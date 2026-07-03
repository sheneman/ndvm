"""PyTorch <-> NDVM autograd boundary (design section 14), Phase 2.

Exposes the native NDVM runtime as a differentiable PyTorch op: a program's bound scalar parameters
flow in as autograd leaves, the scalar output flows out as a tensor, and `loss.backward()` routes
gradients (computed by NDVM's native reverse-mode tape) back to those parameters. External optimizers
and experiment code can therefore use NDVM exactly like any differentiable function, while the
interpreter runtime stays native.

The native extension (`ndvm_native`) is compiled on first use via ``torch.utils.cpp_extension.load``
(which bundles pybind11) from the ndvm C++ sources -- so this needs a C++17 compiler and torch, i.e. an
HPC compute node, not the login node / Mac. The build is cached under ~/.cache/torch_extensions.

    import torch
    from ndvm.python.ndvm_autograd import ndvm_forward
    q = torch.tensor(0.05, requires_grad=True); r = torch.tensor(0.10, requires_grad=True)
    nll = ndvm_forward(KALMAN_SRC, {"q": q, "r": r}, matrices={"obs": (T, 2, obs_flat)})
    nll.backward()                      # q.grad, r.grad now hold dNLL/dq, dNLL/dr
"""
from __future__ import annotations

from pathlib import Path

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # importable without torch; runtime use requires it

_HERE = Path(__file__).resolve().parent
_NDVM = _HERE.parent  # the ndvm/ subtree root
_ext = None


def _get_ext():
    """Load the native extension. Prefer an ahead-of-time build (setup.py build_ext --inplace, which
    needs no ninja); fall back to JIT torch.utils.cpp_extension.load (which does need ninja)."""
    global _ext
    if _ext is None:
        import sys
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        try:
            import ndvm_native as ext  # prebuilt .so next to this file
            _ext = ext
        except Exception:
            from torch.utils.cpp_extension import load
            srcs = [str(_HERE / "ndvm_ext.cpp")] + sorted(str(p) for p in (_NDVM / "src").glob("*.cpp"))
            _ext = load(
                name="ndvm_native",
                sources=srcs,
                extra_include_paths=[str(_NDVM / "include"), str(_NDVM / "src")],
                extra_cflags=["-O2", "-std=c++17"],
                verbose=False,
            )
    return _ext


if torch is not None:

    class NDVMFunction(torch.autograd.Function):
        """Differentiate B independent native NDVM evaluations w.r.t. their bound scalar parameters.

        `params` is a [P, B] tensor (P parameters x B lanes, param-major). One structural walk fits all B
        lanes; the native reverse pass yields the per-lane gradient d(out_b)/d(param_i lane b) in the
        forward call (each lane's output depends only on its own lane's params). backward routes the
        upstream [B] gradient lane-wise: param.grad[i, b] = grad_out[b] * g[i, b]. B == 1 reproduces the
        Phase-2 scalar boundary exactly.
        """

        @staticmethod
        def forward(ctx, src, names, params, B, mnames, mrows, mcols, mdata):  # noqa: D401
            ext = _get_ext()
            outs, grads = ext.eval_and_grad_batched(
                src, list(names), [float(v) for v in params.detach().reshape(-1).tolist()], int(B),
                list(mnames), [int(x) for x in mrows], [int(x) for x in mcols],
                [list(map(float, d)) for d in mdata], True)
            g = torch.as_tensor(grads, dtype=params.dtype, device=params.device).reshape(len(names), int(B))
            ctx.save_for_backward(g)
            return params.new_tensor(outs)  # [B]

        @staticmethod
        def backward(ctx, grad_out):
            (g,) = ctx.saved_tensors  # [P, B]
            return (None, None, grad_out.reshape(1, -1) * g, None, None, None, None, None)

    def ndvm_forward(src: str, params: dict, matrices: dict | None = None):
        """Evaluate `src` through NDVM with differentiable scalar `params` (name -> tensor).

        Each parameter tensor is either a scalar (B=1, or broadcast across the batch) or a 1-D [B] tensor
        of per-lane values; B (the number of independent lanes) is inferred as the common length. One
        native walk fits all B lanes. `matrices` maps a name to (rows, cols, flat_row_major_data), bound as
        a non-differentiated input shared across lanes (read via (ref name k)). Returns a scalar tensor when
        B == 1 (back-compatible) or a [B] tensor otherwise; call .backward() (scalar) or .sum().backward()
        ([B]) to populate each param tensor's .grad (a scalar or [B]) with the per-lane d(output)/d(param).
        """
        names = list(params)

        def _len(t):
            return 1 if t.dim() == 0 else t.reshape(-1).shape[0]
        lens = [_len(params[n]) for n in names]
        B = max(lens) if lens else 1
        for n, L in zip(names, lens):
            if L not in (1, B):
                raise ValueError(f"param {n!r} has length {L}; expected 1 or B={B}")
        rows = []
        for n in names:
            t = params[n].reshape(-1)                  # [1] or [B]
            rows.append(t.expand(B) if t.shape[0] == 1 else t)   # broadcast scalars (differentiable)
        pmat = torch.stack(rows) if rows else torch.empty(0, B)  # [P, B], differentiable

        matrices = matrices or {}
        mnames = list(matrices)
        mrows = [matrices[n][0] for n in mnames]
        mcols = [matrices[n][1] for n in mnames]
        mdata = [matrices[n][2] for n in mnames]
        out = NDVMFunction.apply(src, tuple(names), pmat, B, tuple(mnames), tuple(mrows),
                                 tuple(mcols), tuple(mdata))  # [B]
        return out.reshape(()) if B == 1 else out


def evaluate_population(candidates, nthreads: int = 0):
    """Evaluate a POPULATION of independent candidates in parallel across worker threads (Phase 5).

    Each candidate is ``(src, params, matrices)`` where ``params`` maps a name to a scalar or a length-B
    sequence of per-lane values, and ``matrices`` maps a name to ``(rows, cols, flat_row_major_data)``.
    Returns a list (in candidate order) of dicts ``{ok, err, outs: [B], grads: {name -> [B]}}``. This is
    byte-identical to evaluating each candidate serially, but spread across cores; the co-search inner loop
    packs its whole population into one call. Needs only the native extension (no torch). nthreads<=0 =>
    the NDVM_THREADS env or hardware concurrency.
    """
    ext = _get_ext()

    def _seq(v):
        try:
            return [float(x) for x in v]      # already a sequence
        except TypeError:
            return [float(v)]                 # scalar

    tasks, meta = [], []
    for src, params, matrices in candidates:
        names = list(params)
        seqs = {n: _seq(params[n]) for n in names}
        B = max((len(s) for s in seqs.values()), default=1)
        svals = []
        for n in names:
            s = seqs[n]
            if len(s) not in (1, B):
                raise ValueError(f"candidate param {n!r} has length {len(s)}; expected 1 or B={B}")
            svals.extend(s if len(s) == B else s * B)   # broadcast scalars
        matrices = matrices or {}
        mnames = list(matrices)
        tasks.append((src, list(names), [float(x) for x in svals], int(B), list(mnames),
                      [int(matrices[n][0]) for n in mnames], [int(matrices[n][1]) for n in mnames],
                      [[float(x) for x in matrices[n][2]] for n in mnames], True))
        meta.append((names, B))

    raw = ext.evaluate_batch(tasks, int(nthreads))
    out = []
    for (ok, err, outs, grads), (names, B) in zip(raw, meta):
        gd = {names[i]: list(grads[i * B:(i + 1) * B]) for i in range(len(names))}
        out.append({"ok": ok, "err": err, "outs": list(outs), "grads": gd})
    return out
