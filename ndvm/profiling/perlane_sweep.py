#!/usr/bin/env python3
"""NDVM per-lane sweep: forward+gradient cost vs batch size B (regenerates the "~60x" claim).

Measures the per-lane wall-clock cost of one batched NDVM forward+backward through the PyTorch
autograd boundary (`ndvm.python.ndvm_autograd.ndvm_forward`) on the 80-step 2x2 Kalman NLL, as a
function of the lane count B. One structural walk fits all B lanes, so the per-walk structural
overhead amortizes: per-lane cost (= total / B) falls as B grows. This is the data behind the
"~60x per-lane drop from B=1 to B=256" figure in PHASE3.md.

The object program and binding scheme are reused verbatim from profile_dmci_baseline.py
(`kalman2d_T80`): the program is fed as DATA, the obs sequence is a non-differentiated [T, 2] matrix
bound by name, and only the noise params q, r are differentiated -- here as per-lane [B] leaves.

Requires torch + the prebuilt native extension (setup.py build_ext --inplace). Per project
convention, run on an HPC COMPUTE node (the login node / Mac lacks torch and the .so), single core.

    srun -p eight -n 1 -c 1 --time=00:30:00 \
        bash -lc 'cd /path/to/nncompile-ndvm && .venv/bin/python ndvm/profiling/perlane_sweep.py'

Writes results/perlane_n128.json (rows {B, total_ms, per_lane_ms} + a meta block) and prints a table.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # ndvm/profiling -> ndvm -> repo root

# Batch sizes to sweep; per-lane cost = total / B should fall steeply across this range.
BATCHES = [1, 8, 64, 256, 1024]
REPS = 20        # timed reps per B (median is reported); >= 20 as required
WARMUP = 5       # uncounted warmup reps (extension load, lazy caches, allocator warmup)
KALMAN_T = 80    # 80-step rollout (matches the PHASE3.md / profile_dmci_baseline figure)

# Differentiated noise-param starting values (verbatim from profile_dmci_baseline.kalman2d_T80).
Q0, R0 = 0.05, 0.10


def _try_imports():
    """Import torch + the NDVM autograd boundary, with a clear message on failure."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return None, None, (
            f"cannot import torch ({type(e).__name__}: {e}). "
            "Run on an HPC compute node via .venv/bin/python (the login node / Mac lacks torch).")
    sys.path.insert(0, str(HERE.parent / "python"))
    try:
        import ndvm_autograd  # noqa: E402
    except Exception as e:  # noqa: BLE001
        return None, None, (
            f"cannot import ndvm_autograd / native extension ({type(e).__name__}: {e}). "
            "Build it first: cd ndvm && .venv/bin/python setup.py build_ext --inplace (needs a "
            "C++17 compiler on the compute node).")
    return torch, ndvm_autograd, None


def kalman_src(T: int) -> str:
    # 2D local-level Kalman filter NLL folded through the interpreter -- verbatim shape from
    # profile_dmci_baseline._kalman_src (which itself mirrors kalman_detinv_pilot.py).
    return f"""
(loop ((k 0)
       (x  (vec 0.0 0.0))
       (P  (mat (vec 1.0 0.0) (vec 0.0 1.0)))
       (L  0.0))
  (if (= k {T})
      L
      (let* ((Q     (scale q (eye 2)))
             (R     (scale r (eye 2)))
             (Ppred (+ P Q))
             (y     (ref obs k))
             (e     (- y x))
             (S     (+ Ppred R))
             (Sinv  (inv S))
             (Kg    (matmul Ppred Sinv))
             (xnew  (+ x (matvec Kg e)))
             (Pnew  (matmul (- (eye 2) Kg) Ppred))
             (nll   (+ (log (det S)) (dot e (matvec Sinv e)))))
        (recur (+ k 1) xnew Pnew (+ L nll)))))
"""


def build_obs_matrix(T: int, torch):
    """Non-differentiated obs sequence bound as a [T, 2] matrix: (rows, cols, flat_row_major_data).

    Same seed/shape as profile_dmci_baseline's matrix factory ({"obs": ("randn", (T, 2))},
    Generator().manual_seed(0)), so the rollout sees the identical observation stream. The matrices
    arg to ndvm_forward maps a name to (rows, cols, flat) and is shared (broadcast) across all lanes.
    """
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(T, 2, generator=g)
    flat = obs.reshape(-1).tolist()
    return {"obs": (T, 2, flat)}


def time_one_B(B, src, matrices, torch, ndvm_autograd):
    """Median total ms over REPS of forward+backward for B per-lane Kalman fits in one batched op."""
    def one_eval():
        # Per-lane [B] differentiable leaves (param-major); each lane is an independent fit.
        q = torch.full((B,), Q0, requires_grad=True)
        r = torch.full((B,), R0, requires_grad=True)
        nll = ndvm_autograd.ndvm_forward(src, {"q": q, "r": r}, matrices)  # scalar at B=1, else [B]
        # out[b] depends only on lane b, so .sum().backward() gives each lane its own per-lane grad.
        (nll.sum() if B > 1 else nll).backward()
        # Touch the grads so nothing is optimized away; CPU op, so this also serves as the sync point.
        return float(q.grad[0]) if B > 1 else float(q.grad)

    for _ in range(WARMUP):
        one_eval()

    times_ms = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        one_eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # no-op on CPU runs; honors the "synchronize" requirement
        times_ms.append(1e3 * (time.perf_counter() - t0))

    times_ms.sort()
    n = len(times_ms)
    median_ms = times_ms[n // 2] if n % 2 else 0.5 * (times_ms[n // 2 - 1] + times_ms[n // 2])
    return median_ms


def main():
    torch, ndvm_autograd, err = _try_imports()
    if err is not None:
        print(f"[perlane] {err}")
        return 1

    torch.set_num_threads(1)  # single-core: the per-lane amortization is the signal, not threading
    sys.setrecursionlimit(100_000)

    src = kalman_src(KALMAN_T)
    matrices = build_obs_matrix(KALMAN_T, torch)

    rows = []
    print(f"[perlane] host={socket.gethostname()}  torch={torch.__version__}  "
          f"threads=1  Kalman T={KALMAN_T}  reps={REPS} (warmup={WARMUP})")
    print("=" * 52)
    print(f"{'B':>6s} {'total ms/eval':>16s} {'per-lane ms':>16s}")
    print("-" * 52)
    for B in BATCHES:
        total_ms = time_one_B(B, src, matrices, torch, ndvm_autograd)
        per_lane_ms = total_ms / B
        rows.append({"B": B, "total_ms": total_ms, "per_lane_ms": per_lane_ms})
        print(f"{B:>6d} {total_ms:>16.4f} {per_lane_ms:>16.6f}", flush=True)
    print("=" * 52)
    if len(rows) >= 2:
        speedup = rows[0]["per_lane_ms"] / rows[-1]["per_lane_ms"]
        print(f"[perlane] per-lane drop B={rows[0]['B']} -> B={rows[-1]['B']}: {speedup:.1f}x")

    meta = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "kalman_T": KALMAN_T,
        "params": {"q": Q0, "r": R0},
        "reps": REPS,
        "warmup": WARMUP,
        "num_threads": 1,
        "note": ("NDVM per-lane forward+grad cost vs B for the 80-step 2x2 Kalman NLL through the "
                 "ndvm_forward autograd boundary; regenerates the PHASE3.md ~60x figure. "
                 "Program + bindings reuse profile_dmci_baseline.kalman2d_T80."),
    }
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "perlane_n128.json"
    out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"[perlane] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
