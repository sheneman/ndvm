#!/usr/bin/env python3
"""End-to-end harness for the SECOND client (differentiable stack-bytecode VM).

Reviewer weakness 1 (a single measured client) remediation: show the structural/numeric-split value
representation generalizes to a second front end with a different dispatch model (a stack-bytecode VM,
neural_compiler/clients/bytecode_vm.py) that shares the value contract but contains NO DMCI parser /
evaluator code. For each of four workloads (W1 scalar expr, W2 counted loop, W3 branch, W4 small
matvec/dot) this harness:

  * runs the VM on BOTH value backends (payload-only split vs eager fused [tag,payload] tensor);
  * validates forward agreement and gradient agreement against (i) torch autograd through a plain-torch
    oracle and (ii) central finite-difference (atol ~1e-3);
  * shows batch-independence: one structural walk over B payload lanes, per-lane cost roughly flat
    B=1..256;
  * times eager-VM vs payload-VM and reports the speedup.

Emits the W1-W4 results table, the per-lane-cost batch table, and a small LOC inventory (VM-specific
lines vs reused value/AD interface lines). Run on an HPC compute node (the Mac lacks torch).

    python3 ndvm/profiling/bytecode_vm_e2e.py
"""
from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

# Import the VM module by file path (no package __init__ needed; keeps the track to its two files).
_spec = importlib.util.spec_from_file_location(
    "bytecode_vm", REPO_ROOT / "neural_compiler" / "clients" / "bytecode_vm.py")
bvm = importlib.util.module_from_spec(_spec)
sys.modules["bytecode_vm"] = bvm
_spec.loader.exec_module(bvm)

BytecodeVM = bvm.BytecodeVM
PayloadBackend = bvm.PayloadBackend
EagerBackend = bvm.EagerBackend
INSTRUCTIONS = bvm.INSTRUCTIONS


# ---------------------------------------------------------------------------------------------------------
# Differentiable leaf parameters per workload (scalars, or vectors-as-lists for DOT). Batched [B] variants
# share the same structure with a leading batch axis on each numeric leaf.
# ---------------------------------------------------------------------------------------------------------

def w_params(name, batch=1):
    def leaf(v):
        if batch > 1:
            return torch.full((batch,), float(v), requires_grad=True)
        return torch.tensor(float(v), requires_grad=True)

    if name == "W1_scalar_expr":
        return {"a": leaf(0.3), "x": leaf(1.5), "w": leaf(2.0), "b": leaf(0.7)}
    if name == "W2_counted_loop":
        return {"x0": leaf(0.4), "r": leaf(3.2)}
    if name == "W3_branch":
        # g>0 selects the then-arm; structural selector kept constant across the batch
        return {"g": leaf(1.0), "a": leaf(2.0), "x": leaf(1.5), "b": leaf(3.0)}
    if name == "W4_matvec":
        return {
            "m0": [leaf(1.0), leaf(2.0), leaf(-1.0)],
            "m1": [leaf(0.5), leaf(-0.5), leaf(2.0)],
            "m2": [leaf(-1.0), leaf(1.0), leaf(0.5)],
            "v":  [leaf(1.0), leaf(0.5), leaf(-2.0)],
        }
    raise KeyError(name)


def leaf_list(params):
    """Flatten the param dict to an ordered list of differentiable leaf tensors (for autograd.grad)."""
    leaves, names = [], []
    for k, v in params.items():
        if isinstance(v, list):
            for i, t in enumerate(v):
                leaves.append(t); names.append(f"{k}[{i}]")
        else:
            leaves.append(v); names.append(k)
    return leaves, names


# ---------------------------------------------------------------------------------------------------------
# Validation: forward agreement (eager vs payload vs oracle); gradient agreement (autograd + finite diff).
# ---------------------------------------------------------------------------------------------------------

def run_vm(program, params, backend):
    vm = BytecodeVM(backend)
    return vm.run(program, params)


def central_fd_grad(ref, params, eps=1e-4):
    """Central finite-difference gradient of ref(params).sum() w.r.t. each scalar leaf."""
    leaves, _ = leaf_list(params)
    base = [t.detach().clone() for t in leaves]
    grads = []
    for i, t in enumerate(leaves):
        with torch.no_grad():
            t.copy_(base[i] + eps)
        fp = float(ref(params).reshape(-1).sum().item())
        with torch.no_grad():
            t.copy_(base[i] - eps)
        fm = float(ref(params).reshape(-1).sum().item())
        with torch.no_grad():
            t.copy_(base[i])
        grads.append((fp - fm) / (2 * eps))
    return grads


def validate(name, program, ref):
    params = w_params(name, batch=1)

    # forward, both backends + oracle
    yp = run_vm(program, params, PayloadBackend())
    ye = run_vm(program, params, EagerBackend())
    yo = ref(params)
    yp_s, ye_s, yo_s = (float(t.reshape(-1).sum().item()) for t in (yp, ye, yo))
    fwd_match = abs(yp_s - yo_s) < 1e-5 and abs(ye_s - yo_s) < 1e-5

    # autograd through the payload VM
    leaves, lnames = leaf_list(params)
    gp = torch.autograd.grad(yp.reshape(-1).sum(), leaves, retain_graph=True, allow_unused=True)
    gp = [0.0 if g is None else float(g.reshape(-1).sum().item()) for g in gp]
    # autograd through the eager VM (fresh graph)
    params_e = w_params(name, batch=1)
    leaves_e, _ = leaf_list(params_e)
    ye2 = run_vm(program, params_e, EagerBackend())
    ge = torch.autograd.grad(ye2.reshape(-1).sum(), leaves_e, retain_graph=True, allow_unused=True)
    ge = [0.0 if g is None else float(g.reshape(-1).sum().item()) for g in ge]
    # autograd through the oracle
    params_o = w_params(name, batch=1)
    leaves_o, _ = leaf_list(params_o)
    go = torch.autograd.grad(ref(params_o).reshape(-1).sum(), leaves_o, retain_graph=True, allow_unused=True)
    go = [0.0 if g is None else float(g.reshape(-1).sum().item()) for g in go]
    # central finite difference oracle
    params_fd = w_params(name, batch=1)
    gfd = central_fd_grad(ref, params_fd)

    def close(a, b, atol, rtol=0.0):
        return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))

    grad_vs_autograd = close(gp, go, 1e-5) and close(ge, go, 1e-5)
    # central finite difference (eps=1e-4) carries O(eps^2) truncation plus float rounding, so compare
    # with a combined absolute+relative tolerance (atol 3e-3, rtol 2e-3); the analytic autograd path is
    # the exact check above. (Autograd already matches the oracle to 1e-5 on every leaf.)
    grad_vs_fd = close(gp, gfd, 3e-3, 2e-3) and close(ge, gfd, 3e-3, 2e-3)

    return {
        "name": name,
        "fwd_payload": yp_s, "fwd_eager": ye_s, "fwd_oracle": yo_s,
        "fwd_match": fwd_match,
        "grad_payload": gp, "grad_eager": ge, "grad_oracle": go, "grad_fd": gfd,
        "grad_vs_autograd": grad_vs_autograd, "grad_vs_fd": grad_vs_fd,
        "leaf_names": lnames,
    }


# ---------------------------------------------------------------------------------------------------------
# Timing: eager-VM vs payload-VM, and per-lane cost across batch sizes (batch-independence).
# ---------------------------------------------------------------------------------------------------------

def timed(fn, n=30):
    # warmup
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1e3  # median ms


def time_workload(name, program, batches=(1, 8, 64, 256)):
    rows = {}
    for B in batches:
        params_p = w_params(name, batch=B)
        params_e = w_params(name, batch=B)
        # include backward in the timed unit so the per-lane curve reflects a real fwd+bwd training step
        pb, eb = PayloadBackend(), EagerBackend()
        leaves_p, _ = leaf_list(params_p)
        leaves_e, _ = leaf_list(params_e)

        def step_payload():
            y = BytecodeVM(pb).run(program, params_p)
            torch.autograd.grad(y.reshape(-1).sum(), leaves_p, retain_graph=True, allow_unused=True)

        def step_eager():
            y = BytecodeVM(eb).run(program, params_e)
            torch.autograd.grad(y.reshape(-1).sum(), leaves_e, retain_graph=True, allow_unused=True)

        mp = timed(step_payload)
        me = timed(step_eager)
        rows[B] = {"payload_ms": mp, "eager_ms": me,
                   "payload_per_lane_us": 1e3 * mp / B, "eager_per_lane_us": 1e3 * me / B,
                   "speedup": me / mp}
    return rows


# ---------------------------------------------------------------------------------------------------------
# LOC inventory: VM-specific lines vs reused value/AD interface lines. Honest accounting -- the VM is a
# deliberate second consumer of the same engine, not an independently motivated system.
# ---------------------------------------------------------------------------------------------------------

def loc_inventory():
    def sloc(path):
        n = 0
        for line in Path(path).read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
        return n

    vm_file = REPO_ROOT / "neural_compiler" / "clients" / "bytecode_vm.py"
    value_file = REPO_ROOT / "neural_compiler" / "runtime" / "payload_value.py"
    vm_total = sloc(vm_file)
    value_total = sloc(value_file)
    # the VM imports the numeric value box + tag codes from payload_value; that module is the reused
    # representation/AD interface shared with the tree-walking evaluator (the first client).
    return {
        "vm_specific_sloc": vm_total,
        "reused_value_interface_sloc": value_total,
        "note": ("bytecode_vm.py is the only new client code; the value box (PV, numeric "
                 "constructors, tag codes) + torch-autograd AD come from payload_value.py, shared "
                 "verbatim with the DMCI tree-walking evaluator."),
    }


# ---------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------

def main():
    workloads = bvm.all_workloads()
    assert len(INSTRUCTIONS) == 15, f"instruction count {len(INSTRUCTIONS)}"
    print(f"# instruction set ({len(INSTRUCTIONS)}): {', '.join(INSTRUCTIONS)}")
    print(f"# host torch={torch.__version__}")
    print()

    # --- validation + timing per workload ---
    vrows, trows = {}, {}
    for name, (program, ref) in workloads.items():
        vrows[name] = validate(name, program, ref)
        trows[name] = time_workload(name, program)

    print("=" * 100)
    print("W1-W4 RESULTS: forward + gradient validation, eager-VM vs payload-VM timing (B=1), per-lane @B=256")
    print("=" * 100)
    hdr = (f"{'workload':18} {'eager_ms':>9} {'payload_ms':>11} {'perlane_us@256':>15} "
           f"{'speedup':>8} {'fwd_match':>9} {'grad=AD':>8} {'grad=FD':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name in workloads:
        v = vrows[name]; t = trows[name]
        b1, b256 = t[1], t[256]
        print(f"{name:18} {b1['eager_ms']:9.3f} {b1['payload_ms']:11.3f} "
              f"{b256['payload_per_lane_us']:15.3f} {b1['eager_ms']/b1['payload_ms']:7.2f}x "
              f"{str(v['fwd_match']):>9} {str(v['grad_vs_autograd']):>8} {str(v['grad_vs_fd']):>8}")

    print()
    print("=" * 100)
    print("BATCH-INDEPENDENCE: per-lane cost (us) across B (one structural walk over B payload lanes)")
    print("=" * 100)
    for name in workloads:
        t = trows[name]
        cells = "  ".join(f"B={B}:{t[B]['payload_per_lane_us']:7.3f}us" for B in (1, 8, 64, 256))
        flat = t[256]['payload_per_lane_us'] / t[1]['payload_per_lane_us']
        print(f"{name:18} payload per-lane: {cells}   (B256/B1 = {flat:.3f}x)")

    print()
    print("=" * 100)
    print("GRADIENT detail (workload : leaf : payload-VM / eager-VM / autograd-oracle / finite-diff)")
    print("=" * 100)
    for name in workloads:
        v = vrows[name]
        for nm, gp, ge, go, gfd in zip(v["leaf_names"], v["grad_payload"], v["grad_eager"],
                                       v["grad_oracle"], v["grad_fd"]):
            print(f"{name:18} {nm:8} {gp:+.5f} / {ge:+.5f} / {go:+.5f} / {gfd:+.5f}")

    print()
    print("=" * 100)
    print("LOC INVENTORY (honest: VM-specific vs reused value/AD interface)")
    print("=" * 100)
    loc = loc_inventory()
    print(f"  VM-specific (bytecode_vm.py) SLOC ........ {loc['vm_specific_sloc']}")
    print(f"  reused value/AD interface (payload_value.py) SLOC ... {loc['reused_value_interface_sloc']}")
    print(f"  note: {loc['note']}")

    # overall validation gate
    all_ok = all(v["fwd_match"] and v["grad_vs_autograd"] and v["grad_vs_fd"] for v in vrows.values())
    print()
    print(f"ALL WORKLOADS VALIDATED (fwd + grad=AD + grad=FD): {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
