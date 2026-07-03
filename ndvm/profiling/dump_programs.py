#!/usr/bin/env python3
"""Dump the five canonical baseline programs as .scm source + .binds files for the native CLI drivers
(ndvm_run / boxed_run). The obs matrix for the Kalman program is torch.randn((T,2)) under manual_seed(0),
identical to residual_e2e.make_matrices, so ndvm_run and boxed_run consume the SAME inputs the eager/
tuned-eager/NDVM decomposition uses. Run on a compute node (needs torch). Writes into the given out dir.

    python3 dump_programs.py <out_dir>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
from profile_dmci_baseline import build_programs

ORDER = ["scalar_mul_add", "michaelis_menten", "damped_oscillator", "logistic_map_loop", "kalman2d_T80"]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    progs = build_programs(16, 80)
    for name in ORDER:
        p = progs[name]
        (out / f"{name}.scm").write_text(p["src"].strip() + "\n")
        lines = []
        for k, v in p["params"].items():
            lines.append(f"scalar {k} {float(v)!r}")
        for k, v in p["inputs"].items():
            lines.append(f"scalar {k} {float(v)!r}")
        for mn, (kind, shape) in p.get("matrix", {}).items():
            g = torch.Generator().manual_seed(0)
            t = torch.randn(*shape, generator=g) if kind == "randn" else torch.zeros(*shape)
            flat = t.reshape(-1).tolist()
            lines.append(f"matrix {mn} {shape[0]} {shape[1]} " + " ".join(repr(float(x)) for x in flat))
        (out / f"{name}.binds").write_text("\n".join(lines) + "\n")
        print(f"wrote {name}.scm + {name}.binds")


if __name__ == "__main__":
    main()
