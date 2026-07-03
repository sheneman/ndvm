############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# cli.py: ``nncompile`` -- command-line front-end for the Neural Compiler. Compile a Scheme ``.scm`` program to a...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""``nncompile`` -- command-line front-end for the Neural Compiler.

Compile a Scheme ``.scm`` program to a portable, backend-agnostic ``.ncg`` artifact, emit a
ready-to-import differentiable ``torch.nn.Module``, evaluate a program/artifact on any backend,
or inspect a compiled graph.

    nncompile compile model.scm -o model.ncg
    nncompile emit    model.scm --params a,b -o model.py
    nncompile run     model.ncg --inputs '{"x": 4.0}'
    nncompile info    model.scm

A ``.ncg`` is the compiled program as data; the backend (PyTorch / JAX / NumPy / CuPy) is
chosen at *run* time, not baked into the file -- compile once, differentiate everywhere.
Gradient backends are ``torch`` and ``jax``; ``numpy`` and ``cupy`` are forward-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRAD_BACKENDS = ("torch", "jax")
ALL_BACKENDS = ("torch", "jax", "numpy", "cupy")


def _read(path: str) -> str:
    return Path(path).read_text()


def _compile_auto(source: str, prelude: bool = False):
    """Compile a program, auto-discovering its free variables as inputs.

    The compiler raises ``KeyError("Undefined variable: X")`` for each free variable; we add
    each reported name to the input declaration and retry, so the user need not list inputs by
    hand. Returns the compiled ``ComputeGraph``.
    """
    from neural_compiler.compiler import compile_program

    inputs: dict[str, None] = {}
    seen: set[str] = set()
    while True:
        try:
            return compile_program(source, inputs=inputs or None, prelude=prelude)
        except KeyError as e:
            msg = str(e.args[0]) if e.args else str(e)
            if "Undefined variable:" not in msg:
                raise
            name = msg.split("Undefined variable:")[-1].strip().strip("'\"")
            if not name or name in seen:
                raise
            seen.add(name)
            inputs[name] = None


def _compile_program_graph(source: str, dmci: bool = False, prelude: bool = False):
    """Compile a Scheme program directly, or via DMCI (the meta-circular interpreter path)."""
    if dmci:
        # The tagged evaluator trampolines tail calls (including the interpreter's own
        # scheme-eval/eval-apply loop), so tail recursion runs in constant Python stack.
        # Structural (non-tail) recursion still consumes stack proportional to the interpreted
        # program's genuine recursion depth, so keep a generous limit as a safety margin.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
        from neural_compiler.dmci import compile_dmci

        return compile_dmci(source, prelude=prelude)
    return _compile_auto(source, prelude=prelude)


def _load_or_compile(source_path: str, prelude: bool = False, dmci: bool = False):
    """Return ``(graph, source_text)`` for a ``.ncg`` artifact or a ``.scm`` program."""
    if source_path.endswith(".ncg"):
        from neural_compiler.serialize import from_artifact, load_artifact

        art = load_artifact(source_path)
        return from_artifact(art), art.get("source")
    src = _read(source_path)
    return _compile_program_graph(src, dmci=dmci, prelude=prelude), src.strip()


# --------------------------------------------------------------------------- subcommands
def cmd_compile(args) -> int:
    from neural_compiler.serialize import save_compiled

    src = _read(args.source)
    graph = _compile_program_graph(src, dmci=args.dmci, prelude=args.prelude)
    out = args.output or str(Path(args.source).with_suffix(".ncg"))
    save_compiled(graph, out, source=src.strip())
    print(f"compiled {args.source} -> {out}" + ("   (via DMCI interpreter)" if args.dmci else ""))
    print(f"  inputs: {', '.join(graph.input_names) or '(none)'}   nodes: {len(graph.nodes)}   "
          f"tagged: {graph.uses_tagged_values}")
    return 0


def cmd_emit(args) -> int:
    from neural_compiler.emit import emit_jax_module, emit_torch_module

    src = _read(args.source)
    graph = _compile_program_graph(src, dmci=args.dmci, prelude=args.prelude)
    params = [p.strip() for p in (args.params.split(",") if args.params else []) if p.strip()]
    data = [n for n in graph.input_names if n not in params]
    out = args.output or str(Path(args.source).with_suffix(".py"))
    emitter = emit_jax_module if args.backend == "jax" else emit_torch_module
    try:
        code = emitter(graph, src.strip(), params, data, module_name=Path(out).stem)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    Path(out).write_text(code)
    print(f"emitted {args.backend} module {args.source} -> {out}"
          + ("   (via DMCI interpreter)" if args.dmci else ""))
    print(f"  params (learnable): {', '.join(params) or '(none)'}")
    print(f"  data inputs:        {', '.join(data) or '(none)'}")
    if not params:
        print("  note: no --params given, so the module has no learnable constants "
              "(a fixed function). Pass --params a,b to optimize constants a and b.")
    return 0


def cmd_run(args) -> int:
    from neural_compiler.evaluator import evaluate

    graph, _ = _load_or_compile(args.source, args.prelude, dmci=args.dmci)
    try:
        inputs = json.loads(args.inputs) if args.inputs else {}
    except json.JSONDecodeError as e:
        print(f"error: --inputs must be a JSON object, e.g. '{{\"x\": 4.0}}' ({e})", file=sys.stderr)
        return 2
    missing = [n for n in graph.input_names if n not in inputs]
    if missing:
        print(f"error: missing input value(s) {missing}; the program's inputs are "
              f"{graph.input_names}. Pass them with --inputs.", file=sys.stderr)
        return 2
    eval_kwargs = {}
    # --max-heap raises the tagged-value heap cap (a torch-path knob); pass only for torch.
    # The default cap is intentionally kept: a typical recursive DMCI program uses very little
    # heap (~tens of cells per level), so a sudden overflow usually flags a non-terminating
    # program (e.g. an unsupported operator), which a roomy default would only hide.
    if args.max_heap and args.backend in (None, "torch"):
        eval_kwargs["max_heap"] = args.max_heap
    result = evaluate(graph, inputs, backend=args.backend, **eval_kwargs)
    # Tagged (heap) programs return a raw tagged-value tensor; unwrap the scalar payload.
    if graph.uses_tagged_values and args.backend in (None, "torch"):
        from neural_compiler.runtime.tagged_value import unwrap_number
        result = float(unwrap_number(result))
    print(result if isinstance(result, (int, float)) else getattr(result, "tolist", lambda: result)())
    return 0


def cmd_info(args) -> int:
    graph, source = _load_or_compile(args.source, args.prelude, dmci=args.dmci)
    if graph.uses_tagged_values:
        batchable = "no (heap ops are not batchable)"
    elif graph.has_loops or graph.has_functions:
        batchable = "only if control flow is uniform across the batch (loops/recursion)"
    else:
        batchable = "yes (straight-line)"
    print(f"source:        {source or '(unavailable)'}")
    print(f"inputs:        {', '.join(graph.input_names) or '(none)'}")
    print(f"nodes:         {len(graph.nodes)}")
    print(f"graph depth:   {graph.depth()}")
    print(f"has loops:     {graph.has_loops}")
    print(f"has functions: {graph.has_functions}   (recursion / closures)")
    print(f"tagged values: {graph.uses_tagged_values}   (heap-allocated pairs/lists/closures)")
    print(f"batchable:     {batchable}")
    print(f"backends:      autograd = {', '.join(GRAD_BACKENDS)} ; "
          f"forward-only = numpy, cupy")
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nncompile", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, dmci=True):
        sp.add_argument("--prelude", action="store_true",
                        help="prepend the stdlib PRELUDE (map/filter/fold-left/...)")
        if dmci:
            sp.add_argument("--dmci", action="store_true",
                            help="compile via the meta-circular interpreter (program as DATA "
                                 "through the compiled evaluator) instead of direct compilation")

    c = sub.add_parser("compile", help="compile a .scm program to a portable .ncg artifact")
    c.add_argument("source", help="input Scheme program (.scm)")
    c.add_argument("-o", "--output", help="output .ncg path (default: <source>.ncg)")
    add_common(c)
    c.set_defaults(func=cmd_compile)

    e = sub.add_parser("emit", help="emit a standalone, importable module (torch.nn.Module or JAX)")
    e.add_argument("source", help="input Scheme program (.scm)")
    e.add_argument("-o", "--output", help="output .py path (default: <source>.py)")
    e.add_argument("--params", help="comma-separated input names to expose as learnable "
                   "constants (the rest become data inputs)")
    e.add_argument("--backend", default="torch", choices=["torch", "jax"],
                   help="emit target: torch (nn.Module) or jax (functional apply)")
    add_common(e)
    e.set_defaults(func=cmd_emit)

    r = sub.add_parser("run", help="evaluate a .scm program or .ncg artifact on a backend")
    r.add_argument("source", help="input .scm program or .ncg artifact")
    r.add_argument("--inputs", help='JSON object of input values, e.g. \'{"x": 4.0}\'')
    r.add_argument("--backend", default="torch", choices=list(ALL_BACKENDS),
                   help="evaluation backend (torch/jax are differentiable; numpy/cupy forward-only)")
    r.add_argument("--max-heap", type=int, default=None,
                   help="cap on heap cells for tagged/DMCI programs (default: roomy for --dmci). "
                        "Raise it if a recursive DMCI run reports a heap overflow.")
    add_common(r)
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("info", help="print a compiled program's inputs, structure, and backends")
    i.add_argument("source", help="input .scm program or .ncg artifact")
    add_common(i)
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
