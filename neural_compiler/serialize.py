############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# serialize.py: Serialize a compiled :class:`ComputeGraph` to/from a portable JSON artifact (``.ncg``). A ``.ncg`` file is the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Serialize a compiled :class:`ComputeGraph` to/from a portable JSON artifact (``.ncg``).

A ``.ncg`` file is the *compiled form* of a Scheme program: a backend-agnostic dataflow
graph that any supported backend can evaluate at load time (``evaluate(graph, inputs,
backend=...)``). One artifact runs on PyTorch, JAX, NumPy, or CuPy -- the backend is chosen
when you run it, not baked into the file ("compile once, differentiate everywhere").

JSON (not pickle) is used deliberately: a ``.ncg`` is a pure data description of the graph,
so loading one cannot execute arbitrary code. The schema mirrors the dataclasses in
:mod:`neural_compiler.graph.builder` (nested ``LoopBody``/``FunctionBody`` body graphs are
serialized recursively). Build-time scratch fields (``_next_id``, ``_outer_scope``,
``_captures``, ``_root_ref``) are not stored -- the evaluator never reads them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural_compiler.graph.builder import (
    ComputeGraph,
    FunctionBody,
    GraphNode,
    LoopBody,
)
from neural_compiler.parser.ast_nodes import SchemeChar

FORMAT = "nncompile-graph"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- values
def _value_to_json(v: Any) -> Any:
    """Encode a GraphNode.value (float | int | bool | None | SchemeChar) as JSON."""
    if isinstance(v, SchemeChar):
        return {"__char__": v.code_point}
    # bool must be checked before int (bool is an int subclass); JSON keeps true/false,
    # ints, and floats distinct, so they round-trip as themselves.
    return v


def _value_from_json(v: Any) -> Any:
    if isinstance(v, dict) and "__char__" in v:
        return SchemeChar(v["__char__"])
    return v


# --------------------------------------------------------------------------- nodes
def _node_to_dict(n: GraphNode) -> dict:
    return {
        "node_id": n.node_id,
        "name": n.name,
        "op_type": n.op_type,
        "value": _value_to_json(n.value),
        "input_edges": list(n.input_edges),
        "input_names": list(n.input_names),
        "call_target": n.call_target,
    }


def _node_from_dict(d: dict) -> GraphNode:
    return GraphNode(
        node_id=d["node_id"],
        name=d["name"],
        op_type=d["op_type"],
        value=_value_from_json(d.get("value")),
        input_edges=list(d.get("input_edges", [])),
        input_names=list(d.get("input_names", [])),
        call_target=d.get("call_target"),
    )


# --------------------------------------------------------------------------- bodies
def _loop_to_dict(lb: LoopBody) -> dict:
    return {
        "params": list(lb.params),
        "captures": list(lb.captures),
        "recur_flag_id": lb.recur_flag_id,
        "result_id": lb.result_id,
        # dict[str, list[tuple[list[int], list[str]]]] -> tuples become 2-element lists
        "recur_arg_ids": {
            k: [[list(edges), list(names)] for (edges, names) in v]
            for k, v in lb.recur_arg_ids.items()
        },
        "body_graph": graph_to_dict(lb.body_graph, _top=False),
    }


def _loop_from_dict(d: dict) -> LoopBody:
    return LoopBody(
        params=tuple(d["params"]),
        body_graph=graph_from_dict(d["body_graph"]),
        captures=tuple(d.get("captures", [])),
        recur_flag_id=d.get("recur_flag_id"),
        result_id=d.get("result_id"),
        recur_arg_ids={
            k: [(list(edges), list(names)) for (edges, names) in v]
            for k, v in d.get("recur_arg_ids", {}).items()
        },
    )


def _func_to_dict(fb: FunctionBody) -> dict:
    return {
        "name": fb.name,
        "params": list(fb.params),
        "captures": list(fb.captures),
        "body_graph": graph_to_dict(fb.body_graph, _top=False),
    }


def _func_from_dict(d: dict) -> FunctionBody:
    return FunctionBody(
        name=d["name"],
        params=tuple(d["params"]),
        body_graph=graph_from_dict(d["body_graph"]),
        captures=tuple(d.get("captures", [])),
    )


# --------------------------------------------------------------------------- graph
def graph_to_dict(g: ComputeGraph, _top: bool = True) -> dict:
    """Recursively encode a ComputeGraph as a JSON-serializable dict.

    The recursive ``functions`` registry (recursive programs put each function in scope
    within its own body, so it is shared by reference across every body graph) is serialized
    ONCE at the top level. Nested body graphs omit it and the registry is re-shared on load
    (:func:`from_artifact`), which both breaks the reference cycle and preserves semantics.
    """
    d = {
        "root_id": g.root_id,
        "input_names": list(g.input_names),
        "uses_tagged_values": g.uses_tagged_values,
        "name_to_id": dict(g.name_to_id),
        "nodes": {str(nid): _node_to_dict(n) for nid, n in g.nodes.items()},
        "loops": {str(nid): _loop_to_dict(lb) for nid, lb in g.loops.items()},
    }
    if _top:
        d["functions"] = {name: _func_to_dict(fb) for name, fb in g.functions.items()}
    return d


def graph_from_dict(d: dict) -> ComputeGraph:
    """Reconstruct a ComputeGraph from :func:`graph_to_dict` output.

    Note: nested body graphs are reconstructed with an empty ``functions`` registry;
    :func:`from_artifact` re-shares the top-level registry across them.
    """
    g = ComputeGraph(
        nodes={int(nid): _node_from_dict(nd) for nid, nd in d["nodes"].items()},
        name_to_id=dict(d.get("name_to_id", {})),
        root_id=d.get("root_id"),
        input_names=list(d.get("input_names", [])),
        loops={int(nid): _loop_from_dict(lb) for nid, lb in d.get("loops", {}).items()},
        functions={name: _func_from_dict(fb) for name, fb in d.get("functions", {}).items()},
        uses_tagged_values=d.get("uses_tagged_values", False),
    )
    _share_function_registry(g)  # no-op for graphs without a functions registry (nested bodies)
    return g


def _all_body_graphs(g: ComputeGraph):
    """Yield ``g`` and every nested loop/function body graph, each once."""
    seen, stack = set(), [g]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        stack.extend(lb.body_graph for lb in cur.loops.values())
        stack.extend(fb.body_graph for fb in cur.functions.values())


def _share_function_registry(root: ComputeGraph) -> None:
    """Point every body graph's ``functions`` at the single top-level registry (by reference),
    restoring the shared-registry semantics the compiler relies on for recursion."""
    registry = root.functions
    if not registry:
        return
    for bg in list(_all_body_graphs(root)):
        bg.functions = registry


# --------------------------------------------------------------------------- file I/O
def to_artifact(graph: ComputeGraph, source: str | None = None) -> dict:
    """Wrap a graph in the versioned ``.ncg`` envelope."""
    return {
        "format": FORMAT,
        "version": SCHEMA_VERSION,
        "source": source,
        "graph": graph_to_dict(graph),
    }


def from_artifact(artifact: dict) -> ComputeGraph:
    """Validate the envelope and return the contained ComputeGraph."""
    if artifact.get("format") != FORMAT:
        raise ValueError(f"Not an nncompile graph artifact (format={artifact.get('format')!r})")
    if artifact.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported .ncg schema version {artifact.get('version')!r} "
            f"(this build reads version {SCHEMA_VERSION})")
    return graph_from_dict(artifact["graph"])


def save_compiled(graph: ComputeGraph, path: str | Path, source: str | None = None) -> None:
    """Write a compiled graph to a ``.ncg`` JSON file (``source`` is stored for provenance)."""
    Path(path).write_text(json.dumps(to_artifact(graph, source), indent=2))


def load_compiled(path: str | Path) -> ComputeGraph:
    """Load a ``.ncg`` file written by :func:`save_compiled` into a ComputeGraph.

    The returned graph is evaluated with ``neural_compiler.evaluator.evaluate(graph, inputs,
    backend=...)`` on any supported backend.
    """
    return from_artifact(json.loads(Path(path).read_text()))


def load_artifact(path: str | Path) -> dict:
    """Load the raw ``.ncg`` envelope (including the ``source`` field), without rebuilding."""
    return json.loads(Path(path).read_text())
