"""Describe a model: its module tree and, for a composed model, its dataflow.

``fx.describe(model)`` is pure introspection: nothing is run, nothing is traced, and the model is not changed. It returns plain JSON-able data:

``tree``
    The module tree of any ``eqx.Module``: for every module and container its path, type and aggregated parameter count and bytes; for every
    array leaf its shape, dtype, parameter count and bytes; and the plain static fields (sizes, flags) of a module.
``summary``
    Total parameters, total bytes and the number of array leaves.
``graph``
    Only when the model is built from :func:`foundax.block`, :class:`foundax.Pipe` and the combinators (:func:`foundax.dot`, :func:`foundax.add`,
    :func:`foundax.cat`): the structure as nodes and edges. A block is a node with its in / out channels, a pipe is a chain of nodes, a
    combinator is a node that takes its branches, so a branching model is a branching graph. This is the real dataflow of such a model, read from
    how it was composed. A plain module (an ``fx.fno2d(...)``, an ``eqx.nn.MLP``) has no composition to read, so ``graph`` is ``None``: its
    structure is the tree.

Example::

    d = fx.describe(fx.add(b1 | b2, b3))
    d["summary"]["total_parameters"]
    d["graph"]["nodes"]   # input, the blocks, an "add" node, output
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

from .combinators import AddCombinator, CatCombinator, DotCombinator
from .pipe import _IN_NAMES, _OUT_NAMES, Block, Pipe, _sniff

_PLAIN = (bool, int, float, str)
_COMBINATORS = {
    AddCombinator: ("add", ("a", "b")),
    CatCombinator: ("cat", ("a", "b")),
    DotCombinator: ("dot", ("branch", "trunk")),
}


def _is_array(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, type)


def _leaf(name: str, path: str, x: Any) -> dict:
    n = int(np.prod(x.shape)) if len(x.shape) else 1
    return {
        "name": name,
        "path": path,
        "kind": "array",
        "type": type(x).__name__,
        "shape": [int(s) for s in x.shape],
        "dtype": str(x.dtype),
        "param_count": n,
        "bytes": int(n * np.dtype(x.dtype).itemsize),
        "children": [],
    }


def _node(name: str, path: str, x: Any) -> Optional[dict]:
    """The tree node of ``x`` (an array, a module, a list / tuple / dict), or None for what carries no parameters (None, plain values, functions)."""
    if _is_array(x):
        return _leaf(name, path, x)
    children, static = [], {}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        items = [(f.name, getattr(x, f.name, None)) for f in dataclasses.fields(x)]
        kind = "module"
    elif isinstance(x, (list, tuple)):
        items, kind = [(str(i), v) for i, v in enumerate(x)], "list"
    elif isinstance(x, dict):
        items, kind = [(str(k), v) for k, v in x.items()], "dict"
    else:
        return None
    for key, value in items:
        if isinstance(value, _PLAIN):
            static[key] = value
            continue
        child = _node(key, f"{path}/{key}" if path else key, value)
        if child is not None:
            children.append(child)
    return {
        "name": name,
        "path": path,
        "kind": kind,
        "type": type(x).__name__,
        "static": static,
        "children": children,
        "param_count": sum(c["param_count"] for c in children),
        "bytes": sum(c["bytes"] for c in children),
    }


def _count_leaves(node: dict) -> int:
    return (
        1
        if node["kind"] == "array"
        else sum(_count_leaves(c) for c in node["children"])
    )


class _Graph:
    """The nodes and edges of a composed model. ``flow(obj)`` adds ``obj`` and returns its (entry node ids, exit node ids)."""

    def __init__(self):
        self.nodes: list = []
        self.edges: list = []
        self.takes: dict = {}  # entry node id -> the dot branch ("branch" / "trunk") whose input it takes, for the inputs of a dot

    def add(self, kind: str, label: str, **extra) -> str:
        nid = f"n{len(self.nodes)}"
        self.nodes.append({"id": nid, "kind": kind, "label": label, **extra})
        return nid

    def link(self, a: str, b: str):
        out = next((n.get("out_channels") for n in self.nodes if n["id"] == a), None)
        self.edges.append(
            {"source": a, "target": b, "label": None if out is None else str(out)}
        )

    def flow(self, obj: Any, path: str) -> tuple:
        if isinstance(obj, Pipe):
            entries, exits = [], []
            for i, blk in enumerate(obj.blocks):
                e, x = self.flow(blk, f"{path}/blocks/{i}")
                if exits:
                    for a in exits:
                        for b in e:
                            self.link(a, b)
                else:
                    entries = e
                exits = x
            return entries, exits
        if isinstance(obj, Block) and isinstance(
            obj.module, (Pipe, Block, *_COMBINATORS)
        ):
            return self.flow(
                obj.module, f"{path}/module"
            )  # a block around a composition is that composition
        for cls, (kind, branches) in _COMBINATORS.items():
            if isinstance(obj, cls):
                comb = self.add(kind, kind, path=path)
                entries = []
                for name in branches:
                    e, x = self.flow(getattr(obj, name), f"{path}/{name}")
                    entries += e
                    if kind == "dot":
                        self.takes.update({n: name for n in e})
                    for a in x:
                        self.link(a, comb)
                return entries, [comb]
        module = obj.module if isinstance(obj, Block) else obj
        in_c = obj._in_channels if isinstance(obj, Block) else _sniff(module, _IN_NAMES)
        out_c = (
            obj._out_channels if isinstance(obj, Block) else _sniff(module, _OUT_NAMES)
        )
        nid = self.add(
            "block" if isinstance(obj, Block) else "module",
            obj.name if isinstance(obj, Block) else type(module).__name__,
            type=type(module).__name__,
            path=f"{path}/module" if isinstance(obj, Block) else path,
            in_channels=in_c,
            out_channels=out_c,
        )
        return [nid], [nid]


def _composed(model: Any) -> bool:
    return isinstance(model, (Block, Pipe, *_COMBINATORS))


def _graph(model: Any) -> dict:
    g = _Graph()
    entries, exits = g.flow(model, "")
    # one input node feeds every entry; a dot takes two inputs, u for its branch and y for its trunk
    inputs = {}
    for nid in entries:
        key = g.takes.get(nid)
        if key not in inputs:
            inputs[key] = g.add(
                "input", {"branch": "u", "trunk": "y"}.get(key, "input")
            )
        g.edges.append({"source": inputs[key], "target": nid, "label": None})
    out = g.add("output", "output")
    for a in exits:
        g.link(a, out)
    return {"nodes": g.nodes, "edges": g.edges}


def describe(model: Any) -> dict:
    """The module tree of ``model`` and, when it is composed of blocks, pipes and combinators, its dataflow graph. See the module docstring."""
    root = _node(type(model).__name__, "", model)
    if root is None:
        raise TypeError(
            f"describe() needs an equinox module (or a list / dict of them), got {type(model).__name__}"
        )
    summary = {
        "total_parameters": root["param_count"],
        "total_bytes": root["bytes"],
        "leaf_count": _count_leaves(root),
    }
    return {
        "type": type(model).__name__,
        "tree": root,
        "summary": summary,
        "graph": _graph(model) if _composed(model) else None,
    }


__all__ = ["describe"]
