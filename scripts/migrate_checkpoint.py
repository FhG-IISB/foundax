"""Convert a checkpoint saved with foundax 0.2 into the 0.3 layout of the same model.

``eqx.tree_serialise_leaves`` writes a model's leaves in order and nothing else, so a file only loads into a model
whose leaves come in the same order with the same shapes. foundax 0.3 changes that order for the core models:

- **Fixed arrays are no longer leaves.** Coordinate grids, KAN knots and centres, the SFNO Legendre table, PiT
  distance matrices, GAOT latent positions, PCNO / geofno_nodal modes, the Transformer's sinusoidal encoding and
  DeepONet's fixed Fourier features moved into static ``Constant`` fields, at the same place in each module.
  The 0.2 file holds them as leaves; they are read and dropped, and the model keeps the values it builds from its
  configuration (a 0.2 model may have moved them during training: weight decay, or gradients where nothing stopped
  them; the largest change is reported).

Command line (the model is built by a Python expression with ``fx`` = foundax, ``jax`` and ``jnp`` in scope, with
the same configuration as the saved one)::

    python scripts/migrate_checkpoint.py old.eqx new.eqx --model "fx.fno2d(2, 32, 16, use_positions=True)"

From Python::

    from migrate_checkpoint import load_v02
    model = load_v02("old.eqx", fx.fno2d(2, 32, 16, use_positions=True))
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from foundax.architectures.common import Constant


def _is_constant(x) -> bool:
    return isinstance(x, Constant)


def legacy_view(model):
    """``model`` as the 0.2 layout saw it: every ``Constant`` in a static field becomes a leaf again.

    Returns ``(tree, is_constant)``: a pytree whose leaves are in the 0.2 order, and a pytree of the same structure
    that is ``True`` at the leaves that are constants now.
    """

    def convert(node, mark):
        if not isinstance(node, eqx.Module):
            return False if mark else node
        parts = []
        for field in dataclasses.fields(node):
            value = getattr(node, field.name, None)
            if field.metadata.get("static", False):
                constants = [
                    c
                    for c in jax.tree.leaves(value, is_leaf=_is_constant)
                    if _is_constant(c)
                ]
                parts.extend(True if mark else jnp.asarray(c.value) for c in constants)
            else:
                parts.append(
                    jax.tree.map(lambda n: convert(n, mark), value, is_leaf=_is_module)
                )
        return tuple(parts)

    tree = jax.tree.map(lambda n: convert(n, False), model, is_leaf=_is_module)
    marks = jax.tree.map(lambda n: convert(n, True), model, is_leaf=_is_module)
    return tree, marks


def _is_module(x) -> bool:
    return isinstance(x, eqx.Module)


def load_v02(path, model, *, warn_above: float = 1e-6):
    """Read the 0.2 checkpoint at ``path`` into ``model`` (built in 0.3 with the configuration of the saved one)."""
    old_like, marks = legacy_view(model)
    old = eqx.tree_deserialise_leaves(path, old_like)
    old_leaves, flags = jax.tree.leaves(old), jax.tree.leaves(marks)
    like_leaves = jax.tree.leaves(old_like)
    assert len(old_leaves) == len(flags) == len(like_leaves)
    kept = [leaf for leaf, is_const in zip(old_leaves, flags) if not is_const]
    drift = [
        float(jnp.max(jnp.abs(jnp.asarray(saved) - jnp.asarray(built))))
        for saved, built, is_const in zip(old_leaves, like_leaves, flags)
        if is_const and np.size(built)
    ]
    if drift and max(drift) > warn_above:
        warnings.warn(
            f"the saved fixed arrays differ from the ones the model builds by up to {max(drift):.3g}; "
            "the 0.3 model uses the built ones",
            stacklevel=2,
        )
    return jax.tree.unflatten(jax.tree.structure(model), kept)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "old", help="checkpoint written by foundax 0.2 (eqx.tree_serialise_leaves)"
    )
    p.add_argument("new", help="where to write the 0.3 checkpoint")
    p.add_argument(
        "--model",
        required=True,
        help='Python expression that builds the model, e.g. "fx.fno2d(2, 32, 16)"',
    )
    args = p.parse_args(argv)

    import foundax as fx

    model = eval(args.model, {"fx": fx, "foundax": fx, "jax": jax, "jnp": jnp})  # noqa: S307 (local CLI input)
    migrated = load_v02(args.old, model)
    eqx.tree_serialise_leaves(args.new, migrated)
    n = len(jax.tree.leaves(eqx.filter(migrated, eqx.is_array)))
    print(f"wrote {args.new}: {n} arrays in the 0.3 layout")


if __name__ == "__main__":
    sys.exit(main())
