"""Pipe operator (|) and Block wrapper for composable neural operators.

Uses ``|`` rather than ``>`` because Python treats ``a > b > c`` as a chained
comparison (``(a > b) and (b > c)``), which silently drops intermediate blocks.
``|`` is left-associative with no chaining, so ``b1 | b2 | b3`` is always
``(b1 | b2) | b3`` — a flat three-block pipeline.

Usage::

    import foundax as fx
    import jax

    ks = jax.random.split(jax.random.PRNGKey(0), 4)

    b1 = fx.block(fx.layers.SpectralBlock2d(3,  32, n_modes=16, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(32, 32, n_modes=16, key=ks[1]))
    b3 = fx.block(fx.layers.SpectralBlock2d(32,  1, n_modes=16, key=ks[2]))

    branch = b1 | b2 | b3          # Pipe of 3 blocks
    model  = fx.dot(branch, trunk)  # multi-input combinator
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx

_IN_NAMES  = ("in_channels", "in_features")
_OUT_NAMES = ("out_channels", "out_features", "output_dim")


def _sniff(module: eqx.Module, names: tuple) -> Optional[int]:
    for name in names:
        val = getattr(module, name, None)
        if isinstance(val, int):
            return val
    return None


def _format_pipeline(blocks: list, mismatch_idx: Optional[int] = None) -> str:
    lines = ["  Pipeline:"]
    for i, b in enumerate(blocks):
        in_s   = str(b._in_channels)  if b._in_channels  is not None else "?"
        out_s  = str(b._out_channels) if b._out_channels is not None else "?"
        marker = "  <-- mismatch here" if i == mismatch_idx else ""
        lines.append(f"    [{i}] {b.name:<32s}  in={in_s:<6s} out={out_s}{marker}")
    return "\n".join(lines)


def _check_edge(
    left_name: str,
    left_out:  Optional[int],
    right_name: str,
    right_in:   Optional[int],
    all_blocks: list,
    right_idx:  int,
) -> None:
    if left_out is not None and right_in is not None and left_out != right_in:
        pipeline_str = _format_pipeline(all_blocks, mismatch_idx=right_idx)
        raise ShapeMismatchError(
            f"\nChannel mismatch: '{left_name}' outputs {left_out} channels "
            f"but '{right_name}' expects {right_in}.\n"
            f"{pipeline_str}\n"
            f"Hint: change '{right_name}' in_channels to {left_out}, "
            f"or insert a projection layer between them."
        )


class ShapeMismatchError(ValueError):
    """Raised when adjacent blocks in a Pipe have incompatible channel counts."""


class Block(eqx.Module):
    """Thin wrapper that adds the ``|`` pipe operator to any ``eqx.Module``.

    Create via :func:`foundax.block`.  The wrapped module is still usable on
    its own — ``Block`` is transparent for all other purposes.
    """

    module: eqx.Module
    _in_channels:  Optional[int] = eqx.field(static=True)
    _out_channels: Optional[int] = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __or__(self, other: "Block | Pipe") -> "Pipe":
        if not isinstance(other, (Block, Pipe)):
            raise TypeError(
                f"The right-hand side of '|' must be a Block or Pipe, "
                f"got {type(other).__name__}. "
                f"Wrap it with foundax.block() first."
            )
        if isinstance(other, Pipe):
            all_blocks = [self] + other.blocks
            if other.blocks:
                first = other.blocks[0]
                _check_edge(
                    self.name, self._out_channels,
                    first.name, first._in_channels,
                    all_blocks, 1,
                )
            return Pipe(all_blocks)

        all_blocks = [self, other]
        _check_edge(
            self.name, self._out_channels,
            other.name, other._in_channels,
            all_blocks, 1,
        )
        return Pipe(all_blocks)

    def __call__(self, x, **kwargs):
        return self.module(x, **kwargs)


class Pipe(eqx.Module):
    """An ordered sequence of :class:`Block` objects produced by the ``|`` operator.

    Calling a ``Pipe`` applies each block in order.  ``|`` is also defined on
    ``Pipe`` so pipelines can be extended or merged::

        extended = existing_pipe | new_block
        merged   = pipe_a | pipe_b   # flattened, not nested
    """

    blocks: list  # list[Block]

    def __or__(self, other: "Block | Pipe") -> "Pipe":
        if not isinstance(other, (Block, Pipe)):
            raise TypeError(
                f"The right-hand side of '|' must be a Block or Pipe, "
                f"got {type(other).__name__}. "
                f"Wrap it with foundax.block() first."
            )
        if isinstance(other, Pipe):
            all_blocks = self.blocks + other.blocks
            if self.blocks and other.blocks:
                last  = self.blocks[-1]
                first = other.blocks[0]
                _check_edge(
                    last.name,  last._out_channels,
                    first.name, first._in_channels,
                    all_blocks, len(self.blocks),
                )
            return Pipe(all_blocks)

        all_blocks = self.blocks + [other]
        if self.blocks:
            last = self.blocks[-1]
            _check_edge(
                last.name,  last._out_channels,
                other.name, other._in_channels,
                all_blocks, len(self.blocks),
            )
        return Pipe(all_blocks)

    def __call__(self, x, **kwargs):
        for blk in self.blocks:
            x = blk(x, **kwargs)
        return x


def block(module: eqx.Module, *, name: Optional[str] = None) -> Block:
    """Wrap any ``eqx.Module`` into the composable ``|`` pipe API.

    The module is still fully usable on its own.  ``block()`` only adds the
    ``|`` operator so the module can be chained with others.

    Channel counts are inferred automatically from static fields named
    ``in_channels``, ``in_features``, ``out_channels``, ``out_features``, or
    ``output_dim``.  When none are found the check is deferred to call time.

    Args:
        module: Any ``equinox.Module`` (a single layer, a whole FNO2D, etc.).
        name:   Display name used in error messages.  Defaults to the class name.

    Returns:
        A :class:`Block` wrapping *module*.

    Example::

        b1 = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=16, key=k1))
        b2 = fx.block(fx.layers.SpectralBlock2d(32, 1, n_modes=16, key=k2))
        pipe = b1 | b2 | b3
    """
    in_c  = _sniff(module, _IN_NAMES)
    out_c = _sniff(module, _OUT_NAMES)
    label = name or type(module).__name__
    return Block(module, in_c, out_c, label)
