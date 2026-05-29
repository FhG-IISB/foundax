"""Multi-input combinators for composable neural operators.

Combinators join two pipelines (or any callables) into a single model::

    branch = b1 > b2 > b3
    trunk  = t1 > t2

    model = foundax.dot(branch, trunk)   # DeepONet-style
    model = foundax.add(branch, trunk)   # elementwise sum
    model = foundax.cat(branch, trunk)   # channel concat
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx


class DotCombinator(eqx.Module):
    """DeepONet-style branch · trunk dot-product operator.

    ``branch(u)`` must return a 1-D basis vector of shape ``(C,)``.
    ``trunk(y)``  must return query evaluations of shape ``(N, C)``.

    The output is ``einsum("c,nc->n", branch(u), trunk(y))`` with shape
    ``(N,)``.
    """

    branch: eqx.Module
    trunk: eqx.Module

    def __call__(self, u, y):
        b = self.branch(u)  # (C,)
        t = self.trunk(y)   # (N, C)
        return jnp.einsum("c,nc->n", b, t)


class AddCombinator(eqx.Module):
    """Apply two branches to the same input and sum outputs element-wise.

    Both branches must produce the same output shape.
    """

    a: eqx.Module
    b: eqx.Module

    def __call__(self, x):
        return self.a(x) + self.b(x)


class CatCombinator(eqx.Module):
    """Apply two branches to the same input and concatenate along the channel axis.

    Output channels = ``a``'s output channels + ``b``'s output channels.
    """

    a: eqx.Module
    b: eqx.Module

    def __call__(self, x):
        return jnp.concatenate([self.a(x), self.b(x)], axis=-1)


def dot(branch, trunk) -> DotCombinator:
    """DeepONet-style operator: ``branch(u) · trunk(y) -> (N,)``.

    Args:
        branch: Callable mapping sensor data ``u`` to a basis vector ``(C,)``.
        trunk:  Callable mapping query coordinates ``y`` to evaluations ``(N, C)``.

    Returns:
        A :class:`DotCombinator` whose ``__call__(u, y)`` returns shape ``(N,)``.
    """
    return DotCombinator(branch, trunk)


def add(a, b) -> AddCombinator:
    """Elementwise-sum of two branches applied to the same input.

    Args:
        a: First branch.
        b: Second branch.  Must produce the same output shape as *a*.
    """
    return AddCombinator(a, b)


def cat(a, b) -> CatCombinator:
    """Channel-concatenation of two branches applied to the same input.

    Args:
        a: First branch.
        b: Second branch.  Output channels are concatenated along axis -1.
    """
    return CatCombinator(a, b)
