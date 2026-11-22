"""The FNO spectral layers compute their transforms as truncated DFTs; these tests pin them to the FFT
formulation they replace (rfftn at the transform size -> keep modes -> mix -> irfftn -> crop).

The reference below is the previous FFT implementation, generalised only in its complex dtype so it can
run in float64. Cases cover both transform sizes (linear_conv), odd and even sizes, mode counts that are
clipped, the real axis's Nyquist mode, and upper/lower blocks that keep the same frequency (the later
block wins, as the sequential ``.at[].set`` did).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foundax.architectures.fno import SpectralConv1d, SpectralConv2d, SpectralConv3d


def _reference(x, weights, n_modes, out_channels, linear_conv):
    """The FFT version: rfftn, per-block mode mixing written in block order, irfftn, crop."""
    spatial = x.shape[:-1]
    k = len(spatial) - 1
    sizes = [2 * n - 1 if linear_conv else n for n in spatial]
    X = jnp.fft.rfftn(x, s=sizes, axes=tuple(range(k + 1)), norm="ortho")
    m = [min(n_modes[a], (X.shape[a] + 1) // 2) for a in range(k)] + [
        min(n_modes[k], X.shape[k])
    ]
    out = jnp.zeros(X.shape[:-1] + (out_channels,), X.dtype)
    for w, halves in zip(weights, itertools.product((0, 1), repeat=k)):
        sl = tuple(
            slice(0, m[a]) if h == 0 else slice(X.shape[a] - m[a], X.shape[a])
            for a, h in enumerate(halves)
        ) + (slice(0, m[k]),)
        wb = w[(slice(None), slice(None)) + tuple(slice(0, mm) for mm in m)]
        letters = "abcd"[: k + 1]
        out = out.at[sl].set(
            jnp.einsum(f"{letters}i,io{letters}->{letters}o", X[sl], wb)
        )
    y = jnp.fft.irfftn(out, s=sizes, axes=tuple(range(k + 1)), norm="ortho")
    return y[tuple(slice(0, n) for n in spatial)]


def _layer(dim, c_in, c_out, modes, linear_conv, seed):
    key = jax.random.PRNGKey(seed)
    if dim == 1:
        lay = SpectralConv1d(c_in, c_out, modes[0], linear_conv=linear_conv, key=key)
        ws = [lay.weight_real + 1j * lay.weight_imag]
    elif dim == 2:
        lay = SpectralConv2d(c_in, c_out, *modes, linear_conv=linear_conv, key=key)
        ws = [
            lay.weight_1_real + 1j * lay.weight_1_imag,
            lay.weight_2_real + 1j * lay.weight_2_imag,
        ]
    else:
        lay = SpectralConv3d(c_in, c_out, *modes, linear_conv=linear_conv, key=key)
        ws = [
            getattr(lay, f"weight_{i}_real") + 1j * getattr(lay, f"weight_{i}_imag")
            for i in (1, 2, 3, 4)
        ]
    return lay, ws


CASES = [
    # (spatial shape, requested modes) -- with linear_conv False and True each
    ((7,), (3,)),
    ((8,), (5,)),  # non-linear: 8 // 2 + 1 = 5 keeps the Nyquist mode
    ((6,), (50,)),  # clipped
    ((6, 7), (2, 3)),
    ((8, 8), (4, 5)),  # non-linear: Nyquist on the real axis
    (
        (5, 4),
        (3, 3),
    ),  # non-linear: 3 + 3 rows of 5 overlap in one row; real axis clipped to 3
    ((4, 6), (9, 9)),  # clipped on both axes
    ((5, 4, 6), (2, 2, 3)),
    (
        (5, 5, 4),
        (3, 3, 3),
    ),  # non-linear: overlap along depth and height; Nyquist on the real axis
    ((4, 6, 5), (7, 7, 7)),  # clipped everywhere
]


@pytest.mark.parametrize("linear_conv", [False, True])
@pytest.mark.parametrize("shape,modes", CASES)
def test_matches_the_fft_formulation(shape, modes, linear_conv):
    lay, ws = _layer(len(shape), 3, 2, modes, linear_conv, seed=len(shape))
    x = jax.random.normal(jax.random.PRNGKey(7), shape + (3,), jnp.float32)
    got, ref = lay(x), _reference(x, ws, modes, 2, linear_conv)
    assert got.shape == ref.shape == shape + (2,)
    assert got.dtype == jnp.float32
    scale = float(jnp.abs(ref).max())
    assert float(jnp.abs(got - ref).max()) <= 1e-5 * scale, (
        "truncated DFT differs from the FFT formulation"
    )


@pytest.mark.parametrize("shape,modes", [CASES[1], CASES[5], CASES[8]])
def test_gradients_match_the_fft_formulation(shape, modes):
    # Gradients with respect to the input and the weights: the backward pass is the same map's adjoint.
    lay, ws = _layer(len(shape), 3, 2, modes, False, seed=1)
    x = jax.random.normal(jax.random.PRNGKey(2), shape + (3,), jnp.float32)
    probe = jax.random.normal(jax.random.PRNGKey(3), shape + (2,), jnp.float32)
    gx = jax.grad(lambda z: jnp.sum(lay(z) * probe))(x)
    rx = jax.grad(lambda z: jnp.sum(_reference(z, ws, modes, 2, False) * probe))(x)
    assert float(jnp.abs(gx - rx).max()) <= 1e-5 * float(jnp.abs(rx).max())
    name = "weight_real" if len(shape) == 1 else "weight_1_real"
    gw = jax.grad(
        lambda v: jnp.sum(type(lay).__call__(_with(lay, name, v), x) * probe)
    )(getattr(lay, name))
    ws_of = lambda v: (
        [v + 1j * (lay.weight_imag if len(shape) == 1 else lay.weight_1_imag)] + ws[1:]
    )  # noqa: E731
    rw = jax.grad(lambda v: jnp.sum(_reference(x, ws_of(v), modes, 2, False) * probe))(
        getattr(lay, name)
    )
    assert float(jnp.abs(gw - rw).max()) <= 1e-5 * float(jnp.abs(rw).max())


def _with(lay, name, value):
    import equinox as eqx

    return eqx.tree_at(lambda m: getattr(m, name), lay, value)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_no_fft_is_traced(dim):
    # The fast path: the layer lowers to dense contractions only. An FFT in the jaxpr means the full
    # (padded) transform came back.
    shape = (9,) * dim
    lay, _ = _layer(dim, 4, 4, (3,) * dim, True, seed=0)
    jaxpr = jax.make_jaxpr(lay)(jnp.zeros(shape + (4,), jnp.float32))
    prims = _primitives(jaxpr.jaxpr)
    assert "fft" not in prims, prims
    assert "dot_general" in prims


def _primitives(jaxpr):
    """Every primitive in a jaxpr, including those inside nested jit / custom-derivative sub-jaxprs
    (``jnp.fft`` itself traces as a nested jit, so a top-level scan would never see the FFT)."""
    out = set()
    for e in jaxpr.eqns:
        out.add(e.primitive.name)
        for p in e.params.values():
            for sub in p if isinstance(p, (list, tuple)) else [p]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    out |= _primitives(inner)
    return out


def test_float64_inputs_stay_float64_and_match_tightly():
    # The FFT version forced complex64 for the 2-D and 3-D layers; the DFT keeps the input's precision.
    enable_x64 = (
        jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    )  # moved in newer JAX
    with enable_x64(True):
        lay, ws = _layer(2, 3, 2, (3, 3), True, seed=4)
        lay = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.float64) if isinstance(a, jnp.ndarray) else a, lay
        )
        ws = [w.astype(jnp.complex128) for w in ws]
        x = jax.random.normal(jax.random.PRNGKey(5), (6, 7, 3), jnp.float64)
        got, ref = lay(x), _reference(x, ws, (3, 3), 2, True)
        assert got.dtype == jnp.float64
        np.testing.assert_allclose(
            np.asarray(got),
            np.asarray(ref),
            rtol=0,
            atol=1e-13 * float(jnp.abs(ref).max()),
        )
