"""Tests for KAN structural blocks (Conv1d/3d, SpectralBlock1d/2d/3d, ResBlock, AttentionBlock)."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx
from foundax.architectures.kan import (
    KANConv1d, KANConv3d,
    KANSpectralBlock1d, KANSpectralBlock2d, KANSpectralBlock3d,
    KANResBlock, KANAttentionBlock,
)


def _ks(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


# ── KANConv1d ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("basis", [
    "bspline", "rbf", "fourier", "chebyshev", "wavelet",
    "hermite", "bernstein", "rational",
])
def test_kan_conv1d_shape(basis):
    conv = KANConv1d(in_channels=3, out_channels=6, kernel_size=3,
                     basis=basis, key=_ks(1)[0])
    y = conv(jnp.ones((16, 3)))
    assert y.shape == (16, 6)


def test_kan_conv1d_pipe():
    a = fx.block(KANConv1d(2, 8, basis="rbf", key=_ks(2)[0]))
    b = fx.block(KANConv1d(8, 1, basis="chebyshev", key=_ks(2)[1]))
    pipe = a | b
    assert pipe(jnp.ones((10, 2))).shape == (10, 1)


# ── KANConv3d ──────────────────────────────────────────────────────────────


def test_kan_conv3d_shape():
    conv = KANConv3d(in_channels=2, out_channels=4, kernel_size=3,
                     basis="rbf", key=_ks(1)[0])
    y = conv(jnp.ones((4, 4, 4, 2)))
    assert y.shape == (4, 4, 4, 4)


def test_kan_conv3d_grad():
    conv = KANConv3d(in_channels=2, out_channels=2, kernel_size=3,
                     basis="chebyshev", key=_ks(1)[0])
    x = jax.random.normal(_ks(1, seed=3)[0], (4, 4, 4, 2))
    grad = eqx.filter_grad(lambda m: jnp.mean(m(x) ** 2))(conv)
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_array(g)]
    assert leaves
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))


# ── KANSpectralBlock1d/2d/3d ───────────────────────────────────────────────


def test_kan_spectral_block1d():
    blk = KANSpectralBlock1d(4, 8, n_modes=4, basis="rbf", key=_ks(1)[0])
    assert blk(jnp.ones((16, 4))).shape == (16, 8)
    # field-sniff
    b = fx.block(blk)
    assert b._in_channels == 4 and b._out_channels == 8


def test_kan_spectral_block2d():
    blk = KANSpectralBlock2d(4, 8, n_modes=4, basis="chebyshev", key=_ks(1)[0])
    assert blk(jnp.ones((8, 8, 4))).shape == (8, 8, 8)


def test_kan_spectral_block3d():
    blk = KANSpectralBlock3d(2, 4, n_modes=2, basis="fourier", key=_ks(1)[0])
    assert blk(jnp.ones((4, 4, 4, 2))).shape == (4, 4, 4, 4)


def test_kan_spectral_block2d_pipe_jit():
    p = (
        fx.block(KANSpectralBlock2d(2, 16, n_modes=4, basis="rbf", key=_ks(2)[0]))
        | fx.block(KANSpectralBlock2d(16, 1, n_modes=4, basis="rbf", key=_ks(2)[1]))
    )

    @eqx.filter_jit
    def fwd(m, x):
        return m(x)

    y = fwd(p, jnp.ones((8, 8, 2)))
    assert y.shape == (8, 8, 1)


# ── KANResBlock ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("basis", ["bspline", "rbf", "chebyshev"])
def test_kan_res_block_shape(basis):
    blk = KANResBlock(features=8, basis=basis, key=_ks(1)[0])
    y = blk(jnp.ones((5, 8)))
    assert y.shape == (5, 8)


def test_kan_res_block_actually_residual():
    """With zero-weight inner layers, output ≈ x (residual identity).

    Uses ``rbf`` which has no SiLU skip in the inner KAN, so zeroing
    ``spline_weight`` is sufficient to make the inner contribution vanish.
    """
    blk = KANResBlock(features=8, basis="rbf", key=_ks(1)[0])
    assert blk.kan1.skip is None and blk.kan2.skip is None
    blk = eqx.tree_at(lambda m: (m.kan1.spline_weight, m.kan2.spline_weight),
                      blk, (jnp.zeros_like(blk.kan1.spline_weight),
                            jnp.zeros_like(blk.kan2.spline_weight)))
    x = jax.random.normal(_ks(1, seed=4)[0], (5, 8))
    y = blk(x)
    assert jnp.allclose(y, x, atol=1e-6)


def test_kan_res_block_with_layer_norm():
    blk = KANResBlock(features=8, basis="rbf", use_layer_norm=True,
                      key=_ks(1)[0])
    y = blk(jnp.ones((5, 8)))
    assert y.shape == (5, 8)


# ── KANAttentionBlock ──────────────────────────────────────────────────────


def test_kan_attention_block_shape():
    blk = KANAttentionBlock(features=16, num_heads=4, basis="rbf",
                            key=_ks(1)[0])
    y = blk(jnp.ones((10, 16)))
    assert y.shape == (10, 16)


def test_kan_attention_block_jit():
    blk = KANAttentionBlock(features=16, num_heads=4, basis="chebyshev",
                            key=_ks(1)[0])

    @eqx.filter_jit
    def fwd(m, x):
        return m(x)

    y = fwd(blk, jax.random.normal(_ks(1, seed=5)[0], (8, 16)))
    assert y.shape == (8, 16)


def test_kan_attention_block_grad():
    blk = KANAttentionBlock(features=16, num_heads=4, basis="rbf",
                            key=_ks(1)[0])
    x = jax.random.normal(_ks(1, seed=6)[0], (5, 16))
    grad = eqx.filter_grad(lambda m: jnp.mean(m(x) ** 2))(blk)
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_array(g)]
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))


# ── Cross-cutting factory presence checks ──────────────────────────────────


def test_all_new_factories_exposed():
    for name in (
        "hermite_kan", "laguerre_kan", "bernstein_kan", "relu_kan",
        "rational_kan", "sinc_kan", "gram_kan", "bsrbf_kan",
        "kan_conv1d", "kan_conv3d",
        "kan_res_block", "kan_attention_block",
        "kan_spectral_block1d", "kan_spectral_block2d", "kan_spectral_block3d",
    ):
        assert hasattr(fx, name), f"foundax missing factory '{name}'"


def test_all_new_layers_in_layers_namespace():
    for name in (
        "HermiteKANLayer", "LaguerreKANLayer", "BernsteinKANLayer",
        "ReLUKANLayer", "RationalKANLayer", "SincKANLayer",
        "GramKANLayer", "BSRBFKANLayer",
        "KANConv1d", "KANConv3d",
        "KANResBlock", "KANAttentionBlock",
        "KANSpectralBlock1d", "KANSpectralBlock2d", "KANSpectralBlock3d",
    ):
        assert hasattr(fx.layers, name), f"fx.layers missing '{name}'"
