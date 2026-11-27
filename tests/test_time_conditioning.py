"""Time embedding and DiT conditioning details (B8)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402

KEY = jax.random.PRNGKey(0)


def _sinusoid(t, dim):
    """The DDPM features (Ho et al.): frequencies 10000^(-i / (half - 1))."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / max(half - 1, 1))
    return np.concatenate([np.sin(t * freqs), np.cos(t * freqs)])


def test_embedding_of_two_dimensions_is_finite():
    emb = fx.SinusoidalTimeEmbedding(2, key=KEY)
    a, b = emb(jnp.asarray(0.3)), emb(jnp.asarray(0.7))
    assert bool(jnp.all(jnp.isfinite(a))) and float(jnp.abs(a - b).max()) > 0


def test_features_unchanged_for_wider_embeddings():
    """dim > 2 keeps the DDPM frequencies, so trained models keep their outputs."""
    emb = fx.SinusoidalTimeEmbedding(16, key=KEY)
    want = emb.linear2(
        jax.nn.gelu(emb.linear1(jnp.asarray(_sinusoid(0.37, 16), jnp.float32)))
    )
    np.testing.assert_allclose(emb(jnp.asarray(0.37)), want, rtol=1e-5, atol=1e-6)


def test_t_scale():
    plain = fx.SinusoidalTimeEmbedding(16, key=KEY)
    scaled = fx.SinusoidalTimeEmbedding(16, t_scale=1000.0, key=KEY)
    np.testing.assert_allclose(
        scaled(jnp.asarray(0.25)), plain(jnp.asarray(250.0)), rtol=1e-5, atol=1e-5
    )


def test_embedding_keeps_double_precision():
    with jax.enable_x64(True):
        emb = fx.SinusoidalTimeEmbedding(16, key=KEY)
        out = emb(jnp.asarray(0.37, jnp.float64))
        assert out.dtype == jnp.float64
        np.testing.assert_allclose(
            emb.linear2(jax.nn.gelu(emb.linear1(jnp.asarray(_sinusoid(0.37, 16))))),
            out,
            rtol=1e-12,
        )


DITS = {
    "dit2d": (
        lambda **kw: fx.dit2d(
            2, patch_size=4, hidden_size=16, depth=1, num_heads=2, key=KEY, **kw
        ),
        (16, 16, 2),
    ),
    "dit3d": (
        lambda **kw: fx.dit3d(
            2, patch_size=4, hidden_size=12, depth=1, num_heads=2, key=KEY, **kw
        ),
        (8, 8, 8, 2),
    ),
}


@pytest.mark.parametrize("name", DITS)
def test_dit_final_layer_is_zero_initialised(name):
    """As in DiT (Peebles & Xie): the final adaLN projection and output layer start at zero, so does the output."""
    build, shape = DITS[name]
    model = build()
    final = model.final_layer
    for lin in (final.adaLN_proj, final.proj):
        assert not np.any(lin.weight) and (lin.bias is None or not np.any(lin.bias))
    assert not np.any(model(jax.random.normal(KEY, shape), jnp.asarray(0.5)))


@pytest.mark.parametrize("name", DITS)
def test_dit_t_scale_reaches_the_embedding(name):
    build, _ = DITS[name]
    assert build(t_scale=1000.0).time_embed.t_scale == 1000.0


@pytest.mark.parametrize("factory", ["transolver", "transolver2d", "transolver3d"])
def test_transolver_t_scale_reaches_the_embedding(factory):
    kwargs = {"space_dim": 2, "fun_dim": 1} if factory == "transolver" else {}
    model = getattr(fx, factory)(
        **kwargs,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        n_slices=4,
        time_input=True,
        t_scale=1000.0,
        key=KEY,
    )
    embeddings = [
        m
        for m in jax.tree.leaves(
            model, is_leaf=lambda m: isinstance(m, fx.SinusoidalTimeEmbedding)
        )
        if isinstance(m, fx.SinusoidalTimeEmbedding)
    ]
    assert embeddings and all(e.t_scale == 1000.0 for e in embeddings)
