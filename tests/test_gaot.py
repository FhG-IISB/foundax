import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import foundax as fx
from foundax.architectures.gaot import (
    GAOT,
    AGNO,
    MAGNOConfig,
    TransformerConfig,
    AttentionConfig,
    compute_neighbors_csr,
)


KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _grid_latent(H=8):
    xs = np.linspace(0, 1, H, dtype=np.float32)
    return jnp.asarray(np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2))


def _toy_geometry(H=8, N=40, M=20, radius=0.3, seed=0):
    latent = _grid_latent(H)
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.random((N, 2)).astype(np.float32))
    q = jnp.asarray(rng.random((M, 2)).astype(np.float32))
    enc = [compute_neighbors_csr(np.asarray(x), np.asarray(latent), radius)]
    dec = [compute_neighbors_csr(np.asarray(latent), np.asarray(q), radius)]
    return latent, x, q, enc, dec


def _tiny_gaot(**mc_overrides):
    defaults = dict(
        radius=0.3,
        lifting_channels=8,
        hidden_size=16,
        mlp_layers=2,
        use_attention=False,
        use_geoembed=False,
    )
    defaults.update(mc_overrides)
    mc = MAGNOConfig(**defaults)
    tc = TransformerConfig(
        patch_size=2,
        hidden_size=16,
        num_layers=2,
        attn_config=AttentionConfig(num_heads=2, num_kv_heads=2),
    )
    return GAOT(
        input_size=2,
        output_size=1,
        magno_config=mc,
        transformer_config=tc,
        latent_tokens_size=(8, 8),
        key=KEY,
    )


# ---------------------------------------------------------------------------
# compute_neighbors_csr
# ---------------------------------------------------------------------------


def test_neighbors_csr_keys_present():
    src = np.random.default_rng(0).random((20, 2)).astype(np.float32)
    qry = np.random.default_rng(1).random((10, 2)).astype(np.float32)
    d = compute_neighbors_csr(src, qry, radius=2.0)
    for k in ("neighbors_index", "neighbors_row_splits", "seg_ids", "counts"):
        assert k in d


def test_neighbors_csr_row_splits_and_counts_shape():
    src = np.random.default_rng(0).random((20, 2)).astype(np.float32)
    qry = np.random.default_rng(1).random((10, 2)).astype(np.float32)
    d = compute_neighbors_csr(src, qry, radius=2.0)
    assert d["neighbors_row_splits"].shape[0] == len(qry) + 1
    assert d["counts"].shape[0] == len(qry)
    # row_splits[-1] must equal total edge count
    assert int(d["neighbors_row_splits"][-1]) == int(d["neighbors_index"].shape[0])


def test_neighbors_csr_no_neighbors_for_tiny_radius():
    """Radius smaller than nearest-neighbor distance → empty edge list."""
    src = np.array([[0.0, 0.0]], dtype=np.float32)
    qry = np.array([[1.0, 1.0]], dtype=np.float32)
    d = compute_neighbors_csr(src, qry, radius=0.01)
    assert int(d["counts"][0]) == 0
    assert d["neighbors_index"].shape[0] == 0


# ---------------------------------------------------------------------------
# Full GAOT model — forward pass
# ---------------------------------------------------------------------------


def test_gaot_output_shape_and_finite():
    model = _tiny_gaot()
    latent, x, q, enc, dec = _toy_geometry()
    pn = jax.random.normal(KEY, (40, 2))
    out = model(latent, x, pn, q, enc, dec)
    assert out.shape == (20, 1)
    assert jnp.all(jnp.isfinite(out))


def test_query_coord_none_decodes_onto_input_mesh():
    """When query_coord=None the output has the same leading dim as x_coord."""
    model = _tiny_gaot()
    latent, x, _, enc, _ = _toy_geometry()
    pn = jax.random.normal(KEY, (40, 2))
    dec = [compute_neighbors_csr(np.asarray(latent), np.asarray(x), 0.3)]
    out = model(latent, x, pn, query_coord=None, encoder_nbrs=enc, decoder_nbrs=dec)
    assert out.shape == (40, 1)
    assert jnp.all(jnp.isfinite(out))


def test_gaot_geoembed_statistical():
    model = _tiny_gaot(
        use_geoembed=True,
        embedding_method="statistical",
        use_attention=True,
        attention_type="cosine",
    )
    latent, x, q, enc, dec = _toy_geometry()
    out = model(latent, x, jax.random.normal(KEY, (40, 2)), q, enc, dec)
    assert out.shape == (20, 1)
    assert jnp.all(jnp.isfinite(out))


def test_gaot_dot_product_attn():
    model = _tiny_gaot(use_attention=True, attention_type="dot_product")
    latent, x, q, enc, dec = _toy_geometry()
    out = model(latent, x, jax.random.normal(KEY, (40, 2)), q, enc, dec)
    assert out.shape == (20, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------


def test_fx_gaot_callable():
    """fx.gaot(...) flat-kwarg API must return a working model."""
    model = fx.gaot(
        input_size=2,
        output_size=1,
        radius=0.3,
        lifting_channels=8,
        hidden_size_mlp=16,
        mlp_layers=2,
        use_attention=False,
        use_geoembed=False,
        transformer_hidden_size=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        patch_size=2,
        latent_tokens_size=(8, 8),
        key=KEY,
    )
    latent, x, q, enc, dec = _toy_geometry()
    out = model(latent, x, jax.random.normal(KEY, (40, 2)), q, enc, dec)
    assert out.shape == (20, 1)


def test_gaot_S_shape():
    """fx.gaot.S with a tiny latent grid must produce the right output shape."""
    model = fx.gaot.S(input_size=2, output_size=1, latent_tokens_size=(8, 8), key=KEY)
    latent, x, q, enc, dec = _toy_geometry()
    out = model(latent, x, jax.random.normal(KEY, (40, 2)), q, enc, dec)
    assert out.shape == (20, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_encoder():
    model = _tiny_gaot()
    latent, x, q, enc, dec = _toy_geometry()
    pn = jax.random.normal(KEY, (40, 2))
    yt = jax.random.normal(KEY, (20, 1))

    def loss(m):
        return jnp.mean((m(latent, x, pn, q, enc, dec) - yt) ** 2)

    grads = eqx.filter_grad(loss)(model)
    assert jnp.all(jnp.isfinite(grads.encoder.agno.channel_mlp.fcs[0].weight))
    assert jnp.all(jnp.isfinite(grads.patch_linear.weight))


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


def test_jit_compatibility():
    model = _tiny_gaot()
    latent, x, q, enc, dec = _toy_geometry()
    pn = jax.random.normal(KEY, (40, 2))

    @eqx.filter_jit
    def fwd(m, lat, xc, pn, qc, en, dn):
        return m(lat, xc, pn, qc, en, dn)

    out = fwd(model, latent, x, pn, q, enc, dec)
    assert out.shape == (20, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# vmap batching
# ---------------------------------------------------------------------------


def test_vmap_batch():
    model = _tiny_gaot()
    latent, x, q, enc, dec = _toy_geometry()
    pn_batch = jax.random.normal(KEY, (3, 40, 2))

    out = jax.vmap(lambda f: model(latent, x, f, q, enc, dec))(pn_batch)
    assert out.shape == (3, 20, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# AGNO sub-module
# ---------------------------------------------------------------------------


def _agno_nbrs(N=20, M=10):
    rng = np.random.default_rng(0)
    y = rng.standard_normal((N, 2)).astype(np.float32)
    x = rng.standard_normal((M, 2)).astype(np.float32)
    return (
        jnp.asarray(y),
        jnp.asarray(x),
        compute_neighbors_csr(y, x, radius=100.0),  # large radius → all edges present
    )


def test_agno_no_attn_shape():
    y, x, nbrs = _agno_nbrs()
    agno = AGNO([4, 16, 8], transform_type="linear", use_attn=False, key=KEY)
    out = agno(y=y, neighbors=nbrs, x=x)
    assert out.shape == (10, 8)
    assert jnp.all(jnp.isfinite(out))


def test_agno_cosine_attn_shape():
    y, x, nbrs = _agno_nbrs()
    agno = AGNO(
        [4, 16, 8],
        transform_type="linear",
        use_attn=True,
        attention_type="cosine",
        coord_dim=2,
        key=KEY,
    )
    out = agno(y=y, neighbors=nbrs, x=x)
    assert out.shape == (10, 8)
    assert jnp.all(jnp.isfinite(out))


def test_agno_dot_product_attn_shape():
    y, x, nbrs = _agno_nbrs()
    agno = AGNO(
        [4, 16, 8],
        transform_type="linear",
        use_attn=True,
        attention_type="dot_product",
        coord_dim=2,
        key=KEY,
    )
    out = agno(y=y, neighbors=nbrs, x=x)
    assert out.shape == (10, 8)
    assert jnp.all(jnp.isfinite(out))


def test_invalid_transform_type_raises():
    with pytest.raises(ValueError, match="transform_type"):
        AGNO([4, 16, 8], transform_type="invalid", key=KEY)
