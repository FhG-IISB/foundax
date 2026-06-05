import jax
import jax.numpy as jnp
import pytest

import foundax as fx
from foundax.architectures.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    MultiHeadAttention,
    PositionalEncoding,
    causal_mask,
)


KEY = jax.random.PRNGKey(0)


def test_factory_smoke():
    model = fx.transformer(
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        mlp_features=64,
        vocab_size=100,
        max_len=16,
        dropout_rate=0.0,
    )
    enc_tokens = jnp.arange(8)
    dec_tokens = jnp.arange(8)
    out = model(enc_tokens, dec_tokens)
    assert out.shape == (8, 100)
    assert jnp.all(jnp.isfinite(out))


def test_encoder_smoke():
    enc = TransformerEncoder(
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        qkv_features=32,
        mlp_features=64,
        dropout_rate=0.0,
        vocab_size=100,
        max_len=16,
        key=KEY,
    )
    tokens = jnp.arange(8)
    out = enc(tokens)
    assert out.shape == (8, 32)
    assert jnp.all(jnp.isfinite(out))


def test_decoder_with_causal_mask():
    enc = TransformerEncoder(
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        qkv_features=32,
        mlp_features=64,
        dropout_rate=0.0,
        vocab_size=100,
        max_len=16,
        key=jax.random.PRNGKey(1),
    )
    dec = TransformerDecoder(
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        qkv_features=32,
        mlp_features=64,
        dropout_rate=0.0,
        vocab_size=100,
        max_len=16,
        key=jax.random.PRNGKey(2),
    )
    src = jnp.arange(8)
    tgt = jnp.arange(8)
    enc_out = enc(src)
    out = dec(tgt, enc_out, decoder_self_attention_mask=causal_mask(8))
    assert out.shape == (8, 100)
    assert jnp.all(jnp.isfinite(out))


def test_causal_mask_blocks_future_positions():
    """Changing target token at the last position must not change earlier outputs."""
    dec = TransformerDecoder(
        num_layers=2,
        embed_dim=16,
        num_heads=2,
        qkv_features=16,
        mlp_features=32,
        dropout_rate=0.0,
        vocab_size=50,
        max_len=8,
        key=jax.random.PRNGKey(3),
    )
    enc_out = jax.random.normal(jax.random.PRNGKey(4), (8, 16))
    mask = causal_mask(8)

    tgt_a = jnp.arange(8)
    tgt_b = jnp.arange(8).at[-1].set(42)

    out_a = dec(tgt_a, enc_out, decoder_self_attention_mask=mask)
    out_b = dec(tgt_b, enc_out, decoder_self_attention_mask=mask)

    assert jnp.allclose(out_a[:-1], out_b[:-1], atol=1e-5)
    assert not jnp.allclose(out_a[-1], out_b[-1], atol=1e-5)


def test_qkv_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(
            in_features=32,
            qkv_features=30,
            out_features=32,
            num_heads=4,
            key=KEY,
        )


def test_positional_encoding_shape_and_consistency():
    pe = PositionalEncoding(max_len=16, embed_dim=8)

    x = jnp.zeros((10, 8))
    y = pe(x)
    assert y.shape == (10, 8)

    x2 = jnp.zeros((5, 8))
    y2 = pe(x2)
    assert y2.shape == (5, 8)

    # Same positions must produce the same encoding when called on shorter sequences.
    assert jnp.allclose(y2, y[:5])


def test_dropout_runs_when_key_provided():
    """With dropout > 0 and key passed, forward must still run and produce finite values."""
    model = fx.transformer(
        num_layers=2,
        embed_dim=16,
        num_heads=2,
        mlp_features=32,
        vocab_size=50,
        max_len=8,
        dropout_rate=0.1,
    )
    enc = jnp.arange(6)
    dec = jnp.arange(6)
    out = model(enc, dec, key=jax.random.PRNGKey(99))
    assert out.shape == (6, 50)
    assert jnp.all(jnp.isfinite(out))
