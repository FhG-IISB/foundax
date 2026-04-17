"""Tests for the Equinox Walrus model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_walrus"))

from jax_walrus.model_eqx import IsotropicModel as EqxModel


_CFG = dict(
    hidden_dim=64,
    intermediate_dim=32,
    n_states=4,
    processor_blocks=2,
    groups=4,
    num_heads=4,
    mlp_dim=0,
    max_d=3,
    causal_in_time=False,
    drop_path=0.0,
    bias_type="rel",
    base_kernel_size=((4, 2), (4, 2), (4, 2)),
    use_spacebag=True,
    use_silu=True,
    include_d=(2,),
    encoder_groups=4,
    jitter_patches=False,
    learned_pad=True,
)


class TestWalrusEqx:
    @pytest.fixture(autouse=True)
    def setup(self):
        eqx_cfg = {
            k: v
            for k, v in _CFG.items()
            if k not in ("jitter_patches", "remat", "input_field_drop")
        }
        self.model = EqxModel(**eqx_cfg, key=jax.random.PRNGKey(0))

    def test_forward_2d(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        out = self.model(x, state_labels, bcs, deterministic=True)
        assert jnp.all(jnp.isfinite(out))

    def test_forward_2d_periodic(self):
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[2, 2], [2, 2]]
        out = self.model(x, state_labels, bcs, deterministic=True)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self):
        x = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        out = self.model(x, state_labels, bcs, deterministic=True)
        assert out.shape == (1, 1, 16, 16, 4), f"Shape {out.shape}"

    def test_deterministic_consistency(self):
        x = jax.random.normal(jax.random.PRNGKey(3), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        out1 = np.array(self.model(x, state_labels, bcs, deterministic=True))
        out2 = np.array(self.model(x, state_labels, bcs, deterministic=True))
        assert np.allclose(
            out1, out2, atol=0
        ), "Non-deterministic in deterministic mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
