"""Tests verifying Equinox Walrus matches Flax Walrus forward pass."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from jax_walrus.model import IsotropicModel as FlaxModel
from jax_walrus.model_eqx import IsotropicModel as EqxModel, transfer_weights


# ---------------------------------------------------------------------------
# Small test config (2D only to keep it fast)
# ---------------------------------------------------------------------------
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


def _init_flax(cfg):
    """Initialize Flax model and return (model, params)."""
    model = FlaxModel(**cfg)
    # 2D input: (B, T, H, W, C)
    B, T, H, W, C = 1, 2, 16, 16, 4
    x = jnp.ones((B, T, H, W, C))
    state_labels = jnp.array([0, 1, 2, 3])
    bcs = [[0, 0], [0, 0]]
    rng = jax.random.PRNGKey(42)
    variables = model.init(
        {"params": rng, "dropout": rng, "drop_path": rng, "jitter": rng},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    return model, variables


def _init_eqx(cfg):
    """Initialize Equinox model."""
    # Remove Flax-only keys
    eqx_cfg = {
        k: v
        for k, v in cfg.items()
        if k not in ("jitter_patches", "remat", "input_field_drop")
    }
    key = jax.random.PRNGKey(99)
    return EqxModel(**eqx_cfg, key=key)


def _transfer(flax_vars, eqx_model):
    """Transfer Flax params to Equinox model."""
    return transfer_weights(flax_vars, eqx_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWalrusEqx:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.flax_model, self.flax_vars = _init_flax(_CFG)
        self.eqx_model = _init_eqx(_CFG)
        self.eqx_model = _transfer(self.flax_vars, self.eqx_model)

    def _run_both(self, x, state_labels, bcs):
        flax_out = self.flax_model.apply(
            self.flax_vars,
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        eqx_out = self.eqx_model(
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        return np.array(flax_out), np.array(eqx_out)

    def test_forward_2d(self):
        """2D forward pass equivalence."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        flax_out, eqx_out = self._run_both(x, state_labels, bcs)
        diff = np.max(np.abs(flax_out - eqx_out))
        print(f"2D forward max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"

    def test_forward_2d_periodic(self):
        """2D forward with periodic BCs."""
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[2, 2], [2, 2]]  # periodic
        flax_out, eqx_out = self._run_both(x, state_labels, bcs)
        diff = np.max(np.abs(flax_out - eqx_out))
        print(f"2D periodic forward max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"

    def test_output_shape(self):
        """Output shape correctness."""
        x = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        eqx_out = self.eqx_model(
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        # Non-causal: T_out=1
        assert eqx_out.shape == (1, 1, 16, 16, 4), f"Shape {eqx_out.shape}"

    def test_deterministic_consistency(self):
        """Same input → same output (deterministic mode)."""
        x = jax.random.normal(jax.random.PRNGKey(3), (1, 2, 16, 16, 4))
        state_labels = jnp.array([0, 1, 2, 3])
        bcs = [[0, 0], [0, 0]]
        out1 = np.array(self.eqx_model(x, state_labels, bcs, deterministic=True))
        out2 = np.array(self.eqx_model(x, state_labels, bcs, deterministic=True))
        assert np.allclose(
            out1, out2, atol=0
        ), "Non-deterministic in deterministic mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
