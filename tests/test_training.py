"""Training tests: actual gradient-descent loops using optax.

Requires the ``dev`` pixi environment (``pixi run -e dev test-train``).
All tests are marked ``train`` and are skipped when optax is not importable.

Each test follows the same pattern:
  1. Build a trivial regression dataset.
  2. Build a model from the foundax pipe API.
  3. Run N optimiser steps.
  4. Assert the loss strictly decreased.
  5. Assert the named weights actually changed.
"""

import math
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
optax = pytest.importorskip("optax")

import foundax as fx  # noqa: E402
import foundax.layers as fl  # noqa: E402

pytestmark = pytest.mark.train


# ── helpers ──────────────────────────────────────────────────────────────────


def ks(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


def _make_step_fn(model, opt):
    """Return a JIT-compiled (model, opt_state, x, y) → (model, opt_state, loss) fn."""

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        def loss_fn(m):
            return jnp.mean((m(x) - y) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss

    return step


def _make_step_fn_2in(model, opt):
    """Variant for models with two inputs (u, y) and scalar output."""

    @eqx.filter_jit
    def step(model, opt_state, u, y, target):
        def loss_fn(m):
            return jnp.mean((m(u, y) - target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss

    return step


def _train(model, opt, step_fn, *data, n_steps=20):
    """Run ``n_steps`` of gradient descent; return (initial_loss, final_loss, model)."""
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    losses = []
    for _ in range(n_steps):
        model, opt_state, loss = step_fn(model, opt_state, *data)
        losses.append(float(loss))
    return losses[0], losses[-1], model


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 – 1-D spectral pipe: learn sin(2πx)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrain1DSpectralPipe:
    """A two-block 1-D pipe should be able to fit sin(2πx) in a few Adam steps."""

    def _model(self):
        k1, k2 = ks(2)
        return fx.block(fl.SpectralBlock1d(1, 16, n_modes=8, key=k1)) | fx.block(
            fl.SpectralBlock1d(16, 1, n_modes=8, key=k2)
        )

    def _data(self):
        x = jnp.linspace(0, 1, 64)[:, None]  # (64, 1)
        y = jnp.sin(2 * jnp.pi * x)  # (64, 1)
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss, (
            f"Loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_is_finite_throughout(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        step = _make_step_fn(model, opt)
        for _ in range(10):
            model, opt_state, loss = step(model, opt_state, x, y)
            assert math.isfinite(float(loss)), f"loss became non-finite: {loss}"

    def test_weights_change_after_step(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        w_before = model.blocks[0].module.linear_skip.weight
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        step = _make_step_fn(model, opt)
        model, _, _ = step(model, opt_state, x, y)
        w_after = model.blocks[0].module.linear_skip.weight
        assert not jnp.allclose(w_before, w_after), "weights unchanged after one step"

    def test_named_weights_change_for_all_blocks(self):
        """Both blocks' skip weights should move after training."""
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        w0_before = model.blocks[0].module.linear_skip.weight
        w1_before = model.blocks[1].module.linear_skip.weight
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert not jnp.allclose(trained.blocks[0].module.linear_skip.weight, w0_before)
        assert not jnp.allclose(trained.blocks[1].module.linear_skip.weight, w1_before)

    def test_spectral_weights_change(self):
        """Spectral conv weights should also be updated."""
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        s_before = model.blocks[0].module.spectral_conv.weight_real
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert not jnp.allclose(
            trained.blocks[0].module.spectral_conv.weight_real, s_before
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 – 2-D spectral pipe: learn a constant target field
# ═══════════════════════════════════════════════════════════════════════════


class TestTrain2DSpectralPipe:
    """A 3-block 2-D FNO-style pipe should fit a constant target field."""

    def _model(self):
        k = ks(3)
        return (
            fx.block(fl.SpectralBlock2d(1, 16, n_modes=4, key=k[0]))
            | fx.block(fl.SpectralBlock2d(16, 16, n_modes=4, key=k[1]))
            | fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=k[2]))
        )

    def _data(self):
        x = jnp.ones((8, 8, 1)) * 0.1  # non-zero so all weight gradients are non-zero
        y = jnp.ones((8, 8, 1)) * 0.5
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss, (
            f"Loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )

    def test_output_moves_toward_target(self):
        """After training, output should be closer to the target than initial output."""
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_err = float(jnp.mean(jnp.abs(model(x) - y)))
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y, n_steps=30)
        final_err = float(jnp.mean(jnp.abs(trained(x) - y)))
        assert final_err < init_err, (
            f"Output did not approach target: MAE {init_err:.4f} → {final_err:.4f}"
        )

    def test_all_blocks_receive_gradient_updates(self):
        """Every block's skip weight must change after training."""
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        w_before = [model.blocks[i].module.linear_skip.weight for i in range(3)]
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        for i in range(3):
            assert not jnp.allclose(
                trained.blocks[i].module.linear_skip.weight, w_before[i]
            ), f"block {i} skip weight unchanged after training"


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 – MLP pipe: learn a quadratic
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainMLPPipe:
    """A two-MLP pipe should fit f(x) = x^2 in a few Adam steps."""

    def _model(self):
        k1, k2 = ks(2)
        return fx.block(
            fx.mlp(in_features=1, output_dim=16, hidden_dims=16, key=k1)
        ) | fx.block(fx.mlp(in_features=16, output_dim=1, hidden_dims=8, key=k2))

    def _data(self):
        x = jnp.linspace(-1, 1, 32)[:, None]  # (32, 1)
        y = x**2  # (32, 1)
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_weights_update_in_both_blocks(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        # foundax MLP stores hidden layers in `hidden_layers`
        w0_before = model.blocks[0].module.hidden_layers[0].weight
        w1_before = model.blocks[1].module.hidden_layers[0].weight
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert not jnp.allclose(
            trained.blocks[0].module.hidden_layers[0].weight, w0_before
        )
        assert not jnp.allclose(
            trained.blocks[1].module.hidden_layers[0].weight, w1_before
        )

    def test_sgd_also_decreases_loss(self):
        """SGD with momentum should also work as an optimiser."""
        model = self._model()
        opt = optax.sgd(1e-2, momentum=0.9)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 – AddCombinator training
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainAddCombinator:
    """An AddCombinator (residual connection) should be trainable."""

    def _model(self):
        k1, k2, k3 = ks(3)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        adder = fx.block(fx.add(a, b))
        proj = fx.block(fl.SpectralBlock2d(4, 1, n_modes=4, key=k3))
        return adder | proj

    def _data(self):
        x = jnp.ones((8, 8, 4)) * 0.1
        y = jnp.zeros((8, 8, 1))
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_both_branches_update(self):
        """Both branches of the AddCombinator must receive gradient updates."""
        model = self._model()
        add_block = model.blocks[0].module
        wa_before = add_block.a.module.linear_skip.weight
        wb_before = add_block.b.module.linear_skip.weight
        opt = optax.adam(1e-3)
        x, y = self._data()
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        add_trained = trained.blocks[0].module
        assert not jnp.allclose(add_trained.a.module.linear_skip.weight, wa_before)
        assert not jnp.allclose(add_trained.b.module.linear_skip.weight, wb_before)


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 – CatCombinator training
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainCatCombinator:
    """A CatCombinator followed by a projection block should be trainable."""

    def _model(self):
        k1, k2, k3 = ks(3)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        catter = fx.block(fx.cat(a, b))  # outputs 16 channels
        proj = fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=k3))
        return catter | proj

    def _data(self):
        x = jnp.ones((8, 8, 4)) * 0.2
        y = jnp.zeros((8, 8, 1))
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_both_cat_branches_update(self):
        cat_block = self._model().blocks[0].module
        wa_before = cat_block.a.module.linear_skip.weight
        wb_before = cat_block.b.module.linear_skip.weight
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        cat_trained = trained.blocks[0].module
        assert not jnp.allclose(cat_trained.a.module.linear_skip.weight, wa_before)
        assert not jnp.allclose(cat_trained.b.module.linear_skip.weight, wb_before)


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 – DotCombinator (DeepONet-style) training
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainDotCombinator:
    """Branch · trunk DeepONet model should be trainable on a trivial dataset."""

    def _model(self):
        k1, k2 = ks(2)
        branch = fx.block(fx.mlp(in_features=8, output_dim=16, hidden_dims=16, key=k1))
        trunk = fx.block(fx.mlp(in_features=1, output_dim=16, hidden_dims=16, key=k2))
        return fx.dot(branch, trunk)

    def _data(self):
        # u: sensor values (8,), y: query points (10, 1), target: (10,)
        u = jnp.ones((8,))
        y_pts = jnp.linspace(0, 1, 10)[:, None]
        target = jnp.zeros((10,))
        return u, y_pts, target

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        u, y_pts, target = self._data()
        init_loss, final_loss, _ = _train(
            model, opt, _make_step_fn_2in(model, opt), u, y_pts, target
        )
        assert final_loss < init_loss, (
            f"Loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )

    def test_branch_weights_update(self):
        model = self._model()
        w_before = model.branch.module.hidden_layers[0].weight
        opt = optax.adam(1e-3)
        u, y_pts, target = self._data()
        _, _, trained = _train(
            model, opt, _make_step_fn_2in(model, opt), u, y_pts, target
        )
        assert not jnp.allclose(trained.branch.module.hidden_layers[0].weight, w_before)

    def test_trunk_weights_update(self):
        model = self._model()
        w_before = model.trunk.module.hidden_layers[0].weight
        opt = optax.adam(1e-3)
        u, y_pts, target = self._data()
        _, _, trained = _train(
            model, opt, _make_step_fn_2in(model, opt), u, y_pts, target
        )
        assert not jnp.allclose(trained.trunk.module.hidden_layers[0].weight, w_before)


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 – Long pipe training
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainLongPipe:
    """A 6-block FNO-style 2-D pipeline should train without numerical issues."""

    def _model(self):
        k = ks(6)
        blocks = [
            fx.block(fl.SpectralBlock2d(1, 16, n_modes=4, key=k[0]), name="lift"),
            fx.block(fl.SpectralBlock2d(16, 16, n_modes=4, key=k[1])),
            fx.block(fl.SpectralBlock2d(16, 16, n_modes=4, key=k[2])),
            fx.block(fl.SpectralBlock2d(16, 16, n_modes=4, key=k[3])),
            fx.block(fl.SpectralBlock2d(16, 16, n_modes=4, key=k[4])),
            fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=k[5]), name="project"),
        ]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        return pipe

    def _data(self):
        x = jnp.ones((8, 8, 1)) * 0.1  # non-zero so skip-weight gradients are non-zero
        y = jnp.full((8, 8, 1), 0.3)
        return x, y

    def test_loss_decreases(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_all_6_blocks_receive_updates(self):
        model = self._model()
        w_before = [model.blocks[i].module.linear_skip.weight for i in range(6)]
        opt = optax.adam(1e-3)
        x, y = self._data()
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        for i in range(6):
            assert not jnp.allclose(
                trained.blocks[i].module.linear_skip.weight, w_before[i]
            ), f"block '{trained.blocks[i].name}' (index {i}) weight unchanged"

    def test_no_nan_or_inf_during_training(self):
        model = self._model()
        opt = optax.adam(1e-3)
        x, y = self._data()
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        step = _make_step_fn(model, opt)
        for i in range(20):
            model, opt_state, loss = step(model, opt_state, x, y)
            assert math.isfinite(float(loss)), f"NaN/Inf loss at step {i}"

    def test_param_count_unchanged_after_training(self):
        """Training must not add or remove parameters."""

        def param_count(m):
            return sum(
                math.prod(leaf.shape)
                for leaf in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
            )

        model = self._model()
        before = param_count(model)
        opt = optax.adam(1e-3)
        x, y = self._data()
        _, _, trained = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert param_count(trained) == before

    def test_different_lr_schedules(self):
        """Cosine decay and constant LR should both decrease loss."""
        for schedule in [
            optax.adam(1e-3),
            optax.adam(optax.cosine_decay_schedule(1e-3, decay_steps=20)),
        ]:
            model = self._model()
            x, y = self._data()
            init_loss, final_loss, _ = _train(
                model, schedule, _make_step_fn(model, schedule), x, y
            )
            assert final_loss < init_loss, f"Loss did not decrease with {schedule}"


# ═══════════════════════════════════════════════════════════════════════════
# Section 8 – Gradient clipping and weight decay
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainWithRegularisation:
    """optax chained transforms (clip + adam, adamw) should work with the pipe API."""

    def _model_and_data(self):
        k1, k2 = ks(2)
        model = fx.block(fl.SpectralBlock2d(1, 8, n_modes=4, key=k1)) | fx.block(
            fl.SpectralBlock2d(8, 1, n_modes=4, key=k2)
        )
        x = jnp.zeros((8, 8, 1))
        y = jnp.ones((8, 8, 1)) * 0.5
        return model, x, y

    def test_clip_then_adam_decreases_loss(self):
        model, x, y = self._model_and_data()
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_adamw_weight_decay_decreases_loss(self):
        model, x, y = self._model_and_data()
        opt = optax.adamw(1e-3, weight_decay=1e-4)
        init_loss, final_loss, _ = _train(model, opt, _make_step_fn(model, opt), x, y)
        assert final_loss < init_loss

    def test_gradient_clipping_bounds_update_norm(self):
        """With global-norm clipping, the update norm should not exceed the clip value."""
        model, x, y = self._model_and_data()
        clip_val = 0.5
        opt = optax.chain(optax.clip_by_global_norm(clip_val), optax.sgd(1.0))
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step_with_grads(model, opt_state, x, y):
            loss, grads = eqx.filter_value_and_grad(
                lambda m: jnp.mean((m(x) - y) ** 2)
            )(model)
            updates, new_state = opt.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            # Compute update norm before applying
            update_leaves = jax.tree_util.tree_leaves(eqx.filter(updates, eqx.is_array))
            norm = jnp.sqrt(sum(jnp.sum(u**2) for u in update_leaves))
            return norm, new_state

        update_norm, _ = step_with_grads(model, opt_state, x, y)
        # SGD with lr=1 means update = -clipped_grad; norm ≤ clip_val
        assert float(update_norm) <= clip_val + 1e-5, (
            f"Update norm {float(update_norm):.4f} exceeded clip {clip_val}"
        )
