import math

import pytest

import foundax


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _first_leaf(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves, "no leaves found"
    return leaves[0]


def _tree_mse(tree):
    vals = []
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if arr.dtype.kind in ("f", "c"):
            vals.append(jnp.mean(jnp.real(arr) ** 2))
    assert vals, "no floating leaves found for loss computation"
    return sum(vals) / len(vals)


def _check_loss_and_grads(loss, grads):
    assert math.isfinite(float(loss))
    g0 = _first_leaf(grads)
    assert jnp.all(jnp.isfinite(g0))


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder",
    [
        lambda: foundax.poseidon.T(),
        lambda: foundax.poseidon.B(),
        lambda: foundax.poseidon.L(),
        lambda: foundax.walrus.base(),
        lambda: foundax.morph.Ti(),
        lambda: foundax.morph.S(),
        lambda: foundax.morph.M(),
        lambda: foundax.morph.L(),
        lambda: foundax.mpp.Ti(),
        lambda: foundax.mpp.S(),
        lambda: foundax.mpp.B(),
        lambda: foundax.mpp.L(),
        lambda: foundax.bcat.base(),
        lambda: foundax.pdeformer2.small(),
        lambda: foundax.pdeformer2.base(),
        lambda: foundax.pdeformer2.fast(),
        lambda: foundax.dpot.Ti(),
        lambda: foundax.dpot.S(),
        lambda: foundax.dpot.M(),
        lambda: foundax.dpot.L(),
        lambda: foundax.dpot.H(),
        lambda: foundax.prose.fd_1to1(),
        lambda: foundax.prose.fd_2to1(n_words=64),
        lambda: foundax.prose.ode_2to1(n_words=64, pad_index=0),
        lambda: foundax.prose.pde_2to1(n_words=64, pad_index=0),
    ],
)
def test_heavy_model_builders_construct(builder):
    model = builder()
    assert model is not None
