"""DeepONet combiners: the output depends on both the input function and the query point."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx  # noqa: E402

KEY = jax.random.PRNGKey(0)
COMBINERS = ["dot", "bilinear", "mlp", "attention"]


def _model(combination_type):
    return fx.deeponet(
        combination_type=combination_type,
        n_sensors=10,
        coord_dim=2,
        basis_functions=16,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        key=KEY,
    )


@pytest.mark.parametrize("combination_type", COMBINERS)
def test_output_and_gradient_depend_on_the_query_points(combination_type):
    model = _model(combination_type)
    u = jax.random.normal(jax.random.PRNGKey(1), (10,))
    y = jax.random.uniform(jax.random.PRNGKey(2), (6, 2))
    out = model(u, y)
    assert out.shape[0] == 6
    assert float(jnp.std(out, axis=0).max()) > 1e-4, (
        "the same output at every query point"
    )

    grad_y = jax.grad(lambda y: jnp.sum(model(u, y) ** 2))(y)
    assert float(jnp.abs(grad_y).max()) > 1e-6, (
        "no gradient with respect to the query points"
    )


@pytest.mark.parametrize("combination_type", COMBINERS)
def test_output_depends_on_the_input_function(combination_type):
    model = _model(combination_type)
    y = jax.random.uniform(jax.random.PRNGKey(2), (6, 2))
    a = model(jax.random.normal(jax.random.PRNGKey(1), (10,)), y)
    b = model(jax.random.normal(jax.random.PRNGKey(3), (10,)), y)
    assert float(jnp.abs(a - b).max()) > 1e-4
