"""Every core model follows the foundax convention: one unbatched, channel-last sample in, one sample out.

The caller batches with ``jax.vmap``. For each model the test checks that it runs on one sample, that the output
has the shape of one sample (no batch axis added), and that ``vmap(model)`` over two samples equals the model run
on each sample alone. A model that treats the leading axis as a batch, or mixes samples (a norm over the batch), fails.

Every JAX array input is batched (the second sample perturbs floats and rolls integers, so tokens and edge
indices stay valid); NumPy arrays, the neighbour lists of GAOT, are shared.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

from .test_factory_smoke import CORE, KAN, KEY, _gaot_inputs, _nodal_inputs, _x  # noqa: E402


def _pcno_sample(n=10, e=16, ndim=2, in_dim=3):
    edges = jax.random.randint(jax.random.PRNGKey(2), (e, 2), 0, n)
    nodes = jax.random.uniform(jax.random.PRNGKey(3), (n, ndim))
    return (
        _x(n, in_dim),
        jnp.ones((n, 1)),
        nodes,
        jnp.full((n, 1), 1.0 / n),
        edges,
        _x(e, ndim, seed=4),
    )


# name -> (one sample of inputs, the output shape for one sample)
SAMPLES = {
    "linear": (lambda: (_x(3),), (2,)),
    "mlp": (lambda: (_x(3),), (2,)),
    "fno1d": (lambda: (_x(16, 2),), (1, 16, 1)),
    "fno2d": (lambda: (_x(16, 16, 2),), (16, 16, 1)),
    "fno3d": (lambda: (_x(8, 8, 8, 2),), (8, 8, 8, 1)),
    "unet1d": (lambda: (_x(16, 2),), (16, 1)),
    "unet2d": (lambda: (_x(16, 16, 2),), (16, 16, 1)),
    "unet3d": (lambda: (_x(8, 8, 8, 2),), (8, 8, 8, 1)),
    "transformer": (lambda: (jnp.arange(8) % 50, jnp.arange(8) % 50), (8, 50)),
    "deeponet": (lambda: (_x(10), _x(5, 1)), (5,)),
    "cno2d": (lambda: (_x(16, 16, 2),), (16, 16, 1)),
    "mgno1d": (lambda: (_x(16, 2),), (16, 1)),
    "mgno2d": (lambda: (_x(16, 16, 2),), (16, 16, 1)),
    "geofno": (lambda: (_x(20, 2),), (20, 1)),
    "geofno_nodal": (_nodal_inputs, (10, 1)),
    "pcno": (_pcno_sample, (10, 1)),
    "cgptno": (lambda: (_x(5, 2), [_x(7, 3)]), (5, 1)),
    "gnot": (lambda: (_x(5, 2), [_x(7, 3)]), (5, 1)),
    "moegptno": (lambda: (_x(5, 2), _x(7, 3)), (5, 1)),
    "pit": (lambda: (_x(64, 2),), (64, 1)),
    "pointnet": (lambda: (_x(12, 3),), (12, 1)),
    "gaot_S": (_gaot_inputs, (6, 1)),
    "gaot_M": (_gaot_inputs, (6, 1)),
    "gaot_L": (_gaot_inputs, (6, 1)),
    "dit2d": (lambda: (_x(16, 16, 2), jnp.array(0.5)), (16, 16, 2)),
    "dit3d": (lambda: (_x(8, 8, 8, 2), jnp.array(0.5)), (8, 8, 8, 2)),
    "ffno2d": (lambda: (_x(16, 16, 2),), (16, 16, 2)),
    "ffno3d": (lambda: (_x(8, 8, 8, 2),), (8, 8, 8, 2)),
    "wno1d": (lambda: (_x(16, 2),), (16, 2)),
    "wno2d": (lambda: (_x(16, 16, 2),), (16, 16, 2)),
    "wno3d": (lambda: (_x(8, 8, 8, 2),), (8, 8, 8, 2)),
    "transolver": (lambda: (_x(20, 2), _x(20, 1)), (20, 1)),
    "transolver2d": (lambda: (_x(8, 8, 2), _x(8, 8, 1)), (8, 8, 1)),
    "transolver3d": (lambda: (_x(4, 4, 4, 3), _x(4, 4, 4, 1)), (4, 4, 4, 1)),
    "sfno2d": (lambda: (_x(8, 16, 2),), (8, 16, 1)),
}
SAMPLES |= {"kan." + name: (inputs, shape) for name, (_, inputs, shape) in KAN.items()}
BUILDS = {name: case[0] for name, case in CORE.items()} | {
    "kan." + n: case[0] for n, case in KAN.items()
}

# Known failures, each removed by the commit that fixes it (strict: an unexpected pass fails too).
XFAIL = {
    "unet3d": "B5: the default activation is the string 'celu'",
    "cno2d": "B5: CNO adds a batch axis to one sample",
    "geofno_nodal": "B5: out_dim=1 is squeezed away",
    "pcno": "P2.6: PCNO takes batched, channel-first internals only",
    "cgptno": "P2.6: the GNOT family takes batched input only",
    "gnot": "P2.6: the GNOT family takes batched input only",
    "moegptno": "P2.6: the GNOT family takes batched input only",
}


def _second(x):
    if not isinstance(x, jax.Array):
        return x
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x + 0.5 * jnp.cos(3.0 * x)
    return jnp.roll(x, 1)


def _marks(name):
    return [pytest.mark.xfail(reason=XFAIL[name], strict=True)] if name in XFAIL else []


@pytest.mark.parametrize(
    "name", [pytest.param(n, id=n, marks=_marks(n)) for n in SAMPLES]
)
def test_one_sample_in_one_sample_out_and_vmap(name):
    model = BUILDS[name](KEY)
    inputs, shape = SAMPLES[name]
    a = inputs()
    b = jax.tree.map(_second, a)
    in_axes = jax.tree.map(lambda x: 0 if isinstance(x, jax.Array) else None, a)
    both = jax.tree.map(
        lambda x, y: jnp.stack([x, y]) if isinstance(x, jax.Array) else x, a, b
    )

    run = eqx.filter_jit(lambda m, args: m(*args))
    out_a, out_b = run(model, a), run(model, b)
    assert out_a.shape == shape, "the output of one sample is not one sample"
    batched = eqx.filter_jit(
        lambda m, args: jax.vmap(lambda *xs: m(*xs), in_axes=tuple(in_axes))(*args)
    )(model, both)
    np.testing.assert_allclose(batched, jnp.stack([out_a, out_b]), rtol=1e-4, atol=1e-5)
