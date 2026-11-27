"""Every public factory builds with a small configuration and runs one forward pass on the CPU.

This catches drift between a factory and the class it builds: a shortcut that passes an argument the class no
longer takes, or a renamed option. The core architectures run a real forward pass on unbatched, channel-last
input where the model takes it. The foundation-model shortcuts are only built, abstractly with
``eqx.filter_eval_shape`` so no weights are allocated; their forward passes are tested in their own files.

``test_every_factory_is_covered`` fails when a factory is added to ``fx.__all__`` or ``fx.kan.__all__`` without a
case here.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402
from foundax.gaot import compute_neighbors_csr  # noqa: E402

KEY = jax.random.PRNGKey(0)


def _x(*shape, seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def _gaot_inputs(h=4, n=12, m=6, radius=0.5):
    xs = np.linspace(0, 1, h, dtype=np.float32)
    latent = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    rng = np.random.default_rng(0)
    x = rng.random((n, 2)).astype(np.float32)
    q = rng.random((m, 2)).astype(np.float32)
    enc = [compute_neighbors_csr(x, latent, radius)]
    dec = [compute_neighbors_csr(latent, q, radius)]
    return (jnp.asarray(latent), jnp.asarray(x), _x(n, 2), jnp.asarray(q), enc, dec)


def _pcno_inputs(b=1, n=10, e=16, ndim=2, in_dim=3):
    edges = jax.random.randint(jax.random.PRNGKey(2), (b, e, 2), 0, n)
    return (
        _x(b, n, in_dim),
        jnp.ones((b, n, 1)),
        jax.random.uniform(jax.random.PRNGKey(3), (b, n, ndim)),
        jnp.full((b, n, 1), 1.0 / n),
        edges,
        _x(b, e, ndim, seed=4),
    )


def _nodal_inputs(n=10, in_dim=3):
    return (
        _x(n, in_dim),
        jnp.ones((n, 1)),
        jax.random.uniform(KEY, (n, 2)),
        jnp.full((n, 1), 1.0 / n),
    )


# name -> (build, args of the forward pass, expected output shape)
# Inputs are unbatched where the model takes that; the models documented as batched get a batch of one.
CORE = {
    "linear": (lambda k: fx.linear(3, 2, key=k), lambda: (_x(5, 3),), (5, 2)),
    "mlp": (lambda k: fx.mlp(3, 2, hidden_dims=8, key=k), lambda: (_x(3),), (2,)),
    # FNO1D returns (n_steps, w, d_vars): it is the auto-regressive variant
    "fno1d": (
        lambda k: fx.fno1d(2, 8, 4, n_layers=1, key=k),
        lambda: (_x(16, 2),),
        (1, 16, 1),
    ),
    "fno2d": (
        lambda k: fx.fno2d(2, 8, 4, n_layers=1, d_model=(16, 16), key=k),
        lambda: (_x(16, 16, 2),),
        (16, 16, 1),
    ),
    "fno3d": (
        lambda k: fx.fno3d(2, 8, 4, n_layers=1, d_model=(8, 8, 8), key=k),
        lambda: (_x(8, 8, 8, 2),),
        (8, 8, 8, 1),
    ),
    "unet1d": (
        lambda k: fx.unet1d(2, 1, depth=2, wf=2, key=k),
        lambda: (_x(16, 2),),
        (16, 1),
    ),
    "unet2d": (
        lambda k: fx.unet2d(2, 1, depth=2, wf=2, key=k),
        lambda: (_x(16, 16, 2),),
        (16, 16, 1),
    ),
    "unet3d": (
        lambda k: fx.unet3d(2, 1, depth=2, wf=2, key=k),
        lambda: (_x(8, 8, 8, 2),),
        (8, 8, 8, 1),
    ),
    "transformer": (
        lambda k: fx.transformer(1, 16, 2, 32, 0.0, 50, 16, key=k),
        lambda: (jnp.arange(8) % 50, jnp.arange(8) % 50),
        None,
    ),
    "deeponet": (
        lambda k: fx.deeponet(
            n_sensors=10, basis_functions=8, hidden_dim=16, n_layers=2, key=k
        ),
        lambda: (_x(10), _x(5, 1)),
        None,
    ),
    "cno2d": (
        lambda k: fx.cno2d(
            2,
            1,
            size=16,
            N_layers=1,
            N_res=1,
            N_res_neck=1,
            channel_multiplier=4,
            key=k,
        ),
        lambda: (_x(16, 16, 2),),
        None,
    ),
    "mgno1d": (
        lambda k: fx.mgno1d(16, num_layer=1, num_channel_u=4, num_channel_f=2, key=k),
        lambda: (_x(16, 2),),
        (16, 1),
    ),
    "mgno2d": (
        lambda k: fx.mgno2d(
            (16, 16), num_layer=1, num_channel_u=4, num_channel_f=2, key=k
        ),
        lambda: (_x(16, 16, 2),),
        (16, 16, 1),
    ),
    "geofno": (
        lambda k: fx.geofno(4, 4, 8, s1=8, s2=8, n_layers=1, fc_dim=8, key=k),
        lambda: (_x(20, 2),),
        (20, 1),
    ),
    "geofno_nodal": (
        lambda k: fx.geofno_nodal(
            2, [4, 4], [1.0, 1.0], layers=(8, 8), fc_dim=8, key=k
        ),
        _nodal_inputs,
        None,
    ),
    "pcno": (
        lambda k: fx.pcno(2, [4, 4], [1.0, 1.0], layers=(8, 8), fc_dim=8, key=k),
        _pcno_inputs,
        (1, 10, 1),
    ),
    "cgptno": (
        lambda k: fx.cgptno(2, [3], n_layers=1, n_hidden=8, key=k),
        lambda: (_x(1, 5, 2), [_x(1, 7, 3)]),
        None,
    ),
    "gnot": (
        lambda k: fx.gnot(2, [3], n_layers=1, n_hidden=8, key=k),
        lambda: (_x(1, 5, 2), [_x(1, 7, 3)]),
        None,
    ),
    "moegptno": (
        lambda k: fx.moegptno(2, 3, n_layers=1, n_hidden=8, key=k),
        lambda: (_x(1, 5, 2), _x(1, 7, 3)),
        None,
    ),
    "pit": (
        lambda k: fx.pit(
            2,
            1,
            hid_channels=8,
            n_head=2,
            input_res=(8, 8),
            latent_res=(4, 4),
            output_res=(8, 8),
            key=k,
        ),
        lambda: (_x(1, 64, 2),),
        None,
    ),
    "pointnet": (
        lambda k: fx.pointnet(3, 1, hidden_dims=[8, 8], key=k),
        lambda: (_x(1, 12, 3),),
        None,
    ),
    "gaot_S": (
        lambda k: fx.gaot_S(latent_tokens_size=(4, 4), key=k),
        _gaot_inputs,
        (6, 1),
    ),
    "gaot_M": (
        lambda k: fx.gaot_M(latent_tokens_size=(4, 4), key=k),
        _gaot_inputs,
        (6, 1),
    ),
    "gaot_L": (
        lambda k: fx.gaot_L(latent_tokens_size=(4, 4), key=k),
        _gaot_inputs,
        (6, 1),
    ),
    "dit2d": (
        lambda k: fx.dit2d(
            2, patch_size=4, hidden_size=16, depth=1, num_heads=2, key=k
        ),
        lambda: (_x(16, 16, 2), jnp.array(0.5)),
        (16, 16, 2),
    ),
    "dit3d": (
        lambda k: fx.dit3d(
            2, patch_size=4, hidden_size=12, depth=1, num_heads=2, key=k
        ),
        lambda: (_x(8, 8, 8, 2), jnp.array(0.5)),
        (8, 8, 8, 2),
    ),
    "ffno2d": (
        lambda k: fx.ffno2d(2, 8, n_modes=4, n_layers=1, key=k),
        lambda: (_x(16, 16, 2),),
        (16, 16, 2),
    ),
    "ffno3d": (
        lambda k: fx.ffno3d(2, 8, n_modes=4, n_layers=1, key=k),
        lambda: (_x(8, 8, 8, 2),),
        (8, 8, 8, 2),
    ),
    "wno1d": (lambda k: fx.wno1d(2, 8, depth=1, key=k), lambda: (_x(16, 2),), (16, 2)),
    "wno2d": (
        lambda k: fx.wno2d(2, 8, depth=1, key=k),
        lambda: (_x(16, 16, 2),),
        (16, 16, 2),
    ),
    "wno3d": (
        lambda k: fx.wno3d(2, 8, depth=1, key=k),
        lambda: (_x(8, 8, 8, 2),),
        (8, 8, 8, 2),
    ),
    "transolver": (
        lambda k: fx.transolver(
            2, 1, hidden_dim=16, n_layers=1, n_heads=2, n_slices=4, key=k
        ),
        lambda: (_x(20, 2), _x(20, 1)),
        (20, 1),
    ),
    "transolver2d": (
        lambda k: fx.transolver2d(
            hidden_dim=16, n_layers=1, n_heads=2, n_slices=4, key=k
        ),
        lambda: (_x(8, 8, 2), _x(8, 8, 1)),
        (8, 8, 1),
    ),
    "transolver3d": (
        lambda k: fx.transolver3d(
            hidden_dim=16, n_layers=1, n_heads=2, n_slices=4, key=k
        ),
        lambda: (_x(4, 4, 4, 3), _x(4, 4, 4, 1)),
        (4, 4, 4, 1),
    ),
    "sfno2d": (
        lambda k: fx.sfno2d(
            2, 1, hidden_channels=8, L=8, nlat=8, nlon=16, n_layers=1, key=k
        ),
        lambda: (_x(8, 16, 2),),
        (8, 16, 1),
    ),
}

_KAN_NETS = [
    "kan", "default", "efficient", "fast", "fourier", "chebyshev", "jacobi", "legendre", "wavelet", "taylor",
    "hermite", "laguerre", "bernstein", "relu", "rational", "sinc", "gram", "bsrbf",
]  # fmt: skip
KAN = {
    name: (
        lambda k, name=name: getattr(fx.kan, name)(
            3, 2, hidden_dims=4, num_layers=1, key=k
        ),
        lambda: (_x(3),),
        (2,),
    )
    for name in _KAN_NETS
}
KAN |= {
    "conv1d": (lambda k: fx.kan.conv1d(2, 3, key=k), lambda: (_x(16, 2),), (16, 3)),
    "conv2d": (lambda k: fx.kan.conv2d(2, 3, key=k), lambda: (_x(8, 8, 2),), (8, 8, 3)),
    "conv3d": (
        lambda k: fx.kan.conv3d(2, 3, key=k),
        lambda: (_x(4, 4, 4, 2),),
        (4, 4, 4, 3),
    ),
    "res_block": (lambda k: fx.kan.res_block(4, key=k), lambda: (_x(4),), (4,)),
    "spectral_block1d": (
        lambda k: fx.kan.spectral_block1d(2, 3, 4, key=k),
        lambda: (_x(16, 2),),
        (16, 3),
    ),
    "spectral_block2d": (
        lambda k: fx.kan.spectral_block2d(2, 3, 4, key=k),
        lambda: (_x(8, 8, 2),),
        (8, 8, 3),
    ),
    "spectral_block3d": (
        lambda k: fx.kan.spectral_block3d(2, 3, 2, key=k),
        lambda: (_x(4, 4, 4, 2),),
        (4, 4, 4, 3),
    ),
    "attention_block": (
        lambda k: fx.kan.attention_block(8, num_heads=2, key=k),
        lambda: (_x(5, 8),),
        (5, 8),
    ),
}

# Foundation-model shortcuts in ``fx``: built with their defaults, without allocating weights.
FOUNDATION = [
    "poseidonT", "poseidonB", "poseidonL", "morph_Ti", "morph_S", "morph_M", "morph_L", "mpp_Ti", "mpp_S",
    "mpp_B", "mpp_L", "pdeformer2_small", "pdeformer2_base", "pdeformer2_fast", "dpot_Ti", "dpot_S", "dpot_M",
    "dpot_L", "dpot_H",
]  # fmt: skip
FOUNDATION_ARGS = {
    "prose_fd_1to1": {},
    "prose_fd_2to1": {"n_words": 64},
    "prose_ode_2to1": {"n_words": 64, "pad_index": 0},
    "prose_pde_2to1": {"n_words": 64, "pad_index": 0},
}

# Names in ``fx.__all__`` that are not model factories, or are covered elsewhere.
NOT_FACTORIES = {
    "__version__", "block", "Block", "Pipe", "ShapeMismatchError", "describe", "dot", "add", "cat", "layers",
    "SinusoidalTimeEmbedding", "FiLMLayer", "AdaLayerNorm", "AdaLayerNormZero",
    # namespaces: kan is covered below; the foundation namespaces in test_heavy_models.py and their own files
    "kan", "gaot", "poseidon", "walrus", "morph", "mpp", "bcat", "pdeformer2", "dpot", "prose",
    # downloads its weights from Hugging Face on construction
    "timesfm",
}  # fmt: skip

# Known failures, each removed by the commit that fixes it (strict: an unexpected pass fails too).
XFAIL = {
    name: f"B1: fx.{name} is a copy of the namespace factory and passes arguments it does not take"
    for name in FOUNDATION
}
XFAIL["unet3d"] = (
    "B5: fx.unet3d passes its default activation as the string 'celu', which UNet3D calls"
)


def _params(table, prefix=""):
    return [
        pytest.param(name, *case, id=prefix + name, marks=_marks(prefix + name))
        for name, case in table.items()
    ]


def _marks(name):
    return [pytest.mark.xfail(reason=XFAIL[name], strict=True)] if name in XFAIL else []


def _check_output(out, shape):
    leaves = jax.tree.leaves(out)
    assert leaves, "the forward pass returned nothing"
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "non-finite output"
    if shape is not None:
        assert out.shape == shape


@pytest.mark.parametrize(
    "name, build, inputs, shape", _params(CORE) + _params(KAN, "kan.")
)
def test_factory_builds_and_runs(name, build, inputs, shape):
    model = build(KEY)
    # jit: one compile is faster than running the many small ops of these models one by one
    _check_output(eqx.filter_jit(lambda m, args: m(*args))(model, inputs()), shape)


@pytest.mark.parametrize(
    "name, kwargs",
    [pytest.param(n, {}, id=n, marks=_marks(n)) for n in FOUNDATION]
    + [pytest.param(n, kw, id=n, marks=_marks(n)) for n, kw in FOUNDATION_ARGS.items()],
)
def test_foundation_shortcut_builds(name, kwargs):
    model = eqx.filter_eval_shape(lambda: getattr(fx, name)(**kwargs))
    assert jax.tree.leaves(model), "the model has no arrays"


def test_every_factory_is_covered():
    covered = set(CORE) | set(FOUNDATION) | set(FOUNDATION_ARGS) | NOT_FACTORIES
    assert set(fx.__all__) - covered == set(), "factories without a smoke case"
    assert set(fx.kan.__all__) - set(KAN) == set(), "KAN factories without a smoke case"
