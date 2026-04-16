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


@pytest.mark.heavy
def test_morph_ti_single_forward_backward_pass():
    model = foundax.morph.Ti()

    x = jnp.zeros((1, 1, 1, 1, 8, 8, 8), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    params = variables["params"]

    def loss_fn(p):
        _, _, pred = model.apply({"params": p}, x, deterministic=True)
        return jnp.mean(pred**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_mpp_ti_single_forward_backward_pass():
    model = foundax.mpp.Ti(n_states=12)

    x = jnp.zeros((2, 1, 2, 32, 32), dtype=jnp.float32)
    state_labels = jnp.array([0, 1], dtype=jnp.int32)
    bcs = jnp.zeros((1, 2), dtype=jnp.int32)

    rng = jax.random.PRNGKey(0)
    rng_params, rng_drop = jax.random.split(rng)
    variables = model.init(
        {"params": rng_params, "drop_path": rng_drop},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    params = variables["params"]

    def loss_fn(p):
        y = model.apply(
            {"params": p},
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder", [foundax.poseidon.T, foundax.poseidon.B, foundax.poseidon.L]
)
def test_poseidon_forward_backward_all_variants(builder):
    model = builder()
    x = jnp.zeros((1, 128, 128, 4), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)

    variables = model.init(
        jax.random.PRNGKey(0),
        pixel_values=x,
        time=t,
        deterministic=True,
        return_dict=False,
    )
    params = variables["params"]

    def loss_fn(p):
        out = model.apply(
            {"params": p},
            pixel_values=x,
            time=t,
            deterministic=True,
            return_dict=False,
        )
        return _tree_mse(out)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder", [foundax.morph.Ti, foundax.morph.S, foundax.morph.M, foundax.morph.L]
)
def test_morph_forward_backward_all_variants(builder):
    model = builder()
    x = jnp.zeros((1, 1, 1, 1, 8, 8, 8), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    params = variables["params"]

    def loss_fn(p):
        _, _, pred = model.apply({"params": p}, x, deterministic=True)
        return jnp.mean(pred**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder",
    [
        foundax.mpp.Ti,
        foundax.mpp.S,
        foundax.mpp.B,
        foundax.mpp.L,
    ],
)
def test_mpp_forward_backward_all_variants(builder):
    model = builder(n_states=12)

    x = jnp.zeros((2, 1, 2, 32, 32), dtype=jnp.float32)
    state_labels = jnp.array([0, 1], dtype=jnp.int32)
    bcs = jnp.zeros((1, 2), dtype=jnp.int32)

    rng = jax.random.PRNGKey(0)
    rng_params, rng_drop = jax.random.split(rng)
    variables = model.init(
        {"params": rng_params, "drop_path": rng_drop},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    params = variables["params"]

    def loss_fn(p):
        y = model.apply(
            {"params": p},
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_walrus_forward_backward_base():
    model = foundax.walrus.base()

    x = jnp.zeros((1, 2, 32, 32, 2), dtype=jnp.float32)
    state_labels = jnp.array([0, 1], dtype=jnp.int32)
    bcs = [[2, 2], [2, 2]]

    variables = model.init(
        jax.random.PRNGKey(0),
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    params = variables["params"]

    def loss_fn(p):
        y = model.apply(
            {"params": p},
            x,
            state_labels,
            bcs,
            deterministic=True,
        )
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_bcat_forward_backward_base():
    model = foundax.bcat.base()

    data = jnp.zeros((1, 3, 128, 128, 1), dtype=jnp.float32)
    times = jnp.zeros((1, 3, 1), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), data, times, input_len=2)
    params = variables["params"]

    def loss_fn(p):
        y = model.apply({"params": p}, data, times, input_len=2)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder",
    [foundax.pdeformer2.small, foundax.pdeformer2.base, foundax.pdeformer2.fast],
)
def test_pdeformer2_forward_backward_all_variants(builder):
    foundax.pdeformer2.ensure_repo_on_path("jax_pdeformer2")
    from jax_pdeformer2.utils import create_dummy_inputs

    model = builder()
    inputs = create_dummy_inputs(
        n_graph=1,
        num_scalar=20,
        num_function=2,
        num_branches=4,
        num_points_function=128**2,
        num_points=32,
        key=jax.random.PRNGKey(1),
    )

    variables = model.init(jax.random.PRNGKey(0), **inputs)
    params = variables["params"]

    def loss_fn(p):
        y = model.apply({"params": p}, **inputs)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "builder",
    [foundax.dpot.Ti, foundax.dpot.S, foundax.dpot.M, foundax.dpot.L, foundax.dpot.H],
)
def test_dpot_forward_backward_all_variants(builder):
    model = builder()
    x = jnp.zeros((1, 128, 128, 10, 4), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), x)
    params = variables["params"]

    def loss_fn(p):
        y, cls = model.apply({"params": p}, x)
        return jnp.mean(y**2) + jnp.mean(cls**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_prose_fd_1to1_forward_backward():
    model, variables = foundax.prose.fd_1to1()
    params = variables["params"]

    x = jnp.ones((1, 10, 128, 128, 4), dtype=jnp.float32)
    tin = jnp.zeros((1, 10, 1), dtype=jnp.float32)
    tout = jnp.zeros((1, 10, 1), dtype=jnp.float32)

    def loss_fn(p):
        y = model.apply({"params": p}, x, tin, tout, deterministic=True)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_prose_fd_2to1_forward_backward():
    model, variables = foundax.prose.fd_2to1(n_words=64)
    params = variables["params"]

    x = jnp.ones((1, 10, 128, 128, 4), dtype=jnp.float32)
    tin = jnp.zeros((1, 10, 1), dtype=jnp.float32)
    tout = jnp.zeros((1, 10, 1), dtype=jnp.float32)
    sym = jnp.zeros((1, 48), dtype=jnp.int32)
    sym_mask = jnp.zeros((1, 48), dtype=bool)

    def loss_fn(p):
        y = model.apply({"params": p}, x, tin, tout, sym, sym_mask)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_prose_ode_2to1_forward_backward():
    model, variables = foundax.prose.ode_2to1(n_words=64, pad_index=0)
    params = variables["params"]

    x = jnp.ones((50, 1, 4), dtype=jnp.float32)
    lengths = jnp.asarray([50], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, 50, dtype=jnp.float32)
    text = jnp.zeros((48, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([48], dtype=jnp.int32)

    def loss_fn(p):
        y = model.apply({"params": p}, x, lengths, query_times, text, text_lengths)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)


@pytest.mark.heavy
def test_prose_pde_2to1_forward_backward():
    model, variables = foundax.prose.pde_2to1(n_words=64, pad_index=0)
    params = variables["params"]

    x = jnp.ones((10, 1, 2), dtype=jnp.float32)
    lengths = jnp.asarray([10], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, 10, dtype=jnp.float32)
    text = jnp.zeros((48, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([48], dtype=jnp.int32)

    def loss_fn(p):
        y = model.apply({"params": p}, x, lengths, query_times, text, text_lengths)
        return jnp.mean(y**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    _check_loss_and_grads(loss, grads)
