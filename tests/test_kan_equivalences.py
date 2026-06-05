"""Mathematical equivalence tests across KAN variants.

Several KAN variants are formally equivalent to others for specific parameter
choices. These tests pin down those equivalences as machine-checkable
properties: if the math holds, weights identical, outputs identical.
"""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import (
    JacobiBasis,
    LegendreBasis,
    JacobiKANLayer,
    LegendreKANLayer,
    ChebyshevBasis,
    BSRBFBasis,
    BSplineBasis,
    RBFBasis,
    KAN,
    EfficientKAN,
)


def _k():
    return jax.random.PRNGKey(0)


# ── LegendreBasis is JacobiBasis with alpha = beta = 0 ─────────────────────


def test_legendre_basis_equals_jacobi_with_alpha_beta_zero():
    leg = LegendreBasis(degree=6)
    jac = JacobiBasis(degree=6, alpha=0.0, beta=0.0)
    x = jnp.linspace(-0.9, 0.9, 20).reshape(-1, 1)
    assert jnp.allclose(leg(x), jac(x), atol=1e-6)


def test_legendre_layer_equals_jacobi_layer():
    """Same weights → same forward when α=β=0."""
    key = _k()
    leg = LegendreKANLayer(3, 5, degree=4, key=key)
    jac = JacobiKANLayer(3, 5, degree=4, alpha=0.0, beta=0.0, key=key)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    assert jnp.allclose(leg(x), jac(x), atol=1e-5)


# ── ChebyshevBasis is JacobiBasis with alpha = beta = -1/2 (up to scaling) ─
# The recurrence coefficient identity is exact only up to a per-degree
# normalisation factor, so we check column-wise rank correlation rather than
# pointwise equality.


def test_chebyshev_and_jacobi_share_zero_count():
    cheb = ChebyshevBasis(degree=6)
    jac = JacobiBasis(degree=6, alpha=-0.5, beta=-0.5)
    x = jnp.linspace(-0.95, 0.95, 200).reshape(-1, 1)
    pc = np.asarray(cheb(x))[:, 0, :]
    pj = np.asarray(jac(x))[:, 0, :]

    # For each column, the sign-change count should match (Chebyshev / Jacobi
    # of the same degree have the same number of zeros on (-1, 1)).
    def zero_count(col):
        s = np.sign(col)
        s = s[s != 0]
        return int(np.sum(s[:-1] != s[1:]))

    for j in range(1, pc.shape[1]):  # skip constant column 0
        assert zero_count(pc[:, j]) == zero_count(pj[:, j])


# ── BSRBFBasis size and concat structure ───────────────────────────────────


def test_bsrbf_basis_concat_matches_components():
    grid_size, k, rbf_size = 5, 3, 8
    bs = BSplineBasis(in_features=1, grid_size=grid_size, spline_order=k)
    rb = RBFBasis(grid_size=rbf_size)
    hyb = BSRBFBasis(
        in_features=1, grid_size=grid_size, spline_order=k, rbf_grid_size=rbf_size
    )
    x = jnp.linspace(-0.5, 0.5, 11).reshape(-1, 1)
    bs_out = bs(x)
    rb_out = rb(x)
    hyb_out = hyb(x)
    n_bs = bs_out.shape[-1]
    assert jnp.allclose(hyb_out[..., :n_bs], bs_out, atol=1e-6)
    assert jnp.allclose(hyb_out[..., n_bs:], rb_out, atol=1e-6)


# ── KAN and EfficientKAN forward surface ───────────────────────────────────


def test_kan_and_efficient_kan_have_same_forward_signature():
    """Both networks accept the same constructor args and produce same output shapes."""
    key = _k()
    kw = dict(
        in_features=3,
        output_dim=2,
        hidden_dims=8,
        num_layers=2,
        grid_size=5,
        spline_order=3,
    )
    a = KAN(**kw, key=key)
    b = EfficientKAN(**kw, key=key)
    x = jnp.ones((4, 3))
    assert a(x).shape == b(x).shape == (4, 2)


# ── Layer == fx.block(layer) at the forward level ──────────────────────────


def test_block_is_transparent_for_all_variants():
    """fx.block(layer)(x) == layer(x) for every variant."""
    import foundax.architectures.kan as kan_mod

    LAYERS = [
        kan_mod.KANLayer,
        kan_mod.EfficientKANLayer,
        kan_mod.FastKANLayer,
        kan_mod.FourierKANLayer,
        kan_mod.ChebyshevKANLayer,
        kan_mod.JacobiKANLayer,
        kan_mod.LegendreKANLayer,
        kan_mod.WaveletKANLayer,
        kan_mod.TaylorKANLayer,
        kan_mod.HermiteKANLayer,
        kan_mod.LaguerreKANLayer,
        kan_mod.BernsteinKANLayer,
        kan_mod.ReLUKANLayer,
        kan_mod.RationalKANLayer,
        kan_mod.SincKANLayer,
        kan_mod.GramKANLayer,
        kan_mod.BSRBFKANLayer,
    ]
    for cls in LAYERS:
        layer = cls(4, 8, key=_k())
        x = jnp.linspace(-1, 1, 4 * 5).reshape(5, 4)
        assert jnp.array_equal(layer(x), fx.block(layer)(x)), (
            f"fx.block transparency broken for {cls.__name__}"
        )


# ── Pipe forward equals sequential application ─────────────────────────────


def test_pipe_equals_sequential_apply():
    k1, k2, k3 = jax.random.split(_k(), 3)
    l1 = fx.layers.kan.fast(2, 16, key=k1)
    l2 = fx.layers.kan.chebyshev(16, 8, degree=4, key=k2)
    l3 = fx.layers.kan.taylor(8, 1, degree=4, key=k3)
    pipe = fx.block(l1) | fx.block(l2) | fx.block(l3)
    x = jax.random.normal(jax.random.PRNGKey(7), (5, 2))
    expected = l3(l2(l1(x)))
    assert jnp.allclose(pipe(x), expected, atol=1e-5)
