"""Tests for the Section 4 structural guarantees (checked post-identification)."""
import numpy as np
import jax.numpy as jnp

from dbwm.dynamics.guarantees import (
    is_shaded,
    observability_matrix,
    check_observability,
    cyclic_index,
    check_controllability,
    open_loop_error_bound,
    empirical_error_terms,
    error_bound_report,
    solve_dare,
    steady_state_error,
)


# --------------------------------------------------------------------------- #
# Definition 4.1 / Proposition 4.1 -- shadedness
# --------------------------------------------------------------------------- #
def test_shadedness_detects_a_dead_basis_column():
    """A basis function no observation activates makes Phi_X un-shaded (A2 fails)."""
    phi = np.random.RandomState(0).randn(20, 5)
    phi[:, 2] = 0.0  # basis 2 is never activated -> w_2 is unobservable

    out = is_shaded(jnp.asarray(phi))
    assert not out["shaded"]
    assert out["n_dead"] == 1


def test_swiglu_style_features_are_shaded():
    """Proposition 4.1: a dense (SwiGLU-like) feature matrix is shaded a.s."""
    phi = np.random.RandomState(0).randn(50, 8)
    out = is_shaded(jnp.asarray(phi))
    assert out["shaded"]
    assert out["n_dead"] == 0


# --------------------------------------------------------------------------- #
# Theorem 4.1 -- observability
# --------------------------------------------------------------------------- #
def test_observability_matrix_shape():
    """O_Upsilon stacks Phi A^tau, giving (N*L, r)."""
    r, n = 4, 3
    a = jnp.asarray(np.random.RandomState(0).randn(r, r))
    phi = jnp.asarray(np.random.RandomState(1).randn(n, r))
    o = observability_matrix(phi, a, [0, 1, 2, 3])
    assert o.shape == (n * 4, r)


def test_visual_encoder_is_observable():
    """
    In the visual setting the encoder supplies the full r-vector (Phi_X = I_r), so
    (Phi_X, A) is observable for any invertible A -- Section 2.3.
    """
    r = 6
    rng = np.random.RandomState(0)
    a = jnp.asarray(0.9 * np.eye(r) + 0.05 * rng.randn(r, r))

    out = check_observability(jnp.eye(r), a)
    assert out["observable"]
    assert out["rank"] == r
    assert out["shaded"]


def test_dead_basis_breaks_observability():
    """A zero column in Phi_X drops the observability rank below r."""
    r = 5
    a = jnp.asarray(np.diag([0.9, 0.8, 0.7, 0.6, 0.5]))
    phi = np.eye(r)
    phi[:, 3] = 0.0  # basis 3 invisible to every sensor

    out = check_observability(jnp.asarray(phi), a)
    assert not out["observable"]
    assert out["rank"] == r - 1


def test_cyclic_index_of_distinct_eigenvalues_is_one():
    """Corollary 4.1: distinct eigenvalues -> cyclic index 1 (one sensor suffices)."""
    a = jnp.asarray(np.diag([0.9, 0.7, 0.5, 0.3]))
    assert cyclic_index(a) == 1


def test_cyclic_index_counts_geometric_multiplicity():
    """A repeated eigenvalue with a 3-dim eigenspace needs 3 sensors."""
    a = jnp.asarray(np.diag([0.5, 0.5, 0.5, 0.9]))  # gm(0.5) = 3
    assert cyclic_index(a) == 3


# --------------------------------------------------------------------------- #
# Proposition 4.3 -- controllability
# --------------------------------------------------------------------------- #
def test_controllability_full_rank_actuator():
    """With B_u = I_r every direction is reachable, so the Gramian is PD."""
    r = 4
    a = jnp.asarray(0.5 * np.eye(r))
    out = check_controllability(a, jnp.eye(r), horizon=r)
    assert out["controllable"]
    assert out["rank"] == r


def test_scalar_irrigation_cannot_control_a_large_state():
    """
    A single irrigation scalar driving one direction reaches at most a Krylov
    subspace -- with a diagonal A that is a 1-dim span, so the system is NOT
    controllable. Reporting this honestly is the point of Proposition 4.3.
    """
    r = 6
    a = jnp.asarray(np.diag(np.full(r, 0.9)))  # A acts as a scalar on span(b_u)
    b_u = jnp.zeros((r, 1)).at[0, 0].set(1.0)

    out = check_controllability(a, b_u, horizon=r)
    assert not out["controllable"]
    assert out["rank"] == 1  # only the actuated direction is reachable


# --------------------------------------------------------------------------- #
# Theorem 4.2 -- open-loop prediction error bound
# --------------------------------------------------------------------------- #
def test_error_bound_is_linear_when_rho_is_one():
    """
    rho = 1 (the near-conservative LST design point) gives eps_enc + T eps_dyn --
    LINEAR growth, not the geometric decay a contractive model would suggest.
    """
    a = jnp.eye(4)  # ||A||_2 = 1 exactly
    bound = float(open_loop_error_bound(a, eps_enc=0.1, eps_dyn=0.05, horizon=10))
    assert np.isclose(bound, 0.1 + 10 * 0.05)
    assert np.isfinite(bound)  # the rho != 1 branch must not leak a 0/0


def test_error_bound_is_geometric_when_contractive():
    """rho < 1 gives the geometric-series bound."""
    rho = 0.5
    a = jnp.asarray(rho * np.eye(3))
    eps_enc, eps_dyn, t = 0.2, 0.1, 5

    bound = float(open_loop_error_bound(a, eps_enc, eps_dyn, t))
    expected = rho**t * eps_enc + (rho**t - 1) / (rho - 1) * eps_dyn
    assert np.isclose(bound, expected, rtol=1e-5)


def test_error_bound_grows_with_horizon():
    """Longer open-loop horizon -> looser bound (no free lunch without observations)."""
    a = jnp.eye(3)
    b3 = float(open_loop_error_bound(a, 0.1, 0.05, 3))
    b10 = float(open_loop_error_bound(a, 0.1, 0.05, 10))
    assert b10 > b3


def test_non_normality_is_reported():
    """
    rho(A) <= ||A||_2, with a gap for non-normal A. Radius-clipping to rho_max = 1
    does NOT bound the norm, so this gap must be visible to the user.
    """
    a = jnp.asarray(np.array([[1.0, 10.0], [0.0, 1.0]]))  # rho = 1, ||A||_2 >> 1
    rep = error_bound_report(a, eps_enc=0.1, eps_dyn=0.05, horizon=5)

    assert np.isclose(rep["spectral_radius"], 1.0, atol=1e-5)
    assert rep["spectral_norm"] > 5.0
    assert rep["non_normality"] > 5.0


def test_empirical_error_terms_use_the_known_forcing():
    """
    eps_dyn must be measured AFTER accounting for B u. On a perfectly forced linear
    system the residual is ~0; ignoring the forcing would report a large eps_dyn.
    """
    rng = np.random.RandomState(0)
    r, t = 4, 50
    a = jnp.asarray(0.9 * np.eye(r))
    b = jnp.asarray(rng.randn(r, 2))
    u = jnp.asarray(rng.rand(t, 2).astype(np.float32))

    w = [jnp.asarray(rng.randn(r).astype(np.float32))]
    for i in range(t - 1):
        w.append(a @ w[-1] + b @ u[i])
    w = jnp.stack(w)

    with_forcing = empirical_error_terms(w, w, a, b, u)
    without = empirical_error_terms(w, w, a, None, None)

    assert with_forcing["eps_dyn"] < 1e-3, "forced residual should vanish"
    assert without["eps_dyn"] > 10 * with_forcing["eps_dyn"], (
        "dropping B u must inflate eps_dyn with forcing signal"
    )


# --------------------------------------------------------------------------- #
# Theorem 4.3 -- steady-state observer error (DARE)
# --------------------------------------------------------------------------- #
def test_dare_solution_satisfies_the_riccati_equation():
    """The returned P_inf must actually be a fixed point of the Riccati recursion."""
    r = 4
    rng = np.random.RandomState(0)
    a = jnp.asarray(0.8 * np.eye(r) + 0.02 * rng.randn(r, r))
    q = jnp.asarray(0.1 * np.eye(r))
    sigma2 = 0.05

    p = solve_dare(a, q, sigma2)
    phi = jnp.eye(r)
    s = phi @ p @ phi.T + sigma2 * jnp.eye(r)
    rhs = a @ p @ a.T + q - a @ p @ phi.T @ jnp.linalg.inv(s) @ phi @ p @ a.T

    np.testing.assert_allclose(np.asarray(p), np.asarray(rhs), atol=1e-5)


def test_steady_state_error_is_horizon_independent():
    """
    P_inf is a fixed point: it does not grow with the horizon. This is exactly what
    the Kalman correction buys over the open-loop bound of Theorem 4.2.
    """
    r = 4
    a = jnp.asarray(0.9 * np.eye(r))
    q = jnp.asarray(0.05 * np.eye(r))

    p_short = solve_dare(a, q, 0.05, max_iter=200)
    p_long = solve_dare(a, q, 0.05, max_iter=1000)
    np.testing.assert_allclose(np.asarray(p_short), np.asarray(p_long), atol=1e-6)


def test_steady_state_error_grows_with_process_noise():
    """More process noise -> larger irreducible estimation error."""
    r = 4
    a = jnp.asarray(0.9 * np.eye(r))
    lo = steady_state_error(a, jnp.asarray(0.01 * np.eye(r)), 0.05)
    hi = steady_state_error(a, jnp.asarray(0.50 * np.eye(r)), 0.05)
    assert hi["trace"] > lo["trace"]
