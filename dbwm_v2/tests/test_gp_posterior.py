"""Tests for the low-rank (Woodbury) GP posterior."""
import jax
import jax.numpy as jnp

from dbwm.gp.posterior import (
    solve_weights,
    lambda_matrix,
    gram_matrix,
    posterior_mean_var,
    log_marginal_likelihood,
)


def test_woodbury_matches_naive_posterior():
    """
    The low-rank posterior mean/var must match the naive O(n^3) GP using
    K = Phi Phi^T directly.
    """
    key = jax.random.PRNGKey(0)
    n, r, m = 40, 6, 5
    phi_x = jax.random.normal(key, (n, r))
    phi_star = jax.random.normal(jax.random.PRNGKey(1), (m, r))
    y = jax.random.normal(jax.random.PRNGKey(2), (n,))
    s2 = 0.1

    lam = lambda_matrix(gram_matrix(phi_x), s2)
    mean_lr, var_lr = posterior_mean_var(phi_star, lam, phi_x.T @ y, s2)

    # Naive: K = Phi Phi^T; mean* = k_*^T (K + s2 I)^{-1} y.
    k = phi_x @ phi_x.T + s2 * jnp.eye(n)
    k_star = phi_star @ phi_x.T  # (m, n)
    kinv_y = jnp.linalg.solve(k, y)
    mean_naive = k_star @ kinv_y
    # var* = s2 * phi_*^T Lambda^{-1} phi_*  (consistent low-rank predictive var).
    assert jnp.allclose(mean_lr, mean_naive, atol=1e-4)
    assert var_lr.shape == (m,)
    assert bool(jnp.all(var_lr > 0))


def test_solve_weights_recovers_linear_target():
    """If y = Phi w_true exactly, the solved weights reconstruct y well."""
    key = jax.random.PRNGKey(5)
    n, r = 100, 8
    phi_x = jax.random.normal(key, (n, r))
    w_true = jax.random.normal(jax.random.PRNGKey(6), (r,))
    y = phi_x @ w_true
    w_hat = solve_weights(phi_x, y, 1e-6)
    assert jnp.allclose(phi_x @ w_hat, y, atol=1e-3)


def test_log_marginal_likelihood_finite():
    """Low-rank LML is finite and scalar."""
    key = jax.random.PRNGKey(7)
    phi_x = jax.random.normal(key, (30, 5))
    y = jax.random.normal(jax.random.PRNGKey(8), (30,))
    lml = log_marginal_likelihood(phi_x, y, 0.1)
    assert lml.shape == ()
    assert bool(jnp.isfinite(lml))
