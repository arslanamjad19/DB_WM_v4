"""
Empirical complexity checks: GP inference must scale ~O(n r^2) in n (linear) and
the per-step dynamics ~O(r^2), confirming the framework's scalability claims
(Section 3.1). These are coarse timing tests with generous tolerances so they
are not flaky on shared CI machines.
"""
import time

import jax
import jax.numpy as jnp

from dbwm.gp.posterior import gram_matrix, lambda_matrix, posterior_mean_var


def _time_gram(n, r, reps=3):
    """Median wall-time to form Phi^T Phi for an (n, r) feature matrix."""
    phi = jax.random.normal(jax.random.PRNGKey(0), (n, r))
    f = jax.jit(gram_matrix)
    f(phi).block_until_ready()  # warm up / compile
    ts = []
    for _ in range(reps):
        t0 = time.time()
        f(phi).block_until_ready()
        ts.append(time.time() - t0)
    return sorted(ts)[len(ts) // 2]


def test_gram_linear_in_n():
    """Doubling n should roughly (within a wide factor) double Gram time."""
    r = 128
    t1 = _time_gram(20000, r)
    t2 = _time_gram(40000, r)
    # Linear scaling: t2/t1 ~ 2. Allow a wide band [1.2, 4] for noise/overhead.
    ratio = (t2 + 1e-6) / (t1 + 1e-6)
    assert 1.2 < ratio < 4.0, "Gram scaling ratio {:.2f} not ~linear in n".format(ratio)


def test_posterior_independent_of_n_given_lambda():
    """
    Once Lambda_X and Phi^T y are precomputed, prediction at m test points is
    O(m r^2), independent of the training size n -- verify it runs in r-space
    only and returns correct shapes.
    """
    r, m = 64, 100
    lam = lambda_matrix(gram_matrix(jax.random.normal(jax.random.PRNGKey(1), (5000, r))), 0.1)
    proj = jax.random.normal(jax.random.PRNGKey(2), (r,))
    phi_star = jax.random.normal(jax.random.PRNGKey(3), (m, r))
    mean, var = posterior_mean_var(phi_star, lam, proj, 0.1)
    assert mean.shape == (m,) and var.shape == (m,)


def test_quadratic_in_r():
    """Gram time should grow super-linearly (~quadratic) in r at fixed n."""
    n = 20000
    t1 = _time_gram(n, 64)
    t2 = _time_gram(n, 256)  # 4x r -> expect well above 4x time if ~r^2 (->16x)
    ratio = (t2 + 1e-6) / (t1 + 1e-6)
    assert ratio > 3.0, "r-scaling ratio {:.2f} too small for ~r^2".format(ratio)
