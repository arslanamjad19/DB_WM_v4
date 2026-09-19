"""
Tests for the closed-form least-squares identification and spectral utilities.

Terminology (Section 2.5 / Proposition 4.4): this is a least-squares / finite-section
Koopman fit, NOT a DMD -- the deep basis has already reduced the state to R^r, so a
projected DMD's truncating SVD would remove no dimension. The old ``dbwm.dynamics.dmdc``
names survive only as a deprecated shim, exercised by ``test_deprecated_dmdc_shim``.
"""
import jax
import jax.numpy as jnp

from dbwm.dynamics.identification import (
    least_squares_operator,
    joint_least_squares,
    two_stage_least_squares,
    process_noise_cov,
)
from dbwm.dynamics.transition import spectral_radius, clip_spectral_radius

# Local aliases keep the existing test bodies readable.
edmd = least_squares_operator
dmdc_joint = joint_least_squares
dmdc_two_stage = two_stage_least_squares


def test_deprecated_dmdc_shim_still_reexports():
    """The old DMD-flavoured names remain importable for backwards compatibility."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from dbwm.dynamics import dmdc

    assert dmdc.edmd is least_squares_operator
    assert dmdc.dmdc_joint is joint_least_squares
    assert dmdc.dmdc_two_stage is two_stage_least_squares


def test_edmd_recovers_known_operator():
    """EDMD should recover A from a clean autonomous trajectory w_{t+1}=A w_t."""
    key = jax.random.PRNGKey(0)
    r = 5
    a_true = 0.9 * jax.random.orthogonal(key, r)  # stable-ish
    w0 = jax.random.normal(jax.random.PRNGKey(1), (r,))
    traj = [w0]
    for _ in range(60):
        traj.append(a_true @ traj[-1])
    weights = jnp.stack(traj, axis=0)
    a_hat = edmd(weights, mu=1e-8)
    assert jnp.allclose(a_hat, a_true, atol=1e-3)


def test_dmdc_joint_recovers_A_and_B():
    """Joint DMDc recovers [A B] from a forced linear trajectory."""
    key = jax.random.PRNGKey(2)
    r, ell, t = 4, 2, 80
    a_true = 0.8 * jax.random.orthogonal(key, r)
    b_true = jax.random.normal(jax.random.PRNGKey(3), (r, ell))
    u = jax.random.normal(jax.random.PRNGKey(4), (t, ell))
    w = [jax.random.normal(jax.random.PRNGKey(5), (r,))]
    for i in range(t - 1):
        w.append(a_true @ w[-1] + b_true @ u[i])
    weights = jnp.stack(w, axis=0)
    a_hat, b_hat = dmdc_joint(weights, u, mu=1e-8)
    assert jnp.allclose(a_hat, a_true, atol=1e-2)
    assert jnp.allclose(b_hat, b_true, atol=1e-2)


def test_dmdc_two_stage_quiescent():
    """Two-stage DMDc recovers A from quiescent steps and B from residuals."""
    key = jax.random.PRNGKey(6)
    r, ell, t = 4, 2, 120
    a_true = 0.85 * jax.random.orthogonal(key, r)
    b_true = jax.random.normal(jax.random.PRNGKey(7), (r, ell))
    rng = jax.random.PRNGKey(8)
    # Sparse forcing: nonzero only every 5th step.
    u = jnp.zeros((t, ell))
    idx = jnp.arange(0, t, 5)
    u = u.at[idx].set(jax.random.normal(rng, (idx.shape[0], ell)))
    w = [jax.random.normal(jax.random.PRNGKey(9), (r,))]
    for i in range(t - 1):
        w.append(a_true @ w[-1] + b_true @ u[i])
    weights = jnp.stack(w, axis=0)
    a_hat, b_hat = dmdc_two_stage(weights, u, mu=1e-8, quiescent_threshold=1e-8)
    assert jnp.allclose(a_hat, a_true, atol=1e-2)
    assert jnp.allclose(b_hat, b_true, atol=1e-2)


def test_clip_spectral_radius():
    """Eigenvalue clipping enforces rho(A) <= rho_max."""
    key = jax.random.PRNGKey(10)
    a = 2.0 * jax.random.normal(key, (6, 6))  # likely unstable
    a_clipped = clip_spectral_radius(a, 1.0)
    assert float(spectral_radius(a_clipped)) <= 1.0 + 1e-4


def test_process_noise_psd():
    """Estimated process-noise covariance is symmetric PSD."""
    key = jax.random.PRNGKey(11)
    weights = jax.random.normal(key, (50, 5))
    a = edmd(weights)
    q = process_noise_cov(weights, a, jitter=1e-6)
    assert jnp.allclose(q, q.T, atol=1e-6)
    assert bool(jnp.all(jnp.linalg.eigvalsh(q) > 0))
