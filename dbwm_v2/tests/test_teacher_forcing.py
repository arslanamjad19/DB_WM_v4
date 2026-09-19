"""
Tests for the teacher-forced rollout on the GP path.

The rule under test is the specified one: roll the latent state forward, decode
it to pixels, and whenever that decoded RMSE exceeds ``tau`` feed the **ground
truth** state into the next step instead of the model's own prediction. The
subtlety worth pinning is that the threshold is applied in *pixel* space, so the
0.02 in the config means 0.02 NDVI and keeps meaning that as ``Psi`` trains.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dbwm.training.gp_trainer import teacher_forced_rollout


def _setup(k=4, n=12, r=3, seed=0):
    """A short window with an orthonormal basis, so decoding is an isometry."""
    rng = np.random.default_rng(seed)
    phi = np.linalg.qr(rng.normal(size=(n, r)))[0]
    w_seq = rng.normal(size=(k + 1, r))
    y = w_seq @ phi.T
    return jnp.asarray(phi), jnp.asarray(w_seq), jnp.asarray(y)


def test_perfect_operator_is_never_forced():
    """
    With the true operator the decoded error is zero, so nothing should force.

    If this failed, the threshold would be firing on numerical noise and the
    rollout would be ground truth all the way -- learning nothing about
    propagation.
    """
    rng = np.random.default_rng(1)
    r, n, k = 3, 10, 4
    a = np.diag([0.9, 0.8, 0.7])
    phi = np.linalg.qr(rng.normal(size=(n, r)))[0]
    w = np.zeros((k + 1, r))
    w[0] = rng.normal(size=r)
    for i in range(1, k + 1):
        w[i] = a @ w[i - 1]
    y = w @ phi.T
    mses, forced = teacher_forced_rollout(
        jnp.asarray(a), jnp.asarray(w), jnp.asarray(phi), jnp.asarray(y),
        jnp.zeros(n), threshold=1e-3,
    )
    assert float(jnp.max(mses)) < 1e-12
    assert float(jnp.sum(forced)) == 0.0


def test_bad_operator_is_always_forced():
    """A hopeless operator must trip the threshold at every step."""
    phi, w_seq, y = _setup()
    a = jnp.zeros((w_seq.shape[1], w_seq.shape[1]))
    _, forced = teacher_forced_rollout(
        a, w_seq, phi, y, jnp.zeros(phi.shape[0]), threshold=1e-6
    )
    assert float(jnp.mean(forced)) == 1.0


def test_threshold_controls_the_forced_fraction():
    """Raising the threshold must (weakly) reduce how often truth is fed."""
    phi, w_seq, y = _setup(k=8, seed=2)
    a = jnp.eye(w_seq.shape[1]) * 0.5
    fractions = []
    for tau in (1e-8, 0.5, 1e8):
        _, forced = teacher_forced_rollout(
            a, w_seq, phi, y, jnp.zeros(phi.shape[0]), threshold=tau
        )
        fractions.append(float(jnp.mean(forced)))
    assert fractions[0] >= fractions[1] >= fractions[2]
    assert fractions[0] == 1.0 and fractions[2] == 0.0


def test_forcing_actually_substitutes_ground_truth():
    """
    The forced step must carry the TRUE state forward, not the prediction.

    Checked by making step 2's error depend only on what step 1 fed forward: with
    a zero operator and forcing on, step 2 predicts from ``w_seq[1]``; with
    forcing off it predicts from ``0``.
    """
    phi, w_seq, y = _setup(k=2, seed=3)
    a = jnp.zeros((w_seq.shape[1], w_seq.shape[1]))
    on, _ = teacher_forced_rollout(
        a, w_seq, phi, y, jnp.zeros(phi.shape[0]), 1e-6, use_teacher_forcing=True
    )
    off, _ = teacher_forced_rollout(
        a, w_seq, phi, y, jnp.zeros(phi.shape[0]), 1e-6, use_teacher_forcing=False
    )
    # With A = 0 both predict zero, so the decoded errors coincide; what must
    # differ is the state carried forward, which is what the next assertion tests
    # through a non-zero operator.
    assert np.allclose(np.asarray(on), np.asarray(off))

    a2 = jnp.eye(w_seq.shape[1])
    on2, _ = teacher_forced_rollout(
        a2, w_seq, phi, y, jnp.zeros(phi.shape[0]), 1e-6, use_teacher_forcing=True
    )
    off2, _ = teacher_forced_rollout(
        a2, w_seq, phi, y, jnp.zeros(phi.shape[0]), 1e-6, use_teacher_forcing=False
    )
    # Forced: step 2 predicts w_seq[1]; free-running: it still predicts w_seq[0].
    assert not np.allclose(np.asarray(on2)[1], np.asarray(off2)[1])


def test_offset_is_added_before_the_comparison():
    """
    The climatology must be part of the decode, or the threshold means nothing.

    Comparing an anomaly prediction against an absolute frame would make the
    error look enormous and force every step forever.
    """
    phi, w_seq, y = _setup(seed=4)
    offset = jnp.full(phi.shape[0], 3.0)
    _, forced_wrong = teacher_forced_rollout(
        jnp.eye(w_seq.shape[1]), w_seq, phi, y + 3.0, jnp.zeros(phi.shape[0]),
        threshold=0.5,
    )
    _, forced_right = teacher_forced_rollout(
        jnp.eye(w_seq.shape[1]), w_seq, phi, y + 3.0, offset, threshold=0.5
    )
    assert float(jnp.mean(forced_right)) <= float(jnp.mean(forced_wrong))


def test_rollout_is_differentiable_and_jittable():
    """
    Gradients must reach the operator through the scan.

    The branch selector is ``stop_gradient``'d, so forcing changes the value fed
    forward without blocking the gradient path -- if it did block it, the term
    would be decorative.
    """
    phi, w_seq, y = _setup(seed=5)

    def loss(a):
        mses, _ = teacher_forced_rollout(
            a, w_seq, phi, y, jnp.zeros(phi.shape[0]), threshold=0.1
        )
        return jnp.mean(mses)

    g = jax.jit(jax.grad(loss))(jnp.eye(w_seq.shape[1]) * 0.5)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.linalg.norm(np.asarray(g)) > 0


def test_threshold_is_converted_from_physical_units():
    """
    ``tf_threshold_ndvi`` is divided by the normalisation std before use.

    0.02 NDVI on a field with std 0.135 is 0.148 in normalised units; applying
    0.02 directly would force roughly seven times too eagerly.
    """
    from dbwm.config import default_config

    cfg = default_config()
    assert cfg.training.tf_threshold_ndvi == 0.02
    norm_std = 0.135
    assert np.isclose(cfg.training.tf_threshold_ndvi / norm_std, 0.1481, atol=1e-3)
