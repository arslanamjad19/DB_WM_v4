"""
Loss terms for DB-WM training (Definition 3.1) plus the teacher-forcing rollout.

The total objective is

    L = lambda_recon  * L_recon          (data fit / encoder-state consistency)
      + lambda_dyn    * L_dyn            (teacher-forced dynamics prediction)
      + lambda_dppgp  * L_dPPGP          (calibrated GP: pixel NLL + trace + KL)
      + lambda_spec   * L_spec           (spectral-radius safety)

These are pure-JAX functions operating on already-computed arrays (features,
weights, targets, the transition matrix). The trainer wires them to the model
via ``model.apply`` so gradients flow to every parameter, including ``A``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from dbwm.dynamics.transition import spectral_radius


def gaussian_nll(y: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    """
    Mean per-element negative log-likelihood of a diagonal Gaussian.

    :param y: targets.
    :param mu: predicted means (same shape as ``y``).
    :param var: predicted variances (broadcastable to ``y``).
    :return: scalar mean NLL.
    """
    var = jnp.clip(var, 1e-8, None)
    return 0.5 * jnp.mean((y - mu) ** 2 / var + jnp.log(var) + jnp.log(2.0 * jnp.pi))


def trace_regularizer(feature_norms: jnp.ndarray, sigma_eps2: jnp.ndarray) -> jnp.ndarray:
    """
    Trace regulariser ``(1/b) sum_o (k_b - ||phi(o)||^2) / (2 sigma_eps^2)``
    with ``k_b = max_o ||phi(o)||^2`` (Section 3.2). Encourages uniform prior
    variance across inputs, preventing the rank-1 collapse of Remark 3.1.

    :param feature_norms: ``(N,)`` squared feature norms ``||phi(o)||^2``.
    :param sigma_eps2: scalar observation-noise variance.
    :return: scalar trace regulariser.
    """
    k_b = jnp.max(feature_norms)
    return jnp.mean(k_b - feature_norms) / (2.0 * sigma_eps2)


def spectral_penalty(a: jnp.ndarray, rho_max: float) -> jnp.ndarray:
    """
    Spectral-radius penalty ``(max(0, rho(A) - rho_max))^2`` (Section 5.2).

    :param a: ``(r, r)`` transition matrix.
    :param rho_max: target maximum spectral radius (1.0 for LST).
    :return: scalar penalty.
    """
    rho = spectral_radius(a)
    return jnp.maximum(0.0, rho - rho_max) ** 2


def teacher_forced_rollout(
    a: jnp.ndarray,
    w_seq: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,
    u_seq: Optional[jnp.ndarray] = None,
    threshold: float = 0.01,
    use_teacher_forcing: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Roll the linear dynamics through one trajectory with scheduled-sampling
    teacher forcing (user spec / v3 Sec. 8).

    Starting from ``w_seq[0]`` the model predicts ``w_hat_{t+1} = A w_t + B u_t``.
    The per-step error is ``||w_hat_{t+1} - w_seq[t+1]||^2``. If teacher forcing
    is on and that error exceeds ``threshold``, the *ground-truth* ``w_seq[t+1]``
    is fed as the next state instead of the prediction -- this curbs error
    accumulation while keeping the recursion differentiable (``jnp.where``; the
    branch selector is stop-gradient'd).

    :param a: ``(r, r)`` transition matrix.
    :param w_seq: ``(T, r)`` ground-truth encoded weight trajectory.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u_seq: ``(T, ell)`` forcing or ``None``.
    :param threshold: teacher-forcing error threshold ``tau``.
    :param use_teacher_forcing: enable/disable scheduled sampling.
    :return: ``(per_step_sq_err, w_hat_seq)`` with shapes ``(T-1,)`` and
             ``(T-1, r)``.
    """
    t = w_seq.shape[0]

    def step(carry, idx):
        w_state = carry
        u_t = None if u_seq is None else u_seq[idx]
        w_pred = a @ w_state
        if b is not None and u_t is not None:
            w_pred = w_pred + b @ u_t
        w_true_next = w_seq[idx + 1]
        err = jnp.sum((w_pred - w_true_next) ** 2)
        if use_teacher_forcing:
            feed_truth = jax.lax.stop_gradient(err > threshold)
            next_state = jnp.where(feed_truth, w_true_next, w_pred)
        else:
            next_state = w_pred
        return next_state, (err, w_pred)

    _, (errs, preds) = jax.lax.scan(step, w_seq[0], jnp.arange(t - 1))
    return errs, preds


def dynamics_loss(
    a: jnp.ndarray,
    w_batch: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,
    u_batch: Optional[jnp.ndarray] = None,
    threshold: float = 0.01,
    use_teacher_forcing: bool = True,
) -> jnp.ndarray:
    """
    Mean teacher-forced dynamics-prediction loss over a batch of trajectories.

    :param a: ``(r, r)`` transition matrix.
    :param w_batch: ``(b, T, r)`` encoded weight trajectories.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u_batch: ``(b, T, ell)`` forcing or ``None``.
    :param threshold: teacher-forcing threshold.
    :param use_teacher_forcing: enable scheduled sampling.
    :return: scalar dynamics loss.
    """

    def per_traj(w_seq, u_seq):
        errs, _ = teacher_forced_rollout(
            a, w_seq, b, u_seq, threshold, use_teacher_forcing
        )
        return jnp.mean(errs)

    if u_batch is None:
        losses = jax.vmap(lambda w: per_traj(w, None))(w_batch)
    else:
        losses = jax.vmap(per_traj)(w_batch, u_batch)
    return jnp.mean(losses)
