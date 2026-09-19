"""
Visual Basis Observer: a Kalman filter in the deep-basis weight space R^r
(Algorithm 2 / Theorem 4.3).

The state is the latent weight ``w_t in R^r``. The "measurement" at time ``t`` is
the encoded observation ``phi_theta(o_t) in R^r`` (Section 2.3: in the visual
setting the encoder *is* the measurement map, so the observation matrix is the
identity ``I_r`` and the innovation is ``phi_theta(o_t) - w_{t|t-1}``).

This module provides:

* :func:`kalman_predict` / :func:`kalman_update` -- one PREDICT / UPDATE step.
* :func:`filter_sequence` -- closed-loop filtering over a sequence of encoded
  observations (the inference-time "predict-then-correct" of the user spec).
* :func:`open_loop_forecast` -- ``H``-step open-loop rollout with honest,
  horizon-growing covariance (v3 Sec. 5.3, dynamics-residual inflation).

All operations are ``O(r^2)``-``O(r^3)`` per step (independent of pixel count).
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def kalman_predict(
    w: jnp.ndarray,
    p: jnp.ndarray,
    a: jnp.ndarray,
    q: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,
    u: Optional[jnp.ndarray] = None,
    gamma_dyn: jnp.ndarray | float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kalman PREDICT step ``w_{t|t-1} = A w + B u``,
    ``P_{t|t-1} = A P A^T + Q + Gamma_dyn``.

    :param w: ``(r,)`` posterior mean ``w_{t-1|t-1}``.
    :param p: ``(r, r)`` posterior covariance ``P_{t-1|t-1}``.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u: ``(ell,)`` raw forcing or ``None``.
    :param gamma_dyn: scalar dynamics-residual inflation added to the diagonal
                      (v3 Sec. 5.3: ``Gamma_dyn = gamma * tr(Q) * I_r``).
    :return: ``(w_pred, P_pred)``.
    """
    w_pred = a @ w
    if b is not None and u is not None:
        w_pred = w_pred + b @ u
    r = a.shape[0]
    inflation = gamma_dyn * (jnp.trace(q) / r) * jnp.eye(r)
    p_pred = a @ p @ a.T + q + inflation
    return w_pred, p_pred


def kalman_update(
    w_pred: jnp.ndarray,
    p_pred: jnp.ndarray,
    phi_obs: jnp.ndarray,
    sigma_eps2: jnp.ndarray | float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kalman UPDATE step with identity observation map (measurement = encoded obs).

    Innovation ``nu = phi_obs - w_pred``; ``S = P_pred + sigma_eps^2 I``;
    gain ``K = P_pred S^{-1}``; ``w = w_pred + K nu``; ``P = (I - K) P_pred``.

    :param w_pred: ``(r,)`` predicted mean.
    :param p_pred: ``(r, r)`` predicted covariance.
    :param phi_obs: ``(r,)`` encoded observation ``phi_theta(o_t)``.
    :param sigma_eps2: scalar measurement-noise variance.
    :return: ``(w_post, P_post)``.
    """
    r = w_pred.shape[0]
    innovation = phi_obs - w_pred
    s = p_pred + sigma_eps2 * jnp.eye(r)
    gain = jnp.linalg.solve(s.T, p_pred.T).T  # K = P_pred S^{-1}
    w_post = w_pred + gain @ innovation
    p_post = (jnp.eye(r) - gain) @ p_pred
    return w_post, p_post


def filter_sequence(
    phi_obs_seq: jnp.ndarray,
    a: jnp.ndarray,
    q: jnp.ndarray,
    sigma_eps2: jnp.ndarray | float,
    p0: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    u_seq: Optional[jnp.ndarray] = None,
    gamma_dyn: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Closed-loop filtering over a full sequence of encoded observations.

    This is the inference-time procedure of the user spec: at each step the
    filter first PREDICTS using its prior (the dynamics), then UPDATES using the
    encoded test-set observation as a measurement.

    :param phi_obs_seq: ``(T, r)`` encoded observations ``phi_theta(o_t)``.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param sigma_eps2: measurement-noise variance.
    :param p0: ``(r, r)`` initial covariance (default ``sigma_eps^2 I``).
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u_seq: ``(T, ell)`` forcing or ``None``.
    :param gamma_dyn: dynamics-residual inflation factor.
    :return: ``(w_filtered, P_filtered)`` with shapes ``(T, r)`` and ``(T, r, r)``.
    """
    r = a.shape[0]
    if p0 is None:
        p0 = sigma_eps2 * jnp.eye(r)
    w0 = phi_obs_seq[0]  # initialise state from the first encoded observation

    def step(carry, idx):
        w_prev, p_prev = carry
        u_t = None if u_seq is None else u_seq[idx]
        w_pred, p_pred = kalman_predict(w_prev, p_prev, a, q, b, u_t, gamma_dyn)
        w_post, p_post = kalman_update(w_pred, p_pred, phi_obs_seq[idx + 1], sigma_eps2)
        return (w_post, p_post), (w_post, p_post)

    (_, _), (w_seq, p_seq) = jax.lax.scan(
        step, (w0, p0), jnp.arange(phi_obs_seq.shape[0] - 1)
    )
    # Prepend the initial (unfiltered) state so outputs align with the inputs.
    w_all = jnp.concatenate([w0[None], w_seq], axis=0)
    p_all = jnp.concatenate([p0[None], p_seq], axis=0)
    return w_all, p_all


def open_loop_forecast(
    w0: jnp.ndarray,
    a: jnp.ndarray,
    q: jnp.ndarray,
    horizon: int,
    p0: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    u_seq: Optional[jnp.ndarray] = None,
    gamma_dyn: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    ``H``-step open-loop rollout (no measurement updates) with horizon-growing
    covariance (v3 Sec. 5.3 / Theorem 4.2 regime).

    :param w0: ``(r,)`` initial (e.g. Kalman-corrected) state.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param horizon: number of steps ``H``.
    :param p0: ``(r, r)`` initial covariance (default zeros -- certain start).
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u_seq: ``(H, ell)`` known forcing forecasts or ``None``.
    :param gamma_dyn: dynamics-residual inflation factor.
    :return: ``(w_forecast, P_forecast)`` with shapes ``(H, r)`` and ``(H, r, r)``.
    """
    r = a.shape[0]
    if p0 is None:
        p0 = jnp.zeros((r, r))

    def step(carry, idx):
        w_prev, p_prev = carry
        u_t = None if u_seq is None else u_seq[idx]
        w_next, p_next = kalman_predict(w_prev, p_prev, a, q, b, u_t, gamma_dyn)
        return (w_next, p_next), (w_next, p_next)

    (_, _), (w_seq, p_seq) = jax.lax.scan(step, (w0, p0), jnp.arange(horizon))
    return w_seq, p_seq
