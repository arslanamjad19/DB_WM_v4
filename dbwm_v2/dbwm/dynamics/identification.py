"""
Closed-form system identification by **regularized matrix least squares** in the
deep-basis weight space (Section 2.5, Theorem 4.4, Proposition 4.4, Algorithm 3).

Terminology (Section 2.5 / Remark 4.1)
--------------------------------------
The operator fit here is the empirical **finite-section (Galerkin) approximation**
of the Koopman operator restricted to the learned dictionary
``{phi_i . g_theta}`` -- i.e. plain regularized least squares:

    A_hat = argmin_A ||W_+ - A W_-||_F^2 + mu ||A||_F^2
          = W_+ W_-^T (W_- W_-^T + mu I_r)^{-1}.

This is *not* described as a DMD, and no truncating SVD is performed. The reason
is Proposition 4.4: a projected/reduced-order DMD spends its SVD reducing the raw
``n``-dimensional snapshots to ``r`` coordinates -- but the deep basis
``phi_theta`` has **already** performed exactly that reduction. Applying an SVD to
the (already ``r``-dimensional) weights only re-expresses them in an orthonormal
frame; it removes no dimension and changes no nonzero eigenvalue. The truncation
step is therefore *redundant*, and the reduced-order operator is recovered exactly
by the normal equations at cost ``O(T r^2 + r^3)``.

:func:`reduced_order_operator` implements the DMD-style projected operator purely
so that :func:`verify_proposition_4_4` can *check* this equivalence numerically;
it is never needed on the training path.

Input-affine forcing (Remark 2.1, Corollary 4.3)
------------------------------------------------
With forcing, the joint fit of ``[A B]`` against the augmented regressor
``Omega = [W_- ; Upsilon]`` is the finite-section of the Koopman operator *with
inputs*. ``A`` stays a single, input-INDEPENDENT operator, so its spectrum
(Corollary 4.2) remains well defined -- the property the rejected ``A(a_t)`` form
destroys.

Regimes (Algorithm 3)
---------------------
* :func:`joint_least_squares`     -- Regime A: identify ``[A B]`` jointly
  (dense, high-SNR forcing).
* :func:`two_stage_least_squares` -- Regime B (**default**): ``A`` from *quiescent*
  transitions (``p_t = u_t = 0``), then ``[B_p B_u]`` from the residuals on all
  transitions. Statistically cleaner when forcing is sparse, as in agricultural
  LST, and it sidesteps the collinearity failure mode of Remark 6.1.

All functions take/return ``jnp`` arrays and are jit-compatible.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax.numpy as jnp

from dbwm.dynamics.transition import clip_spectral_radius


def _ridge_solve(target: jnp.ndarray, regressor: jnp.ndarray, mu: float) -> jnp.ndarray:
    """
    Solve ``G = target @ regressor^T (regressor regressor^T + mu I)^-1``.

    :param target: ``(p, m)`` left matrix (e.g. ``W_+``).
    :param regressor: ``(q, m)`` regressor (e.g. ``W_-`` or ``Omega``).
    :param mu: ridge parameter.
    :return: ``(p, q)`` operator ``G``.
    """
    q = regressor.shape[0]
    c = regressor @ regressor.T + mu * jnp.eye(q)  # (q, q)
    d = target @ regressor.T  # (p, q)
    return jnp.linalg.solve(c, d.T).T  # C symmetric, so solve C G^T = D^T


def least_squares_operator(weights: jnp.ndarray, mu: float = 1e-3) -> jnp.ndarray:
    """
    Autonomous operator (Theorem 4.4): the finite-section Koopman / least-squares
    fit ``A = argmin_A sum_t ||w_{t+1} - A w_t||^2 + mu||A||^2``.

    :param weights: ``(T, r)`` weight trajectory (rows are time steps).
    :param mu: ridge parameter.
    :return: ``(r, r)`` transition matrix ``A``. Cost ``O(T r^2 + r^3)``.
    """
    w_minus = weights[:-1].T  # (r, T-1)
    w_plus = weights[1:].T  # (r, T-1)
    return _ridge_solve(w_plus, w_minus, mu)


def joint_least_squares(
    weights: jnp.ndarray, forcing: jnp.ndarray, mu: float = 1e-3
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Regime A (Algorithm 3): joint fit ``[A B] = W_+ Omega^T (Omega Omega^T + mu I)^-1``
    with ``Omega = [W_- ; Upsilon]`` -- the Koopman-with-inputs finite section
    (Corollary 4.3).

    :param weights: ``(T, r)`` weight trajectory.
    :param forcing: ``(T, ell)`` raw forcing (only the first ``T-1`` rows used).
    :param mu: ridge parameter.
    :return: ``(A, B)`` with shapes ``(r, r)`` and ``(r, ell)``.
    """
    r = weights.shape[1]
    w_minus = weights[:-1].T  # (r, T-1)
    w_plus = weights[1:].T  # (r, T-1)
    ups = forcing[:-1].T  # (ell, T-1)
    omega = jnp.concatenate([w_minus, ups], axis=0)  # (r+ell, T-1)
    g = _ridge_solve(w_plus, omega, mu)  # (r, r+ell)
    return g[:, :r], g[:, r:]


def two_stage_least_squares(
    weights: jnp.ndarray,
    forcing: jnp.ndarray,
    mu: float = 1e-3,
    quiescent_threshold: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Regime B (Algorithm 3, **default**): two-stage identification.

    * **Stage I** -- fit ``A`` using only *quiescent* transitions
      (``|u_t^raw| ~ 0``: no rain, no irrigation), so sparse forcing cannot leak
      variance into the autonomous operator.
    * **Stage II** -- fit the forcing directions from the residuals on *all*
      transitions::

          dW_t = w_{t+1} - A w_t
          [B_p  B_u] = dW . Upsilon^T (Upsilon Upsilon^T + mu I_ell)^{-1}

      This is exactly the equation the thesis specifies for learning ``B``.

    :param weights: ``(T, r)`` weight trajectory.
    :param forcing: ``(T, ell)`` raw forcing.
    :param mu: ridge parameter.
    :param quiescent_threshold: rows of ``forcing`` with L1 norm below this count
                                as quiescent.
    :return: ``(A, B)`` with shapes ``(r, r)`` and ``(r, ell)``.
    """
    w_minus = weights[:-1]  # (T-1, r)
    w_plus = weights[1:]  # (T-1, r)
    ups = forcing[:-1]  # (T-1, ell)

    # Stage I: quiescent-only fit, expressed as a 0/1-weighted ridge so it stays
    # jit-friendly (no boolean indexing of dynamic size).
    active = jnp.sum(jnp.abs(ups), axis=-1)  # (T-1,)
    qmask = (active <= quiescent_threshold).astype(weights.dtype)  # (T-1,)
    wm = (w_minus * qmask[:, None]).T  # (r, T-1)
    wp = (w_plus * qmask[:, None]).T  # (r, T-1)
    a = _ridge_solve(wp, wm, mu)

    # Stage II: residual regression onto the inputs -> [B_p B_u].
    dw = (w_plus - w_minus @ a.T).T  # (r, T-1)
    b = _ridge_solve(dw, ups.T, mu)  # (r, ell)
    return a, b


def reduced_order_operator(
    weights: jnp.ndarray, rank: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The *projected* (DMD-style) reduced-order operator, provided only to verify
    Proposition 4.4 -- it is **not** used on the training path.

    With the thin SVD ``W_- = U S V^T``, the operator a projected DMD would fit in
    the orthonormal coordinates ``w~ = U^T w`` is ``A~ = U^T W_+ V S^{-1}``.

    :param weights: ``(T, r)`` weight trajectory.
    :param rank: truncation rank (defaults to the numerical rank of ``W_-``).
    :return: ``(A_tilde, U)`` with shapes ``(rho, rho)`` and ``(r, rho)``.
    """
    w_minus = weights[:-1].T  # (r, T-1)
    w_plus = weights[1:].T  # (r, T-1)
    u, s, vt = jnp.linalg.svd(w_minus, full_matrices=False)
    if rank is None:
        tol = jnp.max(s) * jnp.finfo(s.dtype).eps * max(w_minus.shape)
        rank = int(jnp.sum(s > tol))
    u_r, s_r, v_r = u[:, :rank], s[:rank], vt[:rank].T
    a_tilde = u_r.T @ w_plus @ v_r @ jnp.diag(1.0 / s_r)
    return a_tilde, u_r


def verify_proposition_4_4(
    weights: jnp.ndarray, tol: float = 1e-5
) -> Dict[str, object]:
    """
    Numerically verify Proposition 4.4: the least-squares operator ``A`` and the
    SVD-projected reduced operator ``A~`` share the same nonzero spectrum, so the
    truncating SVD of a projected DMD is redundant once the deep basis has already
    reduced the state to ``R^r``.

    :param weights: ``(T, r)`` weight trajectory.
    :param tol: tolerance on the sorted-eigenvalue mismatch.
    :return: dict with ``max_eig_diff``, ``rank``, and ``equivalent`` (bool).
    """
    a = least_squares_operator(weights, mu=0.0)  # exact LS (mu -> 0)
    a_tilde, u_r = reduced_order_operator(weights)
    rho = a_tilde.shape[0]

    ev_full = jnp.linalg.eigvals(a)
    ev_red = jnp.linalg.eigvals(a_tilde)
    # Compare the rho largest-magnitude eigenvalues of A against A~'s spectrum
    # (A's remaining r - rho eigenvalues are zero: directions unexcited by data).
    idx = jnp.argsort(-jnp.abs(ev_full))[:rho]
    ev_full_top = ev_full[idx]
    ev_full_sorted = ev_full_top[jnp.argsort(jnp.abs(ev_full_top))]
    ev_red_sorted = ev_red[jnp.argsort(jnp.abs(ev_red))]
    max_diff = float(jnp.max(jnp.abs(ev_full_sorted - ev_red_sorted))) if rho else 0.0
    return {
        "max_eig_diff": max_diff,
        "rank": int(rho),
        "equivalent": bool(max_diff < tol),
    }


def persistence_of_excitation(
    weights: jnp.ndarray, forcing: jnp.ndarray, tol: float = 1e-8
) -> Dict[str, object]:
    """
    Check the identifiability condition of Remark 6.1: the augmented regressor
    ``Omega = [W_- ; Upsilon]`` must have full row rank ``r + ell``.

    The dangerous failure mode for LST is forcing that is *collinear* with an
    autonomous mode (e.g. monsoon rain aligned with the seasonal swing), which
    makes ``A`` and ``B_p`` inseparable. This returns the numerical rank and the
    deficiency so the caller can warn.

    :param weights: ``(T, r)`` weight trajectory.
    :param forcing: ``(T, ell)`` forcing.
    :param tol: singular-value tolerance (relative to the largest).
    :return: dict with ``rank``, ``required``, ``deficiency``, ``satisfied``,
             and ``cond`` (condition number of ``Omega``).
    """
    r, ell = weights.shape[1], forcing.shape[1]
    omega = jnp.concatenate([weights[:-1].T, forcing[:-1].T], axis=0)  # (r+ell, T-1)
    s = jnp.linalg.svd(omega, compute_uv=False)
    smax = jnp.max(s)
    rank = int(jnp.sum(s > tol * smax))
    required = r + ell
    cond = float(smax / jnp.maximum(jnp.min(s), 1e-30))
    return {
        "rank": rank,
        "required": required,
        "deficiency": required - rank,
        "satisfied": bool(rank >= required),
        "cond": cond,
    }


def controllability_gramian(
    a: jnp.ndarray, b_u: jnp.ndarray, horizon: int
) -> jnp.ndarray:
    """
    Controllability Gramian of the *irrigation* channel (Proposition 4.3)::

        W_c(0,T) = sum_{k=0}^{T-1} A^k B_u B_u^T (A^k)^T.

    Precipitation is uncontrollable and is correctly **excluded**: it is a
    disturbance, so including it in ``W_c`` would be meaningless.

    :param a: ``(r, r)`` transition matrix.
    :param b_u: ``(r, ell_u)`` irrigation columns of ``B``.
    :param horizon: number of steps ``T``.
    :return: ``(r, r)`` Gramian.
    """
    r = a.shape[0]
    w_c = jnp.zeros((r, r))
    a_k = jnp.eye(r)
    for _ in range(horizon):
        w_c = w_c + a_k @ b_u @ b_u.T @ a_k.T
        a_k = a_k @ a
    return w_c


def process_noise_cov(
    weights: jnp.ndarray,
    a: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,
    forcing: Optional[jnp.ndarray] = None,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """
    Process-noise covariance from one-step residuals (Algorithm 3 step 5)::

        eta_t = w_{t+1} - A w_t - B_p p_t - B_u u_t
        Q     = (1/(T-2)) sum_t eta_t eta_t^T + nu I_r

    The noise is a *residual*, never a regressor.

    :param weights: ``(T, r)`` weight trajectory.
    :param a: ``(r, r)`` transition matrix.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param forcing: ``(T, ell)`` raw forcing or ``None``.
    :param jitter: diagonal conditioning ``nu``.
    :return: ``(r, r)`` covariance ``Q``.
    """
    r = weights.shape[1]
    pred = weights[:-1] @ a.T
    if b is not None and forcing is not None:
        pred = pred + forcing[:-1] @ b.T
    resid = weights[1:] - pred  # (T-1, r)
    t = resid.shape[0]
    q = (resid.T @ resid) / jnp.maximum(t - 1, 1)
    return q + jitter * jnp.eye(r)


def identify(
    weights: jnp.ndarray,
    forcing: Optional[jnp.ndarray],
    cfg,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full closed-form identification (Algorithm 3) dispatched from a
    :class:`~dbwm.config.DynamicsConfig`.

    :param weights: ``(T, r)`` weight trajectory.
    :param forcing: ``(T, ell)`` forcing or ``None`` (pure-temporal).
    :param cfg: dynamics configuration.
    :return: ``(A, B, Q)``. ``B`` is zeros if forcing is disabled/absent.
    """
    r = weights.shape[1]
    if (not cfg.use_forcing) or forcing is None or forcing.shape[1] == 0:
        a = least_squares_operator(weights, cfg.ridge_mu)
        b = jnp.zeros((r, max(cfg.input_dim, 1)))
        if cfg.clip_eigenvalues:
            a = clip_spectral_radius(a, cfg.rho_max)
        q = process_noise_cov(weights, a, jitter=cfg.process_noise_jitter)
        return a, b, q

    if cfg.identification == "joint":
        a, b = joint_least_squares(weights, forcing, cfg.ridge_mu)
    else:
        a, b = two_stage_least_squares(
            weights, forcing, cfg.ridge_mu, cfg.quiescent_threshold
        )
    if cfg.clip_eigenvalues:
        a = clip_spectral_radius(a, cfg.rho_max)
    q = process_noise_cov(weights, a, b, forcing, cfg.process_noise_jitter)
    return a, b, q


def koopman_modes(a: jnp.ndarray, dt: float = 1.0) -> Dict[str, jnp.ndarray]:
    """
    Koopman spectrum of the autonomous operator (Corollary 4.2).

    Well defined precisely *because* ``A`` is input-independent (Remark 2.1).

    :param a: ``(r, r)`` transition matrix.
    :param dt: sampling interval (days for LST/NDVI).
    :return: dict with eigenvalues, frequencies (cycles per unit time) and growth
             rates.
    """
    evals = jnp.linalg.eigvals(a)
    log_lambda = jnp.log(evals + 0j)
    freqs = jnp.abs(jnp.imag(log_lambda)) / (2.0 * jnp.pi * dt)
    growth = jnp.real(log_lambda) / dt
    return {"eigenvalues": evals, "frequencies": freqs, "growth_rates": growth}


def split_input_matrix(
    b: jnp.ndarray, names: Sequence[str]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decompose the learned ``B in R^{r x ell}`` into ``B = [B_p  B_u]``.

    Algorithm 3 fits ``B`` as a single block -- statistically the channels are just
    columns of one regression. But their **control-theoretic roles are opposite**:

    * ``B_p`` (precipitation, incl. its lags) is an *uncontrollable exogenous
      disturbance*. It enters the Kalman predict mean (Algorithm 2) and it must be
      kept **out** of the controllability Gramian (Proposition 4.3).
    * ``B_u`` (irrigation) is the *only* genuinely controllable actuator, and is the
      decision variable the CEM planner optimises.

    Collapsing the two would let the planner "decide" the weather, so the split is
    enforced explicitly rather than left implicit in column order.

    :param b: ``(r, ell)`` input matrix from :func:`identify`.
    :param names: length-``ell`` forcing channel names (``ForcingSeries.names``).
    :return: ``(B_p, B_u)`` with shapes ``(r, ell_p)`` and ``(r, ell_u)``.
    """
    from dbwm.data.forcing import channel_split

    precip_cols, irrig_cols = channel_split(names)
    if b.shape[1] != len(names):
        raise ValueError(
            "B has {} columns but {} forcing channel names were given.".format(
                b.shape[1], len(names)
            )
        )
    b_p = b[:, jnp.asarray(precip_cols, dtype=int)] if precip_cols else jnp.zeros((b.shape[0], 0))
    b_u = b[:, jnp.asarray(irrig_cols, dtype=int)] if irrig_cols else jnp.zeros((b.shape[0], 0))
    return b_p, b_u


def forcing_response_maps(
    b: jnp.ndarray, phi_pixels: jnp.ndarray
) -> jnp.ndarray:
    """
    Spatial forcing-response maps ``R_j(x) = <B[:, j], Psi(x)>``.

    A unit of channel ``j`` (e.g. 1 mm of rain) perturbs the field by
    ``delta f(x) = <B[:, j], Psi(x)>``. For precipitation and irrigation these
    maps should be predominantly **negative** over vegetated / irrigated pixels
    (evaporative cooling of LST), which is a physical sanity check on ``B``.

    :param b: ``(r, ell)`` input matrix.
    :param phi_pixels: ``(N, r)`` spatial basis features at query pixels.
    :return: ``(ell, N)`` response maps, one row per forcing channel.
    """
    return (phi_pixels @ b).T
