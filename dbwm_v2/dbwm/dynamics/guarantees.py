"""
Structural guarantees of Section 4, checked numerically on the *identified*
operator.

Remark 4.1 is what makes this module well posed: every guarantee in Section 4 is
stated purely in terms of ``(A, Phi_X, B, Q, sigma_eps^2)`` and **none of them
references how ``A`` was obtained**. So these are post-identification certificates
-- run them on whatever ``A`` came out of Algorithm 3 (or out of SGD) and they
either hold or they do not.

Contents
--------
* :func:`is_shaded`               -- Definition 4.1 (+ Proposition 4.1 for SwiGLU).
* :func:`observability_matrix` /
  :func:`check_observability`     -- Theorem 4.1 / Proposition 4.2.
* :func:`cyclic_index`            -- Corollary 4.1 (sensor lower bound).
* :func:`check_controllability`   -- Proposition 4.3 (irrigation channel only).
* :func:`open_loop_error_bound`   -- Theorem 4.2 (both the rho != 1 and rho = 1 branches).
* :func:`solve_dare` /
  :func:`steady_state_error`      -- Theorem 4.3 (bounded observer error).

Diagnostics, not training code: these run once after identification, at ``O(r^3)``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import jax.numpy as jnp

from dbwm.dynamics.transition import spectral_radius, spectral_norm


# --------------------------------------------------------------------------- #
# Definition 4.1 / Proposition 4.1 -- shadedness
# --------------------------------------------------------------------------- #
def is_shaded(phi: jnp.ndarray, tol: float = 1e-8) -> Dict[str, object]:
    """
    Definition 4.1: ``Phi_X`` is **shaded** if every basis function is activated by
    at least one observation -- i.e. each *column* ``j`` has some row ``i`` with
    ``phi_j(o_i) != 0``.

    Shadedness is assumption (A2) of Theorem 4.1: a column that is identically zero
    means basis direction ``j`` is invisible to every sensor, so ``w_j`` can never
    be recovered and the observability matrix drops rank.

    Proposition 4.1 says this holds *automatically*, with probability 1, for the
    SwiGLU expansion under any continuous input distribution (the zero set of
    SwiGLU has measure zero). This function is the empirical check of that claim --
    a *dead* basis column here means the expansion has collapsed, not that the
    proposition is wrong.

    :param phi: ``(N, r)`` feature matrix ``Phi_X``.
    :param tol: magnitude below which an entry counts as zero.
    :return: dict with ``shaded`` (bool), ``n_dead`` columns, ``dead_fraction``,
             and ``min_col_absmax`` (the weakest column's peak activation).
    """
    col_absmax = jnp.max(jnp.abs(phi), axis=0)  # (r,)
    dead = col_absmax <= tol
    n_dead = int(jnp.sum(dead))
    return {
        "shaded": bool(n_dead == 0),
        "n_dead": n_dead,
        "dead_fraction": float(n_dead / phi.shape[1]),
        "min_col_absmax": float(jnp.min(col_absmax)),
    }


# --------------------------------------------------------------------------- #
# Theorem 4.1 / Proposition 4.2 -- observability
# --------------------------------------------------------------------------- #
def observability_matrix(
    phi: jnp.ndarray, a: jnp.ndarray, taus: Sequence[int]
) -> jnp.ndarray:
    """
    Generalized observability matrix of Theorem 4.1::

        O_Upsilon = [ Phi_X A^{tau_1} ; ... ; Phi_X A^{tau_L} ]  in  R^{NL x r}.

    :param phi: ``(N, r)`` feature/observation matrix ``Phi_X``.
    :param a: ``(r, r)`` transition matrix.
    :param taus: the time instances ``Upsilon = {tau_1, ..., tau_L}``.
    :return: ``(N*L, r)`` observability matrix.
    """
    blocks = [phi @ jnp.linalg.matrix_power(a, int(tau)) for tau in taus]
    return jnp.concatenate(blocks, axis=0)


def check_observability(
    phi: jnp.ndarray,
    a: jnp.ndarray,
    taus: Optional[Sequence[int]] = None,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """
    Verify Theorem 4.1 / Proposition 4.2: the pair ``(Phi_X, A)`` is observable iff
    ``rank(O_Upsilon) = r``, in which case ``w_0`` is uniquely recoverable from the
    observations at ``Upsilon``.

    The three hypotheses are reported separately so a failure is actionable:

    * **(A1)** ``A`` has distinct eigenvalues (full-rank Jordan form). Reported as
      ``distinct_eigenvalues``; near-duplicates are the usual culprit.
    * **(A2)** ``Phi_X`` is shaded (:func:`is_shaded`).
    * **(A3)** ``|Upsilon| >= r`` distinct time instances.

    Note (Proposition 4.2): observability depends on ``(Phi_X, A)`` **only**. The
    input term ``B u_t^raw`` is a known signal and cannot affect it -- so a model
    with forcing is observable exactly when its autonomous part is.

    :param phi: ``(N, r)`` feature matrix.
    :param a: ``(r, r)`` transition matrix.
    :param taus: time instances; defaults to ``0, 1, ..., r-1`` (which satisfies A3).
    :param tol: relative singular-value tolerance for the rank test.
    :return: dict with ``observable``, ``rank``, ``required`` (= r), the three
             hypothesis flags, and the smallest singular value of ``O_Upsilon``.
    """
    r = a.shape[0]
    if taus is None:
        taus = list(range(r))

    o_mat = observability_matrix(phi, a, taus)
    s = jnp.linalg.svd(o_mat, compute_uv=False)
    smax = jnp.max(s)
    rank = int(jnp.sum(s > tol * jnp.maximum(smax, 1e-30)))

    evals = jnp.linalg.eigvals(a)
    pair_gap = jnp.abs(evals[:, None] - evals[None, :]) + jnp.eye(r) * 1e9
    min_gap = float(jnp.min(pair_gap))

    shade = is_shaded(phi, tol)
    return {
        "observable": bool(rank >= r),
        "rank": rank,
        "required": r,
        "min_singular_value": float(jnp.min(s)),
        # (A1) distinct eigenvalues
        "distinct_eigenvalues": bool(min_gap > tol),
        "min_eigenvalue_gap": min_gap,
        # (A2) shadedness
        "shaded": shade["shaded"],
        "n_dead_basis": shade["n_dead"],
        # (A3) enough time instances
        "n_time_instances": len(taus),
        "time_instances_sufficient": bool(len(taus) >= r),
    }


def cyclic_index(a: jnp.ndarray, tol: float = 1e-6) -> int:
    """
    Corollary 4.1 (sensor lower bound): the minimum number of observations needed
    for observability is the **cyclic index**

        ell = max_i gm(lambda_i),

    the largest *geometric* multiplicity over the distinct eigenvalues of ``A``,
    where ``gm(lambda) = r - rank(A - lambda I)``.

    In the visual setting the encoder supplies a full ``r``-vector each step
    (``N = r`` "sensors"), so this bound is satisfied trivially -- it becomes
    binding only if one sub-samples the basis / uses a sparse sensor set.

    :param a: ``(r, r)`` transition matrix.
    :param tol: tolerance for clustering eigenvalues and for the rank test.
    :return: the cyclic index ``ell >= 1``.
    """
    r = a.shape[0]
    evals = jnp.linalg.eigvals(a)

    # Cluster near-duplicate eigenvalues into "distinct" representatives.
    reps: list = []
    for lam in evals:
        if not any(abs(complex(lam) - complex(m)) <= tol for m in reps):
            reps.append(lam)

    a_c = a.astype(jnp.complex64)
    eye = jnp.eye(r, dtype=jnp.complex64)
    best = 1
    for lam in reps:
        s = jnp.linalg.svd(a_c - lam * eye, compute_uv=False)
        smax = jnp.max(s)
        rank = int(jnp.sum(s > tol * jnp.maximum(smax, 1e-30)))
        best = max(best, r - rank)
    return int(best)


# --------------------------------------------------------------------------- #
# Proposition 4.3 -- controllability (irrigation only)
# --------------------------------------------------------------------------- #
def check_controllability(
    a: jnp.ndarray, b_u: jnp.ndarray, horizon: Optional[int] = None, tol: float = 1e-8
) -> Dict[str, object]:
    """
    Proposition 4.3: the system is controllable iff the Gramian

        W_c(0,T) = sum_{k=0}^{T-1} A^k B_u B_u^T (A^k)^T

    is positive definite. **Precipitation is deliberately excluded** -- it is an
    uncontrollable disturbance, and putting ``B_p`` into ``W_c`` would assert the
    model can *choose the weather*. Pass only the irrigation columns ``B_u``
    (:func:`dbwm.data.forcing.channel_split` gives the indices).

    For a scalar actuator this is a rank-1-per-step accumulation, so full rank
    needs ``T >= r`` steps *and* ``B_u`` not orthogonal to any left-eigenspace of
    ``A``. With ``r`` in the hundreds and a single irrigation scalar, expect this
    to report **not controllable** -- that is the honest, physically correct answer
    (one scalar of water per step cannot steer a 512-dimensional thermal state
    anywhere you like), and ``min_eigenvalue`` / ``rank`` quantify *how much* of
    the state is reachable.

    :param a: ``(r, r)`` transition matrix.
    :param b_u: ``(r, ell_u)`` irrigation columns of ``B``.
    :param horizon: number of steps ``T`` (defaults to ``r``).
    :param tol: relative eigenvalue tolerance for the rank test.
    :return: dict with ``controllable``, ``rank``, ``required``, ``min_eigenvalue``
             and the Gramian's condition number.
    """
    from dbwm.dynamics.identification import controllability_gramian

    r = a.shape[0]
    t = r if horizon is None else horizon
    w_c = controllability_gramian(a, b_u, t)

    evals = jnp.linalg.eigvalsh((w_c + w_c.T) / 2.0)  # symmetric PSD
    emax = jnp.max(evals)
    rank = int(jnp.sum(evals > tol * jnp.maximum(emax, 1e-30)))
    emin = float(jnp.min(evals))
    return {
        "controllable": bool(rank >= r),
        "rank": rank,
        "required": r,
        "horizon": t,
        "min_eigenvalue": emin,
        "max_eigenvalue": float(emax),
        "cond": float(emax / max(emin, 1e-30)),
    }


# --------------------------------------------------------------------------- #
# Theorem 4.2 -- open-loop prediction error bound
# --------------------------------------------------------------------------- #
def open_loop_error_bound(
    a: jnp.ndarray,
    eps_enc: float,
    eps_dyn: float,
    horizon: int,
    rho: Optional[float] = None,
) -> jnp.ndarray:
    """
    Theorem 4.2, the ``T``-step open-loop bound::

        ||w_T - w_T*||  <=  rho^T eps_enc + (rho^T - 1)/(rho - 1) eps_dyn   (rho != 1)
                         =  eps_enc + T eps_dyn                             (rho  = 1)

    **``rho`` here is the spectral NORM ``||A||_2``, not the spectral radius.**
    The framework is deliberate about this and the distinction matters in practice:
    ``rho(A) <= ||A||_2`` always, with equality only for normal ``A``. Clipping the
    *radius* to ``rho_max = 1`` (Algorithm 3 step 4, ``L_spec``) therefore does
    **not** bound the norm -- a non-normal ``A`` can satisfy ``rho(A) = 1`` while
    ``||A||_2 >> 1``, producing real transient amplification over a multi-day
    forecast. :func:`error_bound_report` returns both so the gap is visible.

    For a near-conservative field like LST the design point is ``rho ~ 1``, giving
    the **linear** growth ``eps_enc + T eps_dyn`` -- the honest characterisation of
    a multi-day forecast, rather than the optimistic geometric decay a ``rho < 1``
    model would imply.

    :param a: ``(r, r)`` transition matrix (used to compute ``||A||_2``).
    :param eps_enc: encoder error ``sup_t ||w_t^encoded - w_t^true||``.
    :param eps_dyn: one-step dynamics residual under the *known* forcing,
                    ``sup_t ||w_{t+1}* - A w_t* - B_p p_t - B_u u_t||``.
    :param horizon: prediction horizon ``T``.
    :param rho: override for ``||A||_2`` (computed from ``a`` if ``None``).
    :return: scalar error bound.
    """
    rho_v = spectral_norm(a) if rho is None else jnp.asarray(rho, dtype=jnp.float32)
    t = float(horizon)

    # Guard the (rho - 1) division so the unused branch cannot emit inf/NaN.
    near_one = jnp.abs(rho_v - 1.0) < 1e-6
    denom = jnp.where(near_one, 1.0, rho_v - 1.0)
    rho_t = rho_v**t

    geometric = rho_t * eps_enc + (rho_t - 1.0) / denom * eps_dyn
    linear = eps_enc + t * eps_dyn
    return jnp.where(near_one, linear, geometric)


def empirical_error_terms(
    w_true: jnp.ndarray,
    w_encoded: jnp.ndarray,
    a: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,
    forcing: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Estimate the two constants of Theorem 4.2 from data.

    ``eps_dyn`` is measured **under the known forcing** -- i.e. it is the residual
    the input-affine model cannot explain *after* precipitation and irrigation have
    been accounted for. Measuring it without the ``B u`` term would inflate it with
    forcing signal and make the bound vacuous.

    :param w_true: ``(T, r)`` reference weight trajectory.
    :param w_encoded: ``(T, r)`` encoder outputs (equal to ``w_true`` in the visual
                      case, where the encoder *is* the state).
    :param a: ``(r, r)`` transition matrix.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param forcing: ``(T, ell)`` raw forcing or ``None``.
    :return: dict with ``eps_enc`` and ``eps_dyn`` (sup-norms over the trajectory).
    """
    eps_enc = float(jnp.max(jnp.linalg.norm(w_encoded - w_true, axis=-1)))

    pred = w_true[:-1] @ a.T
    if b is not None and forcing is not None:
        pred = pred + forcing[:-1] @ b.T
    resid = w_true[1:] - pred
    eps_dyn = float(jnp.max(jnp.linalg.norm(resid, axis=-1)))
    return {"eps_enc": eps_enc, "eps_dyn": eps_dyn}


def error_bound_report(
    a: jnp.ndarray, eps_enc: float, eps_dyn: float, horizon: int
) -> Dict[str, object]:
    """
    Theorem 4.2 bound plus the radius/norm gap that governs whether it is tight.

    :param a: ``(r, r)`` transition matrix.
    :param eps_enc: encoder error.
    :param eps_dyn: one-step dynamics residual under known forcing.
    :param horizon: horizon ``T``.
    :return: dict with ``bound``, ``spectral_norm``, ``spectral_radius``,
             ``non_normality`` (norm / radius) and the ``regime`` in force.
    """
    norm = float(spectral_norm(a))
    radius = float(spectral_radius(a))
    bound = float(open_loop_error_bound(a, eps_enc, eps_dyn, horizon))
    regime = "linear (rho ~ 1)" if abs(norm - 1.0) < 1e-6 else (
        "contractive (rho < 1)" if norm < 1.0 else "expansive (rho > 1)"
    )
    return {
        "bound": bound,
        "horizon": horizon,
        "spectral_norm": norm,
        "spectral_radius": radius,
        # > 1 means A is non-normal: radius-clipping does NOT bound the norm, so
        # transient growth is possible even with rho(A) <= 1.
        "non_normality": norm / max(radius, 1e-30),
        "regime": regime,
        "eps_enc": eps_enc,
        "eps_dyn": eps_dyn,
    }


# --------------------------------------------------------------------------- #
# Theorem 4.3 -- steady-state observer error (DARE)
# --------------------------------------------------------------------------- #
def solve_dare(
    a: jnp.ndarray,
    q: jnp.ndarray,
    sigma_eps2: float,
    phi: Optional[jnp.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """
    Theorem 4.3: solve the discrete algebraic Riccati equation in ``R^{r x r}``::

        P = A P A^T + Q - A P Phi^T (Phi P Phi^T + sigma_eps^2 I_N)^{-1} Phi P A^T

    by fixed-point iteration of the Riccati recursion. If ``(Phi, A)`` is observable
    and ``(Q^{1/2}, A)`` stabilizable, ``P_inf`` exists, is unique, and bounds the
    estimation error **independently of the prediction horizon** -- this is exactly
    what buys the Kalman-corrected forecast its horizon-free error, in contrast to
    the open-loop bound of Theorem 4.2 which grows with ``T``.

    Because the dynamics are input-affine, the known forcing ``B u_t^raw`` shifts the
    predict *mean* but never enters this recursion: ``P_inf`` is unchanged by rain or
    irrigation. That is the concrete pay-off of rejecting the action-conditioned
    ``A(a_t)`` form (Remark 2.1) -- with a state-dependent operator the covariance
    recursion would itself become input-dependent and no single ``P_inf`` would exist.

    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param sigma_eps2: measurement-noise variance.
    :param phi: ``(N, r)`` observation matrix; defaults to ``I_r`` (the visual case,
                where the encoder is the measurement map -- Section 2.3).
    :param max_iter: maximum Riccati iterations.
    :param tol: convergence tolerance on ``||P_{k+1} - P_k||_F``.
    :return: ``(r, r)`` steady-state covariance ``P_inf``.
    """
    r = a.shape[0]
    if phi is None:
        phi = jnp.eye(r)
    n = phi.shape[0]

    p = jnp.eye(r)
    for _ in range(max_iter):
        s = phi @ p @ phi.T + sigma_eps2 * jnp.eye(n)  # (N, N) innovation cov
        gain = a @ p @ phi.T @ jnp.linalg.inv(s)  # (r, N)
        p_next = a @ p @ a.T + q - gain @ phi @ p @ a.T
        p_next = (p_next + p_next.T) / 2.0  # keep symmetric against drift
        if float(jnp.linalg.norm(p_next - p)) < tol:
            p = p_next
            break
        p = p_next
    return p


def steady_state_error(
    a: jnp.ndarray,
    q: jnp.ndarray,
    sigma_eps2: float,
    phi: Optional[jnp.ndarray] = None,
) -> Dict[str, object]:
    """
    Theorem 4.3 summary: the horizon-independent steady-state estimation error.

    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param sigma_eps2: measurement-noise variance.
    :param phi: ``(N, r)`` observation matrix (default ``I_r``).
    :return: dict with ``P_inf``, its trace (total state-estimation variance) and
             the implied RMS state error ``sqrt(tr(P_inf) / r)``.
    """
    p_inf = solve_dare(a, q, sigma_eps2, phi)
    r = a.shape[0]
    trace = float(jnp.trace(p_inf))
    return {
        "P_inf": p_inf,
        "trace": trace,
        "rms_state_error": float(jnp.sqrt(jnp.maximum(trace, 0.0) / r)),
        "max_eigenvalue": float(jnp.max(jnp.linalg.eigvalsh((p_inf + p_inf.T) / 2.0))),
    }
