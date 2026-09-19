"""
The pivotal pretest: is the ``L = 1`` innovation sequence white? (v3 Sec. 2.2.0)

Corollary 2.5.1 makes the memory extension **falsifiable**. Fit the v2 model,
compute ``eta_t = w_{t+1} - A w_t - B_p p_t``, and test ``{eta_t}`` for residual
autocorrelation:

* **Rejection** => Proposition 2.5 bites. The Mori-Zwanzig memory kernel is not
  identically zero, the v2 Kalman filter is not minimum-variance for this process,
  and ``Q`` is absorbing coloured memory into a white covariance -- which is exactly
  what produces miscalibrated intervals. The memory extension is then *empirically
  motivated* rather than postulated.
* **Non-rejection** => ``L = 1`` suffices and the whole lift is pure variance
  inflation.

v3 calls this "the pivotal experiment of the paper" and instructs that it be run
before anything else and **reported either way**. :func:`whiteness_pretest` is
therefore wired as a gate in the estimation pipeline, not as an optional extra.

Why not a textbook multivariate portmanteau
--------------------------------------------
Hosking's multivariate Ljung-Box on an ``r``-dimensional residual has ``r^2 K``
degrees of freedom -- at ``r = 256`` and ``K = 12`` that is 786,432 against ``T ~
1200`` samples. The statistic is undefined long before it is unreliable, since
``C_0`` is singular. Two usable routes are provided instead:

1. **Per-component** Ljung-Box across the ``r`` latent coordinates, with a
   Benjamini-Hochberg FDR correction. Interpretable: "37% of latent components show
   significant residual autocorrelation at FDR 5%".
2. **Random-projection** Hosking statistic on ``d << r`` projected dimensions
   (``d^2 K`` dof), which retains the cross-component structure the per-component
   test discards.

Both are reported; either rejecting is grounds to lift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.diagnostics")


def _chi2_sf(x: float, df: int) -> float:
    """
    Upper-tail probability of a chi-square distribution.

    :param x: statistic value.
    :param df: degrees of freedom.
    :return: ``P(X > x)``.
    """
    from scipy import stats

    return float(stats.chi2.sf(x, df))


def one_step_residuals(
    weights: np.ndarray,
    a: np.ndarray,
    b_p: Optional[np.ndarray] = None,
    forcing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    The v2 innovation sequence ``eta_t = w_{t+1} - A w_t - B_p p_t``.

    :param weights: ``(T, r)`` weight trajectory.
    :param a: ``(r, r)`` one-step operator.
    :param b_p: ``(r, ell)`` input matrix or ``None``.
    :param forcing: ``(T, ell)`` inputs or ``None``.
    :return: ``(T-1, r)`` residuals.
    """
    w = np.asarray(weights, dtype=np.float64)
    pred = w[:-1] @ np.asarray(a).T
    if b_p is not None and forcing is not None and np.asarray(b_p).shape[1]:
        pred = pred + np.asarray(forcing, dtype=np.float64)[:-1] @ np.asarray(b_p).T
    return w[1:] - pred


def ljung_box_univariate(x: np.ndarray, lags: int = 12) -> Dict[str, float]:
    """
    Univariate Ljung-Box test on one residual series.

    ``Q = T(T+2) sum_{k=1}^{K} rho_k^2 / (T-k)``, asymptotically ``chi2(K)``.

    :param x: ``(T,)`` residual series.
    :param lags: number of lags ``K``.
    :return: dict with ``statistic``, ``dof`` and ``p_value``.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x.size
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 1e-30:
        return {"statistic": 0.0, "dof": lags, "p_value": 1.0}
    q = 0.0
    for k in range(1, lags + 1):
        rho = float(xc[k:] @ xc[:-k]) / denom
        q += rho**2 / (t - k)
    q *= t * (t + 2)
    return {"statistic": q, "dof": lags, "p_value": _chi2_sf(q, lags)}


def ljung_box_hosking(resid: np.ndarray, lags: int = 12) -> Dict[str, float]:
    """
    Hosking's multivariate portmanteau statistic.

    ``Q = T^2 sum_k (T-k)^{-1} tr(C_k' C_0^{-1} C_k C_0^{-1})``, asymptotically
    ``chi2(d^2 K)``. Only usable for small ``d`` -- apply it to a random projection
    of the residuals, never to the raw ``r``-dimensional series.

    :param resid: ``(T, d)`` residuals with ``d`` small.
    :param lags: number of lags ``K``.
    :return: dict with ``statistic``, ``dof`` and ``p_value``.
    """
    e = np.asarray(resid, dtype=np.float64)
    e = e - e.mean(axis=0, keepdims=True)
    t, d = e.shape
    if t <= lags + d:
        raise ValueError(
            "Need T > K + d for the Hosking statistic; got T={}, K={}, d={}".format(
                t, lags, d
            )
        )
    c0 = (e.T @ e) / t
    c0_inv = np.linalg.pinv(c0)
    q = 0.0
    for k in range(1, lags + 1):
        ck = (e[k:].T @ e[:-k]) / t
        q += np.trace(ck.T @ c0_inv @ ck @ c0_inv) / (t - k)
    q *= t**2
    dof = d * d * lags
    return {"statistic": float(q), "dof": dof, "p_value": _chi2_sf(float(q), dof)}


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up FDR control.

    :param p_values: raw p-values.
    :param alpha: target false-discovery rate.
    :return: boolean array marking rejected hypotheses.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        k = int(np.nonzero(passed)[0].max())
        out[order[: k + 1]] = True
    return out


def whiteness_pretest(
    residuals: np.ndarray,
    lags: int = 12,
    alpha: float = 0.05,
    projection_dim: int = 16,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Run the v3 step-0 pretest and return a verdict.

    :param residuals: ``(T, r)`` one-step innovations from the ``L = 1`` fit.
    :param lags: Ljung-Box lags ``K``.
    :param alpha: significance level / target FDR.
    :param projection_dim: dimension ``d`` of the random projection used for the
        Hosking statistic. Must satisfy ``d^2 K < T`` to be meaningful; it is
        reduced automatically if not.
    :param seed: RNG seed for the projection.
    :return: dict with ``reject``, ``verdict``, ``fraction_significant``,
             ``hosking`` and ``per_component``.
    """
    e = np.asarray(residuals, dtype=np.float64)
    t, r = e.shape

    per = [ljung_box_univariate(e[:, j], lags) for j in range(r)]
    p_raw = np.array([d["p_value"] for d in per])
    rejected = benjamini_hochberg(p_raw, alpha)
    frac = float(rejected.mean())

    # Keep d^2 * K comfortably below T or the chi-square approximation is useless.
    d = int(projection_dim)
    while d > 1 and d * d * lags > max(t // 4, 1):
        d -= 1
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(r, d)) / np.sqrt(r)
    hosking = ljung_box_hosking(e @ proj, lags)
    if d < projection_dim:
        logger.info(
            "Hosking projection reduced from d=%d to d=%d so that d^2 K = %d stays "
            "well below T = %d.", projection_dim, d, d * d * lags, t,
        )

    reject = bool(hosking["p_value"] < alpha or frac > alpha)
    verdict = (
        "REJECT whiteness -- Prop. 2.5 bites: the memory kernel is not zero, so the "
        "L=1 Kalman filter is not minimum-variance and Q is absorbing coloured "
        "memory into a white covariance. The memory lift is EMPIRICALLY MOTIVATED."
        if reject
        else "FAIL TO REJECT whiteness -- L=1 suffices for this record and the "
        "memory lift would be pure variance inflation. Report this and stop."
    )
    logger.info(
        "Whiteness pretest (K=%d lags, alpha=%.3f): Hosking Q=%.1f on %d dof, "
        "p=%.3e; %.1f%% of %d latent components reject at FDR %.2f.",
        lags, alpha, hosking["statistic"], hosking["dof"], hosking["p_value"],
        100.0 * frac, r, alpha,
    )
    logger.info("Verdict: %s", verdict)
    return {
        "reject": reject,
        "verdict": verdict,
        "fraction_significant": frac,
        "n_components": r,
        "n_samples": t,
        "hosking": hosking,
        "per_component_p": p_raw,
        "per_component_rejected": rejected,
        "projection_dim": d,
        "lags": lags,
        "alpha": alpha,
    }


def residual_autocorrelation(
    residuals: np.ndarray, lags: int = 12
) -> Dict[str, np.ndarray]:
    """
    Mean absolute residual autocorrelation per lag, for plotting.

    A white sequence sits inside the ``+-1.96/sqrt(T)`` band at every lag; coloured
    memory shows as a decaying profile whose extent *is* the Mori-Zwanzig depth the
    ``||D_h||``-vs-``L`` sweep measures independently. The two should agree, and
    disagreement is itself informative.

    :param residuals: ``(T, r)`` residuals.
    :param lags: maximum lag.
    :return: dict with ``lags``, ``mean_abs_acf`` and the ``band`` half-width.
    """
    e = np.asarray(residuals, dtype=np.float64)
    e = e - e.mean(axis=0, keepdims=True)
    t = e.shape[0]
    denom = np.maximum(np.sum(e**2, axis=0), 1e-30)
    out = np.zeros(lags)
    for k in range(1, lags + 1):
        rho = np.sum(e[k:] * e[:-k], axis=0) / denom
        out[k - 1] = float(np.mean(np.abs(rho)))
    return {
        "lags": np.arange(1, lags + 1),
        "mean_abs_acf": out,
        "band": 1.96 / np.sqrt(t),
    }
