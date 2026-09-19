"""
Weather emission ``y_t^w = C w_t + d(doy_t) + nu_t`` (the rank-``m`` sensor).

The NDVI raster is a rank-``r`` measurement of the latent state, but it exists only
on dates with a usable frame. The weather record is daily-complete, so an emission
matrix ``C in R^{m x r}`` fitted from the training pairs ``(w_t, y_t^w)`` turns
Rs/Ta/VPD into a **second, always-available sensor**::

    C_ndvi = [I_r  0 ... 0]   rank r, gappy    (the NDVI frame)
    C_w    = [C    0 ... 0]   rank m, daily    (Rs / Ta / VPD)

This is what makes the requested "correct the prediction at every step from t+1 to
t+6" possible at all: during a forecast no NDVI frame is available by construction
(it is the held-out truth), so without a second sensor the covariance would simply
grow monotonically per Theorem 4.2.

On the direction of the regression
----------------------------------
``C`` regresses *weather on state*, which reads backwards causally -- the weather
drives the vegetation, not the reverse. That is deliberate and correct. A Kalman
update is a conditioning operation on a joint Gaussian, not a causal claim: it uses
the fact that ``w_t`` and ``y_t^w`` are *correlated* to sharpen the belief about
``w_t`` given ``y_t^w``. The causal direction is already represented, separately, by
the precipitation input ``B_p``.

The honesty check that matters
------------------------------
A rank-``m`` update is only worth something if ``C w_t`` genuinely explains part of
``y_t^w``. :func:`fit_emission` therefore reports **per-channel out-of-sample R^2**
and the resulting reduction in state uncertainty. If ``R^2`` is near zero the
weather sensor carries no information about the state, the update is a no-op, and
that must be reported rather than hidden behind an impressive-looking pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.emission")


@dataclass
class EmissionModel:
    """
    The fitted weather sensor.

    :ivar c: ``(m, r)`` emission matrix acting on the current weight block.
    :ivar r_cov: ``(m, m)`` measurement-noise covariance from the fit residuals.
    :ivar names: length-``m`` channel names (row order of ``c`` and ``r_cov``).
    :ivar r2: ``(m,)`` in-sample coefficient of determination per channel.
    :ivar r2_holdout: ``(m,)`` held-out ``R^2``, or ``None`` if not evaluated.
    :ivar scale: ``(m,)`` divisor that took the anomalies to standardised units.
    """

    c: np.ndarray
    r_cov: np.ndarray
    names: List[str]
    r2: np.ndarray
    r2_holdout: Optional[np.ndarray] = None
    #: Whether the sensor carries enough information to be worth assimilating.
    #: ``False`` means every channel failed on held-out data, so the update would
    #: inject error rather than remove it -- see :func:`emission_is_usable`.
    usable: bool = True
    scale: Optional[np.ndarray] = None

    @property
    def n_measurements(self) -> int:
        """Measurement dimension ``m``."""
        return self.c.shape[0]

    def predict(self, w: np.ndarray) -> np.ndarray:
        """
        Predicted standardised anomaly ``C w`` for one or more states.

        :param w: ``(r,)`` or ``(n, r)`` weights.
        :return: ``(m,)`` or ``(n, m)`` predicted anomalies.
        """
        return np.asarray(w) @ self.c.T

    def innovation(self, w_pred: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """
        Innovation ``nu = y^w - C w_pred`` in standardised anomaly units.

        The climatology offset ``d(doy)`` has already been removed by
        :mod:`dbwm.data.weather`, so no seasonal term appears here -- which is
        exactly the point: the innovation carries the weather *departure*, not the
        seasonal level that the autonomous operator already accounts for.

        :param w_pred: ``(r,)`` predicted weights.
        :param y_obs: ``(m,)`` observed standardised anomalies.
        :return: ``(m,)`` innovation.
        """
        return np.asarray(y_obs) - self.predict(w_pred)


def fit_emission(
    weights: np.ndarray,
    measurements: np.ndarray,
    names: Sequence[str],
    ridge: float = 1e-3,
    full_covariance: bool = True,
    holdout_fraction: float = 0.15,
    min_holdout_r2: float = 0.0,
    scale: Optional[np.ndarray] = None,
) -> EmissionModel:
    """
    Least-squares fit of the weather emission ``C`` and its noise covariance ``R``.

    ``C = Y W^T (W W^T + ridge I_r)^{-1}`` with ``W`` the training weights and ``Y``
    the standardised weather anomalies.

    ``R`` is estimated from the residuals as a **full** ``m x m`` covariance by
    default. Rs, Ta and VPD residuals are strongly cross-correlated -- a hot clear
    day is simultaneously bright, warm and dry -- so a diagonal ``R`` misstates the
    noise model and yields a gain that is not the minimum-variance one, together
    with a posterior covariance that does not match the actual error.

    Note that the *direction* of that error is not fixed. It is tempting to assume
    a diagonal ``R`` always over-weights the update by counting one physical
    fluctuation ``m`` times, but the opposite occurs under strong positive
    correlation: differencing two highly correlated channels cancels the shared
    disturbance and exposes the state, so the correct full-``R`` filter can extract
    *more* information than the diagonal approximation suggests is available. The
    reason to fit the full matrix is simply that it is the true one, not that it
    errs in a convenient direction.

    :param weights: ``(n, r)`` training weight trajectory.
    :param measurements: ``(n, m)`` standardised weather anomalies.
    :param names: length-``m`` channel names.
    :param ridge: ridge parameter.
    :param full_covariance: fit a full ``R`` (``False`` for a diagonal one).
    :param min_holdout_r2: a channel counts as informative only above this
        held-out ``R^2``. If no channel clears it the emission is marked unusable
        and the observer drops the weather update entirely. 0.0 is the weakest
        defensible bar: "better than predicting the channel's own mean".
    :param holdout_fraction: chronological tail fraction used to report an honest
        out-of-sample ``R^2``. Set to 0 to skip.
    :param scale: ``(m,)`` standardisation divisors, carried for reporting.
    :return: the fitted :class:`EmissionModel`.
    """
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(measurements, dtype=np.float64)
    if w.shape[0] != y.shape[0]:
        raise ValueError(
            "weights has {} rows but measurements has {}".format(w.shape[0], y.shape[0])
        )
    n, r = w.shape
    m = y.shape[1]

    n_val = int(round(holdout_fraction * n)) if holdout_fraction > 0 else 0
    n_fit = n - n_val
    if n_fit < r // 4 or n_fit < 8:
        logger.warning(
            "Only %d samples to fit a %d x %d emission matrix; C is weakly "
            "determined and the ridge is doing most of the work.", n_fit, m, r
        )

    def _solve(wx: np.ndarray, yx: np.ndarray) -> np.ndarray:
        gram = wx.T @ wx + ridge * np.eye(r)
        return np.linalg.solve(gram, wx.T @ yx).T  # (m, r)

    c_hat = _solve(w[:n_fit], y[:n_fit]) if n_val else _solve(w, y)

    r2_hold = None
    if n_val:
        r2_hold = _r2(y[n_fit:], w[n_fit:] @ c_hat.T)
        c_hat = _solve(w, y)  # refit on everything once the holdout has been scored

    pred = w @ c_hat.T
    resid = y - pred
    r2 = _r2(y, pred)

    cov = (resid.T @ resid) / max(n - 1, 1)
    if not full_covariance:
        cov = np.diag(np.diag(cov))
    cov = cov + 1e-8 * max(np.trace(cov) / m, 1e-12) * np.eye(m)

    # Gate on HELD-OUT skill, not in-sample fit. The two diverge dramatically
    # here: C maps a ~150-dimensional latent state onto 3 weather channels from
    # ~1200 dates, so it can interpolate the training weather (in-sample R^2 up
    # to +0.64) while extrapolating catastrophically off it (holdout R^2 down to
    # -948). A Kalman update built from that model does not merely fail to help;
    # it drives the state with a systematically wrong innovation at every one of
    # the ~1581 filter steps, which shows up as a bias in the forecast maps.
    usable = bool(r2_hold is not None and np.any(r2_hold > min_holdout_r2))
    if r2_hold is None:
        usable = True
    model = EmissionModel(
        c=c_hat,
        r_cov=cov,
        names=list(names),
        r2=r2,
        r2_holdout=r2_hold,
        scale=None if scale is None else np.asarray(scale),
        usable=usable,
    )
    _log_emission(model)
    return model


def _r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Per-channel coefficient of determination.

    :param y: ``(n, m)`` targets.
    :param pred: ``(n, m)`` predictions.
    :return: ``(m,)`` R^2 values.
    """
    ss_res = np.sum((y - pred) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


def _log_emission(model: EmissionModel) -> None:
    """
    Report the emission fit, flagging the case where the sensor is uninformative.

    :param model: the fitted model.
    """
    corr = _correlation(model.r_cov)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    logger.info("Weather emission C: %d x %d", *model.c.shape)
    for j, nm in enumerate(model.names):
        hold = "" if model.r2_holdout is None else " | holdout R2 {:+.3f}".format(
            model.r2_holdout[j]
        )
        logger.info("  %-14s in-sample R2 %+.3f%s", nm, float(model.r2[j]), hold)
    logger.info(
        "  R off-diagonal correlations: min %+.2f max %+.2f "
        "(a diagonal R would ignore these and give a non-optimal gain).",
        float(off.min()) if off.size else 0.0,
        float(off.max()) if off.size else 0.0,
    )
    ref = model.r2 if model.r2_holdout is None else model.r2_holdout
    if not model.usable:
        worst = float(np.min(ref))
        logger.warning(
            "Weather emission DISABLED: no channel has positive held-out R^2 "
            "(worst %+.1f). C was fitted in-sample on a %d-dimensional state from "
            "~1200 dates, so it interpolates the training weather and then "
            "extrapolates violently on the test split. Assimilating it would apply "
            "a Kalman update built from a garbage innovation at EVERY filter step, "
            "which biases the state the forecast starts from -- a systematic "
            "offset in the maps, not just extra noise. The rank-%d sensor is "
            "reported and dropped rather than silently trusted.",
            worst, model.c.shape[1], model.n_measurements,
        )
    elif np.all(ref < 0.05):
        logger.warning(
            "Every weather channel has R2 < 0.05 against the latent state. The "
            "rank-%d weather update carries essentially NO information about the "
            "NDVI state, so it will not meaningfully correct the t+1..t+6 forecast. "
            "Report this rather than relying on the update.",
            model.n_measurements,
        )


def _correlation(cov: np.ndarray) -> np.ndarray:
    """
    Correlation matrix of a covariance.

    :param cov: ``(m, m)`` covariance.
    :return: ``(m, m)`` correlation.
    """
    d = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return cov / np.outer(d, d)


def information_gain(
    model: EmissionModel, p_prior: np.ndarray
) -> Dict[str, float]:
    """
    How much does one weather update actually shrink the state uncertainty?

    Computes the posterior ``P+ = P - P C^T (C P C^T + R)^{-1} C P`` and reports the
    trace reduction. This is the quantitative answer to "is a rank-``m`` daily sensor
    worth anything against an ``r``-dimensional state?" -- it can correct at most
    ``m`` directions per step, so the honest expectation is a small per-step gain
    that accumulates over a multi-day gap, not a substitute for an NDVI frame.

    :param model: the fitted emission.
    :param p_prior: ``(r, r)`` prior state covariance.
    :return: dict with ``trace_prior``, ``trace_posterior``, ``trace_reduction`` and
             ``fraction_removed``.
    """
    p = np.asarray(p_prior, dtype=np.float64)
    c = model.c
    s = c @ p @ c.T + model.r_cov
    gain = np.linalg.solve(s, c @ p).T  # (r, m)
    p_post = p - gain @ c @ p
    tr0, tr1 = float(np.trace(p)), float(np.trace(p_post))
    return {
        "trace_prior": tr0,
        "trace_posterior": tr1,
        "trace_reduction": tr0 - tr1,
        "fraction_removed": (tr0 - tr1) / max(tr0, 1e-30),
    }
