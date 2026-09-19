"""
Memory-lifted, two-sensor Kalman observer (v3 Sec. 2.2.6).

The filter runs on the lifted state ``w_bar_t = [w_t; ...; w_{t-L+1}] in R^{Lr}``
with the block-companion ``A_cal``::

    w_bar_{t|t-1} = A_cal w_bar_{t-1|t-1} + B_cal p_{t-1}
    P_{t|t-1}     = A_cal P_{t-1|t-1} A_cal^T + E Q E^T + Gamma_dyn

Two heterogeneous sensors, applied sequentially
------------------------------------------------
================  ==========================  =======  ==============
Sensor            Emission                    Rank     Availability
================  ==========================  =======  ==============
NDVI frame        ``C_ndvi = [I_r 0 ... 0]``  ``r``    gappy
Rs / Ta / VPD     ``C_w = [C 0 ... 0]``       ``m``    every day
================  ==========================  =======  ==============

They are applied as two sequential updates, which is exact because their noises
are independent. The NDVI innovation covariance uses the **per-date GP posterior**
``Sigma_{w_t} = sigma_eps^2 Lambda_{X_t}^{-1}`` rather than a scalar
``sigma_eps^2 I``: the number of valid pixels varies by date, so the confidence in
``w_t^obs`` genuinely varies too, and a scalar would throw that away.

Why the weather sensor is what makes the request possible
----------------------------------------------------------
During a forecast no NDVI frame exists -- it is the held-out truth. Without a
second sensor the covariance would grow monotonically (v2 Thm 4.2, and at
``rho_max = 1`` it grows *linearly and without bound*). The daily rank-``m``
weather sensor is what allows a genuine Bayesian correction at every step from
``t+1`` to ``t+6``.

Two forecast modes, and why both are reported
----------------------------------------------
v3 Sec. 2.2.6 states that *filtering must remain one-step*: by Theorem 2.10 only
``Theta_1`` is compatible with a recursive Bayesian update, while *forecasting*
should use the direct family. Applying a weather update at each of six steps and
using the direct family are therefore two different requests, and this module
implements both rather than silently picking one:

``recursive``
    Roll the one-step lifted filter forward through ``t+1..t+6``, applying the
    weather update at each step. Information **accumulates**: by ``t+6`` the state
    has absorbed six weather updates. Theoretically clean -- every step is a proper
    predict/update pair on ``(A_cal, C_w)``.

``direct``
    Predict ``w_{t+h}`` from the origin with ``Theta_h`` and ``Sigma_h``, then apply
    a single weather update at ``t+h``. Better calibrated (Prop. 2.13) but the
    corrections do not compound across the horizon.

The comparison is itself a result: it measures whether six accumulated rank-``m``
corrections outweigh the calibration advantage of the direct family.

Numerics
--------
Covariance updates use the symmetric form ``P <- P - K S K^T``, which is
algebraically identical to ``P - K C P`` but symmetric by construction, so the
covariance cannot drift asymmetric over ~1600 steps. The companion structure is
exploited in the predict step: the lower block rows of ``A_cal P`` are a *slice* of
``P``, so the cost is ``O(L^2 r^3)`` rather than ``O(L^3 r^3)``.

This module is ``numpy``-based rather than ``jax``: it is closed-form, host-side,
runs once, and needs data-dependent branching for missing observations, which
``lax.scan`` would obscure for no gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.dynamics.memory import (
    MemoryOperator,
    lifted_input_matrix,
    noise_injection_matrix,
    selector_matrix,
)
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.observer")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class LiftedSystem:
    """
    Everything the observer needs, assembled once.

    :ivar op: the identified memory operator (supplies ``A_cal``).
    :ivar b_p: ``(r, ell)`` input matrix.
    :ivar q: ``(r, r)`` process-noise covariance.
    :ivar c_w: ``(m, r)`` weather emission, or ``None`` for NDVI-only filtering.
    :ivar r_w: ``(m, m)`` weather measurement-noise covariance.
    :ivar gamma_dyn: scalar dynamics-residual inflation factor.
    """

    op: MemoryOperator
    b_p: np.ndarray
    q: np.ndarray
    c_w: Optional[np.ndarray] = None
    r_w: Optional[np.ndarray] = None
    gamma_dyn: float = 0.0

    def __post_init__(self):
        self._a_row = self.op.blocks.transpose(1, 0, 2).reshape(
            self.op.r, self.op.lifted_dim
        )
        self._b_cal = lifted_input_matrix(self.op.order, np.asarray(self.b_p))
        self._e = noise_injection_matrix(self.op.order, self.op.r)
        self._s = selector_matrix(self.op.order, self.op.r)

    @property
    def r(self) -> int:
        """Latent dimension."""
        return self.op.r

    @property
    def order(self) -> int:
        """Memory order ``L``."""
        return self.op.order

    @property
    def lifted_dim(self) -> int:
        """``L * r``."""
        return self.op.lifted_dim

    def apply_a(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ``A_cal`` to a lifted vector, exploiting the companion structure.

        :param x: ``(Lr,)`` lifted state.
        :return: ``(Lr,)`` propagated state.
        """
        r = self.r
        out = np.empty_like(x)
        out[:r] = self._a_row @ x
        out[r:] = x[: -r]
        return out

    def process_noise(self) -> np.ndarray:
        """
        The lifted process noise ``E Q E^T + Gamma_dyn``.

        Singular by construction (rank ``r`` of ``Lr``), which is fine: Lemma 2.7
        guarantees ``(A_cal, E)`` is controllable for every memory kernel, so the
        stabilizability hypothesis of v2 Thm 4.3 still holds.

        :return: ``(Lr, Lr)`` covariance.
        """
        n = self.lifted_dim
        out = np.zeros((n, n))
        out[: self.r, : self.r] = self.q
        if self.gamma_dyn:
            out[: self.r, : self.r] += (
                self.gamma_dyn * np.trace(self.q) / self.r * np.eye(self.r)
            )
        return out


@dataclass
class FilterResult:
    """
    Output of a filtering pass.

    :ivar states: ``(T, Lr)`` filtered lifted states ``w_bar_{t|t}``.
    :ivar covariances: ``(T, Lr, Lr)`` filtered covariances, or ``None`` if not kept.
    :ivar priors: ``(T, r)`` one-step-ahead prior means ``w_{t|t-1}`` -- these are
        the quantities scored as one-step forecasts.
    :ivar prior_covariances: ``(T, r, r)`` current-block prior covariances.
    :ivar innovations: per-sensor innovation records, for the whiteness diagnostics.
    :ivar n_ndvi_updates: how many NDVI updates actually fired.
    :ivar n_weather_updates: how many weather updates fired.
    """

    states: np.ndarray
    priors: np.ndarray
    prior_covariances: np.ndarray
    covariances: Optional[np.ndarray] = None
    innovations: Dict[str, np.ndarray] = field(default_factory=dict)
    n_ndvi_updates: int = 0
    n_weather_updates: int = 0

    def current(self) -> np.ndarray:
        """The current-block weights ``w_{t|t}`` (the top block of each state)."""
        r = self.priors.shape[1]
        return self.states[:, :r]

    def smoothed_lag(self, lag: int) -> np.ndarray:
        """
        Fixed-lag smoothed estimates ``w_{t-lag|t}`` (v3 Prop. 2.9).

        The lower blocks of the lifted filter are **not** bookkeeping: they hold
        retro-corrected past weights, updated using present observations. That is
        exactly the mechanism that fills revisit and cloud gaps.

        :param lag: ``j`` in ``w_{t-j|t}``; must be ``< L``.
        :return: ``(T, r)`` smoothed weights.
        """
        r = self.priors.shape[1]
        n_blocks = self.states.shape[1] // r
        if not 0 <= lag < n_blocks:
            raise ValueError("lag must be in [0, {}), got {}".format(n_blocks, lag))
        return self.states[:, lag * r : (lag + 1) * r]


# --------------------------------------------------------------------------- #
# Primitive steps
# --------------------------------------------------------------------------- #
def predict(
    sys: LiftedSystem,
    state: np.ndarray,
    cov: np.ndarray,
    forcing: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lifted PREDICT step.

    Exploits the companion structure: the lower block rows of ``A_cal P`` are a
    slice of ``P``, so only the top block row costs anything.

    :param sys: the assembled system.
    :param state: ``(Lr,)`` posterior mean.
    :param cov: ``(Lr, Lr)`` posterior covariance.
    :param forcing: ``(ell,)`` input ``p_{t-1}`` or ``None``.
    :return: ``(state_pred, cov_pred)``.
    """
    r = sys.r
    new_state = sys.apply_a(state)
    if forcing is not None and sys._b_cal.shape[1]:
        new_state = new_state + sys._b_cal @ np.asarray(forcing)

    # A_cal @ cov, using the shift structure for the lower blocks.
    ap = np.empty_like(cov)
    ap[:r] = sys._a_row @ cov
    ap[r:] = cov[:-r]
    # (A_cal cov) @ A_cal^T, same trick on the right.
    out = np.empty_like(cov)
    out[:, :r] = ap @ sys._a_row.T
    out[:, r:] = ap[:, :-r]
    out = out + sys.process_noise()
    return new_state, 0.5 * (out + out.T)


def update(
    state: np.ndarray,
    cov: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic Kalman UPDATE for an emission acting on the **current block**.

    Uses the symmetric covariance form ``P <- P - K S K^T``. It is algebraically
    identical to ``P - K C P`` but symmetric by construction, so the covariance
    cannot drift asymmetric over the ~1,600 steps of the record.

    :param state: ``(Lr,)`` prior mean.
    :param cov: ``(Lr, Lr)`` prior covariance.
    :param c: ``(k, r)`` emission on the current block.
    :param y: ``(k,)`` observation.
    :param noise: ``(k, k)`` measurement-noise covariance.
    :return: ``(state_post, cov_post, innovation)``.
    """
    c = np.atleast_2d(c)
    r = c.shape[1]
    p_top = cov[:, :r]  # (Lr, r)
    cp = c @ p_top[:r]  # (k, r) = C P_11
    s = cp @ c.T + noise  # (k, k)
    gain = np.linalg.solve(s, (p_top @ c.T).T).T  # (Lr, k)
    innovation = np.asarray(y) - c @ state[:r]
    new_state = state + gain @ innovation
    new_cov = cov - gain @ s @ gain.T
    return new_state, 0.5 * (new_cov + new_cov.T), innovation


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def filter_sequence(
    sys: LiftedSystem,
    w_obs: np.ndarray,
    observed: np.ndarray,
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    obs_cov: Optional[np.ndarray] = None,
    sigma_eps2: float = 1e-2,
    keep_covariances: bool = False,
    initial_cov_scale: float = 1.0,
) -> FilterResult:
    """
    Run the two-sensor lifted filter over a record, tolerating missing frames.

    :param sys: the assembled :class:`LiftedSystem`.
    :param w_obs: ``(T, r)`` GP-solved weights per date (rows where ``observed`` is
        ``False`` are ignored and may be anything).
    :param observed: ``(T,)`` bool -- ``True`` where an NDVI frame exists.
    :param forcing: ``(T, ell)`` precipitation rows (forward-window convention:
        row ``t`` drives ``t -> t+1``, so the predict at ``t`` consumes row ``t-1``).
    :param weather: ``(T, m)`` standardised Rs/Ta/VPD anomalies, or ``None``.
    :param obs_cov: ``(T, r, r)`` per-date GP posterior covariances
        ``sigma_eps^2 Lambda_{X_t}^{-1}``, or ``None`` to use ``sigma_eps2 I``.
    :param sigma_eps2: fallback scalar observation noise.
    :param keep_covariances: retain the full ``(T, Lr, Lr)`` covariance stack
        (memory-hungry: ``1581 x 1792^2`` floats is ~40 GB, so leave this off for
        the full record and use :func:`forecast_from_origin` instead).
    :param initial_cov_scale: multiplier on the initial covariance.
    :return: the :class:`FilterResult`.
    """
    w_obs = np.asarray(w_obs, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    t_total, r = w_obs.shape
    n = sys.lifted_dim

    state = np.zeros(n)
    first = int(np.argmax(observed)) if observed.any() else 0
    for j in range(sys.order):
        state[j * r : (j + 1) * r] = w_obs[first]
    cov = initial_cov_scale * np.eye(n) * max(sigma_eps2, 1e-8)

    states = np.zeros((t_total, n))
    priors = np.zeros((t_total, r))
    prior_covs = np.zeros((t_total, r, r))
    covs = np.zeros((t_total, n, n)) if keep_covariances else None
    innov_ndvi: List[np.ndarray] = []
    innov_w: List[np.ndarray] = []
    n_ndvi = n_weather = 0

    for t in range(t_total):
        if t > 0:
            u = None
            if forcing is not None and np.asarray(forcing).shape[1]:
                u = np.asarray(forcing)[t - 1]
            state, cov = predict(sys, state, cov, u)
        priors[t] = state[:r]
        prior_covs[t] = cov[:r, :r]

        # --- Sensor 1: weather (daily, rank m) ---
        if weather is not None and sys.c_w is not None:
            y_w = np.asarray(weather)[t]
            if np.all(np.isfinite(y_w)):
                state, cov, nu = update(state, cov, sys.c_w, y_w, sys.r_w)
                innov_w.append(nu)
                n_weather += 1

        # --- Sensor 2: NDVI frame (gappy, rank r) ---
        if observed[t]:
            noise = (
                obs_cov[t]
                if obs_cov is not None
                else sigma_eps2 * np.eye(r)
            )
            state, cov, nu = update(state, cov, np.eye(r), w_obs[t], noise)
            innov_ndvi.append(nu)
            n_ndvi += 1

        states[t] = state
        if keep_covariances:
            covs[t] = cov

    logger.info(
        "Lifted filter: %d steps, %d NDVI updates (%.1f%% coverage), %d weather updates.",
        t_total, n_ndvi, 100.0 * n_ndvi / max(t_total, 1), n_weather,
    )
    return FilterResult(
        states=states,
        priors=priors,
        prior_covariances=prior_covs,
        covariances=covs,
        innovations={
            "ndvi": np.asarray(innov_ndvi) if innov_ndvi else np.zeros((0, r)),
            "weather": np.asarray(innov_w) if innov_w else np.zeros((0, 0)),
        },
        n_ndvi_updates=n_ndvi,
        n_weather_updates=n_weather,
    )


def filter_state_at(
    sys: LiftedSystem,
    w_obs: np.ndarray,
    observed: np.ndarray,
    upto: int,
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    obs_cov: Optional[np.ndarray] = None,
    sigma_eps2: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter up to and including step ``upto`` and return that state and covariance.

    Provided because retaining the full covariance stack for the whole record is
    infeasible (``1581 x 1792 x 1792`` floats is ~40 GB).

    :param sys: the assembled system.
    :param w_obs: ``(T, r)`` GP-solved weights.
    :param observed: ``(T,)`` NDVI availability.
    :param upto: last step to assimilate (inclusive).
    :param forcing: ``(T, ell)`` inputs.
    :param weather: ``(T, m)`` weather anomalies.
    :param obs_cov: ``(T, r, r)`` per-date GP posterior covariances.
    :param sigma_eps2: fallback observation noise.
    :return: ``(w_bar_{upto|upto}, P_{upto|upto})``.
    """
    res = filter_sequence(
        sys,
        w_obs[: upto + 1],
        observed[: upto + 1],
        None if forcing is None else forcing[: upto + 1],
        None if weather is None else weather[: upto + 1],
        None if obs_cov is None else obs_cov[: upto + 1],
        sigma_eps2,
        keep_covariances=True,
    )
    return res.states[-1], res.covariances[-1]


# --------------------------------------------------------------------------- #
# Forecasting from an origin, with a weather update at every horizon step
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    """
    An ``H``-step forecast from one origin, before and after the weather update.

    :ivar mean_prior: ``(H, r)`` predicted weights before the weather update.
    :ivar cov_prior: ``(H, r, r)`` predicted current-block covariances.
    :ivar mean: ``(H, r)`` weights after the rank-``m`` weather correction.
    :ivar cov: ``(H, r, r)`` covariances after the correction.
    :ivar innovations: ``(H, m)`` weather innovations, one per horizon step.
    :ivar mode: ``"recursive"`` or ``"direct"``.
    :ivar trace_reduction: ``(H,)`` trace removed by the weather update at each
        step -- the honest measure of what the rank-``m`` sensor bought.
    """

    mean_prior: np.ndarray
    cov_prior: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    innovations: np.ndarray
    mode: str
    trace_reduction: np.ndarray


def forecast_from_origin(
    sys: LiftedSystem,
    state: np.ndarray,
    cov: np.ndarray,
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    family=None,
    mode: str = "recursive",
) -> ForecastResult:
    """
    Forecast ``t+1..t+H`` with a weather update applied at **every** step.

    No NDVI frame is used at any horizon step -- those are the held-out truth. The
    only correction available is the daily rank-``m`` weather sensor, which is
    precisely why the emission matrix exists.

    ``mode="recursive"``
        Roll the one-step lifted filter forward, applying the weather update after
        each predict. Information **accumulates** across the horizon: by ``t+6`` the
        state has absorbed six corrections. This is the mode in which "apply the
        update at each predicted timestep" compounds, and every step is a proper
        predict/update pair on ``(A_cal, C_w)`` -- consistent with v3 Sec. 2.2.6's
        rule that recursive Bayesian updating must use ``Theta_1``.

    ``mode="direct"``
        Predict ``w_{t+h}`` from the origin with ``Theta_h`` and the directly
        estimated ``Sigma_h``, then apply one weather update at ``t+h``. Better
        calibrated (Prop. 2.13), but the corrections do not compound.

    Neither dominates a priori: the recursive mode accumulates six rank-``m``
    corrections while carrying the iterated predictor's bias ``b_h``; the direct
    mode is unbiased and calibrated but corrects once. Reporting both is what makes
    the trade-off measurable rather than assumed.

    :param sys: the assembled system.
    :param state: ``(Lr,)`` filtered state at the origin.
    :param cov: ``(Lr, Lr)`` filtered covariance at the origin.
    :param horizon: ``H``.
    :param forcing: ``(H, ell)`` inputs ``[p_t, ..., p_{t+H-1}]``.
    :param weather: ``(H, m)`` weather anomalies at ``t+1..t+H``, or ``None`` to
        skip the updates (giving the honest uncorrected baseline).
    :param family: a :class:`~dbwm.dynamics.multihorizon.HorizonFamily`, required
        for ``mode="direct"``.
    :param mode: ``"recursive"`` or ``"direct"``.
    :return: the :class:`ForecastResult`.
    """
    r = sys.r
    m = 0 if sys.c_w is None else sys.c_w.shape[0]
    mean_prior = np.zeros((horizon, r))
    cov_prior = np.zeros((horizon, r, r))
    mean_post = np.zeros((horizon, r))
    cov_post = np.zeros((horizon, r, r))
    innov = np.full((horizon, m), np.nan)
    reduction = np.zeros(horizon)

    if mode == "recursive":
        s, p = state.copy(), cov.copy()
        for h in range(horizon):
            u = None
            if forcing is not None and np.asarray(forcing).shape[1]:
                u = np.asarray(forcing)[h]
            s, p = predict(sys, s, p, u)
            mean_prior[h] = s[:r]
            cov_prior[h] = p[:r, :r]
            tr0 = np.trace(p[:r, :r])
            if weather is not None and sys.c_w is not None:
                y = np.asarray(weather)[h]
                if np.all(np.isfinite(y)):
                    s, p, nu = update(s, p, sys.c_w, y, sys.r_w)
                    innov[h] = nu
            mean_post[h] = s[:r]
            cov_post[h] = p[:r, :r]
            reduction[h] = tr0 - np.trace(p[:r, :r])
        return ForecastResult(
            mean_prior, cov_prior, mean_post, cov_post, innov, mode, reduction
        )

    if mode != "direct":
        raise ValueError("mode must be 'recursive' or 'direct', got {!r}".format(mode))
    if family is None:
        raise ValueError("mode='direct' requires a HorizonFamily")

    flat = state.reshape(-1)
    for h in range(horizon):
        mu = family.theta[h] @ flat
        if forcing is not None and np.asarray(forcing).shape[1]:
            for i in range(h + 1):
                mu = mu + family.inputs[h][i] @ np.asarray(forcing)[i]
        # Prop. 2.13: Sigma_h is the DIRECT residual covariance, never propagated Q.
        p_h = family.theta[h] @ cov @ family.theta[h].T + family.sigma[h]
        p_h = 0.5 * (p_h + p_h.T)
        mean_prior[h] = mu
        cov_prior[h] = p_h
        tr0 = np.trace(p_h)
        if weather is not None and sys.c_w is not None:
            y = np.asarray(weather)[h]
            if np.all(np.isfinite(y)):
                mu, p_h, nu = update(mu, p_h, sys.c_w, y, sys.r_w)
                innov[h] = nu
        mean_post[h] = mu
        cov_post[h] = p_h
        reduction[h] = tr0 - np.trace(p_h)
    return ForecastResult(
        mean_prior, cov_prior, mean_post, cov_post, innov, mode, reduction
    )


def forecast_with_assimilation(
    sys: LiftedSystem,
    state: np.ndarray,
    cov: np.ndarray,
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    w_obs: Optional[np.ndarray] = None,
    observed: Optional[np.ndarray] = None,
    obs_cov: Optional[np.ndarray] = None,
    sigma_eps2: float = 1e-2,
) -> Dict[str, np.ndarray]:
    """
    Forecast ``t+1..t+H``, folding in later frames as they arrive.

    This is the operational loop: a forecast is issued, and then over the
    following days some of its target frames actually turn up -- a clear
    acquisition on day ``t+2``, say. The pipeline should use them.

    At each step it predicts, applies the daily rank-``m`` weather update, and
    **then** assimilates that step's NDVI frame if one exists. The rank-``r`` NDVI
    update uses the per-date GP posterior ``sigma_eps^2 Lambda_{X_t}^{-1}``, so a
    frame with fewer valid pixels is trusted less, exactly as in the main filter.

    Two trajectories come back and the distinction is the whole point:

    ``mean_forecast``
        what was predicted for step ``h`` **before** that step's own frame was
        seen. This is the genuine out-of-sample forecast and it is the only one
        that may be scored as such.

    ``mean``
        the corrected state after assimilating everything available up to and
        including step ``h``. Later steps are launched from this, so a frame at
        ``t+2`` improves ``t+3..t+6``. It is a *filtered estimate*, not a
        forecast, and reporting it as forecast skill would be scoring a model on
        data it has already seen.

    :param sys: the assembled system.
    :param state: ``(Lr,)`` filtered state at the origin.
    :param cov: ``(Lr, Lr)`` filtered covariance at the origin.
    :param horizon: ``H``.
    :param forcing: ``(H, ell)`` inputs ``[p_t, ..., p_{t+H-1}]``.
    :param weather: ``(H, m)`` weather anomalies at ``t+1..t+H``; ``NaN`` rows are
        skipped.
    :param w_obs: ``(H, r)`` GP-solved weights at ``t+1..t+H``, where frames exist.
    :param observed: ``(H,)`` bool -- which horizon steps have a frame.
    :param obs_cov: ``(H, r, r)`` per-step GP posterior covariances.
    :param sigma_eps2: fallback scalar observation noise.
    :return: dict with ``mean_forecast``, ``cov_forecast``, ``mean``, ``cov``,
             ``assimilated`` ``(H,)`` bool and ``trace_reduction`` ``(H,)``.
    """
    r = sys.r
    s, p = np.asarray(state, dtype=np.float64).copy(), np.asarray(cov).copy()
    pre_mean = np.zeros((horizon, r))
    pre_cov = np.zeros((horizon, r, r))
    post_mean = np.zeros((horizon, r))
    post_cov = np.zeros((horizon, r, r))
    used = np.zeros(horizon, dtype=bool)
    reduction = np.zeros(horizon)

    for h in range(horizon):
        u = None
        if forcing is not None and np.asarray(forcing).shape[1]:
            u = np.asarray(forcing)[h]
        s, p = predict(sys, s, p, u)

        # Weather first: it is available every day and independent of the frame,
        # and sequential updates are exact for independent noises.
        if weather is not None and sys.c_w is not None:
            y = np.asarray(weather)[h]
            if np.all(np.isfinite(y)):
                s, p, _ = update(s, p, sys.c_w, y, sys.r_w)

        # The forecast for this step, recorded BEFORE its own frame is seen.
        pre_mean[h] = s[:r]
        pre_cov[h] = p[:r, :r]
        tr0 = float(np.trace(p[:r, :r]))

        if (
            w_obs is not None and observed is not None
            and bool(np.asarray(observed)[h])
        ):
            noise = (
                np.asarray(obs_cov)[h] if obs_cov is not None
                else sigma_eps2 * np.eye(r)
            )
            s, p, _ = update(s, p, np.eye(r), np.asarray(w_obs)[h], noise)
            used[h] = True
        post_mean[h] = s[:r]
        post_cov[h] = p[:r, :r]
        reduction[h] = tr0 - float(np.trace(p[:r, :r]))

    if used.any():
        logger.info(
            "Assimilated %d of %d horizon frames (steps %s); those steps' own "
            "forecasts were recorded before the update, so they stay scoreable.",
            int(used.sum()), horizon,
            [int(i) + 1 for i in np.nonzero(used)[0]],
        )
    return {
        "mean_forecast": pre_mean,
        "cov_forecast": pre_cov,
        "mean": post_mean,
        "cov": post_cov,
        "assimilated": used,
        "trace_reduction": reduction,
    }


def rolling_forecast_multi(
    sys: LiftedSystem,
    w_obs: np.ndarray,
    observed: np.ndarray,
    origins: Sequence[int],
    horizon: int,
    variants: Sequence[Dict[str, object]],
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    obs_cov: Optional[np.ndarray] = None,
    sigma_eps2: float = 1e-2,
    family=None,
    keep_cov: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate several forecast variants from a **single** filter pass.

    The filter trajectory does not depend on which forecast variant is being
    scored -- it assimilates the same data either way, and only the *branch* taken
    at each origin differs. Calling :func:`rolling_forecast` once per variant
    therefore re-runs the identical 1581-step filter N times, and at ``r = 256``
    that pass costs ~9 minutes. With the four variants the driver reports
    (recursive / direct, weather on / off) that is ~36 minutes of pure duplication.

    Here the filter runs once and, at each requested origin, every variant is
    branched from the same live state before the filter moves on. Only one
    ``(Lr, Lr)`` covariance is alive at a time, so the saving costs no memory.

    :param sys: the assembled system.
    :param w_obs: ``(T, r)`` GP-solved weights.
    :param observed: ``(T,)`` NDVI availability.
    :param origins: forecast origin indices.
    :param horizon: ``H``.
    :param variants: one dict per variant, e.g.
        ``{"name": "recursive", "mode": "recursive", "use_weather": True}``.
        ``use_weather`` toggles the weather update **inside the forecast branch
        only** -- the shared filter always assimilates whatever is available up to
        the origin. That is deliberate, and it is a different (better-controlled)
        comparison than calling :func:`rolling_forecast` with ``weather=None``,
        which additionally strips weather from the filter and so changes the origin
        state too. Toggling one thing at a time is what isolates the value of the
        ``t+1..t+6`` correction; the fully-stripped version answers the different
        question of what the sensor is worth end to end.
    :param forcing: ``(T, ell)`` inputs.
    :param weather: ``(T, m)`` weather anomalies.
    :param obs_cov: ``(T, r, r)`` per-date GP posterior covariances.
    :param sigma_eps2: fallback observation noise.
    :param family: horizon family (needed by any ``mode="direct"`` variant).
    :param keep_cov: retain full per-horizon covariances (see :func:`rolling_forecast`).
    :return: ``{variant_name: result dict}`` with the same keys
             :func:`rolling_forecast` returns.
    """
    w_obs = np.asarray(w_obs, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    t_total, r = w_obs.shape
    origins = np.asarray(list(origins), dtype=int)
    names = [str(v["name"]) for v in variants]

    state = np.zeros(sys.lifted_dim)
    first = int(np.argmax(observed)) if observed.any() else 0
    for j in range(sys.order):
        state[j * r : (j + 1) * r] = w_obs[first]
    cov = np.eye(sys.lifted_dim) * max(sigma_eps2, 1e-8)

    out = {
        name: {
            "origins": origins,
            "mean": np.zeros((origins.size, horizon, r)),
            "mean_prior": np.zeros((origins.size, horizon, r)),
            "cov_trace": np.zeros((origins.size, horizon)),
            "trace_reduction": np.zeros((origins.size, horizon)),
            "valid": np.zeros((origins.size, horizon), dtype=bool),
            "mode": str(variants[i].get("mode", "recursive")),
            **({"cov": np.zeros((origins.size, horizon, r, r))} if keep_cov else {}),
        }
        for i, name in enumerate(names)
    }
    slot = {int(o): i for i, o in enumerate(origins)}
    want = set(slot)

    for t in range(t_total):
        if t > 0:
            u = None
            if forcing is not None and np.asarray(forcing).shape[1]:
                u = np.asarray(forcing)[t - 1]
            state, cov = predict(sys, state, cov, u)
        if weather is not None and sys.c_w is not None:
            y = np.asarray(weather)[t]
            if np.all(np.isfinite(y)):
                state, cov, _ = update(state, cov, sys.c_w, y, sys.r_w)
        if observed[t]:
            noise = obs_cov[t] if obs_cov is not None else sigma_eps2 * np.eye(r)
            state, cov, _ = update(state, cov, np.eye(r), w_obs[t], noise)

        if t not in want:
            continue
        i = slot[t]
        hi = min(horizon, t_total - 1 - t)
        f_branch = None if forcing is None else np.asarray(forcing)[t : t + horizon]
        if f_branch is not None and f_branch.shape[0] < horizon:
            f_branch = np.concatenate(
                [f_branch, np.zeros((horizon - f_branch.shape[0], f_branch.shape[1]))]
            )
        w_full = None
        if weather is not None:
            w_full = np.asarray(weather)[t + 1 : t + 1 + horizon]
            if w_full.shape[0] < horizon:
                w_full = np.concatenate(
                    [w_full, np.full((horizon - w_full.shape[0], w_full.shape[1]), np.nan)]
                )

        for spec, name in zip(variants, names):
            fc = forecast_from_origin(
                sys, state, cov, horizon, f_branch,
                w_full if spec.get("use_weather", True) else None,
                family, str(spec.get("mode", "recursive")),
            )
            res = out[name]
            res["mean"][i] = fc.mean
            res["mean_prior"][i] = fc.mean_prior
            res["cov_trace"][i] = np.trace(fc.cov, axis1=1, axis2=2)
            res["trace_reduction"][i] = fc.trace_reduction
            res["valid"][i, :hi] = True
            if keep_cov:
                res["cov"][i] = fc.cov

    logger.info(
        "Rolling forecast: 1 filter pass, %d origins x %d variants x H=%d.",
        origins.size, len(variants), horizon,
    )
    return out


def rolling_forecast(
    sys: LiftedSystem,
    w_obs: np.ndarray,
    observed: np.ndarray,
    origins: Sequence[int],
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    weather: Optional[np.ndarray] = None,
    obs_cov: Optional[np.ndarray] = None,
    sigma_eps2: float = 1e-2,
    family=None,
    mode: str = "recursive",
    keep_cov: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run the ``H``-step protocol from every test origin, filtering forward once.

    The filter is advanced through the record a single time; at each requested
    origin the current state is branched to produce a forecast. Crucially, the
    filter's own forward pass assimilates NDVI frames (that is legitimate -- at
    origin ``t`` all data up to ``t`` is available), while each *forecast branch*
    sees only weather.

    :param sys: the assembled system.
    :param w_obs: ``(T, r)`` GP-solved weights.
    :param observed: ``(T,)`` NDVI availability.
    :param origins: forecast origin indices.
    :param horizon: ``H``.
    :param forcing: ``(T, ell)`` inputs.
    :param weather: ``(T, m)`` weather anomalies.
    :param obs_cov: ``(T, r, r)`` per-date GP posterior covariances.
    :param sigma_eps2: fallback observation noise.
    :param family: horizon family (needed for ``mode="direct"``).
    :param mode: ``"recursive"`` or ``"direct"``.
    :param keep_cov: retain the full ``(n_origins, H, r, r)`` covariance stack.
        Off by default: at ``r = 256`` with ~370 origins that array is 1.1 GB per
        variant, and the per-horizon metrics need only the mean. Turn it on for
        calibration work (PIT / coverage), ideally with a stride so the stack
        stays small.
    :return: dict with ``origins``, ``mean`` ``(n_origins, H, r)``, ``mean_prior``,
             ``cov`` (only if ``keep_cov``), ``cov_trace`` ``(n_origins, H)``,
             ``trace_reduction`` and ``valid`` ``(n_origins, H)`` marking horizon
             steps that fall inside the record.
    """
    w_obs = np.asarray(w_obs, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    t_total, r = w_obs.shape
    origins = np.asarray(list(origins), dtype=int)

    state = np.zeros(sys.lifted_dim)
    first = int(np.argmax(observed)) if observed.any() else 0
    for j in range(sys.order):
        state[j * r : (j + 1) * r] = w_obs[first]
    cov = np.eye(sys.lifted_dim) * max(sigma_eps2, 1e-8)

    want = set(origins.tolist())
    means = np.zeros((origins.size, horizon, r))
    means_prior = np.zeros((origins.size, horizon, r))
    covs = np.zeros((origins.size, horizon, r, r)) if keep_cov else None
    cov_traces = np.zeros((origins.size, horizon))
    reductions = np.zeros((origins.size, horizon))
    valid = np.zeros((origins.size, horizon), dtype=bool)
    slot = {int(o): i for i, o in enumerate(origins)}

    for t in range(t_total):
        if t > 0:
            u = None
            if forcing is not None and np.asarray(forcing).shape[1]:
                u = np.asarray(forcing)[t - 1]
            state, cov = predict(sys, state, cov, u)
        if weather is not None and sys.c_w is not None:
            y = np.asarray(weather)[t]
            if np.all(np.isfinite(y)):
                state, cov, _ = update(state, cov, sys.c_w, y, sys.r_w)
        if observed[t]:
            noise = obs_cov[t] if obs_cov is not None else sigma_eps2 * np.eye(r)
            state, cov, _ = update(state, cov, np.eye(r), w_obs[t], noise)

        if t in want:
            i = slot[t]
            hi = min(horizon, t_total - 1 - t)
            f_branch = (
                None
                if forcing is None
                else np.asarray(forcing)[t : t + horizon]
            )
            w_branch = (
                None
                if weather is None
                else np.asarray(weather)[t + 1 : t + 1 + horizon]
            )
            if w_branch is not None and w_branch.shape[0] < horizon:
                pad = np.full((horizon - w_branch.shape[0], w_branch.shape[1]), np.nan)
                w_branch = np.concatenate([w_branch, pad], axis=0)
            if f_branch is not None and f_branch.shape[0] < horizon:
                pad = np.zeros((horizon - f_branch.shape[0], f_branch.shape[1]))
                f_branch = np.concatenate([f_branch, pad], axis=0)
            fc = forecast_from_origin(
                sys, state, cov, horizon, f_branch, w_branch, family, mode
            )
            means[i] = fc.mean
            means_prior[i] = fc.mean_prior
            if keep_cov:
                covs[i] = fc.cov
            cov_traces[i] = np.trace(fc.cov, axis1=1, axis2=2)
            reductions[i] = fc.trace_reduction
            valid[i, :hi] = True

    out = {
        "origins": origins,
        "mean": means,
        "mean_prior": means_prior,
        "cov_trace": cov_traces,
        "trace_reduction": reductions,
        "valid": valid,
        "mode": mode,
    }
    if keep_cov:
        out["cov"] = covs
    return out
