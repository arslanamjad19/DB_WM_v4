"""
Rank and conditioning diagnostics for the latent weight trajectory.

The framework sizes ``r`` against the number of *pixels* (v2 Prop. 3.1 wants
``r << n``, and Sec. 8.1 suggests ``r = 512-1024`` for a 4,500 km^2 scene). That
constraint is necessary but **not sufficient**, and taking it as the whole story
is what breaks the dynamics on a field-scale AOI.

The binding constraint for the *dynamics* is temporal, not spatial: identification
regresses ``w_{t+1}`` on ``w_t``, so what matters is how many directions of
``R^r`` the weight trajectory actually **excites**. For a single agricultural field
the NDVI record spans very few independent spatial patterns -- measured here at a
95%-energy rank of order 10 -- however many pixels it contains.

What goes wrong when ``r`` exceeds that rank
--------------------------------------------
The failure is a chain, and only the last link is visible in the forecast:

1. ``W_-`` is rank-deficient, so the ridge fit ``A_0 = W_+ W_-^T (W_- W_-^T + mu I)^{-1}``
   is near-singular in the unexcited directions -- exactly the zero eigenvalues
   v2 Prop. 4.4 predicts for ``ker W_-^T``.
2. A near-singular, non-symmetric ``A_0`` has a near-**defective** eigenbasis, so
   ``cond(P)`` in :func:`~dbwm.dynamics.memory.real_modal_form` is large.
3. The S2 blocks are ``A_j = P diag(alpha_j) P^{-1}``, whose norm scales with
   ``cond(P)``. Tiny per-mode coefficients then produce blocks with enormous
   norms -- a measured ``sum_j ||A_j||_2 = 162`` at ``rho(A_cal) = 1``.
4. Such an operator is violently **non-normal**. Spectral radius bounds only
   asymptotic growth; the transient ``sup_k ||A_cal^k||`` can be orders of
   magnitude larger. One predict step then amplifies the state error instead of
   propagating it.
5. The one-step forecast is therefore the *worst* horizon, and later horizons look
   better only because the observer keeps correcting. A forecast whose error
   *decreases* with lead time is the signature of this, not of a good model.

So the diagnostics here are not decoration: ``rho <= 1`` passing while the forecast
is worse than climatology is precisely the case they exist to catch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.conditioning")


@dataclass
class RankReport:
    """
    Effective rank of a weight trajectory.

    :ivar singular_values: ``(min(T, r),)`` singular values of the centred
        trajectory.
    :ivar rank_95: directions holding 95% of the temporal variance.
    :ivar rank_99: directions holding 99%.
    :ivar numerical_rank: singular values above ``tol * s_max``.
    :ivar r: the basis dimension in use.
    :ivar over_parameterised: ``True`` when ``r`` materially exceeds ``rank_99``.
    """

    singular_values: np.ndarray
    rank_95: int
    rank_99: int
    numerical_rank: int
    r: int
    over_parameterised: bool

    def suggested_r(self, headroom: float = 2.0, minimum: int = 8) -> int:
        """
        A defensible ``r`` for the dynamics: the 99%-energy rank with headroom.

        Headroom matters because the basis must still *represent* the field, not
        merely span its temporal variation; going all the way down to ``rank_99``
        would start costing reconstruction accuracy.

        :param headroom: multiplier on ``rank_99``.
        :param minimum: floor.
        :return: the suggested basis dimension.
        """
        return int(max(minimum, min(self.r, round(headroom * self.rank_99))))


def weight_rank(
    weights: np.ndarray, valid: Optional[np.ndarray] = None, tol: float = 1e-8
) -> RankReport:
    """
    Measure how many directions of ``R^r`` the weight trajectory excites.

    :param weights: ``(T, r)`` weight trajectory.
    :param valid: ``(T,)`` mask of usable rows.
    :param tol: relative singular-value tolerance for the numerical rank.
    :return: the :class:`RankReport`.
    """
    w = np.asarray(weights, dtype=np.float64)
    if valid is not None:
        w = w[np.asarray(valid, dtype=bool)]
    if w.shape[0] < 2:
        raise ValueError("need at least 2 rows to measure rank")
    centred = w - w.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    energy = np.cumsum(s**2) / max(float(np.sum(s**2)), 1e-30)
    r = w.shape[1]
    rank_95 = int(np.searchsorted(energy, 0.95) + 1)
    rank_99 = int(np.searchsorted(energy, 0.99) + 1)
    numerical = int(np.sum(s > tol * s[0])) if s.size else 0
    return RankReport(
        singular_values=s,
        rank_95=rank_95,
        rank_99=rank_99,
        numerical_rank=numerical,
        r=r,
        over_parameterised=bool(r > 4 * rank_99),
    )


def log_rank_report(report: RankReport) -> None:
    """
    Report the rank and, when ``r`` is oversized, say what it will break.

    :param report: the :class:`RankReport`.
    """
    logger.info(
        "Weight-trajectory rank: 95%% energy in %d directions, 99%% in %d, "
        "numerical rank %d, against r = %d.",
        report.rank_95, report.rank_99, report.numerical_rank, report.r,
    )
    if not report.over_parameterised:
        return
    logger.warning(
        "r = %d is %.0fx the 99%%-energy rank (%d). This is the binding constraint "
        "for the DYNAMICS, and it is temporal, not spatial: r << n_pixels is "
        "necessary but not sufficient.\n"
        "  Expect, in order: a near-singular A_0 (so Thm 2.8(i) fails at EVERY L, "
        "including L=1, where it cannot mean over-lagging); an ill-conditioned "
        "modal basis; S2 blocks with huge norms despite rho <= 1; and a one-step "
        "forecast that is worse than climatology because the operator is "
        "non-normal.\n"
        "  Suggested: --r %d",
        report.r, report.r / max(report.rank_99, 1), report.rank_99,
        report.suggested_r(),
    )


def transient_amplification(a_cal: np.ndarray, horizon: int = 12) -> Dict[str, float]:
    """
    Measure ``sup_k ||A_cal^k||_2`` -- the growth ``rho <= 1`` does not bound.

    For a **normal** operator ``||A^k|| = rho^k``, so ``rho <= 1`` guarantees no
    growth. For a non-normal one the two decouple: the powers can grow by orders
    of magnitude before eventually decaying, and the peak is what a finite-horizon
    forecast actually experiences. This is the quantity that explains a one-step
    prediction being worse than the state it started from.

    :param a_cal: ``(Lr, Lr)`` companion matrix.
    :param horizon: how many powers to examine.
    :return: dict with ``rho``, ``norm_1``, ``peak``, ``peak_step`` and
             ``non_normality`` (``peak / max(rho, 1)``).
    """
    a = np.asarray(a_cal, dtype=np.float64)
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    powers, cur = [], np.eye(a.shape[0])
    for _ in range(horizon):
        cur = cur @ a
        powers.append(float(np.linalg.norm(cur, 2)))
    peak = float(max(powers))
    return {
        "rho": rho,
        "norm_1": powers[0],
        "peak": peak,
        "peak_step": int(np.argmax(powers) + 1),
        "non_normality": peak / max(rho, 1.0),
        "powers": np.asarray(powers),
    }


def log_transient(report: Dict[str, float], threshold: float = 5.0) -> None:
    """
    Report transient growth and warn when the operator is badly non-normal.

    :param report: output of :func:`transient_amplification`.
    :param threshold: amplification above which to warn.
    """
    logger.info(
        "Transient growth: rho = %.4f, ||A|| = %.2f, peak ||A^k|| = %.2f at k = %d "
        "(non-normality %.1fx).",
        report["rho"], report["norm_1"], report["peak"], report["peak_step"],
        report["non_normality"],
    )
    if report["non_normality"] > threshold:
        logger.warning(
            "The lifted operator is strongly NON-NORMAL: rho = %.3f satisfies the "
            "stability constraint, yet powers peak at %.1fx. Spectral radius bounds "
            "only asymptotic behaviour, so a %d-step forecast can be amplified "
            "rather than damped -- which shows up as the one-step horizon being the "
            "WORST one. Usually caused by r exceeding the excited rank; reduce r "
            "before anything else.",
            report["rho"], report["peak"], report["peak_step"],
        )
