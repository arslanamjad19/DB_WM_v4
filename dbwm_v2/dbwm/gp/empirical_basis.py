"""
Closing the representation floor: climatology offset + residual empirical modes.

The measured problem
--------------------
On the real Sayedanwala record the learned spatial basis reaches a median
reconstruction ``R^2 = 0.958``. That is a **hard floor on every forecast**, because
the forecast is decoded through the same basis: no dynamics model, however good,
can produce a field the basis cannot represent. In normalised units that floor is

    eps_enc = sqrt(1 - 0.958) * (within-date spatial std) ~= 0.18

and v2 Theorem 4.2 then multiplies it by the operator norm. With the measured
``||A_cal||_2 = 4.13`` that predicts a one-step error of ``4.13 * 0.18 = 0.76``
normalised = **0.103 physical** -- against an observed 0.1026. The one-step error
is, to three significant figures, *nothing but the representation error amplified
by the operator*. Fixing the dynamics alone therefore cannot help; the floor has to
come down too.

Why a coordinate MLP hits a floor here
--------------------------------------
``Psi`` is a smooth function of ``(lon, lat)``. This AOI is an agricultural mosaic:
piecewise-constant plots with sharp boundaries, each following its own crop
calendar. Representing a step edge with smooth features is the classic spectral-bias
problem -- Fourier features postpone it but do not remove it, and rank ``r = 256``
buys only so much. The residual is not noise; it is *structure the basis cannot
reach*.

The fix, and why it is inside the framework rather than a departure
-------------------------------------------------------------------
Two data-driven blocks are appended to the feature matrix:

**1. A climatology offset** ``m(s)``, the per-pixel training mean. The plot mosaic
is largely static, so a single fixed field removes most of the spatial variance at
zero model cost. Weights then describe the *anomaly*, which is what the dynamics
should have been modelling all along -- and an anomaly state is far better
conditioned for a linear operator than one carrying a large static mean.

**2. Empirical orthogonal functions** of what ``Psi`` leaves behind. The residual
``R = (Y - m) - P_Psi (Y - m)`` is what the learned basis provably cannot express;
its leading left-singular vectors are the optimal rank-``q`` completion in the L2
sense, with no smoothness prior to fight the plot edges.

This is not a departure from the DBK construction. v2 Theorem 2.1 says the *ideal*
basis is the Mercer eigenfunction set ``phi_i = sqrt(lambda_i) psi_i`` of the kernel;
EOFs are precisely the empirical estimate of those eigenfunctions for the covariance
the data actually has. The E-GP paper lists the same object ("Low-Rank
Approximations": diagonalise the Gram matrix to get ``[lambda_i, phi_i]``) as one of
its three admissible feature maps. Concatenating two feature blocks yields
``k(x,x') = <Psi(x),Psi(x')> + <U(x),U(x')>``, a sum of PSD kernels, so every
Woodbury/observability result that depends only on ``Phi_X`` carries over unchanged.

The honest limitation
---------------------
The EOF block is **transductive**: it is defined at the grid cells it was fitted on
and, unlike ``Psi``, cannot be evaluated at a coordinate outside that grid. For this
application the grid is fixed across all dates, so this costs nothing; for
transfer to a new scene the EOF block must be refitted or Nystrom-extended, and
``Psi`` alone is the part that transfers. This is reported, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.empbasis")


@dataclass
class AugmentedBasis:
    """
    The feature matrix actually used for encoding and decoding.

    :ivar phi: ``(n_pixels, r + q)`` features, learned block then empirical block.
    :ivar offset: ``(n_pixels,)`` climatology, added back on decode.
    :ivar r_learned: width of the learned ``Psi`` block.
    :ivar q_empirical: width of the appended EOF block.
    :ivar explained: fraction of the residual variance the EOF block captures.
    :ivar r2_before: median reconstruction ``R^2`` using ``Psi`` alone.
    :ivar r2_after: median reconstruction ``R^2`` using the augmented basis.
    """

    phi: np.ndarray
    offset: np.ndarray
    r_learned: int
    q_empirical: int
    explained: float
    r2_before: float
    r2_after: float
    #: Reconstruction RMSE (normalised) on the training dates the modes were fitted
    #: to, and on a held-out chronological tail they were not. The RATIO is the
    #: number that matters: the basis is data-driven, so it can fit the frames it
    #: saw arbitrarily well while generalising poorly, and the held-out value --
    #: not the fitted one -- is the floor every forecast inherits.
    rmse_fit: float = float("nan")
    rmse_holdout: float = float("nan")
    #: ``(n_pixels,)`` per-pixel held-out residual variance, normalised units.
    #: This is the honest input to the predictive band; the in-sample residual
    #: understates it by the generalisation gap.
    holdout_variance: Optional[np.ndarray] = None
    #: The ``q``-versus-held-out-RMSE table the selection walked.
    selection: Optional[Dict[str, object]] = None

    @property
    def r(self) -> int:
        """Total feature width."""
        return self.phi.shape[1]

    @property
    def generalisation_gap(self) -> float:
        """Held-out reconstruction RMSE divided by the fitted one."""
        return float(self.rmse_holdout / max(self.rmse_fit, 1e-12))


def _reconstruction_r2(
    phi: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float
) -> float:
    """
    Median per-date ``R^2`` of the least-squares round trip through ``phi``.

    Scored exactly as :meth:`~dbwm.gp.state.GPStateExtractor.solve_sequence`
    scores it, so the number reported here is the number that will appear in the
    pipeline log rather than an optimistic variant of it.

    :param phi: ``(n, k)`` features on valid pixels.
    :param y: ``(T, n)`` values on valid pixels.
    :param offset: ``(n,)`` climatology.
    :param ridge: Tikhonov ridge (the GP's ``sigma_eps^2``).
    :return: median ``R^2`` over dates.
    """
    lam = phi.T @ phi + ridge * np.eye(phi.shape[1])
    resid = y - offset
    w = np.linalg.solve(lam, phi.T @ resid.T).T
    pred = w @ phi.T + offset
    ss_res = np.sum((y - pred) ** 2, axis=1)
    ss_tot = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ok = ss_tot > 1e-30
    return float(np.median(1.0 - ss_res[ok] / ss_tot[ok])) if ok.any() else float("nan")


def _round_trip(
    phi: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float
) -> Tuple[float, np.ndarray]:
    """
    Reconstruction RMSE and per-pixel residual variance through a given basis.

    :param phi: ``(n, k)`` features on valid pixels.
    :param y: ``(T, n)`` values on valid pixels.
    :param offset: ``(n,)`` climatology.
    :param ridge: Tikhonov ridge (the GP's ``sigma_eps^2``).
    :return: ``(rmse, per-pixel variance)`` in normalised units.
    """
    lam = phi.T @ phi + ridge * np.eye(phi.shape[1])
    w = np.linalg.solve(lam, phi.T @ (y - offset).T).T
    resid = y - (w @ phi.T + offset)
    return float(np.sqrt(np.mean(resid**2))), (resid**2).mean(axis=0)


def _shape_modes(
    phi_v: np.ndarray, directions: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    """
    Orthogonalise a set of residual directions against ``Psi`` and match its scale.

    :param phi_v: ``(n, r)`` learned features on valid pixels.
    :param directions: ``(n, q)`` candidate empirical directions.
    :param lam: ``(r, r)`` ``Psi^T Psi + ridge I``, precomputed.
    :return: ``(n, q)`` the empirical block.
    """
    modes = directions - phi_v @ np.linalg.solve(lam, phi_v.T @ directions)
    modes, _ = np.linalg.qr(modes)
    return modes * float(np.linalg.norm(phi_v) / max(np.linalg.norm(modes), 1e-30))


def _select_mode_count(
    phi_v: np.ndarray,
    y: np.ndarray,
    ridge: float,
    n_modes: int,
    holdout_fraction: float,
    use_climatology: bool,
) -> Dict[str, object]:
    """
    Choose ``q`` by **held-out** reconstruction, not by training-residual energy.

    Why the energy criterion is the wrong one here
    ----------------------------------------------
    ``energy`` asks how much of the *training* residual the leading ``q`` modes
    span, and stops at a fixed share of it (or at the ``n_modes`` cap, whichever
    binds first). Neither stopping rule ever looks at data the modes were not
    fitted to, so neither can tell a **mode-starved** basis from an **over-fitted**
    one -- and those call for opposite actions. EOFs are a data-driven basis;
    unlike ``Psi`` they carry no smoothness prior to stop them chasing per-frame
    structure.

    On the real record that is exactly what happened. With 256 modes selected by
    energy (99.641% of the training residual), reconstruction came out at **0.0016
    NDVI on training dates and 0.044 on test dates -- a 28x generalisation gap**.
    Because the identified operator had been shrunk to persistence, the forecast
    *was* the origin's reconstruction, so that 0.044 became an exact additive
    floor under every horizon::

        ubRMSE(h) = sqrt( persistence(h)^2 + 0.0437^2 )

    which reproduces all six measured values to within 0.0005 NDVI. No change to
    the dynamics can move a term that is already present before the operator runs.

    The fix is to select ``q`` the way any other hyper-parameter is selected: on
    data the modes were not fitted to. The split is **chronological** (a tail of
    the training window), never random, because random frames from the same weeks
    are near-duplicates and would report an in-sample number under a held-out
    name.

    :param phi_v: ``(n, r)`` learned features on valid pixels.
    :param y: ``(T, n)`` training values on valid pixels, chronological.
    :param ridge: the GP ``sigma_eps^2``.
    :param n_modes: cap on ``q``.
    :param holdout_fraction: tail fraction of the training window held out.
    :param use_climatology: whether a per-pixel offset is part of the basis.
    :return: dict with ``q``, ``grid`` (``[{q, rmse}]``), ``rmse_fit``,
             ``rmse_holdout`` and the held-out per-pixel ``variance``.
    """
    n_dates = y.shape[0]
    cut = int(round(n_dates * (1.0 - holdout_fraction)))
    cut = int(np.clip(cut, 2, n_dates - 2))
    y_fit, y_hold = y[:cut], y[cut:]

    # Everything -- the climatology AND the modes -- is fitted on y_fit only, or
    # the "held-out" score would have seen its own answer.
    offset = y_fit.mean(axis=0) if use_climatology else np.zeros(y.shape[1])
    lam = phi_v.T @ phi_v + ridge * np.eye(phi_v.shape[1])
    resid = y_fit - offset
    left_over = resid - (np.linalg.solve(lam, phi_v.T @ resid.T).T @ phi_v.T)
    _, _, vt = np.linalg.svd(left_over, full_matrices=False)

    cap = int(min(n_modes, vt.shape[0]))
    grid = sorted({0, *[q for q in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                                    384, 512) if q <= cap], cap})
    rows, best = [], None
    for q in grid:
        block = (
            phi_v if q == 0
            else np.concatenate([phi_v, _shape_modes(phi_v, vt[:q].T, lam)], axis=1)
        )
        rmse_h, var_h = _round_trip(block, y_hold, offset, ridge)
        rmse_f, _ = _round_trip(block, y_fit, offset, ridge)
        rows.append({"q": int(q), "rmse_holdout": rmse_h, "rmse_fit": rmse_f})
        if best is None or rmse_h < best["rmse_holdout"]:
            best = {"q": int(q), "rmse_holdout": rmse_h, "rmse_fit": rmse_f,
                    "variance": var_h}

    logger.info(
        "Empirical-mode selection on a held-out tail of %d/%d training dates:",
        y_hold.shape[0], n_dates,
    )
    for row in rows:
        logger.info(
            "  q = %4d   fit %.5f   held-out %.5f   gap %5.1fx%s",
            row["q"], row["rmse_fit"], row["rmse_holdout"],
            row["rmse_holdout"] / max(row["rmse_fit"], 1e-12),
            "   <- selected" if row["q"] == best["q"] else "",
        )
    worst = max(rows, key=lambda r: r["rmse_holdout"] / max(r["rmse_fit"], 1e-12))
    logger.info(
        "Selected q = %d by held-out RMSE %.5f. The largest-q candidate (q = %d) "
        "fits training %.0fx better than it generalises -- that gap is the "
        "forecast floor, and it is invisible to the energy criterion.",
        best["q"], best["rmse_holdout"], worst["q"],
        worst["rmse_holdout"] / max(worst["rmse_fit"], 1e-12),
    )

    # Which way does the curve point at the top end? The two answers call for
    # opposite actions, and the fitted-vs-held-out gap alone cannot tell them
    # apart -- with a data-driven basis the fitted error is near zero by
    # construction (the modes ARE the training residual's SVD), so a large gap is
    # expected and is not by itself evidence of too many modes.
    if len(rows) >= 2 and best["q"] == cap:
        prev = rows[-2]["rmse_holdout"]
        gain = (prev - best["rmse_holdout"]) / max(prev, 1e-12)
        logger.warning(
            "The --eof-modes CAP IS BINDING: q = %d was the largest offered and "
            "it won, still improving held-out RMSE by %.1f%% over q = %d. The "
            "basis is mode-STARVED, not over-fitted: raise --eof-modes (and r if "
            "that saturates too). An agricultural mosaic needs roughly one "
            "independent direction per parcel to be representable at all, so a "
            "cap below the parcel count is a hard floor on every forecast.",
            best["q"], 100.0 * gain, rows[-2]["q"],
        )
    elif best["q"] < cap:
        logger.info(
            "Held-out error turns back up beyond q = %d, so the mode count is "
            "genuinely optimal rather than capped: the extra directions describe "
            "the fitted frames rather than the field.", best["q"],
        )
    return {"q": best["q"], "grid": rows, "rmse_fit": best["rmse_fit"],
            "rmse_holdout": best["rmse_holdout"], "variance": best["variance"]}


def _holdout_report(
    phi_v: np.ndarray,
    y: np.ndarray,
    ridge: float,
    q: int,
    holdout_fraction: float,
    use_climatology: bool,
) -> Dict[str, object]:
    """
    Score one fixed ``q`` on a held-out tail, without selecting anything.

    Used when the mode count came from somewhere else (the ``energy`` rule, or an
    explicit ``--eof-modes``): the generalisation gap is reported either way, so
    an ablation cannot quietly hide its own cost.

    :param phi_v: ``(n, r)`` learned features on valid pixels.
    :param y: ``(T, n)`` training values, chronological.
    :param ridge: the GP ``sigma_eps^2``.
    :param q: the mode count actually used.
    :param holdout_fraction: tail fraction held out.
    :param use_climatology: whether a per-pixel offset is part of the basis.
    :return: dict with ``q``, ``rmse_fit``, ``rmse_holdout`` and ``variance``.
    """
    n_dates = y.shape[0]
    cut = int(np.clip(round(n_dates * (1.0 - holdout_fraction)), 2, n_dates - 2))
    y_fit, y_hold = y[:cut], y[cut:]
    offset = y_fit.mean(axis=0) if use_climatology else np.zeros(y.shape[1])
    lam = phi_v.T @ phi_v + ridge * np.eye(phi_v.shape[1])

    block = phi_v
    if q > 0:
        resid = y_fit - offset
        left_over = resid - (np.linalg.solve(lam, phi_v.T @ resid.T).T @ phi_v.T)
        _, _, vt = np.linalg.svd(left_over, full_matrices=False)
        k = int(min(q, vt.shape[0]))
        if k > 0:
            block = np.concatenate(
                [phi_v, _shape_modes(phi_v, vt[:k].T, lam)], axis=1
            )
    rmse_h, var_h = _round_trip(block, y_hold, offset, ridge)
    rmse_f, _ = _round_trip(block, y_fit, offset, ridge)
    return {"q": int(q), "rmse_fit": rmse_f, "rmse_holdout": rmse_h,
            "variance": var_h, "grid": None}


def build_augmented_basis(
    phi: np.ndarray,
    frames: np.ndarray,
    valid_flat: np.ndarray,
    train_observed: np.ndarray,
    n_modes: int = 96,
    energy: float = 0.999,
    ridge: float = 1e-2,
    use_climatology: bool = True,
    select: str = "holdout",
    holdout_fraction: float = 0.2,
) -> AugmentedBasis:
    """
    Append a climatology offset and residual EOFs to the learned basis.

    Every quantity is estimated on **training, observed** dates only. Fitting the
    climatology or the modes on the full record would leak the evaluation period
    into the decoder, and would do so invisibly: the forecast metrics would improve
    for a reason that has nothing to do with the dynamics.

    :param phi: ``(n_pixels, r)`` learned features on the full grid.
    :param frames: ``(T, H, W)`` normalised frames.
    :param valid_flat: ``(n_pixels,)`` bool, the date-invariant validity mask.
    :param train_observed: ``(T,)`` bool, training dates that carry a frame.
    :param n_modes: cap on the number of empirical modes ``q``. 0 disables them.
    :param energy: fraction of the *residual* variance to retain.
    :param ridge: ridge used when scoring the round trip (the GP ``sigma_eps^2``).
    :param use_climatology: append the per-pixel training mean.
    :param select: how ``q`` is chosen. ``"holdout"`` (default) minimises
        reconstruction error on a chronological tail of the training window --
        see :func:`_select_mode_count` for why the alternative overfits.
        ``"energy"`` restores the previous cumulative-variance rule as an
        ablation.
    :param holdout_fraction: tail fraction used by ``select="holdout"``.
    :return: the :class:`AugmentedBasis`.
    """
    phi = np.asarray(phi, dtype=np.float64)
    valid = np.asarray(valid_flat, dtype=bool).reshape(-1)
    sel = np.asarray(train_observed, dtype=bool)
    n_pix = phi.shape[0]

    y = np.asarray(frames, dtype=np.float64).reshape(frames.shape[0], -1)[sel][:, valid]
    if y.shape[0] < 2:
        raise ValueError("need at least 2 training frames to build an empirical basis")
    phi_v = phi[valid]

    offset_v = y.mean(axis=0) if use_climatology else np.zeros(y.shape[1])
    r2_before = _reconstruction_r2(phi_v, y, np.zeros(y.shape[1]), ridge)

    # ---- How many empirical modes? --------------------------------------- #
    chosen = None
    if n_modes > 0 and str(select).lower() == "holdout":
        chosen = _select_mode_count(
            phi_v, y, ridge, n_modes, holdout_fraction, use_climatology
        )

    resid = y - offset_v
    blocks = [phi_v]
    explained = 0.0
    if n_modes > 0:
        # What Psi provably cannot express: project the anomaly onto span(Psi) and
        # keep the orthogonal complement. Fitting EOFs to the raw anomaly instead
        # would re-learn directions the basis already covers, wasting modes and
        # making Lambda_X badly conditioned through near-collinear columns.
        lam = phi_v.T @ phi_v + ridge * np.eye(phi_v.shape[1])
        coef = np.linalg.solve(lam, phi_v.T @ resid.T)
        left_over = resid - (coef.T @ phi_v.T)

        _, s, vt = np.linalg.svd(left_over, full_matrices=False)
        power = np.cumsum(s**2) / max(float(np.sum(s**2)), 1e-30)
        if chosen is not None:
            q = int(min(chosen["q"], vt.shape[0]))
        else:
            # The ablation path. `energy` is monotone in q, so it can only ever be
            # stopped by the n_modes cap -- it has no opinion on generalisation.
            q = int(min(np.searchsorted(power, energy) + 1, n_modes, vt.shape[0]))
        if q > 0:
            blocks.append(_shape_modes(phi_v, vt[:q].T, lam))
            explained = float(power[q - 1])
    else:
        q = 0

    phi_v_aug = np.concatenate(blocks, axis=1)
    r2_after = _reconstruction_r2(phi_v_aug, y, offset_v, ridge)

    # The held-out numbers are what the forecast actually inherits. Measure them
    # even when `select="energy"`, so the ablation reports its own cost.
    if chosen is None and n_modes >= 0:
        chosen = _holdout_report(
            phi_v, y, ridge, q, holdout_fraction, use_climatology
        )
    hold_var = np.zeros(n_pix)
    hold_var[valid] = chosen["variance"]

    phi_aug = np.zeros((n_pix, phi_v_aug.shape[1]))
    phi_aug[valid] = phi_v_aug
    offset = np.zeros(n_pix)
    offset[valid] = offset_v

    logger.info(
        "Augmented basis: learned r=%d + %d empirical modes (%.3f%% of the residual "
        "variance) + %s climatology.",
        phi.shape[1], q, 100.0 * explained,
        "a" if use_climatology else "no",
    )
    logger.info(
        "Reconstruction R2: %.4f -> %.4f  (encoding RMSE %.4f -> %.4f normalised). "
        "v2 Thm 4.2 multiplies this by ||A_cal||, so it is the term that sets the "
        "forecast floor.",
        r2_before, r2_after, np.sqrt(max(1 - r2_before, 0)),
        np.sqrt(max(1 - r2_after, 0)),
    )
    logger.info(
        "Reconstruction RMSE fitted %.5f vs HELD OUT %.5f (%.0fx gap, normalised). "
        "The held-out value is the one every forecast inherits: with the operator "
        "shrunk to persistence the forecast IS the origin's reconstruction, so "
        "ubRMSE(h) = sqrt(persistence(h)^2 + this^2) exactly.",
        chosen["rmse_fit"], chosen["rmse_holdout"],
        chosen["rmse_holdout"] / max(chosen["rmse_fit"], 1e-12),
    )
    if chosen["rmse_holdout"] > 4.0 * max(chosen["rmse_fit"], 1e-12):
        logger.warning(
            "The empirical basis OVERFITS: it reconstructs the frames it was "
            "fitted on %.0fx better than frames it was not. q = %d modes were "
            "fitted to %d training dates, and the excess directions describe "
            "those particular frames rather than the field. This is a "
            "representation failure, not a dynamics one -- it is present before "
            "the operator runs and no memory order, rho cap or shrinkage can "
            "reduce it. Lower --eof-modes, or leave --eof-select holdout to "
            "choose it by generalisation.",
            chosen["rmse_holdout"] / max(chosen["rmse_fit"], 1e-12), q,
            y.shape[0],
        )
    if r2_after < 0.99:
        logger.warning(
            "Reconstruction R2 is still %.4f. With a unit-norm operator the best "
            "achievable one-step RMSE is about %.4f of the field std, so a target "
            "below that is unreachable no matter what the dynamics do. Raise "
            "--eof-modes or r.", r2_after, np.sqrt(max(1 - r2_after, 0)),
        )
    return AugmentedBasis(
        phi=phi_aug, offset=offset, r_learned=phi.shape[1], q_empirical=q,
        explained=explained, r2_before=r2_before, r2_after=r2_after,
        rmse_fit=chosen["rmse_fit"], rmse_holdout=chosen["rmse_holdout"],
        holdout_variance=hold_var, selection={"grid": chosen.get("grid"),
                                              "rule": str(select)},
    )
