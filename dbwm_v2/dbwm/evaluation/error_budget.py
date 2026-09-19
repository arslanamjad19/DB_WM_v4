"""
Where the forecast error actually comes from -- reported, not guessed.

The v4 forecast passes through four lossy stages, and each one puts a floor under
everything after it:

===  ==========================  =================================================
1    representation              ``y -> w -> y``: what the basis cannot express
2    subspace truncation         ``w -> z -> w``: what the POD basis discards
3    operator gain               ``max_h ||S A_cal^h||``: amplification of 1 and 2
4    dynamics residual           what the operator gets wrong about the future
===  ==========================  =================================================

v2 Theorem 4.2 composes them as ``rho^h eps_enc + ((rho^h - 1)/(rho - 1))
eps_dyn``, so terms 1-2 arrive at the output *multiplied* by term 3. Two
consequences follow, and both were live on the real record:

* a forecast can be no better than ``gain x (representation + truncation)``, so
  chasing a target below that number is chasing something unreachable;
* with ``gain = 4.13`` and ``eps_enc = 0.18`` the predicted one-step error was
  0.1028 NDVI against an observed 0.1026 -- i.e. the "dynamics error" was
  essentially zero and the whole problem lay upstream of the operator.

Reporting the decomposition makes that visible immediately rather than after a
day of tuning the wrong stage. The rule of thumb it supports: **fix the largest
term, then re-measure**, because the binding constraint moves.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.budget")


def error_budget(
    extractor,
    weights: np.ndarray,
    frames: np.ndarray,
    valid_flat: np.ndarray,
    observed: np.ndarray,
    scale: float,
    gain: float,
    subspace=None,
    persistence: Optional[float] = None,
    units: str = "NDVI",
) -> Dict[str, float]:
    """
    Decompose the achievable forecast error into its stages, in physical units.

    :param extractor: the :class:`~dbwm.gp.state.GPStateExtractor`.
    :param weights: ``(T, r)`` encoded states.
    :param frames: ``(T, H, W)`` normalised frames.
    :param valid_flat: ``(n_pixels,)`` validity.
    :param observed: ``(T,)`` which dates carry a frame.
    :param scale: normalisation std, to report in physical units.
    :param gain: ``max_h ||S A_cal^h||_2``.
    :param subspace: the :class:`~dbwm.dynamics.subspace.LatentSubspace`, or ``None``.
    :param persistence: measured persistence RMSE at ``h = 1``, if available.
    :param units: modality unit symbol (``"NDVI"``, ``"K"``), printed in the
        header so the five numbers under it are never read on the wrong scale.
    :return: the budget, in physical units.
    """
    obs = np.asarray(observed, dtype=bool)
    flat = np.asarray(frames).reshape(frames.shape[0], -1)[:, valid_flat][obs]
    w = np.asarray(weights)[obs]

    recon = extractor.decode(w)[:, valid_flat]
    eps_enc = float(np.sqrt(np.mean((recon - flat) ** 2)) * scale)

    # Subspace truncation, measured the way it ACTUALLY enters the forecast.
    #
    # The forecast carries the origin's unmodelled component forward unchanged
    # (LatentSubspace.reconstruct_with_complement), so the truncation error at
    # the origin cancels exactly. What survives is only the *drift* of that
    # component over the horizon: ||complement(w_{t+h}) - complement(w_t)||.
    # Measuring the raw magnitude ||complement(w)|| instead -- which is what a
    # plain project/reconstruct round trip gives -- overstates the forecast
    # impact, on a slowly varying field by a large factor, and would send you
    # off to raise --subspace-energy when there is nothing there to win.
    eps_trunc = 0.0
    if subspace is not None:
        comp = subspace.complement(w)
        # Map to pixels with the LINEAR part of the decoder only. `decode` adds
        # the climatology offset, which is correct for a state but wrong for a
        # difference: the offset cancels between two decodes, so including it
        # here would report the magnitude of the climatology (~0.39 NDVI)
        # instead of the truncation, and would name it the binding term forever.
        phi_v = extractor.phi[valid_flat]
        static = float(np.sqrt(np.mean((comp @ phi_v.T) ** 2)) * scale)
        drift = comp[1:] - comp[:-1]
        eps_trunc = float(np.sqrt(np.mean((drift @ phi_v.T) ** 2)) * scale)

    upstream = float(np.sqrt(eps_enc**2 + eps_trunc**2))
    floor = gain * upstream

    budget = {
        "representation": eps_enc,
        "subspace_truncation": eps_trunc,
        "subspace_static_carried": None if subspace is None else static,
        "upstream_total": upstream,
        "operator_gain": float(gain),
        "achievable_floor_h1": floor,
    }
    if persistence is not None:
        budget["persistence_h1"] = float(persistence)

    logger.info("=" * 78)
    logger.info(
        "ERROR BUDGET at h = 1 (%s; the target must sit above this)", units
    )
    logger.info(
        "  representation   (y -> w -> y)          %.5f", eps_enc
    )
    if subspace is not None:
        logger.info(
            "  subspace drift   (1-step complement)     %.5f   "
            "(static part %.5f is carried forward, not paid)", eps_trunc, static,
        )
    else:
        logger.info(
            "  subspace drift                          %.5f   "
            "(no subspace projection)", eps_trunc,
        )
    logger.info(
        "  upstream total   (quadrature)           %.5f", upstream
    )
    logger.info(
        "  operator gain    max_h ||S A^h||        %.3fx", gain
    )
    logger.info(
        "  --> achievable floor  gain x upstream   %.5f", floor
    )
    if persistence is not None:
        logger.info(
            "  persistence reference (h = 1)           %.5f", persistence
        )
        if floor > persistence:
            logger.warning(
                "The achievable floor (%.5f) is ABOVE persistence (%.5f): the "
                "model cannot beat 'predict today's field' no matter how good "
                "the dynamics are, because the loss happens before the operator "
                "is applied. Reduce the largest term above first -- raise "
                "--eof-modes for representation, --subspace-energy for "
                "truncation, or the gain cap for amplification.",
                floor, persistence,
            )
    dominant = max(
        ("representation", eps_enc), ("subspace drift", eps_trunc),
        key=lambda kv: kv[1],
    )
    logger.info(
        "  binding term: %s. Fix that one, then re-measure -- the constraint "
        "moves once it is relieved.", dominant[0],
    )
    logger.info("=" * 78)
    return budget


def bias_attribution(
    extractor,
    weights: np.ndarray,
    frames: np.ndarray,
    valid_flat: np.ndarray,
    observed: np.ndarray,
    scale: float,
    forecast_bias: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Split the forecast bias into the stage that introduces it.

    ``RMSE^2 = bias^2 + ubRMSE^2`` says how much of the error is a whole-field
    offset; it does not say **where the offset comes from**, and the three
    candidates need completely different fixes. This walks the same field through
    each stage and differences them:

    ===================  ====================================================
    ``encode_bias``      ``mean(decode(w_t) - y_t)`` on observed dates. The
                         basis round trip alone. Nonzero here means the ridge
                         ``Lambda = Phi^T Phi + sigma_eps^2 I`` is shrinking
                         ``w`` toward zero -- i.e. toward the climatology
                         offset -- so the decode is pulled toward the training
                         mean field. On a falling seasonal limb that is a
                         *positive* bias, and it is present before any
                         dynamics run.
    ``forecast_bias``    the reported ``h``-step bias, if supplied.
    ``dynamics_bias``    ``forecast_bias[0] - encode_bias``. What the filter and
                         the operator add on top of the round trip: a
                         mis-specified weather emission pulling the state every
                         step, or an operator with a nonzero mean increment.
    ``drift_per_step``   slope of ``forecast_bias`` in ``h``. A static offset
                         has slope 0; a drifting operator does not.
    ===================  ====================================================

    Reading it: a large ``encode_bias`` is fixed upstream (more modes, a smaller
    ridge, a seasonal rather than annual climatology). A large ``dynamics_bias``
    is fixed at the sensor or the operator. Chasing the wrong one is the failure
    this exists to prevent.

    :param extractor: the :class:`~dbwm.gp.state.GPStateExtractor`.
    :param weights: ``(T, r)`` GP-solved weights.
    :param frames: ``(T, H, W)`` normalised frames.
    :param valid_flat: ``(n_pixels,)`` validity.
    :param observed: ``(T,)`` dates to average over -- pass the **test** mask to
        attribute the reported test bias.
    :param scale: normalisation std, for physical units.
    :param forecast_bias: the reported per-horizon bias, physical units.
    :return: the attribution, in physical units.
    """
    obs = np.asarray(observed, dtype=bool)
    valid = np.asarray(valid_flat, dtype=bool).reshape(-1)
    y = np.asarray(frames).reshape(frames.shape[0], -1)[obs][:, valid]
    recon = extractor.decode(np.asarray(weights)[obs])[:, valid]
    encode = float(np.mean(recon - y) * scale)

    out: Dict[str, float] = {"encode_bias": encode, "n_dates": int(obs.sum())}
    logger.info("=" * 78)
    logger.info("BIAS ATTRIBUTION (physical units, over %d dates)", int(obs.sum()))
    logger.info("  encode/decode round trip  %+.5f", encode)
    if forecast_bias is not None:
        fb = np.asarray(forecast_bias, dtype=float)
        finite = np.isfinite(fb)
        if finite.any():
            out["forecast_bias_h1"] = float(fb[finite][0])
            out["dynamics_bias"] = float(fb[finite][0] - encode)
            logger.info("  forecast at h = 1        %+.5f", out["forecast_bias_h1"])
            logger.info(
                "  --> added by filter+op   %+.5f", out["dynamics_bias"]
            )
        if finite.sum() >= 2:
            h = np.arange(1, fb.size + 1)[finite]
            slope = float(np.polyfit(h, fb[finite], 1)[0])
            out["drift_per_step"] = slope
            logger.info(
                "  drift per horizon step   %+.5f  (%s)", slope,
                "static offset, not a drifting operator" if abs(slope) < 1e-4
                else "the operator has a nonzero mean increment",
            )
        dominant = max(
            ("the basis round trip", abs(encode)),
            ("the filter / operator", abs(out.get("dynamics_bias", 0.0))),
            key=lambda kv: kv[1],
        )
        logger.info(
            "  binding: %s. Fix that stage -- work on the other one cannot move "
            "this term.", dominant[0],
        )
    logger.info("=" * 78)
    return out


def worst_pixel_attribution(
    errors_by_lead: Dict[int, np.ndarray],
    frames: np.ndarray,
    origins_by_lead: Dict[int, np.ndarray],
    valid_flat: np.ndarray,
    scale: float,
    repr_rms: Optional[np.ndarray] = None,
    n_worst: int = 5,
    units: str = "NDVI",
) -> Dict[str, object]:
    """
    Say whether the worst pixels are *fixable* or *unpredictable*, and which.

    A single number like "worst pixel 0.30" is not actionable, because two
    completely different situations produce it and only one of them can be
    engineered away:

    **Representation failure.** The basis cannot render that pixel -- a small
    parcel, a sharp boundary. The error is then essentially the *same at every
    lead*, because it is present the moment the field is encoded and no
    propagation is involved. This is fixable: more empirical modes, a larger
    ``r``, a finer grid.

    **An unpredicted event.** The ground genuinely changed between the origin and
    the target -- a harvest, an irrigation, a burn. The error appears at ``t+1``
    and persists, and it is accompanied by a large *observed* change at that
    pixel. No autonomous linear operator driven by NDVI history and weather can
    know the date a farmer cuts a specific plot, so this is **not** fixable by
    tuning the dynamics; the honest response is a wide predictive interval, not a
    better point forecast.

    The discriminator is the observed change ``|y_{t+h} - y_t|`` at the pixel. If
    the truth barely moved and the model is still wrong, that is representation.
    If the truth moved by as much as the error, that is an event.

    :param errors_by_lead: ``{lead: (n_origins, H, W)}`` forecast-minus-truth.
    :param frames: ``(T, H, W)`` normalised frames.
    :param origins_by_lead: ``{lead: (n_origins,)}`` origin indices per lead.
    :param valid_flat: ``(n_pixels,)`` validity.
    :param scale: normalisation std, for physical units.
    :param repr_rms: ``(n_pixels,)`` representation residual RMS, physical units.
    :param n_worst: how many pixels to report.
    :param units: modality label, used only in the reported text.
    :return: the per-pixel verdicts.
    """
    leads = sorted(errors_by_lead)
    if not leads:
        return {"pixels": []}
    shape = errors_by_lead[leads[0]].shape[1:]
    flat = np.asarray(frames).reshape(frames.shape[0], -1)

    # Rank pixels by RMSE at the FIRST lead, pooled over origins. Counted by hand
    # rather than with nanmean: masked pixels are NaN in every origin, and nanmean
    # warns "Mean of empty slice" for each of them.
    e1 = errors_by_lead[leads[0]]
    ok1 = np.isfinite(e1)
    n1 = ok1.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse1 = np.where(
            n1 > 0,
            np.sqrt(np.where(ok1, e1, 0.0).__pow__(2).sum(axis=0) / np.maximum(n1, 1)),
            np.nan,
        )
    finite = np.isfinite(rmse1)
    if not finite.any():
        return {"pixels": []}
    order = np.argsort(np.where(finite, rmse1, -np.inf).ravel())[::-1][:n_worst]

    rows = []
    for idx in order:
        row, col = np.unravel_index(int(idx), shape)
        per_lead, moved = [], []
        for lead in leads:
            e = errors_by_lead[lead][:, row, col]
            e = e[np.isfinite(e)]
            per_lead.append(float(np.sqrt(np.mean(e**2))) if e.size else float("nan"))
            org = np.asarray(origins_by_lead[lead])
            tgt = org + lead
            ok = tgt < flat.shape[0]
            if ok.any():
                delta = np.abs(flat[tgt[ok], idx] - flat[org[ok], idx]) * scale
                moved.append(float(np.mean(delta)))
            else:  # pragma: no cover - guarded by the caller
                moved.append(float("nan"))
        slope = (
            float(np.polyfit(leads, per_lead, 1)[0]) if len(leads) >= 2 else 0.0
        )
        # Static + the truth barely moved => the basis simply cannot render it.
        # Comparable to the observed motion => an event nothing could foresee.
        static = abs(slope) < 0.1 * max(per_lead[0], 1e-9)
        event = moved[0] > 0.5 * per_lead[0]
        rows.append({
            "row": int(row), "col": int(col),
            "rmse_by_lead": per_lead,
            "growth_per_lead": slope,
            "observed_change_by_lead": moved,
            "representation_rms": (
                None if repr_rms is None else float(np.asarray(repr_rms).ravel()[idx])
            ),
            "verdict": (
                "unpredicted event (the ground moved as much as the error)"
                if event else
                "representation (flat across leads, truth barely moved)"
                if static else
                "dynamics (error grows with lead)"
            ),
        })

    logger.info("=" * 78)
    logger.info("WORST-PIXEL ATTRIBUTION (top %d by t+%d RMSE)", len(rows), leads[0])
    for r in rows:
        logger.info(
            "  (r%d, c%d)  RMSE by lead %s  growth %+.4f/step  observed move "
            "%.4f  -> %s",
            r["row"], r["col"],
            np.array2string(np.asarray(r["rmse_by_lead"]), precision=4),
            r["growth_per_lead"], r["observed_change_by_lead"][0], r["verdict"],
        )
    n_event = sum(1 for r in rows if r["verdict"].startswith("unpredicted"))
    if n_event:
        logger.info(
            "  %d of %d worst pixels are unpredicted EVENTS. No autonomous linear "
            "operator driven by %s history and weather can anticipate the day a "
            "parcel is harvested or irrigated, so these are not reducible by "
            "tuning L, rho or the shrinkage. The correct response is the wide "
            "predictive interval they now carry, not a better point forecast. "
            "(For LST the same argument covers an irrigation or a canal turn: the "
            "surface cools by several kelvin within a day of wetting, and nothing "
            "in the point weather record announces which parcel was watered.)",
            n_event, len(rows), units,
        )
    n_repr = sum(1 for r in rows if r["verdict"].startswith("representation"))
    if n_repr:
        logger.info(
            "  %d of %d are REPRESENTATION failures -- flat across leads, so the "
            "error is already present at encoding time. These ARE reducible: "
            "raise --eof-modes or r.", n_repr, len(rows),
        )
    logger.info("=" * 78)
    return {"pixels": rows}


def persistence_rmse(
    frames: np.ndarray,
    valid_flat: np.ndarray,
    observed: np.ndarray,
    origins: Sequence[int],
    horizon: int,
    scale: float,
) -> np.ndarray:
    """
    RMSE of "predict today's field" at each horizon -- the bar any forecast must clear.

    On a daily, gap-filled NDVI product the field barely moves between
    consecutive dates, so this is a genuinely strong baseline rather than a straw
    man. It is also the honest way to decide whether an accuracy target is
    reachable: if persistence is already at 0.019 NDVI at ``t+6``, a target of
    0.01 there is asking the dynamics to halve the error of the best trivial
    forecast, which is a much stronger claim than it looks.

    :param frames: ``(T, H, W)`` normalised frames.
    :param valid_flat: ``(n_pixels,)`` validity.
    :param observed: ``(T,)`` observation mask.
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param scale: normalisation std.
    :return: ``(H,)`` RMSE in physical units.
    """
    flat = np.asarray(frames).reshape(frames.shape[0], -1)[:, valid_flat]
    obs = np.asarray(observed, dtype=bool)
    origins = np.asarray(origins)
    out = np.full(horizon, np.nan)
    for h in range(horizon):
        tgt = origins + h + 1
        ok = (tgt < flat.shape[0]) & obs[np.clip(tgt, 0, flat.shape[0] - 1)] & obs[origins]
        if ok.any():
            out[h] = float(
                np.sqrt(np.mean((flat[tgt[ok]] - flat[origins[ok]]) ** 2)) * scale
            )
    return out
