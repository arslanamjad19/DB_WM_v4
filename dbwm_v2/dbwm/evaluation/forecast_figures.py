"""
Decode a latent forecast to pixels and render the per-date figure set.

Three entry points need exactly the same product -- ``run_ndvi_v4`` (the full
pipeline), ``infer_v4`` (score a checkpoint on the test split) and
``forecast_v4`` (operational forecast from an arbitrary date). Before this module
each built its own triptych call, and they had already drifted: one passed the
valid mask and one did not, one de-normalised and one plotted z-scores under a
label that said NDVI. Decoding is subtle enough -- subspace lift, unmodelled
complement, normalisation, mask -- that having it in one place is the only way the
three stay comparable.

What gets rendered, and why each piece is there
-----------------------------------------------
**One triptych per (origin date, lead).** Named by the date it was issued from, so
``dbwm_ndvi_triptych_2025-06-14_t+3.png`` is unambiguous without opening it. The
panels carry the *target* date too, because "t+3 from the 14th" and "the 17th" are
the same statement and a reader should not have to do the arithmetic.

**A predictive-interval panel.** Drawn from the same
:func:`~dbwm.evaluation.geotiff_export.uncertainty_band` that fills band 2 of the
exported GeoTIFF, so the figure and the raster cannot disagree.

**Per-pixel RMSE / MAE maps per lead, aggregated over every origin.** A single
date cannot separate RMSE from MAE at a pixel -- both are ``|e|`` -- so the
question "which pixel is worst" only becomes well posed once several hundred
origins are pooled. That map is also the one that distinguishes a registration
problem (one hot boundary) from a dynamics problem (diffuse error).

Forecasts past the end of the archive
--------------------------------------
A genuine forward forecast has no truth. Those steps are still decoded, plotted
and exported -- they are the product -- but the observed and error panels are
omitted rather than drawn against a fabricated frame, and they are excluded from
every metric.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.evaluation import v4_plots
from dbwm.evaluation.geotiff_export import uncertainty_band
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.figures")


def decode_forecast(
    weights: np.ndarray,
    extractor,
    ds,
    subspace=None,
    w_origin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Lift, decode and de-normalise latent forecasts into physical-unit maps.

    The subspace lift carries the origin's **unmodelled** component forward
    unchanged rather than asserting it collapses to zero: the operator says
    nothing about the orthogonal complement, so persisting it is the honest
    default, and ``reconstruct()`` alone would charge the full truncation error to
    every horizon step. This must match what the scorer does or the picture and
    the number stop describing the same forecast.

    :param weights: ``(n, k)`` latent forecasts (subspace coordinates if reduced).
    :param extractor: the :class:`~dbwm.gp.state.GPStateExtractor`.
    :param ds: the calendar dataset, for ``(mean, std)`` and the frame shape.
    :param subspace: the :class:`~dbwm.dynamics.subspace.LatentSubspace`, or ``None``.
    :param w_origin: ``(n, r)`` full-``r`` origin states, one per row of
        ``weights``, supplying the complement to carry forward.
    :return: ``(n, H, W)`` fields in physical units.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    if subspace is not None:
        w = (
            subspace.reconstruct_with_complement(w, np.atleast_2d(w_origin))
            if w_origin is not None else subspace.reconstruct(w)
        )
    scale, offset = float(ds.std[0]), float(ds.mean[0])
    pix = extractor.decode(w) * scale + offset
    return pix.reshape(-1, *ds.shape)


def _mask_flat(ds) -> np.ndarray:
    """
    The pixel mask used for every score and every figure.

    :param ds: the calendar dataset.
    :return: ``(n_pixels,)`` bool.
    """
    return ds.common_mask()


def error_stack(
    forecast: Dict[str, np.ndarray],
    origins: Sequence[int],
    ds,
    extractor,
    lead: int,
    subspace=None,
    w_origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast-minus-truth at one lead, for every origin that has a truth frame.

    :param forecast: one variant's output from ``rolling_forecast_multi``.
    :param origins: forecast origin indices into the calendar.
    :param ds: the calendar dataset.
    :param extractor: the GP state extractor.
    :param lead: ``h``, one-based.
    :param subspace: the latent subspace, or ``None``.
    :param w_origin: ``(T, r)`` GP-solved weights.
    :return: ``(n_scored, H, W)`` errors with NaN outside the mask, and the
             ``(n_scored,)`` origin indices they came from.
    """
    origins = np.asarray(origins, dtype=int)
    h = lead - 1
    ok = np.asarray(forecast["valid"], dtype=bool)[:, h]
    tgt = origins + lead
    ok = ok & (tgt < ds.n_steps)
    ok[ok] &= ds.observed[tgt[ok]]
    if not ok.any():
        return np.zeros((0, *ds.shape)), np.zeros(0, dtype=int)

    org = origins[ok]
    pred = decode_forecast(
        np.asarray(forecast["mean"])[ok, h], extractor, ds, subspace,
        None if w_origin is None else np.asarray(w_origin)[org],
    )
    scale, offset = float(ds.std[0]), float(ds.mean[0])
    truth = np.asarray(ds.frames)[tgt[ok]] * scale + offset
    err = pred - truth
    mask = _mask_flat(ds).reshape(ds.shape)
    return np.where(mask[None], err, np.nan), org


def render_horizon_figures(
    fig_dir: str,
    name: str,
    forecast: Dict[str, np.ndarray],
    origins: Sequence[int],
    ds,
    extractor,
    subspace=None,
    w_origin: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    units: str = "NDVI",
    variant: str = "recursive",
    plot_origins: Optional[Sequence[int]] = None,
    per_pixel: bool = True,
    ci_level: float = 0.95,
    repr_var: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Render the ``t+1..t+H`` triptychs for chosen origins, plus per-pixel error maps.

    :param fig_dir: directory to write into.
    :param name: filename stem (the experiment name).
    :param forecast: one variant's output from ``rolling_forecast_multi``.
    :param origins: every origin the forecast covers, aligned to its first axis.
    :param ds: the calendar dataset.
    :param extractor: the GP state extractor.
    :param subspace: the latent subspace, or ``None``.
    :param w_origin: ``(T, r)`` GP-solved weights, for the complement carry.
    :param horizon: how many leads to render (default: all).
    :param units: modality label; fixes the colour range via
        :func:`~dbwm.evaluation.v4_plots.value_range`.
    :param variant: forecast variant label, printed on each figure.
    :param plot_origins: calendar indices to draw triptychs for. ``None`` picks the
        latest origin whose whole horizon is observed, so every panel shows real
        data instead of a gap.
    :param per_pixel: also write the aggregated per-pixel RMSE/MAE/bias maps.
    :param ci_level: nominal coverage for the interval panel.
    :param repr_var: ``(n_pixels,)`` representation variance, physical units.
        Without it the interval panel omits the dominant variance term and the
        reported coverage comes out far below nominal for that reason alone.
    :return: dict with ``figures`` (paths) and ``pixel_metrics``
             (``{lead: {metric: summary}}``).
    """
    os.makedirs(fig_dir, exist_ok=True)
    origins = np.asarray(origins, dtype=int)
    mean = np.asarray(forecast["mean"])
    valid = np.asarray(forecast["valid"], dtype=bool)
    hmax = mean.shape[1] if horizon is None else min(horizon, mean.shape[1])
    trace = forecast.get("cov_trace")
    scale = float(ds.std[0])
    mask2d = _mask_flat(ds).reshape(ds.shape)

    wanted = _select_origins(plot_origins, origins, valid, ds, hmax)
    figures: List[str] = []
    pixel_metrics: Dict[int, Dict[str, float]] = {}

    for h in range(hmax):
        lead = h + 1
        # --- per-pixel aggregation across every scorable origin at this lead --- #
        if per_pixel:
            errs, org = error_stack(
                forecast, origins, ds, extractor, lead, subspace, w_origin
            )
            if errs.shape[0]:
                maps = v4_plots.per_pixel_error_maps(errs)
                p = v4_plots.plot_pixel_error_maps(
                    maps,
                    os.path.join(fig_dir, "{}_pixel_error_t+{}.png".format(name, lead)),
                    lead, units=units, n_origins=int(errs.shape[0]),
                )
                if p:
                    figures.append(p)
                ext = v4_plots.pixel_extremes(maps["rmse"], maps["mae"])
                pixel_metrics[lead] = {
                    "n_origins": int(errs.shape[0]),
                    "field_rmse": float(np.nanmean(maps["rmse"])),
                    "field_mae": float(np.nanmean(maps["mae"])),
                    **{
                        "{}_{}".format(k, f): v[f]
                        for k, v in ext.items() for f in ("row", "col", "value")
                    },
                }

        # --- one triptych per requested origin ---------------------------- #
        for i in wanted:
            o = int(origins[i])
            if not valid[i, h]:
                continue
            tgt = o + lead
            beyond = tgt >= ds.n_steps
            target_date = None if beyond else ds.dates[tgt]
            truth = None
            if not beyond and ds.observed[tgt]:
                truth = np.asarray(ds.frames)[tgt] * scale + float(ds.mean[0])
            pred = decode_forecast(
                mean[i, h], extractor, ds, subspace,
                None if w_origin is None else np.asarray(w_origin)[o],
            )[0]
            sigma = None
            if trace is not None:
                sigma = uncertainty_band(
                    np.asarray(trace)[i, h], extractor, subspace, scale, repr_var
                ).reshape(ds.shape)
            figures.append(v4_plots.plot_forecast_triptych(
                truth, pred,
                os.path.join(
                    fig_dir,
                    "{}_triptych_{}_t+{}.png".format(name, ds.dates[o].isoformat(), lead),
                ),
                lead=lead, valid_mask=mask2d, units=units,
                target_date=target_date, origin_date=ds.dates[o],
                sigma=sigma, ci_level=ci_level, variant=variant,
            ))

    if pixel_metrics:
        worst = max(pixel_metrics.items(), key=lambda kv: kv[1]["field_rmse"])
        logger.info(
            "Per-pixel error maps written for %d lead(s); worst lead t+%d "
            "(field-mean per-pixel RMSE %.4f, hottest pixel r%d c%d at %.4f).",
            len(pixel_metrics), worst[0], worst[1]["field_rmse"],
            worst[1]["max_rmse_row"], worst[1]["max_rmse_col"],
            worst[1]["max_rmse_value"],
        )
    return {"figures": [p for p in figures if p], "pixel_metrics": pixel_metrics}


def _select_origins(
    plot_origins: Optional[Sequence[int]],
    origins: np.ndarray,
    valid: np.ndarray,
    ds,
    horizon: int,
) -> List[int]:
    """
    Turn requested calendar dates/indices into positions in the forecast array.

    With nothing requested, the default is the **latest** origin whose whole
    horizon is observed. That is deliberate: an origin picked at random may have a
    gap at ``t+4``, and a figure set where two of six panels are blank reads as a
    broken model rather than as a missing acquisition.

    :param plot_origins: calendar indices to draw, or ``None``.
    :param origins: the forecast's origin indices.
    :param valid: ``(n_origins, H)`` validity.
    :param ds: the calendar dataset.
    :param horizon: number of leads being rendered.
    :return: positions into the forecast arrays.
    """
    if plot_origins is not None:
        pos = {int(o): i for i, o in enumerate(origins)}
        out = []
        for o in plot_origins:
            if int(o) in pos:
                out.append(pos[int(o)])
            else:
                logger.warning(
                    "Requested plot origin %s is not among the %d scored origins; "
                    "skipped.",
                    ds.dates[int(o)] if 0 <= int(o) < ds.n_steps else o, origins.size,
                )
        return out
    for i in range(len(origins) - 1, -1, -1):
        o = int(origins[i])
        tgt = o + np.arange(1, horizon + 1)
        if (
            valid[i, :horizon].all()
            and (tgt < ds.n_steps).all()
            and ds.observed[tgt].all()
        ):
            return [i]
    return [len(origins) - 1] if len(origins) else []


def resolve_plot_dates(
    wanted: Sequence[str], ds, origins: Sequence[int]
) -> List[int]:
    """
    Map ``YYYY-MM-DD`` strings onto forecast-origin calendar indices.

    :param wanted: requested origin dates.
    :param ds: the calendar dataset.
    :param origins: the origins actually scored.
    :return: calendar indices, in the order given.
    :raises SystemExit: on an unparseable date.
    """
    index = {d: i for i, d in enumerate(ds.dates)}
    scored = set(int(o) for o in origins)
    out: List[int] = []
    for raw in wanted:
        try:
            day = dt.date.fromisoformat(str(raw))
        except ValueError as exc:
            raise SystemExit(
                "--plot-date {!r} is not a YYYY-MM-DD date ({})".format(raw, exc)
            )
        if day not in index:
            raise SystemExit(
                "{} is outside the archive calendar {} .. {}.".format(
                    day, ds.dates[0], ds.dates[-1]
                )
            )
        t = index[day]
        if t not in scored:
            raise SystemExit(
                "{} is not a scored forecast origin. Origins are the observed "
                "test-split dates with L-1 observed days behind them and a full "
                "horizon ahead; the nearest scored ones are {}.".format(
                    day,
                    ", ".join(
                        str(ds.dates[o])
                        for o in sorted(scored, key=lambda o: abs(o - t))[:3]
                    ),
                )
            )
        out.append(t)
    return out
