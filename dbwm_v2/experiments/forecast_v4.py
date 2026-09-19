"""
Operational forecast: give it a date, get ``t+1..t+6``.

``infer_v4`` scores a checkpoint over the test split -- a retrospective exercise
on a fixed archive. This script answers the other question, the one an end user
actually asks:

    "It is 2026-05-02. I have the last seven days of 30 m NDVI and the weather
    record. What does the field look like on the 3rd through the 8th?"

That date is past the end of the archive the model was trained on, which is the
normal case for a forecast and the reason this needs its own entry point. Nothing
is refitted: the basis ``Psi``, the empirical block, the memory kernel, the
horizon family, the weather emission, the normalisation and the weather
climatology all come from the checkpoint.

What it does, and the three places it refuses rather than guesses
-----------------------------------------------------------------
1. Build a **local** calendar ``[date - (context-1), date + H]`` and load whatever
   rasters exist in it. Missing days are marked unobserved; the lifted filter
   predicts through them and Prop. 2.9 fixed-lag smoothing retro-corrects them.
2. Normalise with the **checkpoint's** ``(mean, std)``. Re-deriving them from a
   dozen frames would put the state on a different scale from the operator.
3. Weather through :func:`~dbwm.data.weather.weather_for_window`, with the
   training climatology and both training scale vectors.
4. Filter up to and including the origin, then forecast the horizon with a
   weather update at every step.

It refuses, with a message naming the fix, when: the origin has fewer than ``L``
consecutive observed days behind it (the delay line would be part prediction);
the pixel grid does not match the checkpoint's (the EOF block is transductive, so
a different ``--max-pixels`` silently changes the basis); or the checkpoint
predates the fields this path needs.

Receiving measurements after the fact
--------------------------------------
``--assimilate-future`` turns on the correction loop. If a frame for ``t+2``
exists by the time this runs, it is assimilated and ``t+3..t+6`` are relaunched
from the corrected state. Two trajectories are always produced and always kept
apart: ``forecast`` (what was predicted for each step *before* that step's own
frame was seen -- the only one scoreable as forecast skill) and ``corrected``
(the filtered estimate). Only the first is reported as a forecast metric.

Usage
-----
    python -m experiments.forecast_v4 --ckpt _v4_ckpt/dbwm_v4_v4.pkl \\
        --date 2026-05-02 --ndvi-dir <dir> --weather-csv <csv> --export-geotiff

    python -m experiments.forecast_v4 --ckpt <pkl> --date 2026-05-02 \\
        --ndvi-dir <dir> --weather-csv <csv> --assimilate-future
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Dict, List, Optional

import numpy as np

from dbwm.platform import ensure_working_backend, preflight

_PREFLIGHT = preflight()

import jax.numpy as jnp  # noqa: E402

from dbwm.data.ndvi_dataset import load_calendar_dataset  # noqa: E402
from dbwm.data.weather import weather_for_window  # noqa: E402
from dbwm.evaluation import forecast_figures  # noqa: E402
from dbwm.evaluation.geotiff_export import export_forecast_maps  # noqa: E402
from dbwm.evaluation.metrics import field_error_metrics  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.inference.lifted_kalman import (  # noqa: E402
    LiftedSystem, filter_state_at, forecast_from_origin,
    forecast_with_assimilation,
)
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.training.gp_trainer import SpatialGPModule  # noqa: E402
from experiments._v4_common import config_from_dict, load_checkpoint  # noqa: E402

logger = configure_logger(level="INFO", name="dbwm.forecast_v4")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Forecast t+1..t+H from a date, using a trained DB-WM v4 model."
    )
    p.add_argument("--ckpt", required=True, help="path to a trained v4 .pkl")
    p.add_argument(
        "--date", required=True, metavar="YYYY-MM-DD",
        help="the forecast ORIGIN: the last day whose raster is known. Outputs "
             "are t+1..t+H after it, so --date 2026-05-02 forecasts the 3rd "
             "through the 8th.",
    )
    p.add_argument("--ndvi-dir", default=None)
    p.add_argument("--lst-dir", default=None)
    p.add_argument("--weather-csv", default=None)
    p.add_argument("--results-dir", default="./_v4_forecast")
    p.add_argument("--horizon", type=int, default=None,
                   help="H (default: the checkpoint's)")
    p.add_argument(
        "--context-days", type=int, default=7,
        help="how many days of raster history to load before the origin "
             "(default 7). Raised automatically to the checkpoint's memory order "
             "L if that is larger, since the lift conditions on w_t..w_{t-L+1}.",
    )
    p.add_argument("--forecast-mode", default="recursive",
                   choices=["recursive", "direct"])
    p.add_argument(
        "--assimilate-future", action="store_true",
        help="fold in any horizon frames that already exist, correcting the "
             "remaining steps. Each step's own forecast is still recorded before "
             "its frame is used, so the reported forecast metrics stay honest.",
    )
    p.add_argument("--no-weather-update", action="store_true",
                   help="ablate the Rs/Ta/VPD correction; B_p p_t stays")
    p.add_argument("--export-geotiff", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--max-pixels", type=int, default=250_000,
                   help="must match the value the checkpoint was trained with")
    p.add_argument("--ram-budget-gb", type=float, default=8.0)
    p.add_argument("--require-accelerator", action="store_true")
    return p.parse_args()


def _require(ckpt: Dict[str, object], key: str, why: str):
    """
    Fetch a checkpoint field this path cannot work without.

    :param ckpt: the loaded checkpoint.
    :param key: field name.
    :param why: what it is needed for, quoted in the error.
    :return: the value.
    :raises SystemExit: if absent.
    """
    val = ckpt.get(key)
    if val is None:
        raise SystemExit(
            "This checkpoint has no '{}', which the operational forecast needs "
            "to {}. It predates experiments/forecast_v4.py; retrain with "
            "`python -m experiments.train_v4 ...` to record it.".format(key, why)
        )
    return val


def load_window(args, cfg, ckpt, origin: dt.date, context: int, horizon: int):
    """
    Load the local raster window and normalise it with the checkpoint's statistics.

    :param args: parsed arguments.
    :param cfg: the checkpoint's config.
    :param ckpt: the loaded checkpoint.
    :param origin: the forecast origin date.
    :param context: days of history to load, including the origin.
    :param horizon: ``H``.
    :return: the :class:`~dbwm.data.ndvi_dataset.CalendarDataset`.
    """
    start = origin - dt.timedelta(days=context - 1)
    end = origin + dt.timedelta(days=horizon)
    data_dir = cfg.data_dir()
    logger.info(
        "Window %s .. %s (%d context days + %d horizon days), reading %s from %s.",
        start, end, context, horizon, cfg.data.modality.upper(), data_dir,
    )
    ds = load_calendar_dataset(
        data_dir, start, end, end + dt.timedelta(days=1),
        nodata=cfg.data.nodata_value,
        normalize=False,               # the checkpoint's statistics are applied below
        max_pixels=args.max_pixels,
        ram_budget_gb=args.ram_budget_gb,
    )

    mean = float(np.asarray(_require(ckpt, "norm_mean", "de-normalise the output")).reshape(-1)[0])
    std = float(np.asarray(_require(ckpt, "norm_std", "normalise the input")).reshape(-1)[0])
    frames = (np.asarray(ds.frames, dtype=np.float32) - mean) / std
    frames = np.where(ds.valid_mask, frames, 0.0).astype(np.float32)
    frames[~ds.observed] = 0.0
    ds.frames = frames
    ds.mean = np.array([mean], dtype=np.float32)
    ds.std = np.array([std], dtype=np.float32)
    logger.info(
        "Applied the TRAINING normalisation (mean %.4f, std %.4f) rather than "
        "re-deriving it from %d frames.", mean, std, int(ds.observed.sum()),
    )

    expected = ckpt.get("n_pixels")
    if expected is not None and int(expected) != int(ds.coords.shape[0]):
        raise SystemExit(
            "Pixel-grid mismatch: this window is {} px ({}x{}) but the checkpoint "
            "was fitted on {} px{}. The empirical (EOF) basis block is defined "
            "per pixel, not as a function of coordinates, so it cannot be "
            "evaluated on a different grid. Re-run with the --max-pixels the "
            "checkpoint used (it was trained with the same flag).".format(
                ds.coords.shape[0], ds.shape[0], ds.shape[1], expected,
                "" if ckpt.get("grid_shape") is None
                else " ({}x{})".format(*ckpt["grid_shape"]),
            )
        )
    return ds


def check_origin(ds, origin: dt.date, order: int) -> int:
    """
    Verify the origin can actually seed the memory lift, and return its index.

    :param ds: the window dataset.
    :param origin: the requested origin date.
    :param order: memory order ``L``.
    :return: the calendar index of the origin.
    :raises SystemExit: if the origin or its history is unusable.
    """
    index = {d: i for i, d in enumerate(ds.dates)}
    if origin not in index:  # pragma: no cover - the window is built around it
        raise SystemExit("{} is not in the loaded window.".format(origin))
    t = index[origin]
    if not ds.observed[t]:
        raise SystemExit(
            "No {} raster for {}. The origin is the last day whose field is "
            "KNOWN; forecasting from a date with no image would start the filter "
            "from a fabricated state. Pick the most recent date that has a "
            "raster{}.".format(
                "NDVI/LST", origin,
                "" if not ds.observed.any() else
                " -- the latest in this window is {}".format(
                    ds.dates[int(np.nonzero(ds.observed)[0][-1])]
                ),
            )
        )
    lo = t - order + 1
    if lo < 0 or not ds.observed[lo : t + 1].all():
        missing = [str(ds.dates[j]) for j in range(max(lo, 0), t + 1)
                   if not ds.observed[j]]
        raise SystemExit(
            "{} does not have {} consecutive observed days behind it (memory "
            "order L = {}); missing {}. Part of the delay line would be the "
            "filter's own predictions, which is not what a conditioned forecast "
            "means. Widen --context-days only helps if the rasters exist; "
            "otherwise pick a later origin.".format(
                origin, order, order,
                missing or "history before the start of the window",
            )
        )
    logger.info(
        "Origin %s (index %d): %d observed days behind it, %d of the %d horizon "
        "days already have a raster.",
        origin, t, order,
        int(ds.observed[t + 1 :].sum()), int(ds.n_steps - t - 1),
    )
    return t


def main():
    """Load a checkpoint, forecast from one date, score what can be scored."""
    args = build_args()
    backend = ensure_working_backend(allow_cpu_fallback=not args.require_accelerator)
    logger.info("Backend: %s", backend["backend"])

    try:
        origin = dt.date.fromisoformat(args.date)
    except ValueError as exc:
        raise SystemExit("--date {!r} is not YYYY-MM-DD ({})".format(args.date, exc))

    ckpt = load_checkpoint(args.ckpt)
    cfg = config_from_dict(ckpt["config"])
    if args.ndvi_dir:
        cfg.data.ndvi_dir = args.ndvi_dir
    if args.lst_dir:
        cfg.data.lst_dir = args.lst_dir
    if args.weather_csv:
        cfg.weather.csv_path = args.weather_csv
    horizon = args.horizon or cfg.horizons.horizon
    order = cfg.memory.order
    context = max(args.context_days, order)
    if context > args.context_days:
        logger.info(
            "Raised --context-days %d -> %d: the memory lift conditions on "
            "w_t..w_{t-L+1} with L = %d.", args.context_days, context, order,
        )
    diag = ckpt["diagnostics"]
    logger.info(
        "Checkpoint %s: r = %d (+%d EOF modes), L = %d, k = %d | reconstruction "
        "R2 %.4f | blend s = %s | max gain %.3f",
        args.ckpt, cfg.basis.r, diag.get("eof_modes", 0), order, diag["k"],
        diag.get("reconstruction_r2", float("nan")), diag.get("persistence_blend"),
        diag.get("max_forecast_gain", float("nan")),
    )

    ds = load_window(args, cfg, ckpt, origin, context, horizon)
    t0 = check_origin(ds, origin, order)

    # ---- Weather, with the TRAINING climatology and scales ---------------- #
    wt = weather_for_window(
        ds.dates, cfg.weather.csv_path,
        _require(ckpt, "weather_climatology", "build weather anomalies"),
        _require(ckpt, "weather_forcing_scale", "scale the precipitation input"),
        _require(ckpt, "weather_measurement_scale", "scale the Rs/Ta/VPD anomalies"),
        precip_lags=int(ckpt.get("weather_precip_lags", cfg.weather.precip_lags)),
        measurement_cols=list(ckpt.get("measurement_names", cfg.weather.measurement_cols)),
    )
    forcing = np.asarray(wt.forcing, dtype=np.float64)
    measurement = np.asarray(wt.measurement, dtype=np.float64)

    # ---- Rebuild the EXACT decoder the operator was identified against ---- #
    params = ckpt["params"]
    sigma_eps2 = float(ckpt["sigma_eps2"])
    model = SpatialGPModule(cfg.basis)
    extractor = build_extractor(
        lambda c: model.apply(params, jnp.asarray(c), method=model.features),
        ds.coords, sigma_eps2, ds.static_mask,
        extra_features=ckpt.get("extra_features"),
        offset=ckpt.get("basis_offset"),
    )
    gp = extractor.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    w, w_cov = gp["weights"], gp["covariances"]

    subspace = ckpt.get("subspace")
    w_dyn = subspace.project(w) if subspace is not None else w
    cov_dyn = subspace.project_covariance(w_cov) if subspace is not None else w_cov

    op, family, em = ckpt["op"], ckpt["family"], ckpt["emission"]
    use_wx = em.usable and not args.no_weather_update
    if not em.usable and not args.no_weather_update:
        logger.warning(
            "The weather emission was marked unusable at training time (no "
            "channel with positive held-out R^2), so the observer runs without "
            "it. Only B_p p_t carries weather into this forecast."
        )
    system = LiftedSystem(
        op=op, b_p=np.asarray(ckpt["b_p"]), q=np.asarray(ckpt["q"]),
        c_w=em.c if use_wx else None,
        r_w=em.r_cov if use_wx else None,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )

    # ---- Filter up to the origin, then forecast --------------------------- #
    # `upto=t0` is what keeps this out-of-sample: every frame at or before the
    # origin is assimilated, and nothing after it can reach the forecast branch.
    state, cov = filter_state_at(
        system, w_dyn, ds.observed, t0, forcing, measurement, cov_dyn, sigma_eps2
    )
    f_branch = forcing[t0 : t0 + horizon]
    if f_branch.shape[0] < horizon:  # pragma: no cover - window is sized for it
        f_branch = np.vstack(
            [f_branch, np.zeros((horizon - f_branch.shape[0], f_branch.shape[1]))]
        )
    w_branch = measurement[t0 + 1 : t0 + 1 + horizon]
    if args.no_weather_update:
        w_branch = None

    future = np.arange(t0 + 1, t0 + 1 + horizon)
    have_future = ds.observed[future]
    if args.assimilate_future and have_future.any():
        res = forecast_with_assimilation(
            system, state, cov, horizon, f_branch, w_branch,
            w_obs=w_dyn[future], observed=have_future, obs_cov=cov_dyn[future],
            sigma_eps2=sigma_eps2,
        )
        mean_fc, cov_fc = res["mean_forecast"], res["cov_forecast"]
        mean_corr, cov_corr = res["mean"], res["cov"]
        assimilated = res["assimilated"]
    else:
        if args.assimilate_future:
            logger.info(
                "--assimilate-future had nothing to fold in: none of the %d "
                "horizon dates has a raster yet.", horizon,
            )
        fc = forecast_from_origin(
            system, state, cov, horizon, f_branch, w_branch, family,
            args.forecast_mode,
        )
        mean_fc, cov_fc = fc.mean, fc.cov
        mean_corr, cov_corr = fc.mean, fc.cov
        assimilated = np.zeros(horizon, dtype=bool)

    # ---- Score whatever has truth ---------------------------------------- #
    os.makedirs(args.results_dir, exist_ok=True)
    repr_var = ckpt.get("repr_variance")
    if repr_var is None:
        logger.warning(
            "This checkpoint carries no representation variance: the "
            "predictive interval omits its dominant term and understates the "
            "forecast's real uncertainty. Retrain to record it."
        )
    scored = _score(mean_fc, ds, extractor, t0, horizon, subspace, w)
    _log_scores(ds, t0, horizon, scored, assimilated)

    # Shaped like one origin of `rolling_forecast_multi` so the figure and
    # GeoTIFF writers -- the ones every other entry point uses -- apply unchanged.
    packed = {
        "origins": np.array([t0]),
        "mean": mean_fc[None],
        "mean_prior": mean_fc[None],
        "cov_trace": np.trace(cov_fc, axis1=1, axis2=2)[None],
        "trace_reduction": np.zeros((1, horizon)),
        "valid": np.ones((1, horizon), dtype=bool),
        "mode": args.forecast_mode,
    }
    figures: List[str] = []
    if not args.no_plots:
        fig_dir = os.path.join(args.results_dir, "figures")
        rendered = forecast_figures.render_horizon_figures(
            fig_dir, cfg.name, packed,
            np.array([t0]), ds, extractor, subspace=subspace, w_origin=w,
            horizon=horizon, units=cfg.data.modality.upper(),
            variant=args.forecast_mode, plot_origins=[t0], per_pixel=False,
            repr_var=repr_var,
        )
        figures = rendered["figures"]
        logger.info("Wrote %d figures to %s", len(figures), os.path.abspath(fig_dir))

    exported: List[str] = []
    if args.export_geotiff:
        exported = export_forecast_maps(
            os.path.join(args.results_dir, "geotiff"), packed, np.array([t0]),
            ds.dates, extractor, ds, subspace=subspace, horizon=horizon,
            max_origins=None, variant=args.forecast_mode, w_origin=w,
            sigma_from=packed["cov_trace"], repr_var=repr_var,
        )

    payload = {
        "checkpoint": os.path.abspath(args.ckpt),
        "origin_date": origin.isoformat(),
        "horizon": int(horizon),
        "context_days": int(context),
        "memory_order": int(order),
        "forecast_mode": args.forecast_mode,
        "weather_update": bool(use_wx),
        "window": [str(ds.dates[0]), str(ds.dates[-1])],
        "target_dates": [str(ds.dates[t0 + h + 1]) for h in range(horizon)],
        "truth_available": [bool(v) for v in have_future],
        "assimilated": [bool(v) for v in assimilated],
        "metrics": {k: np.asarray(v).tolist() for k, v in scored.items()},
        "figures": [os.path.abspath(p) for p in figures],
        "geotiffs": [os.path.abspath(p) for p in exported],
    }
    if args.assimilate_future and assimilated.any():
        payload["corrected_metrics"] = {
            k: np.asarray(v).tolist() for k, v in
            _score(mean_corr, ds, extractor, t0, horizon, subspace, w).items()
        }
        payload["corrected_metrics_note"] = (
            "FILTERED estimates, not forecasts: each step here has seen its own "
            "frame where one existed. Never report these as forecast skill."
        )
    path = os.path.join(args.results_dir, "forecast.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    logger.info("Results written to %s", os.path.abspath(args.results_dir))


def _score(mean, ds, extractor, t0, horizon, subspace, w) -> Dict[str, np.ndarray]:
    """
    Score the horizon in physical units, skipping steps with no truth.

    :param mean: ``(H, k)`` latent forecasts.
    :param ds: the window dataset.
    :param extractor: the GP state extractor.
    :param t0: origin index.
    :param horizon: ``H``.
    :param subspace: the latent subspace, or ``None``.
    :param w: ``(T, r)`` GP-solved weights.
    :return: ``{metric: (H,)}`` with NaN where no raster exists.
    """
    mask_flat = ds.common_mask()
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]
    scale = float(ds.std[0])
    keys = ("ubrmse", "mae", "rmse", "bias")
    out = {k: np.full(horizon, np.nan) for k in keys}
    for h in range(horizon):
        tgt = t0 + h + 1
        if tgt >= ds.n_steps or not ds.observed[tgt]:
            continue
        wv = np.asarray(mean)[h]
        if subspace is not None:
            wv = subspace.reconstruct_with_complement(wv[None], w[t0][None])[0]
        pred = extractor.decode(wv)[mask_flat]
        m = field_error_metrics(pred, flat[tgt], scale=scale)
        for k in keys:
            out[k][h] = m[k]
    return out


def _log_scores(ds, t0, horizon, scored, assimilated) -> None:
    """
    Print the per-step table, saying plainly which rows are forecasts of the unknown.

    :param ds: the window dataset.
    :param t0: origin index.
    :param horizon: ``H``.
    :param scored: output of :func:`_score`.
    :param assimilated: ``(H,)`` which steps folded in their own frame.
    """
    logger.info("=" * 78)
    logger.info("FORECAST FROM %s", ds.dates[t0])
    logger.info("  step   date         ubRMSE      MAE     RMSE      bias   status")
    n_scored = 0
    for h in range(horizon):
        tgt = t0 + h + 1
        date = ds.dates[tgt] if tgt < ds.n_steps else "-"
        if np.isfinite(scored["ubrmse"][h]):
            n_scored += 1
            status = "scored" + (" (frame also assimilated)" if assimilated[h] else "")
            logger.info(
                "  t+%-3d  %s   %7.4f  %7.4f  %7.4f  %+8.4f   %s",
                h + 1, date, scored["ubrmse"][h], scored["mae"][h],
                scored["rmse"][h], scored["bias"][h], status,
            )
        else:
            logger.info(
                "  t+%-3d  %s   %7s  %7s  %7s  %8s   FORECAST (no raster yet)",
                h + 1, date, "--", "--", "--", "--",
            )
    logger.info("-" * 78)
    if n_scored == 0:
        logger.info(
            "None of the %d horizon days has a raster yet, so nothing can be "
            "scored -- which is exactly what a genuine forward forecast looks "
            "like. The maps and GeoTIFFs are still written.", horizon,
        )
    else:
        logger.info(
            "%d of %d steps had a raster to score against; the rest are "
            "unverifiable forward predictions.", n_scored, horizon,
        )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
