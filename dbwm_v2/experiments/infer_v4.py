"""
Forecast ``t+1..t+H`` from a trained DB-WM v4 checkpoint.

Mirrors ``experiments/run_inference.py``: load a checkpoint, run the test-split
predict-then-correct loop, and write the same product set -- triptychs, the error
curve, the Koopman spectrum -- plus the two things the v4 model adds, namely a
per-horizon error breakdown and georeferenced forecast rasters.

The weather enters at both places the framework requires, and the distinction
matters for reading the outputs:

* **precipitation** drives the state through ``B_p p_t``, inside the predict step,
  so it shapes the forecast itself;
* **Rs, Ta, VPD** arrive as measurements ``y_t^w = C w_t + noise`` and correct the
  state at every step of the horizon, so they shape the *correction*.

``--no-weather`` ablates only the second. It leaves precipitation forcing intact,
so the reported difference is attributable to the measurement update rather than to
having quietly removed all weather from the model.

Usage
-----
    python -m experiments.infer_v4 --ckpt _v4_ckpt/dbwm_v4_v4.pkl --export-geotiff
    python -m experiments.infer_v4 --smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np

from dbwm.platform import ensure_working_backend, preflight

_PREFLIGHT = preflight()

import jax.numpy as jnp  # noqa: E402

from dbwm.data.seasons import purge_boundary_windows  # noqa: E402
from dbwm.dynamics.memory import lifted_spectrum  # noqa: E402
from dbwm.evaluation import forecast_figures, v4_plots  # noqa: E402
from dbwm.evaluation.geotiff_export import export_forecast_maps  # noqa: E402
from dbwm.evaluation.metrics import field_error_metrics  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.inference.lifted_kalman import (  # noqa: E402
    LiftedSystem, rolling_forecast_multi,
)
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.training.gp_trainer import SpatialGPModule  # noqa: E402
from experiments._v4_common import (  # noqa: E402
    add_data_args, config_from_dict, load_checkpoint, load_inputs,
)

logger = configure_logger(level="INFO", name="dbwm.infer_v4")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="DB-WM v4 inference / forecasting.")
    p.add_argument("--ckpt", default=None, help="path to a trained v4 .pkl")
    p.add_argument("--results-dir", default="./_v4_infer")
    p.add_argument("--forecast-stride", type=int, default=1,
                   help="score every Nth test origin")
    p.add_argument(
        "--origin-date", nargs="+", default=None, metavar="YYYY-MM-DD",
        help="forecast t+1..t+H from these specific dates instead of sweeping "
             "the whole test split. Each must be an observed date on the "
             "calendar with at least L-1 observed days before it (the memory "
             "lift needs w_t ... w_{t-L+1}); the horizon may run past the end "
             "of the archive, in which case the unscoreable steps are still "
             "predicted and exported but reported as having no truth.",
    )
    p.add_argument(
        "--allow-train-origin", action="store_true",
        help="permit --origin-date inside the TRAINING window. Off by default "
             "because a forecast from a date the operator was fitted on is not "
             "an out-of-sample result and must not be reported as one.",
    )
    p.add_argument("--forecast-mode", default="both",
                   choices=["recursive", "direct", "both"])
    p.add_argument("--no-weather", action="store_true",
                   help="ablate the weather MEASUREMENT update only; B_p p_t stays")
    p.add_argument("--export-geotiff", action="store_true")
    p.add_argument("--export-origins", type=int, default=20)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--plot-date", nargs="+", default=None, metavar="YYYY-MM-DD",
        help="draw the t+1..t+H triptych sequence from these origins. Defaults "
             "to every --origin-date given, or to the latest fully observed "
             "origin when sweeping the split.",
    )
    p.add_argument("--no-pixel-maps", action="store_true",
                   help="skip the per-lead per-pixel RMSE/MAE/bias maps")
    p.add_argument("--require-accelerator", action="store_true")
    # The calendar args are accepted so a checkpoint can be scored on a different
    # window; by default they are taken from the checkpoint's own config.
    add_data_args(p)
    return p.parse_args()


def resolve_origin_dates(
    wanted, ds, order: int, horizon: int, train_sel, allow_train: bool = False
) -> np.ndarray:
    """
    Turn requested calendar dates into forecast-origin indices, refusing bad ones.

    A date is a valid origin only if three things hold, and each failure is
    reported specifically rather than as a generic error, because they have
    different fixes:

    **The date is on the calendar and observed.** The archive has 13 missing
    dates; forecasting "from" one of them would silently start the filter at a
    fabricated state.

    **It has ``L - 1`` observed days before it.** The memory lift conditions on
    ``w_t, ..., w_{t-L+1}``. Starting inside a gap would fill part of the delay
    line with the filter's own predictions and report the result as if it had
    been conditioned on data.

    **It is outside the training window**, unless explicitly allowed. A forecast
    from a date the operator was fitted on is an in-sample result; reporting one
    beside out-of-sample numbers would be misleading, so it takes a deliberate
    flag.

    The horizon is allowed to run past the end of the archive. Those steps are
    genuinely predicted and exported -- they are what a forward forecast *is* --
    but they carry no truth, so they are excluded from the metrics rather than
    scored against nothing.

    :param wanted: iterable of ``YYYY-MM-DD`` strings.
    :param ds: the calendar dataset.
    :param order: memory order ``L``.
    :param horizon: ``H``.
    :param train_sel: ``(T,)`` training mask.
    :param allow_train: permit origins inside the training window.
    :return: ``(n,)`` origin indices.
    """
    index = {d: i for i, d in enumerate(ds.dates)}
    out = []
    for raw in wanted:
        try:
            day = dt.date.fromisoformat(str(raw))
        except ValueError as exc:
            raise SystemExit(
                "--origin-date {!r} is not a YYYY-MM-DD date ({})".format(raw, exc)
            )
        if day not in index:
            raise SystemExit(
                "{} is outside the archive calendar {} .. {}.".format(
                    day, ds.dates[0], ds.dates[-1]
                )
            )
        t = index[day]
        if not ds.observed[t]:
            raise SystemExit(
                "{} has no raster (one of the {} missing dates). Pick an "
                "observed date: forecasting from a gap would start the filter "
                "from a fabricated state.".format(day, int((~ds.observed).sum()))
            )
        lo = t - order + 1
        if lo < 0 or not ds.observed[lo : t + 1].all():
            missing = [
                str(ds.dates[j]) for j in range(max(lo, 0), t + 1)
                if not ds.observed[j]
            ]
            raise SystemExit(
                "{} does not have {} consecutive observed days behind it "
                "(memory order L = {}); missing {}. The delay line would be "
                "part prediction, which is not what a conditioned forecast "
                "means.".format(day, order, order, missing or "history before the archive")
            )
        if train_sel[t] and not allow_train:
            raise SystemExit(
                "{} is inside the TRAINING window. A forecast from a date the "
                "operator was fitted on is in-sample; pass --allow-train-origin "
                "if you want it anyway, and label it as such.".format(day)
            )
        out.append(t)
        last = t + horizon
        if last >= ds.n_steps:
            n_beyond = last - ds.n_steps + 1
            logger.info(
                "%s: %d of the %d horizon steps fall past the end of the "
                "archive (%s). They are predicted and exported but cannot be "
                "scored.", day, n_beyond, horizon, ds.dates[-1],
            )
    logger.info(
        "Forecasting from %d requested origin(s): %s",
        len(out), ", ".join(str(ds.dates[t]) for t in out),
    )
    return np.asarray(sorted(set(out)))


def _extend_archive_end(current: str, origin_dates, horizon: int) -> str:
    """
    Widen the calendar so a requested origin, and its whole horizon, fit inside it.

    Refusing an origin past ``archive_end`` would make the operational case --
    "forecast the next six days from today" -- impossible, since today is always
    past the end of a fixed archive window. Extending instead is safe because the
    loader marks any date without a raster as unobserved: the filter predicts
    through them, and :func:`resolve_origin_dates` still refuses to *start* from
    one.

    :param current: the checkpoint's ``archive_end``.
    :param origin_dates: requested ``YYYY-MM-DD`` origins.
    :param horizon: ``H``, so the targets fit too.
    :return: the (possibly unchanged) archive end, ISO formatted.
    """
    end = dt.date.fromisoformat(current)
    needed = end
    for raw in origin_dates:
        try:
            day = dt.date.fromisoformat(str(raw))
        except ValueError:
            continue  # resolve_origin_dates reports this properly
        needed = max(needed, day + dt.timedelta(days=horizon))
    if needed > end:
        logger.info(
            "Extending the calendar %s -> %s so the requested origin(s) and their "
            "%d-step horizon fit. Dates with no raster are marked unobserved; the "
            "forecast is still produced for them but cannot be scored.",
            end, needed, horizon,
        )
        return needed.isoformat()
    return current


def score(forecasts, ds, extractor, origins, horizon, subspace, mask_flat,
          w_true=None):
    """
    Decode each variant's forecasts and score them in physical units.

    Scoring happens in pixel space, not latent space: a latent RMSE is not the
    quantity the thesis reports, and the two can diverge badly when the basis is
    ill-conditioned.

    :param forecasts: output of ``rolling_forecast_multi``.
    :param ds: the dataset.
    :param extractor: the GP state extractor.
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param subspace: latent subspace or ``None``.
    :param mask_flat: ``(n_pixels,)`` validity.
    :return: ``{variant: {"pixel_rmse": (H,), "n": (H,)}}``.
    """
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]
    scale = float(ds.std[0])
    keys = ("rmse", "bias", "ubrmse", "mae")
    out = {}
    for name, fc in forecasts.items():
        acc = {k: np.full(horizon, np.nan) for k in keys}
        counts = np.zeros(horizon, dtype=int)
        for h in range(horizon):
            ok = fc["valid"][:, h]
            org = origins[ok]
            tgt = org + h + 1
            # Steps past the end of the archive are genuine forecasts with no
            # truth: predicted and exported, but never scored against nothing.
            keep = (tgt < ds.n_steps) & ds.observed[np.clip(tgt, 0, ds.n_steps - 1)]
            if not keep.any():
                continue
            wv = fc["mean"][ok, h][keep]
            if subspace is not None:
                wv = (
                    subspace.reconstruct_with_complement(wv, w_true[org[keep]])
                    if w_true is not None else subspace.reconstruct(wv)
                )
            pred = extractor.decode(wv)[:, mask_flat]
            m = field_error_metrics(pred, flat[tgt[keep]], scale=scale)
            for k in keys:
                acc[k][h] = m[k]
            counts[h] = int(keep.sum())
        out[name] = {"pixel_{}".format(k): acc[k] for k in keys}
        out[name]["pixel_rmse"] = acc["rmse"]
        out[name]["n"] = counts
        logger.info("%-22s ubRMSE %s", name, np.round(acc["ubrmse"], 4))
        logger.info("%-22s MAE    %s", name, np.round(acc["mae"], 4))
        logger.info("%-22s (RMSE %s | bias %s | n = %s)", name,
                    np.round(acc["rmse"], 4), np.round(acc["bias"], 4),
                    counts.tolist())
    return out


def main():
    """Load a checkpoint (or smoke-train one), forecast, score, and export."""
    args = build_args()
    backend = ensure_working_backend(allow_cpu_fallback=not args.require_accelerator)
    logger.info("Backend: %s", backend["backend"])

    if args.ckpt is None:
        raise SystemExit(
            "infer_v4 needs a checkpoint. Train one first:\n"
            "  python -m experiments.train_v4 --smoke --ckpt-dir ./_v4_ckpt\n"
            "then pass --ckpt ./_v4_ckpt/dbwm_smoke_v4.pkl"
        )
    ckpt = load_checkpoint(args.ckpt)
    cfg = config_from_dict(ckpt["config"])
    diag = ckpt["diagnostics"]
    logger.info(
        "Loaded %s: r = %d (+%d EOF modes), L = %d, H = %d, k = %d | "
        "reconstruction R2 %.4f | blend s = %s | max gain %.3f | rho = %.4f",
        args.ckpt, cfg.basis.r, diag.get("eof_modes", 0), cfg.memory.order,
        cfg.horizons.horizon, diag["k"], diag.get("reconstruction_r2", float("nan")),
        diag.get("persistence_blend"), diag.get("max_forecast_gain", float("nan")),
        diag["rho"],
    )

    # The calendar is rebuilt from the CHECKPOINT's config, so the test split is the
    # one the operator has never seen. Overriding it on the command line is allowed
    # (scoring a model on a later window is a legitimate thing to want) but is
    # announced, because it silently invalidates the train/test separation if the
    # new window overlaps training.
    for attr, field in (("archive_start", "archive_start"),
                        ("archive_end", "archive_end"),
                        ("split_date", "split_date")):
        given = getattr(args, attr)
        if given and given != getattr(cfg.seasons, field):
            logger.warning("Overriding checkpoint %s %s -> %s.",
                           field, getattr(cfg.seasons, field), given)
            setattr(cfg.seasons, field, given)

    # A requested origin may sit past the checkpoint's archive window -- that is
    # the normal case for an operational forecast, where the point is to predict
    # forward from today. Extend the calendar to cover it plus the horizon rather
    # than refusing: whatever rasters exist in the directory are loaded, the rest
    # of the window is marked unobserved, and the unscoreable steps are still
    # predicted and exported (see `resolve_origin_dates`).
    if args.origin_date and not args.archive_end:
        cfg.seasons.archive_end = _extend_archive_end(
            cfg.seasons.archive_end, args.origin_date, cfg.horizons.horizon
        )
    inp = load_inputs(cfg, args, logger)
    ds = inp.ds

    params = ckpt["params"]
    sigma_eps2 = float(ckpt["sigma_eps2"])
    model = SpatialGPModule(cfg.basis)
    # Rebuild the EXACT decoder the operator was identified against. The EOF
    # block is transductive, so it is carried in the checkpoint rather than
    # recomputed here -- recomputing it on this window would silently change the
    # basis, and with it the meaning of every stored Theta_h.
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
    # Assimilate the weather sensor only if it cleared the held-out gate at
    # training time; --no-weather additionally forces it off as an ablation.
    use_wx = em.usable and not args.no_weather
    if not em.usable and not args.no_weather:
        logger.warning(
            "Weather emission was marked unusable at training time (no channel "
            "with positive held-out R^2); the observer runs without it."
        )
    system = LiftedSystem(
        op=op, b_p=np.asarray(ckpt["b_p"]), q=np.asarray(ckpt["q"]),
        c_w=em.c if use_wx else None,
        r_w=em.r_cov if use_wx else None,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )

    if args.origin_date:
        origins = resolve_origin_dates(
            args.origin_date, ds, cfg.memory.order, cfg.horizons.horizon,
            inp.train_sel, allow_train=args.allow_train_origin,
        )
    else:
        _, scorable = purge_boundary_windows(
            inp.train_idx, inp.test_idx, cfg.memory.order
        )
        origins = np.array([
            t for t in scorable[:: max(args.forecast_stride, 1)]
            if t + cfg.horizons.horizon < ds.n_steps and ds.observed[t]
        ])
        if origins.size == 0:
            raise SystemExit("No scorable test origins in this window.")

    variants = []
    if args.forecast_mode in ("recursive", "both"):
        variants.append({"name": "recursive", "mode": "recursive",
                         "use_weather": not args.no_weather})
    if args.forecast_mode in ("direct", "both"):
        variants.append({"name": "direct", "mode": "direct",
                         "use_weather": not args.no_weather})
    logger.info("Forecasting %d origins x %d variants, horizon %d.",
                origins.size, len(variants), cfg.horizons.horizon)

    forecasts = rolling_forecast_multi(
        system, w_dyn, ds.observed, origins, cfg.horizons.horizon, variants,
        inp.forcing, inp.measurement, cov_dyn, sigma_eps2, family=family,
    )

    mask_flat = ds.common_mask()
    metrics = score(forecasts, ds, extractor, origins, cfg.horizons.horizon,
                    subspace, mask_flat, w_true=w)

    os.makedirs(args.results_dir, exist_ok=True)
    # The predictive band's third variance term, fitted at training time. An
    # older checkpoint has none; the band is then narrower than the error and
    # the reported coverage is not interpretable, so say so rather than let it
    # pass as a calibration result.
    repr_var = ckpt.get("repr_variance")
    if repr_var is None:
        logger.warning(
            "This checkpoint carries no representation variance, so the "
            "predictive interval omits its dominant term and the coverage "
            "printed on each triptych will read far below nominal for that "
            "reason alone. Retrain with experiments.train_v4 to record it."
        )
    pixel_metrics = {}
    if not args.no_plots:
        fig_dir = os.path.join(args.results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        # ubRMSE, not RMSE: a whole-field offset and genuine structural
        # disagreement have different fixes, and the horizon curve is read to
        # decide which one to work on.
        v4_plots.plot_rmse_by_horizon(
            metrics, os.path.join(fig_dir, cfg.name + "_ubrmse_by_horizon.png"),
            key="pixel_ubrmse", units=cfg.modality_spec().units,
        )
        v4_plots.plot_koopman_spectrum(
            lifted_spectrum(op), os.path.join(fig_dir, cfg.name + "_koopman.png"),
            title="{} | lifted Koopman spectrum (L = {})".format(
                cfg.name, cfg.memory.order),
        )
        primary_name = "recursive" if "recursive" in forecasts else next(iter(forecasts))
        # Default to drawing exactly the origins that were asked for; sweeping
        # the whole split instead draws the latest fully observed one.
        plot_dates = None
        if args.plot_date:
            plot_dates = forecast_figures.resolve_plot_dates(
                args.plot_date, ds, origins
            )
        elif args.origin_date:
            plot_dates = list(origins)
        rendered = forecast_figures.render_horizon_figures(
            fig_dir, cfg.name, forecasts[primary_name], origins, ds, extractor,
            subspace=subspace, w_origin=w, horizon=cfg.horizons.horizon,
            units=cfg.data.modality.upper(), variant=primary_name,
            plot_origins=plot_dates, per_pixel=not args.no_pixel_maps,
            repr_var=repr_var,
        )
        pixel_metrics = rendered["pixel_metrics"]
        logger.info(
            "Saved %d figures to %s", len(rendered["figures"]) + 2,
            os.path.abspath(fig_dir),
        )

    # Written after the figures so the per-pixel summary -- which lead is worst,
    # and at which pixel -- lands in the same file as the field averages.
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as fh:
        json.dump(
            {"checkpoint": os.path.abspath(args.ckpt),
             "diagnostics": ckpt["diagnostics"],
             "n_origins": int(origins.size),
             "origin_dates": [str(ds.dates[int(o)]) for o in origins],
             "weather_update": not args.no_weather,
             "metrics": {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
                         for k, v in metrics.items()},
             "per_pixel_error": {str(k): v for k, v in pixel_metrics.items()}},
            fh, indent=2, default=float,
        )

    if args.export_geotiff:
        primary_name = "recursive" if "recursive" in forecasts else next(iter(forecasts))
        export_forecast_maps(
            os.path.join(args.results_dir, "geotiff"), forecasts[primary_name],
            origins, ds.dates, extractor, ds, subspace=subspace,
            max_origins=args.export_origins, variant=primary_name, w_origin=w,
            sigma_from=forecasts[primary_name].get("cov_trace"),
            repr_var=repr_var,
        )

    logger.info("Results written to %s", os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
