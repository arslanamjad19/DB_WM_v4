"""
End-to-end DB-WM v4 pipeline for NDVI (and, with ``--modality lst``, for LST).

Runs the full sequence of the v3 estimation pipeline:

============  =====================================================================
Stage         What happens
============  =====================================================================
0             Calendar reindex + season split, with the coverage table checked
1             Weather loaded and split by role: p_t -> B_p, (Rs,Ta,VPD) -> C
2             Spatial basis ``Psi`` trained by dPPGP, then FROZEN
3             GP posterior per date: w_t, Sigma_{w_t}
4             **Whiteness pretest** (v3 step 0) -- the pivotal, falsifiable check
5             Memory kernel {A_j}, B_p, Q + Lemma 2.6 / 2.7 / Thm 2.8 certificates
6             Weather emission C, R  (+ information gain)
7             Semigroup-shrunk horizon family {Theta_h, B^(h), Sigma_h} + ||D_h||
8             Memory-depth sweep: ||D_h|| vs L  (the Mori-Zwanzig measurement)
9             Rolling t+1..t+6 forecast with a weather update at EVERY step
10            Per-horizon metrics, PIT / coverage, and the seasonal statistics
============  =====================================================================

Usage::

    # Synthetic smoke run, no Drive needed (~2 minutes on CPU)
    python -m experiments.run_ndvi_v4 --smoke

    # Real data
    python -m experiments.run_ndvi_v4 \\
        --ndvi-dir    /content/drive/MyDrive/NDVI_Downscaled_30m/NDVI_downscaled_30m \\
        --weather-csv /content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P/sayedanwala_historical_weather_2022_2026.csv \\
        --epochs 50 --r 256 --memory-order 7 --horizon 6

Every path is a flag, so only the four ``--*-dir`` / ``--*-csv`` arguments need
changing once the Drive folders are mounted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
from typing import Dict

import numpy as np

# Preflight BEFORE importing JAX. A Colab image with a stale CUDA plugin breaks
# the PJRT ABI for *every* backend, CPU included, the moment JAX initialises --
# so the only reliable escape is to set JAX_PLATFORMS before that happens.
from dbwm.platform import describe_execution_plan, ensure_working_backend, preflight

_PREFLIGHT = preflight()

import jax.numpy as jnp  # noqa: E402  (must follow preflight)

from dbwm.config import default_config, smoke_config
from dbwm.data.ndvi_dataset import load_calendar_dataset
from dbwm.data.seasons import (
    KHARIF, RABI, ZAID, assert_split_targets, coverage_table, format_coverage,
    purge_boundary_windows, split_indices,
)
from dbwm.data.weather import build_weather_table, synthetic_weather
from dbwm.dynamics.diagnostics import (
    one_step_residuals, residual_autocorrelation, whiteness_pretest,
)
from dbwm.dynamics.conditioning import (
    log_rank_report, log_transient, transient_amplification, weight_rank,
)
from dbwm.dynamics.emission import fit_emission, information_gain
from dbwm.dynamics.subspace import build_subspace
from dbwm.dynamics.memory import (
    enforce_stability, identify_memory, lifted_spectrum, memory_profile,
    observability_certificate, spectral_radius_lifted,
    enforce_forecast_gain, forecast_gain, calibrate_persistence_blend,
    enforce_spectral_radius, is_persistence,
)
from dbwm.dynamics.multihorizon import (
    coverage_comparison, fit_horizon_family, memory_depth_sweep, semigroup_defect,
)
from dbwm.evaluation.error_budget import (
    bias_attribution, error_budget, persistence_rmse, worst_pixel_attribution,
)
from dbwm.evaluation.metrics import field_error_metrics
from dbwm.evaluation.geotiff_export import (
    export_forecast_maps, representation_variance,
)
from dbwm.evaluation.spatiotemporal_stats import format_season_table, season_reports
from dbwm.evaluation import forecast_figures, v4_plots
from dbwm.gp.empirical_basis import build_augmented_basis
from dbwm.gp.state import build_extractor
from dbwm.inference.lifted_kalman import LiftedSystem, rolling_forecast_multi
from dbwm.log_utils import configure_logger
from dbwm.training.gp_trainer import train_spatial_basis
from experiments._v4_common import add_model_args, apply_model_args

logger = configure_logger(level="INFO", name="dbwm.v4")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="DB-WM v4 (memory + multi-horizon) pipeline")
    p.add_argument("--modality", default="ndvi", choices=["ndvi", "lst"])
    p.add_argument("--ndvi-dir", default=None)
    p.add_argument("--lst-dir", default=None)
    p.add_argument("--weather-csv", default=None)
    p.add_argument("--results-dir", default="./_v4_out")
    p.add_argument("--archive-start", default="2022-01-01")
    p.add_argument("--archive-end", default="2026-04-30")
    p.add_argument("--split-date", default="2025-04-15")
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--memory-order", type=int, default=None, help="L")
    p.add_argument("--horizon", type=int, default=None, help="H")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--parameterization", default=None, choices=["s1", "s2", "s3", "unstructured"])
    p.add_argument("--estimator", default=None, choices=["shrunk", "direct", "iterated"])
    p.add_argument("--forecast-mode", default="both", choices=["recursive", "direct", "both"])
    p.add_argument("--no-sweep", action="store_true", help="skip the ||D_h|| vs L sweep")
    p.add_argument(
        "--forecast-stride", type=int, default=1,
        help="score every Nth test origin. The branch cost is linear in the origin "
             "count, so a stride of 3 cuts stage 9 threefold at negligible cost to "
             "an RMSE-vs-horizon curve averaged over hundreds of origins.",
    )
    p.add_argument(
        "--keep-forecast-cov", action="store_true",
        help="retain full (origins, H, r, r) forecast covariances. ~1.1 GB per "
             "variant at r=256; needed only for PIT/coverage work.",
    )
    p.add_argument("--no-stats", action="store_true", help="skip the seasonal statistics")
    p.add_argument("--no-separability", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic end-to-end run")
    p.add_argument("--synthetic", action="store_true", help="synthetic data, full size")
    p.add_argument(
        "--max-pixels", type=int, default=250_000,
        help="per-frame pixel budget; larger frames are read DECIMATED so the "
             "native array is never allocated. 0 = native resolution. A full-scene "
             "30 m archive is ~6.8M px/frame = ~50 GB over 1581 dates.",
    )
    p.add_argument(
        "--ram-budget-gb", type=float, default=8.0,
        help="refuse to build a frame stack larger than this, with instructions, "
             "instead of being OOM-killed minutes into the load.",
    )
    p.add_argument(
        "--subspace-energy", type=float, default=0.999,
        help="fraction of TEMPORAL variance the dynamics subspace retains. r keeps "
             "its full size for representation; only the dynamics are reduced. 0 "
             "disables the projection. Raised from 0.995 because the discarded "
             "0.5%% is itself a forecast-error term (it shows up as 'subspace "
             "truncation' in the error budget), and once the representation floor "
             "is closed it becomes binding. The conditioning argument that "
             "motivated aggressive truncation -- a near-singular A_0 in unexcited "
             "directions -- is answered instead by the increment parameterisation, "
             "which makes A_0 = I + G and therefore never singular. Raising it further is a real trade-off, not a free win: k grows, the filter costs O(L^2 (Lk)^3), and the operator is fitted from the same ~1200 transitions so it gets noisier -- at k=94 on the smoke record every blend candidate was rejected by the gain cap and the model fell back to persistence. Use the printed error budget: raise this only while 'subspace truncation' is the binding term.",
    )
    p.add_argument("--max-k", type=int, default=None,
                   help="hard cap on the dynamics subspace dimension k")
    p.add_argument(
        "--auto-r", action="store_true",
        help="after the GP solve, re-run with r matched to the rank the weight "
             "trajectory actually excites. r must satisfy r << n_pixels for the GP "
             "(v2 Prop 3.1) AND r <~ excited rank for the DYNAMICS; only the first "
             "is usually checked, and the second is what breaks the forecast.",
    )
    p.add_argument(
        "--export-geotiff", action="store_true",
        help="write t+1..t+H forecast rasters as 3-band GeoTIFFs "
             "(forecast / std / error) in the source CRS",
    )
    p.add_argument("--export-origins", type=int, default=20,
                   help="how many forecast origins to export, evenly spaced")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    p.add_argument(
        "--plot-date", nargs="+", default=None, metavar="YYYY-MM-DD",
        help="draw the t+1..t+H triptych sequence from these forecast origins "
             "instead of the default (the latest origin whose whole horizon is "
             "observed). Each must be a scored origin; the error message lists "
             "the nearest ones if it is not. Files are named by the date, so "
             "several may be given.",
    )
    p.add_argument(
        "--no-pixel-maps", action="store_true",
        help="skip the per-pixel RMSE/MAE/bias maps. They pool every origin at "
             "each lead, which is the only aggregation under which 'the worst "
             "pixel' is a well-posed question -- on a single date per-pixel RMSE "
             "and MAE are both just |error|.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--require-accelerator", action="store_true",
        help="fail instead of silently falling back to CPU when the GPU/TPU "
             "backend is broken",
    )
    add_model_args(p)
    return p.parse_args()


def _apply_overrides(cfg, args):
    """Fold CLI overrides into the config."""
    cfg.data.modality = args.modality
    if args.ndvi_dir:
        cfg.data.ndvi_dir = args.ndvi_dir
    if args.lst_dir:
        cfg.data.lst_dir = args.lst_dir
    if args.weather_csv:
        cfg.weather.csv_path = args.weather_csv
    if args.r:
        cfg.basis.r = args.r
    if args.memory_order:
        cfg.memory.order = args.memory_order
    if args.horizon:
        cfg.horizons.horizon = args.horizon
    if args.epochs:
        cfg.training.n_epochs = args.epochs
    if args.parameterization:
        cfg.memory.parameterization = args.parameterization
    if args.estimator:
        cfg.horizons.estimator = args.estimator
    cfg.seasons.archive_start = args.archive_start
    cfg.seasons.archive_end = args.archive_end
    cfg.seasons.split_date = args.split_date
    cfg.inference.results_dir = args.results_dir
    apply_model_args(cfg, args)
    return cfg.sync_input_dim()


def main():
    """Run the full v4 pipeline."""
    args = build_args()
    cfg = smoke_config() if args.smoke else default_config()
    cfg = _apply_overrides(cfg, args)
    synthetic = args.smoke or args.synthetic
    os.makedirs(args.results_dir, exist_ok=True)

    # Probe JAX before any real work. A mismatched CUDA plugin otherwise blows up
    # inside stage 2 -- after data loading and weather assembly -- with a traceback
    # pointing at jax.random.PRNGKey, which is not where the problem is.
    backend = ensure_working_backend(allow_cpu_fallback=not args.require_accelerator)
    logger.info("%s", describe_execution_plan(str(backend["backend"])))

    start = dt.date.fromisoformat(cfg.seasons.archive_start)
    end = dt.date.fromisoformat(cfg.seasons.archive_end)
    split = dt.date.fromisoformat(cfg.seasons.split_date)
    if args.smoke:
        end = start + dt.timedelta(days=729)
        split = start + dt.timedelta(days=560)
        # Write the effective window BACK into the config. Everything downstream
        # -- the figure's climatology reference, the saved summary, any
        # checkpoint -- reads cfg.seasons, and leaving it at the real-data
        # defaults made the smoke split (2023-07-15) disagree with the recorded
        # one (2025-04-15), which is past the smoke archive entirely: the
        # climatology line was then averaged over an empty test set.
        cfg.seasons.archive_end = end.isoformat()
        cfg.seasons.split_date = split.isoformat()

    # ---------------- Stage 0: calendar + season split ---------------- #
    logger.info("=" * 78)
    spec = cfg.modality_spec()
    logger.info(
        "STAGE 0  Calendar reindex and season split  [%s, %s; every metric below "
        "is in %s]", spec.long_name, spec.units, spec.units,
    )
    ds = load_calendar_dataset(
        None if synthetic else cfg.data_dir(), start, end, split,
        nodata=cfg.data.nodata_value,
        height=cfg.data.image_height if synthetic else None,
        width=cfg.data.image_width if synthetic else None,
        max_pixels=args.max_pixels,
        ram_budget_gb=args.ram_budget_gb,
        synthetic=synthetic,
        synthetic_shape=(
            (cfg.data.image_height or 32, cfg.data.image_width or 32)
            if args.smoke else (48, 44)
        ),
        seed=args.seed,
        modality=spec.label,
        physical_range=cfg.physical_range(),
        drop_empty=cfg.drop_empty_frames(),
        min_valid_fraction=cfg.modality_spec().min_valid_fraction,
    )
    table = coverage_table(start, end, split)
    logger.info("Season coverage:\n%s", format_coverage(table))
    if not args.smoke:
        assert_split_targets(
            start, end, split,
            {KHARIF: cfg.seasons.target_kharif,
             RABI: cfg.seasons.target_rabi,
             ZAID: cfg.seasons.target_zaid},
            cfg.seasons.target_tolerance,
        )
    train_idx, test_idx = split_indices(ds.dates, split)
    train_sel = np.zeros(ds.n_steps, dtype=bool)
    train_sel[train_idx] = True

    # ---------------- Stage 1: weather, split by role ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 1  Weather: p_t -> B_p (input) | Rs/Ta/VPD -> C (measurement)")
    # Real weather is used whenever a readable CSV is given, even alongside
    # synthetic rasters. That combination is deliberately supported: it exercises
    # the whole weather path -- schema resolution, climatology, quiescent
    # detection -- before the Drive rasters are mounted. Note the emission R^2
    # will be meaningless there, because a synthetic field has no physical
    # relationship to the real weather; the low-R2 warning firing is the correct
    # behaviour, not a failure.
    use_real_weather = (not synthetic) or bool(
        args.weather_csv and os.path.exists(args.weather_csv)
    )
    if not use_real_weather:
        wt = synthetic_weather(
            ds.dates, train_sel, cfg.weather.precip_lags,
            cfg.weather.measurement_cols, seed=args.seed,
        )
    else:
        wt = build_weather_table(
            ds.dates, cfg.weather.csv_path, train_sel, cfg.weather.precip_lags,
            cfg.weather.measurement_cols, cfg.weather.n_harmonics,
            cfg.weather.scale_forcing,
        )
    forcing = np.asarray(wt.forcing, dtype=np.float64)
    weather = np.asarray(wt.measurement, dtype=np.float64)

    # ---------------- Stage 2: train Psi, then freeze ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 2  Training the spatial basis Psi (dPPGP), then freezing it")
    model, params, train_metrics = train_spatial_basis(
        cfg, ds.frames[train_sel], ds.valid_mask[train_sel], ds.observed[train_sel],
        ds.coords, seed=args.seed,
        rollout_steps=cfg.training.rollout_steps,
        lambda_rollout=cfg.training.lambda_rollout,
        tf_threshold=cfg.teacher_forcing_threshold(),
        use_teacher_forcing=cfg.training.teacher_forcing,
        norm_std=float(ds.std[0]),
    )
    sigma_eps2 = float(model.apply(params, method=model.sigma_eps2))

    # ---------------- Stage 3: GP posterior per date ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 3  GP posterior: w_t = Lambda^-1 Phi^T y_t per date")
    psi = np.concatenate(
        [np.asarray(model.apply(params, jnp.asarray(ds.coords[i:i + 20000]),
                                method=model.features))
         for i in range(0, ds.coords.shape[0], 20000)], axis=0)
    mask_static = ds.common_mask()
    # Close the representation floor BEFORE anything downstream sees w_t: every
    # forecast is decoded through this basis, so its residual is a hard lower
    # bound on the forecast error that no dynamics model can get under.
    aug = build_augmented_basis(
        psi, ds.frames, mask_static, train_sel & ds.observed,
        n_modes=cfg.basis.eof_modes, energy=cfg.basis.eof_energy,
        ridge=sigma_eps2, use_climatology=cfg.basis.use_climatology,
        select=cfg.basis.eof_select,
        holdout_fraction=cfg.basis.eof_holdout_fraction,
    )
    extractor = build_extractor(
        lambda c: model.apply(params, jnp.asarray(c), method=model.features),
        ds.coords, sigma_eps2, ds.static_mask,
        extra_features=aug.phi[:, aug.r_learned:] if aug.q_empirical else None,
        offset=aug.offset if cfg.basis.use_climatology else None,
    )
    gp = extractor.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    w = gp["weights"]
    w_cov = gp["covariances"]

    # How many directions of R^r does the trajectory actually excite? This is the
    # binding constraint for the DYNAMICS and it is temporal, not spatial: r << n
    # is necessary but not sufficient, and exceeding the excited rank produces a
    # near-singular A_0, an ill-conditioned modal basis, and a non-normal operator
    # whose one-step forecast is worse than climatology.
    rank_report = weight_rank(w, ds.observed & train_sel)
    log_rank_report(rank_report)

    # Decouple representation from dynamics. Keeping the full r for the GP while
    # fitting the dynamics on the excited subspace is what v2 Prop. 4.4 licenses:
    # the directions removed are ker W_-^T, on which the least-squares operator
    # already had zero eigenvalues.
    subspace = None
    if args.subspace_energy and args.subspace_energy > 0:
        subspace = build_subspace(
            w, ds.observed & train_sel, args.subspace_energy, args.max_k
        )
        if subspace.k >= subspace.r:
            subspace = None

    if args.auto_r and rank_report.over_parameterised:
        new_r = rank_report.suggested_r()
        logger.warning(
            "--auto-r: refitting the basis with r = %d (was %d) to match the "
            "excited rank.", new_r, cfg.basis.r,
        )
        cfg.basis.r = new_r
        model, params, train_metrics = train_spatial_basis(
            cfg, ds.frames[train_sel], ds.valid_mask[train_sel],
            ds.observed[train_sel], ds.coords, seed=args.seed,
        )
        sigma_eps2 = float(model.apply(params, method=model.sigma_eps2))
        extractor = build_extractor(
            lambda c: model.apply(params, jnp.asarray(c), method=model.features),
            ds.coords, sigma_eps2, ds.static_mask,
        )
        gp = extractor.solve_sequence(
            ds.frames, ds.valid_mask, ds.observed, want_cov=True
        )
        w, w_cov = gp["weights"], gp["covariances"]
        rank_report = weight_rank(w, ds.observed & train_sel)
        log_rank_report(rank_report)

    # ---------------- Stage 4: the pivotal whiteness pretest ---------------- #
    logger.info("=" * 78)
    if subspace is not None:
        w_dyn = subspace.project(w)
        cov_dyn = subspace.project_covariance(w_cov)
        logger.info(
            "Dynamics now run in the %d-dim subspace; decoding still uses r = %d.",
            subspace.k, subspace.r,
        )
    else:
        w_dyn, cov_dyn = w, w_cov

    logger.info("STAGE 4  Whiteness pretest (v3 step 0) -- run before anything else")
    op1, b1 = identify_memory(
        w_dyn[train_sel], 1, forcing[train_sel], "unstructured",
        cfg.dynamics.ridge_mu, valid=ds.observed[train_sel],
    )
    resid = one_step_residuals(w_dyn[train_sel], op1.blocks[0], b1, forcing[train_sel])
    pretest = whiteness_pretest(
        resid, cfg.memory.ljung_box_lags, cfg.memory.ljung_box_alpha,
        cfg.memory.ljung_box_projection, args.seed,
    )
    acf = residual_autocorrelation(resid, cfg.memory.ljung_box_lags)
    if not pretest["reject"]:
        logger.warning(
            "The pretest did NOT reject whiteness. Per v3 Corollary 2.5.1 the "
            "memory lift is then pure variance inflation for this record. "
            "Continuing so the L=1 baseline is still produced, but report this "
            "result -- it is the pivotal experiment, and it came out negative."
        )

    # ---------------- Stage 5: memory kernel + certificates ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 5  Memory kernel {A_j}, B_p, Q and the Section 4 certificates")
    op, b_p = identify_memory(
        w_dyn[train_sel], cfg.memory.order, forcing[train_sel],
        cfg.memory.parameterization, cfg.dynamics.ridge_mu,
        cfg.dynamics.quiescent_threshold, cfg.memory.reduced_rank,
        valid=ds.observed[train_sel], increment=cfg.memory.increment,
    )
    blend_report = {}
    if cfg.memory.enforce_stability:
        if cfg.memory.stability_metric == "forecast_gain":
            # Let the operator EARN its deviation from persistence on held-out
            # data (v3 Thm 2.12 shrinkage), with the gain cap as a hard rail.
            op, blend_report = calibrate_persistence_blend(
                op, b_p, w_dyn[train_sel], forcing[train_sel],
                ds.observed[train_sel], cfg.horizons.horizon,
                gain_max=cfg.memory.gain_max,
            )
        else:
            op = enforce_stability(op, cfg.dynamics.rho_max)
    # The gain cap bounds ||S A^h|| over h = 1..H only. A non-normal operator can
    # satisfy it while rho(A_cal) > 1, i.e. while being unstable -- on the real
    # record rho reached 1.37 with the 6-step gain still under 2. DynamicsConfig
    # declares rho_max = 1.0; before this rail it was declared and never applied
    # in the forecast_gain path, which is the whole reason rho was free to drift
    # above 1. Both rails now run.
    op, radius_report = enforce_spectral_radius(op, cfg.dynamics.rho_max)
    blend_report = dict(blend_report or {})
    blend_report["spectral_radius_rail"] = radius_report
    q = _process_noise(w_dyn[train_sel], op, b_p, forcing[train_sel],
                       ds.observed[train_sel], cfg.dynamics.process_noise_jitter)
    cert = observability_certificate(op, np.eye(op.r))
    if is_persistence(op):
        # Thm 2.8(i) necessarily "fails" here: A_{L-1} = 0 exactly because the
        # blend shrank the memory kernel away. That is a selected outcome, not
        # over-lagging, and the usual advice (reduce L) does not apply.
        logger.info(
            "The operator IS persistence (A_0 = I, A_j = 0 for j >= 1), so "
            "Thm 2.8(i) is vacuously violated: there is no memory kernel to "
            "observe. Ignore the over-lagging advice above -- the finding is "
            "that the memory kernel did not earn its place on held-out data, "
            "which is reported in `persistence_blend` in summary.json."
        )
    rho = spectral_radius_lifted(op)
    # rho <= 1 bounds only ASYMPTOTIC growth. For a non-normal operator the powers
    # can peak far higher, and that peak is what a 6-step forecast experiences.
    transient = transient_amplification(op.companion(), horizon=2 * cfg.horizons.horizon)
    log_transient(transient)
    logger.info(
        "rho(A_cal) = %.4f | max_h ||S A_cal^h|| = %.4f | Thm 2.8: observable=%s "
        "detectable=%s (s_min(A_last)/||A_0|| = %.2e)",
        rho, float(np.max(forecast_gain(op, cfg.horizons.horizon))),
        cert["observable"], cert["detectable"], cert["a_last_relative_smin"],
    )

    # ---------------- Stage 6: weather emission ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 6  Weather emission C (the daily rank-m sensor)")
    obs_train = train_sel & ds.observed
    emission = fit_emission(
        w_dyn[obs_train], weather[obs_train], list(wt.measurement_names),
        cfg.weather.emission_ridge, cfg.weather.full_noise_covariance,
        scale=wt.measurement_scale,
    )
    gain = information_gain(emission, q)
    logger.info(
        "One weather update removes %.1f%% of the state-uncertainty trace.",
        100.0 * gain["fraction_removed"],
    )

    # ---------------- Stage 7: horizon family ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 7  Semigroup-shrunk multi-horizon family")
    family = fit_horizon_family(
        w_dyn[train_sel], op, b_p, cfg.horizons.horizon, forcing[train_sel],
        ds.observed[train_sel], cfg.horizons.estimator, cfg.horizons.ridge_mu,
        cfg.horizons.nu, cfg.horizons.nu_grid, cfg.horizons.nu_selection,
        cfg.horizons.nu_holdout_fraction, q=q,
    )
    defect = semigroup_defect(family)
    cov_cmp = coverage_comparison(
        family, w_dyn[train_sel], op, b_p, forcing[train_sel], ds.observed[train_sel]
    )

    # ---------------- Stage 8: memory-depth sweep ---------------- #
    sweep = None
    if cfg.memory.sweep_memory_order and not args.no_sweep:
        logger.info("=" * 78)
        logger.info("STAGE 8  Memory-depth sweep: ||D_h|| vs L (Mori-Zwanzig depth)")
        sweep = memory_depth_sweep(
            w_dyn[train_sel], range(1, cfg.memory.order + 1), cfg.horizons.horizon,
            forcing[train_sel], ds.observed[train_sel],
            cfg.memory.parameterization, cfg.horizons.ridge_mu,
            estimator=cfg.horizons.estimator, nu_selection=cfg.horizons.nu_selection,
        )

    # ---------------- Stage 9: rolling t+1..t+6 forecast ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 9  Rolling t+1..t+%d forecast, weather update at EVERY step",
                cfg.horizons.horizon)
    # The weather sensor is assimilated only if it earned it on held-out data.
    # The shared filter pass applies this update at every one of the ~1581 steps,
    # so an emission that fails out of sample does not just add noise -- it biases
    # the state each forecast starts from, and the per-variant `use_weather` flag
    # cannot undo it because that only toggles the forecast branch.
    system = LiftedSystem(
        op=op, b_p=b_p, q=q,
        c_w=emission.c if emission.usable else None,
        r_w=emission.r_cov if emission.usable else None,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )
    _, scorable = purge_boundary_windows(train_idx, test_idx, cfg.memory.order)
    origins = np.array(
        [t for t in scorable if t + cfg.horizons.horizon < ds.n_steps and ds.observed[t]]
    )
    if args.forecast_stride > 1:
        origins = origins[:: args.forecast_stride]
    modes = ["recursive", "direct"] if args.forecast_mode == "both" else [args.forecast_mode]

    # One filter pass, all variants branched from the same live state at each
    # origin. Running rolling_forecast once per variant would repeat the identical
    # 1581-step filter -- ~9 min each at r=256 -- for no information gain.
    #
    # The no-weather baseline toggles the update inside the FORECAST BRANCH only;
    # the shared filter still assimilates everything available up to the origin.
    # That isolates the value of the t+1..t+6 correction instead of confounding it
    # with a degraded origin state.
    variants = []
    for mode in modes:
        variants.append({"name": mode, "mode": mode, "use_weather": True})
        variants.append(
            {"name": mode + "_no_weather", "mode": mode, "use_weather": False}
        )
    logger.info(
        "Stage 9: %d origins (stride %d) x %d variants, single filter pass.",
        origins.size, args.forecast_stride, len(variants),
    )
    forecasts = rolling_forecast_multi(
        system, w_dyn, ds.observed, origins, cfg.horizons.horizon, variants,
        forcing, weather, cov_dyn, sigma_eps2, family=family,
        keep_cov=args.keep_forecast_cov,
    )

    # ---------------- Error budget: which stage is binding? ---------------- #
    pers = persistence_rmse(
        ds.frames, mask_static, ds.observed, origins, cfg.horizons.horizon,
        float(ds.std[0]),
    )
    budget = error_budget(
        extractor, w, ds.frames, mask_static, ds.observed, float(ds.std[0]),
        gain=float(np.max(forecast_gain(op, cfg.horizons.horizon))),
        subspace=subspace, persistence=float(pers[0]), units=spec.units,
    )
    logger.info(
        "persistence RMSE by horizon (the bar to beat): %s %s",
        np.array2string(pers, precision=spec.decimals), spec.units,
    )

    # Per-pixel representation variance: the third variance term the predictive
    # band was missing entirely. Estimated on TRAINING dates only, so nothing
    # about the evaluation period reaches the interval.
    repr_stats = representation_variance(
        extractor, w, ds.frames, mask_static, train_sel & ds.observed,
        float(ds.std[0]),
    )
    # The IN-SAMPLE residual is not the right input to the band. The empirical
    # block is the SVD of the training residual, so it reconstructs the frames it
    # was fitted on almost exactly by construction -- on the real record 0.0016
    # NDVI against 0.044 on dates it had not seen, a 28x gap. Feeding the smaller
    # number to the interval would restate the same under-dispersion the
    # representation term was added to fix. `build_augmented_basis` measures the
    # held-out version on a chronological tail of TRAINING, which leaks nothing
    # and is the number every forecast actually inherits.
    scale2 = float(ds.std[0]) ** 2
    if aug.holdout_variance is not None:
        band_var = np.asarray(aug.holdout_variance) * scale2
        logger.info(
            "Representation residual: in-sample RMS %.4f, HELD-OUT RMS %.4f "
            "(%.0fx). The band uses the held-out value -- the in-sample one is "
            "near zero by construction and would under-disperse the interval.",
            repr_stats["rms"], float(np.sqrt(np.mean(band_var[mask_static]))),
            aug.generalisation_gap,
        )
    else:  # pragma: no cover - only for a basis built without the holdout path
        band_var = repr_stats["variance"]
        logger.warning(
            "No held-out representation variance available; the band falls back "
            "to the in-sample residual and will read over-confident."
        )

    # ---------------- Stage 10: metrics + seasonal statistics ---------------- #
    logger.info("=" * 78)
    logger.info("STAGE 10  Per-horizon metrics and the seasonal statistics")
    metrics = _score_forecasts(
        forecasts, w, ds, extractor, origins, cfg.horizons.horizon, subspace
    )
    # ubRMSE and MAE are reported alongside RMSE because RMSE^2 = bias^2 +
    # ubRMSE^2 conflates two errors with different cures: a whole-field offset
    # (correctable, and usually a symptom of something specific) and genuine
    # structural disagreement (what a better model must actually reduce).
    fmt = lambda a: np.array2string(a, precision=spec.decimals, suppress_small=True)
    logger.info("All forecast metrics below are in %s.", spec.units)
    for name, per_h in metrics.items():
        logger.info("%-24s ubRMSE %s %s", name, fmt(per_h["pixel_ubrmse"]), spec.units)
        logger.info("%-24s MAE    %s %s", name, fmt(per_h["pixel_mae"]), spec.units)
        logger.info("%-24s (RMSE  %s | bias %s) %s", name,
                    fmt(per_h["pixel_rmse"]), fmt(per_h["pixel_bias"]), spec.units)
        b, u = per_h["pixel_bias"][0], per_h["pixel_ubrmse"][0]
        if np.isfinite(b) and np.isfinite(u) and abs(b) > 0.5 * u:
            logger.warning(
                "%s: the h=1 bias (%+.4f) is %.0f%% of the ubRMSE (%.4f), i.e. "
                "%.0f%% of the squared error is a constant offset rather than "
                "structural disagreement. Look for a mis-specified sensor, a "
                "climatology fitted on a different period, or a shrinkage anchor "
                "pulling toward the wrong state -- not at the dynamics.",
                name, b, 100.0 * abs(b) / u, u,
                100.0 * b**2 / (b**2 + u**2),
            )

    # Where does the offset actually enter -- the basis, or the filter/operator?
    # RMSE^2 = bias^2 + ubRMSE^2 says how big it is, never where it comes from,
    # and the two candidates need opposite fixes.
    primary_name = "recursive" if "recursive" in metrics else sorted(metrics)[0]
    bias_report = bias_attribution(
        extractor, w, ds.frames, mask_static, (~train_sel) & ds.observed,
        float(ds.std[0]),
        forecast_bias=metrics[primary_name]["pixel_bias"],
    )

    # Is the worst pixel a basis failure (fixable) or an unpredicted ground
    # event (not fixable by any autonomous linear operator)?
    err_by_lead, org_by_lead = {}, {}
    for h in range(1, cfg.horizons.horizon + 1):
        e, o = forecast_figures.error_stack(
            forecasts[primary_name], origins, ds, extractor, h, subspace, w
        )
        if e.shape[0]:
            err_by_lead[h], org_by_lead[h] = e, o
    worst_report = worst_pixel_attribution(
        err_by_lead, ds.frames, org_by_lead, mask_static, float(ds.std[0]),
        repr_rms=np.sqrt(band_var), units=spec.units,
    ) if err_by_lead else {"pixels": []}

    stats = {}
    if not args.no_stats:
        for name, sel in (("train", train_sel), ("test", ~train_sel)):
            keep = sel & ds.observed
            stats[name] = season_reports(
                ds.denormalize(ds.frames[keep]), ds.valid_mask[keep],
                [d for d, k in zip(ds.dates, keep) if k], name,
                # Metres, converted from degrees when the archive is on a
                # geographic CRS -- the covariogram bins C(h, u) by ground
                # distance, so a degree-valued GSD would mis-state every
                # separation by ~10^5.
                pixel_size=ds.pixel_size,
                run_separability=not args.no_separability,
                min_abs_mean=spec.cv_min_abs_mean,
            )
        logger.info("Seasonal statistics:\n%s", format_season_table(stats))
        caveat = spec.cv_caveat()
        if caveat:
            logger.info("CV column: %s", caveat)

    # ---------------- Figures and georeferenced rasters ---------------- #
    figures: List[str] = []
    pixel_metrics: Dict[int, Dict[str, float]] = {}
    if not args.no_plots:
        fig_dir = os.path.join(args.results_dir, "figures")
        plot_dates = (
            forecast_figures.resolve_plot_dates(args.plot_date, ds, origins)
            if args.plot_date else None
        )
        figures, pixel_metrics = _make_figures(
            fig_dir, cfg, ds, forecasts, metrics, op, sweep, transient, stats,
            forecast_dates=[ds.dates[o] for o in origins],
            extractor=extractor, subspace=subspace, origins=origins,
            w_true=w, plot_dates=plot_dates, per_pixel=not args.no_pixel_maps,
            repr_var=band_var,
        )
        logger.info("Wrote %d figures to %s", len(figures), fig_dir)

    exported: List[str] = []
    if args.export_geotiff:
        primary = "recursive" if "recursive" in forecasts else modes[0]
        exported = export_forecast_maps(
            os.path.join(args.results_dir, "forecast_geotiff"),
            forecasts[primary], origins, ds.dates, extractor, ds,
            subspace=subspace, horizon=cfg.horizons.horizon, w_origin=w,
            max_origins=args.export_origins, variant=primary,
            sigma_from=forecasts[primary].get("cov_trace"),
            repr_var=band_var,
        )

    _save(args.results_dir, cfg, ds, pretest, acf, op, b_p, q, cert, rho, emission,
          gain, family, defect, cov_cmp, sweep, metrics, stats, train_metrics, gp,
          rank_report, transient, budget=budget, persistence=pers,
          blend=blend_report, basis=aug, pixel_metrics=pixel_metrics,
          repr_stats=repr_stats, bias_report=bias_report,
          worst_report=worst_report)
    logger.info("=" * 78)
    try:
        import resource

        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        logger.info("Peak resident memory: %.2f GB", peak_gb)
    except Exception:  # pragma: no cover - platform dependent
        pass
    logger.info("Done. Results written to %s", os.path.abspath(args.results_dir))


def _process_noise(weights, op, b_p, forcing, valid, jitter):
    """
    Residual process-noise covariance ``Q`` for the memory model.

    ``Q`` is a *residual*, never a regressor (v2 Algorithm 3 step 5).

    :param weights: ``(T, r)`` weights.
    :param op: the memory operator.
    :param b_p: ``(r, ell)`` input matrix.
    :param forcing: ``(T, ell)`` inputs.
    :param valid: ``(T,)`` observation validity.
    :param jitter: diagonal conditioning.
    :return: ``(r, r)`` covariance.
    """
    from dbwm.dynamics.memory import build_lifted_design

    w_bar, targets, origins = build_lifted_design(weights, op.order, 1, valid)
    pred = np.einsum("jrs,njs->nr", op.blocks, w_bar)
    if b_p.shape[1]:
        pred = pred + forcing[origins] @ b_p.T
    resid = targets[0] - pred
    q = (resid.T @ resid) / max(resid.shape[0] - 1, 1)
    return q + jitter * np.eye(op.r)


def _score_forecasts(forecasts, w_true, ds, extractor, origins, horizon,
                     subspace=None):
    """
    Score every forecast variant in **pixel** space, per horizon.

    Latent RMSE is reported too, but pixel RMSE is the number that matters: it is
    what a user of the NDVI map actually experiences, and it folds in whatever the
    basis fails to represent.

    :param forecasts: ``{name: rolling_forecast output}``.
    :param w_true: ``(T, r)`` GP-solved weights.
    :param ds: the dataset (for masks and de-normalisation).
    :param extractor: the GP state extractor (for decoding).
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param subspace: if the dynamics ran reduced, the :class:`LatentSubspace` used
        to lift predictions back to ``R^r`` before decoding.
    :return: ``{name: {"latent_rmse": (H,), "pixel_rmse": (H,)}}``.
    """
    mask_flat = ds.common_mask()
    scale = float(ds.std[0])
    out = {}
    for name, fc in forecasts.items():
        lat = np.zeros(horizon)
        pix = np.zeros(horizon)
        extra = {k: np.zeros(horizon) for k in ("bias", "ubrmse", "mae")}
        for h in range(horizon):
            ok = fc["valid"][:, h]
            org_idx = origins[ok]
            tgt_idx = org_idx + h + 1
            obs = ds.observed[tgt_idx]
            if not obs.any():
                lat[h] = pix[h] = np.nan
                for v in extra.values():
                    v[h] = np.nan
                continue
            pred_w = fc["mean"][ok, h][obs]
            if subspace is not None:
                # Lift back to R^r, carrying the origin's UNMODELLED component
                # forward unchanged. The operator says nothing about the
                # orthogonal complement, so persisting it is the honest default;
                # plain reconstruct() would instead assert it collapses to zero
                # and charge the full truncation error to every horizon.
                pred_w = subspace.reconstruct_with_complement(
                    pred_w, w_true[org_idx[obs]]
                )
            true_w = w_true[tgt_idx[obs]]
            lat[h] = float(np.sqrt(np.mean((pred_w - true_w) ** 2)))
            pred_pix = extractor.decode(pred_w)[:, mask_flat]
            true_pix = ds.frames[tgt_idx[obs]].reshape(obs.sum(), -1)[:, mask_flat]
            m = field_error_metrics(pred_pix, true_pix, scale=scale)
            pix[h] = m["rmse"]
            for k in extra:
                extra[k][h] = m[k]
        out[name] = {
            "latent_rmse": lat, "pixel_rmse": pix,
            "pixel_bias": extra["bias"], "pixel_ubrmse": extra["ubrmse"],
            "pixel_mae": extra["mae"],
        }
    return out


def _save(results_dir, cfg, ds, pretest, acf, op, b_p, q, cert, rho, emission, gain,
          family, defect, cov_cmp, sweep, metrics, stats, train_metrics, gp,
          rank_report, transient, budget=None, persistence=None, blend=None,
          basis=None, pixel_metrics=None, repr_stats=None, bias_report=None,
          worst_report=None):
    """Persist the operators, diagnostics and a human-readable summary."""
    with open(os.path.join(results_dir, "model.pkl"), "wb") as fh:
        pickle.dump(
            {
                "memory_blocks": op.blocks,
                "memory_coefficients": op.coefficients,
                "parameterization": op.parameterization,
                "b_p": b_p,
                "q": q,
                "emission_c": emission.c,
                "emission_r": emission.r_cov,
                "theta": family.theta,
                "sigma": family.sigma,
                "nu": family.nu,
                "config": cfg.to_dict(),
            },
            fh,
        )
    from dbwm.data.modality import get_modality

    spec = get_modality(cfg.data.modality)
    summary = {
        "config": cfg.to_dict(),
        # Stated at the top level so a summary.json can never be read on the
        # wrong scale: every metric in this file is in `units`.
        "modality": spec.label,
        "units": spec.units,
        "pixel_size_m": float(ds.pixel_size),
        "crs": ds.crs,
        "basis_training": train_metrics,
        "gp_reconstruction_r2_median": float(
            np.nanmedian(gp["reconstruction_r2"][ds.observed])
        ),
        # The error budget is the first thing to read when a target is missed:
        # it names the binding stage and the floor the forecast cannot get under.
        # Persisting it makes that decision reproducible from the artefacts
        # rather than only from the console.
        "error_budget": budget,
        # The variance term the predictive band used to omit, and the
        # per-pixel bias that goes with it.
        "bias_attribution": bias_report,
        "worst_pixel_attribution": worst_report,
        "representation_residual": (
            None if repr_stats is None else {
                "rms_in_sample": repr_stats["rms"],
                "mean_bias": repr_stats["mean_bias"],
                "n_dates": repr_stats["n_dates"],
                # The held-out value is the forecast floor; the in-sample one is
                # near zero by construction for a data-driven basis.
                "rms_holdout": (
                    None if basis is None
                    else float(basis.rmse_holdout) * float(ds.std[0])
                ),
                "generalisation_gap": (
                    None if basis is None else float(basis.generalisation_gap)
                ),
                "eof_selection": None if basis is None else basis.selection,
            }
        ),
        "persistence_rmse_by_horizon": (
            None if persistence is None else np.asarray(persistence).tolist()
        ),
        "persistence_blend": blend,
        "augmented_basis": None if basis is None else {
            "r_learned": int(basis.r_learned),
            "q_empirical": int(basis.q_empirical),
            "residual_variance_explained": float(basis.explained),
            "reconstruction_r2_psi_only": float(basis.r2_before),
            "reconstruction_r2_augmented": float(basis.r2_after),
        },
        "whiteness_pretest": {
            "reject": bool(pretest["reject"]),
            "verdict": pretest["verdict"],
            "fraction_significant": pretest["fraction_significant"],
            "hosking_p": pretest["hosking"]["p_value"],
        },
        "residual_acf": {
            "lags": acf["lags"].tolist(),
            "mean_abs_acf": acf["mean_abs_acf"].tolist(),
            "band": float(acf["band"]),
        },
        "weight_rank": {
            "rank_95": rank_report.rank_95,
            "rank_99": rank_report.rank_99,
            "numerical_rank": rank_report.numerical_rank,
            "r": rank_report.r,
            "over_parameterised": bool(rank_report.over_parameterised),
            "suggested_r": rank_report.suggested_r(),
        },
        "transient": {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in transient.items()},
        "memory": {
            "order": int(op.order),
            "rho_lifted": float(rho),
            "norm_budget": float(op.norm_budget()),
            "observability": {k: (float(v) if isinstance(v, float) else v)
                              for k, v in cert.items()},
        },
        "emission": {
            "r2": emission.r2.tolist(),
            "r2_holdout": None if emission.r2_holdout is None else emission.r2_holdout.tolist(),
            "names": emission.names,
            "information_gain": gain,
        },
        "horizons": {
            "nu": family.nu.tolist(),
            "defect": defect["defect"].tolist(),
            "defect_normalized": defect["defect_normalized"].tolist(),
            "structural_residual": defect["structural_residual"],
            "coverage": {k: np.asarray(v).tolist() for k, v in cov_cmp.items()},
        },
        "memory_depth_sweep": (
            None if sweep is None
            else {k: np.asarray(v).tolist() for k, v in sweep.items()}
        ),
        "forecast_metrics": {
            k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            for k, v in metrics.items()
        },
        # Where the error lives, not just how large it is: the best- and
        # worst-predicted pixel per lead, pooled over every origin. Persisted
        # so the figure can be re-read as numbers without re-running the run.
        "per_pixel_error": (
            None if not pixel_metrics
            else {str(k): v for k, v in pixel_metrics.items()}
        ),
    }
    if stats:
        summary["seasonal_statistics"] = {
            split: {
                season: {
                    **rep.statistics.summary(),
                    "separability_rejected": (
                        None if rep.separability is None else bool(rep.separability.reject)
                    ),
                    "separability_p": (
                        None if rep.separability is None else rep.separability.p_value
                    ),
                    "temporal_correlation": (
                        None if rep.separability is None
                        else rep.separability.temporal_correlation
                    ),
                }
                for season, rep in per.items()
            }
            for split, per in stats.items()
        }
        np.savez_compressed(
            os.path.join(results_dir, "seasonal_maps.npz"),
            **{
                f"{split}_{season}_{field}": getattr(rep.statistics, field)
                for split, per in stats.items()
                for season, rep in per.items()
                for field in ("mean", "variance", "cv", "cv_robust")
            },
        )
    with open(os.path.join(results_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


def _make_figures(fig_dir, cfg, ds, forecasts, metrics, op, sweep, transient,
                  stats, forecast_dates, extractor=None, subspace=None,
                  origins=None, w_true=None, plot_dates=None, per_pixel=True,
                  repr_var=None):
    """
    Render the v4 figure set.

    Matches the legacy v2 outputs (triptych, error curve, Koopman spectrum) and
    adds the diagnostics that only exist because of the memory / multi-horizon
    extension: the horizon curve against climatology, the Mori-Zwanzig depth, and
    the transient-growth plot that catches a non-normal operator passing
    ``rho <= 1`` while still amplifying the forecast.

    The per-date products -- the ``t+1..t+H`` triptychs and the per-pixel error
    maps -- are delegated to :mod:`dbwm.evaluation.forecast_figures` so that this
    script, ``infer_v4`` and ``forecast_v4`` render byte-identical figures from
    the same decode path.

    :param fig_dir: output directory.
    :param cfg: experiment config.
    :param ds: the dataset.
    :param forecasts: ``{variant: rolling_forecast_multi output}``.
    :param metrics: ``{variant: {"pixel_rmse": ...}}``.
    :param op: the identified memory operator.
    :param sweep: memory-depth sweep output, or ``None``.
    :param transient: transient-amplification report.
    :param stats: ``{split: {season: SeasonReport}}``.
    :param forecast_dates: origin dates aligned to the forecast arrays.
    :param extractor: GP state extractor, needed to decode the triptych panels.
    :param subspace: latent subspace, if the dynamics ran reduced.
    :param origins: forecast origin indices.
    :param plot_dates: calendar indices of the origins to draw triptychs for, or
        ``None`` for the latest fully observed one.
    :param per_pixel: also write the aggregated per-pixel error maps.
    :param repr_var: ``(n_pixels,)`` representation variance for the interval
        panel, in physical units.
    :return: ``(paths written, {lead: per-pixel summary})``.
    """
    import numpy as np

    from dbwm.dynamics.memory import lifted_spectrum

    os.makedirs(fig_dir, exist_ok=True)
    out = []
    name = cfg.name
    units = cfg.data.modality.upper()

    # Climatology reference: predicting the training mean. Without it the horizon
    # curve cannot be read as "useful" or "worse than doing nothing".
    train_obs = np.array([d < dt.date.fromisoformat(cfg.seasons.split_date)
                          for d in ds.dates]) & ds.observed
    mask_flat = ds.common_mask()
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]
    test_obs = (~train_obs) & ds.observed
    clim = None
    if train_obs.any() and test_obs.any():
        clim_mean = flat[train_obs].mean(axis=0)
        clim = float(
            np.sqrt(np.mean((flat[test_obs] - clim_mean) ** 2)) * float(ds.std[0])
        )
    else:
        # No held-out dates on this side of the split: a climatology reference
        # would be an average over nothing. Omit the line rather than draw NaN.
        logger.warning(
            "Climatology reference skipped: %d training and %d test observations "
            "either side of %s.",
            int(train_obs.sum()), int(test_obs.sum()), cfg.seasons.split_date,
        )

    primary = "recursive" if "recursive" in forecasts else sorted(forecasts)[0]
    fc = forecasts[primary]
    origins_arr = np.asarray(origins if origins is not None else fc.get("origins", []))
    pixel_metrics = {}
    if extractor is not None and origins_arr.size:
        rendered = forecast_figures.render_horizon_figures(
            fig_dir, name, fc, origins_arr, ds, extractor,
            subspace=subspace, w_origin=w_true, horizon=cfg.horizons.horizon,
            units=units, variant=primary, plot_origins=plot_dates,
            per_pixel=per_pixel, repr_var=repr_var,
        )
        out.extend(rendered["figures"])
        pixel_metrics = rendered["pixel_metrics"]

    out.append(v4_plots.plot_rmse_by_horizon(
        metrics, os.path.join(fig_dir, f"{name}_rmse_by_horizon.png"),
        key="pixel_ubrmse", climatology=clim, units=units,
    ))
    out.append(v4_plots.plot_koopman_spectrum(
        lifted_spectrum(op), os.path.join(fig_dir, f"{name}_koopman.png"),
    ))
    out.append(v4_plots.plot_transient_growth(
        transient, os.path.join(fig_dir, f"{name}_transient.png"),
    ))
    if sweep is not None:
        out.append(v4_plots.plot_memory_depth(
            sweep, os.path.join(fig_dir, f"{name}_memory_depth.png"),
        ))
    for split, per in (stats or {}).items():
        out.append(v4_plots.plot_seasonal_statistics(
            per, os.path.join(fig_dir, f"{name}_seasonal_{split}.png"), split,
            units=units,
        ))
        for season, rep in per.items():
            cg = rep.covariograms.get("anomaly")
            if cg is not None:
                out.append(v4_plots.plot_separability(
                    cg, os.path.join(fig_dir, f"{name}_separability_{split}_{season}.png"),
                    season, split,
                ))
    return [p for p in out if p], pixel_metrics

if __name__ == "__main__":
    main()
