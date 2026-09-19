"""
DB-WM v4 vs E-GP / Kernel Observers -- one matched-protocol comparison.

What this replaces, and why
---------------------------
The previous version of this script was a 72-cell ablation grid over the DB-WM
memory order, parameterization, estimator and subspace flag, with the E-GP bolted
on at the end. On the real record it ran for five hours and produced a table that
could not answer the question it was for, for three reasons:

**The grid axes were inert.** Every one of the 72 cells reported
``Persistence blend selected: s = 0.00`` -- the memory kernel was shrunk away in
all of them because it never beat persistence on held-out data. With ``A_j = 0``
for ``j >= 1`` the memory order and the parameterization have nothing to act on,
which is why ``L1_s1``, ``L1_s2`` and ``L1_s3`` returned byte-identical numbers.
The grid was 72 near-duplicates of one model plus noise from the ``Theta_h`` fit.

**The comparison was not like-for-like.** DB-WM was scored with a climatology
offset, 96 empirical basis modes, a persistence-blended operator and a POD
subspace; the E-GP was handed the raw field and none of that scaffolding. A 10x
gap measured that way is mostly protocol, not method.

**The E-GP was mis-configured in three specific ways**, all of them in our code
rather than in the paper -- see :mod:`dbwm.baselines.egp`. It entered each
forecast having forgotten the history DB-WM had assimilated; it used the cyclic
index as the sensor *count* when Proposition 2 gives it as a lower *bound*,
leaving 290 of 300 centres unobserved; and its bandwidth search hit the lower
edge of an absolute grid because wider kernels made the Gram matrix singular.

So this script does one thing: run **both methods once, under the same protocol,
and report the full metric set** that ``run_ndvi_v4.py`` reports.

The comparison ladder
---------------------
=========================  ====================================================
row                        what it is
=========================  ====================================================
``climatology``            predict the training mean field. The floor.
``persistence``            predict today's field. The bar that matters.
``EGP_forecast``           E-GP filtered on the FULL frame up to the origin,
                           then free-running t+1..t+6. **The like-for-like
                           comparator**: same information as DB-WM, no NDVI
                           after the origin.
``EGP_sensors``            same, but the filter sees only the N sensing pixels.
                           The gap to ``EGP_forecast`` is what the paper's
                           sensor economy costs on this record.
``EGP_feedback``           N-sensor filter that keeps correcting *through* the
                           horizon. Filtering, not forecasting -- it reads part
                           of the frame it is scored on. Labelled, never the
                           headline.
``DBWM_recursive``         the lifted memory filter, iterated forecast.
``DBWM_direct``            the same filter, direct horizon family ``Theta_h``.
=========================  ====================================================

Every row is scored by the same :func:`~dbwm.evaluation.metrics.field_error_metrics`
on the same origins and the same valid pixels, and reported as ubRMSE / MAE /
RMSE / bias per horizon, alongside the reconstruction error of whichever basis
produced it and the error budget that bounds it.

Usage
-----
    python -m experiments.compare_v4 --smoke
    python -m experiments.compare_v4 --ndvi-dir <dir> --weather-csv <csv> \\
        --r 256 --memory-order 7 --horizon 6 --epochs 300 --eof-modes 256
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Dict, List, Optional

import numpy as np

from dbwm.platform import preflight

_PREFLIGHT = preflight()

import jax.numpy as jnp  # noqa: E402

from dbwm.config import default_config, smoke_config  # noqa: E402
from dbwm.data.seasons import purge_boundary_windows  # noqa: E402
from dbwm.dynamics.emission import fit_emission  # noqa: E402
from dbwm.dynamics.memory import (  # noqa: E402
    build_lifted_design, calibrate_persistence_blend, forecast_gain,
    identify_memory, is_persistence,
)
from dbwm.dynamics.multihorizon import fit_horizon_family  # noqa: E402
from dbwm.dynamics.subspace import build_subspace  # noqa: E402
from dbwm.evaluation import v4_plots  # noqa: E402
from dbwm.evaluation.error_budget import error_budget, persistence_rmse  # noqa: E402
from dbwm.evaluation.metrics import field_error_metrics  # noqa: E402
from dbwm.gp.empirical_basis import build_augmented_basis  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.inference.lifted_kalman import (  # noqa: E402
    LiftedSystem, rolling_forecast_multi,
)
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.training.gp_trainer import train_spatial_basis  # noqa: E402
from experiments._v4_common import (  # noqa: E402
    add_data_args, add_model_args, apply_model_args, load_inputs,
)

logger = configure_logger(level="INFO", name="dbwm.compare")

METRIC_KEYS = ("ubrmse", "mae", "rmse", "bias")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DB-WM v4 vs E-GP / Kernel Observers, matched protocol"
    )
    add_data_args(p)
    p.add_argument("--results-dir", default="./_v4_compare")
    # DB-WM configuration -- the SAME knobs run_ndvi_v4 takes, so the compared
    # model is the model that pipeline produces rather than a variant of it.
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--memory-order", type=int, default=None, help="L")
    p.add_argument("--horizon", type=int, default=None, help="H")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--parameterization", default=None,
                   choices=["s1", "s2", "s3", "unstructured"])
    p.add_argument("--estimator", default=None,
                   choices=["shrunk", "direct", "iterated"])
    p.add_argument("--subspace-energy", type=float, default=0.999)
    p.add_argument("--max-k", type=int, default=None)
    # E-GP configuration.
    p.add_argument("--egp-centers", type=int, default=300,
                   help="M, the dictionary size (the paper uses 300-600)")
    p.add_argument("--egp-meas", default="rational",
                   choices=["rational", "random"],
                   help="sensor placement: Algorithms 2-3, or uniform random")
    p.add_argument("--egp-sensors", type=int, default=None,
                   help="N sensors. Default: search upward for observability "
                        "(the cyclic index is a LOWER BOUND, not a count)")
    p.add_argument("--egp-sensor-rule", default="observable",
                   choices=["observable", "cyclic_index"],
                   help="'cyclic_index' reproduces Proposition 2 literally; it "
                        "is an ablation, and on this record it is unobservable")
    p.add_argument("--egp-no-climatology", action="store_true",
                   help="ablate the shared climatology from the E-GP only, to "
                        "show what the protocol asymmetry was worth")
    p.add_argument("--forecast-stride", type=int, default=1)
    p.add_argument("--no-plots", action="store_true")
    add_model_args(p)
    return p.parse_args()


def _score(pred_flat, tgt_idx, flat, scale) -> Dict[str, float]:
    """
    Score decoded predictions against truth, in physical units.

    :param pred_flat: ``(n, n_pixels)`` predictions on valid pixels.
    :param tgt_idx: ``(n,)`` target calendar indices.
    :param flat: ``(T, n_pixels)`` truth on valid pixels.
    :param scale: normalisation std.
    :return: the metric dict.
    """
    return field_error_metrics(pred_flat, flat[tgt_idx], scale=scale)


def _per_horizon(name, decode_fn, origins, horizon, ds, flat, scale) -> Dict:
    """
    Evaluate one method at every horizon on the shared origins.

    ``decode_fn(h)`` returns ``(mask, predictions)`` for horizon ``h``: the
    boolean origin mask it could score, and the decoded fields for those origins.
    Routing every method through one scorer is what makes the table comparable --
    a row computed with its own masking or its own de-normalisation is not
    measuring the same thing as its neighbours.

    :param name: row label, for the log.
    :param decode_fn: callable described above.
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param ds: the dataset.
    :param flat: ``(T, n_pixels)`` truth on valid pixels.
    :param scale: normalisation std.
    :return: ``{metric: (H,) array, "n": (H,) counts}``.
    """
    acc = {k: np.full(horizon, np.nan) for k in METRIC_KEYS}
    counts = np.zeros(horizon, dtype=int)
    for h in range(horizon):
        mask, pred = decode_fn(h)
        if pred is None or not np.any(mask):
            continue
        tgt = np.asarray(origins)[mask] + h + 1
        keep = (tgt < ds.n_steps) & ds.observed[np.clip(tgt, 0, ds.n_steps - 1)]
        if not keep.any():
            continue
        m = _score(pred[keep], tgt[keep], flat, scale)
        for k in METRIC_KEYS:
            acc[k][h] = m[k]
        counts[h] = int(keep.sum())
    logger.info("%-16s ubRMSE %s", name,
                np.array2string(acc["ubrmse"], precision=4, suppress_small=True))
    logger.info("%-16s MAE    %s", name,
                np.array2string(acc["mae"], precision=4, suppress_small=True))
    out = {k: acc[k] for k in METRIC_KEYS}
    out["n"] = counts
    return out


def _reconstruction(pred_flat, truth_flat, scale) -> Dict[str, float]:
    """
    Encode-decode fidelity of a basis, the floor under every forecast it makes.

    Reported for both methods because v2 Theorem 4.2 multiplies it by the
    operator gain at every horizon: a basis that reconstructs worse cannot be
    rescued by better dynamics, so a forecast table without it cannot say whether
    a gap came from the representation or from the model of time.

    :param pred_flat: ``(T, n_pixels)`` round-tripped fields.
    :param truth_flat: ``(T, n_pixels)`` observed fields.
    :param scale: normalisation std.
    :return: dict with ``r2`` and ``rmse``.
    """
    ss_res = np.sum((pred_flat - truth_flat) ** 2, axis=1)
    ss_tot = np.sum(
        (truth_flat - truth_flat.mean(axis=1, keepdims=True)) ** 2, axis=1
    )
    ok = ss_tot > 1e-30
    return {
        "r2": float(np.median(1.0 - ss_res[ok] / ss_tot[ok])) if ok.any() else float("nan"),
        "rmse": float(np.sqrt(np.mean((pred_flat - truth_flat) ** 2)) * scale),
    }


def run_dbwm(cfg, inp, args, origins, flat, mask_flat) -> Dict:
    """
    Train and evaluate DB-WM v4 exactly as ``run_ndvi_v4`` would.

    :param cfg: the experiment config.
    :param inp: the loaded inputs.
    :param args: parsed arguments.
    :param origins: shared forecast origins.
    :param flat: ``(T, n_pixels)`` truth on valid pixels.
    :param mask_flat: ``(n_pixels,)`` validity.
    :return: rows plus diagnostics.
    """
    ds, train_sel = inp.ds, inp.train_sel
    scale = float(ds.std[0])

    logger.info("Training the spatial basis Psi (r = %d, %d epochs) ...",
                cfg.basis.r, cfg.training.n_epochs)
    model, params, _ = train_spatial_basis(
        cfg, ds.frames[train_sel], ds.valid_mask[train_sel], ds.observed[train_sel],
        ds.coords, seed=args.seed,
        rollout_steps=cfg.training.rollout_steps,
        lambda_rollout=cfg.training.lambda_rollout,
        tf_threshold=cfg.teacher_forcing_threshold(),
        use_teacher_forcing=cfg.training.teacher_forcing,
        norm_std=scale,
    )
    sigma_eps2 = float(model.apply(params, method=model.sigma_eps2))
    psi = np.concatenate(
        [np.asarray(model.apply(params, jnp.asarray(ds.coords[i:i + 20000]),
                                method=model.features))
         for i in range(0, ds.coords.shape[0], 20000)], axis=0)
    aug = build_augmented_basis(
        psi, ds.frames, mask_flat, train_sel & ds.observed,
        n_modes=cfg.basis.eof_modes, energy=cfg.basis.eof_energy,
        ridge=sigma_eps2, use_climatology=cfg.basis.use_climatology,
    )
    extractor = build_extractor(
        lambda c: model.apply(params, jnp.asarray(c), method=model.features),
        ds.coords, sigma_eps2, ds.static_mask,
        extra_features=aug.phi[:, aug.r_learned:] if aug.q_empirical else None,
        offset=aug.offset if cfg.basis.use_climatology else None,
    )
    gp = extractor.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    w, w_cov = gp["weights"], gp["covariances"]

    obs = ds.observed
    recon = _reconstruction(
        extractor.decode(w[obs])[:, mask_flat], flat[obs], scale
    )

    subspace = None
    if args.subspace_energy and args.subspace_energy > 0:
        subspace = build_subspace(
            w, ds.observed & train_sel, args.subspace_energy, args.max_k
        )
        if subspace.k >= subspace.r:
            subspace = None
    w_dyn = subspace.project(w) if subspace is not None else w
    cov_dyn = subspace.project_covariance(w_cov) if subspace is not None else w_cov

    op, b_p = identify_memory(
        w_dyn[train_sel], cfg.memory.order, inp.forcing[train_sel],
        cfg.memory.parameterization, cfg.dynamics.ridge_mu,
        cfg.dynamics.quiescent_threshold, cfg.memory.reduced_rank,
        valid=ds.observed[train_sel], increment=cfg.memory.increment,
    )
    op, blend = calibrate_persistence_blend(
        op, b_p, w_dyn[train_sel], inp.forcing[train_sel],
        ds.observed[train_sel], cfg.horizons.horizon, gain_max=cfg.memory.gain_max,
    )
    if is_persistence(op):
        logger.info(
            "The DB-WM operator reduced to persistence on held-out data. That is "
            "the finding, not a failure to report: the memory kernel earned "
            "nothing beyond the basis on this record."
        )

    w_bar, targets, org = build_lifted_design(
        w_dyn[train_sel], cfg.memory.order, 1, ds.observed[train_sel]
    )
    pred = np.einsum("jrs,njs->nr", op.blocks, w_bar)
    if b_p.shape[1]:
        pred = pred + inp.forcing[train_sel][org] @ b_p.T
    resid = targets[0] - pred
    q = (resid.T @ resid) / max(resid.shape[0] - 1, 1)
    q += cfg.dynamics.process_noise_jitter * np.eye(op.r)

    obs_train = train_sel & ds.observed
    em = fit_emission(
        w_dyn[obs_train], inp.measurement[obs_train],
        list(inp.weather.measurement_names), cfg.weather.emission_ridge,
        cfg.weather.full_noise_covariance,
    )
    family = fit_horizon_family(
        w_dyn[train_sel], op, b_p, cfg.horizons.horizon, inp.forcing[train_sel],
        ds.observed[train_sel], cfg.horizons.estimator, cfg.horizons.ridge_mu,
        nu_grid=cfg.horizons.nu_grid, nu_selection=cfg.horizons.nu_selection, q=q,
    )
    system = LiftedSystem(
        op=op, b_p=b_p, q=q,
        c_w=em.c if em.usable else None,
        r_w=em.r_cov if em.usable else None,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )
    forecasts = rolling_forecast_multi(
        system, w_dyn, ds.observed, origins, cfg.horizons.horizon,
        [{"name": "recursive", "mode": "recursive", "use_weather": True},
         {"name": "direct", "mode": "direct", "use_weather": True}],
        inp.forcing, inp.measurement, cov_dyn, sigma_eps2, family=family,
    )

    rows = {}
    for variant, fc in forecasts.items():
        def decode_fn(h, fc=fc):
            ok = fc["valid"][:, h]
            if not np.any(ok):
                return ok, None
            wv = fc["mean"][ok, h]
            if subspace is not None:
                wv = subspace.reconstruct_with_complement(
                    wv, w[np.asarray(origins)[ok]]
                )
            return ok, extractor.decode(wv)[:, mask_flat]
        rows["DBWM_" + variant] = _per_horizon(
            "DBWM_" + variant, decode_fn, origins, cfg.horizons.horizon,
            ds, flat, scale,
        )

    pers = persistence_rmse(
        ds.frames, mask_flat, ds.observed, origins, cfg.horizons.horizon, scale
    )
    budget = error_budget(
        extractor, w, ds.frames, mask_flat, ds.observed, scale,
        gain=float(np.max(forecast_gain(op, cfg.horizons.horizon))),
        subspace=subspace, persistence=float(pers[0]),
        units=cfg.modality_spec().units,
    )
    return {
        "rows": rows,
        "reconstruction": recon,
        "error_budget": budget,
        "persistence_blend": blend.get("blend"),
        "operator_is_persistence": bool(is_persistence(op)),
        "max_forecast_gain": float(np.max(forecast_gain(op, cfg.horizons.horizon))),
        "augmented_basis": {
            "r_learned": int(aug.r_learned),
            "q_empirical": int(aug.q_empirical),
            "residual_variance_explained": float(aug.explained),
            "reconstruction_r2_psi_only": float(aug.r2_before),
            "reconstruction_r2_augmented": float(aug.r2_after),
        },
        "emission_usable": bool(em.usable),
        "subspace_k": int(subspace.k) if subspace is not None else None,
    }


def run_egp(cfg, inp, args, origins, flat, mask_flat) -> Dict:
    """
    Fit and evaluate the E-GP under the same protocol.

    :param cfg: the experiment config.
    :param inp: the loaded inputs.
    :param args: parsed arguments.
    :param origins: shared forecast origins.
    :param flat: ``(T, n_pixels)`` truth on valid pixels.
    :param mask_flat: ``(n_pixels,)`` validity.
    :return: rows plus diagnostics.
    """
    from dbwm.baselines.egp import EGPBaseline, EGPConfig

    ds, train_sel = inp.ds, inp.train_sel
    scale = float(ds.std[0])
    coords = np.asarray(ds.coords, dtype=np.float64)[mask_flat]

    egp = EGPBaseline(EGPConfig(
        n_centers=min(args.egp_centers, coords.shape[0] // 4),
        meas_type=args.egp_meas,
        n_measurements=args.egp_sensors,
        sensor_rule=args.egp_sensor_rule,
        use_climatology=(cfg.basis.use_climatology
                         and not args.egp_no_climatology),
        seed=args.seed,
    ))
    egp.fit(coords, flat[train_sel & ds.observed])

    obs = ds.observed
    recon = _reconstruction(egp.decode(egp.encode(flat[obs])), flat[obs], scale)
    logger.info(
        "E-GP reconstruction: R2 %.4f (RMSE %.4f %s) -- the floor under every "
        "E-GP forecast, the counterpart of DB-WM's augmented-basis R2.",
        recon["r2"], recon["rmse"], cfg.modality_spec().units,
    )

    rows = {}
    for name, kw in (
        ("EGP_forecast", dict(assimilate="full", feedback=False)),
        ("EGP_sensors", dict(assimilate="sensors", feedback=False)),
        ("EGP_feedback", dict(assimilate="sensors", feedback=True)),
    ):
        fc = egp.filtered_rolling_forecast(
            flat, list(origins), cfg.horizons.horizon, ds.observed, **kw
        )

        def decode_fn(h, fc=fc):
            ok = fc["valid"][:, h]
            return ok, (fc["mean"][ok, h] if np.any(ok) else None)

        rows[name] = _per_horizon(
            name, decode_fn, origins, cfg.horizons.horizon, ds, flat, scale
        )
    return {"rows": rows, "reconstruction": recon, "report": dict(egp.report)}


def reference_rows(ds, origins, horizon, train_sel, flat, scale) -> Dict:
    """
    Climatology and persistence, through the same scorer as everything else.

    :param ds: the dataset.
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param train_sel: training mask.
    :param flat: ``(T, n_pixels)`` truth on valid pixels.
    :param scale: normalisation std.
    :return: the two reference rows.
    """
    clim = flat[train_sel & ds.observed].mean(axis=0)
    origins = np.asarray(origins)
    out = {}
    for name in ("climatology", "persistence"):
        def decode_fn(h, name=name):
            ok = np.ones(origins.size, dtype=bool)
            if name == "climatology":
                return ok, np.repeat(clim[None], origins.size, axis=0)
            return ok, flat[origins]
        out[name] = _per_horizon(name, decode_fn, origins, horizon, ds, flat, scale)
    return out


def main():
    """Run the matched comparison."""
    args = build_args()
    cfg = smoke_config() if args.smoke else default_config()
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
    if args.archive_start:
        cfg.seasons.archive_start = args.archive_start
    if args.archive_end:
        cfg.seasons.archive_end = args.archive_end
    if args.split_date:
        cfg.seasons.split_date = args.split_date
    apply_model_args(cfg, args)
    cfg.sync_input_dim()
    os.makedirs(args.results_dir, exist_ok=True)

    inp = load_inputs(cfg, args, logger)
    ds = inp.ds
    scale = float(ds.std[0])
    mask_flat = ds.common_mask()
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]

    # ONE set of origins for every row. Different origin sets are the easiest way
    # to make an unfair table without noticing.
    _, scorable = purge_boundary_windows(
        inp.train_idx, inp.test_idx, cfg.memory.order
    )
    origins = np.array([
        t for t in scorable[:: max(args.forecast_stride, 1)]
        if t + cfg.horizons.horizon < ds.n_steps and ds.observed[t]
    ])
    if origins.size == 0:
        raise SystemExit("No scorable test origins in this window.")
    logger.info("=" * 78)
    logger.info(
        "Scoring every method on the SAME %d test origins, the same %d valid "
        "pixels, and the same metric function.",
        origins.size, int(mask_flat.sum()),
    )

    results: Dict[str, Dict] = {}
    logger.info("-" * 78)
    results.update(reference_rows(
        ds, origins, cfg.horizons.horizon, inp.train_sel, flat, scale
    ))

    logger.info("-" * 78)
    logger.info("E-GP / KERNEL OBSERVERS")
    egp = run_egp(cfg, inp, args, origins, flat, mask_flat)
    results.update(egp["rows"])

    logger.info("-" * 78)
    logger.info("DB-WM v4")
    dbwm = run_dbwm(cfg, inp, args, origins, flat, mask_flat)
    results.update(dbwm["rows"])

    # ---------------- the table ---------------- #
    logger.info("=" * 78)
    logger.info(
        "COMPARISON  (physical %s; ubRMSE is the headline)",
        cfg.modality_spec().units,
    )
    hdr = "{:<16} {:>8} {:>8} {:>8} {:>9} {:>9}".format(
        "method", "ubRMSE1", "MAE1", "RMSE1", "bias1", "ubRMSE6"
    )
    logger.info(hdr)
    logger.info("-" * len(hdr))
    order = sorted(results, key=lambda k: np.nan_to_num(results[k]["ubrmse"][0], nan=9e9))
    last = cfg.horizons.horizon - 1
    for k in order:
        r = results[k]
        logger.info(
            "{:<16} {:>8.4f} {:>8.4f} {:>8.4f} {:>+9.4f} {:>9.4f}".format(
                k, r["ubrmse"][0], r["mae"][0], r["rmse"][0], r["bias"][0],
                r["ubrmse"][last],
            )
        )
    logger.info("-" * len(hdr))
    logger.info(
        "reconstruction R2 -- DB-WM %.4f (RMSE %.4f) | E-GP %.4f (RMSE %.4f)",
        dbwm["reconstruction"]["r2"], dbwm["reconstruction"]["rmse"],
        egp["reconstruction"]["r2"], egp["reconstruction"]["rmse"],
    )
    logger.info(
        "E-GP: M = %s centres, sigma = %.4f (%.2f spacings), N = %s sensors "
        "(%s rule, cyclic index %s), rank(O) = %s/%s",
        egp["report"].get("n_centers"), egp["report"].get("lengthscale", float("nan")),
        egp["report"].get("bandwidth_over_spacing") or float("nan"),
        egp["report"].get("n_measurements"), egp["report"].get("sensor_rule"),
        egp["report"].get("cyclic_index"), egp["report"].get("observability_rank"),
        egp["report"].get("n_centers"),
    )
    logger.info(
        "DB-WM: persistence blend s = %s, max gain %.3f, %d EOF modes, k = %s",
        dbwm["persistence_blend"], dbwm["max_forecast_gain"],
        dbwm["augmented_basis"]["q_empirical"], dbwm["subspace_k"],
    )

    payload = {
        "origins": int(origins.size),
        "valid_pixels": int(mask_flat.sum()),
        "config": cfg.to_dict(),
        "metrics": {
            k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            for k, v in results.items()
        },
        "dbwm": {k: v for k, v in dbwm.items() if k != "rows"},
        "egp": {k: v for k, v in egp.items() if k != "rows"},
    }
    with open(os.path.join(args.results_dir, "comparison.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    if not args.no_plots:
        v4_plots.plot_rmse_by_horizon(
            {k: {"pixel_rmse": np.asarray(v["ubrmse"])} for k, v in results.items()},
            os.path.join(args.results_dir, "ubrmse_by_horizon.png"),
            climatology=float(results["climatology"]["ubrmse"][0]),
            units=cfg.modality_spec().units,
        )
        v4_plots.plot_ablation_bars(
            {k: {"pixel_rmse_h1": float(v["ubrmse"][0])} for k, v in results.items()},
            os.path.join(args.results_dir, "ubrmse_h1.png"),
        )
    logger.info("Written to %s", os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
