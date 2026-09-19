"""
Train a DB-WM v4 model and checkpoint it.

Mirrors ``experiments/train_dbwm.py`` for the memory-augmented, multi-horizon
model: fit the spatial basis, solve the GP posterior per date, identify the memory
kernel, the weather emission and the direct horizon family, report the theoretical
certificates, and write a ``.pkl`` that ``experiments/infer_v4.py`` can forecast
from without re-touching the training split.

What is checkpointed and why
----------------------------
The v2 checkpoint stores ``(params, A, B, Q)``. The v4 model needs more, and every
extra field is load-bearing at inference time:

``blocks``
    the ``L`` memory blocks ``A_1..A_L``, not one ``A``. Storing only ``A_1`` would
    forecast a memoryless model that happens to have been fitted with memory.
``subspace``
    the POD basis. The dynamics live in ``R^k`` and the basis decodes from ``R^r``,
    so without it the saved operator cannot be connected to the saved ``Psi``.
``family``
    the direct horizon operators ``Theta_h``. Re-deriving them from ``A`` at
    inference would give the *iterated* predictor, which v3 Prop. 2.13 shows
    under-covers -- silently reporting optimistic intervals.
``c``/``r_cov``
    the weather emission. Rebuilding it from the test split would leak.

Usage
-----
    python -m experiments.train_v4 --ndvi-dir <dir> --weather-csv <csv>
    python -m experiments.train_v4 --smoke
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from dbwm.platform import ensure_working_backend, preflight

_PREFLIGHT = preflight()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dbwm.config import default_config, smoke_config  # noqa: E402
from dbwm.dynamics.conditioning import (  # noqa: E402
    log_rank_report, log_transient, transient_amplification, weight_rank,
)
from dbwm.dynamics.diagnostics import (  # noqa: E402
    one_step_residuals, whiteness_pretest,
)
from dbwm.dynamics.emission import fit_emission, information_gain  # noqa: E402
from dbwm.dynamics.memory import (  # noqa: E402
    build_lifted_design, calibrate_persistence_blend, enforce_spectral_radius,
    enforce_stability, forecast_gain, identify_memory, lifted_spectrum,
    observability_certificate,
)
from dbwm.dynamics.multihorizon import fit_horizon_family  # noqa: E402
from dbwm.dynamics.subspace import build_subspace  # noqa: E402
from dbwm.evaluation.geotiff_export import representation_variance  # noqa: E402
from dbwm.gp.empirical_basis import build_augmented_basis  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.training.gp_trainer import train_spatial_basis  # noqa: E402
from experiments._v4_common import (  # noqa: E402
    add_data_args, add_model_args, apply_model_args, load_inputs,
    save_checkpoint,
)

logger = configure_logger(level="INFO", name="dbwm.train_v4")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train a DB-WM v4 model.")
    add_data_args(p)
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--memory-order", type=int, default=None, help="L")
    p.add_argument("--horizon", type=int, default=None, help="H")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--parameterization", default=None,
                   choices=["s1", "s2", "s3", "unstructured"])
    p.add_argument("--estimator", default=None,
                   choices=["shrunk", "direct", "iterated"])
    p.add_argument("--ckpt-dir", default="./_v4_ckpt")
    p.add_argument("--name", default=None, help="checkpoint stem")
    p.add_argument("--subspace-energy", type=float, default=0.995,
                   help="temporal variance the dynamics subspace retains; 0 disables")
    p.add_argument("--max-k", type=int, default=None)
    p.add_argument("--require-accelerator", action="store_true")
    add_model_args(p)
    return p.parse_args()


def report_certificates(op, rho_max, order, horizon):
    """
    Log the v3 guarantees, and say what a failure actually means.

    :param op: the stabilised memory operator.
    :param rho_max: stability cap.
    :param order: memory order ``L``.
    :param horizon: ``H``, used to set the transient-growth window.
    :return: ``(rho, certificate, transient)``.
    """
    spec = lifted_spectrum(op)
    rho = float(np.max(np.abs(spec)))
    logger.info("Lifted spectral radius rho(script-A) = %.4f (cap %.2f)", rho, rho_max)

    amp = transient_amplification(op.companion(), horizon=2 * horizon)
    log_transient(amp)

    cert = observability_certificate(op, np.eye(op.r))
    if cert["observable"]:
        logger.info("Thm 2.8: lifted pair observable (A_%d nonsingular, "
                    "no mode in ker Phi_X).", order)
    else:
        # Both causes are failures of Thm 2.8(i) -- A_{L-1} singular -- but they
        # call for opposite fixes, and confusing them sends the user the wrong way.
        logger.warning(
            "Thm 2.8(i) FAILS -- cause: %s. %s", cert["cause"],
            "A_0 is ITSELF rank deficient: r exceeds the rank the trajectory "
            "excites, so the deepest lag carries no independent information. This "
            "is NOT over-lagging -- it would fail at L = 1 too. Reduce r, or leave "
            "the dynamics subspace enabled."
            if cert["cause"] == "a0_rank_deficient" else
            "A_0 is well conditioned but A_{L-1} is not, so the memory order "
            "genuinely over-shoots what the record supports. Reduce L using the "
            "||D_h||-vs-L sweep.",
        )
    return rho, cert, amp


def main():
    """Train, identify, certify, checkpoint."""
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
    if args.name:
        cfg.name = args.name
    apply_model_args(cfg, args)
    cfg.sync_input_dim()

    backend = ensure_working_backend(allow_cpu_fallback=not args.require_accelerator)
    logger.info("Backend: %s", backend["backend"])

    inp = load_inputs(cfg, args, logger)
    ds, train_sel = inp.ds, inp.train_sel

    logger.info("Training the spatial basis: r = %d, %d epochs.",
                cfg.basis.r, cfg.training.n_epochs)
    model, params, metrics = train_spatial_basis(
        cfg, ds.frames[train_sel], ds.valid_mask[train_sel], ds.observed[train_sel],
        ds.coords, seed=args.seed,
        rollout_steps=cfg.training.rollout_steps,
        lambda_rollout=cfg.training.lambda_rollout,
        tf_threshold=cfg.teacher_forcing_threshold(),
        use_teacher_forcing=cfg.training.teacher_forcing,
        norm_std=float(ds.std[0]),
    )
    sigma_eps2 = float(model.apply(params, method=model.sigma_eps2))
    mask_static = ds.common_mask()
    # Close the representation floor before anything downstream sees w_t.
    psi = np.concatenate(
        [np.asarray(model.apply(params, jnp.asarray(ds.coords[i:i + 20000]),
                                method=model.features))
         for i in range(0, ds.coords.shape[0], 20000)], axis=0)
    aug = build_augmented_basis(
        psi, ds.frames, mask_static, train_sel & ds.observed,
        n_modes=cfg.basis.eof_modes, energy=cfg.basis.eof_energy,
        ridge=sigma_eps2, use_climatology=cfg.basis.use_climatology,
        select=cfg.basis.eof_select,
        holdout_fraction=cfg.basis.eof_holdout_fraction,
    )
    extra = aug.phi[:, aug.r_learned:] if aug.q_empirical else None
    offset = aug.offset if cfg.basis.use_climatology else None
    extractor = build_extractor(
        lambda c: model.apply(params, jnp.asarray(c), method=model.features),
        ds.coords, sigma_eps2, ds.static_mask,
        extra_features=extra, offset=offset,
    )
    gp = extractor.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    w, w_cov = gp["weights"], gp["covariances"]
    logger.info("GP posterior solved on %d observed dates; sigma_eps^2 = %.3e.",
                int(ds.observed.sum()), sigma_eps2)

    rank = weight_rank(w, ds.observed & train_sel)
    log_rank_report(rank)

    # The third variance term of the predictive band, estimated on TRAINING dates
    # and carried in the checkpoint so infer_v4/forecast_v4 report the same
    # interval this run would. Without it the band omits its dominant term.
    repr_stats = representation_variance(
        extractor, w, ds.frames, mask_static, train_sel & ds.observed,
        float(ds.std[0]),
    )
    logger.info(
        "Representation residual (training): RMS %.4f, mean bias %+.4f over %d "
        "dates -- added to the predictive variance at inference time.",
        repr_stats["rms"], repr_stats["mean_bias"], repr_stats["n_dates"],
    )

    subspace = None
    if args.subspace_energy and args.subspace_energy > 0:
        subspace = build_subspace(w, ds.observed & train_sel, args.subspace_energy,
                                  max_k=args.max_k)
        if subspace.k >= subspace.r:
            subspace = None
    w_dyn = subspace.project(w) if subspace is not None else w

    # v3 step 0, run BEFORE fitting any memory: the pretest asks whether the L=1
    # innovations are already white. If they are, Corollary 2.5.1 says the memory
    # kernel has nothing to explain and any L > 1 fit is estimating noise.
    op1, b1 = identify_memory(
        w_dyn[train_sel], 1, inp.forcing[train_sel], "unstructured",
        cfg.dynamics.ridge_mu, valid=ds.observed[train_sel],
    )
    resid = one_step_residuals(
        w_dyn[train_sel], op1.blocks[0], b1, inp.forcing[train_sel]
    )
    pre = whiteness_pretest(
        resid, cfg.memory.ljung_box_lags, cfg.memory.ljung_box_alpha,
        cfg.memory.ljung_box_projection, args.seed,
    )
    if not pre["reject"]:
        logger.warning(
            "The pretest did NOT reject whiteness: the L=1 innovations are already "
            "white, so per v3 Cor. 2.5.1 the memory kernel has nothing left to "
            "explain and L = %d is fitting noise. Proceeding, but treat any "
            "improvement over L=1 as unsupported.", cfg.memory.order,
        )

    op, b_p = identify_memory(
        w_dyn[train_sel], cfg.memory.order, inp.forcing[train_sel],
        cfg.memory.parameterization, cfg.dynamics.ridge_mu,
        cfg.dynamics.quiescent_threshold, cfg.memory.reduced_rank,
        valid=ds.observed[train_sel], increment=cfg.memory.increment,
    )
    blend_report = {}
    if cfg.memory.stability_metric == "forecast_gain":
        op, blend_report = calibrate_persistence_blend(
            op, b_p, w_dyn[train_sel], inp.forcing[train_sel],
            ds.observed[train_sel], cfg.horizons.horizon,
            gain_max=cfg.memory.gain_max,
        )
    else:
        op = enforce_stability(op, cfg.dynamics.rho_max)
    # Run the spectral-radius rail as WELL as the gain cap. The gain cap bounds
    # ||S A^h|| over h = 1..H; a non-normal operator can satisfy it while
    # rho(A_cal) > 1 and therefore diverge beyond the fitted horizon. rho_max was
    # declared in DynamicsConfig and applied nowhere in this branch.
    op, radius_report = enforce_spectral_radius(op, cfg.dynamics.rho_max)
    blend_report = dict(blend_report or {})
    blend_report["spectral_radius_rail"] = radius_report
    rho, cert, transient = report_certificates(
        op, cfg.dynamics.rho_max, cfg.memory.order, cfg.horizons.horizon,
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
    gain = information_gain(em, q)
    logger.info(
        "Weather emission: %d measurements; one update removes %.1f%% of the "
        "state-uncertainty trace.",
        em.c.shape[0], 100.0 * gain["fraction_removed"],
    )

    family = fit_horizon_family(
        w_dyn[train_sel], op, b_p, cfg.horizons.horizon, inp.forcing[train_sel],
        ds.observed[train_sel], cfg.horizons.estimator, cfg.horizons.ridge_mu,
        nu_grid=cfg.horizons.nu_grid, nu_selection=cfg.horizons.nu_selection, q=q,
    )
    logger.info("Horizon family (%s): nu = %s", family.estimator,
                np.array2string(np.asarray(family.nu), precision=2))
    logger.info("Semigroup defect ||D_h||/||S A^h|| = %s (h >= 2)",
                np.array2string(np.asarray(family.defect_relative), precision=3))

    ckpt_path = os.path.join(args.ckpt_dir, cfg.name + "_v4.pkl")
    save_checkpoint(ckpt_path, {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "sigma_eps2": sigma_eps2,
        "op": op,
        "b_p": np.asarray(b_p),
        "q": np.asarray(q),
        "emission": em,
        "family": family,
        "subspace": subspace,
        # The augmented basis must travel with the checkpoint: the EOF block is
        # transductive (defined on this grid, not as a function of coordinates),
        # so infer_v4 cannot rebuild it from Psi alone. Without it the decoder at
        # inference would differ from the one the operator was identified against.
        "extra_features": extra,
        "basis_offset": offset,
        "config": cfg.to_dict(),
        "norm_mean": np.asarray(ds.mean),
        "norm_std": np.asarray(ds.std),
        # The pixel grid the basis was evaluated on. The empirical (EOF) block is
        # transductive -- defined per pixel, not as a function of coordinates --
        # so a checkpoint is only valid on the grid it was fitted on. Recording
        # the shape lets forecast_v4 refuse a mismatched --max-pixels with a
        # message that names the fix, instead of silently decoding nonsense.
        "grid_shape": tuple(int(v) for v in ds.shape),
        "n_pixels": int(ds.coords.shape[0]),
        "forcing_names": list(inp.weather.forcing_names),
        "measurement_names": list(inp.weather.measurement_names),
        "weather_climatology": getattr(inp.weather, "climatology", None),
        # Both scale vectors travel with the climatology: an operational window
        # of a dozen days must not re-estimate either, or its anomalies -- the
        # quantity C consumes -- would be on a different scale from training.
        "weather_forcing_scale": np.asarray(inp.weather.forcing_scale),
        "weather_measurement_scale": np.asarray(inp.weather.measurement_scale),
        "weather_precip_lags": int(cfg.weather.precip_lags),
        # Per-pixel representation variance and bias, physical units. The
        # predictive interval is wrong without the first, and the second
        # localises where the basis is systematically off.
        "repr_variance": np.asarray(
            aug.holdout_variance * float(ds.std[0]) ** 2
            if aug.holdout_variance is not None else repr_stats["variance"]
        ),
        "repr_bias": np.asarray(repr_stats["bias"]),
        "diagnostics": {
            "rho": rho,
            "observable": bool(cert["observable"]),
            "observability_cause": cert.get("cause"),
            "transient_peak": float(transient["peak"]),
            "rank_95": int(rank.rank_95),
            "rank_99": int(rank.rank_99),
            "k": int(subspace.k) if subspace is not None else int(cfg.basis.r),
            "whiteness_reject": bool(pre["reject"]),
            "persistence_blend": blend_report.get("blend"),
            "spectral_radius_rail": radius_report,
            "representation_rms": float(repr_stats["rms"]),
            "representation_rms_holdout": float(aug.rmse_holdout * float(ds.std[0])),
            "representation_gap": float(aug.generalisation_gap),
            "eof_selection": aug.selection,
            "max_forecast_gain": float(
                np.max(forecast_gain(op, cfg.horizons.horizon))
            ),
            "reconstruction_r2": float(aug.r2_after),
            "reconstruction_r2_psi_only": float(aug.r2_before),
            "eof_modes": int(aug.q_empirical),
            "information_gain": {k: float(v) for k, v in gain.items()},
            "train_metrics": {k: float(v) for k, v in metrics.items()},
        },
    })
    logger.info("Saved checkpoint to %s", os.path.abspath(ckpt_path))
    logger.info("Forecast with: python -m experiments.infer_v4 --ckpt %s", ckpt_path)


if __name__ == "__main__":
    main()
