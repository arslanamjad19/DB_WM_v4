"""
Run inference / forecasting with a trained DB-WM v2 checkpoint.

Performs the test-time predict-then-correct loop (Algorithm 2) on the test split,
saves prediction triptychs (truth / prediction / uncertainty), the per-step error
curve, and the Koopman spectrum of the learned operator.

Usage
-----
    python -m experiments.run_inference --ckpt <path>.pkl
    python -m experiments.run_inference --smoke
"""
from __future__ import annotations

import os
import argparse
import pickle

import numpy as np
import jax.numpy as jnp

from dbwm.config import (
    ExperimentConfig,
    smoke_config,
    DataConfig,
    ForcingConfig,
    BackboneConfig,
    BasisConfig,
    DynamicsConfig,
    TrainingConfig,
    InferenceConfig,
    PlanningConfig,
)
from dbwm.models import DBWM
from dbwm.data.geotiff_dataset import load_dataset
from dbwm.training.trainer import train, identify_dynamics_closed_form
from dbwm.inference.observer import closed_loop_filter
from dbwm.dynamics.identification import koopman_modes
from dbwm.evaluation.metrics import evaluate_sequence, rmse
from dbwm.evaluation.plots import (
    plot_prediction_triptych,
    plot_error_over_time,
    plot_koopman_spectrum,
)
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.exp.infer")


def _config_from_dict(d) -> ExperimentConfig:
    """
    Reconstruct an ExperimentConfig from a saved ``to_dict()``.

    Every sub-config is restored, ``forcing`` included: dropping it would silently
    fall back to the *default* forcing channels, so a checkpoint trained with (say)
    two precipitation zones would be reloaded expecting one, and ``B`` would be
    applied against a mismatched ``u_t^raw``.
    """
    return ExperimentConfig(
        data=DataConfig(**d["data"]),
        forcing=ForcingConfig(**d["forcing"]),
        backbone=BackboneConfig(**d["backbone"]),
        basis=BasisConfig(**d["basis"]),
        dynamics=DynamicsConfig(**d["dynamics"]),
        training=TrainingConfig(**d["training"]),
        inference=InferenceConfig(**d["inference"]),
        planning=PlanningConfig(**d.get("planning", {})),
        name=d["name"],
    )


def main():
    """Load a checkpoint (or smoke-train one), run inference, and plot."""
    ap = argparse.ArgumentParser(description="DB-WM v2 inference / forecasting.")
    ap.add_argument("--ckpt", default=None, help="path to a trained .pkl checkpoint")
    ap.add_argument("--smoke", action="store_true", help="train tiny model then infer")
    args = ap.parse_args()

    if args.smoke or args.ckpt is None:
        logger.info("No checkpoint: running a smoke train+infer on synthetic data.")
        cfg = smoke_config()
        forcing_cfg = cfg.forcing if cfg.dynamics.use_forcing else None
        train_ds, test_ds = load_dataset(cfg.data, forcing_cfg)
        model = DBWM(cfg)
        state = train(model, train_ds, cfg)
        a, b, q = identify_dynamics_closed_form(model, state.params, train_ds, cfg)
        params = state.params
    else:
        with open(args.ckpt, "rb") as f:
            ckpt = pickle.load(f)
        cfg = _config_from_dict(ckpt["config"])
        model = DBWM(cfg)
        params = ckpt["params"]
        a, b, q = jnp.asarray(ckpt["A"]), jnp.asarray(ckpt["B"]), jnp.asarray(ckpt["Q"])
        # Forcing must be rebuilt for the test split too: the Kalman predict step
        # consumes u_t^raw, so omitting it would score a pure-autonomous forecast
        # against a model that was trained with rain and irrigation.
        forcing_cfg = cfg.forcing if cfg.dynamics.use_forcing else None
        _, test_ds = load_dataset(cfg.data, forcing_cfg)

    sigma2 = float(model.apply(params, method=model.sigma_eps2))
    b_mat = b if (cfg.dynamics.use_forcing and test_ds.forcing is not None) else None
    out = closed_loop_filter(model, params, test_ds, a, q, sigma2, cfg, b=b_mat)

    truth = test_ds.frames[..., 0]
    metrics_prior = evaluate_sequence(out["prior_maps"], truth, test_ds.valid_mask)
    metrics_filt = evaluate_sequence(out["filtered_maps"], truth, test_ds.valid_mask)
    logger.info("Prior (forecast)  metrics: %s", metrics_prior)
    logger.info("Filtered (corrected) metrics: %s", metrics_filt)

    results_dir = cfg.inference.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Triptych for the first test step.
    plot_prediction_triptych(
        truth[0], out["prior_maps"][0], out["var_map"],
        os.path.join(results_dir, cfg.name + "_triptych.png"),
        title="{} | one-step forecast".format(cfg.name),
    )
    # Error-over-time curve (prior forecast vs filtered).
    per_step_prior = [rmse(out["prior_maps"][i], truth[i], test_ds.valid_mask[i]) for i in range(truth.shape[0])]
    per_step_filt = [rmse(out["filtered_maps"][i], truth[i], test_ds.valid_mask[i]) for i in range(truth.shape[0])]
    plot_error_over_time(
        {"prior (forecast)": per_step_prior, "filtered (corrected)": per_step_filt},
        os.path.join(results_dir, cfg.name + "_rmse_over_time.png"),
        title="{} | test RMSE".format(cfg.name),
    )
    # Koopman spectrum.
    modes = koopman_modes(a)
    plot_koopman_spectrum(
        np.asarray(modes["eigenvalues"]),
        os.path.join(results_dir, cfg.name + "_koopman.png"),
        title="{} | Koopman spectrum".format(cfg.name),
    )
    logger.info("Saved figures to %s", results_dir)


if __name__ == "__main__":
    main()
