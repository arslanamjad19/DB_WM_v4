"""
Comparative analysis for the thesis: DB-WM v2 (ours) vs E-GP / Kernel Observer.

Runs the full ablation grid

    DB-WM:  {resnet, deit}  x  {swiglu, rbf}   (+ gelu optionally)
    Baseline:  E-GP / Kernel Observer (stationary RBF feature space, O(M^3))

trains each on the same train split, evaluates the test-time predict-then-correct
forecast on the same test split, and writes a metrics table (CSV) plus comparison
bar charts. This is the figure/table-generating entry point for the paper.

Usage
-----
    python -m experiments.compare_baselines --smoke
    python -m experiments.compare_baselines --modality lst --epochs 50
"""
from __future__ import annotations

import os
import argparse
import time
import csv

import numpy as np
import jax.numpy as jnp

from dbwm.config import default_config, smoke_config
from dbwm.models import DBWM
from dbwm.data.geotiff_dataset import load_dataset
from dbwm.training.trainer import train, identify_dynamics_closed_form
from dbwm.inference.observer import closed_loop_filter
from dbwm.baselines.egp import EGPBaseline, EGPConfig
from dbwm.evaluation.metrics import evaluate_sequence
from dbwm.evaluation.plots import plot_metric_bars, plot_error_over_time
from dbwm.evaluation.metrics import rmse
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.exp.compare")


def _base_cfg(args):
    """Build the base config shared by all DB-WM ablations."""
    cfg = smoke_config() if args.smoke else default_config()
    cfg.data.modality = args.modality
    if not args.smoke:
        cfg.training.n_epochs = args.epochs
        cfg.basis.r = args.r
        cfg.data.use_synthetic = args.synthetic
        if args.lst_dir:
            cfg.data.lst_dir = args.lst_dir
        if args.ndvi_dir:
            cfg.data.ndvi_dir = args.ndvi_dir
    return cfg


def run_dbwm(cfg, backbone, expansion, train_ds, test_ds):
    """Train one DB-WM ablation and return (metrics, per_step_rmse, train_time)."""
    cfg.backbone.kind = backbone
    cfg.basis.expansion = expansion
    cfg.name = "DBWM-{}-{}".format(backbone, expansion)
    model = DBWM(cfg)
    t0 = time.time()
    state = train(model, train_ds, cfg)
    a, b, q = identify_dynamics_closed_form(model, state.params, train_ds, cfg)
    train_time = time.time() - t0
    sigma2 = float(model.apply(state.params, method=model.sigma_eps2))
    out = closed_loop_filter(model, state.params, test_ds, a, q, sigma2, cfg)
    truth = test_ds.frames[..., 0]
    metrics = evaluate_sequence(out["prior_maps"], truth, test_ds.valid_mask)
    per_step = [rmse(out["prior_maps"][i], truth[i], test_ds.valid_mask[i]) for i in range(truth.shape[0])]
    metrics["train_time_s"] = train_time
    return metrics, per_step


def run_egp(train_ds, test_ds, smoke):
    """Train the E-GP baseline and return (metrics, per_step_rmse)."""
    ecfg = EGPConfig(n_centers=64 if smoke else 400, n_measurements=64 if smoke else 400)
    t0 = time.time()
    egp = EGPBaseline(ecfg).fit(train_ds)
    out = egp.closed_loop_filter(test_ds)
    train_time = time.time() - t0
    truth = test_ds.frames[..., 0]
    metrics = evaluate_sequence(out["prior_maps"], truth, test_ds.valid_mask)
    per_step = [rmse(out["prior_maps"][i], truth[i], test_ds.valid_mask[i]) for i in range(truth.shape[0])]
    metrics["train_time_s"] = train_time
    return metrics, per_step


def main():
    """Run the full ablation grid + baseline and write tables/plots."""
    ap = argparse.ArgumentParser(description="DB-WM vs E-GP comparative analysis.")
    ap.add_argument("--modality", default="lst", choices=["lst", "ndvi"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--r", type=int, default=512)
    ap.add_argument("--expansions", nargs="+", default=["swiglu", "rbf"])
    ap.add_argument("--backbones", nargs="+", default=["resnet", "deit"])
    ap.add_argument("--lst-dir", dest="lst_dir", default=None)
    ap.add_argument("--ndvi-dir", dest="ndvi_dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = _base_cfg(args)
    # Pass the forcing config: without it every DB-WM variant would be silently
    # trained with B = 0 and compared against E-GP as if it used rain/irrigation.
    forcing_cfg = cfg.forcing if cfg.dynamics.use_forcing else None
    train_ds, test_ds = load_dataset(cfg.data, forcing_cfg)
    results, curves = {}, {}

    for backbone in args.backbones:
        for expansion in args.expansions:
            name = "DBWM-{}-{}".format(backbone, expansion)
            logger.info("=== %s ===", name)
            m, per_step = run_dbwm(cfg, backbone, expansion, train_ds, test_ds)
            results[name] = m
            curves[name] = per_step

    logger.info("=== E-GP / Kernel Observer baseline ===")
    m, per_step = run_egp(train_ds, test_ds, args.smoke)
    results["E-GP"] = m
    curves["E-GP"] = per_step

    results_dir = cfg.inference.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # CSV table.
    csv_path = os.path.join(results_dir, "comparison_{}.csv".format(args.modality))
    metric_keys = ["rmse", "mae", "r2", "ssim", "train_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + metric_keys)
        for name, m in results.items():
            writer.writerow([name] + [round(m[k], 5) for k in metric_keys])
    logger.info("Wrote metrics table: %s", csv_path)

    # Bar charts + error-over-time.
    for metric in ["rmse", "mae", "ssim"]:
        plot_metric_bars(
            results, metric,
            os.path.join(results_dir, "bars_{}_{}.png".format(metric, args.modality)),
            title="{} ({})".format(metric.upper(), args.modality),
        )
    plot_error_over_time(
        curves,
        os.path.join(results_dir, "rmse_over_time_{}.png".format(args.modality)),
        title="Forecast RMSE over test horizon ({})".format(args.modality),
    )
    logger.info("Comparison complete. Results in %s", results_dir)
    for name, m in results.items():
        logger.info("%-18s rmse=%.4f mae=%.4f ssim=%.4f time=%.1fs",
                    name, m["rmse"], m["mae"], m["ssim"], m["train_time_s"])


if __name__ == "__main__":
    main()
