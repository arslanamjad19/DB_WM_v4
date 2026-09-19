"""
Publication-quality plots for the DB-WM v2 thesis experiments.

Uses a non-interactive Matplotlib backend so figures render on headless
machines (Colab / compute nodes). Every function saves a PNG and returns its
path.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path: str):
    """Create the parent directory of ``path`` if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_prediction_triptych(
    truth, prediction, variance, save_path: str, title: str = "", cmap: str = "jet"
) -> str:
    """
    Plot ground truth, prediction and per-pixel uncertainty side by side.

    :param truth: ``(H, W)`` ground-truth map.
    :param prediction: ``(H, W)`` predicted map.
    :param variance: ``(H, W)`` predictive variance map.
    :param save_path: output PNG path.
    :param title: figure suptitle.
    :param cmap: colormap for the fields.
    :return: ``save_path``.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, name, cm in zip(
        axes,
        [truth, prediction, np.sqrt(np.maximum(variance, 0))],
        ["Ground truth", "Prediction", "Predictive std"],
        [cmap, cmap, "viridis"],
    ):
        im = ax.imshow(np.asarray(data), cmap=cm)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_error_over_time(
    series: Dict[str, List[float]], save_path: str, ylabel: str = "RMSE", title: str = ""
) -> str:
    """
    Plot one or more error curves over the test horizon (e.g. DB-WM vs E-GP).

    :param series: mapping ``label -> list of per-step errors``.
    :param save_path: output PNG path.
    :param ylabel: y-axis label.
    :param title: figure title.
    :return: ``save_path``.
    """
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(8, 5))
    for label, vals in series.items():
        plt.plot(np.arange(len(vals)), vals, marker="o", ms=3, label=label)
    plt.xlabel("Test time step")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_koopman_spectrum(eigenvalues, save_path: str, title: str = "Koopman spectrum") -> str:
    """
    Plot the eigenvalues of the transition operator in the complex plane with the
    unit circle (Corollary 4.2).

    :param eigenvalues: complex array of eigenvalues.
    :param save_path: output PNG path.
    :param title: figure title.
    :return: ``save_path``.
    """
    _ensure_dir(save_path)
    ev = np.asarray(eigenvalues)
    fig = plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="unit circle")
    plt.scatter(ev.real, ev.imag, c="crimson", s=20)
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_metric_bars(
    results: Dict[str, Dict[str, float]], metric: str, save_path: str, title: str = ""
) -> str:
    """
    Bar chart comparing models on a single metric (for the ablation tables).

    :param results: mapping ``model_name -> {metric: value}``.
    :param metric: which metric key to plot.
    :param save_path: output PNG path.
    :param title: figure title.
    :return: ``save_path``.
    """
    _ensure_dir(save_path)
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    fig = plt.figure(figsize=(max(6, len(names) * 1.5), 5))
    plt.bar(names, vals, color="steelblue")
    plt.ylabel(metric.upper())
    if title:
        plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
