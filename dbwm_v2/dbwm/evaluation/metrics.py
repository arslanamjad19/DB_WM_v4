"""
Evaluation metrics for spatiotemporal LST/NDVI forecasting.

All metrics operate on host-side ``numpy`` arrays (predictions vs ground truth),
optionally masking invalid pixels, and return plain floats suitable for tables.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _apply_mask(pred, true, mask):
    """Flatten and mask predictions / targets to valid pixels only."""
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    if mask is None:
        return pred, true
    mask = np.asarray(mask).reshape(-1).astype(bool)
    return pred[mask], true[mask]


def rmse(pred, true, mask: Optional[np.ndarray] = None) -> float:
    """
    Root-mean-square error.

    :param pred: predicted array.
    :param true: ground-truth array.
    :param mask: optional boolean validity mask.
    :return: RMSE.
    """
    p, t = _apply_mask(pred, true, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred, true, mask: Optional[np.ndarray] = None) -> float:
    """
    Mean absolute error.

    :param pred: predicted array.
    :param true: ground-truth array.
    :param mask: optional boolean validity mask.
    :return: MAE.
    """
    p, t = _apply_mask(pred, true, mask)
    return float(np.mean(np.abs(p - t)))


def field_error_metrics(
    pred, true, mask: Optional[np.ndarray] = None, scale: float = 1.0
) -> dict:
    """
    RMSE, bias, **unbiased RMSE** and MAE for a predicted field.

    Why the unbiased RMSE is the more informative headline
    ------------------------------------------------------
    RMSE conflates two errors with completely different causes and cures::

        RMSE^2 = bias^2 + ubRMSE^2

    The **bias** is a whole-field offset -- the model sitting systematically high
    or low. It usually comes from something correctable and specific: a
    climatology fitted on a different period, a filter driven by a mis-specified
    sensor, a shrinkage anchor pulling toward the wrong state. On the real record
    a bias of +0.064 NDVI against an RMSE of 0.104 meant **38% of the squared
    error was a constant offset**, which no amount of work on the dynamics would
    have touched.

    The **ubRMSE** is the part that varies pixel to pixel -- genuine disagreement
    about spatial structure. That is the quantity a better model has to reduce,
    and the one worth comparing across models, because a bias can be removed by
    subtraction while structural error cannot.

    Reporting only RMSE therefore hides which of the two is being fought. This is
    standard practice in the land-surface validation literature for exactly that
    reason.

    MAE is reported alongside because RMSE is dominated by the tail: on a field
    with sharp plot boundaries a handful of misplaced edges can move RMSE
    substantially while most of the map is fine, and ``RMSE / MAE`` makes that
    visible (≈1.25 for Gaussian errors, higher when a few pixels dominate).

    :param pred: predicted field.
    :param true: observed field.
    :param mask: optional validity mask.
    :param scale: multiply every metric by this (e.g. the normalisation std, to
        report in physical NDVI units).
    :return: dict with ``rmse``, ``bias``, ``ubrmse``, ``mae``, ``rmse_over_mae``.
    """
    p, t = _apply_mask(np.asarray(pred), np.asarray(true), mask)
    err = p - t
    finite = np.isfinite(err)
    err = err[finite]
    if err.size == 0:
        return {k: float("nan")
                for k in ("rmse", "bias", "ubrmse", "mae", "rmse_over_mae")}
    rmse_v = float(np.sqrt(np.mean(err**2)))
    bias_v = float(np.mean(err))
    # Exact identity, not an approximation: ubRMSE is the std of the error.
    ub_v = float(np.sqrt(max(rmse_v**2 - bias_v**2, 0.0)))
    mae_v = float(np.mean(np.abs(err)))
    return {
        "rmse": rmse_v * scale,
        "bias": bias_v * scale,
        "ubrmse": ub_v * scale,
        "mae": mae_v * scale,
        "rmse_over_mae": rmse_v / mae_v if mae_v > 0 else float("nan"),
    }


def r2_score(pred, true, mask: Optional[np.ndarray] = None) -> float:
    """
    Coefficient of determination ``R^2``.

    :param pred: predicted array.
    :param true: ground-truth array.
    :param mask: optional boolean validity mask.
    :return: R^2.
    """
    p, t = _apply_mask(pred, true, mask)
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def ssim(pred, true) -> float:
    """
    Structural similarity index over a 2-D map (mean over the image).

    Falls back to a correlation-based proxy if scikit-image is unavailable.

    :param pred: ``(H, W)`` predicted map.
    :param true: ``(H, W)`` ground-truth map.
    :return: SSIM in ``[-1, 1]``.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    rng = float(true.max() - true.min()) + 1e-8
    try:
        from skimage.metrics import structural_similarity as sk_ssim

        return float(sk_ssim(true, pred, data_range=rng))
    except Exception:
        a = (pred - pred.mean()).ravel()
        b = (true - true.mean()).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(a @ b / denom)


def nll_gaussian(pred_mean, pred_var, true, mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Gaussian negative log-likelihood (calibration metric).

    :param pred_mean: predicted means.
    :param pred_var: predicted variances (broadcastable).
    :param true: ground truth.
    :param mask: optional boolean validity mask.
    :return: mean NLL.
    """
    pred_mean = np.asarray(pred_mean).reshape(-1)
    var = np.asarray(pred_var).reshape(-1)
    true = np.asarray(true).reshape(-1)
    var = np.clip(var, 1e-8, None)
    if mask is not None:
        m = np.asarray(mask).reshape(-1).astype(bool)
        pred_mean, var, true = pred_mean[m], var[m], true[m]
    return float(
        0.5 * np.mean((true - pred_mean) ** 2 / var + np.log(var) + np.log(2 * np.pi))
    )


def evaluate_sequence(
    pred_maps, true_maps, masks: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Aggregate metrics over a sequence of predicted vs true maps.

    :param pred_maps: ``(T, H, W)`` predictions.
    :param true_maps: ``(T, H, W)`` ground truth.
    :param masks: ``(T, H, W)`` validity masks or ``None``.
    :return: dict of mean metrics (``rmse``, ``mae``, ``r2``, ``ssim``).
    """
    pred_maps = np.asarray(pred_maps)
    true_maps = np.asarray(true_maps)
    t = pred_maps.shape[0]
    out = {"rmse": [], "mae": [], "r2": [], "ssim": []}
    for i in range(t):
        m = None if masks is None else masks[i]
        out["rmse"].append(rmse(pred_maps[i], true_maps[i], m))
        out["mae"].append(mae(pred_maps[i], true_maps[i], m))
        out["r2"].append(r2_score(pred_maps[i], true_maps[i], m))
        out["ssim"].append(ssim(pred_maps[i], true_maps[i]))
    return {k: float(np.mean(v)) for k, v in out.items()}
