"""
High-level inference / forecasting on top of the Kalman observer.

Ties together the trained :class:`~dbwm.models.db_wm.DBWM`, the identified
dynamics ``(A, B, Q)`` and the weight-space Kalman filter to produce LST / NDVI
*map* forecasts with per-pixel uncertainty.

Two inference modes (both required by the thesis spec):

* :func:`closed_loop_filter` -- the test-time "predict-then-correct" loop: at
  each test step the model PREDICTS the next state from its prior (dynamics),
  decodes that prior to an LST/NDVI map (the *forecast* scored against the held-
  out frame), then UPDATES the state with the encoded test observation.
* :func:`open_loop_forecast_maps` -- a pure ``H``-step rollout from a single
  starting frame with no further observations (honest multi-day forecast).

Pixel-space decoding is done in coordinate batches so the full ``n x r`` matrix
is never held in memory (``O(nr)`` per map).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from dbwm.models.db_wm import DBWM
from dbwm.data.geotiff_dataset import SpatiotemporalDataset, make_pixel_grid
from dbwm.inference.kalman import filter_sequence, open_loop_forecast


def encode_sequence(model: DBWM, params, frames: np.ndarray) -> jnp.ndarray:
    """
    Encode a stack of frames to weight states ``w_t = phi_theta(o_t)``.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param frames: ``(T, H, W, C)`` frames.
    :return: ``(T, r)`` encoded weights.
    """
    return model.apply(params, jnp.asarray(frames), train=False, method=model.encode)


def decode_maps(
    model: DBWM,
    params,
    w_seq: jnp.ndarray,
    grid: np.ndarray,
    height: int,
    width: int,
    pixel_batch: int = 20000,
) -> np.ndarray:
    """
    Decode latent states to full ``(T, H, W)`` field maps in coordinate batches.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param w_seq: ``(T, r)`` latent states.
    :param grid: ``(H*W, 2)`` pixel coordinate grid.
    :param height: map height.
    :param width: map width.
    :param pixel_batch: number of pixels decoded per chunk.
    :return: ``(T, H, W)`` decoded maps.
    """
    n_pix = grid.shape[0]
    outs = []
    for start in range(0, n_pix, pixel_batch):
        coords = jnp.asarray(grid[start : start + pixel_batch])
        phi = model.apply(params, coords, method=model.spatial_features)  # (b, r)
        outs.append(np.asarray(w_seq @ phi.T))  # (T, b)
    maps = np.concatenate(outs, axis=1)  # (T, n_pix)
    return maps.reshape(w_seq.shape[0], height, width)


def variance_map(
    model: DBWM,
    params,
    grid: np.ndarray,
    height: int,
    width: int,
    pixel_batch: int = 20000,
) -> np.ndarray:
    """
    Per-pixel (state-independent) predictive variance map ``||L^T Psi(x)||^2 + sigma^2``.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param grid: ``(H*W, 2)`` pixel grid.
    :param height: map height.
    :param width: map width.
    :param pixel_batch: pixels per chunk.
    :return: ``(H, W)`` variance map.
    """
    n_pix = grid.shape[0]
    outs = []
    for start in range(0, n_pix, pixel_batch):
        coords = jnp.asarray(grid[start : start + pixel_batch])
        outs.append(np.asarray(model.apply(params, coords, method=model.variational_var)))
    return np.concatenate(outs, axis=0).reshape(height, width)


def closed_loop_filter(
    model: DBWM,
    params,
    ds: SpatiotemporalDataset,
    a: jnp.ndarray,
    q: jnp.ndarray,
    sigma_eps2: float,
    cfg,
    b: Optional[jnp.ndarray] = None,
) -> dict:
    """
    Test-time predict-then-correct filtering over a dataset split.

    For each test step the filter predicts the next state (prior), decodes it to
    a forecast map, then assimilates the encoded test frame as a measurement.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param ds: the (test) :class:`SpatiotemporalDataset`.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param sigma_eps2: measurement-noise variance.
    :param cfg: experiment config.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :return: dict with ``prior_maps`` ``(T, H, W)``, ``filtered_maps`` ``(T, H, W)``,
             ``var_map`` ``(H, W)``, ``w_filtered`` ``(T, r)``.
    """
    h, w_, _ = ds.image_shape
    grid = make_pixel_grid(h, w_)
    phi_obs = encode_sequence(model, params, ds.frames)  # (T, r)
    u_seq = None if (b is None or ds.forcing is None) else jnp.asarray(ds.forcing)

    w_filt, p_filt = filter_sequence(
        phi_obs, a, q, sigma_eps2, b=b, u_seq=u_seq,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )

    # Prior (one-step-ahead) states -- these ARE the forecasts that get scored, so
    # they must use the same predict mean as Algorithm 2:
    #     w_{t|t-1} = A w_{t-1|t-1} + B u_{t-1}^raw.
    # Dropping the B u term here would silently evaluate a *pure-autonomous*
    # forecast while claiming the forced model, hiding whatever skill B_p / B_u add.
    # Row t-1 of the forcing is the input driving t-1 -> t (forward-window
    # convention of dbwm.data.forcing), so the predict step at t consumes u[t-1].
    w_prior_rest = w_filt[:-1] @ a.T  # (T-1, r)
    if u_seq is not None:
        w_prior_rest = w_prior_rest + u_seq[:-1] @ b.T
    w_prior = jnp.concatenate([phi_obs[:1], w_prior_rest], axis=0)

    prior_maps = decode_maps(model, params, w_prior, grid, h, w_)
    filtered_maps = decode_maps(model, params, w_filt, grid, h, w_)
    var = variance_map(model, params, grid, h, w_)
    return {
        "prior_maps": prior_maps,
        "filtered_maps": filtered_maps,
        "var_map": var,
        "w_filtered": np.asarray(w_filt),
    }


def open_loop_forecast_maps(
    model: DBWM,
    params,
    start_frame: np.ndarray,
    a: jnp.ndarray,
    q: jnp.ndarray,
    cfg,
    b: Optional[jnp.ndarray] = None,
    u_seq: Optional[jnp.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``H``-step open-loop forecast of LST/NDVI maps from a single starting frame.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param start_frame: ``(H, W, C)`` initial observation.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param cfg: experiment config.
    :param b: ``(r, ell)`` input matrix or ``None``.
    :param u_seq: ``(H, ell)`` forcing forecasts or ``None``.
    :return: ``(forecast_maps, forecast_pixel_var)`` each ``(H, H_img, W_img)``;
             the variance map blends the propagated state covariance with the
             per-pixel basis variance.
    """
    h, w_, _ = start_frame.shape
    grid = make_pixel_grid(h, w_)
    w0 = model.apply(params, jnp.asarray(start_frame)[None], train=False, method=model.encode)[0]
    w_fc, p_fc = open_loop_forecast(
        w0, a, q, cfg.inference.horizon, b=b, u_seq=u_seq,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )
    maps = decode_maps(model, params, w_fc, grid, h, w_)

    # Per-pixel forecast variance phi(x)^T P_k phi(x) for each horizon step.
    var_maps = []
    phi_all = model.apply(params, jnp.asarray(grid), method=model.spatial_features)  # (n, r)
    for k in range(w_fc.shape[0]):
        vk = jnp.einsum("nr,rs,ns->n", phi_all, p_fc[k], phi_all)
        var_maps.append(np.asarray(vk).reshape(h, w_))
    return maps, np.stack(var_maps, axis=0)
