"""
One-off preprocessing of the exogenous forcing ``u_t^raw = [p_t, ..., u_t]``.

Warping a multi-year stack of daily precipitation rasters onto the 30 m LST/NDVI
grid is by far the most expensive step in the pipeline and it does not depend on
any model parameter -- so it is done **once**, per modality, and cached to NPZ.
``DataConfig.forcing_path`` then points training at the cached table.

Why per modality
----------------
LST and NDVI are separate models with separate acquisition calendars (Landsat
thermal and NDVI composites generally do not share dates). The forcing is defined
*relative to the acquisition grid* -- row ``t`` accumulates the water arriving
between ``d_t`` and ``d_{t+1}`` -- so an LST forcing table is simply not valid for
NDVI. Run this script once for each.

What it does
------------
1. Read the reference grid (CRS / affine / size) from the modality's first tile.
2. Warp every daily precipitation raster onto that grid (pixel-by-pixel coincident),
   reducing each to ``J_p`` zonal scalars immediately so the 30 m stack is never
   held in memory.
3. Accumulate into the forward windows ``(d_t, d_{t+1}]`` and append the lag
   channels (wet-soil memory).
4. Aggregate the irrigation CSV onto the same windows.
5. Scale by *training-split* statistics only (no test leakage), then save.
6. Report the identifiability diagnostics that decide whether Algorithm 3 can even
   separate ``A`` from ``B`` (Remark 6.1) -- see the notes printed at the end.

Usage
-----
    python -m experiments.preprocess_forcing --modality lst
    python -m experiments.preprocess_forcing --modality ndvi --out forcing_ndvi.npz
    python -m experiments.preprocess_forcing --modality lst --synthetic   # no rasters
"""
from __future__ import annotations

import os
import argparse

import numpy as np

from dbwm.config import default_config
from dbwm.data.forcing import (
    build_forcing,
    synthetic_forcing,
    save_forcing,
    log_forcing_summary,
    channel_split,
)
from dbwm.data.geotiff_dataset import _discover_frames
from dbwm.data.raster_align import read_reference_grid
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.exp.forcing")


def main():
    """Build and cache the forcing table for one modality."""
    ap = argparse.ArgumentParser(
        description="Preprocess precipitation + irrigation forcing for DB-WM v2."
    )
    ap.add_argument(
        "--modality", choices=["lst", "ndvi"], default="lst",
        help="which acquisition grid to build the forcing against",
    )
    ap.add_argument("--out", default=None, help="output .npz path")
    ap.add_argument("--precip-dir", default=None, help="override precipitation raster dir")
    ap.add_argument("--irrigation-csv", default=None, help="override irrigation CSV path")
    ap.add_argument(
        "--n-zones", type=int, default=None,
        help="J_p: 1 -> AOI-mean scalar; >1 -> zonal means",
    )
    ap.add_argument(
        "--lags", type=int, default=None,
        help="number of lagged precipitation channels (wet-soil memory)",
    )
    ap.add_argument(
        "--synthetic", action="store_true",
        help="generate sparse synthetic forcing instead of reading rasters",
    )
    args = ap.parse_args()

    cfg = default_config()
    cfg.data.modality = args.modality
    if args.precip_dir:
        cfg.forcing.precip_dir = args.precip_dir
    if args.irrigation_csv:
        cfg.forcing.irrigation_csv = args.irrigation_csv
    if args.n_zones is not None:
        cfg.forcing.n_precip_zones = args.n_zones
    if args.lags is not None:
        cfg.forcing.precip_lags = args.lags
    cfg.sync_input_dim()

    out_path = args.out or "forcing_{}.npz".format(args.modality)

    if args.synthetic:
        import datetime as dt

        dates = [dt.date(2022, 1, 1) + dt.timedelta(days=8 * i) for i in range(60)]
        logger.info("Synthetic mode: %d fake acquisition dates.", len(dates))
        fs = synthetic_forcing(dates, cfg.forcing)
        log_forcing_summary(fs)
    else:
        data_dir = cfg.data.lst_dir if args.modality == "lst" else cfg.data.ndvi_dir
        dated = _discover_frames(data_dir)
        dates = [d for d, _ in dated]
        logger.info(
            "%s: %d acquisitions (%s -> %s).",
            args.modality.upper(), len(dates), dates[0], dates[-1],
        )

        # The reference grid comes from the imagery, so precipitation is warped
        # onto the *target* geometry -- never the other way round.
        grid = read_reference_grid(dated[0][1])
        n_train = max(2, int(round(cfg.data.train_fraction * len(dates))))
        fs = build_forcing(dates, cfg.forcing, grid=grid, n_train=n_train)

    save_forcing(fs, out_path)

    # --- Identifiability report (Remark 6.1) ---
    precip_cols, irrig_cols = channel_split(fs.names)
    logger.info("Forcing table: T=%d, ell=%d -> %s", fs.values.shape[0], fs.ell, out_path)
    logger.info(
        "  B_p columns (uncontrollable disturbance): %s",
        [fs.names[j] for j in precip_cols],
    )
    logger.info(
        "  B_u columns (controllable actuator):      %s",
        [fs.names[j] for j in irrig_cols],
    )

    if fs.ell > 1:
        # Collinear forcing channels are the failure mode of Remark 6.1: if rain and
        # irrigation move together, no amount of data can separate B_p from B_u.
        c = np.corrcoef(fs.values.T)
        iu = np.triu_indices(fs.ell, k=1)
        worst = int(np.argmax(np.abs(c[iu])))
        i, j = iu[0][worst], iu[1][worst]
        logger.info(
            "  Max |corr| between channels: %.3f (%s vs %s)",
            abs(c[i, j]), fs.names[i], fs.names[j],
        )
        if abs(c[i, j]) > 0.9:
            logger.warning(
                "Channels '%s' and '%s' are near-collinear (|r|=%.2f). Algorithm 3 "
                "cannot cleanly separate their forcing directions -- consider "
                "dropping one, or merging them into a single channel.",
                fs.names[i], fs.names[j], abs(c[i, j]),
            )

    logger.info("Point DataConfig.forcing_path at %s to use this table.", out_path)


if __name__ == "__main__":
    main()
