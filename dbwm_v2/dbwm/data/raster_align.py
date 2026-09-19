"""
Geospatial raster alignment utilities.

The precipitation product (CHIRPS 0.05 deg ~5.5 km, GPM IMERG 0.1 deg ~11 km, or
ERA5 0.25 deg) lives on a *completely different* grid from the 30 m downscaled
LST/NDVI rasters: different CRS (usually EPSG:4326 lon/lat vs. the study area's
UTM Zone 43N / EPSG:32643), different resolution, different extent.

To use precipitation as the forcing ``p_t`` it must be made **pixel-by-pixel
coincident** with the LST/NDVI grid. That means a full warp -- reproject +
resample onto the *exact* reference grid (same CRS, affine transform, width and
height) -- not a naive array resize, which would silently mis-register the fields
by kilometres.

This module provides:

* :class:`ReferenceGrid`  -- the target grid read once from an LST/NDVI tile.
* :func:`align_raster_to_grid` -- warp any GeoTIFF onto that grid.

Resampling choice matters physically:

* ``bilinear`` (default) -- smooth interpolation of the coarse rain field down to
  30 m. Appropriate when going coarse -> fine (the usual case here).
* ``nearest``  -- preserves the native coarse blocks (blocky but conservative in
  the sense that no new values are invented).
* ``average``  -- area-weighted; correct when going fine -> coarse (downscaling
  the other way), not for coarse -> fine.

Note this warp does **not** perform rainfall *downscaling* in the geophysical
sense: it re-registers a coarse field onto a fine grid. Since the DB-WM forcing
is ultimately reduced to an AOI mean (or a few zonal means), sub-pixel rainfall
structure is averaged away anyway, so the interpolation choice has little effect
on ``p_t`` -- but the *geometric registration* is essential.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.raster")

_RESAMPLING = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "cubic": "cubic",
    "average": "average",
}


@dataclass
class ReferenceGrid:
    """
    The target raster grid that every other layer is aligned to.

    :ivar crs: coordinate reference system (e.g. ``EPSG:32643``).
    :ivar transform: affine geotransform of the grid.
    :ivar width: grid width in pixels.
    :ivar height: grid height in pixels.
    """

    crs: object
    transform: object
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        """``(height, width)`` of the grid."""
        return (self.height, self.width)


def read_reference_grid(path: str) -> ReferenceGrid:
    """
    Read the reference grid (CRS / transform / size) from an LST or NDVI GeoTIFF.

    Uses the raster's *native* grid, so alignment happens at full resolution
    before any downsampling for the backbone.

    :param path: path to a representative LST/NDVI GeoTIFF.
    :return: the :class:`ReferenceGrid`.
    """
    import rasterio

    with rasterio.open(path) as src:
        grid = ReferenceGrid(
            crs=src.crs, transform=src.transform, width=src.width, height=src.height
        )
    logger.info(
        "Reference grid: %dx%d, CRS=%s", grid.width, grid.height, grid.crs
    )
    return grid


def align_raster_to_grid(
    path: str,
    grid: ReferenceGrid,
    resampling: str = "bilinear",
    nodata: float = np.nan,
) -> np.ndarray:
    """
    Warp a GeoTIFF (any CRS/resolution/extent) onto the reference grid so it is
    pixel-by-pixel coincident with the LST/NDVI rasters.

    :param path: path to the source raster (e.g. a daily precipitation tile).
    :param grid: the target :class:`ReferenceGrid`.
    :param resampling: one of ``{"nearest", "bilinear", "cubic", "average"}``.
    :param nodata: fill value for pixels outside the source extent.
    :return: ``(grid.height, grid.width)`` float32 array on the reference grid.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    if resampling not in _RESAMPLING:
        raise ValueError(
            "Unknown resampling '{}'; choose from {}.".format(
                resampling, sorted(_RESAMPLING)
            )
        )
    method = getattr(Resampling, _RESAMPLING[resampling])

    dst = np.full(grid.shape, nodata, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=nodata,
            resampling=method,
        )
    return dst


def aoi_mean(
    raster: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """
    Spatially average a raster over the AOI, ignoring NaNs and invalid pixels.

    :param raster: ``(H, W)`` aligned raster.
    :param mask: optional ``(H, W)`` boolean validity mask.
    :return: scalar AOI mean (0.0 if no valid pixel).
    """
    valid = np.isfinite(raster)
    if mask is not None:
        valid &= mask.astype(bool)
    if not valid.any():
        return 0.0
    return float(raster[valid].mean())


def zonal_means(
    raster: np.ndarray, zones: np.ndarray, n_zones: int
) -> np.ndarray:
    """
    Average a raster within each of ``n_zones`` spatial zones.

    :param raster: ``(H, W)`` aligned raster.
    :param zones: ``(H, W)`` integer zone labels in ``[0, n_zones)``.
    :param n_zones: number of zones ``J_p``.
    :return: ``(n_zones,)`` float32 array of zonal means.
    """
    out = np.zeros(n_zones, dtype=np.float32)
    finite = np.isfinite(raster)
    for j in range(n_zones):
        sel = finite & (zones == j)
        if sel.any():
            out[j] = raster[sel].mean()
    return out


def make_zone_labels(height: int, width: int, n_zones: int) -> np.ndarray:
    """
    Partition the AOI into ``n_zones`` contiguous blocks (a simple default when
    no administrative/district zone raster is supplied).

    Zones are laid out as a near-square grid of tiles, so each zone is a spatially
    contiguous block of the study area.

    :param height: grid height.
    :param width: grid width.
    :param n_zones: number of zones.
    :return: ``(H, W)`` int32 zone-label array with values in ``[0, n_zones)``.
    """
    if n_zones <= 1:
        return np.zeros((height, width), dtype=np.int32)
    n_rows = int(np.floor(np.sqrt(n_zones)))
    n_cols = int(np.ceil(n_zones / n_rows))
    row_idx = (np.arange(height) * n_rows // height).clip(0, n_rows - 1)
    col_idx = (np.arange(width) * n_cols // width).clip(0, n_cols - 1)
    labels = row_idx[:, None] * n_cols + col_idx[None, :]
    return np.minimum(labels, n_zones - 1).astype(np.int32)


def load_zone_raster(path: str, grid: ReferenceGrid, n_zones: int) -> np.ndarray:
    """
    Load an external zone/district raster and align it to the reference grid.

    Nearest-neighbour resampling is used so integer labels are preserved.

    :param path: path to a categorical zone GeoTIFF.
    :param grid: target reference grid.
    :param n_zones: expected number of zones (labels are clipped into range).
    :return: ``(H, W)`` int32 zone labels.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Zone raster not found: {}".format(path))
    arr = align_raster_to_grid(path, grid, resampling="nearest", nodata=0.0)
    arr = np.nan_to_num(arr, nan=0.0)
    return np.clip(arr.astype(np.int32), 0, n_zones - 1)
