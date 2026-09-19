"""
LST / NDVI GeoTIFF dataset loading, preprocessing and chronological splitting.

Responsibilities
----------------
1. Discover GeoTIFF frames in a Google-Drive folder, parse their acquisition
   dates from the filenames, and order them chronologically.
2. Capture the native **reference grid** (CRS / affine / size) of the imagery so
   the precipitation rasters can be warped onto it pixel-by-pixel
   (:mod:`dbwm.data.raster_align`).
3. Read + resample each frame to a fixed ``(H, W)`` grid, mask no-data pixels,
   and normalise using *training-split* statistics only (no test leakage).
4. Split 85% / 15% chronologically (the test set is strictly in the future).
5. Build and attach the per-step forcing ``u_t^raw = [p_t, ..., u_t]``
   (:mod:`dbwm.data.forcing`), against **this modality's own date grid** -- LST
   and NDVI are separate models with generally different acquisition dates.
6. Provide a synthetic generator (frames + sparse forcing) so the whole pipeline
   runs without the real data.

Host-side I/O uses ``numpy`` (per the codebase convention: ``np`` for file I/O,
``jnp`` only inside jit'd compute). Frames are returned as ``float32`` arrays of
shape ``(n_frames, H, W, C)`` (channels-last, the Flax convention).
"""
from __future__ import annotations

import os
import glob
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from dbwm.config import DataConfig, ForcingConfig
from dbwm.data.forcing import (
    ForcingSeries,
    build_forcing,
    load_forcing,
    parse_date,
    synthetic_forcing,
)
from dbwm.data.raster_align import ReferenceGrid, read_reference_grid
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.data")


@dataclass
class SpatiotemporalDataset:
    """
    A chronologically ordered stack of LST/NDVI frames plus optional forcing.

    :ivar frames: ``(n, H, W, C)`` float32 array, normalised.
    :ivar dates: list of :class:`datetime.date`, one per frame.
    :ivar forcing: ``(n, ell)`` float32 array of ``[p_t, ..., u_t]`` or ``None``.
    :ivar mean: per-channel training mean used for normalisation.
    :ivar std: per-channel training std used for normalisation.
    :ivar valid_mask: ``(n, H, W)`` boolean array, ``True`` where pixel valid.
    :ivar forcing_names: channel names for ``forcing`` (length ``ell``).
    :ivar forcing_scale: per-channel divisor applied to ``forcing``.
    """

    frames: np.ndarray
    dates: List[dt.date]
    forcing: Optional[np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    valid_mask: np.ndarray
    forcing_names: Optional[List[str]] = None
    forcing_scale: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        """Number of time steps."""
        return self.frames.shape[0]

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)`` of a single frame."""
        return self.frames.shape[1:]

    @property
    def ell(self) -> int:
        """Forcing dimension (0 if no forcing)."""
        return 0 if self.forcing is None else self.forcing.shape[1]

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Invert normalisation to recover physical units (deg C / NDVI units).

        :param x: normalised array broadcastable to ``(..., C)``.
        :return: de-normalised array.
        """
        return x * self.std + self.mean


def _read_geotiff(path: str, height: int, width: int, nodata: float):
    """
    Read a single GeoTIFF, resample to ``(height, width)`` and return the first
    band plus a validity mask.

    :param path: path to the GeoTIFF.
    :param height: target height.
    :param width: target width.
    :param nodata: sentinel value marking invalid pixels.
    :return: ``(band, mask)`` with ``band`` shape ``(H, W)`` float32 and
             ``mask`` shape ``(H, W)`` bool.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        band = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        file_nodata = src.nodata

    mask = np.isfinite(band)
    if file_nodata is not None:
        mask &= band != file_nodata
    mask &= band != nodata
    band = np.where(mask, band, 0.0).astype(np.float32)
    return band, mask


def _make_synthetic(cfg: DataConfig) -> Tuple[np.ndarray, List[dt.date], np.ndarray]:
    """
    Generate a synthetic spatiotemporal field with smooth diurnal/seasonal
    structure so the dynamics are genuinely learnable (low-rank, near-periodic).

    :param cfg: data configuration.
    :return: ``(frames, dates, valid_mask)``.
    """
    rng = np.random.RandomState(0)
    n, h, w, c = (
        cfg.synthetic_n_frames,
        cfg.image_height,
        cfg.image_width,
        cfg.n_channels,
    )
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij"
    )
    modes = [np.sin(2 * xx), np.cos(3 * yy), np.exp(-(xx**2 + yy**2)), xx * yy]
    frames = np.zeros((n, h, w, c), dtype=np.float32)
    for t in range(n):
        season = np.sin(2 * np.pi * t / 365.0)
        weekly = np.sin(2 * np.pi * t / 7.0)
        field = (
            (1.0 + 0.5 * season) * modes[0]
            + (0.5 * weekly) * modes[1]
            + (0.3 + 0.2 * season) * modes[2]
            + 0.1 * weekly * modes[3]
        )
        field = field + 0.02 * rng.randn(h, w)
        frames[t, :, :, 0] = field.astype(np.float32)
    base = dt.date(2022, 1, 1)
    dates = [base + dt.timedelta(days=i) for i in range(n)]
    valid_mask = np.ones((n, h, w), dtype=bool)
    return frames, dates, valid_mask


def _discover_frames(data_dir: str) -> List[Tuple[dt.date, str]]:
    """
    List dated GeoTIFF frames in a modality folder, chronologically ordered.

    :param data_dir: LST or NDVI folder.
    :return: list of ``(date, path)``.
    """
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.tif"))
        + glob.glob(os.path.join(data_dir, "*.tiff"))
    )
    if not paths:
        raise FileNotFoundError(
            "No GeoTIFFs found in {}. Set DataConfig.use_synthetic=True for a "
            "smoke run, or point lst_dir/ndvi_dir at your Drive folder.".format(data_dir)
        )
    dated = []
    for p in paths:
        d = parse_date(p)
        if d is None:
            logger.warning("Skipping undated frame: %s", os.path.basename(p))
            continue
        dated.append((d, p))
    if not dated:
        raise ValueError(
            "Found {} GeoTIFFs in {} but none had a parsable date in the filename "
            "(expected YYYYMMDD / YYYY-MM-DD / YYYYDDD).".format(len(paths), data_dir)
        )
    dated.sort(key=lambda x: x[0])
    return dated


def load_dataset(
    cfg: DataConfig, forcing_cfg: Optional[ForcingConfig] = None
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset]:
    """
    Load one modality (LST *or* NDVI) and return a chronological
    ``(train, test)`` pair with forcing attached.

    Normalisation statistics *and* forcing scaling are computed on the training
    split only, so there is no test-set leakage.

    :param cfg: data configuration.
    :param forcing_cfg: forcing configuration, or ``None`` for pure-temporal.
    :return: ``(train_ds, test_ds)`` :class:`SpatiotemporalDataset` instances.
    """
    grid: Optional[ReferenceGrid] = None

    if cfg.use_synthetic:
        logger.info("Generating synthetic dataset (%d frames).", cfg.synthetic_n_frames)
        frames, dates, mask = _make_synthetic(cfg)
    else:
        data_dir = cfg.lst_dir if cfg.modality == "lst" else cfg.ndvi_dir
        dated = _discover_frames(data_dir)
        logger.info(
            "Found %d %s frames in %s (%s -> %s).",
            len(dated), cfg.modality.upper(), data_dir, dated[0][0], dated[-1][0],
        )
        # Reference grid from the first tile: precipitation is warped onto THIS.
        grid = read_reference_grid(dated[0][1])
        bands, masks, dates = [], [], []
        for d, p in dated:
            band, m = _read_geotiff(p, cfg.image_height, cfg.image_width, cfg.nodata_value)
            bands.append(band)
            masks.append(m)
            dates.append(d)
        frames = np.stack(bands, axis=0)[..., None]  # (n, H, W, 1)
        if cfg.n_channels > 1:
            frames = np.repeat(frames, cfg.n_channels, axis=-1)
        mask = np.stack(masks, axis=0)

    n = frames.shape[0]

    # Chronological split (no shuffle): first 85% train, last 15% test.
    n_train = int(round(cfg.train_fraction * n))
    n_train = max(2, min(n - 1, n_train))
    logger.info("Chronological split: %d train / %d test frames.", n_train, n - n_train)

    # --- Forcing (built against THIS modality's acquisition dates) ---
    fs: Optional[ForcingSeries] = None
    if forcing_cfg is not None and forcing_cfg.ell() > 0:
        if cfg.forcing_path and os.path.exists(cfg.forcing_path):
            logger.info("Loading precomputed forcing from %s", cfg.forcing_path)
            fs = load_forcing(cfg.forcing_path)
            if len(fs.dates) != n:
                raise ValueError(
                    "Precomputed forcing has {} rows but the {} dataset has {} frames. "
                    "Re-run experiments/preprocess_forcing.py for this modality.".format(
                        len(fs.dates), cfg.modality, n
                    )
                )
        elif cfg.use_synthetic:
            fs = synthetic_forcing(dates, forcing_cfg)
        else:
            # AOI mask: a pixel is valid if it is valid in most frames.
            aoi = mask.mean(axis=0) > 0.5
            fs = build_forcing(
                dates, forcing_cfg, grid=grid, valid_mask=aoi, n_train=n_train
            )

    # Normalisation from the training split only (valid pixels).
    if cfg.normalize:
        train_vals = frames[:n_train][mask[:n_train]]
        mean = np.array([train_vals.mean()], dtype=np.float32)
        std = np.array([train_vals.std() + 1e-6], dtype=np.float32)
    else:
        mean = np.array([0.0], dtype=np.float32)
        std = np.array([1.0], dtype=np.float32)
    frames_norm = ((frames - mean) / std).astype(np.float32)
    frames_norm = np.where(mask[..., None], frames_norm, 0.0).astype(np.float32)

    def _slice(a: int, b: int) -> SpatiotemporalDataset:
        return SpatiotemporalDataset(
            frames=frames_norm[a:b],
            dates=dates[a:b],
            forcing=None if fs is None else fs.values[a:b],
            mean=mean,
            std=std,
            valid_mask=mask[a:b],
            forcing_names=None if fs is None else fs.names,
            forcing_scale=None if fs is None else fs.scale,
        )

    return _slice(0, n_train), _slice(n_train, n)


def make_pixel_grid(height: int, width: int) -> np.ndarray:
    """
    Build the fixed normalised pixel-coordinate grid used by the spatial basis.

    Coordinates are in ``[-1, 1]^2`` (row-major).

    :param height: grid height.
    :param width: grid width.
    :return: ``(H * W, 2)`` float32 array of ``(x, y)`` coordinates.
    """
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
