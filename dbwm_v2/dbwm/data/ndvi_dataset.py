"""
Calendar-reindexed NDVI/LST loading for the v4 pipeline.

Differences from :mod:`dbwm.data.geotiff_dataset` (the v2 encoder-path loader),
each of them load-bearing for the memory lift:

**A complete daily calendar.** The archive holds ~1,568 frames over the 1,581 days
from 2022-01-01 to 2026-04-30, so ~13 dates have no GeoTIFF. Frames are placed on
the *full* daily index with the absent dates marked unobserved, rather than
compressed to the observed dates. The memory lift assumes ``w_t, ..., w_{t-6}`` are
six **consecutive days**; compressing would make some "lag-6" span seven or eight
real days and would bias the Mori-Zwanzig memory-depth measurement. The filter
predicts through the gaps, the daily weather sensor still fires, and Prop. 2.9
fixed-lag smoothing retro-corrects them.

**Native resolution.** The Sayedanwala tiles are 135 x 125 at 30 m. Resampling up
to 256^2 would invent 3.7x more pixels than the sensor recorded.

**Per-date validity.** ~48% of the bounding box is valid (a field clip, or cloud).
The mask is kept per date and, if it turns out to be date-invariant, that fact is
detected and exploited: ``Phi_X`` and ``Lambda_X`` can then be built once instead
of per date, turning an ``O(T n r^2)`` cost into ``O(n r^2)``.

**Real geographic coordinates.** The spatial basis is evaluated on coordinates
derived from the GeoTIFF affine transform, normalised to ``[-1, 1]^2`` over the
tile, so ``Psi`` learns structure in metres rather than in array indices.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.data.forcing import parse_date
from dbwm.data.seasons import season_labels, split_indices
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.ndvi")


@dataclass
class CalendarDataset:
    """
    A daily-complete stack of frames with per-date validity and season labels.

    :ivar frames: ``(T, H, W)`` float32, normalised; rows where ``observed`` is
        ``False`` are zero-filled and must never be read as data.
    :ivar valid_mask: ``(T, H, W)`` bool, per-date pixel validity.
    :ivar observed: ``(T,)`` bool, ``True`` where a GeoTIFF existed for that date.
    :ivar dates: ``(T,)`` the complete daily calendar.
    :ivar coords: ``(H*W, 2)`` pixel coordinates normalised to ``[-1, 1]^2``.
    :ivar mean: normalisation mean (training split only).
    :ivar std: normalisation std (training split only).
    :ivar static_mask: ``(H, W)`` mask valid on every observed date, or ``None`` if
        the per-date masks differ.
    :ivar pixel_size: ground sample distance in **metres**, converted from degrees
        when the raster sits on a geographic CRS (see :func:`ground_sample_distance`).
    :ivar transform: the GeoTIFF affine coefficients, kept for georeferencing.
    :ivar crs: the coordinate reference system string.
    :ivar modality: ``"NDVI"`` or ``"LST"``. Carried on the dataset because every
        presentation surface downstream -- band names, colour ramps, unit labels,
        exported filenames -- needs it, and reading it off the config in each of
        them is how the two drift.
    """

    frames: np.ndarray
    valid_mask: np.ndarray
    observed: np.ndarray
    dates: List[dt.date]
    coords: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    static_mask: Optional[np.ndarray] = None
    pixel_size: float = 30.0
    transform: Optional[Tuple[float, ...]] = None
    crs: Optional[str] = None
    modality: str = "NDVI"
    #: Cache for :meth:`common_mask`, which several stages call per run.
    _common_mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_steps(self) -> int:
        """Length of the daily calendar."""
        return self.frames.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """``(H, W)`` of a frame."""
        return self.frames.shape[1:3]

    @property
    def mask_is_static(self) -> bool:
        """Whether every observed date shares the same validity mask."""
        return self.static_mask is not None

    def coverage(self) -> np.ndarray:
        """
        Fraction of **observed** dates on which each pixel is valid.

        :return: ``(n_pixels,)`` in ``[0, 1]``; all zeros if nothing is observed.
        """
        obs = np.asarray(self.observed, dtype=bool)
        if not obs.any():  # pragma: no cover - an empty archive fails earlier
            return np.zeros(int(np.prod(self.shape)))
        return self.valid_mask[obs].reshape(int(obs.sum()), -1).mean(axis=0)

    def common_mask(self, min_pixels: int = 64) -> np.ndarray:
        """
        The pixel mask every score, basis fit and figure is computed on.

        Replaces an expression that was written out by hand in nine places::

            ds.static_mask.reshape(-1) if ds.mask_is_static
            else ds.valid_mask.reshape(ds.n_steps, -1).all(axis=0)

        and was wrong in the second branch, silently. The intersection ran over
        **every calendar step**, including the unobserved ones -- whose masks are
        all-``False`` by construction, since no raster was read into them. So the
        moment a run had one missing date *and* a non-static mask, the common mask
        was identically empty and everything downstream operated on zero pixels.
        NDVI never hit it because its mask is date-invariant and the first branch
        applied; LST hits it immediately.

        The intersection is therefore taken over observed dates only. If that is
        still too small to work with, it is relaxed to a coverage rule -- pixels
        valid on at least a given fraction of observed dates -- rather than
        failing, because on a real thermal archive a strict intersection is
        fragile: one cloudy date at one pixel excludes that pixel for the whole
        run. The relaxation is logged, with the number it settled on, so a
        shrinking mask is visible rather than inferred from a later error.

        :param min_pixels: fall back to a coverage rule below this count.
        :return: ``(n_pixels,)`` boolean mask.
        """
        if self._common_mask is not None:
            return self._common_mask

        obs = np.asarray(self.observed, dtype=bool)
        cov = self.coverage()
        mask = cov >= 1.0
        if int(mask.sum()) < min_pixels:
            for frac in (0.99, 0.95, 0.9, 0.75, 0.5):
                relaxed = cov >= frac
                if int(relaxed.sum()) >= min_pixels:
                    logger.warning(
                        "Strict date-intersection leaves only %d valid pixel(s) "
                        "over %d observed dates; relaxed to pixels valid on >= "
                        "%.0f%% of them, giving %d. A pixel excluded by the "
                        "strict rule is one that was invalid on at least one "
                        "date -- cloud, or the physical-range gate -- not one "
                        "that is never observed.",
                        int(mask.sum()), int(obs.sum()), 100 * frac,
                        int(relaxed.sum()),
                    )
                    mask = relaxed
                    break
        self._common_mask = mask
        return mask

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Recover physical units (NDVI index / degrees C).

        :param x: normalised values.
        :return: de-normalised values.
        """
        return x * self.std + self.mean

    def season_of_each_step(self) -> np.ndarray:
        """``(T,)`` season name per calendar date."""
        return season_labels(self.dates)

    def subset(self, idx: np.ndarray) -> "CalendarDataset":
        """
        Slice the dataset along time, keeping everything else intact.

        :param idx: integer indices into the calendar.
        :return: a new :class:`CalendarDataset`.
        """
        idx = np.asarray(idx, dtype=int)
        return CalendarDataset(
            frames=self.frames[idx],
            valid_mask=self.valid_mask[idx],
            observed=self.observed[idx],
            dates=[self.dates[i] for i in idx],
            coords=self.coords,
            mean=self.mean,
            std=self.std,
            static_mask=self.static_mask,
            pixel_size=self.pixel_size,
            transform=self.transform,
            crs=self.crs,
            modality=self.modality,
        )


def probe_raster(path: str) -> Dict[str, object]:
    """
    Read one raster's geometry **without** reading its pixels.

    Used to size the run before any data is loaded: at ~6.8M pixels per frame and
    1,581 dates, materialising the full stack is 50 GB, and finding that out three
    minutes into a load is far too late.

    :param path: file path.
    :return: dict with ``shape``, ``n_pixels``, ``transform``, ``crs``,
             ``pixel_size``.
    """
    import rasterio

    with rasterio.open(path) as src:
        return {
            "shape": (int(src.height), int(src.width)),
            "n_pixels": int(src.height) * int(src.width),
            "transform": tuple(src.transform)[:6],
            "crs": None if src.crs is None else str(src.crs),
            "pixel_size": float(abs(src.transform.a)),
        }


def ground_sample_distance(
    transform: Optional[Sequence[float]],
    crs: Optional[str],
    fallback: float = 30.0,
) -> float:
    """
    Ground sample distance in **metres**, whatever CRS the raster is on.

    Why this is not simply ``abs(transform.a)``
    -------------------------------------------
    The NDVI archive is EPSG:32643 (UTM 43N) with a 30.0 m pixel, so the affine
    ``a`` coefficient *is* the GSD in metres and the distinction never arose. The
    LST archive is EPSG:4326: its ``a`` is ``0.00026949...`` **degrees**. Taking
    that as a metre count understates every ground distance by a factor of about
    10^5, and the number is not cosmetic -- it is the bin width of the
    spatiotemporal covariogram ``C(h, u)``, so every separation and the
    separability test built on them would be computed against a nonsense axis.

    The conversion uses the local scale of the ellipsoid at the tile's own
    latitude: one degree of latitude is ~111.32 km, one degree of longitude is
    that times ``cos(lat)``. At the Sayedanwala latitude (31.08 deg N) a square
    0.000269 deg pixel is ~25.7 m east-west and ~30.0 m north-south -- exactly the
    anisotropy the raster report quotes. A single scalar GSD cannot express both,
    so the **geometric mean** is returned: it is the side of the square with the
    same ground area, which is the right summary for an isotropic distance bin.

    :param transform: affine coefficients ``(a, b, c, d, e, f)``, or ``None``.
    :param crs: the CRS string; a geographic one triggers the conversion.
    :param fallback: value returned when the transform is unavailable.
    :return: the GSD in metres.
    """
    if transform is None:
        return float(fallback)
    a, b, c, d, e, f = [float(v) for v in transform[:6]]
    dx, dy = abs(a), abs(e)
    if not _is_geographic(crs):
        return float(dx) if dx > 0 else float(fallback)

    # Centre latitude of the tile, from the affine origin. `f` is the top edge and
    # `e` is negative for a north-up raster, so this is only exact at the origin
    # row; over a ~4 km tile the cosine varies by <0.1% and the refinement is not
    # worth the extra plumbing.
    lat = f
    m_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * np.radians(lat))
    m_per_deg_lon = 111_412.84 * np.cos(np.radians(lat)) - 93.5 * np.cos(
        3 * np.radians(lat)
    )
    gx, gy = dx * abs(m_per_deg_lon), dy * abs(m_per_deg_lat)
    if gx <= 0 or gy <= 0:  # pragma: no cover - degenerate transform
        return float(fallback)
    return float(np.sqrt(gx * gy))


def _is_geographic(crs: Optional[str]) -> bool:
    """
    Whether a CRS string denotes a geographic (degree-valued) reference system.

    Uses ``rasterio``/``pyproj`` when the string parses, and falls back to the
    handful of EPSG codes that actually appear in this project so the function is
    still usable in a test that has no CRS database.

    :param crs: the CRS string, e.g. ``"EPSG:4326"``.
    :return: ``True`` for a geographic CRS.
    """
    if not crs:
        return False
    try:
        from rasterio.crs import CRS

        return bool(CRS.from_user_input(str(crs)).is_geographic)
    except Exception:
        return str(crs).upper().replace(" ", "") in {"EPSG:4326", "EPSG:4979", "WGS84"}


def decimation_for(n_pixels: int, max_pixels: Optional[int],
                   shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Choose a decimated read shape that keeps a frame under ``max_pixels``.

    Decimation is not a compromise here, it is what v2 Sec. 8.1 prescribes
    ("subsampling to n ~ 1e5 training pixels per day is sufficient"). The GP
    posterior is a **rank-r** reconstruction: with ``r = 256``, 250k well-spread
    pixels already satisfy Prop. 3.1's ``n >> r`` by a factor of ~1000, and the
    extra 6.5M pixels add essentially nothing to ``Lambda_X`` while costing 50 GB
    of RAM. What they *would* buy is finer output maps, which is a rendering
    concern handled at decode time, not a reason to hold the whole stack.

    :param n_pixels: pixels in the native frame.
    :param max_pixels: budget, or ``None``/``0`` for native resolution.
    :param shape: native ``(H, W)``.
    :return: ``(out_h, out_w)``, or ``None`` to read natively.
    """
    if not max_pixels or n_pixels <= max_pixels:
        return None
    scale = (max_pixels / float(n_pixels)) ** 0.5
    h, w = shape
    return (max(1, int(round(h * scale))), max(1, int(round(w * scale))))


def _read_frame(
    path: str, nodata: float, height: Optional[int], width: Optional[int],
    physical_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Read one GeoTIFF band plus its validity mask and georeferencing.

    When ``height``/``width`` are given, rasterio performs a **decimated read**:
    the full-resolution array is never allocated, so peak memory is the decimated
    size rather than the native one. The reported ``pixel_size`` and affine
    transform are rescaled to match, which matters because the spatiotemporal
    covariogram bins ``C(h, u)`` by distance in metres -- leaving the native GSD
    in place would mis-state every separation.

    :param path: file path.
    :param nodata: sentinel marking invalid pixels.
    :param height: target height, or ``None`` for native.
    :param width: target width, or ``None`` for native.
    :param physical_range: ``(lo, hi)`` outside which a pixel is marked invalid,
        or ``None`` to accept any finite value. See :func:`load_calendar_dataset`
        for why an LST archive needs this and NDVI does not.
    :return: ``(band, mask, meta)``.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        out_shape = None
        if height is not None and width is not None:
            out_shape = (height, width)
        band = src.read(
            1,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        file_nodata = src.nodata
        native = (int(src.height), int(src.width))
        a, b, c, d, e, f = tuple(src.transform)[:6]
        sy = native[0] / band.shape[0]
        sx = native[1] / band.shape[1]
        meta = {
            # Rescale so ground geometry stays correct after decimation.
            "transform": (a * sx, b * sy, c, d * sx, e * sy, f),
            "crs": None if src.crs is None else str(src.crs),
            "pixel_size": float(abs(a) * sx),
            "native_shape": native,
            "decimation": (sy, sx),
        }

    mask = np.isfinite(band)
    if file_nodata is not None:
        mask &= band != file_nodata
    mask &= band != nodata
    if physical_range is not None:
        lo, hi = float(physical_range[0]), float(physical_range[1])
        # Applied BEFORE anything is averaged into the normalisation statistics,
        # which is the only place it can do any good: a single 1e4 K artefact
        # anywhere in the archive moves the training mean and std, and every
        # frame in the run is then standardised against it.
        in_range = (band >= lo) & (band <= hi)
        # Counted against the pixels that survived the no-data checks, so the
        # ~52% of the bounding box outside the field clip is not reported as an
        # out-of-range anomaly.
        meta["n_out_of_range"] = int((mask & ~in_range).sum())
        mask &= in_range
    band = np.where(mask, band, 0.0).astype(np.float32)
    return band, mask, meta


def discover_frames(data_dir: str) -> List[Tuple[dt.date, str]]:
    """
    List dated GeoTIFFs in a modality folder, chronologically, rejecting duplicates.

    Duplicate dates are an error rather than a warning: the calendar index assumes
    one frame per day, and silently keeping the last one would drop data without
    trace.

    :param data_dir: folder holding the rasters.
    :return: ``[(date, path), ...]`` sorted by date.
    :raises ValueError: if two files map to the same date.
    """
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.tif"))
        + glob.glob(os.path.join(data_dir, "*.tiff"))
    )
    if not paths:
        raise FileNotFoundError(
            "No GeoTIFFs found in {}. Point --ndvi-dir/--lst-dir at the Drive "
            "folder, or use --synthetic.".format(data_dir)
        )
    seen: Dict[dt.date, str] = {}
    undated = []
    for p in paths:
        d = parse_date(p)
        if d is None:
            undated.append(os.path.basename(p))
            continue
        if d in seen:
            raise ValueError(
                "Two files map to {}: {} and {}. The daily calendar assumes one "
                "frame per date.".format(d, os.path.basename(seen[d]), os.path.basename(p))
            )
        seen[d] = p
    if undated:
        logger.warning(
            "%d file(s) had no parsable date and were skipped (e.g. %s).",
            len(undated), undated[:3],
        )
    if not seen:
        raise ValueError(
            "Found {} GeoTIFFs in {} but none had a parsable date "
            "(expected YYYYMMDD / YYYY-MM-DD / YYYYDDD).".format(len(paths), data_dir)
        )
    return sorted(seen.items())


def make_coordinate_grid(
    shape: Tuple[int, int], transform: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Pixel-centre coordinates normalised to ``[-1, 1]^2``.

    Built from the affine transform when available so the basis ``Psi`` sees real
    ground geometry (anisotropic pixels, north-up or not) rather than array indices.

    :param shape: ``(H, W)``.
    :param transform: affine coefficients ``(a, b, c, d, e, f)``, or ``None``.
    :return: ``(H*W, 2)`` float32 coordinates.
    """
    h, w = shape
    rows, cols = np.meshgrid(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
    )
    if transform is None:
        x, y = cols, rows
    else:
        a, b, c, d, e, f = [float(v) for v in transform[:6]]
        x = a * (cols + 0.5) + b * (rows + 0.5) + c
        y = d * (cols + 0.5) + e * (rows + 0.5) + f

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    return np.stack([_norm(x).ravel(), _norm(y).ravel()], axis=-1).astype(np.float32)


def load_calendar_dataset(
    data_dir: Optional[str],
    archive_start: dt.date,
    archive_end: dt.date,
    split_date: dt.date,
    nodata: float = -9999.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    normalize: bool = True,
    max_pixels: Optional[int] = 250_000,
    ram_budget_gb: float = 8.0,
    synthetic: bool = False,
    synthetic_shape: Tuple[int, int] = (32, 32),
    seed: int = 0,
    modality: str = "NDVI",
    physical_range: Optional[Tuple[float, float]] = None,
    drop_empty: bool = False,
    min_valid_fraction: float = 0.0,
) -> CalendarDataset:
    """
    Load one modality onto the complete daily calendar ``[archive_start, archive_end]``.

    Normalisation statistics come from the **training** side of ``split_date`` only.

    :param data_dir: folder of dated GeoTIFFs, or ``None`` when ``synthetic``.
    :param archive_start: first calendar date (inclusive).
    :param archive_end: last calendar date (inclusive).
    :param split_date: first date of the test split, used for the normalisation
        statistics so no test-period information leaks in.
    :param nodata: no-data sentinel.
    :param height: resample height, or ``None`` for native.
    :param width: resample width, or ``None`` for native.
    :param normalize: standardise using training-split statistics.
    :param max_pixels: per-frame pixel budget. Frames larger than this are read
        **decimated**, so the native array is never allocated. ``None``/``0`` reads
        natively. See :func:`decimation_for` for why this costs the model nothing.
    :param ram_budget_gb: refuse to build a stack larger than this, with
        instructions, rather than letting the process be OOM-killed minutes in.
    :param synthetic: generate a synthetic archive instead of reading files.
    :param synthetic_shape: ``(H, W)`` for the synthetic archive.
    :param seed: RNG seed for the synthetic path.
    :param modality: ``"NDVI"`` or ``"LST"``, recorded on the dataset so the
        presentation layer (band names, ramps, unit labels) does not have to
        re-derive it.
    :param physical_range: ``(lo, hi)`` in physical units, outside which a pixel is
        treated as no-data. ``None`` disables the gate.

        NDVI needs no gate: the index is bounded by its own definition and the
        product is already clipped. A **downscaled LST** product is different --
        where the downscaling extrapolates beyond its training support it can emit
        physically impossible temperatures, and a single such pixel anywhere in a
        1,581-date archive shifts the training mean and std that every frame is
        then standardised against.

        This must be a *plausibility* bound (250-350 K), never the display range:
        set to the latter it discarded 8.4% of the real archive, because 47 degC
        bare soil in June is ordinary data and an unremarkable colour at the same
        time.
    :param drop_empty: treat a raster with no usable pixels as an **unobserved**
        date rather than as an empty observation. A file can exist and be entirely
        no-data -- a failed downscaling, a fully clouded overpass -- and counting
        it as observed poisons the date-intersection mask that the basis fit, the
        GP solve and every metric are computed on, emptying the run.
    :param min_valid_fraction: with ``drop_empty``, the fraction of the bounding
        box that must be valid for a frame to count as an observation.
    :return: the assembled :class:`CalendarDataset`.
    """
    calendar = [
        archive_start + dt.timedelta(days=i)
        for i in range((archive_end - archive_start).days + 1)
    ]
    index = {d: i for i, d in enumerate(calendar)}
    t_total = len(calendar)

    if synthetic:
        frames, masks, observed, meta = _synthetic_archive(
            calendar, synthetic_shape, seed, modality=modality
        )
    else:
        if data_dir is None:
            raise ValueError("data_dir is required unless synthetic=True")
        dated = discover_frames(data_dir)
        outside = [d for d, _ in dated if d not in index]
        if outside:
            logger.warning(
                "%d frame(s) fall outside [%s, %s] and were dropped (e.g. %s).",
                len(outside), archive_start, archive_end, outside[:3],
            )
        dated = [(d, p) for d, p in dated if d in index]
        if not dated:
            raise ValueError(
                "No frames fall inside [{}, {}].".format(archive_start, archive_end)
            )
        # Size the run BEFORE reading anything. A full-scene archive is ~6.8M
        # pixels per frame; at 1,581 dates that is a 50 GB stack, and discovering
        # it three minutes into the load is far too late.
        probe = probe_raster(dated[0][1])
        native_h, native_w = probe["shape"]
        decimated = False
        if height is None or width is None:
            dec = decimation_for(probe["n_pixels"], max_pixels, probe["shape"])
            if dec is not None:
                height, width = dec
                decimated = True
        n_px = (height * width) if (height and width) else probe["n_pixels"]
        est_gb = len(dated) * n_px * 5 / 1024**3  # float32 frame + bool mask

        logger.info(
            "Raster geometry: native %d x %d = %s px, %.1f m GSD. Reading at "
            "%d x %d = %s px -> estimated stack %.2f GB.",
            native_h, native_w, f"{probe['n_pixels']:,}", probe["pixel_size"],
            height or native_h, width or native_w, f"{n_px:,}", est_gb,
        )
        if decimated:
            logger.info(
                "Decimated read: the native array is never allocated. With r basis "
                "functions the GP posterior is a rank-r reconstruction, so v2 "
                "Sec. 8.1's ~1e5 pixels/day already satisfies Prop. 3.1's n >> r "
                "by ~1000x; the discarded pixels buy output resolution, not "
                "estimation accuracy."
            )
        if est_gb > ram_budget_gb:
            raise MemoryError(
                "Estimated frame stack is {:.1f} GB, over the {:.1f} GB budget.\n"
                "  native {} x {} px x {} dates.\n"
                "Options:\n"
                "  --max-pixels 250000   decimate on read (default; never "
                "allocates the native array)\n"
                "  --max-pixels 50000    smaller still, if RAM is tight\n"
                "  --ram-budget-gb N     raise the budget if you really have the "
                "memory\n"
                "Reading natively here would need ~{:.0f} GB.".format(
                    est_gb, ram_budget_gb, native_h, native_w, len(dated),
                    len(dated) * probe["n_pixels"] * 5 / 1024**3,
                )
            )

        band0, mask0, meta = _read_frame(
            dated[0][1], nodata, height, width, physical_range
        )
        h, w = band0.shape
        frames = np.zeros((t_total, h, w), dtype=np.float32)
        masks = np.zeros((t_total, h, w), dtype=bool)
        observed = np.zeros(t_total, dtype=bool)
        n_out_of_range = 0
        empty: List[dt.date] = []
        min_valid_px = int(np.ceil(min_valid_fraction * h * w))
        for d, p in dated:
            band, mask, fmeta = _read_frame(p, nodata, height, width, physical_range)
            n_out_of_range += int(fmeta.get("n_out_of_range", 0))
            if drop_empty and int(mask.sum()) <= min_valid_px:
                # A raster that exists but carries no usable pixels is NOT an
                # observation, and marking it as one is not a harmless
                # bookkeeping choice. `observed` gates the date-intersection mask,
                # the GP solve and the Kalman update, so a single all-no-data
                # frame drives the common mask to zero pixels and the run dies in
                # Stage 2 with "fewer than 64 pixels are reliably valid" -- an
                # error that names the AOI when the cause is one bad file.
                #
                # Left unobserved, the date is handled by machinery that already
                # exists and is correct for it: the filter predicts through the
                # gap and Prop. 2.9 fixed-lag smoothing retro-corrects it, exactly
                # as for a date with no raster at all.
                empty.append(d)
                continue
            if band.shape != (h, w):
                raise ValueError(
                    "Frame {} has shape {} but the first frame has {}. All tiles "
                    "must share a grid.".format(os.path.basename(p), band.shape, (h, w))
                )
            i = index[d]
            frames[i], masks[i], observed[i] = band, mask, True

        if empty:
            logger.warning(
                "%s: %d raster(s) of %d carry no usable pixels (<= %d valid) and "
                "are treated as UNOBSERVED dates, not as empty observations "
                "(e.g. %s). The filter predicts through them and fixed-lag "
                "smoothing retro-corrects them, exactly as for a date with no "
                "file. Counting them as observed would drive the date-"
                "intersection mask to zero pixels and fail Stage 2.",
                modality, len(empty), len(dated), min_valid_px,
                ", ".join(str(d) for d in empty[:5]),
            )
        if physical_range is not None and n_out_of_range:
            total = masks.sum() + n_out_of_range
            logger.warning(
                "%s: %d pixel(s) (%.4f%% of otherwise-valid data) fell outside the "
                "physical range [%.2f, %.2f] and were marked no-data. A "
                "downscaled product extrapolating past its training support is "
                "the usual cause; pass --no-physical-range to keep them and see "
                "what they do to the normalisation.",
                modality, n_out_of_range, 100.0 * n_out_of_range / max(total, 1),
                physical_range[0], physical_range[1],
            )

    n_missing = int((~observed).sum())
    logger.info(
        "Calendar %s .. %s: %d days, %d observed, %d missing (%.1f%%).",
        archive_start, archive_end, t_total, int(observed.sum()), n_missing,
        100.0 * n_missing / max(t_total, 1),
    )
    if n_missing:
        gaps = _gap_runs(observed)
        logger.info(
            "Missing-date runs: %d (longest %d days). The filter predicts through "
            "these; the daily weather sensor still fires and Prop. 2.9 fixed-lag "
            "smoothing retro-corrects them.",
            len(gaps), max(g[1] for g in gaps),
        )
    coverage = float(observed.mean()) if t_total else 0.0
    if coverage < 0.5:
        # The memory lift conditions on w_t..w_{t-L+1} being L CONSECUTIVE days
        # (v3 Def. 2.3), and a forecast origin additionally needs L-1 observed
        # days behind it. On a gap-filled daily product that is nearly free; on a
        # native-revisit archive (Landsat is 16-day, so ~6% coverage) almost no
        # date qualifies and Stage 9 would score zero origins after an hour of
        # training. Said here rather than discovered there.
        logger.warning(
            "%s coverage is only %.1f%% of the daily calendar (%d of %d days, "
            "longest observed run %d days). The memory lift needs L consecutive "
            "observed days per origin, so at this density most origins will be "
            "rejected. If this archive is at native revisit rather than daily "
            "gap-filled, either gap-fill it first or reduce --memory-order to "
            "what the runs actually support.",
            modality, 100.0 * coverage, int(observed.sum()), t_total,
            _longest_run(observed),
        )

    # A date-invariant mask lets Phi_X / Lambda_X be built ONCE instead of per date.
    obs_masks = masks[observed]
    static = None
    if obs_masks.shape[0] and np.all(obs_masks == obs_masks[0]):
        static = obs_masks[0].copy()
        logger.info(
            "Validity mask is date-invariant (%d of %d pixels valid, %.1f%%): "
            "Phi_X and Lambda_X will be cached once.",
            int(static.sum()), static.size, 100.0 * static.mean(),
        )
    else:
        frac = obs_masks.mean(axis=(1, 2)) if obs_masks.shape[0] else np.zeros(0)
        logger.info(
            "Validity mask varies by date (valid fraction %.1f%%-%.1f%%): "
            "Lambda_X is rebuilt per date, and the GP posterior covariance "
            "sigma_eps^2 Lambda_{X_t}^{-1} then genuinely differs between dates.",
            100.0 * float(frac.min()) if frac.size else 0.0,
            100.0 * float(frac.max()) if frac.size else 0.0,
        )

    train_idx, _ = split_indices(calendar, split_date)
    train_sel = np.zeros(t_total, dtype=bool)
    train_sel[train_idx] = True
    usable = train_sel & observed
    if normalize and usable.any() and masks[usable].any():
        vals = frames[usable][masks[usable]]
        mean = np.array([vals.mean()], dtype=np.float32)
        std = np.array([vals.std() + 1e-6], dtype=np.float32)
        # The archive's ACTUAL span, so the fixed display range can be checked
        # against it rather than assumed to fit. Clipping is legitimate -- it
        # keeps t+1 and t+6 comparable -- but it should be a known quantity.
        lo, hi = np.percentile(vals, [0.1, 99.9])
        logger.info(
            "%s training values: min %.3f, p0.1 %.3f, mean %.3f, p99.9 %.3f, "
            "max %.3f (std %.3f).",
            modality, float(vals.min()), float(lo), float(mean[0]), float(hi),
            float(vals.max()), float(std[0]),
        )
    else:
        mean = np.array([0.0], dtype=np.float32)
        std = np.array([1.0], dtype=np.float32)
    frames = ((frames - mean) / std).astype(np.float32)
    frames = np.where(masks, frames, 0.0).astype(np.float32)
    frames[~observed] = 0.0

    transform = meta.get("transform")
    crs = meta.get("crs")
    gsd = ground_sample_distance(transform, crs, float(meta.get("pixel_size", 30.0)))
    if _is_geographic(crs):
        logger.info(
            "%s rasters are on a geographic CRS (%s): the affine pixel size is "
            "%.3e degrees, converted to %.2f m for the covariogram distance axis. "
            "Note the pixel is anisotropic on the ground (~%.1f m E-W vs ~%.1f m "
            "N-S at this latitude); %.2f m is the equal-area geometric mean.",
            modality, crs, abs(float(transform[0])) if transform else float("nan"),
            gsd,
            abs(float(transform[0])) * 111_412.84 * np.cos(np.radians(float(transform[5])))
            if transform else float("nan"),
            abs(float(transform[4])) * 111_132.92 if transform else float("nan"),
            gsd,
        )

    return CalendarDataset(
        frames=frames,
        valid_mask=masks,
        observed=observed,
        dates=calendar,
        coords=make_coordinate_grid(frames.shape[1:3], transform),
        mean=mean,
        std=std,
        static_mask=static,
        pixel_size=gsd,
        transform=transform,
        crs=crs,
        modality=str(modality).upper(),
    )


def _gap_runs(observed: np.ndarray) -> List[Tuple[int, int]]:
    """
    Locate runs of consecutive unobserved dates.

    :param observed: ``(T,)`` bool.
    :return: ``[(start_index, length), ...]``.
    """
    out, start = [], None
    for i, ok in enumerate(observed):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(observed) - start))
    return out


def _longest_run(observed: np.ndarray) -> int:
    """
    Length of the longest run of consecutive observed dates.

    This is the hard ceiling on the memory order the archive can support: an
    origin needs ``L`` consecutive observed days behind it, so no origin exists at
    all once ``L`` exceeds this.

    :param observed: ``(T,)`` bool.
    :return: the longest run length.
    """
    best = run = 0
    for ok in np.asarray(observed, dtype=bool):
        run = run + 1 if ok else 0
        best = max(best, run)
    return int(best)


def _synthetic_archive(
    calendar: Sequence[dt.date], shape: Tuple[int, int], seed: int,
    modality: str = "NDVI",
):
    """
    Generate a synthetic archive with seasonal structure, gaps and a field clip.

    The synthetic field is deliberately built with (a) an annual phenological cycle,
    (b) genuine temporal memory beyond one step, and (c) a static invalid region --
    so the whiteness pretest, the memory-depth sweep and the masked-pixel handling
    are all exercised without the real data.

    The same spatial modes and the same AR(2) temporal structure are used for both
    modalities; only the **level and amplitude** are rescaled, to roughly 300 K
    with an 8 K annual swing for LST. That is not cosmetic: a smoke run in LST
    mode applies the LST physical-range gate, and an NDVI-valued synthetic field
    (everything near 0.55) would be masked out entirely, leaving the pipeline with
    zero valid pixels and an error a long way from its cause. Keeping the
    *structure* identical is what lets an LST smoke run and an NDVI smoke run be
    compared as the same test.

    Note the LST field is built with the **opposite** seasonal phase to NDVI, which
    is the physically right relationship over an irrigated field: greenness peaks
    when evaporative cooling is strongest, so the surface is coolest when the
    canopy is densest.

    :param calendar: the daily calendar.
    :param shape: ``(H, W)``.
    :param seed: RNG seed.
    :param modality: ``"NDVI"`` or ``"LST"``, selecting the level and amplitude.
    :return: ``(frames, masks, observed, meta)``.
    """
    rng = np.random.RandomState(seed)
    t_total = len(calendar)
    h, w = shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij"
    )
    modes = np.stack(
        [
            np.sin(2.0 * xx),
            np.cos(3.0 * yy),
            np.exp(-(xx**2 + yy**2)),
            xx * yy,
        ]
    )

    doy = np.array([d.timetuple().tm_yday for d in calendar], dtype=np.float64)
    phase = 2.0 * np.pi * (doy - 1.0) / 365.25
    amps = np.zeros((t_total, 4))
    amps[:, 0] = 0.6 + 0.35 * np.sin(phase - 0.7)
    amps[:, 1] = 0.25 * np.cos(phase)
    # An AR(2) component gives the record real memory, so the L=1 whiteness
    # pretest has something to reject.
    e = rng.randn(t_total) * 0.08
    for i in range(2, t_total):
        amps[i, 2] = 0.7 * amps[i - 1, 2] + 0.25 * amps[i - 2, 2] + e[i]
    amps[:, 3] = 0.1 * np.sin(phase * 2.0)

    frames = np.einsum("tk,khw->thw", amps, modes).astype(np.float32)
    if str(modality).upper().startswith("LST"):
        # ~300 K mean, ~8 K annual swing, 0.2 K observation noise. The sign flip
        # gives the physically correct anti-phase with greenness.
        frames = (300.0 - 8.0 * frames).astype(np.float32)
        frames += 0.2 * rng.randn(t_total, h, w).astype(np.float32)
    else:
        frames += 0.55 + 0.01 * rng.randn(t_total, h, w).astype(np.float32)

    # A static field clip: roughly half the bounding box is outside the parcel.
    clip = (xx + 0.35) ** 2 + (yy - 0.1) ** 2 < 0.9
    masks = np.broadcast_to(clip, (t_total, h, w)).copy()

    observed = np.ones(t_total, dtype=bool)
    n_missing = max(1, int(round(0.008 * t_total)))
    observed[rng.choice(t_total, size=n_missing, replace=False)] = False

    meta = {
        "transform": (30.0, 0.0, 419670.0, 0.0, -30.0, 3440790.0),
        "crs": "EPSG:32643",
        "pixel_size": 30.0,
        "native_shape": (h, w),
    }
    return frames, masks, observed, meta
