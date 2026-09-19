"""
Exogenous forcing pipeline: precipitation ``p_t`` and irrigation ``u_t``.

This builds the raw input vector ``u_t^raw = [p_t, u_t]`` and hence the regressor

    Upsilon = [u_1^raw, ..., u_{T-1}^raw]  in  R^{ell x (T-1)}

from which Algorithm 3 Stage II *learns* the forcing directions

    [B_p  B_u] = dW . Upsilon^T . (Upsilon Upsilon^T + mu I_ell)^{-1}  in R^{r x ell}

(``B`` is never supplied -- only ``Upsilon`` is; see
:mod:`dbwm.dynamics.identification`).

Temporal index convention (load-bearing)
----------------------------------------
The dynamics are ``w_{t+1} = A w_t + B_p p_t + B_u u_t``, so **row ``t`` of the
forcing is the forcing that acts during the transition ``w_t -> w_{t+1}``**: the
water that arrives between acquisition ``d_t`` and acquisition ``d_{t+1}``. The
framework fixes this in three independent places -- Algorithm 3 pairs row ``t`` of
``Upsilon`` with the residual ``w_{t+1} - A w_t``; Algorithm 2 predicts with
``u_{t-1}^raw``; and Section 8.1 calls ``p_t`` the "known precipitation *forecast*"
used to predict ``f_{t+1}`` (a forecast is necessarily in the future relative to
``w_t``).

Accumulating *backwards* into ``(d_{t-1}, d_t]`` -- the rain already baked into
``w_t`` -- would regress every transition on the previous step's rain, shifting all
channels one step late and corrupting ``B_p`` / ``B_u``. Antecedent rain is not
discarded: it enters through the **lag channels**, which is what they are for.

Precipitation (uncontrollable exogenous disturbance)
---------------------------------------------------
Supplied as GeoTIFF rasters (CHIRPS / GPM IMERG / ERA5). Processing:

1. **Geometric alignment.** Each precipitation raster is warped onto the LST/NDVI
   reference grid (CRS + affine + shape) so it is pixel-by-pixel coincident with
   the target imagery -- see :mod:`dbwm.data.raster_align`.
2. **Temporal accumulation.** Satellite acquisitions are irregular. For step ``t``
   we accumulate *daily* rainfall (mm) over the forward half-open window
   ``(d_t, d_{t+1}]`` per the convention above. Long gaps are capped by
   ``max_accum_days`` so a data outage does not create an enormous spurious
   forcing spike.
3. **Spatial reduction.** The aligned, accumulated raster is reduced to
   ``n_precip_zones`` scalars: 1 -> a single AOI mean; >1 -> zonal means
   (``J_p`` zonal scalars).
4. **Wet-soil memory.** ``precip_lags`` lagged copies are appended as extra
   channels. Under the forward convention the lag-1 channel at step ``t`` is the
   rain of ``(d_{t-1}, d_t]`` -- the antecedent soil moisture already present when
   ``w_t`` was observed, which is exactly what LST responds to.

Irrigation (controllable actuator)
----------------------------------
Supplied as a time series (CSV: a date column + one or more magnitude columns --
applied depth mm, valve 0-1, or canal discharge). It is aggregated onto the *same*
forward windows as the precipitation ("sum" for accumulated depth, "mean" for a
rate), giving ``J_u`` scalars per step. Sharing the grid means ``u_t`` is both the
quantity Algorithm 3 identifies ``B_u`` from *and* the quantity the Algorithm 2
planner decides.

Scaling
-------
Channels are optionally divided by their training standard deviation. This is
**scale-only, never mean-centred**: centring would shift ``p_t = 0`` off zero and
destroy the "quiescent transition" (``p_t = u_t = 0``) detection that the default
two-stage identification (Algorithm 3, Regime B) depends on.
"""
from __future__ import annotations

import os
import re
import glob
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.config import ForcingConfig
from dbwm.data.raster_align import (
    ReferenceGrid,
    align_raster_to_grid,
    aoi_mean,
    zonal_means,
    make_zone_labels,
)
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.forcing")


# --------------------------------------------------------------------------- #
# Date parsing
# --------------------------------------------------------------------------- #
_DATE_RE = [
    re.compile(r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})"),  # YYYYMMDD / YYYY-MM-DD
]
_DOY_RE = re.compile(r"(\d{4})[-_.]?(\d{3})(?!\d)")  # YYYYDDD (day-of-year)


def parse_date(name: str) -> Optional[dt.date]:
    """
    Extract an acquisition date from a filename.

    Supports ``YYYYMMDD``, ``YYYY-MM-DD``, ``YYYY_MM_DD`` and ``YYYYDDD``
    (year + day-of-year) patterns.

    :param name: file name or path.
    :return: a :class:`datetime.date`, or ``None`` if no date is found.
    """
    base = os.path.basename(name)
    for rx in _DATE_RE:
        m = rx.search(base)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return dt.date(y, mo, d)
            except ValueError:
                pass
    m = _DOY_RE.search(base)
    if m:
        y, doy = int(m.group(1)), int(m.group(2))
        try:
            return dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
        except ValueError:
            return None
    return None


def discover_dated_rasters(directory: str) -> List[Tuple[dt.date, str]]:
    """
    Find all dated GeoTIFFs in a directory, sorted chronologically.

    :param directory: folder containing rasters.
    :return: list of ``(date, path)`` sorted by date. Undated files are skipped.
    """
    paths = sorted(
        glob.glob(os.path.join(directory, "*.tif"))
        + glob.glob(os.path.join(directory, "*.tiff"))
    )
    out = []
    for p in paths:
        d = parse_date(p)
        if d is None:
            logger.warning("Skipping undated raster: %s", os.path.basename(p))
            continue
        out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


# --------------------------------------------------------------------------- #
# Precipitation
# --------------------------------------------------------------------------- #
def daily_precip_zonal_series(
    precip_dir: str,
    grid: ReferenceGrid,
    cfg: ForcingConfig,
    zones: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[dt.date, np.ndarray]:
    """
    Align every daily precipitation raster onto the reference grid and reduce it
    to zonal (or AOI-mean) scalars.

    Each raster is warped once and immediately reduced, so the full aligned
    30 m stack is **never** held in memory -- only ``J_p`` floats per day.

    :param precip_dir: directory of daily precipitation GeoTIFFs.
    :param grid: LST/NDVI :class:`ReferenceGrid` to align onto.
    :param cfg: forcing configuration.
    :param zones: ``(H, W)`` zone labels, or ``None`` to build a default grid of
                  ``cfg.n_precip_zones`` contiguous blocks.
    :param valid_mask: ``(H, W)`` AOI validity mask (used when ``n_precip_zones``
                       is 1).
    :return: mapping ``date -> (J_p,)`` daily precipitation in mm.
    """
    j_p = cfg.n_precip_zones
    if j_p > 1 and zones is None:
        zones = make_zone_labels(grid.height, grid.width, j_p)

    dated = discover_dated_rasters(precip_dir)
    if not dated:
        raise FileNotFoundError(
            "No dated precipitation GeoTIFFs found in {}.".format(precip_dir)
        )
    logger.info("Aligning %d daily precipitation rasters to the reference grid.", len(dated))

    series: Dict[dt.date, np.ndarray] = {}
    for date, path in dated:
        arr = align_raster_to_grid(path, grid, resampling=cfg.precip_resampling)
        # Rain is non-negative; warping can introduce tiny negative undershoot.
        arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
        if j_p == 1:
            series[date] = np.array([aoi_mean(arr, valid_mask)], dtype=np.float32)
        else:
            series[date] = zonal_means(arr, zones, j_p)
    return series


def accumulation_windows(
    dates: Sequence[dt.date], max_accum_days: int = 32
) -> List[Tuple[Optional[dt.date], Optional[dt.date]]]:
    """
    The half-open accumulation window ``(d_t, d_{t+1}]`` for every step ``t``.

    **Index convention (load-bearing -- see** :func:`accumulate_between_acquisitions` **).**
    Row ``t`` is the window the forcing ``u_t^raw`` *drives*: the interval from
    acquisition ``d_t`` to acquisition ``d_{t+1}``. The final row has no successor
    acquisition and is therefore ``(None, None)``.

    :param dates: acquisition dates (chronological), length ``T``.
    :param max_accum_days: cap on the window length in days (guards long gaps).
    :return: length-``T`` list of ``(start, end)``; the window is ``(start, end]``.
    """
    t = len(dates)
    windows: List[Tuple[Optional[dt.date], Optional[dt.date]]] = []
    for i in range(t):
        if i == t - 1:
            windows.append((None, None))  # no successor -> forcing undefined
            continue
        start, end = dates[i], dates[i + 1]
        gap = min(max((end - start).days, 1), max_accum_days)
        # Truncate over-long gaps by pulling the *start* forward, so the window
        # always ends at the acquisition it forecasts into.
        windows.append((end - dt.timedelta(days=gap), end))
    return windows


def accumulate_between_acquisitions(
    daily: Dict[dt.date, np.ndarray],
    dates: Sequence[dt.date],
    n_zones: int,
    max_accum_days: int = 32,
) -> np.ndarray:
    """
    Accumulate daily precipitation onto the acquisition grid, using the **forward**
    window ``(d_t, d_{t+1}]``.

    Index convention
    ----------------
    The dynamics are ``w_{t+1} = A w_t + B_p p_t + B_u u_t`` (Section 2.2), so the
    forcing carrying index ``t`` is the forcing that acts *during the transition*
    ``w_t -> w_{t+1}`` -- i.e. the rain that falls **between acquisition ``d_t`` and
    acquisition ``d_{t+1}``**. Section 8.1 makes this explicit by calling ``p_t``
    the "known precipitation *forecast*" used to predict ``f_{t+1}``: it is in the
    future relative to the state ``w_t``, and Algorithm 3 pairs row ``t`` of
    ``Upsilon`` with the residual ``w_{t+1} - A w_t``.

    Accumulating *backwards* (``(d_{t-1}, d_t]``, the rain already reflected in
    ``w_t``) would regress each transition on the *previous* step's rain, shifting
    every channel one step late and corrupting ``B_p`` / ``B_u``.

    The last row has no successor acquisition, so its forcing is undefined and is
    returned as zeros; Algorithm 3 only ever uses rows ``0 .. T-2``
    (``Upsilon = [u_1^raw, ..., u_{T-1}^raw]``), so it is never a regressor.

    Antecedent (already-fallen) rain is *not* lost -- it enters through the lag
    channels of :func:`add_lag_channels`, where lag ``k`` at step ``t`` is the rain
    of window ``(d_{t-k}, d_{t-k+1}]``. Lag 1 is exactly the wet-soil memory
    present at the moment ``w_t`` was observed.

    :param daily: mapping ``date -> (J_p,)`` daily precipitation in mm.
    :param dates: the acquisition dates (chronological), length ``T``.
    :param n_zones: ``J_p``.
    :param max_accum_days: cap on the accumulation window (guards long data gaps).
    :return: ``(T, J_p)`` precipitation in mm; row ``t`` covers ``(d_t, d_{t+1}]``
             and the final row is zero.
    """
    t = len(dates)
    out = np.zeros((t, n_zones), dtype=np.float32)
    for i, (start, end) in enumerate(accumulation_windows(dates, max_accum_days)):
        if start is None or end is None:
            continue  # final step: no successor acquisition
        total = np.zeros(n_zones, dtype=np.float32)
        day = start + dt.timedelta(days=1)  # half-open: (start, end]
        while day <= end:
            if day in daily:
                total += daily[day]
            day += dt.timedelta(days=1)
        out[i] = total
    return out


def add_lag_channels(series: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Append lagged copies of a forcing series (wet-soil memory).

    Lag ``k`` at step ``t`` is ``series[t - k]``, zero-padded at the start.

    Under the forward-window convention of :func:`accumulate_between_acquisitions`
    (row ``t`` = rain over ``(d_t, d_{t+1}]``), the lag-1 channel at step ``t`` is
    the rain of window ``(d_{t-1}, d_t]`` -- precisely the antecedent soil moisture
    *already present* when ``w_t`` was observed. That is the wet-soil memory the
    thesis spec asks for: LST responds to how wet the ground already is, not only
    to rain falling during the step.

    :param series: ``(T, J)`` forcing series.
    :param n_lags: number of lags to append (0 -> unchanged).
    :return: ``(T, J * (1 + n_lags))`` array ``[s_t, s_{t-1}, ..., s_{t-n_lags}]``.
    """
    if n_lags <= 0:
        return series.astype(np.float32)
    chans = [series]
    for k in range(1, n_lags + 1):
        lagged = np.zeros_like(series)
        lagged[k:] = series[:-k]
        chans.append(lagged)
    return np.concatenate(chans, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Irrigation
# --------------------------------------------------------------------------- #
def load_irrigation_series(
    csv_path: str, dates: Sequence[dt.date], cfg: ForcingConfig
) -> np.ndarray:
    """
    Load the irrigation time series and aggregate it onto the acquisition grid.

    The CSV must contain a date column (``cfg.irrigation_date_col``) and one or
    more magnitude columns (``cfg.irrigation_value_cols``) -- applied depth in mm,
    a 0-1 valve indicator, or canal discharge.

    Rows are aggregated over the **same forward windows** ``(d_t, d_{t+1}]`` used
    for precipitation (:func:`accumulate_between_acquisitions`), so rain and
    irrigation share one temporal grid and one index convention: ``u_t`` is the
    water applied during the transition ``w_t -> w_{t+1}``. This is also exactly
    the quantity the planner *decides* (Algorithm 2: the plan ``u_{t:t+H}`` is a
    sequence of future applications), so the identification grid and the control
    grid coincide.

    The final row has no successor acquisition and is left at zero (never used as
    a regressor).

    :param csv_path: path to the irrigation CSV.
    :param dates: acquisition dates (chronological), length ``T``.
    :param cfg: forcing configuration.
    :return: ``(T, J_u)`` irrigation magnitudes; row ``t`` covers ``(d_t, d_{t+1}]``.
    """
    import pandas as pd

    j_u = len(cfg.irrigation_value_cols)
    t = len(dates)
    if not os.path.exists(csv_path):
        logger.warning(
            "Irrigation CSV %s not found: irrigation channel set to zero.", csv_path
        )
        return np.zeros((t, j_u), dtype=np.float32)

    df = pd.read_csv(csv_path)
    missing = [
        c
        for c in (cfg.irrigation_date_col,) + tuple(cfg.irrigation_value_cols)
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            "Irrigation CSV {} is missing column(s) {}. Found: {}".format(
                csv_path, missing, list(df.columns)
            )
        )
    df[cfg.irrigation_date_col] = pd.to_datetime(df[cfg.irrigation_date_col]).dt.date

    out = np.zeros((t, j_u), dtype=np.float32)
    for i, (start, end) in enumerate(accumulation_windows(dates, cfg.max_accum_days)):
        if start is None or end is None:
            continue  # final step: no successor acquisition
        sel = df[
            (df[cfg.irrigation_date_col] > start) & (df[cfg.irrigation_date_col] <= end)
        ]
        if sel.empty:
            continue
        vals = sel[list(cfg.irrigation_value_cols)].to_numpy(dtype=np.float32)
        out[i] = vals.sum(axis=0) if cfg.irrigation_aggregation == "sum" else vals.mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
IRRIGATION_PREFIX = "irrig"
PRECIP_PREFIX = "precip"


def channel_split(names: Sequence[str]) -> Tuple[List[int], List[int]]:
    """
    Split forcing channel names into precipitation and irrigation column indices.

    This is what lets ``B in R^{r x ell}`` be decomposed into ``B = [B_p  B_u]``
    *after* Algorithm 3 has learned it as one block: the identification fits every
    channel jointly, but the **control-theoretic roles differ**. ``B_p`` is an
    uncontrollable disturbance (it may not appear in the controllability Gramian,
    Proposition 4.3) and ``B_u`` is the actuator the planner is allowed to choose.

    :param names: length-``ell`` channel names (as built by :func:`build_forcing`).
    :return: ``(precip_cols, irrig_cols)`` index lists partitioning ``0..ell-1``.
    """
    precip = [j for j, nm in enumerate(names) if nm.startswith(PRECIP_PREFIX)]
    irrig = [j for j, nm in enumerate(names) if not nm.startswith(PRECIP_PREFIX)]
    return precip, irrig


@dataclass
class ForcingSeries:
    """
    The assembled forcing for one modality's acquisition grid.

    Row ``t`` is the forcing acting over ``(d_t, d_{t+1}]`` -- the transition
    ``w_t -> w_{t+1}`` (see :func:`accumulate_between_acquisitions`).

    :ivar values: ``(T, ell)`` raw forcing ``[p_t, p_{t-1}, ..., u_t]``.
    :ivar scale: ``(ell,)`` per-channel scale divisor applied (1.0 if unscaled).
    :ivar names: human-readable channel names, length ``ell``.
    :ivar dates: the acquisition dates, length ``T``.
    """

    values: np.ndarray
    scale: np.ndarray
    names: List[str]
    dates: List[dt.date]

    @property
    def ell(self) -> int:
        """Forcing dimension ``ell``."""
        return self.values.shape[1]

    @property
    def precip_cols(self) -> List[int]:
        """
        Column indices of the *precipitation* channels (including its lags).

        These are the columns of ``B`` that form ``B_p``: an **uncontrollable
        exogenous disturbance**. Proposition 4.3 excludes them from the
        controllability Gramian.
        """
        return channel_split(self.names)[0]

    @property
    def irrig_cols(self) -> List[int]:
        """
        Column indices of the *irrigation* channels -- the columns of ``B`` that
        form ``B_u``, the only genuinely **controllable** actuator.
        """
        return channel_split(self.names)[1]

    def physical(self) -> np.ndarray:
        """Return the forcing in original physical units (undo the scaling)."""
        return self.values * self.scale


def build_forcing(
    dates: Sequence[dt.date],
    cfg: ForcingConfig,
    grid: Optional[ReferenceGrid] = None,
    valid_mask: Optional[np.ndarray] = None,
    n_train: Optional[int] = None,
) -> ForcingSeries:
    """
    Build the full forcing series ``u_t^raw = [p_t, p_{t-1}, ..., u_t]`` for a set
    of acquisition dates.

    :param dates: acquisition dates (chronological), length ``T``.
    :param cfg: forcing configuration.
    :param grid: LST/NDVI reference grid (required if ``use_precip``).
    :param valid_mask: ``(H, W)`` AOI mask for the AOI-mean reduction.
    :param n_train: number of training steps used to compute the scaling stats
                    (avoids leaking test-period statistics). Defaults to all.
    :return: the :class:`ForcingSeries`.
    """
    t = len(dates)
    channels, names = [], []

    if cfg.use_precip:
        if grid is None:
            raise ValueError("A ReferenceGrid is required to build precipitation forcing.")
        daily = daily_precip_zonal_series(cfg.precip_dir, grid, cfg, valid_mask=valid_mask)
        p = accumulate_between_acquisitions(
            daily, dates, cfg.n_precip_zones, cfg.max_accum_days
        )  # (T, J_p)
        p = add_lag_channels(p, cfg.precip_lags)  # (T, J_p*(1+lags))
        channels.append(p)
        for lag in range(cfg.precip_lags + 1):
            for j in range(cfg.n_precip_zones):
                suffix = "" if lag == 0 else "_lag{}".format(lag)
                zone = "" if cfg.n_precip_zones == 1 else "_z{}".format(j)
                names.append("precip{}{}".format(zone, suffix))

    if cfg.use_irrigation:
        u = load_irrigation_series(cfg.irrigation_csv, dates, cfg)  # (T, J_u)
        channels.append(u)
        names.extend(list(cfg.irrigation_value_cols))

    if not channels:
        return ForcingSeries(
            values=np.zeros((t, 0), dtype=np.float32),
            scale=np.ones(0, dtype=np.float32),
            names=[],
            dates=list(dates),
        )

    values = np.concatenate(channels, axis=1).astype(np.float32)  # (T, ell)

    # Scale-only (no centring!) so p_t = 0 stays exactly 0 for Stage-I quiescence.
    scale = np.ones(values.shape[1], dtype=np.float32)
    if cfg.scale_forcing:
        n_fit = n_train if n_train is not None else t
        train_vals = values[:n_fit]
        s = train_vals.std(axis=0)
        scale = np.where(s > 1e-8, s, 1.0).astype(np.float32)
        values = (values / scale).astype(np.float32)

    fs = ForcingSeries(values=values, scale=scale, names=names, dates=list(dates))
    log_forcing_summary(fs)
    return fs


def log_forcing_summary(fs: ForcingSeries) -> Dict[str, float]:
    """
    Log diagnostics that matter for identifiability (Remark 6.1).

    In particular the fraction of **quiescent** steps (all channels zero), which
    is what Stage I of the two-stage identification fits ``A`` on -- if this is
    near 0 there are no quiescent transitions and ``joint`` identification should
    be used instead.

    :param fs: the forcing series.
    :return: dict of summary statistics.
    """
    if fs.ell == 0:
        logger.info("Forcing: none (pure-temporal, B_p = B_u = 0).")
        return {"quiescent_fraction": 1.0}
    active = np.abs(fs.values).sum(axis=1)
    quiescent = float((active <= 1e-8).mean())
    stats = {"quiescent_fraction": quiescent}
    logger.info(
        "Forcing: ell=%d channels %s | quiescent steps: %.1f%%",
        fs.ell, fs.names, 100.0 * quiescent,
    )
    for j, nm in enumerate(fs.names):
        col = fs.values[:, j]
        nz = float((np.abs(col) > 1e-8).mean())
        logger.info(
            "  %-16s nonzero %5.1f%% | mean %.3f | max %.3f (scaled units)",
            nm, 100.0 * nz, float(col.mean()), float(col.max()),
        )
    if quiescent < 0.05:
        logger.warning(
            "Very few quiescent steps (%.1f%%): two-stage Stage-I has little data. "
            "Consider DynamicsConfig.identification='joint'.", 100.0 * quiescent,
        )
    return stats


def save_forcing(fs: ForcingSeries, path: str) -> str:
    """
    Persist a forcing series to NPZ (so the expensive raster warp runs once).

    :param fs: the forcing series.
    :param path: output ``.npz`` path.
    :return: ``path``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        values=fs.values,
        scale=fs.scale,
        names=np.array(fs.names, dtype=object),
        dates=np.array([d.isoformat() for d in fs.dates], dtype=object),
    )
    logger.info("Saved forcing (%d x %d) to %s", fs.values.shape[0], fs.ell, path)
    return path


def load_forcing(path: str) -> ForcingSeries:
    """
    Load a forcing series previously written by :func:`save_forcing`.

    :param path: the ``.npz`` path.
    :return: the :class:`ForcingSeries`.
    """
    z = np.load(path, allow_pickle=True)
    return ForcingSeries(
        values=z["values"].astype(np.float32),
        scale=z["scale"].astype(np.float32),
        names=[str(x) for x in z["names"]],
        dates=[dt.date.fromisoformat(str(x)) for x in z["dates"]],
    )


def synthetic_forcing(
    dates: Sequence[dt.date], cfg: ForcingConfig, seed: int = 0
) -> ForcingSeries:
    """
    Generate sparse synthetic forcing (monsoon-like rain bursts + periodic
    irrigation) so the ``B_p`` / ``B_u`` identification path is exercised by
    tests and smoke runs without real rasters.

    Rain and irrigation are made **mutually exclusive** (a farmer does not
    irrigate while it rains), matching the real data property.

    :param dates: acquisition dates.
    :param cfg: forcing configuration.
    :param seed: RNG seed.
    :return: the :class:`ForcingSeries`.
    """
    rng = np.random.RandomState(seed)
    t = len(dates)
    j_p = cfg.n_precip_zones if cfg.use_precip else 0
    j_u = len(cfg.irrigation_value_cols) if cfg.use_irrigation else 0

    rain_days = rng.rand(t) < 0.25  # ~25% of steps have rain
    channels, names = [], []
    if cfg.use_precip:
        p = np.zeros((t, j_p), dtype=np.float32)
        p[rain_days] = rng.gamma(2.0, 6.0, size=(int(rain_days.sum()), j_p)).astype(np.float32)
        p = add_lag_channels(p, cfg.precip_lags)
        channels.append(p)
        for lag in range(cfg.precip_lags + 1):
            for j in range(j_p):
                suffix = "" if lag == 0 else "_lag{}".format(lag)
                zone = "" if j_p == 1 else "_z{}".format(j)
                names.append("precip{}{}".format(zone, suffix))
    if cfg.use_irrigation:
        u = np.zeros((t, j_u), dtype=np.float32)
        # Irrigate every 4th step, but never when it rained (mutual exclusivity).
        irrigate = (np.arange(t) % 4 == 0) & (~rain_days)
        u[irrigate] = rng.uniform(10.0, 40.0, size=(int(irrigate.sum()), j_u)).astype(np.float32)
        channels.append(u)
        names.extend(list(cfg.irrigation_value_cols))

    if not channels:
        return ForcingSeries(np.zeros((t, 0), np.float32), np.ones(0, np.float32), [], list(dates))

    values = np.concatenate(channels, axis=1).astype(np.float32)
    scale = np.ones(values.shape[1], dtype=np.float32)
    if cfg.scale_forcing:
        s = values.std(axis=0)
        scale = np.where(s > 1e-8, s, 1.0).astype(np.float32)
        values = (values / scale).astype(np.float32)
    return ForcingSeries(values=values, scale=scale, names=names, dates=list(dates))
