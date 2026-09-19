"""
Point weather from Open-Meteo, split into its two structurally distinct roles.

The AOI is a single ~3.75 x 4.05 km field (see the raster report) while the
ERA5/IFS grid the archive serves is 9-25 km, so the whole AOI sits inside one
grid cell and the record is a **single scalar per day per variable**, applied
uniformly across the field. This is registration, not downscaling.

Two roles, never mixed
----------------------
**Precipitation is an input.** It enters the dynamics through ``B_p`` in the
predict step (v2 Sec. 2.2), because rain *drives* the vegetation state::

    w_{t+1} = sum_j A_j w_{t-j} + B_p p_t + eta_t

**Rs / Ta / VPD are measurements.** They enter through the emission matrix ``C``
in the Kalman update, because they are observed quantities correlated with the
state rather than actuators of it::

    y_t^w = C w_t + d(doy_t) + nu_t,     nu_t ~ N(0, R),   C in R^{3 x r}

Keeping a channel out of ``Upsilon`` once it is in ``C`` is not bookkeeping: a
channel that is both a known input *and* a measurement makes the innovation
correlated with the input, which destroys the minimum-variance property of the
filter and with it the calibration claims of v3 Prop. 2.13.

Why precipitation stays raw but Rs/Ta/VPD get a climatology
-----------------------------------------------------------
Ta and Rs are near-perfect annual sinusoids. So is NDVI's dominant autonomous
Koopman mode. Feeding raw levels into a regressor block alongside the state is
exactly the collinearity failure of v2 Remark 6.1 -- the estimator cannot
attribute annual variance between ``A`` and ``B``, and the ridge, not the data,
decides. Carrying the day-of-year climatology ``d(doy)`` as a **known offset** in
the emission equation leaves the annual cycle with the autonomous operator (where
it physically belongs) and makes the innovation carry the *departure*, which is
the informative part.

Precipitation needs no such treatment: it is spiky and zero on most days, which is
also what keeps the quiescent set of Algorithm 3 Stage I non-empty.

Temporal index convention (load-bearing)
----------------------------------------
Since ``w_{t+1} = ... + B_p p_t``, row ``t`` of the forcing is the water arriving in
the **forward** window ``(d_t, d_{t+1}]`` -- the interval it drives. On the daily
grid that is ``precip_sum(d_{t+1})``. Antecedent rain (already baked into ``w_t``)
is not lost: it is exactly what the lag channels carry. Accumulating backwards
shifts every channel one step late and corrupts ``B_p`` silently.

Measurement rows carry no such shift: ``y_t^w`` is the weather observed on day
``t``, concurrent with ``w_t``.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.weather")

#: Canonical name of the input channel.
PRECIP_COL = "precip_mm"

#: Canonical measurement block. Order is fixed: it defines the row order of
#: ``C`` and ``R``.
DEFAULT_MEASUREMENT_COLS: Tuple[str, ...] = (
    "swrad_mj_m2",   # Rs / PAR proxy, daily shortwave sum (MJ m^-2)
    "ta_mean_c",     # 2 m air temperature, daily mean (deg C)
    "vpd_mean_kpa",  # vapour pressure deficit, daily mean (kPa)
)

#: Alternates, IF the source file provides them. The Sayedanwala archive supplies
#: only a single daily Ta and VPD, so these are absent there.
ALTERNATE_MEASUREMENT_COLS: Tuple[str, ...] = (
    "ta_max_c", "ta_min_c", "vpd_max_kpa",
)

#: Accepted spellings for each canonical channel.
#:
#: Two archives are in play and they do not agree on names: the Open-Meteo
#: collector emits ``precip_mm`` / ``ta_mean_c`` / ``swrad_mj_m2`` /
#: ``vpd_mean_kpa``, while the Sayedanwala historical file uses ``Precip_mm`` /
#: ``Ta_C`` / ``Sw_rad_mj_m2`` / ``VPD_kpa``. Resolving aliases here means the
#: rest of the pipeline -- and the config -- only ever sees canonical names, so a
#: new source file needs an entry in this table rather than changes anywhere else.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "precip_mm": (
        "precip_mm", "precipitation", "precipitation_sum", "precip", "p", "p_mm",
        "rain", "rain_mm", "prcp",
    ),
    "swrad_mj_m2": (
        "swrad_mj_m2", "sw_rad_mj_m2", "shortwave_radiation_sum", "shortwave_radiation",
        "swr", "rs", "sw_rad", "srad", "solar_radiation", "par",
    ),
    "ta_mean_c": (
        "ta_mean_c", "ta_c", "ta", "temperature_2m_mean", "temperature_2m",
        "air_temperature", "t2m", "tair", "temp_c",
    ),
    "vpd_mean_kpa": (
        "vpd_mean_kpa", "vpd_kpa", "vpd", "vapour_pressure_deficit_mean",
        "vapour_pressure_deficit", "vapor_pressure_deficit",
    ),
    "ta_max_c": ("ta_max_c", "temperature_2m_max", "tmax", "t_max_c"),
    "ta_min_c": ("ta_min_c", "temperature_2m_min", "tmin", "t_min_c"),
    "vpd_max_kpa": ("vpd_max_kpa", "vapour_pressure_deficit_max", "vpd_max"),
}

#: Accepted spellings for the date column.
DATE_ALIASES: Tuple[str, ...] = ("date", "time", "datetime", "day", "timestamp")

#: Date formats tried in order. Ambiguity between ``%m/%d/%Y`` and ``%d/%m/%Y``
#: is resolved by :func:`parse_date_column`, not by guessing.
DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d",
)

_DAYS_PER_YEAR = 365.25


def _normalise(name: str) -> str:
    """
    Reduce a column name to a comparison key: lowercase, alphanumerics only.

    :param name: raw column name.
    :return: normalised key.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_columns(columns: Sequence[str]) -> Dict[str, str]:
    """
    Map canonical channel names onto the actual column names in a file.

    :param columns: the file's column names.
    :return: ``{canonical: actual}`` for every canonical channel present.
    """
    lookup = {_normalise(c): c for c in columns}
    out: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            actual = lookup.get(_normalise(alias))
            if actual is not None:
                out[canonical] = actual
                break
    return out


def resolve_date_column(columns: Sequence[str]) -> str:
    """
    Find the date column by alias.

    :param columns: the file's column names.
    :return: the actual date column name.
    :raises ValueError: if no candidate is found.
    """
    lookup = {_normalise(c): c for c in columns}
    for alias in DATE_ALIASES:
        actual = lookup.get(_normalise(alias))
        if actual is not None:
            return actual
    raise ValueError(
        "No date column found among {}; expected one of {}.".format(
            list(columns), list(DATE_ALIASES)
        )
    )


def parse_date_column(raw) -> List[dt.date]:
    """
    Parse a date column, disambiguating ``%m/%d/%Y`` from ``%d/%m/%Y`` by evidence.

    ``1/2/2022`` is genuinely ambiguous, and guessing a locale would silently
    reorder a third of the record. Two pieces of evidence are used, in order:

    1. **Parseability.** A day-of-month above 12 is decisive: the Sayedanwala
       archive contains ``1/31/2022``, which only ``%m/%d/%Y`` can read, so the
       other reading is eliminated outright.
    2. **Monotonicity**, but only as a *tie-break* between formats that both parse
       every row -- the case where all days happen to be 12 or lower. A daily
       archive can only come out chronological under the correct reading.

    Monotonicity is deliberately not a hard requirement: a file that is merely
    unsorted is still perfectly usable, and :func:`read_weather_csv` sorts it.
    Rejecting it would confuse "ambiguous" with "out of order".

    :param raw: the raw date column (a pandas Series or sequence).
    :return: parsed dates, in file order (not sorted).
    :raises ValueError: if no format parses the column at all.
    """
    import pandas as pd

    candidates = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if not parsed.isna().any():
            candidates.append((fmt, parsed))

    if not candidates:
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.isna().any():
            raise ValueError(
                "Could not parse the date column with any of {} nor by inference. "
                "First few values: {}".format(list(DATE_FORMATS), list(raw)[:5])
            )
        logger.warning(
            "Date column parsed by inference rather than an explicit format; "
            "verify the day/month order is what you intend."
        )
        return [d.date() for d in parsed]

    if len(candidates) > 1:
        monotone = [c for c in candidates if c[1].is_monotonic_increasing]
        if len(monotone) == 1:
            chosen = monotone[0]
        else:
            chosen = candidates[0]
            others = [f for f, _ in candidates[1:]]
            logger.warning(
                "Date column is ambiguous: %s all parse every row and monotonicity "
                "does not separate them. Using %s. If the archive is not in that "
                "order, the day/month reading is wrong.",
                [f for f, _ in candidates], chosen[0],
            )
    else:
        chosen = candidates[0]
    return [d.date() for d in chosen[1]]


def _harmonic_design(dates: Sequence[dt.date], n_harmonics: int) -> np.ndarray:
    """
    Build the harmonic regression design matrix for a day-of-year climatology.

    Columns are ``[1, cos(2*pi*k*phase), sin(2*pi*k*phase)]`` for ``k = 1..K``,
    where ``phase = (doy - 1) / 365.25``. Using a fractional-year phase rather than
    an integer day index keeps leap and non-leap years on the same cycle.

    :param dates: dates to evaluate at.
    :param n_harmonics: number of annual harmonics ``K``.
    :return: ``(n, 1 + 2K)`` design matrix.
    """
    doy = np.asarray([d.timetuple().tm_yday for d in dates], dtype=np.float64)
    phase = 2.0 * np.pi * (doy - 1.0) / _DAYS_PER_YEAR
    cols = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * phase))
        cols.append(np.sin(k * phase))
    return np.stack(cols, axis=-1)


@dataclass
class Climatology:
    """
    Day-of-year harmonic climatology, fitted on the training split only.

    :ivar coeffs: ``(1 + 2K, m)`` regression coefficients, one column per channel.
    :ivar n_harmonics: number of annual harmonics ``K``.
    :ivar names: channel names, length ``m``.
    """

    coeffs: np.ndarray
    n_harmonics: int
    names: List[str]

    def predict(self, dates: Sequence[dt.date]) -> np.ndarray:
        """
        Evaluate the climatology at arbitrary dates.

        :param dates: dates to evaluate at.
        :return: ``(n, m)`` climatological values, in physical units.
        """
        return _harmonic_design(dates, self.n_harmonics) @ self.coeffs

    def amplitude(self) -> np.ndarray:
        """
        Peak-to-trough amplitude of the fundamental annual harmonic per channel.

        Reported as a sanity check: a channel whose annual amplitude dwarfs its
        residual std is one whose *raw* form would have been hopelessly collinear
        with the seasonal Koopman mode.

        :return: ``(m,)`` amplitudes ``2 * sqrt(a1^2 + b1^2)``.
        """
        a1, b1 = self.coeffs[1], self.coeffs[2]
        return 2.0 * np.sqrt(a1**2 + b1**2)


def fit_climatology(
    dates: Sequence[dt.date],
    values: np.ndarray,
    n_harmonics: int = 2,
    names: Optional[Sequence[str]] = None,
) -> Climatology:
    """
    Least-squares fit of an annual harmonic climatology.

    **Fit on the training split only** -- a climatology fitted on the full record
    leaks test-period seasonal structure into the training-time offset.

    :param dates: training dates.
    :param values: ``(n, m)`` channel values in physical units.
    :param n_harmonics: number of annual harmonics.
    :param names: channel names.
    :return: the fitted :class:`Climatology`.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[0] != len(dates):
        raise ValueError(
            "values has {} rows but {} dates were given".format(values.shape[0], len(dates))
        )
    design = _harmonic_design(dates, n_harmonics)
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return Climatology(
        coeffs=coeffs,
        n_harmonics=n_harmonics,
        names=list(names) if names is not None else [f"ch{i}" for i in range(values.shape[1])],
    )


# --------------------------------------------------------------------------- #
# Lag channels
# --------------------------------------------------------------------------- #
def add_lag_channels(series: np.ndarray, n_lags: int) -> Tuple[np.ndarray, List[str]]:
    """
    Append lagged copies of each column (wet-soil memory).

    Lag ``k`` at row ``t`` is row ``t - k``; rows before the start are zero-filled,
    which is the physically right choice for precipitation (unknown antecedent rain
    is treated as none) and is why the forcing must never be mean-centred.

    :param series: ``(n, j)`` base channels.
    :param n_lags: number of lag channels to append per base channel.
    :return: ``((n, j*(1+n_lags)) array, suffix names)``.
    """
    if n_lags <= 0:
        return series, [""]
    out, suffixes = [series], [""]
    for k in range(1, n_lags + 1):
        lagged = np.zeros_like(series)
        if k < series.shape[0]:
            lagged[k:] = series[:-k]
        out.append(lagged)
        suffixes.append("_lag{}".format(k))
    return np.concatenate(out, axis=-1), suffixes


# --------------------------------------------------------------------------- #
# The weather table
# --------------------------------------------------------------------------- #
@dataclass
class WeatherTable:
    """
    Weather aligned to a model time axis and split by role.

    :ivar dates: the model time axis (one entry per step).
    :ivar forcing: ``(n, ell)`` input block ``Upsilon`` rows -- precipitation and its
        lags, **scaled only** (never centred), forward-window aligned.
    :ivar forcing_names: length-``ell`` channel names.
    :ivar forcing_scale: ``(ell,)`` divisor applied to ``forcing``.
    :ivar measurement: ``(n, m)`` standardised **anomalies** ``y_t^w`` used by the
        Kalman update.
    :ivar measurement_names: length-``m`` channel names (row order of ``C``/``R``).
    :ivar measurement_raw: ``(n, m)`` physical-unit values.
    :ivar measurement_clim: ``(n, m)`` climatological offset ``d(doy_t)``, physical
        units.
    :ivar measurement_scale: ``(m,)`` divisor applied to the anomalies.
    :ivar climatology: the fitted :class:`Climatology`.
    :ivar forcing_valid: ``(n,)`` bool -- ``False`` for the final step, whose forward
        window has no successor observation.
    """

    dates: List[dt.date]
    forcing: np.ndarray
    forcing_names: List[str]
    forcing_scale: np.ndarray
    measurement: np.ndarray
    measurement_names: List[str]
    measurement_raw: np.ndarray
    measurement_clim: np.ndarray
    measurement_scale: np.ndarray
    climatology: Climatology
    forcing_valid: np.ndarray

    @property
    def ell(self) -> int:
        """Input dimension ``ell`` (precipitation + lags)."""
        return self.forcing.shape[1]

    @property
    def n_measurements(self) -> int:
        """Measurement dimension ``m`` (rows of ``C``)."""
        return self.measurement.shape[1]

    def physical_forcing(self) -> np.ndarray:
        """Undo the forcing scaling, recovering mm."""
        return self.forcing * self.forcing_scale

    def physical_measurement(self) -> np.ndarray:
        """Undo standardisation and re-add the climatology, recovering SI units."""
        return self.measurement * self.measurement_scale + self.measurement_clim


def read_weather_csv(path: str) -> Tuple[List[dt.date], Dict[str, np.ndarray]]:
    """
    Read a daily weather CSV and return it under **canonical** channel names.

    Column names are resolved through :data:`COLUMN_ALIASES` and the date column
    through :func:`parse_date_column`, so both the Open-Meteo collector output
    (``precip_mm``, ``ta_mean_c``, ...) and the Sayedanwala historical file
    (``Precip_mm``, ``Ta_C``, ``Sw_rad_mj_m2``, ``VPD_kpa``, ``Date`` in
    ``M/D/YYYY``) load through the same path.

    Rows are sorted by date. Unrecognised numeric columns are passed through under
    their original names rather than dropped, so nothing is silently lost.

    :param path: path to the CSV.
    :return: ``(dates, {canonical_or_original_name: values})``.
    :raises FileNotFoundError: if the file is absent.
    :raises ValueError: if no date column can be identified or parsed.
    """
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Weather CSV not found: {}. Point WeatherConfig.csv_path at the "
            "historical weather file.".format(path)
        )
    df = pd.read_csv(path)
    date_col = resolve_date_column(df.columns)
    df = df.copy()
    df["__parsed_date__"] = parse_date_column(df[date_col])
    df = df.sort_values("__parsed_date__").reset_index(drop=True)
    dates = list(df["__parsed_date__"])

    resolved = resolve_columns([c for c in df.columns if c != date_col])
    cols: Dict[str, np.ndarray] = {}
    for canonical, actual in resolved.items():
        cols[canonical] = pd.to_numeric(df[actual], errors="coerce").to_numpy(np.float64)
    renamed = set(resolved.values()) | {date_col, "__parsed_date__"}
    for c in df.columns:
        if c in renamed:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols[c] = df[c].to_numpy(np.float64)

    mapped = {k: v for k, v in resolved.items() if k != v}
    if mapped:
        logger.info(
            "Weather columns resolved to canonical names: %s",
            ", ".join(f"{v} -> {k}" for k, v in sorted(mapped.items())),
        )
    return dates, cols


def _align_to_axis(
    axis: Sequence[dt.date],
    src_dates: Sequence[dt.date],
    values: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Reindex source rows onto the model time axis **by date**.

    Positional joins are the silent-failure mode here: the raster stack has gaps
    while the weather record is daily-complete, so aligning by position shifts every
    channel by the number of preceding gaps.

    :param axis: model time axis.
    :param src_dates: dates of ``values``.
    :param values: ``(n_src, m)`` source values.
    :param label: name used in error messages.
    :return: ``(len(axis), m)`` reindexed values.
    :raises ValueError: if any axis date has no source row.
    """
    lookup = {d: i for i, d in enumerate(src_dates)}
    missing = [d for d in axis if d not in lookup]
    if missing:
        raise ValueError(
            "{}: {} axis dates absent from the weather record (e.g. {}). "
            "The archive must cover the full modelling window.".format(
                label, len(missing), missing[:5]
            )
        )
    idx = np.asarray([lookup[d] for d in axis], dtype=int)
    return values[idx]


def _forward_window_accumulate(
    axis: Sequence[dt.date],
    src_dates: Sequence[dt.date],
    daily: np.ndarray,
    max_accum_days: int = 32,
) -> np.ndarray:
    """
    Accumulate a daily flux over the forward window ``(d_t, d_{t+1}]``.

    Row ``t`` gets the total that arrives in the interval it *drives*. On a
    complete daily axis this reduces to "row ``t`` = the value on day ``t+1``";
    on a gapped axis it correctly **sums** the intervening days.

    Doing this from the *source* record rather than by shifting the axis-aligned
    series matters: shifting takes only the last day of a multi-day gap and drops
    the rest, which is silent (shapes match, ``B_p`` stays nonzero) and shows up
    only as an inexplicably weak precipitation response.

    :param axis: model time axis.
    :param src_dates: dates of the daily record.
    :param daily: ``(n_src,)`` daily values.
    :param max_accum_days: cap on the window length, guarding pathological gaps.
    :return: ``(len(axis),)`` accumulated values; the final row is 0 (no successor).
    """
    lookup = {d: float(v) for d, v in zip(src_dates, np.asarray(daily).reshape(-1))}
    out = np.zeros(len(axis), dtype=np.float64)
    for t in range(len(axis) - 1):
        start, end = axis[t], axis[t + 1]
        span = (end - start).days
        if span <= 0:
            continue
        first = end - dt.timedelta(days=min(span, max_accum_days) - 1)
        total, day = 0.0, first
        while day <= end:
            total += lookup.get(day, 0.0)
            day += dt.timedelta(days=1)
        out[t] = total
    return out


def _interpolate_nans(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """
    Linearly interpolate interior NaNs and edge-fill, reporting what was patched.

    :param values: ``(n, m)`` values.
    :param names: channel names for the log message.
    :return: ``(n, m)`` values with no NaNs.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    n = out.shape[0]
    grid = np.arange(n, dtype=np.float64)
    for j in range(out.shape[1]):
        bad = ~np.isfinite(out[:, j])
        if not bad.any():
            continue
        if bad.all():
            raise ValueError("Weather channel '{}' is entirely missing.".format(names[j]))
        logger.warning(
            "Weather channel '%s': %d/%d values missing -- linearly interpolated.",
            names[j], int(bad.sum()), n,
        )
        out[bad, j] = np.interp(grid[bad], grid[~bad], out[~bad, j])
    return out


def build_weather_table(
    axis: Sequence[dt.date],
    csv_path: str,
    train_mask: np.ndarray,
    precip_lags: int = 1,
    measurement_cols: Sequence[str] = DEFAULT_MEASUREMENT_COLS,
    n_harmonics: int = 2,
    scale_forcing: bool = True,
    max_accum_days: int = 32,
) -> WeatherTable:
    """
    Load the weather record and split it into the input and measurement blocks.

    All statistics -- the climatology, the forcing scale, the measurement scale --
    are estimated on the **training rows only**, so nothing about the test period
    reaches the model through normalisation.

    :param axis: model time axis (calendar-reindexed, one entry per step).
    :param csv_path: path to the historical weather CSV.
    :param train_mask: ``(n,)`` bool, ``True`` for training steps.
    :param precip_lags: number of lagged precipitation channels.
    :param measurement_cols: columns forming the measurement block, in the order
                             that fixes the rows of ``C`` and ``R``.
    :param n_harmonics: annual harmonics in the climatology.
    :param scale_forcing: divide forcing channels by their training std.
    :param max_accum_days: cap on the forward accumulation window, guarding
        pathological gaps in the frame calendar.
    :return: the assembled :class:`WeatherTable`.
    """
    src_dates, cols = read_weather_csv(csv_path)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape[0] != len(axis):
        raise ValueError(
            "train_mask has {} entries but the axis has {}".format(
                train_mask.shape[0], len(axis)
            )
        )
    if not train_mask.any():
        raise ValueError("train_mask selects no rows")

    missing_cols = [c for c in (PRECIP_COL,) + tuple(measurement_cols) if c not in cols]
    if missing_cols:
        raise ValueError(
            "Weather CSV {} is missing required columns {}. Available: {}".format(
                csv_path, missing_cols, sorted(cols)
            )
        )

    n = len(axis)

    # ---- Input block: precipitation, forward-window aligned --------------- #
    # Row t drives the transition t -> t+1, i.e. the window (d_t, d_{t+1}].
    # Accumulated from the SOURCE daily record so that a gapped axis sums the
    # intervening days instead of silently keeping only the last one.
    src_precip = _interpolate_nans(cols[PRECIP_COL][:, None], [PRECIP_COL])[:, 0]
    forward = _forward_window_accumulate(
        axis, src_dates, src_precip, max_accum_days
    )[:, None]
    forcing_valid = np.ones(n, dtype=bool)
    forcing_valid[-1] = False  # no successor observation to drive

    forcing, suffixes = add_lag_channels(forward, precip_lags)
    forcing_names = ["precip{}".format(s) for s in suffixes]

    if scale_forcing:
        # Scale ONLY, never centre: centring would move p_t = 0 off zero and
        # destroy the quiescent-transition detection Stage I relies on.
        scale = forcing[train_mask].std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
    else:
        scale = np.ones(forcing.shape[1])
    forcing = forcing / scale

    # ---- Measurement block: Rs / Ta / VPD as anomalies -------------------- #
    meas_names = list(measurement_cols)
    raw = np.stack([cols[c] for c in meas_names], axis=-1)
    raw = _align_to_axis(axis, src_dates, raw, "measurements")
    raw = _interpolate_nans(raw, meas_names)

    train_dates = [d for d, m in zip(axis, train_mask) if m]
    clim_model = fit_climatology(train_dates, raw[train_mask], n_harmonics, meas_names)
    clim = clim_model.predict(axis)

    anomaly = raw - clim
    meas_scale = anomaly[train_mask].std(axis=0)
    meas_scale = np.where(meas_scale > 1e-8, meas_scale, 1.0)

    table = WeatherTable(
        dates=list(axis),
        forcing=forcing.astype(np.float32),
        forcing_names=forcing_names,
        forcing_scale=scale.astype(np.float32),
        measurement=(anomaly / meas_scale).astype(np.float32),
        measurement_names=meas_names,
        measurement_raw=raw.astype(np.float32),
        measurement_clim=clim.astype(np.float32),
        measurement_scale=meas_scale.astype(np.float32),
        climatology=clim_model,
        forcing_valid=forcing_valid,
    )
    log_weather_summary(table, train_mask)
    return table


def weather_for_window(
    axis: Sequence[dt.date],
    csv_path: str,
    climatology: Climatology,
    forcing_scale: np.ndarray,
    measurement_scale: np.ndarray,
    precip_lags: int = 1,
    measurement_cols: Sequence[str] = DEFAULT_MEASUREMENT_COLS,
    max_accum_days: int = 32,
) -> WeatherTable:
    """
    Weather for a short operational window, using the **training** climatology.

    Two things separate this from :func:`build_weather_table` and both are
    required for an operational forecast rather than a retrospective one.

    **Nothing is estimated here.** The climatology and both scale vectors come
    from the checkpoint, so a window of a dozen days cannot refit an annual
    harmonic on itself. Refitting would be silently catastrophic: a two-week
    window fits an annual cycle to a fortnight of weather, and the resulting
    anomalies -- the quantity the Kalman update consumes -- would be arbitrary.

    **Missing future weather is tolerated, differently by role.** The horizon may
    run past the end of the archive, and the two weather roles then need opposite
    fallbacks:

    * a missing **measurement** (Rs/Ta/VPD) means *no correction is available*, so
      the row is left ``NaN`` and the observer skips that update. Substituting the
      climatology would set the anomaly to exactly zero, which is not "unknown" --
      it is the confident claim that the day is exactly average, and the filter
      would treat it as such;
    * a missing **input** (precipitation) means *assume no rain*, which is the
      only defensible default for a forcing that must enter the predict step and
      is zero on most days anyway. It is logged, because a forecast made under
      an assumed-dry horizon should be read as one.

    :param axis: the window's daily calendar.
    :param csv_path: path to the weather CSV.
    :param climatology: the training-fitted :class:`Climatology`.
    :param forcing_scale: ``(ell,)`` training forcing divisor.
    :param measurement_scale: ``(m,)`` training anomaly divisor.
    :param precip_lags: lagged precipitation channels, as at training time.
    :param measurement_cols: measurement channels, in the training row order.
    :param max_accum_days: cap on the forward accumulation window.
    :return: the assembled :class:`WeatherTable`, with ``NaN`` measurement rows
             wherever the record does not reach.
    """
    src_dates, cols = read_weather_csv(csv_path)
    meas_names = list(measurement_cols)
    missing_cols = [c for c in (PRECIP_COL,) + tuple(meas_names) if c not in cols]
    if missing_cols:
        raise ValueError(
            "Weather CSV {} is missing required columns {}. Available: {}".format(
                csv_path, missing_cols, sorted(cols)
            )
        )
    n = len(axis)

    # ---- Input block: precipitation, forward-window aligned ---------------- #
    src_precip = _interpolate_nans(cols[PRECIP_COL][:, None], [PRECIP_COL])[:, 0]
    forward = _forward_window_accumulate(
        axis, src_dates, src_precip, max_accum_days
    )[:, None]
    forcing, suffixes = add_lag_channels(forward, precip_lags)
    forcing_names = ["precip{}".format(s) for s in suffixes]
    scale = np.asarray(forcing_scale, dtype=np.float64).reshape(-1)
    if scale.shape[0] != forcing.shape[1]:
        raise ValueError(
            "forcing_scale has {} entries but the window built {} channels "
            "({}). The checkpoint's precip_lags must match.".format(
                scale.shape[0], forcing.shape[1], forcing_names
            )
        )
    forcing = forcing / scale

    # ---- Measurement block: anomalies against the TRAINING climatology ----- #
    lookup = {d: i for i, d in enumerate(src_dates)}
    raw = np.full((n, len(meas_names)), np.nan)
    have = np.zeros(n, dtype=bool)
    src = np.stack([cols[c] for c in meas_names], axis=-1)
    for t, d in enumerate(axis):
        i = lookup.get(d)
        if i is not None and np.all(np.isfinite(src[i])):
            raw[t] = src[i]
            have[t] = True

    clim = climatology.predict(list(axis))
    mscale = np.asarray(measurement_scale, dtype=np.float64).reshape(-1)
    anomaly = (raw - clim) / np.where(mscale > 1e-12, mscale, 1.0)

    n_dry = int(np.sum(~have))
    if n_dry:
        gaps = [str(d) for d, ok in zip(axis, have) if not ok]
        logger.warning(
            "Weather record covers %d of %d window dates. The %d uncovered "
            "date(s) %s carry NO measurement update (the observer skips them) "
            "and are forced with precipitation = 0, i.e. the forecast assumes a "
            "dry horizon there. Supply forecast weather to change that.",
            int(have.sum()), n, n_dry,
            gaps if n_dry <= 8 else gaps[:8] + ["..."],
        )

    forcing_valid = np.ones(n, dtype=bool)
    forcing_valid[-1] = False
    return WeatherTable(
        dates=list(axis),
        forcing=forcing.astype(np.float32),
        forcing_names=forcing_names,
        forcing_scale=scale.astype(np.float32),
        measurement=anomaly.astype(np.float32),
        measurement_names=meas_names,
        measurement_raw=raw.astype(np.float32),
        measurement_clim=clim.astype(np.float32),
        measurement_scale=mscale.astype(np.float32),
        climatology=climatology,
        forcing_valid=forcing_valid,
    )


def log_weather_summary(table: WeatherTable, train_mask: np.ndarray) -> Dict[str, float]:
    """
    Report the diagnostics that decide whether identification can succeed.

    Two numbers matter:

    * **Quiescent fraction** -- the share of training transitions with ``p_t = 0``.
      Algorithm 3 Stage I fits the autonomous operator on exactly these, so a low
      value means Stage I is starved and joint identification is the better regime.
    * **Seasonal amplitude / residual std** per measurement channel -- how much of
      each channel the climatology absorbed. A large ratio confirms that the raw
      channel would have been collinear with the annual Koopman mode.

    :param table: the assembled table.
    :param train_mask: ``(n,)`` training mask.
    :return: dict of the reported scalars.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    usable = train_mask & table.forcing_valid
    base = table.forcing[usable, 0]
    quiescent = float(np.mean(np.abs(base) <= 1e-8)) if base.size else 0.0

    amp = table.climatology.amplitude()
    resid_std = table.measurement[train_mask].std(axis=0) * table.measurement_scale
    ratios = amp / np.maximum(resid_std, 1e-8)

    logger.info(
        "Weather: ell=%d input channel(s) %s | %d measurement channel(s) %s",
        table.ell, table.forcing_names, table.n_measurements, table.measurement_names,
    )
    logger.info(
        "Quiescent (rain-free) training transitions: %.1f%% -- Stage I fits A on these.",
        100.0 * quiescent,
    )
    for j, nm in enumerate(table.measurement_names):
        logger.info(
            "  %-14s annual amplitude %.3f vs residual std %.3f (ratio %.1f)",
            nm, float(amp[j]), float(resid_std[j]), float(ratios[j]),
        )
    if quiescent < 0.25:
        logger.warning(
            "Quiescent fraction %.1f%% is low; two-stage identification has few "
            "rain-free transitions to fit A on. Consider identification='joint'.",
            100.0 * quiescent,
        )
    return {
        "quiescent_fraction": quiescent,
        **{"amp_ratio_" + nm: float(ratios[j]) for j, nm in enumerate(table.measurement_names)},
    }


def synthetic_weather(
    axis: Sequence[dt.date],
    train_mask: np.ndarray,
    precip_lags: int = 1,
    measurement_cols: Sequence[str] = DEFAULT_MEASUREMENT_COLS,
    seed: int = 0,
) -> WeatherTable:
    """
    Generate a physically plausible synthetic weather record for smoke runs/tests.

    Precipitation is sparse and gamma-distributed with a monsoon-weighted arrival
    rate (so quiescent transitions dominate, as in the real record); the measurement
    channels are annual sinusoids plus AR(1) noise, so the climatology fit has
    something real to remove.

    :param axis: model time axis.
    :param train_mask: ``(n,)`` training mask.
    :param precip_lags: lagged precipitation channels.
    :param measurement_cols: measurement channel names.
    :param seed: RNG seed.
    :return: a :class:`WeatherTable` built by the same code path as the real data.
    """
    rng = np.random.RandomState(seed)
    n = len(axis)
    doy = np.asarray([d.timetuple().tm_yday for d in axis], dtype=np.float64)
    phase = 2.0 * np.pi * (doy - 1.0) / _DAYS_PER_YEAR

    # Monsoon-weighted Bernoulli-Gamma precipitation.
    rate = 0.06 + 0.28 * np.clip(np.sin(phase - 1.9), 0.0, None) ** 2
    wet = rng.random_sample(n) < rate
    precip = np.where(wet, rng.gamma(shape=1.4, scale=6.0, size=n), 0.0)

    def _ar1(sigma: float) -> np.ndarray:
        e = rng.normal(0.0, sigma, size=n)
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = 0.65 * out[i - 1] + e[i]
        return out

    seasonal = {
        "swrad_mj_m2": 16.0 - 8.0 * np.cos(phase - 0.35),
        "ta_mean_c": 24.0 - 11.0 * np.cos(phase - 0.5),
        "ta_max_c": 31.0 - 11.0 * np.cos(phase - 0.5),
        "ta_min_c": 17.0 - 10.0 * np.cos(phase - 0.5),
        "vpd_mean_kpa": 1.2 - 0.8 * np.cos(phase - 0.6),
        "vpd_max_kpa": 2.4 - 1.5 * np.cos(phase - 0.6),
    }
    noise_scale = {
        "swrad_mj_m2": 1.6, "ta_mean_c": 1.8, "ta_max_c": 2.2,
        "ta_min_c": 1.7, "vpd_mean_kpa": 0.18, "vpd_max_kpa": 0.35,
    }

    src = {PRECIP_COL: precip}
    for c in measurement_cols:
        if c not in seasonal:
            raise ValueError("No synthetic generator for weather column '{}'".format(c))
        src[c] = seasonal[c] + _ar1(noise_scale[c])

    return _build_from_arrays(
        axis, src, train_mask, precip_lags, measurement_cols, n_harmonics=2
    )


def _build_from_arrays(
    axis: Sequence[dt.date],
    cols: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    precip_lags: int,
    measurement_cols: Sequence[str],
    n_harmonics: int,
) -> WeatherTable:
    """
    Assemble a :class:`WeatherTable` from in-memory arrays already on the axis.

    Shared by :func:`build_weather_table` (via CSV) and :func:`synthetic_weather`
    so both paths exercise identical alignment, scaling and climatology code.

    :param axis: model time axis.
    :param cols: ``{column: (n,) values}`` already aligned to ``axis``.
    :param train_mask: ``(n,)`` training mask.
    :param precip_lags: lagged precipitation channels.
    :param measurement_cols: measurement channel names.
    :param n_harmonics: annual harmonics.
    :return: the assembled table.
    """
    import tempfile

    # Round-trip through the CSV reader so the synthetic path cannot drift from
    # the real one (same alignment, NaN handling, scaling and climatology code).
    import csv as _csv

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = _csv.writer(fh)
            names = [PRECIP_COL] + [c for c in measurement_cols if c != PRECIP_COL]
            writer.writerow(["date"] + names)
            for i, d in enumerate(axis):
                writer.writerow([d.isoformat()] + [float(cols[c][i]) for c in names])
        return build_weather_table(
            axis, path, train_mask, precip_lags, measurement_cols, n_harmonics
        )
    finally:
        if os.path.exists(path):
            os.remove(path)
