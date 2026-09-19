"""
Tests for the point-weather loader and its role split.

Two things here are load-bearing and silent when wrong, so both are pinned:

* the **forward-window** alignment of the precipitation input, and
* the fact that all normalisation statistics come from the training split only.
"""
import csv
import datetime as dt
import os

import numpy as np
import pytest

from dbwm.data import weather as W
from dbwm.data.seasons import split_indices


def _axis(n=800, start=dt.date(2022, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _train_mask(axis, cut_frac=0.75):
    m = np.zeros(len(axis), dtype=bool)
    m[: int(len(axis) * cut_frac)] = True
    return m


def _write_csv(path, axis, cols):
    names = list(cols)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date"] + names)
        for i, d in enumerate(axis):
            w.writerow([d.isoformat()] + [float(cols[c][i]) for c in names])


def test_forward_window_alignment(tmp_path):
    """Row t of the forcing is the rain of day t+1 -- the window (d_t, d_{t+1}].

    Since ``w_{t+1} = ... + B_p p_t``, row ``t`` must carry the water that arrives
    in the interval it *drives*. Accumulating backwards shifts every channel one
    step late and corrupts ``B_p`` silently: shapes still match and ``B_p`` is still
    nonzero, so the only symptom is an inexplicably weak precipitation response.
    """
    axis = _axis(40)
    precip = np.zeros(40)
    precip[10] = 25.0  # rain falls on day 10
    cols = {W.PRECIP_COL: precip}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.linspace(1.0, 2.0, 40)
    path = str(tmp_path / "w.csv")
    _write_csv(path, axis, cols)

    tbl = W.build_weather_table(
        axis, path, _train_mask(axis), precip_lags=1, scale_forcing=False
    )
    base = tbl.forcing[:, 0]
    lag1 = tbl.forcing[:, 1]
    # Day 10's rain drives the transition 9 -> 10, so it sits in row 9.
    assert base[9] == pytest.approx(25.0)
    assert base[10] == pytest.approx(0.0)
    # The lag channel carries the rain already present when w_t was observed.
    assert lag1[10] == pytest.approx(25.0)


def test_final_row_forcing_is_marked_invalid(tmp_path):
    """The last step has no successor, so its forward window is undefined."""
    axis = _axis(30)
    cols = {W.PRECIP_COL: np.ones(30)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.ones(30)
    path = str(tmp_path / "w.csv")
    _write_csv(path, axis, cols)
    tbl = W.build_weather_table(axis, path, _train_mask(axis))
    assert tbl.forcing_valid[:-1].all()
    assert not tbl.forcing_valid[-1]


def test_join_is_by_date_not_position(tmp_path):
    """A weather record longer than the axis must still align correctly.

    The raster stack has gaps while the weather record is daily-complete, so a
    positional join would shift every channel by the number of preceding gaps.
    """
    full = _axis(60)
    cols = {W.PRECIP_COL: np.arange(60, dtype=float)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.arange(60, dtype=float)
    path = str(tmp_path / "w.csv")
    _write_csv(path, full, cols)
    axis = full[:40]
    tbl = W.build_weather_table(
        axis, path, _train_mask(axis), precip_lags=0, scale_forcing=False
    )
    # Measurements are concurrent with the state: row t carries day t's value.
    i = axis.index(full[12])
    assert tbl.measurement_raw[i, 0] == pytest.approx(12.0)
    # On a complete daily axis the forward window is exactly the next day.
    assert tbl.forcing[i, 0] == pytest.approx(13.0)


def test_gapped_axis_accumulates_the_whole_forward_window(tmp_path):
    """With a gap, row t must SUM the window (d_t, d_{t+1}], not just its last day.

    Shifting the axis-aligned series instead of accumulating from the source record
    keeps only the final day of a multi-day gap and silently discards the rest --
    shapes still match and ``B_p`` is still nonzero, so the only symptom would be an
    unexplained weakening of the precipitation response. This is why the v4 pipeline
    also reindexes the frame stack onto a complete daily calendar.
    """
    full = _axis(60)
    cols = {W.PRECIP_COL: np.arange(60, dtype=float)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.arange(60, dtype=float)
    path = str(tmp_path / "w.csv")
    _write_csv(path, full, cols)

    # Axis skips days 5, 6 and 7 -- exactly what the 13 missing NDVI frames create
    # if the calendar is NOT reindexed.
    axis = [d for i, d in enumerate(full[:40]) if i not in (5, 6, 7)]
    tbl = W.build_weather_table(
        axis, path, _train_mask(axis), precip_lags=1, scale_forcing=False
    )
    i4 = axis.index(full[4])
    assert axis[i4 + 1] == full[8]
    # Window (day4, day8] = days 5+6+7+8 = 26, not 8 (last day) and not 5 (first).
    assert tbl.forcing[i4, 0] == pytest.approx(26.0)
    # The lag channel carries the previous window, so no rain is double-counted.
    assert tbl.forcing[i4 + 1, 1] == pytest.approx(26.0)


def test_missing_axis_date_raises(tmp_path):
    """Silently dropping an unmatched date would misalign the whole series."""
    cols = {W.PRECIP_COL: np.ones(10)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.ones(10)
    path = str(tmp_path / "w.csv")
    _write_csv(path, _axis(10), cols)
    axis = _axis(20)  # 10 dates beyond the record
    with pytest.raises(ValueError, match="absent from the weather record"):
        W.build_weather_table(axis, path, _train_mask(axis))


def test_forcing_is_scaled_but_never_centred(tmp_path):
    """Zero rain must stay exactly zero, or Stage I loses its quiescent set."""
    axis = _axis(200)
    rng = np.random.RandomState(0)
    precip = np.where(rng.random_sample(200) < 0.2, rng.gamma(2.0, 5.0, 200), 0.0)
    cols = {W.PRECIP_COL: precip}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = rng.randn(200) + 10.0
    path = str(tmp_path / "w.csv")
    _write_csv(path, axis, cols)
    tbl = W.build_weather_table(axis, path, _train_mask(axis), scale_forcing=True)
    dry = precip[1:] == 0.0
    assert np.all(tbl.forcing[:-1][dry, 0] == 0.0)
    assert tbl.forcing[:, 0].std() > 0


def test_climatology_removes_the_annual_cycle():
    """Ta/Rs anomalies must be far smaller than the raw seasonal swing.

    This is the whole point of carrying d(doy) as a known offset: the annual cycle
    stays with the autonomous operator instead of making the measurement channel
    collinear with the seasonal Koopman mode (v2 Remark 6.1).
    """
    axis = _axis(1200)
    mask = _train_mask(axis)
    tbl = W.synthetic_weather(axis, mask)
    raw_std = tbl.measurement_raw.std(axis=0)
    anom_std = (tbl.measurement * tbl.measurement_scale).std(axis=0)
    assert np.all(anom_std < 0.5 * raw_std)
    amp = tbl.climatology.amplitude()
    assert np.all(amp / anom_std > 2.0)


def test_climatology_is_fitted_on_training_rows_only():
    """A climatology fitted on everything would leak test-period seasonality."""
    axis = _axis(1200)
    mask = _train_mask(axis)
    tbl = W.synthetic_weather(axis, mask)
    # Standardised training anomalies are exactly zero-mean / unit-variance;
    # the test rows are not, precisely because they had no say in the statistics.
    assert np.allclose(tbl.measurement[mask].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(tbl.measurement[mask].std(axis=0), 1.0, atol=1e-5)
    assert not np.allclose(tbl.measurement[~mask].mean(axis=0), 0.0, atol=1e-3)


def test_roles_are_disjoint():
    """Precipitation is an input; Rs/Ta/VPD are measurements. Never both."""
    axis = _axis(400)
    tbl = W.synthetic_weather(axis, _train_mask(axis))
    assert tbl.ell == 2
    assert tbl.n_measurements == 3
    assert not (set(tbl.forcing_names) & set(tbl.measurement_names))
    assert all(n.startswith("precip") for n in tbl.forcing_names)


def test_quiescent_fraction_is_high_enough_for_stage_one():
    """Rain-free transitions must dominate, or two-stage identification starves.

    With Rs/Ta/VPD moved into the emission, the only input is precipitation, which
    is zero on most days -- which is exactly what rescues Algorithm 3 Regime B.
    """
    axis = _axis(1200)
    mask = _train_mask(axis)
    tbl = W.synthetic_weather(axis, mask)
    stats = W.log_weather_summary(tbl, mask)
    assert stats["quiescent_fraction"] > 0.5


def test_nan_interpolation(tmp_path):
    """Interior gaps are interpolated; an entirely missing channel is an error."""
    axis = _axis(50)
    precip = np.ones(50)
    precip[10] = np.nan
    cols = {W.PRECIP_COL: precip}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.ones(50)
    path = str(tmp_path / "w.csv")
    _write_csv(path, axis, cols)
    tbl = W.build_weather_table(axis, path, _train_mask(axis), scale_forcing=False)
    assert np.all(np.isfinite(tbl.forcing))
    assert tbl.forcing[9, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Schema adaptation: the real Sayedanwala archive
# --------------------------------------------------------------------------- #
REAL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "sayedanwala_historical_weather_2022_2026.csv",
)


def test_column_aliases_resolve_both_schemas():
    """The collector schema and the Sayedanwala schema map to the same canonicals.

    Two archives are in play and they do not agree on names. Resolving aliases at
    the reader means config, tests and every downstream module only ever see
    canonical names.
    """
    collector = ["date", "precip_mm", "swrad_mj_m2", "ta_mean_c", "vpd_mean_kpa"]
    sayedanwala = ["Ta_C", "Precip_mm", "Sw_rad_mj_m2", "Date", "VPD_kpa"]
    a = W.resolve_columns(collector)
    b = W.resolve_columns(sayedanwala)
    assert set(a) == set(b) == {"precip_mm", "swrad_mj_m2", "ta_mean_c", "vpd_mean_kpa"}
    assert b["ta_mean_c"] == "Ta_C"
    assert b["vpd_mean_kpa"] == "VPD_kpa"
    assert b["swrad_mj_m2"] == "Sw_rad_mj_m2"


def test_date_column_resolved_by_alias():
    assert W.resolve_date_column(["Ta_C", "Date", "VPD_kpa"]) == "Date"
    assert W.resolve_date_column(["date", "x"]) == "date"
    with pytest.raises(ValueError, match="No date column found"):
        W.resolve_date_column(["Ta_C", "VPD_kpa"])


def test_ambiguous_dates_disambiguated_by_monotonicity():
    """1/2/2022 is genuinely ambiguous; guessing a locale would reorder the record.

    Restricting to day-of-month <= 12 makes BOTH M/D/Y and D/M/Y parseable, so the
    only thing that can separate them is which one comes out chronological.
    """
    import pandas as pd

    us = ["1/{}/2022".format(d) for d in range(1, 13)]  # Jan 1..12 under M/D/Y
    parsed = W.parse_date_column(pd.Series(us))
    assert parsed[0] == dt.date(2022, 1, 1)
    assert parsed[-1] == dt.date(2022, 1, 12)
    assert all(
        (parsed[i + 1] - parsed[i]).days == 1 for i in range(len(parsed) - 1)
    )

    # Same strings read as D/M/Y would be 1 Jan, 1 Feb, ... 1 Dec -- also monotone,
    # but M/D/Y is tried first and wins, and both agree the series is ordered.
    eu = ["{}/1/2022".format(d) for d in range(1, 13)]  # D/M/Y reading is 1 Jan..
    parsed_eu = W.parse_date_column(pd.Series(eu))
    assert parsed_eu[0] == dt.date(2022, 1, 1)


def test_unparsable_dates_are_rejected():
    import pandas as pd

    with pytest.raises(ValueError, match="Could not parse the date column"):
        W.parse_date_column(pd.Series(["not-a-date", "also-not", "nope"]))


def test_unsorted_but_valid_dates_are_accepted():
    """Merely unsorted is not ambiguous; read_weather_csv sorts afterwards."""
    import pandas as pd

    out = W.parse_date_column(pd.Series(["2022-01-05", "2022-01-01", "2022-01-09"]))
    assert out == [dt.date(2022, 1, 5), dt.date(2022, 1, 1), dt.date(2022, 1, 9)]


def test_rows_are_sorted_by_date(tmp_path):
    """A shuffled file must load in chronological order, not file order."""
    axis = _axis(30)
    cols = {W.PRECIP_COL: np.arange(30, dtype=float)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.arange(30, dtype=float)
    path = str(tmp_path / "w.csv")
    rows = list(range(30))
    np.random.RandomState(0).shuffle(rows)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        names = [W.PRECIP_COL] + list(W.DEFAULT_MEASUREMENT_COLS)
        wr.writerow(["date"] + names)
        for i in rows:
            wr.writerow([axis[i].isoformat()] + [float(cols[c][i]) for c in names])
    dates, out = W.read_weather_csv(path)
    assert dates == sorted(dates)
    assert out["ta_mean_c"][0] == pytest.approx(0.0)
    assert out["ta_mean_c"][-1] == pytest.approx(29.0)


def test_extra_source_rows_beyond_the_axis_are_allowed(tmp_path):
    """The archive runs to 2026-05-01 while the rasters stop at 2026-04-30.

    The extra row is not an error -- it supplies the forward window for the final
    axis date.
    """
    src = _axis(40)
    cols = {W.PRECIP_COL: np.ones(40)}
    for c in W.DEFAULT_MEASUREMENT_COLS:
        cols[c] = np.ones(40)
    path = str(tmp_path / "w.csv")
    _write_csv(path, src, cols)
    axis = src[:30]
    tbl = W.build_weather_table(
        axis, path, _train_mask(axis), precip_lags=0, scale_forcing=False
    )
    assert tbl.forcing.shape[0] == 30
    # Every interior window is well defined and picks up its successor day.
    assert np.all(tbl.physical_forcing()[:-1, 0] == pytest.approx(1.0))
    # The final row stays zero and flagged invalid. The forward window is defined
    # by CONSECUTIVE AXIS dates, so the last one has no successor regardless of how
    # far the source record extends -- and there is no w_{t+1} for it to drive.
    assert tbl.physical_forcing()[-1, 0] == pytest.approx(0.0)
    assert not tbl.forcing_valid[-1]


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real archive not present")
def test_real_sayedanwala_archive_loads_and_is_daily_complete():
    """End-to-end check against the actual archive file."""
    dates, cols = W.read_weather_csv(REAL_CSV)
    assert dates[0] == dt.date(2022, 1, 1)
    assert dates[-1] == dt.date(2026, 5, 1)
    assert len(dates) == 1582
    assert set(W.DEFAULT_MEASUREMENT_COLS) | {W.PRECIP_COL} <= set(cols)
    gaps = {(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)}
    assert gaps == {1}
    for name, arr in cols.items():
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"
    assert np.all(cols[W.PRECIP_COL] >= 0.0)


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real archive not present")
def test_real_archive_supports_two_stage_identification():
    """Precipitation must be zero often enough for Algorithm 3 Stage I to work.

    This is the property that moving Rs/Ta/VPD into the emission was meant to
    preserve; with always-on forcing the quiescent set would be empty.
    """
    axis = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(1581)]
    mask = np.array([d < dt.date(2025, 4, 15) for d in axis])
    tbl = W.build_weather_table(axis, REAL_CSV, mask)
    stats = W.log_weather_summary(tbl, mask)
    assert stats["quiescent_fraction"] > 0.5


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real archive not present")
def test_real_archive_climatology_removes_the_dominant_annual_cycle():
    """Ta is ~91% seasonal; raw levels would be collinear with the annual mode."""
    axis = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(1581)]
    mask = np.array([d < dt.date(2025, 4, 15) for d in axis])
    tbl = W.build_weather_table(axis, REAL_CSV, mask)
    amp = tbl.climatology.amplitude()
    resid = (tbl.measurement * tbl.measurement_scale).std(axis=0)
    ta = tbl.measurement_names.index("ta_mean_c")
    assert amp[ta] / resid[ta] > 5.0
    # Precipitation is NOT seasonally dominated, which is why it stays raw.
    p = tbl.physical_forcing()[:, 0]
    assert p.std() > 0 and float((p == 0).mean()) > 0.5
