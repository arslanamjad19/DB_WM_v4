"""
Tests for the operational forecast path: date in, ``t+1..t+H`` out.

Three pieces are new and each has a way of being silently wrong:

**The assimilating forecast loop.** It must record each step's forecast *before*
that step's own frame is used, or the reported "forecast skill" is measured
against data the model has already seen. That is a leak, and it looks like a
brilliant result.

**Operational weather.** A twelve-day window must not refit an annual harmonic
climatology on itself, and it must distinguish a missing measurement (no
correction available) from a missing input (assume no rain). Substituting the
climatology for a missing measurement sets the anomaly to exactly zero, which is
not "unknown" -- it is the confident claim that the day is exactly average.

**The figure/decode path.** The picture and the number must describe the same
forecast, which means one decode function with one subspace-lift convention.
"""
import datetime as dt

import numpy as np
import pytest

from dbwm.evaluation import forecast_figures, v4_plots
from dbwm.evaluation.geotiff_export import _blocking
from dbwm.inference import lifted_kalman as K
from tests.test_lifted_kalman import build_system


# --------------------------------------------------------------------------- #
# forecast_with_assimilation
# --------------------------------------------------------------------------- #
def test_assimilation_matches_plain_forecast_when_no_frames_arrive():
    """
    With nothing to fold in, the loop must reproduce the recursive forecast exactly.

    Any divergence would mean the operational path quietly runs a different model
    from the one ``infer_v4`` scores.
    """
    sysm, w, forcing, weather = build_system(n=800, order=3)
    state = np.tile(w[10], sysm.order)
    cov = np.eye(sysm.lifted_dim) * 1e-2
    f, y = forcing[10:16], weather[11:17]

    plain = K.forecast_from_origin(sysm, state, cov, 6, f, y, mode="recursive")
    got = K.forecast_with_assimilation(sysm, state, cov, 6, f, y)
    assert np.allclose(got["mean_forecast"], plain.mean, atol=1e-10)
    assert np.allclose(got["mean"], plain.mean, atol=1e-10)
    assert not got["assimilated"].any()


def test_assimilated_frames_improve_the_steps_that_follow():
    """
    A frame at t+2 must sharpen t+3..t+6, and must not touch t+1 or t+2's forecast.

    This is the operational claim: measurements arriving mid-horizon are worth
    using. It is also where a leak would hide, so the earlier steps are checked
    to be bit-identical.
    """
    sysm, w, forcing, weather = build_system(n=800, order=3)
    order, r = sysm.order, sysm.r
    observed = np.array([False, True, False, False, False, False])
    obs_cov = np.stack([np.eye(r) * 1e-6] * 6)
    plain_err, fed_err = [], []

    # Averaged over many origins, not judged on one: a single draw of the
    # process noise can favour either side, and the claim is about the mean.
    for t in range(order, 700, 17):
        # A CORRECT delay line: [w_t, w_{t-1}, ..., w_{t-L+1}]. Repeating w_t
        # would start every forecast from a state the model never produces.
        state = np.concatenate([w[t - j] for j in range(order)])
        cov = np.eye(sysm.lifted_dim) * 1e-2
        f, y = forcing[t : t + 6], weather[t + 1 : t + 7]
        truth = w[t + 1 : t + 7]

        plain = K.forecast_with_assimilation(sysm, state, cov, 6, f, y)
        fed = K.forecast_with_assimilation(
            sysm, state, cov, 6, f, y, w_obs=truth, observed=observed,
            obs_cov=obs_cov,
        )
        # Steps at or before the assimilated one must be bit-identical: this is
        # where a leak would hide.
        assert np.allclose(
            fed["mean_forecast"][:2], plain["mean_forecast"][:2], atol=1e-10
        )
        assert fed["assimilated"].tolist() == observed.tolist()
        plain_err.append(np.linalg.norm(plain["mean_forecast"][2:] - truth[2:]))
        fed_err.append(np.linalg.norm(fed["mean_forecast"][2:] - truth[2:]))

    assert np.mean(fed_err) < np.mean(plain_err)


def test_assimilation_records_the_forecast_before_using_the_frame():
    """
    At an assimilated step, ``mean_forecast`` must differ from ``mean``.

    If they were equal the loop would have written the corrected state into the
    forecast slot, and every reported metric at that step would be scoring the
    model on the frame it was just handed.
    """
    sysm, w, forcing, weather = build_system(n=600, order=3)
    t = 200
    state = np.tile(w[t], sysm.order)
    cov = np.eye(sysm.lifted_dim) * 1e-2
    truth = w[t + 1 : t + 4]
    got = K.forecast_with_assimilation(
        sysm, state, cov, 3, forcing[t : t + 3], weather[t + 1 : t + 4],
        w_obs=truth, observed=np.ones(3, dtype=bool),
        obs_cov=np.stack([np.eye(sysm.r) * 1e-6] * 3),
    )
    assert not np.allclose(got["mean_forecast"], got["mean"])
    # The corrected state is essentially the frame itself at this noise level.
    assert np.linalg.norm(got["mean"] - truth) < np.linalg.norm(
        got["mean_forecast"] - truth
    )
    assert (got["trace_reduction"] > 0).all()


# --------------------------------------------------------------------------- #
# Operational weather
# --------------------------------------------------------------------------- #
def _weather_csv(tmp_path, dates):
    """Write a minimal weather CSV covering ``dates``."""
    import csv

    path = tmp_path / "wx.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", "precip_mm", "swrad_mj_m2", "ta_mean_c", "vpd_mean_kpa"])
        for i, d in enumerate(dates):
            wr.writerow([d.isoformat(), 1.5 if i % 5 == 0 else 0.0,
                         16.0 + i * 0.1, 24.0 + i * 0.2, 1.2])
    return str(path)


def _climatology():
    """A two-harmonic climatology over the three measurement channels."""
    from dbwm.data.weather import fit_climatology

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    vals = np.stack([
        16.0 + 2 * np.sin(np.arange(400) / 58.0),
        24.0 + 5 * np.cos(np.arange(400) / 58.0),
        1.2 + 0.3 * np.sin(np.arange(400) / 58.0),
    ], axis=-1)
    return fit_climatology(dates, vals, 2,
                           ["swrad_mj_m2", "ta_mean_c", "vpd_mean_kpa"])


def test_window_weather_uses_the_supplied_climatology(tmp_path):
    """
    Nothing is estimated in the window: the climatology and scales come in.

    A twelve-day window fitting an annual harmonic on itself would produce
    arbitrary anomalies, and the anomaly is exactly what the Kalman update
    consumes.
    """
    from dbwm.data.weather import weather_for_window

    axis = [dt.date(2024, 6, 1) + dt.timedelta(days=i) for i in range(12)]
    csv = _weather_csv(tmp_path, axis)
    clim = _climatology()
    wt = weather_for_window(
        axis, csv, clim, forcing_scale=np.array([2.0, 2.0]),
        measurement_scale=np.array([1.0, 2.0, 0.5]), precip_lags=1,
    )
    assert wt.climatology is clim
    assert wt.measurement.shape == (12, 3)
    assert np.allclose(wt.measurement_scale, [1.0, 2.0, 0.5])
    # The anomaly is (raw - clim) / scale, computed with the GIVEN climatology.
    expected = (wt.measurement_raw - clim.predict(axis)) / np.array([1.0, 2.0, 0.5])
    assert np.allclose(wt.measurement, expected, atol=1e-4)


def test_window_weather_leaves_uncovered_dates_unusable(tmp_path):
    """
    Dates past the end of the record get NaN measurements and zero precipitation.

    The two roles need opposite fallbacks: a missing measurement means "no
    correction available" (the observer skips it), while a missing input means
    "assume no rain", which is the only defensible default for a forcing that
    must enter the predict step.
    """
    from dbwm.data.weather import weather_for_window

    axis = [dt.date(2024, 6, 1) + dt.timedelta(days=i) for i in range(10)]
    csv = _weather_csv(tmp_path, axis[:6])          # record stops early
    wt = weather_for_window(
        axis, csv, _climatology(), np.array([2.0, 2.0]), np.ones(3), precip_lags=1,
    )
    assert np.isfinite(wt.measurement[:6]).all()
    assert np.isnan(wt.measurement[6:]).all()
    # Precipitation for the uncovered tail is zero, not NaN: it must be usable.
    assert np.isfinite(wt.forcing).all()
    assert np.allclose(wt.forcing[6:, 0], 0.0)


def test_window_weather_rejects_a_mismatched_forcing_scale(tmp_path):
    """A checkpoint with different precip lags must fail loudly, not broadcast."""
    from dbwm.data.weather import weather_for_window

    axis = [dt.date(2024, 6, 1) + dt.timedelta(days=i) for i in range(8)]
    csv = _weather_csv(tmp_path, axis)
    with pytest.raises(ValueError, match="precip_lags"):
        weather_for_window(
            axis, csv, _climatology(), np.array([2.0, 2.0, 2.0]), np.ones(3),
            precip_lags=1,
        )


# --------------------------------------------------------------------------- #
# Decode / figure path
# --------------------------------------------------------------------------- #
class _Extractor:
    """Minimal stand-in for the GP state extractor."""

    def __init__(self, phi):
        self.phi = np.asarray(phi, dtype=float)
        self.sigma_eps2 = 1e-4

    def decode(self, w):
        """Linear decode with no offset."""
        return np.asarray(w) @ self.phi.T


class _DS:
    """Minimal stand-in for the calendar dataset."""

    def __init__(self, frames, dates, mask):
        self.frames = frames
        self.dates = dates
        self.static_mask = mask
        self.mask_is_static = True
        self.valid_mask = np.broadcast_to(mask, frames.shape)
        self.observed = np.ones(frames.shape[0], dtype=bool)
        self.mean = np.array([0.5])
        self.std = np.array([2.0])
        self.transform = (30.0, 0.0, 0.0, 0.0, -30.0, 0.0)
        self.crs = "EPSG:32643"
        self.pixel_size = 30.0

    @property
    def n_steps(self):
        return self.frames.shape[0]

    @property
    def shape(self):
        return self.frames.shape[1:3]


def _tiny_case(seed=0):
    """A 4x3 grid, 10 dates, rank-3 basis -- enough to exercise the decode path."""
    rng = np.random.default_rng(seed)
    h, w, r, t = 4, 3, 3, 10
    phi = rng.normal(size=(h * w, r))
    weights = rng.normal(size=(t, r))
    frames = (weights @ phi.T).reshape(t, h, w)
    mask = np.ones((h, w), dtype=bool)
    mask[0, 0] = False
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(t)]
    return _Extractor(phi), _DS(frames, dates, mask), weights


def test_decode_forecast_denormalises_and_reshapes():
    """The decoded map must be in physical units and frame-shaped."""
    ex, ds, w = _tiny_case()
    got = forecast_figures.decode_forecast(w[:2], ex, ds)
    assert got.shape == (2, *ds.shape)
    expected = (w[:2] @ ex.phi.T).reshape(2, *ds.shape) * 2.0 + 0.5
    assert np.allclose(got, expected)


def test_error_stack_masks_and_skips_unobserved_targets():
    """
    Invalid pixels are NaN and targets past the record are dropped, not scored.

    A step beyond the archive is a genuine forecast with no truth; scoring it
    against a zero-filled frame would silently invent a target.
    """
    ex, ds, w = _tiny_case()
    ds.observed[7] = False
    origins = np.array([2, 4, 6])
    fc = {"mean": np.stack([w[o + 1 : o + 3] for o in origins]),
          "valid": np.ones((3, 2), dtype=bool)}
    errs, org = forecast_figures.error_stack(fc, origins, ds, ex, lead=1)
    assert np.isnan(errs[:, 0, 0]).all()            # the masked pixel
    assert np.allclose(errs[:, 1, 1], 0.0, atol=1e-9)  # exact reconstruction
    errs2, org2 = forecast_figures.error_stack(fc, origins, ds, ex, lead=2)
    # origin 6 -> target 8 is fine; origin 4 -> target 6 fine; but with
    # ds.observed[7] False, origin 6 at lead 1 drops out.
    errs1, org1 = forecast_figures.error_stack(fc, origins, ds, ex, lead=1)
    assert 6 not in org1.tolist()
    assert len(org2) == 3


def test_render_horizon_figures_names_files_by_date(tmp_path):
    """
    Every triptych must be identifiable from its filename alone.

    A directory of ``*_t+3.png`` is unusable: the lead is only meaningful
    relative to an origin date.
    """
    ex, ds, w = _tiny_case()
    origins = np.array([3, 5])
    fc = {
        "mean": np.stack([w[o + 1 : o + 4] for o in origins]),
        "valid": np.ones((2, 3), dtype=bool),
        "cov_trace": np.full((2, 3), 0.01),
    }
    out = forecast_figures.render_horizon_figures(
        str(tmp_path), "exp", fc, origins, ds, ex, horizon=3,
        plot_origins=[5], per_pixel=True,
    )
    names = sorted(p.split("/")[-1] for p in out["figures"])
    assert "exp_triptych_2025-01-06_t+1.png" in names
    assert "exp_triptych_2025-01-06_t+3.png" in names
    assert "exp_pixel_error_t+2.png" in names
    # The per-lead summary names the worst pixel, which a field average cannot.
    assert set(out["pixel_metrics"]) == {1, 2, 3}
    assert "max_rmse_row" in out["pixel_metrics"][1]


def test_render_horizon_figures_defaults_to_a_fully_observed_origin(tmp_path):
    """
    With no origin requested, pick one whose whole horizon has data.

    A figure set with two blank panels reads as a broken model rather than as a
    missing acquisition.
    """
    ex, ds, w = _tiny_case()
    ds.observed[9] = False                      # would blank t+2 from origin 7
    origins = np.array([5, 7])
    fc = {
        "mean": np.stack([w[o + 1 : o + 3] for o in origins]),
        "valid": np.ones((2, 2), dtype=bool),
        "cov_trace": np.full((2, 2), 0.01),
    }
    out = forecast_figures.render_horizon_figures(
        str(tmp_path), "exp", fc, origins, ds, ex, horizon=2, per_pixel=False,
    )
    names = " ".join(out["figures"])
    assert "2025-01-06" in names          # origin 5, fully observed
    assert "2025-01-08" not in names      # origin 7 would hit the gap


def test_resolve_plot_dates_rejects_a_non_origin():
    """
    A date that was never scored must be refused, with the nearest ones named.

    Silently plotting nothing is the failure mode this replaces.
    """
    _, ds, _ = _tiny_case()
    origins = np.array([3, 5, 7])
    assert forecast_figures.resolve_plot_dates(["2025-01-06"], ds, origins) == [5]
    with pytest.raises(SystemExit, match="not a scored forecast origin"):
        forecast_figures.resolve_plot_dates(["2025-01-05"], ds, origins)
    with pytest.raises(SystemExit, match="outside the archive calendar"):
        forecast_figures.resolve_plot_dates(["2030-01-01"], ds, origins)


# --------------------------------------------------------------------------- #
# GeoTIFF blocking
# --------------------------------------------------------------------------- #
def test_geotiff_blocks_are_multiples_of_sixteen():
    """
    GDAL rejects tile dimensions that are not multiples of 16.

    The Sayedanwala grid is 135 x 125 and 125 is not one, so passing the raster
    width straight through -- as this used to -- raised ``RasterBlockError``
    before a single file was written.
    """
    got = _blocking(135, 125)
    assert got["tiled"] is True
    assert got["blockxsize"] % 16 == 0 and got["blockysize"] % 16 == 0
    assert got["blockxsize"] <= 125 and got["blockysize"] <= 135
    # Too small to tile at all: fall back to strips rather than fail.
    assert _blocking(10, 8) == {"tiled": False}


def test_small_raster_writes_without_tiling(tmp_path):
    """A raster below one tile must still be written."""
    from dbwm.evaluation.geotiff_export import write_forecast_geotiff

    rasterio = pytest.importorskip("rasterio")
    p = write_forecast_geotiff(
        str(tmp_path / "s.tif"), np.full((9, 7), 0.4), None, None,
        (30.0, 0.0, 0.0, 0.0, -30.0, 0.0), "EPSG:32643",
    )
    with rasterio.open(p) as src:
        assert src.shape == (9, 7)
