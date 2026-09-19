"""
Tests for the v4 figure set.

Plot code is easy to leave broken, because a figure that renders looks like a
figure that is right. These tests check the two things that actually matter: every
entry point produces a non-trivial file rather than raising, and the numbers that
reach the axes are the numbers that were passed in -- a silently mis-scaled or
NaN-swallowed curve is worse than no plot at all.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dbwm.evaluation import v4_plots  # noqa: E402


def _nonempty(path):
    """A rendered PNG, not a zero-byte stub."""
    return os.path.exists(path) and os.path.getsize(path) > 1000


def test_triptych_renders(tmp_path):
    """Truth / forecast / error render together for one lead time."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(16, 20))
    fc = truth + 0.1 * rng.normal(size=(16, 20))
    p = v4_plots.plot_forecast_triptych(
        truth, fc, str(tmp_path / "t.png"), lead=3
    )
    assert _nonempty(p)


def test_triptych_masks_invalid_pixels(tmp_path):
    """Masked pixels must not drive the colour scale."""
    truth = np.zeros((8, 8))
    truth[0, 0] = 1e6  # an absurd value, outside the field
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 0] = False
    p = v4_plots.plot_forecast_triptych(
        truth, truth.copy(), str(tmp_path / "m.png"), lead=1, valid_mask=mask
    )
    assert _nonempty(p)


def test_rmse_by_horizon_plots_every_variant(tmp_path):
    """Each named variant becomes its own labelled line."""
    metrics = {
        "recursive": {"pixel_rmse": np.array([0.05, 0.07, 0.09])},
        "direct": {"pixel_rmse": np.array([0.05, 0.06, 0.08])},
    }
    p = v4_plots.plot_rmse_by_horizon(metrics, str(tmp_path / "r.png"))
    assert _nonempty(p)


def test_rmse_by_horizon_draws_the_climatology_line(tmp_path):
    """
    The climatology reference must be drawable.

    Without it a reader cannot tell a good forecast from one that is merely
    plotted, since the axis auto-scales to whatever the model produced.
    """
    metrics = {"recursive": {"pixel_rmse": np.array([0.3, 0.31, 0.32])}}
    p = v4_plots.plot_rmse_by_horizon(
        metrics, str(tmp_path / "c.png"), climatology=0.29
    )
    assert _nonempty(p)


def test_rmse_by_horizon_tolerates_missing_horizons(tmp_path):
    """A horizon with no scorable origin is NaN and must not raise."""
    metrics = {"recursive": {"pixel_rmse": np.array([0.05, np.nan, 0.09])}}
    p = v4_plots.plot_rmse_by_horizon(metrics, str(tmp_path / "n.png"))
    assert _nonempty(p)


def test_koopman_spectrum_renders_complex_eigenvalues(tmp_path):
    """Conjugate pairs and the unit circle both need to draw."""
    eig = np.array([0.9 + 0.2j, 0.9 - 0.2j, 0.5 + 0j, -0.3 + 0j])
    p = v4_plots.plot_koopman_spectrum(eig, str(tmp_path / "k.png"))
    assert _nonempty(p)


def test_memory_depth_sweep_renders(tmp_path):
    """
    The ||D_h||-vs-L sweep is what ``L`` is selected from, so it must plot.

    Keys mirror ``memory_depth_sweep`` exactly: the defect is the *normalised* one
    (raw ``||D_h||/||S A^h||`` inflates with ``h`` for any dissipative field), and
    the second panel carries the Thm 2.8(i) certificate per order.
    """
    sweep = {
        "orders": np.array([1, 3, 5, 7]),
        "defect_normalized": np.array(
            [[0.00, 0.00], [0.02, 0.03], [0.02, 0.03], [0.05, 0.09]]
        ),
        "a_last_relative_smin": np.array([1.0, 0.4, 0.05, 1e-3]),
    }
    p = v4_plots.plot_memory_depth(sweep, str(tmp_path / "d.png"))
    assert _nonempty(p)


def test_transient_growth_renders(tmp_path):
    """The non-normality curve is the diagnostic for a bad forecast."""
    transient = {
        "rho": 0.99, "peak": 12.0, "peak_step": 6, "non_normality": 12.0,
        "powers": np.array([2.9, 5.0, 8.0, 12.0, 9.0, 6.0]),
    }
    p = v4_plots.plot_transient_growth(transient, str(tmp_path / "g.png"))
    assert _nonempty(p)


def test_ablation_bars_rank_by_metric(tmp_path):
    """Cells missing the metric are skipped rather than plotted as zero."""
    results = {
        "L1_s2": {"pixel_rmse_h1": 0.09},
        "L7_s2": {"pixel_rmse_h1": 0.07},
        "climatology": {"pixel_rmse_h1": 0.29},
        "broken": {"error": "singular"},
    }
    p = v4_plots.plot_ablation_bars(results, str(tmp_path / "a.png"))
    assert _nonempty(p)


def test_ablation_bars_with_no_usable_cells(tmp_path):
    """
    An all-failed grid must not crash the run that produced it.

    The grid catches per-cell exceptions so one singular fit cannot kill a
    multi-hour sweep; the plot has to survive the same situation, and returns an
    empty path rather than an empty figure that looks like a result.
    """
    p = v4_plots.plot_ablation_bars({"broken": {"error": "x"}}, str(tmp_path / "e.png"))
    assert p == ""
    assert not os.path.exists(str(tmp_path / "e.png"))


def test_ablation_bars_skips_non_finite_metrics(tmp_path):
    """A NaN cell must be dropped, not barred at zero as if it were the best."""
    results = {
        "good": {"pixel_rmse_h1": 0.09},
        "nan_cell": {"pixel_rmse_h1": float("nan")},
    }
    p = v4_plots.plot_ablation_bars(results, str(tmp_path / "f.png"))
    assert _nonempty(p)


def _spy_save(monkeypatch, sink):
    """
    Replace ``_save`` with a spy that records the figure before writing it.

    :param monkeypatch: pytest's monkeypatch fixture.
    :param sink: dict to record ``clims`` and ``title`` into.
    """
    original = v4_plots._save

    def spy(fig, path, dpi=150):
        sink.setdefault("clims", [])
        for ax in fig.axes:
            for im in ax.get_images():
                sink["clims"].append(im.get_clim())
        sink["title"] = fig._suptitle.get_text() if fig._suptitle else ""
        sink["dpi"] = dpi
        sink["n_axes"] = len([a for a in fig.axes if a.get_images()])
        return original(fig, path, dpi)

    monkeypatch.setattr(v4_plots, "_save", spy)


def test_triptych_uses_the_fixed_zero_to_one_ndvi_scale(tmp_path, monkeypatch):
    """
    Field panels must be pinned to [0, 1], not to [-1, 1] and not stretched.

    Fixed, because auto-scaling makes a t+1 and a t+6 map incomparable: each
    fills its own range, so a forecast that has drifted still looks fine on its
    own axes. [0, 1] rather than the physical [-1, 1], because this field lives
    in [0.1, 0.7] and on [-1, 1] every panel is one flat wash with the plot
    boundaries invisible.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    t = np.random.default_rng(0).uniform(0.2, 0.7, (16, 16))
    v4_plots.plot_forecast_triptych(t, t + 0.05, str(tmp_path / "s.png"), 1)
    assert sink["clims"][0] == (0.0, 1.0)
    assert sink["clims"][1] == (0.0, 1.0)
    # The error panel stays symmetric about zero rather than fixed.
    assert sink["clims"][2][0] == -sink["clims"][2][1]


def test_triptych_title_reports_ubrmse_and_mae(tmp_path, monkeypatch):
    """
    The headline must be ubRMSE and MAE, with RMSE/bias kept as context.

    RMSE alone cannot say whether two maps differ by a whole-field offset or by
    spatial structure, and those have different fixes.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    t = np.zeros((8, 8))
    v4_plots.plot_forecast_triptych(t, t + 0.1, str(tmp_path / "t.png"), 2)
    assert "ubRMSE" in sink["title"] and "MAE" in sink["title"]
    # A pure offset: ubRMSE must be 0 and MAE must equal the offset.
    assert "ubRMSE 0.0000" in sink["title"]
    assert "MAE 0.1000" in sink["title"]


def test_triptych_titles_carry_the_dates(tmp_path, monkeypatch):
    """
    The origin and target dates belong on the figure, not in the reader's head.

    A directory of ``*_t+3.png`` files is unusable without them: "t+3" is only
    meaningful relative to an origin.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    t = np.full((8, 8), 0.4)
    v4_plots.plot_forecast_triptych(
        t, t + 0.02, str(tmp_path / "d.png"), 3,
        origin_date="2025-06-14", target_date="2025-06-17",
    )
    assert "2025-06-14" in sink["title"] and "2025-06-17" in sink["title"]


def test_triptych_reports_the_extreme_pixels(tmp_path, monkeypatch):
    """
    The best- and worst-predicted pixels are named, with their coordinates.

    A field-average ubRMSE cannot distinguish one bad plot boundary from a
    diffuse miss, and those call for different work.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    truth = np.full((6, 7), 0.4)
    fc = truth.copy()
    fc[2, 5] += 0.25  # one deliberately terrible pixel
    v4_plots.plot_forecast_triptych(truth, fc, str(tmp_path / "e.png"), 1)
    assert "worst pixel (r2, c5) 0.2500" in sink["title"]
    # On a single date per-pixel RMSE and MAE are the same statistic, and the
    # figure says so rather than printing one number twice under two names.
    assert "single date" in sink["title"]


def test_triptych_adds_an_interval_panel_and_coverage(tmp_path, monkeypatch):
    """
    Given per-pixel sigma, a fourth panel appears and coverage is measured.

    Coverage beside the realised error is the only way to see whether the
    model's own claim about its accuracy holds spatially; per v3 Prop. 2.13 an
    iterated covariance under-covers whenever the semigroup defect is nonzero.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    truth = np.full((6, 6), 0.4)
    v4_plots.plot_forecast_triptych(
        truth, truth + 0.01, str(tmp_path / "c.png"), 1,
        sigma=np.full((6, 6), 0.05),
    )
    assert sink["n_axes"] == 4
    # 1.96 * 0.05 = 0.098 > 0.01, so every pixel is inside the interval.
    assert "empirical coverage 100.0%" in sink["title"]


def test_triptych_without_truth_drops_the_observed_and_error_panels(tmp_path, monkeypatch):
    """
    A genuine forward forecast has no truth and must not invent one.

    The target date may be past the end of the archive -- that is what a
    forecast *is* -- so the observed and error panels are omitted rather than
    drawn against a fabricated frame.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    p = v4_plots.plot_forecast_triptych(
        None, np.full((6, 6), 0.4), str(tmp_path / "f.png"), 6,
        sigma=np.full((6, 6), 0.05), target_date="2026-05-08",
    )
    assert _nonempty(p)
    assert sink["n_axes"] == 2          # forecast + interval only
    assert "ubRMSE" not in sink["title"]


def test_field_dpi_guarantees_visible_pixels(tmp_path, monkeypatch):
    """
    Resolution must scale with the array, or single-pixel structure is lost.

    Below roughly three device pixels per raster cell a one-pixel plot boundary
    is indistinguishable from a rendering artefact.
    """
    sink = {}
    _spy_save(monkeypatch, sink)
    big = np.full((400, 380), 0.4)
    v4_plots.plot_forecast_triptych(big, big + 0.01, str(tmp_path / "b.png"), 1)
    assert sink["dpi"] >= 4.0 * 400 / 4.0     # px_per_cell * rows / panel inches
    assert v4_plots._field_dpi((8, 8), 4.0) == 200.0   # small arrays hit the floor


def test_per_pixel_maps_separate_rmse_from_mae():
    """
    Pooled over origins, per-pixel RMSE and MAE are genuinely different maps.

    That is the whole reason to aggregate: on a single date both are |e|, so
    "which pixel has the worst RMSE" only becomes a distinct question once
    several origins are pooled. A pixel with one huge miss and many small ones
    must outrank a pixel with steady middling error on RMSE but not on MAE.
    """
    errs = np.zeros((10, 2, 1))
    errs[:, 0, 0] = 0.15              # steady: RMSE = MAE = 0.15
    errs[0, 1, 0] = 1.0               # one spike, otherwise perfect:
    maps = v4_plots.per_pixel_error_maps(errs)   # RMSE 0.316, MAE 0.10
    assert maps["rmse"][1, 0] > maps["rmse"][0, 0]     # spike wins on RMSE
    assert maps["mae"][1, 0] < maps["mae"][0, 0]       # but loses on MAE
    ext = v4_plots.pixel_extremes(maps["rmse"], maps["mae"])
    assert (ext["max_rmse"]["row"], ext["max_rmse"]["col"]) == (1, 0)
    assert (ext["max_mae"]["row"], ext["max_mae"]["col"]) == (0, 0)


def test_per_pixel_maps_ignore_masked_pixels():
    """A pixel that is NaN in every origin yields NaN, not zero."""
    errs = np.full((5, 3, 3), 0.05)
    errs[:, 0, 0] = np.nan
    maps = v4_plots.per_pixel_error_maps(errs)
    assert np.isnan(maps["rmse"][0, 0])
    assert maps["n"][0, 0] == 0
    assert np.isclose(maps["rmse"][1, 1], 0.05)


def test_pixel_error_maps_render(tmp_path):
    """The aggregated per-lead maps must produce a real figure."""
    errs = np.random.default_rng(0).normal(0, 0.04, (12, 20, 18))
    maps = v4_plots.per_pixel_error_maps(errs)
    p = v4_plots.plot_pixel_error_maps(
        maps, str(tmp_path / "pp.png"), lead=2, n_origins=12
    )
    assert _nonempty(p)


def test_value_range_is_fixed_per_registered_modality_and_derived_otherwise():
    """
    Each registered modality has its own fixed range; unknown labels derive one.

    This test previously asserted that LST fell back to robust percentiles of the
    data, on the grounds that "LST has no canonical range". That was wrong on the
    argument's own terms: the reason NDVI is pinned is that a ``t+1`` and a ``t+6``
    panel must share a colourbar, and a data-derived range gives each panel its
    own -- so a drifted forecast still looks fine on its own axes. LST needs the
    same property and now carries the archive's declared span, 283.15-320.15 K.

    Worse, in the call path that actually renders figures the range was requested
    with no ``data`` at all, so LST fell through to the NDVI default and every
    310 K field was drawn on ``[0, 1]``: one flat saturated wash.

    The percentile fallback remains for genuinely unregistered labels -- a
    residual field, an ablation series -- which do have no canonical range.
    """
    assert v4_plots.value_range("NDVI") == (0.0, 1.0)
    assert v4_plots.value_range("NDVI", data=np.array([300.0, 310.0])) == (0.0, 1.0)
    assert v4_plots.value_range("LST") == (283.15, 320.15)
    assert v4_plots.value_range(
        "LST", data=np.linspace(290.0, 320.0, 500)
    ) == (283.15, 320.15)
    lo, hi = v4_plots.value_range("residual", data=np.linspace(290.0, 320.0, 500))
    assert 289.0 < lo < 292.0 and 318.0 < hi < 321.0
    # An explicit override always wins.
    assert v4_plots.value_range("NDVI", vlim=(-1.0, 1.0)) == (-1.0, 1.0)
    assert v4_plots.value_range("LST", vlim=(-1.0, 1.0)) == (-1.0, 1.0)


def test_memory_ablation_renders_all_three_panels(tmp_path):
    """
    The L ablation needs skill, observability and defect together.

    Any one alone misleads: a score table cannot show that L = 7 has lost
    observability (Thm 2.8(i)), and the certificate cannot show that the extra
    lags bought nothing.
    """
    rows = [
        {"order": L, "pixel_ubrmse": np.linspace(0.05, 0.08, 6),
         "a_last_relative_smin": 10.0 ** -L,
         "defect_normalized": np.linspace(0.01, 0.05, 6), "blend": 0.5}
        for L in (2, 4, 6, 7)
    ]
    p = v4_plots.plot_memory_ablation(
        rows, str(tmp_path / "abl.png"),
        persistence=np.linspace(0.06, 0.09, 6), climatology=0.29,
    )
    assert _nonempty(p)


def test_memory_ablation_with_no_usable_rows(tmp_path):
    """A sweep where every order failed must not crash the run that produced it."""
    p = v4_plots.plot_memory_ablation(
        [{"order": 2, "error": "singular"}], str(tmp_path / "z.png")
    )
    assert p == ""
