"""
Tests for the per-season spatiotemporal statistics.

The separability test is validated the only way that means anything: on fields
whose separability is known by construction, checking both the false-positive rate
and the power.
"""
import datetime as dt

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from dbwm.data.seasons import KHARIF, RABI, ZAID
from dbwm.evaluation import spatiotemporal_stats as S


# --------------------------------------------------------------------------- #
# Synthetic fields with known covariance structure
# --------------------------------------------------------------------------- #
def separable_field(seed=0, t=200, h=36, w=36, rho=0.85, scale=3.0):
    """AR(1) in time with a FIXED spatial kernel: C(h,u) = C_s(h) rho^u exactly."""
    rng = np.random.RandomState(seed)
    out = np.zeros((t, h, w))
    prev = gaussian_filter(rng.randn(h, w), scale)
    for i in range(t):
        prev = rho * prev + np.sqrt(1 - rho**2) * gaussian_filter(rng.randn(h, w), scale)
        out[i] = prev
    return out + 0.6


def advecting_field(seed=0, t=200, h=36, w=36, v=1.5, scale=3.0):
    """A frozen field translating at v px/step: C(h,u) = C_F(h - v u). Non-separable."""
    rng = np.random.RandomState(seed)
    big = gaussian_filter(rng.randn(h + int(v * t) + 10, w), scale)
    return np.stack(
        [big[int(round(v * i)) : int(round(v * i)) + h, :w] for i in range(t)]
    ) + 0.6


# --------------------------------------------------------------------------- #
# Field statistics
# --------------------------------------------------------------------------- #
def test_mean_and_variance_recover_known_values():
    t, h, w = 120, 8, 8
    rng = np.random.RandomState(0)
    truth_mean = np.linspace(0.2, 0.8, h * w).reshape(h, w)
    frames = truth_mean[None] + rng.randn(t, h, w) * 0.1
    stats = S.field_statistics(frames, np.ones((t, h, w), dtype=bool))
    # SE of the mean is 0.1/sqrt(120) ~ 0.009, so allow ~5 sigma.
    assert np.allclose(stats.mean, truth_mean, atol=0.05)
    assert np.allclose(stats.variance, 0.01, atol=0.004)
    assert stats.n_frames == t


def test_cv_is_suppressed_where_the_mean_crosses_zero():
    """NDVI crosses zero, so an unguarded sigma/mu map is dominated by spikes."""
    t, h, w = 60, 4, 4
    rng = np.random.RandomState(1)
    means = np.zeros((h, w))
    means[0, 0] = 0.001   # essentially zero -> CV must be suppressed
    means[1, 1] = 0.7     # healthy vegetation -> CV must survive
    frames = means[None] + rng.randn(t, h, w) * 0.05
    stats = S.field_statistics(frames, np.ones((t, h, w), dtype=bool), min_abs_mean=0.05)
    assert np.isnan(stats.cv[0, 0])
    assert np.isfinite(stats.cv[1, 1])
    assert stats.cv_masked_fraction > 0


def test_robust_cv_is_reported_alongside():
    t, h, w = 80, 5, 5
    rng = np.random.RandomState(2)
    frames = 0.6 + rng.randn(t, h, w) * 0.1
    stats = S.field_statistics(frames, np.ones((t, h, w), dtype=bool))
    assert np.all(np.isfinite(stats.cv_robust))
    # IQR/median and sigma/mu measure the same thing on a Gaussian, up to ~1.35.
    assert 0.5 < np.nanmedian(stats.cv_robust) / np.nanmedian(stats.cv) < 2.5


def test_pixels_with_too_few_observations_are_excluded():
    t, h, w = 30, 4, 4
    frames = np.ones((t, h, w))
    mask = np.ones((t, h, w), dtype=bool)
    mask[:, 0, 0] = False
    mask[:2, 1, 1] = True
    mask[2:, 1, 1] = False
    stats = S.field_statistics(frames, mask, min_observations=3)
    assert not stats.valid[0, 0]
    assert not stats.valid[1, 1]
    assert np.isnan(stats.mean[0, 0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        S.field_statistics(np.zeros((4, 3, 3)), np.ones((5, 3, 3), dtype=bool))


# --------------------------------------------------------------------------- #
# Covariogram
# --------------------------------------------------------------------------- #
def test_covariance_decays_with_distance_and_lag():
    frames = separable_field(seed=3)
    mask = np.ones(frames.shape, dtype=bool)
    cg = S.spatiotemporal_covariance(frames, mask, "anomaly", 30.0, lags=(0, 1, 2, 3))
    assert np.all(np.diff(cg.spatial) < 1e-9)      # decays in space
    assert np.all(np.diff(cg.temporal) < 1e-9)     # decays in time
    assert cg.c0 > cg.spatial[0]
    assert cg.cov.shape == (cg.distances.size, cg.lags.size)


def test_centering_choice_is_mandatory_and_changes_the_answer():
    """Global vs anomaly centring measure different things; conflating them is the trap."""
    rng = np.random.RandomState(4)
    t, h, w = 80, 24, 24
    static = gaussian_filter(rng.randn(h, w), 4.0)
    static = 3.0 * static / static.std()  # a genuinely dominant permanent pattern
    frames = static[None] + rng.randn(t, h, w) * 0.2
    mask = np.ones((t, h, w), dtype=bool)

    with pytest.raises(ValueError, match="must be 'anomaly' or 'global'"):
        S.spatiotemporal_covariance(frames, mask, "whatever")

    glob = S.spatiotemporal_covariance(frames, mask, "global", 30.0, lags=(0, 1))
    anom = S.spatiotemporal_covariance(frames, mask, "anomaly", 30.0, lags=(0, 1))
    # The static pattern dominates the global covariance and is absent from the anomaly one.
    assert glob.c0 > 10 * anom.c0


def test_temporally_white_field_has_no_lagged_covariance():
    rng = np.random.RandomState(5)
    t, h, w = 100, 24, 24
    frames = np.stack([gaussian_filter(rng.randn(h, w), 3.0) for _ in range(t)])
    mask = np.ones((t, h, w), dtype=bool)
    cg = S.spatiotemporal_covariance(frames, mask, "anomaly", 30.0, lags=(0, 1, 2))
    assert cg.temporal[0] > 0
    assert abs(cg.temporal[1]) < 0.05 * cg.temporal[0]


def test_separability_ratio_is_near_one_for_a_separable_field():
    frames = separable_field(seed=6)
    mask = np.ones(frames.shape, dtype=bool)
    cg = S.spatiotemporal_covariance(
        frames, mask, "anomaly", 30.0,
        distance_bins=[40.0, 80.0, 130.0], lags=(0, 1, 2), n_pairs=4000,
    )
    finite = np.isfinite(cg.ratio)
    assert np.nanmedian(np.abs(cg.ratio[finite] - 1.0)) < 0.25


# --------------------------------------------------------------------------- #
# Separability test: calibration and power
# --------------------------------------------------------------------------- #
def test_separability_not_rejected_on_a_separable_field():
    """Type-I control. A separable field must usually survive the test."""
    n_reject = 0
    for seed in range(6):
        # A larger grid is not incidental: the bootstrap can only see spatial
        # sampling noise if there are enough distinct pixels for the sampled pairs
        # to be approximately independent.
        res = S.separability_test(
            separable_field(seed=seed, t=260, h=48, w=48),
            np.ones((260, 48, 48), dtype=bool),
            distances=(60.0, 150.0, 300.0), lags=(1, 2, 3),
            n_bootstrap=250, block_length=40, seed=seed,
        )
        n_reject += res.reject
    assert n_reject <= 2, f"over-rejecting a separable field: {n_reject}/6"


def test_separability_rejected_on_an_advecting_field():
    """Power. C(h,u) = C_F(h - v u) cannot factorise, and the test must see it."""
    n_reject = 0
    for seed in range(4):
        res = S.separability_test(
            advecting_field(seed=seed, t=260, h=48, w=48),
            np.ones((260, 48, 48), dtype=bool),
            distances=(60.0, 150.0, 300.0), lags=(1, 2, 3),
            n_bootstrap=250, block_length=40, seed=seed,
        )
        n_reject += res.reject
        assert "REJECTED" in res.verdict() or not res.reject
    assert n_reject >= 3, f"missing a clearly non-separable field: {n_reject}/4"


def test_white_field_is_flagged_as_trivially_separable(caplog):
    """A temporally white field factorises trivially; non-rejection means nothing."""
    rng = np.random.RandomState(7)
    t, h, w = 150, 30, 30
    frames = np.stack([gaussian_filter(rng.randn(h, w), 3.0) for _ in range(t)]) + 0.6
    with caplog.at_level("WARNING"):
        res = S.separability_test(
            frames, np.ones((t, h, w), dtype=bool),
            distances=(60.0, 150.0), lags=(1, 2), n_bootstrap=120, block_length=30,
        )
    assert abs(res.temporal_correlation) < 0.15
    assert any("TRIVIALLY separable" in rec.getMessage() for rec in caplog.records)


def test_bootstrap_resamples_pairs_not_just_time():
    """Holding pixel pairs fixed hides spatial sampling noise and over-rejects.

    Regression test for exactly that defect: the precomputed products must retain a
    pair-group axis for the bootstrap to resample.
    """
    frames = separable_field(seed=8, t=120)
    pre = S._lagged_product_series(
        frames, np.ones(frames.shape, dtype=bool), "anomaly", 30.0,
        (60.0, 150.0), (0, 1, 2), 2000, 0,
    )
    assert pre["pair"].ndim == 4          # (n_h, n_u, T, n_groups)
    assert pre["pair"].shape[3] > 1
    a = S._contrasts_from_blocks(pre, [(0, 120)], (1, 2), np.array([0, 0, 1]))
    b = S._contrasts_from_blocks(pre, [(0, 120)], (1, 2), np.array([2, 3, 4]))
    assert not np.allclose(a, b)          # different pair groups -> different estimates


# --------------------------------------------------------------------------- #
# Per-season driver
# --------------------------------------------------------------------------- #
def test_season_reports_cover_every_season():
    t, h, w = 730, 20, 20
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(t)]
    frames = separable_field(seed=9, t=t, h=h, w=w)
    mask = np.ones((t, h, w), dtype=bool)
    reps = S.season_reports(
        frames, mask, dates, "train", run_separability=False, lags=(0, 1, 2)
    )
    assert set(reps) == {KHARIF, RABI, ZAID}
    for season, rep in reps.items():
        assert rep.split == "train"
        assert rep.n_frames > 100
        assert "anomaly" in rep.covariograms and "global" in rep.covariograms
        assert np.isfinite(rep.statistics.summary()["mean_of_mean"])


def test_season_with_too_few_frames_is_skipped(caplog):
    t, h, w = 40, 12, 12
    dates = [dt.date(2022, 6, 1) + dt.timedelta(days=i) for i in range(t)]
    frames = separable_field(seed=10, t=t, h=h, w=w)
    with caplog.at_level("WARNING"):
        reps = S.season_reports(
            frames, np.ones((t, h, w), dtype=bool), dates, "test",
            run_separability=False, min_frames=100,
        )
    assert reps == {}
    assert any("skipping" in rec.message for rec in caplog.records)


def test_format_season_table_renders():
    t, h, w = 400, 16, 16
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(t)]
    frames = separable_field(seed=11, t=t, h=h, w=w)
    reps = S.season_reports(
        frames, np.ones((t, h, w), dtype=bool), dates, "train",
        run_separability=False, lags=(0, 1),
    )
    table = S.format_season_table({"train": reps})
    assert "season" in table and "separable" in table
    assert len(table.splitlines()) >= 3
