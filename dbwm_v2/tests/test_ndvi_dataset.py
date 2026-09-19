"""
Tests for the calendar-reindexed dataset and the GP-posterior state extraction.

The calendar reindex is what makes "lag-6" mean six *days*; the GP path is what
makes a ~48%-valid mask a non-issue. Both are pinned here.
"""
import datetime as dt

import numpy as np
import pytest

from dbwm.data import ndvi_dataset as ND
from dbwm.gp.state import GPStateExtractor

START, END, SPLIT = dt.date(2022, 1, 1), dt.date(2026, 4, 30), dt.date(2025, 4, 15)


def _dataset(shape=(24, 20), seed=0):
    return ND.load_calendar_dataset(
        None, START, END, SPLIT, synthetic=True, synthetic_shape=shape, seed=seed
    )


def _fourier_basis(coords, r=48, seed=0):
    """A fixed random-Fourier basis standing in for a trained Psi."""
    rng = np.random.RandomState(seed)
    w = rng.randn(2, r // 2) * 3.0
    b = rng.rand(r // 2) * 2 * np.pi
    return np.concatenate(
        [np.cos(coords @ w + b), np.sin(coords @ w + b)], axis=-1
    ) / np.sqrt(r // 2)


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def test_calendar_is_daily_complete_and_matches_the_archive_span():
    """1581 daily steps over 2022-01-01..2026-04-30, matching the real archive."""
    ds = _dataset()
    assert ds.n_steps == 1581
    assert ds.dates[0] == START and ds.dates[-1] == END
    diffs = {(ds.dates[i + 1] - ds.dates[i]).days for i in range(ds.n_steps - 1)}
    assert diffs == {1}  # every step is exactly one day -- what the lift assumes


def test_missing_dates_are_marked_not_dropped():
    """Gaps must remain in the index, so a 'lag-6' never spans 7 or 8 real days."""
    ds = _dataset()
    assert (~ds.observed).sum() > 0
    assert ds.observed.sum() < ds.n_steps
    # Unobserved steps carry no data anyone could mistake for a measurement.
    assert np.allclose(ds.frames[~ds.observed], 0.0)


def test_duplicate_dates_are_rejected(tmp_path):
    """Two files for one date would silently drop data under a last-one-wins rule."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    prof = dict(
        driver="GTiff", height=4, width=4, count=1, dtype="float32",
        crs="EPSG:32643", transform=from_origin(419670.0, 3440790.0, 30.0, 30.0),
    )
    for name in ("NDVI_20220101_a.tif", "NDVI_20220101_b.tif"):
        with rasterio.open(str(tmp_path / name), "w", **prof) as dst:
            dst.write(np.ones((4, 4), dtype="float32"), 1)
    with pytest.raises(ValueError, match="Two files map to"):
        ND.discover_frames(str(tmp_path))


def test_static_mask_is_detected():
    """A date-invariant field clip must be recognised so Phi_X can be cached."""
    ds = _dataset()
    assert ds.mask_is_static
    assert ds.static_mask.shape == ds.shape
    assert 0.0 < ds.static_mask.mean() < 1.0


def test_normalisation_uses_training_rows_only():
    """Statistics must not see the test period."""
    ds = _dataset()
    train = np.array([d < SPLIT for d in ds.dates])
    sel = train & ds.observed
    vals = ds.frames[sel][ds.valid_mask[sel]]
    assert abs(float(vals.mean())) < 1e-4
    assert abs(float(vals.std()) - 1.0) < 1e-3
    test_sel = (~train) & ds.observed
    test_vals = ds.frames[test_sel][ds.valid_mask[test_sel]]
    assert abs(float(test_vals.mean())) > 1e-4  # test rows had no say


def test_coordinates_come_from_the_affine_transform():
    """Psi must see ground geometry, not array indices."""
    ds = _dataset(shape=(24, 20))
    assert ds.coords.shape == (24 * 20, 2)
    assert np.isclose(ds.coords.min(), -1.0) and np.isclose(ds.coords.max(), 1.0)
    # e = -30 in the transform, so y decreases down the rows: row 0 is the top.
    y = ds.coords[:, 1].reshape(24, 20)
    assert y[0, 0] > y[-1, 0]


def test_subset_preserves_geometry():
    ds = _dataset()
    idx = np.arange(100, 200)
    sub = ds.subset(idx)
    assert sub.n_steps == 100
    assert sub.dates[0] == ds.dates[100]
    assert np.shares_memory(sub.coords, ds.coords) or np.allclose(sub.coords, ds.coords)
    assert sub.mask_is_static == ds.mask_is_static


def test_gap_runs_are_reported():
    observed = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)
    runs = ND._gap_runs(observed)
    assert runs == [(2, 2), (5, 1)]


def test_synthetic_archive_matches_the_real_one_in_shape():
    """The synthetic path reproduces 1581 days / ~1568 frames so gaps get exercised."""
    ds = _dataset()
    assert ds.n_steps == 1581
    assert 1550 <= int(ds.observed.sum()) <= 1580


# --------------------------------------------------------------------------- #
# GP-posterior state extraction
# --------------------------------------------------------------------------- #
def test_gp_solve_reconstructs_the_field():
    """w_t = Lambda^-1 Phi^T y_t must actually summarise the raster."""
    ds = _dataset(shape=(28, 24))
    phi = _fourier_basis(ds.coords, r=64)
    ex = GPStateExtractor(phi, 1e-3, ds.static_mask.reshape(-1))
    out = ex.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=False)
    assert np.nanmedian(out["reconstruction_r2"][ds.observed]) > 0.9


def test_unobserved_dates_get_zero_weight_and_huge_covariance():
    """Nothing downstream may mistake a missing frame for a measurement."""
    ds = _dataset(shape=(20, 18))
    ex = GPStateExtractor(_fourier_basis(ds.coords, r=32), 1e-3, ds.static_mask.reshape(-1))
    out = ex.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    miss = ~ds.observed
    assert np.allclose(out["weights"][miss], 0.0)
    assert np.all(np.trace(out["covariances"][miss], axis1=1, axis2=2) > 1e6)
    assert np.all(out["n_valid"][miss] == 0)


def test_invalid_pixels_are_omitted_not_imputed():
    """Corrupting masked-out pixels must not change w_t at all.

    This is the concrete advantage of the GP path over the encoder path: an
    invalid pixel is simply absent from Phi_{X_t}, whereas a CNN must be fed a
    zero-filled grid and cannot help but see the fill value.
    """
    ds = _dataset(shape=(24, 20))
    phi = _fourier_basis(ds.coords, r=48)
    ex = GPStateExtractor(phi, 1e-3, ds.static_mask.reshape(-1))
    t = 500
    w_clean, _ = ex.solve_frame(ds.frames[t], ds.valid_mask[t])
    corrupted = ds.frames[t].copy()
    corrupted[~ds.valid_mask[t]] = 1e6
    w_corrupt, _ = ex.solve_frame(corrupted, ds.valid_mask[t])
    assert np.allclose(w_clean, w_corrupt, atol=1e-9)


def test_posterior_covariance_shrinks_with_more_valid_pixels():
    """Fewer valid pixels must mean less confidence -- a scalar sigma^2 I loses this."""
    ds = _dataset(shape=(28, 24))
    phi = _fourier_basis(ds.coords, r=48)
    ex = GPStateExtractor(phi, 1e-3)
    full = ds.static_mask.reshape(-1).copy()
    sparse = full.copy()
    idx = np.nonzero(sparse)[0]
    sparse[idx[: int(0.8 * idx.size)]] = False  # keep only 20% of the pixels
    _, cov_full = ex.solve_frame(ds.frames[400], full, want_cov=True)
    _, cov_sparse = ex.solve_frame(ds.frames[400], sparse, want_cov=True)
    assert np.trace(cov_sparse) > np.trace(cov_full)


def test_cache_matches_the_uncached_path():
    """The static-mask fast path must be numerically identical to recomputing."""
    ds = _dataset(shape=(24, 20))
    phi = _fourier_basis(ds.coords, r=40)
    m = ds.static_mask.reshape(-1)
    cached = GPStateExtractor(phi, 1e-3, m)
    plain = GPStateExtractor(phi, 1e-3, None)
    for t in (10, 700, 1400):
        a, ca = cached.solve_frame(ds.frames[t], m, want_cov=True)
        b, cb = plain.solve_frame(ds.frames[t], m, want_cov=True)
        assert np.allclose(a, b, atol=1e-10)
        assert np.allclose(ca, cb, atol=1e-10)


def test_decode_variance_is_positive_and_includes_noise():
    ds = _dataset(shape=(16, 16))
    phi = _fourier_basis(ds.coords, r=32)
    ex = GPStateExtractor(phi, 5e-3)
    var = ex.decode_variance(np.eye(32) * 0.1)
    assert np.all(var >= 5e-3 - 1e-12)
    assert np.all(np.isfinite(var))


def test_low_reconstruction_quality_is_warned(caplog):
    """A basis that cannot represent the field invalidates everything downstream."""
    ds = _dataset(shape=(20, 18))
    rng = np.random.RandomState(3)
    phi = rng.randn(ds.coords.shape[0], 4) * 1e-3  # far too weak to fit anything
    ex = GPStateExtractor(phi, 1.0, ds.static_mask.reshape(-1))
    with caplog.at_level("WARNING"):
        ex.solve_sequence(ds.frames[:50], ds.valid_mask[:50], ds.observed[:50], False)
    assert any("spatial basis Psi is not" in rec.getMessage() for rec in caplog.records)
