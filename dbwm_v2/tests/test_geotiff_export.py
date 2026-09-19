"""
Tests for the forecast GeoTIFF export.

The export is the only place a latent forecast becomes a product someone opens in
QGIS, so the properties worth pinning are the ones a reader would otherwise take on
trust: that georeferencing survives, that invalid pixels stay invalid, and that the
uncertainty band means what its name says.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dbwm.evaluation.geotiff_export import (  # noqa: E402
    _uncertainty_band, export_forecast_maps, write_forecast_geotiff,
)


class _Extractor:
    """Minimal stand-in exposing the two attributes the band needs."""

    def __init__(self, phi, sigma_eps2=0.01):
        self.phi = phi
        self.sigma_eps2 = sigma_eps2

    def decode(self, w):
        return np.asarray(w) @ self.phi.T


class _Subspace:
    """Minimal stand-in for a rank-``k`` latent subspace."""

    def __init__(self, basis):
        self.basis = basis
        self.r = basis.shape[0]

    @property
    def k(self):
        return self.basis.shape[1]

    def reconstruct(self, z):
        return np.asarray(z) @ self.basis.T


class _DS:
    """Minimal calendar-dataset stand-in."""

    def __init__(self, frames, dates, transform, crs, mask):
        self.frames = frames
        self.dates = dates
        self.transform = transform
        self.crs = crs
        self.shape = frames.shape[1:]
        self.n_steps = frames.shape[0]
        self.observed = np.ones(self.n_steps, dtype=bool)
        self.static_mask = mask
        self.mask_is_static = True
        self.valid_mask = np.broadcast_to(mask, frames.shape)
        self.mean = np.array([0.3])
        self.std = np.array([0.2])
        self.modality = "ndvi"


def _write(tmp_path, **kw):
    """Write a small 3-band raster and return its path."""
    path = os.path.join(str(tmp_path), "f.tif")
    mean = kw.pop("mean", np.arange(12.0).reshape(3, 4))
    return write_forecast_geotiff(
        path, mean, kw.pop("std", np.full((3, 4), 0.05)),
        kw.pop("error", np.zeros((3, 4))),
        kw.pop("transform", (30.0, 0.0, 419670.0, 0.0, -30.0, 3440790.0)),
        kw.pop("crs", "EPSG:32643"), **kw,
    )


def test_georeferencing_and_band_names_survive(tmp_path):
    """CRS, affine transform and band descriptions must round-trip."""
    path = _write(tmp_path)
    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:32643"
        assert (src.transform.a, src.transform.e) == (30.0, -30.0)
        assert (src.transform.c, src.transform.f) == (419670.0, 3440790.0)
        assert src.descriptions == ("NDVI_forecast", "NDVI_std", "NDVI_error")
        assert src.count == 3


def test_invalid_pixels_are_written_as_nodata(tmp_path):
    """
    Pixels outside the field clip must never carry a number.

    The basis extrapolates everywhere, so writing its extrapolation over the ~40%
    of the bounding box outside the field would let a reader mistake it for a
    prediction about ground that was never observed.
    """
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1:3] = True
    path = _write(tmp_path, valid_mask=mask)
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
    assert band.mask.sum() == 10
    assert not band.mask[1, 1] and not band.mask[1, 2]


def test_non_finite_values_become_nodata(tmp_path):
    """A NaN in the error band must not be written as a finite number."""
    err = np.zeros((3, 4))
    err[0, 0] = np.nan
    err[2, 3] = np.inf
    path = _write(tmp_path, error=err)
    with rasterio.open(path) as src:
        band = src.read(3, masked=True)
    assert band.mask[0, 0] and band.mask[2, 3]
    assert band.mask.sum() == 2


def test_omitted_bands_reduce_the_count(tmp_path):
    """With no truth frame there is no error band, and the count must reflect it."""
    path = os.path.join(str(tmp_path), "one.tif")
    write_forecast_geotiff(
        path, np.zeros((3, 4)), None, None,
        (30.0, 0.0, 0.0, 0.0, -30.0, 0.0), "EPSG:32643",
    )
    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.descriptions == ("NDVI_forecast",)


def test_uncertainty_band_conserves_total_power():
    """
    The band must lift the trace, not broadcast its square root.

    Broadcasting ``sqrt(trace)`` overstates the per-pixel deviation by ``sqrt(k)``.
    The isotropic lift instead spreads the trace over the ``k`` coordinates, so with
    an orthonormal basis (``||Psi(s)||^2 = 1``) the recovered variance is exactly
    ``trace / k`` plus the observation noise.
    """
    k, n = 8, 20
    phi = np.zeros((n, k))
    phi[:, 0] = 1.0  # ||Psi(s)||^2 = 1 everywhere
    ex = _Extractor(phi, sigma_eps2=0.0)
    trace, scale = 4.0, 1.0
    band = _uncertainty_band(trace, ex, None, scale)
    assert np.allclose(band, np.sqrt(trace / k))
    # The naive flat band would have been sqrt(4) = 2, i.e. sqrt(k) = 2.83x larger.
    assert band[0] < np.sqrt(trace)


def test_uncertainty_band_varies_where_the_basis_does():
    """Uncertainty must be larger where the basis has less support."""
    phi = np.array([[1.0, 0.0], [3.0, 0.0], [0.5, 0.0]])
    band = _uncertainty_band(2.0, _Extractor(phi, 0.0), None, 1.0)
    assert band[1] > band[0] > band[2]
    assert band.std() > 0


def test_uncertainty_band_respects_the_subspace():
    """
    With a subspace, the lift goes through ``U`` before ``Psi``.

    The dynamics only claim variance inside the subspace, so a direction the model
    never modelled must not acquire uncertainty from it.
    """
    r, k, n = 6, 2, 5
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(n, r))
    basis = np.linalg.qr(rng.normal(size=(r, k)))[0]
    ex = _Extractor(phi, sigma_eps2=0.0)
    got = _uncertainty_band(3.0, ex, _Subspace(basis), 1.0)
    want = np.sqrt((3.0 / k) * np.einsum("nk,nk->n", phi @ basis, phi @ basis))
    assert np.allclose(got, want)
    # Using the ambient r instead of k would inflate every pixel.
    assert np.all(got > np.sqrt((3.0 / r) * np.einsum("nk,nk->n",
                                                      phi @ basis, phi @ basis)))


def test_scale_returns_physical_units():
    """The band is multiplied by the normalisation std, like the mean band."""
    phi = np.ones((4, 2))
    a = _uncertainty_band(1.0, _Extractor(phi, 0.0), None, 1.0)
    b = _uncertainty_band(1.0, _Extractor(phi, 0.0), None, 0.25)
    assert np.allclose(b, 0.25 * a)


def test_export_writes_one_file_per_valid_step(tmp_path):
    """One raster per (origin, horizon) pair that the filter marked valid."""
    import datetime as dt

    rng = np.random.default_rng(1)
    t, h_img, w_img, r = 8, 4, 5, 3
    frames = rng.normal(size=(t, h_img, w_img))
    mask = np.ones((h_img, w_img), dtype=bool)
    mask[0, 0] = False
    ds = _DS(frames, [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(t)],
             (30.0, 0.0, 100.0, 0.0, -30.0, 200.0), "EPSG:32643", mask)
    ex = _Extractor(rng.normal(size=(h_img * w_img, r)))
    origins = [1, 3]
    fc = {
        "mean": rng.normal(size=(2, 2, r)),
        "valid": np.array([[True, True], [True, False]]),
        "cov_trace": np.full((2, 2), 0.5),
    }
    paths = export_forecast_maps(
        str(tmp_path), fc, origins, ds.dates, ex, ds,
        sigma_from=fc["cov_trace"],
    )
    assert len(paths) == 3
    assert all(os.path.exists(p) for p in paths)
    assert any("2024-01-02_t+1" in p for p in paths)
    with rasterio.open(paths[0]) as src:
        assert src.read(1, masked=True).mask[0, 0]  # the invalid pixel stayed masked
        assert src.tags()["DBWM_LEAD_DAYS"] == "1"


def test_export_caps_the_number_of_origins(tmp_path):
    """``max_origins`` must actually bound how much gets written."""
    import datetime as dt

    rng = np.random.default_rng(2)
    t, r = 30, 2
    frames = rng.normal(size=(t, 3, 3))
    ds = _DS(frames, [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(t)],
             (30.0, 0.0, 0.0, 0.0, -30.0, 0.0), "EPSG:32643",
             np.ones((3, 3), dtype=bool))
    ex = _Extractor(rng.normal(size=(9, r)))
    origins = list(range(20))
    fc = {"mean": rng.normal(size=(20, 1, r)), "valid": np.ones((20, 1), bool)}
    paths = export_forecast_maps(str(tmp_path), fc, origins, ds.dates, ex, ds,
                                 max_origins=4)
    assert len(paths) == 4
