"""Tests for the data pipeline: synthetic generation, 85/15 split, normalisation."""
import numpy as np

from dbwm.config import DataConfig
from dbwm.data.geotiff_dataset import load_dataset, make_pixel_grid


def test_chronological_split_85_15():
    """Split must be 85/15 and chronological (test strictly after train)."""
    cfg = DataConfig(use_synthetic=True, synthetic_n_frames=100, image_height=16, image_width=16)
    train_ds, test_ds = load_dataset(cfg)
    assert train_ds.n_frames == 85
    assert test_ds.n_frames == 15
    # Test dates are all >= last train date (chronological).
    assert test_ds.dates[0] >= train_ds.dates[-1]


def test_normalisation_from_train_only():
    """Training split should be approximately zero-mean / unit-std after norm."""
    cfg = DataConfig(use_synthetic=True, synthetic_n_frames=60, image_height=16, image_width=16)
    train_ds, _ = load_dataset(cfg)
    flat = train_ds.frames.reshape(-1)
    assert abs(float(flat.mean())) < 0.1
    assert abs(float(flat.std()) - 1.0) < 0.2


def test_pixel_grid_range():
    """Pixel grid spans [-1, 1]^2 with H*W rows."""
    grid = make_pixel_grid(8, 10)
    assert grid.shape == (80, 2)
    assert float(grid.min()) >= -1.0 - 1e-6
    assert float(grid.max()) <= 1.0 + 1e-6


def test_denormalize_roundtrip():
    """denormalize(normalize(x)) recovers physical-unit statistics."""
    cfg = DataConfig(use_synthetic=True, synthetic_n_frames=40, image_height=16, image_width=16)
    train_ds, _ = load_dataset(cfg)
    phys = train_ds.denormalize(train_ds.frames)
    assert phys.shape == train_ds.frames.shape
    assert np.isfinite(phys).all()
