"""
Tests for the v4 checkpoint contract.

``train_v4`` and ``infer_v4`` communicate only through a pickle, so every
disagreement between them is silent by construction: a forecast still runs, it is
just wrong. These tests pin the failure modes that would otherwise go unnoticed --
loading a v2 checkpoint as if it were v4, and losing a sub-config on the way back
in.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from dbwm.config import default_config
from experiments._v4_common import (
    CKPT_VERSION, config_from_dict, load_checkpoint, save_checkpoint,
)


def test_round_trip_preserves_arrays(tmp_path):
    """What goes in comes back bit-identical."""
    blocks = np.arange(12.0).reshape(3, 2, 2)
    path = save_checkpoint(str(tmp_path / "c.pkl"), {"blocks": blocks, "rho": 0.9})
    got = load_checkpoint(path)
    assert np.array_equal(got["blocks"], blocks)
    assert got["rho"] == 0.9
    assert got["ckpt_version"] == CKPT_VERSION


def test_v2_checkpoint_is_refused(tmp_path):
    """
    A v2 checkpoint must raise, not load.

    v2 stores a single ``A``; v4 stores ``L`` blocks. Loading the former without
    complaint would forecast a memoryless model while every log line still claimed
    ``L = 7``, which is the kind of error that survives to a thesis table.
    """
    path = str(tmp_path / "v2.pkl")
    with open(path, "wb") as fh:
        pickle.dump({"params": {}, "A": np.eye(2), "B": np.zeros((2, 1)),
                     "Q": np.eye(2), "config": {}}, fh)
    with pytest.raises(ValueError, match="not a DB-WM v4 checkpoint"):
        load_checkpoint(path)


def test_older_v4_version_is_refused(tmp_path):
    """A future format bump must invalidate old checkpoints rather than misread them."""
    path = str(tmp_path / "old.pkl")
    with open(path, "wb") as fh:
        pickle.dump({"ckpt_version": CKPT_VERSION - 1}, fh)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_config_round_trip_keeps_every_v4_subconfig():
    """
    ``weather``, ``seasons``, ``memory`` and ``horizons`` must all survive.

    Dropping any of them silently reverts to defaults: a model trained at ``L = 7``
    with four precipitation lags would reload expecting the default lags, and
    ``B_p`` would then be applied against a mismatched ``p_t``.
    """
    cfg = default_config()
    cfg.memory.order = 7
    cfg.horizons.horizon = 6
    cfg.weather.precip_lags = 4
    cfg.seasons.split_date = "2025-04-15"
    cfg.basis.r = 128

    back = config_from_dict(cfg.to_dict())
    assert back.memory.order == 7
    assert back.horizons.horizon == 6
    assert back.weather.precip_lags == 4
    assert back.seasons.split_date == "2025-04-15"
    assert back.basis.r == 128
    assert back.weather.measurement_cols == cfg.weather.measurement_cols


def test_config_round_trip_reruns_sync_input_dim():
    """
    The reloaded config must be internally consistent, not merely field-equal.

    ``sync_input_dim`` derives the forcing width from the lag count; a config that
    skipped it would carry a stale width and mis-shape ``B_p``.
    """
    cfg = default_config()
    cfg.weather.precip_lags = 5
    cfg.sync_input_dim()
    back = config_from_dict(cfg.to_dict())
    assert back.to_dict() == cfg.to_dict()


def test_checkpoint_directory_is_created(tmp_path):
    """Saving into a directory that does not exist yet must work."""
    path = os.path.join(str(tmp_path), "nested", "deeper", "c.pkl")
    save_checkpoint(path, {"x": 1})
    assert os.path.exists(path)
