"""
Shared plumbing for the v4 entry points.

``train_v4`` and ``infer_v4`` must agree on three things or their results are not
comparable, and the disagreement is silent:

* **the calendar** -- the same archive window, split date and decimation, or the
  test frames are not the frames the operator was identified against;
* **the weather roles** -- precipitation into ``B_p``, the three measurements into
  ``C``, with the *training* climatology, since refitting the harmonics on the test
  split would leak;
* **the normalisation** -- ``(mean, std)`` from training, or the decoded rasters are
  in the wrong units.

Rather than have each script rebuild all of it (and drift), both call
:func:`load_inputs`, and the checkpoint carries every field needed to reproduce the
split at inference time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dbwm.config import (
    BackboneConfig, BasisConfig, DataConfig, DynamicsConfig, ExperimentConfig,
    ForcingConfig, HorizonConfig, InferenceConfig, MemoryConfig, PlanningConfig,
    SeasonConfig, TrainingConfig, WeatherConfig,
)
from dbwm.data.ndvi_dataset import load_calendar_dataset
from dbwm.data.seasons import split_indices
from dbwm.data.weather import build_weather_table, synthetic_weather

CKPT_VERSION = 4


def add_data_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register the data/calendar arguments shared by both entry points.

    :param p: the parser.
    :return: the same parser.
    """
    p.add_argument("--modality", default="ndvi", choices=["ndvi", "lst"])
    p.add_argument("--ndvi-dir", default=None)
    p.add_argument("--lst-dir", default=None)
    p.add_argument("--weather-csv", default=None)
    # Default to None, not to the calendar literals: ``infer_v4`` must be able to
    # tell "the user asked for this window" from "argparse filled in a default",
    # or every inference run would silently overwrite the checkpoint's own split.
    p.add_argument("--archive-start", default=None, help="default: 2022-01-01")
    p.add_argument("--archive-end", default=None, help="default: 2026-04-30")
    p.add_argument("--split-date", default=None, help="default: 2025-04-15")
    p.add_argument("--max-pixels", type=int, default=250_000,
                   help="per-frame pixel budget; larger frames are read DECIMATED")
    p.add_argument("--ram-budget-gb", type=float, default=8.0)
    p.add_argument("--smoke", action="store_true", help="tiny synthetic end-to-end run")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def add_model_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register the knobs that control the error budget, shared by every entry point.

    These four stages each put a floor under the forecast, and the printed error
    budget names which one is currently binding. They are registered together so
    the same flag means the same thing in every script -- a knob that exists in
    ``run_ndvi_v4`` but not in ``train_v4`` is a trap, because the documented fix
    for a diagnosis silently does nothing in the other entry point.

    :param p: the parser.
    :return: the same parser.
    """
    g = p.add_argument_group(
        "error budget",
        "Stages that bound the forecast before the dynamics are applied. Run "
        "once, read the ERROR BUDGET block, then raise whichever term it names "
        "as binding -- the binding term moves once relieved.",
    )
    g.add_argument(
        "--eof-modes", type=int, default=None,
        help="cap on the number of empirical modes appended to Psi (default "
             "96). These close the REPRESENTATION term: a smooth coordinate MLP "
             "cannot express sharp plot boundaries, which is what pinned "
             "reconstruction R^2 at 0.958. 0 disables them and recovers the "
             "learned basis alone.",
    )
    g.add_argument(
        "--eof-energy", type=float, default=None,
        help="fraction of the RESIDUAL variance the empirical modes retain "
             "(default 0.999). Interacts with --eof-modes: whichever binds first "
             "wins.",
    )
    g.add_argument(
        "--eof-select", default=None, choices=["holdout", "energy"],
        help="how the empirical mode count q is chosen (default holdout). "
             "'holdout' minimises reconstruction error on a chronological tail "
             "of TRAINING that the modes were not fitted to; 'energy' restores "
             "the cumulative-training-variance rule, which is monotone in q and "
             "so can only ever stop at the --eof-modes cap. The holdout path "
             "also reports the fitted-vs-held-out gap, which is the term every "
             "forecast inherits and the only one that says whether the basis is "
             "mode-starved or over-fitted.",
    )
    g.add_argument(
        "--eof-holdout-fraction", type=float, default=None,
        help="tail fraction of TRAINING used to select q (default 0.2)",
    )
    g.add_argument(
        "--no-climatology", action="store_true",
        help="do not subtract the per-pixel training mean. The mosaic is largely "
             "static, so this removes most of the spatial variance for free; "
             "disabling it is an ablation, not a speed-up.",
    )
    g.add_argument(
        "--gain-max", type=float, default=None,
        help="hard rejection threshold on max_h ||S A_cal^h||_2 during blend "
             "selection (default 2.0). This is the AMPLIFICATION term: v2 "
             "Thm 4.2 multiplies the encoding error by it at every horizon. Not "
             "a target -- the blend weight itself is chosen by held-out h-step "
             "error. 1.0 exactly admits nothing but persistence, since "
             "persistence already sits at gain 1.",
    )
    g.add_argument(
        "--absolute-operator", action="store_true",
        help="fit w_{t+1} directly instead of the increment w_{t+1} - w_t. Same "
             "model class, but the ridge then shrinks toward ZERO rather than "
             "toward persistence -- an ablation of the fix, not a default.",
    )
    g.add_argument(
        "--no-teacher-forcing", action="store_true",
        help="train Psi on single frames only (v2 Remark 6.1 schedule (a)), "
             "disabling the K-step teacher-forced rollout term.",
    )
    g.add_argument(
        "--no-physical-range", action="store_true",
        help="keep raster pixels outside the modality's PLAUSIBILITY range "
             "(LST: 250-350 K) instead of masking them. An ablation of the "
             "quality gate: a downscaled product extrapolating past its training "
             "support can emit impossible temperatures, and one such pixel moves "
             "the training mean and std that every frame is standardised "
             "against. No effect on NDVI, which declares no range. Note this is "
             "NOT the display range -- a 47 degC pixel is ordinary data that "
             "merely clips on the colourbar.",
    )
    g.add_argument(
        "--lst-valid-range", type=float, nargs=2, default=None,
        metavar=("LO", "HI"),
        help="override the LST plausibility range, in kelvin (default 250 350). "
             "Widen it if your product legitimately ranges further; the loader "
             "reports the archive's actual min/max and percentiles every run, so "
             "you can see whether the default is cutting real data.",
    )
    g.add_argument(
        "--keep-empty-frames", action="store_true",
        help="count an all-no-data raster as an OBSERVED date instead of skipping "
             "it. Almost never what you want: `observed` gates the date-"
             "intersection mask, so one empty frame drives it to zero pixels and "
             "the run fails in Stage 2 with a message naming the AOI rather than "
             "the file. Provided so the behaviour can be ablated, not used.",
    )
    return p


def apply_model_args(cfg, args):
    """
    Fold the shared error-budget arguments into a config.

    :param cfg: an :class:`~dbwm.config.ExperimentConfig`.
    :param args: parsed arguments.
    :return: the same config, mutated.
    """
    if getattr(args, "eof_modes", None) is not None:
        cfg.basis.eof_modes = args.eof_modes
    if getattr(args, "eof_energy", None) is not None:
        cfg.basis.eof_energy = args.eof_energy
    if getattr(args, "eof_select", None) is not None:
        cfg.basis.eof_select = args.eof_select
    if getattr(args, "eof_holdout_fraction", None) is not None:
        cfg.basis.eof_holdout_fraction = args.eof_holdout_fraction
    if getattr(args, "no_climatology", False):
        cfg.basis.use_climatology = False
    if getattr(args, "gain_max", None) is not None:
        cfg.memory.gain_max = args.gain_max
    if getattr(args, "absolute_operator", False):
        cfg.memory.increment = False
    if getattr(args, "no_teacher_forcing", False):
        cfg.training.rollout_steps = 0
        cfg.training.lambda_rollout = 0.0
    if getattr(args, "no_physical_range", False):
        cfg.data.apply_physical_range = False
    if getattr(args, "lst_valid_range", None) is not None:
        lo, hi = args.lst_valid_range
        cfg.data.valid_range_override = (float(lo), float(hi))
    if getattr(args, "keep_empty_frames", False):
        cfg.data.drop_empty_frames = False
    # Last, so it sees the final modality: renames the run only if the name is
    # still an untouched default.
    cfg.apply_modality_defaults()
    return cfg


@dataclass
class Inputs:
    """
    Everything both entry points need from disk.

    :ivar ds: the calendar dataset.
    :ivar weather: the role-split weather table.
    :ivar train_sel: ``(T,)`` boolean training mask over the calendar.
    :ivar train_idx: training indices.
    :ivar test_idx: test indices.
    :ivar forcing: ``(T, ell)`` precipitation forcing for ``B_p``.
    :ivar measurement: ``(T, m)`` weather measurements for ``C``.
    :ivar split: the split date.
    """

    ds: Any
    weather: Any
    train_sel: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    forcing: np.ndarray
    measurement: np.ndarray
    split: dt.date


def load_inputs(cfg: ExperimentConfig, args, logger) -> Inputs:
    """
    Build the calendar dataset and the role-split weather table.

    The weather climatology is always fitted on the TRAINING mask only, whichever
    entry point calls this, so a forecast never sees a seasonal cycle estimated
    partly from its own evaluation period.

    :param cfg: the experiment config (already override-folded).
    :param args: parsed arguments carrying the data paths.
    :param logger: where to report the split.
    :return: the :class:`Inputs` bundle.
    """
    synthetic = bool(getattr(args, "smoke", False) or getattr(args, "synthetic", False))
    start = dt.date.fromisoformat(cfg.seasons.archive_start)
    end = dt.date.fromisoformat(cfg.seasons.archive_end)
    split = dt.date.fromisoformat(cfg.seasons.split_date)
    if getattr(args, "smoke", False):
        end = start + dt.timedelta(days=729)
        split = start + dt.timedelta(days=560)
        # Record the window actually used, so the checkpoint's split_date matches
        # the data the operator was fitted on. Leaving the real-data defaults in
        # place makes infer_v4 rebuild a different split from the same file.
        cfg.seasons.archive_end = end.isoformat()
        cfg.seasons.split_date = split.isoformat()

    ds = load_calendar_dataset(
        None if synthetic else cfg.data_dir(), start, end, split,
        nodata=cfg.data.nodata_value,
        height=cfg.data.image_height if synthetic else None,
        width=cfg.data.image_width if synthetic else None,
        max_pixels=getattr(args, "max_pixels", 250_000),
        ram_budget_gb=getattr(args, "ram_budget_gb", 8.0),
        synthetic=synthetic,
        synthetic_shape=(
            (cfg.data.image_height or 32, cfg.data.image_width or 32)
            if getattr(args, "smoke", False) else (48, 44)
        ),
        seed=getattr(args, "seed", 0),
        modality=cfg.modality_spec().label,
        physical_range=cfg.physical_range(),
        drop_empty=cfg.drop_empty_frames(),
        min_valid_fraction=cfg.modality_spec().min_valid_fraction,
    )
    train_idx, test_idx = split_indices(ds.dates, split)
    train_sel = np.zeros(ds.n_steps, dtype=bool)
    train_sel[train_idx] = True

    use_real = (not synthetic) or (
        getattr(args, "weather_csv", None) and os.path.exists(args.weather_csv)
    )
    wt = (
        build_weather_table(
            ds.dates, cfg.weather.csv_path, train_sel, cfg.weather.precip_lags,
            cfg.weather.measurement_cols, cfg.weather.n_harmonics,
            cfg.weather.scale_forcing,
        )
        if use_real else
        synthetic_weather(ds.dates, train_sel, cfg.weather.precip_lags,
                          cfg.weather.measurement_cols, seed=getattr(args, "seed", 0))
    )
    spec = cfg.modality_spec()
    logger.info(
        "%s | calendar %s..%s | %d dates, %d observed | train %d / test %d "
        "(split %s) | grid %dx%d @ %.1f m | units %s",
        spec.label, ds.dates[0], ds.dates[-1], ds.n_steps, int(ds.observed.sum()),
        len(train_idx), len(test_idx), split, ds.shape[0], ds.shape[1],
        ds.pixel_size, spec.units,
    )
    return Inputs(
        ds=ds, weather=wt, train_sel=train_sel, train_idx=train_idx,
        test_idx=test_idx,
        forcing=np.asarray(wt.forcing, dtype=np.float64),
        measurement=np.asarray(wt.measurement, dtype=np.float64),
        split=split,
    )


def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """
    Rebuild an :class:`ExperimentConfig` from a checkpoint's ``to_dict()``.

    Every sub-config is restored explicitly. Dropping ``weather``, ``memory`` or
    ``seasons`` would silently fall back to defaults, so a model trained at ``L=7``
    with four precipitation lags would be reloaded expecting the default lags and
    ``B_p`` would be applied against a mismatched ``p_t``.

    :param d: the saved dict.
    :return: the config.
    """
    known = {
        "data": DataConfig, "forcing": ForcingConfig, "backbone": BackboneConfig,
        "basis": BasisConfig, "dynamics": DynamicsConfig, "training": TrainingConfig,
        "inference": InferenceConfig, "planning": PlanningConfig,
        "weather": WeatherConfig, "seasons": SeasonConfig, "memory": MemoryConfig,
        "horizons": HorizonConfig,
    }
    kwargs = {k: cls(**d[k]) for k, cls in known.items() if k in d}
    return ExperimentConfig(name=d.get("name", "dbwm_v4"), **kwargs).sync_input_dim()


def save_checkpoint(path: str, payload: Dict[str, Any]) -> str:
    """
    Pickle a v4 checkpoint.

    :param path: destination ``.pkl``.
    :param payload: the checkpoint contents.
    :return: the path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = dict(payload)
    payload["ckpt_version"] = CKPT_VERSION
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a v4 checkpoint, refusing a v2 one rather than misreading it.

    A v2 checkpoint has a single ``A``; a v4 one has ``blocks`` of shape
    ``(L, k, k)``. Loading the former here would treat ``A`` as one memory block and
    quietly forecast with ``L = 1`` and no lag structure.

    :param path: the ``.pkl``.
    :return: the checkpoint dict.
    :raises ValueError: if the file is not a v4 checkpoint.
    """
    with open(path, "rb") as fh:
        ckpt = pickle.load(fh)
    if ckpt.get("ckpt_version") != CKPT_VERSION:
        raise ValueError(
            "{} is not a DB-WM v4 checkpoint (ckpt_version={!r}). A v2 checkpoint "
            "stores a single A and would be silently forecast as L=1; retrain with "
            "experiments.train_v4.".format(path, ckpt.get("ckpt_version"))
        )
    return ckpt
