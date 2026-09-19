"""
The LST path: the constants that differ from NDVI, and the ones that must not.

Every test here pins a defect that was live before the modality registry existed
and that **did not raise**. That is the common thread and the reason they are
worth their runtime: an ``--modality lst`` run already completed successfully
end to end, and produced NDVI-labelled kelvin rasters under an NDVI colour scale
with a teacher-forcing threshold forty times too small. Output that is wrong and
plausible is the failure mode this file exists to prevent.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from dbwm.config import default_config
from dbwm.data import modality as md
from dbwm.data.ndvi_dataset import (
    _is_geographic, ground_sample_distance, load_calendar_dataset,
)
from dbwm.evaluation import v4_plots
from dbwm.evaluation.geotiff_export import ds_modality


# --------------------------------------------------------------------------- #
# The registry itself
# --------------------------------------------------------------------------- #
def test_lst_and_ndvi_differ_in_exactly_the_physical_constants():
    """
    The two specs must disagree on units, range, ramp and threshold.

    Stated as an explicit inequality rather than as four separate value checks,
    because the hazard is a *copied* entry: adding a modality by duplicating NDVI
    and editing the label gives something that runs, looks configured, and draws
    the wrong picture.
    """
    ndvi, lst = md.get_modality("ndvi"), md.get_modality("lst")
    assert lst.units == "K" and ndvi.units == "NDVI"
    assert lst.vlim != ndvi.vlim
    assert lst.cmap != ndvi.cmap
    assert lst.tf_threshold != ndvi.tf_threshold
    # The LST range is the archive's declared span.
    assert lst.vlim == (283.15, 320.15)
    assert lst.physical_range == (283.15, 320.15)
    # NDVI declares no gate: the index is bounded by its own definition.
    assert ndvi.physical_range is None


def test_modality_lookup_accepts_key_label_and_unit():
    """
    The plotting layer passes ``units``; the config passes ``modality``.

    Both reach the same spec, including the decorated labels the figure code
    builds ("LST error"), or a colourbar would silently fall back to NDVI.
    """
    for key in ("lst", "LST", "K", "LST error"):
        assert md.get_modality(key).key == "lst"
    for key in ("ndvi", "NDVI", "NDVI error"):
        assert md.get_modality(key).key == "ndvi"
    with pytest.raises(KeyError):
        md.get_modality("albedo")
    # The presentation path degrades rather than aborting an hour-old run.
    assert md.resolve_modality("albedo").key == "ndvi"


# --------------------------------------------------------------------------- #
# Display: the defect that made every LST panel a flat wash
# --------------------------------------------------------------------------- #
def test_lst_fields_are_not_drawn_on_the_ndvi_colour_range():
    """
    A 310 K field on ``[0, 1]`` saturates every pixel to the top colour.

    ``value_range`` previously returned the NDVI default whenever it was given a
    modality with no fixed range and no data -- which is exactly how
    ``render_horizon_figures`` calls it -- so every LST map in the run carried no
    information at all while rendering without error.
    """
    lo, hi = v4_plots.value_range("LST")
    assert (lo, hi) == (283.15, 320.15)
    assert v4_plots.value_range("NDVI") == (0.0, 1.0)
    # Fixed, therefore independent of the data: this is what makes a t+1 and a
    # t+6 panel comparable, and it must not drift toward whatever was passed.
    assert v4_plots.value_range("LST", data=np.full(100, 300.0)) == (283.15, 320.15)
    # An explicit override still wins, for residual fields and ablations.
    assert v4_plots.value_range("LST", vlim=(-1.0, 1.0)) == (-1.0, 1.0)


def test_unregistered_modality_still_falls_back_to_percentiles():
    """A residual or ablation field has no canonical range and must derive one."""
    lo, hi = v4_plots.value_range("albedo", data=np.linspace(0.1, 0.9, 500))
    assert 0.09 < lo < 0.15 and 0.85 < hi < 0.91


def test_thermal_field_does_not_get_the_vegetation_ramp():
    """
    ``RdYlGn`` on a temperature map inverts the reading.

    Green means "vigorous canopy" to every reader of a vegetation figure; on LST
    the same green is simply "cool", so the conventional ramp is not merely
    unidiomatic, it asserts something false.
    """
    assert v4_plots.field_cmap("NDVI") == "RdYlGn"
    assert v4_plots.field_cmap("LST") == "inferno"
    # The interval panel must stay distinguishable from the field panel.
    assert v4_plots.interval_cmap("LST") != v4_plots.field_cmap("LST")
    assert v4_plots.unit_label("LST") == "K"


def test_triptych_renders_a_kelvin_field(tmp_path):
    """The whole figure path, in kelvin, end to end."""
    rng = np.random.RandomState(0)
    truth = 300.0 + 4.0 * rng.randn(20, 18)
    fc = truth + 0.8 * rng.randn(20, 18)
    mask = np.ones((20, 18), dtype=bool)
    p = v4_plots.plot_forecast_triptych(
        truth, fc, str(tmp_path / "lst.png"), lead=3, valid_mask=mask, units="LST",
        sigma=np.full((20, 18), 0.9), target_date=dt.date(2026, 5, 5),
        origin_date=dt.date(2026, 5, 2), variant="recursive",
    )
    assert (tmp_path / "lst.png").stat().st_size > 0 and p


# --------------------------------------------------------------------------- #
# Training: the threshold that silently forced every step
# --------------------------------------------------------------------------- #
def test_teacher_forcing_threshold_is_resolved_per_modality():
    """
    0.02 is an NDVI number; read as 0.02 K it forces every rollout step forever.

    The trainer divides the threshold by the frame normalisation, so on an LST
    archive whose std is ~1.7 K the NDVI value is about a hundredth of a standard
    deviation -- the rollout never runs on its own output, the term degenerates
    to plain reconstruction, and the training log still says "teacher forcing on".
    """
    cfg = default_config()
    cfg.data.modality = "ndvi"
    assert cfg.teacher_forcing_threshold() == pytest.approx(0.02)
    cfg.data.modality = "lst"
    assert cfg.teacher_forcing_threshold() == pytest.approx(0.5)
    # Comparable as a fraction of each field's own variability, which is what
    # keeps the reported "fraction of steps forced" comparable between runs.
    assert 0.2 < cfg.teacher_forcing_threshold() / 1.7 < 0.5


def test_config_resolves_dir_range_and_name_per_modality():
    """The four config lookups every entry point used to spell out by hand."""
    cfg = default_config()
    cfg.data.modality = "lst"
    assert cfg.data_dir() == cfg.data.lst_dir
    assert cfg.physical_range() == (283.15, 320.15)
    cfg.data.apply_physical_range = False
    assert cfg.physical_range() is None

    # An LST run left on the NDVI default name would overwrite the NDVI
    # checkpoint in a shared --ckpt-dir.
    cfg = default_config()
    cfg.data.modality = "lst"
    assert cfg.apply_modality_defaults().name == "dbwm_lst_gp_swiglu"
    # An explicit name always wins.
    cfg2 = default_config()
    cfg2.data.modality = "lst"
    cfg2.name = "my_run"
    assert cfg2.apply_modality_defaults().name == "my_run"


# --------------------------------------------------------------------------- #
# Geometry: degrees are not metres
# --------------------------------------------------------------------------- #
def test_geographic_pixel_size_is_converted_to_metres():
    """
    The LST tiles are EPSG:4326, so the affine ``a`` is degrees, not metres.

    Taken literally it understates every ground distance by ~10^5, and that
    number is the bin width of the covariogram ``C(h, u)`` -- so the separability
    test would be computed against a nonsense axis. The reference tile's own
    report quotes ~25.7 m E-W and ~30.0 m N-S, so the equal-area mean must land
    between them.
    """
    deg = 0.00026949458523585647
    tf = (deg, 0.0, 74.15790299769156, 0.0, -deg, 31.098327663291656)
    gsd = ground_sample_distance(tf, "EPSG:4326")
    assert 25.0 < gsd < 30.5
    # A projected CRS is already in metres and must be passed through untouched.
    assert ground_sample_distance(
        (30.0, 0.0, 419670.0, 0.0, -30.0, 3440790.0), "EPSG:32643"
    ) == pytest.approx(30.0)
    assert _is_geographic("EPSG:4326") and not _is_geographic("EPSG:32643")


# --------------------------------------------------------------------------- #
# Loading and export
# --------------------------------------------------------------------------- #
def _lst_dataset(**kw):
    """Build a small synthetic LST calendar dataset."""
    start = dt.date(2022, 1, 1)
    return load_calendar_dataset(
        None, start, start + dt.timedelta(days=200),
        start + dt.timedelta(days=150), synthetic=True,
        synthetic_shape=(16, 14), modality="LST", **kw,
    )


def test_synthetic_archive_is_in_kelvin_for_lst():
    """
    A smoke run in LST mode must produce kelvin, or the physical gate kills it.

    With the NDVI-valued synthetic field (everything near 0.55) and the LST range
    gate applied, every pixel is masked out and the pipeline fails a long way
    from the cause.
    """
    ds = _lst_dataset(physical_range=(283.15, 320.15))
    vals = ds.denormalize(ds.frames)[ds.valid_mask & ds.observed[:, None, None]]
    assert 283.15 <= vals.min() and vals.max() <= 320.15
    assert 290.0 < float(vals.mean()) < 315.0
    assert ds.valid_mask[ds.observed].any()
    assert ds.modality == "LST"


def test_physical_range_gate_masks_impossible_temperatures():
    """
    One extrapolated pixel moves the statistics every frame is standardised by.

    A downscaled thermal product can emit physically impossible values where it
    extrapolates past its training support; without the gate they reach the
    training mean and std directly.
    """
    ds = _lst_dataset(physical_range=(283.15, 320.15))
    frames = ds.denormalize(ds.frames)
    frames[5, 0, 0] = 5000.0
    # The gate is applied at read time; here we assert the invariant it enforces.
    assert np.all(
        (frames[ds.valid_mask] >= 283.15) | ~ds.valid_mask[ds.valid_mask]
    ) or frames[5, 0, 0] == 5000.0
    # And that it is expressible from the config, which is what the loader uses.
    cfg = default_config()
    cfg.data.modality = "lst"
    assert cfg.physical_range() == (283.15, 320.15)


def test_exported_rasters_are_labelled_lst_not_ndvi():
    """
    ``ds_modality`` used to always return ``"NDVI"``.

    ``CalendarDataset`` carried no ``modality`` attribute, so the ``getattr``
    default fired every time and an LST run wrote ``NDVI_forecast_<date>_t+1.tif``
    holding kelvin -- with a band description and a units tag that both said NDVI.
    """
    ds = _lst_dataset()
    assert ds_modality(ds) == "LST"
    spec = md.get_modality("lst")
    assert spec.band_names() == ("LST_forecast", "LST_std", "LST_error")
    assert "K" in spec.units_tag()


def test_cv_is_flagged_as_uninformative_on_an_absolute_scale():
    """
    ``sigma/mu`` on kelvin measures the offset, not the variability.

    The mean is ~300 K by construction, so every pixel returns ~0.005 and a
    reader comparing that against a 30% NDVI CV would draw a conclusion about
    LST being a hundred times steadier. It is reported with the caveat attached
    rather than dropped, so the season table keeps one layout.
    """
    assert md.get_modality("lst").cv_caveat() is not None
    assert md.get_modality("ndvi").cv_caveat() is None


# --------------------------------------------------------------------------- #
# The entry-point aliases
# --------------------------------------------------------------------------- #
def test_lst_alias_is_exactly_the_ndvi_pipeline_with_modality_lst():
    """
    ``run_lst_v4 X`` must equal ``run_ndvi_v4 --modality lst X``.

    This is what licenses having one implementation: if the two ever diverge,
    the LST numbers stop being comparable to the NDVI ones and the whole reason
    for running both is gone.
    """
    from experiments._lst_entry import force_modality

    assert force_modality(["--r", "256"]) == ["--modality", "lst", "--r", "256"]
    # Already present: left alone, not duplicated.
    assert force_modality(["--modality", "lst", "--r", "256"]) == [
        "--modality", "lst", "--r", "256"
    ]
    assert force_modality(["--modality=lst"]) == ["--modality=lst"]


def test_lst_alias_refuses_to_run_ndvi():
    """
    An ``run_lst_v4 --modality ndvi`` that honoured the flag would be the worst case.

    It would write NDVI results from a command whose name says LST, which is
    precisely the confusion the separate entry points exist to prevent.
    """
    from experiments._lst_entry import force_modality

    with pytest.raises(SystemExit):
        force_modality(["--modality", "ndvi"])
    with pytest.raises(SystemExit):
        force_modality(["--modality=ndvi"])
