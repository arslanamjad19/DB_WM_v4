"""
Per-modality physical conventions: units, display range, ramp, thresholds.

The v4 pipeline is modality-agnostic in its *mathematics* -- the GP posterior, the
memory lift, the horizon family and the lifted filter never look at what the
pixels mean. It is **not** modality-agnostic in its *presentation and its physical
constants*, and before this module those constants were NDVI literals scattered
across the plotting, export and training code. Running ``--modality lst`` did not
fail; it produced NDVI-labelled output with an NDVI colour scale, which is worse
than failing because the numbers look plausible.

Four constants genuinely differ between a vegetation index and a land-surface
temperature, and each one is load-bearing:

**Display range.** NDVI is pinned to ``[0, 1]`` so a ``t+1`` and a ``t+6`` panel
are comparable. LST needs the same *property* -- a fixed range across the horizon
sequence -- but a completely different *value*: the Sayedanwala archive lives in
``[283.15, 320.15] K``. Drawing a 310 K field on ``[0, 1]`` saturates every pixel
to the top colour, so the map carries no information at all.

**Colour ramp.** ``RdYlGn`` is the vegetation ramp: red = stressed, green =
vigorous. On a thermal field that mapping is meaningless and actively misleading
(green would read as "healthy" where it means "cool"). LST uses ``inferno`` --
perceptually uniform, monotone in lightness, and the conventional dark = cold /
bright = hot thermal encoding.

**Teacher-forcing threshold.** ``TrainingConfig`` states the scheduled-sampling
threshold in *physical* units and the trainer divides it by the frame
normalisation, so "forced whenever the rollout is worse than tau" is literally
true. ``0.02`` is an NDVI number. Left in place for LST it is 0.02 **K**, roughly
a fortieth of the field's own standard deviation, so every step would be forced
forever and the rollout term would silently degenerate into plain reconstruction
-- with the training log still reporting that teacher forcing was "on".

**Physical validity range.** NDVI is bounded by the index definition and the
loader needs no gate. A downscaled LST product can emit physically impossible
values where the downscaling extrapolates (a few hundred K, or a residual
sentinel), and those propagate straight into the normalisation statistics. The
declared archive range gives a cheap, honest quality gate.

Everything else -- the season calendar, the weather roles, the memory order, the
horizon family -- is genuinely shared and is *not* duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModalitySpec:
    """
    The physical and presentational conventions of one observable.

    :ivar key: lowercase identifier used by ``DataConfig.modality``.
    :ivar label: uppercase label used in filenames, band names and titles.
    :ivar units: the unit symbol printed on colourbars and metric tables.
    :ivar long_name: human-readable name for logs and GeoTIFF tags.
    :ivar vlim: fixed ``(vmin, vmax)`` for field panels, or ``None`` to derive
        robust percentiles from the data. Fixing it is what makes a ``t+1`` and a
        ``t+6`` map comparable as a sequence.
    :ivar cmap: matplotlib ramp for field panels.
    :ivar uncertainty_cmap: ramp for the predictive-interval panel, chosen to be
        visually distinct from ``cmap`` so the two panels cannot be confused.
    :ivar tf_threshold: scheduled-sampling threshold in **physical** units.
    :ivar physical_range: ``(lo, hi)`` outside which a pixel is treated as invalid
        by the loader, or ``None`` to disable the gate.

        **This is a plausibility bound, not the display range**, and the two must
        not be confused -- the first release of this module set them equal and the
        gate then deleted 8.4% of the real LST archive. ``vlim`` is the scale a
        figure is drawn on, where a value outside simply clips to the end colour;
        ``physical_range`` decides what is *data*, where a value outside is
        discarded. A surface temperature of 47 degC is entirely ordinary on bare
        soil in Lahore in June and completely uninteresting as a colour, so the
        same pair of numbers cannot serve both roles.
    :ivar drop_empty_frames: treat a raster with no valid pixels as **unobserved**
        rather than as an observation that happens to be empty. See
        :func:`~dbwm.data.ndvi_dataset.load_calendar_dataset` for why one such
        frame is enough to empty the whole run.
    :ivar min_valid_fraction: a frame with a smaller fraction of valid pixels than
        this is also treated as unobserved. A frame carrying nine good pixels is
        not a usable observation of the field, and it damages the common mask in
        exactly the way an empty one does.
    :ivar cv_min_abs_mean: pixels with ``|mu|`` below this get a ``NaN`` coefficient
        of variation.
    :ivar cv_is_informative: whether ``CV = sigma/mu`` means anything for this
        observable. On an absolute temperature scale it does not (see
        :meth:`cv_caveat`), and reporting it without saying so invites a reader to
        compare a 0.5% LST "CV" against a 30% NDVI one.
    :ivar decimals: digits used when formatting a metric in these units.
    """

    key: str
    label: str
    units: str
    long_name: str
    vlim: Optional[Tuple[float, float]]
    cmap: str
    uncertainty_cmap: str
    tf_threshold: float
    physical_range: Optional[Tuple[float, float]]
    cv_min_abs_mean: float
    cv_is_informative: bool
    decimals: int = 4
    drop_empty_frames: bool = False
    min_valid_fraction: float = 0.0

    @property
    def metric_units(self) -> str:
        """
        Unit string for a metric table heading, e.g. ``"K"`` or ``"NDVI"``.

        :return: the unit symbol.
        """
        return self.units

    def format(self, value: float) -> str:
        """
        Format one metric in these units, at this modality's precision.

        NDVI errors sit around 0.05 and need four decimals; LST errors sit around
        1 K and four decimals is noise. Carrying the precision with the modality
        keeps every table readable without per-call formatting strings.

        :param value: the metric value.
        :return: the formatted string, without the unit symbol.
        """
        return "{:.{d}f}".format(float(value), d=self.decimals)

    def band_names(self) -> Tuple[str, str, str]:
        """
        GeoTIFF band descriptions for the exported forecast rasters.

        :return: ``(forecast, std, error)`` band names.
        """
        return (
            "{}_forecast".format(self.label),
            "{}_std".format(self.label),
            "{}_error".format(self.label),
        )

    def units_tag(self) -> str:
        """
        The ``DBWM_UNITS`` dataset tag written into every exported raster.

        :return: the tag text.
        """
        return "{} ({}), physical units, de-normalised".format(
            self.long_name, self.units
        )

    def cv_caveat(self) -> Optional[str]:
        """
        Why the coefficient of variation should not be read for this modality.

        :return: an explanatory string, or ``None`` when CV is informative.
        """
        if self.cv_is_informative:
            return None
        return (
            "CV = sigma/mu is not informative on an absolute temperature scale: "
            "the mean is ~300 K by construction, so every pixel returns ~0.005 "
            "and the map measures the Kelvin offset rather than the field's "
            "variability. Read sigma^2 (or sigma) directly; the CV panel is kept "
            "only so the season table has the same columns for both modalities."
        )


#: NDVI: the v4 defaults, unchanged. Listed explicitly rather than left implicit
#: so the LST entry is a peer rather than a special case bolted on beside it.
NDVI = ModalitySpec(
    key="ndvi",
    label="NDVI",
    units="NDVI",
    long_name="Normalised Difference Vegetation Index",
    # Physically NDVI is in [-1, 1], but this field never leaves ~[0.10, 0.70];
    # on [-1, 1] about 70% of the ramp goes to values the scene never takes.
    vlim=(0.0, 1.0),
    cmap="RdYlGn",
    uncertainty_cmap="magma",
    tf_threshold=0.02,
    physical_range=None,
    cv_min_abs_mean=0.05,
    cv_is_informative=True,
    decimals=4,
)

#: LST in **kelvin**, matching the archive rasters.
#:
#: The rasters are stored as absolute temperature (the reference tile runs
#: 307.7-316.7 K with ``scale=1.0, offset=0.0``), and they are left that way
#: rather than converted to degrees Celsius. The conversion is a constant offset,
#: so it changes no error metric -- RMSE, ubRMSE, MAE and bias are all identical
#: in K and in degC -- while a silent unit change between the input raster and the
#: exported one is a real hazard for anyone opening both in QGIS.
#:
#: The display range is the declared scale, 283.15-320.15 K (10-47 degC). As with
#: NDVI the point is that it is *fixed*, so a drifted t+6 map cannot hide behind
#: its own auto-scaled colourbar. Values outside it clip to the end colour, which
#: is a rendering decision and costs no data; the separate ``physical_range``
#: below is what decides validity.
LST = ModalitySpec(
    key="lst",
    label="LST",
    units="K",
    long_name="Land Surface Temperature",
    vlim=(283.15, 320.15),
    cmap="inferno",
    uncertainty_cmap="viridis",
    # 0.5 K. Two independent arguments land in the same place: it is the accuracy
    # a downscaled thermal product is usually quoted at, and it holds the NDVI
    # ratio -- 0.02 NDVI is ~0.36 of that field's spatial std (0.055), and 0.5 K
    # is ~0.3 of this field's (1.7 K on the reference tile). A threshold that is
    # a fixed fraction of the field's own variability is what keeps the fraction
    # of forced steps comparable between the two runs, which is the only way the
    # two teacher-forcing logs can be read side by side.
    tf_threshold=0.5,
    # PLAUSIBILITY, not display -- see the ``physical_range`` note on ModalitySpec.
    #
    # This was originally set equal to ``vlim`` above, on the reading that
    # 283.15-320.15 K was the archive's span. It is not: on the real record that
    # gate marked **8.4% of otherwise-valid pixels** as no-data, and they were not
    # artefacts. 320.15 K is 47 degC, which 30 m bare soil in Lahore exceeds on
    # ordinary June afternoons, and 283.15 K is 10 degC, which winter mornings go
    # below. Deleting them lost real data and -- because the discarded pixels are
    # scattered across dates -- also shredded the date-intersection mask that the
    # basis, the GP solve and every metric are computed on.
    #
    # 250-350 K (-23 to +77 degC) is instead a bound no land-surface temperature
    # on this AOI can legitimately cross, so what it catches is what it is meant
    # to catch: leftover sentinels and gross downscaling extrapolation. Override
    # with --lst-valid-range if your product needs a different one.
    physical_range=(250.0, 350.0),
    # Never binds on an absolute scale; present so the call signature is uniform.
    cv_min_abs_mean=1.0,
    cv_is_informative=False,
    # 1 mK precision. Four decimals on a field whose errors are ~1 K prints noise.
    decimals=3,
    # The LST archive contains rasters that exist but are entirely no-data (a
    # failed MoCoLSK downscaling, a fully cloudy overpass). They must not count as
    # observations. NDVI's archive has none and keeps the stricter behaviour, so
    # an empty NDVI frame stays loud rather than being silently skipped.
    drop_empty_frames=True,
    # A frame with under 1% of the bounding box valid tells you nothing about the
    # field and harms the common mask exactly as an empty one does.
    min_valid_fraction=0.01,
)

_REGISTRY: Dict[str, ModalitySpec] = {NDVI.key: NDVI, LST.key: LST}


def get_modality(key: str) -> ModalitySpec:
    """
    Look up a modality specification by key or label.

    Accepts either the config key (``"ndvi"``) or the display label (``"NDVI"``,
    and the ``units`` string the plotting functions pass around), so the plotting
    layer does not need to know which of the two it was handed.

    :param key: modality key, label, or unit string.
    :return: the :class:`ModalitySpec`.
    :raises KeyError: if the key is not registered.
    """
    k = str(key).strip().lower()
    if k in _REGISTRY:
        return _REGISTRY[k]
    # The plotting layer passes `units`, which for LST is "K" and for NDVI is
    # "NDVI"; and callers sometimes pass a decorated label ("LST error").
    for spec in _REGISTRY.values():
        if k == spec.units.lower() or k.startswith(spec.key):
            return spec
    raise KeyError(
        "Unknown modality {!r}. Registered: {}.".format(
            key, ", ".join(sorted(_REGISTRY))
        )
    )


def resolve_modality(key: str, default: ModalitySpec = NDVI) -> ModalitySpec:
    """
    Look up a modality, falling back instead of raising.

    Used on the presentation path, where an unrecognised label (a residual field,
    an ablation label) should degrade to sensible defaults rather than abort a run
    that has already done an hour of compute.

    :param key: modality key, label, or unit string.
    :param default: what to return when ``key`` is unknown.
    :return: the :class:`ModalitySpec`.
    """
    try:
        return get_modality(key)
    except KeyError:
        return default
