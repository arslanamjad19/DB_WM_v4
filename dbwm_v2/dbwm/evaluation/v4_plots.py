"""
Figures for the v4 pipeline.

Covers the three the legacy v2 run produced -- forecast triptych, error-vs-time,
Koopman spectrum -- and adds the ones that only exist because of the memory /
multi-horizon extension:

* **RMSE by horizon**, with and without the weather sensor, which is the plot that
  answers "what does the ``t+1..t+6`` correction actually buy?"
* **Mori-Zwanzig memory depth**, ``||D_h||`` against ``L`` (v3 Sec. 2.2.3) -- a
  standalone physical measurement of the field's memory, independent of any
  forecasting gain.
* **Transient growth** ``||A^k||`` against ``rho^k``, which is how a non-normal
  operator that satisfies ``rho <= 1`` is caught amplifying a forecast anyway.
* **Seasonal covariance and separability**, the ``rho(h,u)`` surface whose
  departure from 1 is what the formal test measures.

Matplotlib is imported with the ``Agg`` backend so the module is safe in a
headless Colab/CI run.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dbwm.data.modality import resolve_modality  # noqa: E402
from dbwm.log_utils import configure_logger  # noqa: E402

logger = configure_logger(level="INFO", name="dbwm.plots")

_SEASON_COLOURS = {"kharif": "#c1440e", "rabi": "#1f6f8b", "zaid": "#3f7d20"}

#: Fixed display range for NDVI field panels.
#:
#: A vegetation index is physically bounded by ``[-1, 1]``, and the earlier
#: figures used that as the colour range. Over an irrigated agricultural field
#: the entire archive lives in roughly ``[0.10, 0.70]``, so ``[-1, 1]`` spends
#: about 70% of the ramp on values the scene never takes: every panel comes out
#: as one flat wash of colour and a reader cannot see the plot boundaries at all.
#: ``[0, 1]`` keeps the scale **fixed** -- which is the property that makes ``t+1``
#: and ``t+6`` comparable as a sequence -- while handing the data most of the ramp.
#: Water and bare soil can be slightly negative; those pixels clip to the bottom
#: colour, which is the correct reading for a vegetation map and is stated on the
#: colourbar rather than left implicit.
NDVI_VLIM: Tuple[float, float] = (0.0, 1.0)

#: Field colour ramp. ``RdYlGn`` is the conventional vegetation ramp -- red for
#: bare or stressed ground, green for vigorous canopy -- and is monotone in
#: lightness, so greenness is read directly off the map instead of through the
#: legend. ``YlGn`` (the previous default) has almost no contrast in the lower
#: half of its range, which is exactly where this field sits.
#:
#: This is the **NDVI** default and is kept under its historical name. Every
#: drawing routine now resolves its ramp from the modality instead (see
#: :func:`field_cmap`), because a vegetation ramp on a thermal field is not
#: merely unconventional -- it inverts the reading, since green would signal
#: "healthy" where it actually means "cool".
FIELD_CMAP = "RdYlGn"

#: Diverging ramp for the signed error panel.
ERROR_CMAP = "RdBu_r"

#: Ramp for the predictive-interval panel: sequential, bright = uncertain, so
#: the pixels the model is least sure about are the ones that stand out.
UNCERTAINTY_CMAP = "magma"

#: Colour for masked / no-data pixels. Explicit rather than left as the axis
#: background, so "outside the field clip" cannot be misread as a pale value on
#: whichever ramp the panel happens to use.
NODATA_COLOUR = "0.88"


def _cmap(name: str):
    """
    A colour map whose "bad" (NaN) colour is the explicit no-data grey.

    Left at the default, masked pixels take the axis background -- white -- which
    on a pale-topped ramp is indistinguishable from a real high value. Naming the
    colour makes "outside the field clip" a distinct visual category.

    :param name: matplotlib colour-map name.
    :return: a copy with :data:`NODATA_COLOUR` set as the bad colour.
    """
    cm = plt.get_cmap(name).copy()
    cm.set_bad(NODATA_COLOUR)
    return cm


def _save(fig, path: str, dpi: float = 150) -> str:
    """
    Write a figure and close it.

    :param fig: the figure.
    :param path: destination path.
    :param dpi: output resolution. Field panels raise this so each *data* pixel
        gets several device pixels -- a 135 x 125 tile drawn at 150 dpi in a
        4-inch axis gives ~4 device px per cell, and any anti-aliasing on top of
        that erases the single-pixel structure the maps exist to show.
    :return: the path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def value_range(
    units: str = "NDVI",
    data: Optional[np.ndarray] = None,
    vlim: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Fixed display range for a field panel, by modality.

    Every registered modality carries its own canonical span
    (:class:`~dbwm.data.modality.ModalitySpec`): NDVI ``[0, 1]``, LST
    ``[283.15, 320.15] K``. The *property* that matters is the same for both and
    is the reason the range is pinned at all -- a ``t+1`` and a ``t+6`` panel must
    share a colourbar, or a forecast that has drifted still looks fine on its own
    auto-scaled axes.

    The values differ because the fields do. Before the modality registry, an LST
    run reached this function with ``units="LST"`` and no ``data``, fell through
    to the NDVI default, and drew a 310 K field on ``[0, 1]``: every pixel
    saturates to the top colour and the panel carries no information whatsoever.

    An unregistered label (a residual field, an ablation series) still has no
    canonical range and falls back to robust percentiles of the supplied data;
    the caller is then expected to reuse the returned pair across the sequence.

    :param units: modality label or unit string, e.g. ``"NDVI"``, ``"LST"``, ``"K"``.
    :param data: values to derive a range from when the modality has no fixed one.
    :param vlim: explicit override; returned unchanged when given.
    :return: ``(vmin, vmax)``.
    """
    if vlim is not None:
        return float(vlim[0]), float(vlim[1])
    spec = resolve_modality(units, default=None)
    if spec is not None and spec.vlim is not None:
        return float(spec.vlim[0]), float(spec.vlim[1])
    if data is None:
        return NDVI_VLIM
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:  # pragma: no cover - fully masked input
        return NDVI_VLIM
    lo, hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def field_cmap(units: str = "NDVI") -> str:
    """
    Colour ramp for a field panel of this modality.

    :param units: modality label or unit string.
    :return: a matplotlib colour-map name.
    """
    spec = resolve_modality(units, default=None)
    return FIELD_CMAP if spec is None else spec.cmap


def interval_cmap(units: str = "NDVI") -> str:
    """
    Colour ramp for the predictive-interval panel of this modality.

    Chosen per modality to stay visually distinct from :func:`field_cmap`: the
    default ``magma`` sits beside NDVI's ``RdYlGn`` fine, but beside LST's
    ``inferno`` the two are near-identical and a reader glancing at a four-panel
    figure would read the uncertainty map as another temperature map.

    :param units: modality label or unit string.
    :return: a matplotlib colour-map name.
    """
    spec = resolve_modality(units, default=None)
    return UNCERTAINTY_CMAP if spec is None else spec.uncertainty_cmap


def unit_label(units: str = "NDVI") -> str:
    """
    The unit symbol printed on a colourbar or an axis, e.g. ``"K"``.

    A colourbar reading "LST" states what is drawn but not in what; for a
    temperature that is the one thing a reader has to know before comparing the
    number to anything else.

    :param units: modality label or unit string.
    :return: the unit symbol, falling back to the label itself when unregistered.
    """
    spec = resolve_modality(units, default=None)
    return str(units) if spec is None else spec.units


def _field_dpi(
    shape: Tuple[int, int],
    panel_inches: float,
    px_per_cell: float = 4.0,
    min_dpi: float = 200.0,
    max_dpi: float = 600.0,
) -> float:
    """
    Resolution that guarantees at least ``px_per_cell`` device pixels per raster cell.

    "Granular visibility of pixels" is a resolution requirement, not a styling
    one: below roughly three device pixels per cell, a single-pixel plot boundary
    is indistinguishable from a rendering artefact, and no choice of colour map
    recovers it.

    :param shape: ``(H, W)`` of the array being drawn.
    :param panel_inches: width of one panel in inches.
    :param px_per_cell: target device pixels per data cell.
    :param min_dpi: floor, so small arrays still render crisply.
    :param max_dpi: ceiling, so a large tile does not produce a 200 MB PNG.
    :return: the dpi to save at.
    """
    h, w = int(shape[0]), int(shape[1])
    needed = px_per_cell * max(h, w) / max(panel_inches, 1e-6)
    return float(min(max(min_dpi, needed), max_dpi))


def per_pixel_error_maps(errors: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-pixel RMSE / MAE / bias / ubRMSE across a stack of forecast errors.

    Aggregating over origins is what makes RMSE and MAE **different** quantities
    at a pixel. For a single date they coincide identically -- both are ``|e|`` --
    so a per-date figure that printed them as two numbers would be printing the
    same number twice. Over many origins at the same lead they separate exactly
    as they do globally: RMSE is driven by the pixel's worst days, MAE by its
    typical day, and the ratio localises where a few large misses (a harvest
    boundary, a cloud edge) dominate the score.

    :param errors: ``(n_origins, H, W)`` forecast-minus-truth, NaN where invalid.
    :return: dict of ``(H, W)`` maps: ``rmse``, ``mae``, ``bias``, ``ubrmse``,
             ``n`` (finite samples per pixel).
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim == 2:
        e = e[None]
    finite = np.isfinite(e)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(finite, e, 0.0)
        bias = np.where(n > 0, safe.sum(axis=0) / np.maximum(n, 1), np.nan)
        mse = np.where(n > 0, (safe**2).sum(axis=0) / np.maximum(n, 1), np.nan)
        mae = np.where(n > 0, np.abs(safe).sum(axis=0) / np.maximum(n, 1), np.nan)
    rmse = np.sqrt(mse)
    ub = np.sqrt(np.maximum(mse - bias**2, 0.0))
    return {"rmse": rmse, "mae": mae, "bias": bias, "ubrmse": ub, "n": n}


def pixel_extremes(
    rmse_map: np.ndarray, mae_map: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Locate the best- and worst-predicted pixels under each metric.

    Reported because a field-average ubRMSE says nothing about *where* the error
    lives, and the two failure patterns it can hide need different responses: a
    single hot pixel at a plot boundary is a registration or mixed-pixel problem,
    while a diffuse spread is a dynamics problem.

    :param rmse_map: ``(H, W)`` per-pixel RMSE (or ``|error|`` for a single date).
    :param mae_map: ``(H, W)`` per-pixel MAE; defaults to ``rmse_map``, which is
        the correct identification for a single origin.
    :return: ``{"max_rmse"|"min_rmse"|"max_mae"|"min_mae": {"row","col","value"}}``.
    """
    rmse_map = np.asarray(rmse_map, dtype=float)
    mae_map = rmse_map if mae_map is None else np.asarray(mae_map, dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    for name, arr, pick in (
        ("max_rmse", rmse_map, np.nanargmax), ("min_rmse", rmse_map, np.nanargmin),
        ("max_mae", mae_map, np.nanargmax), ("min_mae", mae_map, np.nanargmin),
    ):
        if not np.isfinite(arr).any():
            out[name] = {"row": -1, "col": -1, "value": float("nan")}
            continue
        idx = int(pick(arr))
        row, col = np.unravel_index(idx, arr.shape)
        out[name] = {"row": int(row), "col": int(col), "value": float(arr[row, col])}
    return out


def plot_forecast_triptych(
    truth: Optional[np.ndarray],
    forecast: np.ndarray,
    path: str,
    lead: int,
    valid_mask: Optional[np.ndarray] = None,
    units: str = "NDVI",
    vlim: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    target_date=None,
    origin_date=None,
    sigma: Optional[np.ndarray] = None,
    ci_level: float = 0.95,
    rmse_map: Optional[np.ndarray] = None,
    mae_map: Optional[np.ndarray] = None,
    variant: Optional[str] = None,
) -> str:
    """
    Observed / forecast / error / predictive interval for one lead time and date.

    Why the field scale is pinned, per modality
    -------------------------------------------
    Auto-scaling each figure to its own percentiles makes panels incomparable
    *between* figures: a ``t+1`` and a ``t+6`` map drawn on different limits look
    equally good even when one has drifted. So the scale is pinned -- to ``[0, 1]``
    for NDVI (rather than the physical ``[-1, 1]``, on which every panel is one
    flat wash with the plot boundaries invisible) and to ``[283.15, 320.15] K``
    for LST. :func:`value_range` resolves which, from ``units``.

    The error panel keeps a symmetric diverging scale about zero, since its job is
    to show the *sign* and spatial pattern of the disagreement, not its level.

    What the fourth panel adds
    --------------------------
    The predictive interval is the model's own claim about how wrong it expects to
    be, so putting it beside the realised error is the only way to see whether
    that claim holds *spatially*. The reported coverage is the fraction of valid
    pixels with ``|error| <= z * sigma``; a value far below the nominal level means
    the intervals are too tight, which per v3 Prop. 2.13 is what an iterated
    covariance does whenever the semigroup defect is nonzero.

    The title reports **ubRMSE and MAE** first, with RMSE and bias as context. The
    identity ``RMSE^2 = bias^2 + ubRMSE^2`` splits the error into a whole-field
    offset and genuine structural disagreement; those have different causes and
    different fixes, so a single RMSE hides which one is being fought. Beneath
    them the best- and worst-predicted pixels are named and marked, because a
    field average cannot distinguish one bad boundary from a diffuse miss.

    :param truth: ``(H, W)`` observed field, or ``None`` for a forecast whose
        target date has no raster yet -- the observed and error panels are then
        omitted rather than drawn against fabricated truth.
    :param forecast: ``(H, W)`` predicted field, in physical units.
    :param path: output path.
    :param lead: lead time in days, for the title.
    :param valid_mask: ``(H, W)`` bool; invalid pixels are blanked.
    :param units: colourbar label and modality key for :func:`value_range`.
    :param vlim: fixed ``(vmin, vmax)`` for the field panels; ``None`` takes the
        modality's canonical range via :func:`value_range`.
    :param cmap: field colour ramp; ``None`` takes the modality's via
        :func:`field_cmap`.
    :param target_date: date the forecast is valid for; printed in the title.
    :param origin_date: date the forecast was issued from; printed in the title.
    :param sigma: ``(H, W)`` per-pixel predictive standard deviation, physical
        units. Enables the interval panel and the coverage read-out.
    :param ci_level: nominal two-sided coverage for the interval panel.
    :param rmse_map: ``(H, W)`` per-pixel RMSE aggregated over origins, when the
        figure summarises more than one. Defaults to this date's ``|error|``.
    :param mae_map: ``(H, W)`` per-pixel MAE, likewise.
    :param variant: forecast variant label (``recursive`` / ``direct`` / ...).
    :return: the path written.
    """
    f = np.array(forecast, dtype=float)
    t = None if truth is None else np.array(truth, dtype=float)
    if valid_mask is not None:
        m = ~np.asarray(valid_mask, dtype=bool)
        f[m] = np.nan
        if t is not None:
            t[m] = np.nan
    err = None if t is None else f - t
    sig = None
    if sigma is not None:
        sig = np.array(sigma, dtype=float)
        if valid_mask is not None:
            sig[~np.asarray(valid_mask, dtype=bool)] = np.nan

    vmin, vmax = value_range(
        units,
        data=f if t is None else np.concatenate([f.ravel(), t.ravel()]),
        vlim=vlim,
    )
    cmap = field_cmap(units) if cmap is None else cmap
    # The colourbar states the UNIT, not the modality: a bar reading "LST" says
    # what is drawn but not in what, and for a temperature that is the one thing
    # a reader needs before comparing the number to anything else.
    ulab = unit_label(units)

    panels = []
    err_panel = -1
    if t is not None:
        panels.append((t, _panel_title("Observed", target_date), cmap, (vmin, vmax), ulab))
    panels.append(
        (f, _panel_title("Forecast t+{}".format(lead), target_date), cmap,
         (vmin, vmax), ulab)
    )
    if err is not None:
        elim = float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else 1.0
        elim = max(elim, 1e-6)
        err_panel = len(panels)
        panels.append(
            (err, "Forecast - Observed", ERROR_CMAP, (-elim, elim),
             "{} error".format(ulab))
        )
    z = float(_normal_quantile(0.5 + 0.5 * ci_level))
    if sig is not None:
        half = z * sig
        top = float(np.nanmax(half)) if np.isfinite(half).any() else 1.0
        panels.append(
            (half, "{:.0f}% interval half-width".format(100 * ci_level),
             interval_cmap(units), (0.0, max(top, 1e-6)), "+/- {}".format(ulab))
        )

    n = len(panels)
    panel_inches = 4.0
    fig, axes = plt.subplots(
        1, n, figsize=(panel_inches * n + 1.2, panel_inches + 0.9),
        constrained_layout=True, squeeze=False,
    )
    axes = axes[0]
    for ax, (data, title, cm, lim, label) in zip(axes, panels):
        im = ax.imshow(
            data, cmap=_cmap(cm), vmin=lim[0], vmax=lim[1],
            interpolation="nearest", resample=False,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    lines = [_headline(units, lead, origin_date, target_date, variant)]
    if err is not None:
        e = err[np.isfinite(err)]
        if e.size:
            rmse = float(np.sqrt(np.mean(e**2)))
            bias = float(np.mean(e))
            ubrmse = float(np.sqrt(max(rmse**2 - bias**2, 0.0)))
            d = resolve_modality(units).decimals
            lines.append(
                "ubRMSE {v[0]:.{d}f} {u}   MAE {v[1]:.{d}f} {u}      "
                "(RMSE {v[2]:.{d}f} {u}, bias {v[3]:+.{d}f} {u})".format(
                    v=(ubrmse, float(np.mean(np.abs(e))), rmse, bias), d=d, u=ulab,
                )
            )
            ext = pixel_extremes(
                np.abs(err) if rmse_map is None else rmse_map,
                np.abs(err) if mae_map is None else mae_map,
            )
            lines.append(_extreme_line(ext, rmse_map is None, decimals=d))
            if err_panel >= 0:
                _mark_extremes(axes[err_panel], ext)
    if sig is not None:
        lines.append(_coverage_line(
            err, sig, z, ci_level, decimals=resolve_modality(units).decimals,
            unit=ulab,
        ))

    fig.suptitle("\n".join(lines), fontsize=10, linespacing=1.5)
    return _save(fig, path, dpi=_field_dpi(f.shape, panel_inches))


def _panel_title(base: str, date=None) -> str:
    """
    Panel heading, with the date it refers to when one is known.

    :param base: the panel's role, e.g. ``"Observed"``.
    :param date: the date shown, or ``None``.
    :return: the heading.
    """
    return base if date is None else "{}\n{}".format(base, date)


def _headline(units, lead, origin_date, target_date, variant) -> str:
    """
    First title line: what is being shown, issued when, valid when.

    :param units: modality label.
    :param lead: lead time in steps.
    :param origin_date: issue date, or ``None``.
    :param target_date: valid date, or ``None``.
    :param variant: forecast variant label, or ``None``.
    :return: the line.
    """
    head = "{} forecast  t+{}".format(str(units).upper(), lead)
    if origin_date is not None:
        head += "   |   issued from {}".format(origin_date)
    if target_date is not None:
        head += "   valid {}".format(target_date)
    if variant:
        head += "   [{}]".format(variant)
    return head


def _extreme_line(
    ext: Dict[str, Dict[str, float]], single_date: bool, decimals: int = 4
) -> str:
    """
    Title line naming the best- and worst-predicted pixels.

    For a single date, per-pixel RMSE and MAE are the *same* statistic
    (``|e|``), and the line says so instead of printing one number twice under
    two names.

    :param ext: output of :func:`pixel_extremes`.
    :param single_date: whether the maps came from one origin.
    :param decimals: precision, from the modality -- four decimals is right for an
        NDVI error of ~0.05 and prints pure noise on an LST error of ~1 K.
    :return: the line.
    """
    def _fmt(key):
        d = ext[key]
        return "(r{:d}, c{:d}) {:.{p}f}".format(
            int(d["row"]), int(d["col"]), d["value"], p=decimals
        )

    if single_date:
        return (
            "worst pixel {}   |   best pixel {}      "
            "(single date: per-pixel RMSE = MAE = |error|)".format(
                _fmt("max_rmse"), _fmt("min_rmse")
            )
        )
    return (
        "worst RMSE {}   best RMSE {}   |   worst MAE {}   best MAE {}".format(
            _fmt("max_rmse"), _fmt("min_rmse"), _fmt("max_mae"), _fmt("min_mae")
        )
    )


def _coverage_line(err, sigma, z, ci_level, decimals: int = 4, unit: str = "") -> str:
    """
    Title line reporting the predictive interval and, if truth exists, its coverage.

    :param err: ``(H, W)`` signed error, or ``None``.
    :param sigma: ``(H, W)`` predictive standard deviation.
    :param z: normal quantile for ``ci_level``.
    :param ci_level: nominal coverage.
    :param decimals: precision, from the modality.
    :param unit: unit symbol appended to the half-width.
    :return: the line.
    """
    half = z * np.asarray(sigma, dtype=float)
    ok = np.isfinite(half)
    mean_half = float(np.mean(half[ok])) if ok.any() else float("nan")
    line = "{:.0f}% predictive interval: mean half-width +/-{:.{p}f}{u}".format(
        100 * ci_level, mean_half, p=decimals, u=" " + unit if unit else "",
    )
    if err is not None:
        both = ok & np.isfinite(err)
        if both.any():
            cov = float(np.mean(np.abs(np.asarray(err)[both]) <= half[both]))
            line += "   |   empirical coverage {:.1f}% (nominal {:.0f}%)".format(
                100 * cov, 100 * ci_level
            )
    return line


def _mark_extremes(ax, ext: Dict[str, Dict[str, float]]) -> None:
    """
    Ring the best- and worst-predicted pixels on the error panel.

    :param ax: the error axis.
    :param ext: output of :func:`pixel_extremes`.
    """
    for key, colour, marker, label in (
        ("max_rmse", "#111111", "o", "worst"),
        ("min_rmse", "#1f6f8b", "s", "best"),
    ):
        d = ext[key]
        if d["row"] < 0:
            continue
        ax.plot(
            d["col"], d["row"], marker=marker, mfc="none", mec=colour,
            ms=11, mew=1.6, ls="none", label=label,
        )
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85, handlelength=1.0)


def _normal_quantile(p: float) -> float:
    """
    Standard-normal quantile, without pulling in SciPy for one number.

    :param p: probability in ``(0, 1)``.
    :return: ``Phi^{-1}(p)``.
    """
    from math import erf, sqrt

    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def plot_pixel_error_maps(
    maps: Dict[str, np.ndarray],
    path: str,
    lead: int,
    units: str = "NDVI",
    n_origins: Optional[int] = None,
) -> str:
    """
    Per-pixel RMSE, MAE and bias at one lead, aggregated over every origin.

    This is where "which pixel is worst" is a well-posed question. On a single
    date the per-pixel RMSE and MAE are the same number; across a few hundred
    origins they separate, and the difference localises the pixels whose score is
    carried by a handful of bad days rather than by steady disagreement.

    :param maps: output of :func:`per_pixel_error_maps`.
    :param path: output path.
    :param lead: lead time, for the title.
    :param units: colourbar label.
    :param n_origins: how many origins were aggregated, for the title.
    :return: the path written.
    """
    rmse, mae, bias = maps["rmse"], maps["mae"], maps["bias"]
    if not np.isfinite(rmse).any():  # pragma: no cover - nothing scorable
        return ""
    spec = resolve_modality(units)
    ulab = unit_label(units)
    top = float(np.nanpercentile(rmse[np.isfinite(rmse)], 99))
    blim = float(np.nanmax(np.abs(bias[np.isfinite(bias)]))) if np.isfinite(bias).any() else 1.0
    ext = pixel_extremes(rmse, mae)

    panel_inches = 4.0
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * panel_inches + 1.2, panel_inches + 0.9),
        constrained_layout=True,
    )
    for ax, data, title, cm, lim in (
        (axes[0], rmse, "per-pixel RMSE", "magma_r", (0.0, max(top, 1e-6))),
        (axes[1], mae, "per-pixel MAE", "magma_r", (0.0, max(top, 1e-6))),
        (axes[2], bias, "per-pixel bias", ERROR_CMAP, (-blim, blim)),
    ):
        im = ax.imshow(data, cmap=_cmap(cm), vmin=lim[0], vmax=lim[1],
                       interpolation="nearest", resample=False)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(ulab, fontsize=9)
    _mark_extremes(axes[0], {"max_rmse": ext["max_rmse"], "min_rmse": ext["min_rmse"]})
    _mark_extremes(axes[1], {"max_rmse": ext["max_mae"], "min_rmse": ext["min_mae"]})

    fig.suptitle(
        "Per-pixel error at t+{} [{}]{}\n{}".format(
            lead, ulab,
            "" if n_origins is None else "  ({} origins)".format(n_origins),
            _extreme_line(ext, single_date=False, decimals=spec.decimals),
        ),
        fontsize=10, linespacing=1.5,
    )
    return _save(fig, path, dpi=_field_dpi(rmse.shape, panel_inches))


def plot_rmse_by_horizon(
    metrics: Dict[str, Dict[str, np.ndarray]],
    path: str,
    key: str = "pixel_rmse",
    climatology: Optional[float] = None,
    units: str = "NDVI",
) -> str:
    """
    Per-horizon error for every forecast variant.

    The climatology line is not optional context -- it is the threshold that
    decides whether the model is useful at all. A curve above it means the
    dynamics are actively harmful, which is exactly the failure a non-normal
    operator produces, and it is invisible without the reference.

    :param metrics: ``{variant: {key: (H,) array}}``.
    :param path: output path.
    :param key: which metric to plot.
    :param climatology: error of predicting the training mean, if known.
    :param units: modality label, for the y-axis unit.
    :return: the path written.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, per in sorted(metrics.items()):
        vals = np.asarray(per[key], dtype=float)
        h = np.arange(1, vals.size + 1)
        dashed = "no_weather" in name
        ax.plot(
            h, vals, marker="o", ms=4,
            ls="--" if dashed else "-",
            lw=1.6 if not dashed else 1.2,
            alpha=0.75 if dashed else 1.0,
            label=name.replace("_", " "),
        )
    if climatology is not None:
        ax.axhline(climatology, color="0.35", ls=":", lw=1.4,
                   label="climatology (predict the mean)")
    ax.set_xlabel("forecast horizon h (days)")
    ax.set_ylabel("{}  ({})".format(key.replace("_", " "), unit_label(units)))
    ax.set_title("Forecast error by horizon")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_memory_depth(sweep: Dict[str, np.ndarray], path: str) -> str:
    """
    ``||D_h||`` against memory order ``L`` -- the Mori-Zwanzig depth (v3 Sec. 2.2.3).

    Plotted in the horizon-comparable normalisation, because the raw
    ``||D_h|| / ||S A^h||`` inflates with ``h`` for any dissipative field and would
    show a spurious trend.

    :param sweep: output of ``memory_depth_sweep``.
    :param path: output path.
    :return: the path written.
    """
    orders = np.asarray(sweep["orders"])
    d = np.asarray(sweep["defect_normalized"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for h in range(d.shape[1]):
        ax1.plot(orders, d[:, h], marker="o", ms=4, label=f"h={h + 1}")
    ax1.set_xlabel("memory order L")
    ax1.set_ylabel(r"normalised $\|D_h\|$")
    ax1.set_title("Semigroup defect vs memory order\n(the elbow is the MZ depth)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    smin = np.asarray(sweep["a_last_relative_smin"])
    ax2.semilogy(orders, smin, marker="s", ms=5, color="#8c2f39")
    ax2.axhline(1e-2, color="0.4", ls="--", lw=1.2, label="Thm 2.8(i) threshold")
    ax2.set_xlabel("memory order L")
    ax2.set_ylabel(r"$s_{\min}(A_{L-1}) / \|A_0\|$")
    ax2.set_title("Observability certificate\n(a drop marks over-lagging)")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=8)
    return _save(fig, path)


def plot_koopman_spectrum(
    eigenvalues: np.ndarray, path: str, dt_days: float = 1.0, title: str = "Koopman spectrum"
) -> str:
    """
    Lifted spectrum on the unit disc, annotated with mode periods.

    :param eigenvalues: complex eigenvalues of ``A_cal``.
    :param path: output path.
    :param dt_days: sampling interval, for converting to a period in days.
    :param title: figure title.
    :return: the path written.
    """
    ev = np.asarray(eigenvalues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), color="0.6", lw=1.0, ls="--")
    ax1.scatter(ev.real, ev.imag, s=16, alpha=0.7, edgecolor="none", color="#1f6f8b")
    ax1.set_xlabel(r"Re $\lambda$")
    ax1.set_ylabel(r"Im $\lambda$")
    ax1.set_title(title + r"  ($\rho$ = {:.3f})".format(float(np.max(np.abs(ev)))))
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.25)

    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.log(ev.astype(complex) + 0j)
        period = 2 * np.pi * dt_days / np.abs(np.imag(log))
        growth = np.real(log) / dt_days
    keep = np.isfinite(period) & (period < 4000) & (np.abs(np.imag(log)) > 1e-8)
    if keep.any():
        ax2.scatter(period[keep], growth[keep], s=18, alpha=0.7, color="#3f7d20")
        ax2.set_xscale("log")
        ax2.axvline(365.25, color="#c1440e", ls="--", lw=1.2, label="annual")
        ax2.legend(fontsize=8)
    ax2.axhline(0.0, color="0.4", lw=1.0)
    ax2.set_xlabel("mode period (days)")
    ax2.set_ylabel("growth rate (1/day)")
    ax2.set_title("Koopman modes (Cor. 4.2)")
    ax2.grid(alpha=0.25, which="both")
    return _save(fig, path)


def plot_transient_growth(transient: Dict[str, object], path: str) -> str:
    """
    ``||A^k||`` against ``rho^k`` -- the growth the spectral radius does not bound.

    For a normal operator the two curves coincide. A gap is the diagnosis for a
    one-step forecast that amplifies rather than propagates.

    :param transient: output of ``transient_amplification``.
    :param path: output path.
    :return: the path written.
    """
    powers = np.asarray(transient["powers"], dtype=float)
    rho = float(transient["rho"])
    k = np.arange(1, powers.size + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(k, powers, marker="o", ms=4, label=r"$\|A_{cal}^k\|_2$ (actual)")
    ax.semilogy(k, rho**k, ls="--", color="0.4",
                label=r"$\rho^k$ (what $\rho\leq1$ promises)")
    ax.set_xlabel("step k")
    ax.set_ylabel("amplification")
    ax.set_title(
        r"Transient growth: $\rho$ = {:.3f}, peak {:.1f}$\times$ at k = {}".format(
            rho, float(transient["peak"]), int(transient["peak_step"])
        )
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_seasonal_statistics(
    reports: Dict[str, object], path: str, split: str, units: str = "NDVI"
) -> str:
    """
    Per-season mean, variance and CV maps for one split.

    The **mean** panel carries the same fixed field scale and ramp as every other
    map of this modality in the run (:func:`value_range`, :func:`field_cmap`), so
    a kharif mean and a rabi mean can be compared by eye. Variance and CV are not
    field values and keep their own sequential ramps.

    A note on the CV panel for LST: on an absolute temperature scale ``sigma/mu``
    is not a meaningful normalisation -- the mean is ~300 K by construction, so
    every pixel returns ~0.005 and the panel measures the Kelvin offset rather
    than the field's variability. It is drawn so the season figure has the same
    layout for both modalities, and the caveat is logged once by
    :func:`~dbwm.evaluation.spatiotemporal_stats.field_statistics`. Read the
    variance panel instead.

    :param reports: ``{season: SeasonReport}``.
    :param path: output path.
    :param split: split label for the title.
    :param units: modality label, which fixes the mean panel's range and ramp.
    :return: the path written.
    """
    from dbwm.data.seasons import SEASON_LABELS, SEASONS

    seasons = [s for s in SEASONS if s in reports]
    if not seasons:  # pragma: no cover - nothing to draw
        return ""
    cmap = field_cmap(units)
    vmin, vmax = value_range(
        units,
        data=np.concatenate([
            np.asarray(reports[s].statistics.mean, dtype=float).ravel()
            for s in seasons
        ]),
    )
    panel_inches = 3.6
    fig, axes = plt.subplots(
        len(seasons), 3,
        figsize=(3 * panel_inches + 1.2, panel_inches * len(seasons) + 0.8),
        squeeze=False, constrained_layout=True,
    )
    shape = np.asarray(reports[seasons[0]].statistics.mean).shape
    for row, season in enumerate(seasons):
        st = reports[season].statistics
        for col, (data, title, cmap, lim) in enumerate((
            (st.mean, r"mean $\mu(s)$  [{}]".format(unit_label(units)), cmap,
             (vmin, vmax)),
            (st.variance, r"variance $\sigma^2(s)$", "magma", (None, None)),
            (st.cv, r"CV $= \sigma/\mu$", "cividis", (None, None)),
        )):
            ax = axes[row][col]
            im = ax.imshow(
                data, cmap=_cmap(cmap), vmin=lim[0], vmax=lim[1],
                interpolation="nearest", resample=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(SEASON_LABELS[season], fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Seasonal spatiotemporal statistics -- {split}", fontsize=12)
    return _save(fig, path, dpi=_field_dpi(shape, panel_inches))


def plot_separability(covariogram, path: str, season: str, split: str) -> str:
    """
    ``C(h, u)`` and the separability ratio ``rho(h, u)``, identically 1 under H0.

    :param covariogram: a :class:`~dbwm.evaluation.spatiotemporal_stats.Covariogram`.
    :param path: output path.
    :param season: season label.
    :param split: split label.
    :return: the path written.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, u in enumerate(covariogram.lags):
        ax1.plot(covariogram.distances, covariogram.cov[:, j], marker="o", ms=3,
                 label=f"u={u}")
    ax1.set_xlabel("separation h (m)")
    ax1.set_ylabel("C(h, u)")
    ax1.set_title(f"Spatiotemporal covariance -- {season} / {split}")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    ratio = np.asarray(covariogram.ratio, dtype=float)
    finite = np.isfinite(ratio)
    lim = float(np.nanpercentile(np.abs(ratio[finite] - 1.0), 95)) if finite.any() else 1.0
    im = ax2.imshow(
        ratio, cmap="RdBu_r", vmin=1 - lim, vmax=1 + lim, aspect="auto",
        origin="lower",
        extent=[covariogram.lags[0] - 0.5, covariogram.lags[-1] + 0.5,
                0, len(covariogram.distances)],
    )
    ax2.set_xlabel("temporal lag u")
    ax2.set_ylabel("distance bin")
    ax2.set_title(r"Separability ratio $\rho(h,u)$ (= 1 under $H_0$)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    return _save(fig, path)


def plot_memory_ablation(
    rows: Sequence[Dict[str, object]],
    path: str,
    metric: str = "pixel_ubrmse",
    persistence: Optional[np.ndarray] = None,
    climatology: Optional[float] = None,
    units: str = "NDVI",
) -> str:
    """
    The memory-order ablation: forecast skill, observability and defect against ``L``.

    Three panels, because the choice of ``L`` is governed by three different
    statements in the framework and any one alone would mislead:

    **(a) Held-out error by horizon.** What the extra lags actually buy
    out-of-sample. Persistence is drawn because on a daily NDVI record it is a
    strong baseline, not a straw man, and a memory order that fails to beat it has
    not earned its parameters.

    **(b) Theorem 2.8(i).** ``s_min(A_{L-1}) / ||A_0||`` -- if the true order is
    ``L* < L`` then ``A_{L-1} -> 0``, the lifted pair loses observability, and the
    excess lags carry no recoverable state. A drop here is the signature of
    over-lagging, and it is why v3 insists the order must be *selected* rather
    than maximised.

    **(c) The Mori-Zwanzig defect.** Normalised ``||D_h||`` (v3 Def. 2.6): nonzero
    exactly to the extent that an order-``L`` linear Markov model is misspecified.
    Its elbow against ``L`` is the memory depth of the field -- a physical
    measurement that stands independently of any forecasting gain.

    :param rows: one dict per order, with ``order``, ``metric`` ``(H,)``,
        ``a_last_relative_smin``, ``defect_normalized`` ``(H,)`` and optionally
        ``blend`` and ``max_gain``.
    :param path: output path.
    :param metric: which per-horizon key to plot in panel (a).
    :param persistence: ``(H,)`` persistence error, for the reference line.
    :param climatology: scalar climatology error, for the reference line.
    :param units: modality label, for the y-axis unit.
    :return: the path written.
    """
    usable = [r for r in rows if r.get(metric) is not None]
    if not usable:  # pragma: no cover - every cell failed
        return ""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(sorted(usable, key=lambda d: int(d["order"]))):
        vals = np.asarray(r[metric], dtype=float)
        h = np.arange(1, vals.size + 1)
        label = "L = {}".format(int(r["order"]))
        if r.get("blend") is not None:
            label += "  (s = {:.2f})".format(float(r["blend"]))
        axes[0].plot(
            h, vals, marker="o", ms=4, lw=1.7,
            color=cmap(i / max(len(usable) - 1, 1)), label=label,
        )
    if persistence is not None:
        p = np.asarray(persistence, dtype=float)
        axes[0].plot(np.arange(1, p.size + 1), p, ls="--", lw=1.5, color="0.35",
                     label="persistence")
    if climatology is not None:
        axes[0].axhline(climatology, ls=":", lw=1.4, color="#8c2f39",
                        label="climatology")
    axes[0].set_xlabel("forecast horizon h (days)")
    axes[0].set_ylabel(metric.replace("_", " ") + "  ({})".format(unit_label(units)))
    axes[0].set_title("(a) Held-out skill vs memory order")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    orders = [int(r["order"]) for r in usable]
    smin = [float(r.get("a_last_relative_smin", np.nan)) for r in usable]
    axes[1].semilogy(orders, smin, marker="s", ms=6, color="#8c2f39")
    axes[1].axhline(1e-2, color="0.4", ls="--", lw=1.2, label="Thm 2.8(i) threshold")
    axes[1].set_xlabel("memory order L")
    axes[1].set_ylabel(r"$s_{\min}(A_{L-1}) / \|A_0\|_2$")
    axes[1].set_title("(b) Observability certificate\n(a drop marks over-lagging)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    any_defect = False
    for i, r in enumerate(usable):
        d = r.get("defect_normalized")
        if d is None:
            continue
        d = np.asarray(d, dtype=float)
        axes[2].plot(np.arange(1, d.size + 1), d, marker="o", ms=4,
                     color=cmap(i / max(len(usable) - 1, 1)),
                     label="L = {}".format(int(r["order"])))
        any_defect = True
    axes[2].set_xlabel("forecast horizon h")
    axes[2].set_ylabel(r"normalised $\|D_h\|$")
    axes[2].set_title("(c) Mori-Zwanzig defect\n(elbow in L = memory depth)")
    axes[2].grid(alpha=0.25)
    if any_defect:
        axes[2].legend(fontsize=8)
    return _save(fig, path)


def plot_ablation_bars(
    results: Dict[str, Dict[str, float]], path: str, metric: str = "pixel_rmse_h1"
) -> str:
    """
    Ablation-grid comparison, sorted best-first.

    :param results: ``{run label: {metric: value}}``.
    :param path: output path.
    :param metric: which metric to bar.
    :return: the path written.
    """
    items = [(k, v[metric]) for k, v in results.items() if metric in v and np.isfinite(v[metric])]
    if not items:  # pragma: no cover
        return ""
    items.sort(key=lambda kv: kv[1])
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels) + 3), 4.4))
    colours = ["#8c2f39" if "egp" in l.lower() else "#1f6f8b" for l in labels]
    ax.bar(range(len(vals)), vals, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Ablation grid (lower is better; E-GP baseline in red)")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)
