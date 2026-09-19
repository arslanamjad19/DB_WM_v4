"""
Write forecast maps back out as georeferenced GeoTIFFs.

A latent forecast is only useful as a product once it is decoded to pixels and
carried back into the source CRS, so it can be opened beside the input rasters in
QGIS and differenced against them. Each horizon step gets three bands:

===  ===================================================================
1    ``<MODALITY>_forecast`` -- the decoded mean, in physical units
2    ``<MODALITY>_std``      -- the per-pixel predictive standard deviation
3    ``<MODALITY>_error``    -- forecast minus truth, where a truth frame exists
===  ===================================================================

``<MODALITY>`` is ``NDVI`` (index units) or ``LST`` (kelvin), taken from the
dataset rather than hard-coded, along with the ``DBWM_UNITS`` tag. LST is kept in
**kelvin**, matching the source rasters: the Celsius conversion is a constant
offset that changes no error metric, whereas a raster silently in different units
from the one beside it in QGIS is a genuine hazard.

Two details make the output trustworthy rather than merely plausible:

**Invalid pixels stay invalid.** The basis will happily extrapolate a value
anywhere, including over the 52% of the bounding box outside the field clip.
Those pixels are written as the source no-data sentinel, so a reader cannot mistake
basis extrapolation for a prediction about ground that was never observed.

**Uncertainty travels with the mean.** Band 2 is not decoration: the forecast
covariance grows with horizon (and, per v3 Prop. 2.13, the honest estimate comes
from the direct residual covariance rather than from propagating ``Q``), so a
``t+6`` map with no uncertainty band would misrepresent what the model claims.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from dbwm.data.modality import resolve_modality
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.geotiff")

def _stats(err: np.ndarray) -> Dict[str, float]:
    """
    Error statistics for one exported raster, written into its tags.

    ``ubRMSE`` is carried alongside ``RMSE`` because ``RMSE^2 = bias^2 +
    ubRMSE^2``: a reader comparing two rasters needs to know whether they differ
    by a whole-field offset or by spatial structure, and RMSE alone cannot say.

    :param err: ``(H, W)`` forecast-minus-truth, with NaN outside the clip.
    :return: dict with ``rmse``, ``bias``, ``ubrmse`` and ``mae``.
    """
    e = np.asarray(err)
    e = e[np.isfinite(e)]
    if e.size == 0:  # pragma: no cover - fully masked frame
        return {k: float("nan") for k in ("rmse", "bias", "ubrmse", "mae")}
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))
    return {
        "rmse": rmse,
        "bias": bias,
        "ubrmse": float(np.sqrt(max(rmse**2 - bias**2, 0.0))),
        "mae": float(np.mean(np.abs(e))),
    }


#: Fallback band names. The real ones come from the modality
#: (:meth:`~dbwm.data.modality.ModalitySpec.band_names`), so an LST export is
#: labelled ``LST_forecast`` / ``LST_std`` / ``LST_error`` in kelvin rather than
#: inheriting the NDVI names -- a raster whose band description disagrees with
#: its contents is worse than one with no description, because QGIS shows the
#: description and nobody checks.
BAND_NAMES = ("NDVI_forecast", "NDVI_std", "NDVI_error")


def _blocking(h: int, w: int) -> Dict[str, object]:
    """
    Tiling options GDAL will actually accept for a raster of this size.

    GDAL requires internal tile dimensions to be **multiples of 16**. Passing the
    raster's own width -- which is what this did -- is therefore invalid for
    almost every real tile: the Sayedanwala grid is 135 x 125, and 125 is not a
    multiple of 16, so ``--export-geotiff`` raised ``RasterBlockError`` before
    writing a single file.

    The fix rounds each block down to a multiple of 16 and falls back to a
    striped (untiled) layout when the raster is too small to tile at all. Tiling
    is a storage-layout choice with no effect on the pixel values, so degrading
    it is free; failing to write the raster is not.

    :param h: raster height.
    :param w: raster width.
    :return: keyword arguments for the rasterio profile.
    """
    bx = min(256, (w // 16) * 16)
    by = min(256, (h // 16) * 16)
    if bx >= 16 and by >= 16:
        return {"tiled": True, "blockxsize": bx, "blockysize": by}
    return {"tiled": False}


def write_forecast_geotiff(
    path: str,
    mean_map: np.ndarray,
    std_map: Optional[np.ndarray],
    error_map: Optional[np.ndarray],
    transform: Sequence[float],
    crs: Optional[str],
    valid_mask: Optional[np.ndarray] = None,
    nodata: float = -9999.0,
    tags: Optional[Dict[str, str]] = None,
    band_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Write one forecast step as a 3-band GeoTIFF.

    :param path: output path.
    :param mean_map: ``(H, W)`` decoded forecast in physical units.
    :param std_map: ``(H, W)`` predictive standard deviation, or ``None``.
    :param error_map: ``(H, W)`` forecast minus truth, or ``None``.
    :param transform: affine coefficients ``(a, b, c, d, e, f)``.
    :param crs: coordinate reference system string.
    :param valid_mask: ``(H, W)`` bool; ``False`` pixels are written as ``nodata``
        so basis extrapolation outside the field is never read as a prediction.
    :param nodata: no-data sentinel, matching the source rasters.
    :param tags: extra dataset tags recording provenance.
    :param band_names: ``(forecast, std, error)`` descriptions; defaults to the
        NDVI names in :data:`BAND_NAMES`.
    :return: the path written.
    """
    import rasterio
    from rasterio.transform import Affine

    names_all = tuple(band_names) if band_names else BAND_NAMES
    h, w = mean_map.shape
    bands = [mean_map]
    names = [names_all[0]]
    if std_map is not None:
        bands.append(std_map)
        names.append(names_all[1])
    if error_map is not None:
        bands.append(error_map)
        names.append(names_all[2])

    stack = np.stack(bands).astype(np.float32)
    if valid_mask is not None:
        stack = np.where(np.asarray(valid_mask, dtype=bool)[None], stack, nodata)
    stack = np.where(np.isfinite(stack), stack, nodata).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    profile = dict(
        driver="GTiff", height=h, width=w, count=len(bands), dtype="float32",
        nodata=nodata, compress="deflate",
        transform=Affine(*[float(v) for v in transform[:6]]),
        **_blocking(h, w),
    )
    if crs:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, nm in enumerate(names, start=1):
            dst.set_band_description(i, nm)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def representation_variance(
    extractor,
    weights: np.ndarray,
    frames: np.ndarray,
    valid_flat: np.ndarray,
    train_observed: np.ndarray,
    scale: float,
) -> Dict[str, np.ndarray]:
    """
    Per-pixel variance of what the basis cannot express, in physical units.

    Why the predictive band was under-dispersed by a factor of two
    --------------------------------------------------------------
    The forecast error at a pixel has three independent sources::

        var_total(s) = Psi(s)^T Sigma_w Psi(s)  +  sigma_eps^2  +  var_repr(s)
                       \\__________________/       \\_________/     \\________/
                        propagated latent          observation      REPRESENTATION
                        uncertainty                noise            (was missing)

    Only the first two were being reported. The third -- the part of the field the
    basis provably cannot reach, which is exactly what the error budget calls the
    representation floor -- was omitted, and it is the *dominant* term here.

    The consequence was measurable on the real record. At ``t+1`` the reported
    95% half-width was 0.0638, implying ``sigma = 0.0326``, against a realised
    ubRMSE of 0.0653: the model's own uncertainty was **exactly half** the error it
    actually made, a 4x under-statement in variance. Empirical coverage came out at
    58.4% against a nominal 95%, and 58% is precisely what a ``N(bias, ubRMSE^2)``
    error distribution gives against that half-width -- so the shortfall is fully
    explained by this missing term plus the bias, not by anything subtle.

    Restoring it does two things at once. Coverage becomes meaningful, and -- because
    the residual is *spatially structured*, largest at the plot boundaries and small
    parcels the smooth basis cannot resolve -- the pixels the model gets worst are
    the pixels it now says it is least sure about. A wide interval on a pixel the
    model cannot represent is the honest output; a narrow one is a false promise.

    Estimated on **training, observed** dates only, so no test-period information
    reaches the band.

    :param extractor: the :class:`~dbwm.gp.state.GPStateExtractor`.
    :param weights: ``(T, r)`` GP-solved weights.
    :param frames: ``(T, H, W)`` normalised frames.
    :param valid_flat: ``(n_pixels,)`` validity mask.
    :param train_observed: ``(T,)`` bool, training dates carrying a frame.
    :param scale: normalisation std, to return physical units.
    :return: dict with ``variance`` and ``bias`` ``(n_pixels,)``, both physical,
             plus scalar ``rms`` and ``mean_bias`` summaries.
    """
    sel = np.asarray(train_observed, dtype=bool)
    valid = np.asarray(valid_flat, dtype=bool).reshape(-1)
    n_pix = valid.shape[0]
    y = np.asarray(frames, dtype=np.float64).reshape(frames.shape[0], -1)[sel]
    recon = extractor.decode(np.asarray(weights)[sel])
    resid = (recon - y) * float(scale)

    var = np.zeros(n_pix)
    bias = np.zeros(n_pix)
    if resid.shape[0]:
        bias[valid] = resid[:, valid].mean(axis=0)
        # Mean SQUARE, not variance about the mean: the band must cover the total
        # error, and a pixel the basis renders systematically 0.2 too high is not
        # made reliable by that offset being consistent.
        var[valid] = (resid[:, valid] ** 2).mean(axis=0)
    return {
        "variance": var,
        "bias": bias,
        "rms": float(np.sqrt(var[valid].mean())) if valid.any() else float("nan"),
        "mean_bias": float(bias[valid].mean()) if valid.any() else float("nan"),
        "n_dates": int(sel.sum()),
    }


def uncertainty_band(trace, extractor, subspace, scale, repr_var=None) -> np.ndarray:
    """
    Turn a latent covariance trace into a per-pixel standard deviation.

    The full ``(r, r)`` forecast covariance is not retained -- at ``r = 256`` over
    hundreds of origins that is gigabytes that nothing else reads -- so only its
    trace survives to export time. Writing ``sqrt(trace)`` into every pixel, as a
    flat band, is wrong twice over: it overstates the magnitude by ``sqrt(k)``,
    since the trace sums ``k`` coordinate variances rather than giving one; and it
    erases the spatial structure, when the whole point of a basis model is that
    uncertainty is *not* spatially uniform -- it is large where the basis has little
    support and small where the field is well covered.

    So the trace is spread isotropically over the latent coordinates,
    ``Sigma ~= (trace / k) I``, and lifted through the basis exactly:

        ``var(s) = (trace / k) * ||Psi(s)||^2 + sigma_eps^2``

    This is exact for an isotropic covariance and, for an anisotropic one, correct
    in total power while still tracking where the basis is thin. It is an
    approximation and is tagged as such in the GeoTIFF, so nobody mistakes it for a
    calibrated interval; a calibrated band needs the full covariance, which
    ``--keep-forecast-cov`` retains.

    :param trace: scalar trace of the latent forecast covariance.
    :param extractor: the GP state extractor, for ``Psi`` and ``sigma_eps^2``.
    :param subspace: the latent subspace, or ``None`` when dynamics ran at full ``r``.
    :param scale: normalisation std, to return the band in physical units.
    :param repr_var: ``(n_pixels,)`` representation variance in **physical** units,
        from :func:`representation_variance`. Omitting it under-states the band by
        the dominant term -- on the real record by a factor of two in sigma -- so
        it is optional only for backwards compatibility, never as a default.
    :return: ``(n_pixels,)`` standard deviations in physical units.
    """
    k = subspace.k if subspace is not None else extractor.phi.shape[1]
    per_coord = max(float(trace), 0.0) / max(k, 1)
    if subspace is not None:
        # The dynamics only claim variance inside the subspace, so the isotropic
        # lift must go through U before Psi.
        psi_sub = extractor.phi @ subspace.basis          # (n, k)
        power = np.einsum("nk,nk->n", psi_sub, psi_sub)
    else:
        power = np.einsum("nr,nr->n", extractor.phi, extractor.phi)
    var = (per_coord * power + extractor.sigma_eps2) * scale**2
    if repr_var is not None:
        # Added in physical units, AFTER the scale: the representation residual is
        # measured on decoded maps, not in normalised latent coordinates.
        var = var + np.asarray(repr_var, dtype=float).reshape(-1)
    return np.sqrt(np.maximum(var, 0.0))


#: Historical private name. The band is now also drawn as the predictive-interval
#: panel of the triptych, so the figure and the raster carry the *same* number
#: rather than two independently derived ones.
_uncertainty_band = uncertainty_band


def export_forecast_maps(
    out_dir: str,
    forecast: Dict[str, np.ndarray],
    origins: Sequence[int],
    dates: Sequence[dt.date],
    extractor,
    ds,
    subspace=None,
    horizon: Optional[int] = None,
    max_origins: Optional[int] = 20,
    variant: str = "recursive",
    sigma_from: Optional[np.ndarray] = None,
    w_origin: Optional[np.ndarray] = None,
    repr_var: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Decode and write ``t+1..t+H`` forecast rasters for a set of origins.

    Filenames are ``<modality>_forecast_<origin date>_t+<h>.tif``, so a whole
    origin's horizon sorts together and is trivially matched to the input frame it
    should be compared against.

    :param out_dir: directory to write into.
    :param forecast: one variant's output from ``rolling_forecast_multi``.
    :param origins: forecast origin indices into the calendar.
    :param dates: the calendar.
    :param extractor: the :class:`~dbwm.gp.state.GPStateExtractor` for decoding.
    :param ds: the :class:`~dbwm.data.ndvi_dataset.CalendarDataset`.
    :param subspace: the :class:`~dbwm.dynamics.subspace.LatentSubspace`, if the
        dynamics ran reduced -- predictions are lifted before decoding.
    :param horizon: number of steps to export (defaults to all).
    :param max_origins: cap on origins exported, evenly spaced. A full run has
        hundreds of origins and writing every one is rarely what is wanted.
    :param variant: label recorded in the dataset tags.
    :param w_origin: ``(T, r)`` GP-solved weights. When a subspace is in use the
        origin's unmodelled component is carried forward rather than dropped, so
        the exported raster matches the scored forecast exactly.
    :param sigma_from: ``(n_origins, H)`` **trace** of the latent forecast
        covariance, i.e. ``forecast["cov_trace"]``. It is turned into a per-pixel
        band by :func:`uncertainty_band`, which lifts it through the basis rather
        than writing it flat.
    :param repr_var: ``(n_pixels,)`` representation variance in physical units.
        Without it band 2 under-states the error by its dominant term.
    :return: the list of paths written.
    """
    mean = np.asarray(forecast["mean"])
    valid = np.asarray(forecast["valid"], dtype=bool)
    n_org, hmax, _ = mean.shape
    horizon = hmax if horizon is None else min(horizon, hmax)

    sel = np.arange(n_org)
    if max_origins is not None and n_org > max_origins:
        sel = np.unique(np.linspace(0, n_org - 1, max_origins).round().astype(int))

    h_img, w_img = ds.shape
    mask_flat = ds.common_mask()
    scale, offset = float(ds.std[0]), float(ds.mean[0])
    spec = resolve_modality(ds_modality(ds))
    written: List[str] = []
    os.makedirs(out_dir, exist_ok=True)

    for i in sel:
        o = int(origins[i])
        for h in range(horizon):
            if not valid[i, h]:
                continue
            tgt = o + h + 1
            if tgt >= ds.n_steps:
                continue
            wv = mean[i, h]
            if subspace is not None:
                wv = (
                    subspace.reconstruct_with_complement(wv, w_origin[o])
                    if w_origin is not None else subspace.reconstruct(wv)
                )
            pred = extractor.decode(wv) * scale + offset

            std_map = None
            if sigma_from is not None:
                std_map = uncertainty_band(
                    sigma_from[i, h], extractor, subspace, scale, repr_var
                )

            err = None
            if ds.observed[tgt]:
                truth = ds.frames[tgt].reshape(-1) * scale + offset
                err = np.where(mask_flat, pred - truth, np.nan).reshape(h_img, w_img)

            path = os.path.join(
                out_dir,
                "{}_forecast_{}_t+{}.tif".format(
                    ds_modality(ds), dates[o].isoformat(), h + 1
                ),
            )
            write_forecast_geotiff(
                path,
                pred.reshape(h_img, w_img),
                None if std_map is None else std_map.reshape(h_img, w_img),
                err,
                ds.transform or (30.0, 0.0, 0.0, 0.0, -30.0, 0.0),
                ds.crs,
                valid_mask=mask_flat.reshape(h_img, w_img),
                band_names=spec.band_names(),
                tags={
                    "DBWM_MODALITY": spec.label,
                    "DBWM_VARIANT": variant,
                    "DBWM_ORIGIN_DATE": dates[o].isoformat(),
                    "DBWM_TARGET_DATE": dates[tgt].isoformat(),
                    "DBWM_LEAD_DAYS": h + 1,
                    "DBWM_TRUTH_AVAILABLE": bool(ds.observed[tgt]),
                    **({} if err is None else {
                        "DBWM_UBRMSE": "{:.6f}".format(_stats(err)["ubrmse"]),
                        "DBWM_MAE": "{:.6f}".format(_stats(err)["mae"]),
                        "DBWM_RMSE": "{:.6f}".format(_stats(err)["rmse"]),
                        "DBWM_BIAS": "{:+.6f}".format(_stats(err)["bias"]),
                    }),
                    "DBWM_UNITS": spec.units_tag(),
                    "DBWM_STD_METHOD": (
                        "isotropic lift of the latent covariance trace through "
                        "Psi, plus sigma_eps^2, plus the per-pixel "
                        "representation variance" if repr_var is not None else
                        "isotropic lift of the latent covariance trace through "
                        "Psi; NO representation term -- band under-states the "
                        "error, see representation_variance()"
                    ),
                },
            )
            written.append(path)

    logger.info(
        "Exported %d forecast GeoTIFFs (%d origins x up to %d horizons) to %s",
        len(written), len(sel), horizon, out_dir,
    )
    return written


def ds_modality(ds) -> str:
    """
    Modality label for filenames and band descriptions.

    :class:`~dbwm.data.ndvi_dataset.CalendarDataset` now carries ``modality``, set
    by the loader from the config. Before it did not, so this ``getattr`` always
    hit its default and an ``--modality lst`` run wrote files called
    ``NDVI_forecast_<date>_t+1.tif`` holding kelvin -- silently, since the export
    itself succeeded.

    :param ds: the dataset.
    :return: ``"NDVI"`` or ``"LST"``.
    """
    return str(getattr(ds, "modality", None) or "NDVI").upper()
