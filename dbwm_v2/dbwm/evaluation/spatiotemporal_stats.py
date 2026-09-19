"""
Per-season spatiotemporal statistics of the NDVI/LST field.

Four analyses, computed separately for every season (Kharif / Rabi / Zaid) on both
the training and the test split:

1. **Temporal mean field** ``mu(s)`` -- the average state at each pixel.
2. **Spatiotemporal variance** ``sigma^2(s)`` -- volatility over time, which flags
   dynamic areas such as fields under an active harvest cycle.
3. **Coefficient of variation** ``CV(s) = sigma(s) / mu(s)`` -- variance normalised
   by baseline, so a stable dense canopy and a volatile sparse one are comparable.
4. **Spatiotemporal covariance** ``C(h, u) = Cov(Z(s,t), Z(s+h, t+u))`` and a test
   of **separability**, ``C(h,u) = C_s(h) C_t(u)``.

Two traps this module is built to avoid
---------------------------------------
**CV is not safe on NDVI.** NDVI crosses zero (bare soil, water), so ``sigma/mu``
diverges wherever ``mu ~ 0`` and the resulting map is dominated by meaningless
spikes. Pixels with ``|mu|`` below a threshold are therefore masked and *the masked
fraction is reported*, and a robust ``IQR / |median|`` variant is emitted alongside.
Quietly returning the raw ratio would produce a map that looks informative and is
not.

**Centring changes what C(h,u) means.** Removing a single global mean leaves the
static spatial pattern in the covariance, so ``C(h,0)`` mostly measures the
*permanent* field layout -- field boundaries, soil type, a canal. Removing the
per-pixel temporal mean ``mu(s)`` instead measures the *dynamic* part: how
anomalies co-vary in space and time. These answer different questions, and
conflating them is the usual way this analysis goes wrong. Both are computed, and
:func:`spatiotemporal_covariance` requires the choice to be explicit.
"""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.data.seasons import SEASON_LABELS, SEASONS, season_masks
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.stats")


# --------------------------------------------------------------------------- #
# First- and second-order pixel statistics
# --------------------------------------------------------------------------- #
@dataclass
class FieldStatistics:
    """
    Per-pixel temporal statistics over one season x split subset.

    :ivar mean: ``(H, W)`` temporal mean field ``mu(s)``.
    :ivar variance: ``(H, W)`` temporal variance ``sigma^2(s)``.
    :ivar cv: ``(H, W)`` coefficient of variation, NaN where ``|mu|`` is too small.
    :ivar cv_robust: ``(H, W)`` robust variant ``IQR / |median|``.
    :ivar n_frames: number of frames contributing.
    :ivar count: ``(H, W)`` number of valid observations per pixel.
    :ivar cv_masked_fraction: fraction of valid pixels where CV was suppressed.
    :ivar valid: ``(H, W)`` pixels with enough observations to be summarised.
    """

    mean: np.ndarray
    variance: np.ndarray
    cv: np.ndarray
    cv_robust: np.ndarray
    n_frames: int
    count: np.ndarray
    cv_masked_fraction: float
    valid: np.ndarray

    def summary(self) -> Dict[str, float]:
        """Scalar digest of the maps, for tables and logs."""
        v = self.valid
        if not v.any():
            return {"n_frames": self.n_frames, "n_pixels": 0}
        return {
            "n_frames": self.n_frames,
            "n_pixels": int(v.sum()),
            "mean_of_mean": float(np.nanmean(self.mean[v])),
            "mean_of_std": float(np.nanmean(np.sqrt(self.variance[v]))),
            "median_cv": float(np.nanmedian(self.cv[v])),
            "cv_masked_fraction": self.cv_masked_fraction,
        }


def field_statistics(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    min_abs_mean: float = 0.05,
    min_observations: int = 3,
) -> FieldStatistics:
    """
    Temporal mean, variance and coefficient of variation per pixel.

    :param frames: ``(T, H, W)`` field values.
    :param valid_mask: ``(T, H, W)`` bool, ``True`` where the pixel is observed.
    :param min_abs_mean: pixels with ``|mu| <`` this get ``NaN`` CV. NDVI crosses
        zero, so an unguarded ``sigma/mu`` diverges on bare soil and water and the
        resulting map is dominated by meaningless spikes.
    :param min_observations: minimum valid frames for a pixel to be summarised.
    :return: the :class:`FieldStatistics`.
    """
    frames = np.asarray(frames, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if frames.shape != valid_mask.shape:
        raise ValueError(
            "frames {} and valid_mask {} must have the same shape".format(
                frames.shape, valid_mask.shape
            )
        )
    count = valid_mask.sum(axis=0)
    enough = count >= min_observations

    masked = np.where(valid_mask, frames, np.nan)
    # Pixels with no (or one) valid observation legitimately produce all-NaN and
    # zero-dof slices; they are excluded below via `enough`, so the warnings are
    # expected noise rather than a signal.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked, axis=0)
        variance = np.nanvar(masked, axis=0, ddof=1)
        median = np.nanmedian(masked, axis=0)
        q75 = np.nanpercentile(masked, 75, axis=0)
        q25 = np.nanpercentile(masked, 25, axis=0)

    mean = np.where(enough, mean, np.nan)
    variance = np.where(enough, variance, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.sqrt(variance) / mean
        cv_robust = (q75 - q25) / np.abs(median)
    suppress = ~enough | (np.abs(mean) < min_abs_mean)
    cv = np.where(suppress, np.nan, cv)
    cv_robust = np.where(~enough | (np.abs(median) < min_abs_mean), np.nan, cv_robust)

    n_valid = int(enough.sum())
    frac = float((suppress & enough).sum()) / max(n_valid, 1)
    if frac > 0.05:
        logger.warning(
            "CV suppressed at %.1f%% of summarisable pixels (|mu| < %.3f). On a "
            "zero-crossing field such as NDVI this is expected near bare soil and "
            "water; read the robust variant instead of the raw ratio there. On an "
            "absolute scale such as LST in kelvin it should never fire, so if it "
            "does the frames are not in the units the run thinks they are.",
            100.0 * frac, min_abs_mean,
        )
    return FieldStatistics(
        mean=mean,
        variance=variance,
        cv=cv,
        cv_robust=cv_robust,
        n_frames=int(frames.shape[0]),
        count=count,
        cv_masked_fraction=frac,
        valid=enough,
    )


# --------------------------------------------------------------------------- #
# Spatiotemporal covariance C(h, u)
# --------------------------------------------------------------------------- #
@dataclass
class Covariogram:
    """
    Empirical spatiotemporal covariance and the separability diagnostic.

    :ivar distances: ``(n_h,)`` bin-centre separations in metres (``h``).
    :ivar lags: ``(n_u,)`` temporal lags in steps (``u``).
    :ivar cov: ``(n_h, n_u)`` empirical ``C(h, u)``.
    :ivar c0: scalar ``C(0, 0)`` -- the field variance.
    :ivar spatial: ``(n_h,)`` marginal ``C(h, 0)``.
    :ivar temporal: ``(n_u,)`` marginal ``C(0, u)``.
    :ivar ratio: ``(n_h, n_u)`` separability ratio, identically 1 under separability.
    :ivar n_pairs: ``(n_h,)`` pair counts per distance bin.
    :ivar centering: which mean was removed (``"anomaly"`` or ``"global"``).
    """

    distances: np.ndarray
    lags: np.ndarray
    cov: np.ndarray
    c0: float
    spatial: np.ndarray
    temporal: np.ndarray
    ratio: np.ndarray
    n_pairs: np.ndarray
    centering: str

    def correlation(self) -> np.ndarray:
        """``C(h, u) / C(0, 0)`` -- the normalised correlogram."""
        return self.cov / max(self.c0, 1e-30)


def _pixel_coordinates(shape: Tuple[int, int], pixel_size: float) -> np.ndarray:
    """
    Metric coordinates of every pixel centre.

    :param shape: ``(H, W)``.
    :param pixel_size: ground sample distance in metres (30 for the NDVI tiles).
    :return: ``(H*W, 2)`` coordinates in metres.
    """
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([xx.ravel() * pixel_size, yy.ravel() * pixel_size], axis=-1)


def _sample_pairs(
    shape: Tuple[int, int],
    valid_flat: np.ndarray,
    lo: float,
    hi: float,
    pixel_size: float,
    n_pairs: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample pixel pairs whose separation falls in ``[lo, hi)`` metres.

    Pairs are drawn by offsetting random valid pixels rather than by enumerating
    all ``N^2`` pairs, which would be ~6.7e7 for this AOI before any lag loop.

    :param shape: ``(H, W)``.
    :param valid_flat: ``(H*W,)`` bool validity.
    :param lo: minimum separation (metres).
    :param hi: maximum separation (metres, exclusive).
    :param pixel_size: ground sample distance.
    :param n_pairs: target number of pairs.
    :param rng: random state.
    :return: ``(idx_a, idx_b, distances)``.
    """
    h, w = shape
    valid_idx = np.nonzero(valid_flat)[0]
    if valid_idx.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    # More pairs than pixels does not buy more information: the pairs start
    # sharing endpoints, so they stop being independent and any bootstrap built on
    # them understates its own variance. Cap and say so.
    if n_pairs > valid_idx.size:
        logger.debug(
            "Requested %d pairs from only %d valid pixels; capping to keep the "
            "sampled pairs approximately independent.", n_pairs, valid_idx.size,
        )
        n_pairs = int(valid_idx.size)

    lo_px = max(lo / pixel_size, 0.0)
    hi_px = hi / pixel_size
    out_a, out_b, out_d = [], [], []
    attempts, max_attempts = 0, 24
    while sum(len(x) for x in out_a) < n_pairs and attempts < max_attempts:
        attempts += 1
        k = n_pairs * 3
        base = valid_idx[rng.randint(0, valid_idx.size, size=k)]
        by, bx = base // w, base % w
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        rad = np.sqrt(rng.uniform(max(lo_px, 0.0) ** 2, max(hi_px, 1e-6) ** 2, size=k))
        oy = np.rint(by + rad * np.sin(theta)).astype(int)
        ox = np.rint(bx + rad * np.cos(theta)).astype(int)
        inside = (oy >= 0) & (oy < h) & (ox >= 0) & (ox < w)
        if not inside.any():
            continue
        base, oy, ox = base[inside], oy[inside], ox[inside]
        by, bx = base // w, base % w
        other = oy * w + ox
        ok = valid_flat[other] & (other != base)
        if not ok.any():
            continue
        base, other = base[ok], other[ok]
        d = pixel_size * np.hypot(
            (other // w) - (base // w), (other % w) - (base % w)
        )
        keep = (d >= lo) & (d < hi)
        out_a.append(base[keep])
        out_b.append(other[keep])
        out_d.append(d[keep])
    if not out_a:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    a = np.concatenate(out_a)[:n_pairs]
    b = np.concatenate(out_b)[:n_pairs]
    d = np.concatenate(out_d)[:n_pairs]
    return a, b, d


def spatiotemporal_covariance(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    centering: str,
    pixel_size: float = 30.0,
    distance_bins: Optional[Sequence[float]] = None,
    lags: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    n_pairs: int = 4000,
    seed: int = 0,
) -> Covariogram:
    """
    Empirical ``C(h, u) = Cov(Z(s,t), Z(s+h, t+u))`` on a binned separation grid.

    :param frames: ``(T, H, W)`` field values on a **uniform** time grid.
    :param valid_mask: ``(T, H, W)`` validity.
    :param centering: ``"anomaly"`` removes the per-pixel temporal mean ``mu(s)``
        and measures the *dynamic* covariance; ``"global"`` removes one scalar mean
        and leaves the static field pattern in, so ``C(h,0)`` then mostly measures
        the permanent layout. There is no default -- the choice changes the meaning
        of every number returned.
    :param pixel_size: ground sample distance in metres.
    :param distance_bins: bin **edges** in metres. Defaults to a log-ish ladder from
        one pixel to roughly a third of the tile.
    :param lags: temporal lags ``u`` in steps.
    :param n_pairs: pixel pairs sampled per distance bin.
    :param seed: RNG seed.
    :return: the :class:`Covariogram`.
    """
    if centering not in ("anomaly", "global"):
        raise ValueError(
            "centering must be 'anomaly' or 'global' (explicitly): they measure "
            "different things -- the dynamic covariance versus the total structure."
        )
    frames = np.asarray(frames, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    t, h, w = frames.shape

    # Restrict to pixels observed in every frame so a lagged product is never
    # taken between a real value and a fabricated one.
    always = valid_mask.all(axis=0)
    if always.sum() < 16:
        always = valid_mask.mean(axis=0) > 0.9
    z = np.where(valid_mask, frames, np.nan)

    if centering == "anomaly":
        with np.errstate(invalid="ignore"):
            z = z - np.nanmean(z, axis=0, keepdims=True)
    else:
        z = z - np.nanmean(z[:, always])
    z = np.where(np.isfinite(z), z, 0.0).reshape(t, h * w)

    valid_flat = always.ravel()
    c0 = float(np.mean(z[:, valid_flat] ** 2))

    if distance_bins is None:
        span = pixel_size * min(h, w) / 3.0
        distance_bins = np.unique(
            np.rint(
                np.geomspace(pixel_size, max(span, 2 * pixel_size), 9) / pixel_size
            ).astype(int)
        ) * pixel_size
    edges = np.asarray(distance_bins, dtype=float)
    lags = np.asarray(list(lags), dtype=int)

    rng = np.random.RandomState(seed)
    centres, cov_rows, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        a, b, d = _sample_pairs(
            (h, w), valid_flat, lo, hi, pixel_size, n_pairs, rng
        )
        if a.size < 32:
            continue
        za, zb = z[:, a], z[:, b]
        row = np.zeros(lags.size)
        for j, u in enumerate(lags):
            if u == 0:
                row[j] = float(np.mean(za * zb))
            elif u < t:
                # Symmetrised so C(h, u) does not depend on pair orientation.
                row[j] = 0.5 * float(
                    np.mean(za[: t - u] * zb[u:]) + np.mean(zb[: t - u] * za[u:])
                )
            else:
                row[j] = np.nan
        centres.append(float(np.mean(d)))
        cov_rows.append(row)
        counts.append(int(a.size))

    if not cov_rows:
        raise ValueError(
            "No distance bin collected enough valid pixel pairs. The AOI may be "
            "too sparsely valid for the requested bins."
        )
    cov = np.stack(cov_rows, axis=0)
    centres = np.asarray(centres)
    counts = np.asarray(counts)

    # Temporal marginal C(0, u): lag-u autocovariance at zero separation.
    temporal = np.zeros(lags.size)
    zz = z[:, valid_flat]
    for j, u in enumerate(lags):
        temporal[j] = float(np.mean(zz[: t - u] * zz[u:])) if u < t else np.nan
    spatial = cov[:, list(lags).index(0)] if 0 in lags else cov[:, 0]

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = cov * c0 / (spatial[:, None] * temporal[None, :])

    return Covariogram(
        distances=centres,
        lags=lags,
        cov=cov,
        c0=c0,
        spatial=spatial,
        temporal=temporal,
        ratio=ratio,
        n_pairs=counts,
        centering=centering,
    )


# --------------------------------------------------------------------------- #
# Formal test of separability
# --------------------------------------------------------------------------- #
@dataclass
class SeparabilityTest:
    """
    Outcome of the formal separability test.

    :ivar statistic: the chi-square statistic.
    :ivar dof: degrees of freedom (number of contrasts tested).
    :ivar p_value: upper-tail probability.
    :ivar reject: ``True`` if separability is rejected at ``alpha``.
    :ivar contrasts: ``(p,)`` estimated contrasts, zero under separability.
    :ivar max_abs_ratio_deviation: largest ``|rho(h,u) - 1|`` over the tested grid.
    :ivar alpha: significance level used.
    :ivar temporal_correlation: lag-1 temporal correlation of the field. A nearly
        white field is trivially separable, so a non-rejection at
        ``|temporal_correlation| ~ 0`` is uninformative rather than a finding.
    """

    statistic: float
    dof: int
    p_value: float
    reject: bool
    contrasts: np.ndarray
    max_abs_ratio_deviation: float
    alpha: float
    temporal_correlation: float = float("nan")

    def verdict(self) -> str:
        """One-line human-readable interpretation."""
        if self.reject:
            return (
                "Separability REJECTED (chi2 = {:.1f}, {} dof, p = {:.2e}): space and "
                "time cannot be modelled independently -- C(h,u) != C_s(h) C_t(u). "
                "A separable covariance would misstate the joint structure.".format(
                    self.statistic, self.dof, self.p_value
                )
            )
        return (
            "Separability NOT rejected (chi2 = {:.1f}, {} dof, p = {:.2f}): the "
            "record is consistent with C(h,u) = C_s(h) C_t(u), so spatial and "
            "temporal dependence may be modelled independently.".format(
                self.statistic, self.dof, self.p_value
            )
        )


def _lagged_product_series(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    centering: str,
    pixel_size: float,
    distances: Sequence[float],
    lags: Sequence[int],
    n_pairs: int,
    seed: int,
    n_groups: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute per-timestep lagged products, so a bootstrap replicate is pure slicing.

    For each distance ``i``, lag ``u`` and **pair group** ``g`` this stores

    ``m[i, u, t, g] = mean over the pairs in group g of Z(s,t) Z(s+h_i, t+u)``

    together with the temporal series ``mt[u, t]`` and ``c0[t]``. Every covariance
    the test needs is then an *average of a slice* of these arrays.

    This matters three times over. It removes pixel-pair sampling from the
    bootstrap inner loop, which is what made the naive version intractable; it lets
    a replicate average only within its own blocks, so a lagged product never spans
    a block boundary (which a concatenate-then-lag approach does silently); and by
    keeping the pairs in ``n_groups`` groups it lets the bootstrap resample
    **pairs as well as time**.

    That last point is not cosmetic. Holding the pairs fixed across replicates
    leaves spatial sampling noise out of the variance estimate entirely, which
    understates the variance of every contrast and makes the separability test
    reject far above its nominal level.

    :param frames: ``(T, H, W)`` field values.
    :param valid_mask: ``(T, H, W)`` validity.
    :param centering: ``"anomaly"`` or ``"global"``.
    :param pixel_size: ground sample distance in metres.
    :param distances: target separations in metres.
    :param lags: temporal lags, must include 0.
    :param n_pairs: pixel pairs per separation.
    :param seed: RNG seed.
    :param n_groups: number of pair groups the bootstrap can resample over.
    :return: dict with ``pair`` ``(n_h, n_u, T, n_groups)``, ``temporal``
             ``(n_u, T)``, ``c0`` ``(T,)``, ``lags`` and ``distances``.
    """
    frames = np.asarray(frames, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    t, h, w = frames.shape
    lags = np.asarray(sorted(set(list(lags) + [0])), dtype=int)

    always = valid_mask.all(axis=0)
    if always.sum() < 16:
        always = valid_mask.mean(axis=0) > 0.9
    z = np.where(valid_mask, frames, np.nan)
    # A pixel invalid across the whole subset legitimately has no mean; it is
    # excluded by `always` below, so the all-NaN slice is expected.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if centering == "anomaly":
            z = z - np.nanmean(z, axis=0, keepdims=True)
        else:
            z = z - np.nanmean(z[:, always])
    z = np.where(np.isfinite(z), z, 0.0).reshape(t, h * w)
    valid_flat = always.ravel()

    edges = _edges_around(distances, pixel_size)
    rng = np.random.RandomState(seed)
    pair_series, centres = [], []
    for i in range(len(edges) - 1):
        a, b, d = _sample_pairs(
            (h, w), valid_flat, edges[i], edges[i + 1], pixel_size, n_pairs, rng
        )
        if a.size < 32:
            continue
        za, zb = z[:, a], z[:, b]
        groups = np.array_split(np.arange(a.size), min(n_groups, a.size))
        rows = np.full((lags.size, t, len(groups)), np.nan)
        for g, idx in enumerate(groups):
            ga, gb = za[:, idx], zb[:, idx]
            for j, u in enumerate(lags):
                if u == 0:
                    rows[j, :, g] = np.mean(ga * gb, axis=1)
                elif u < t:
                    # Symmetrised so C(h, u) does not depend on pair orientation.
                    rows[j, : t - u, g] = 0.5 * (
                        np.mean(ga[: t - u] * gb[u:], axis=1)
                        + np.mean(gb[: t - u] * ga[u:], axis=1)
                    )
        pair_series.append(rows)
        centres.append(float(np.mean(d)))
    if not pair_series:
        raise ValueError("No distance bin collected enough valid pixel pairs.")

    zz = z[:, valid_flat]
    temporal = np.full((lags.size, t), np.nan)
    for j, u in enumerate(lags):
        if u < t:
            temporal[j, : t - u] = np.mean(zz[: t - u] * zz[u:], axis=1)
    return {
        "pair": np.stack(pair_series, axis=0),
        "temporal": temporal,
        "c0": np.mean(zz**2, axis=1),
        "lags": lags,
        "distances": np.asarray(centres),
    }


def _contrasts_from_blocks(
    pre: Dict[str, np.ndarray],
    blocks: Sequence[Tuple[int, int]],
    lags: Sequence[int],
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Evaluate the separability contrasts over time blocks and a pair-group subset.

    ``G(h,u) = C(h,u) - C(h,0) C(0,u) / C(0,0)``, vanishing under separability.
    Averages are taken only within blocks, so lag-``u`` products never straddle a
    boundary.

    :param pre: output of :func:`_lagged_product_series`.
    :param blocks: ``[(start, end), ...]`` half-open time intervals.
    :param lags: lags to test (0 excluded from the contrast set).
    :param groups: pair-group indices to average over (``None`` = all). Resampling
        these is what lets the bootstrap see spatial sampling noise.
    :return: ``(n_h * n_lags,)`` contrasts.
    """
    all_lags = list(pre["lags"])
    i0 = all_lags.index(0)

    def _avg(series: np.ndarray, u: int) -> float:
        vals = [series[s : max(s, e - u)] for s, e in blocks]
        vals = [v for v in vals if v.size]
        if not vals:
            return np.nan
        cat = np.concatenate(vals)
        cat = cat[np.isfinite(cat)]
        return float(cat.mean()) if cat.size else np.nan

    def _avg_pairs(series2d: np.ndarray, u: int) -> float:
        sub = series2d if groups is None else series2d[:, groups]
        # Time steps beyond T-u hold NaN by construction; averaging over an
        # all-NaN row is expected and is filtered out by `_avg`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return _avg(np.nanmean(sub, axis=1), u)

    c0 = _avg(pre["c0"], 0)
    out = []
    for i in range(pre["pair"].shape[0]):
        c_h0 = _avg_pairs(pre["pair"][i, i0], 0)
        for u in lags:
            j = all_lags.index(u)
            c_hu = _avg_pairs(pre["pair"][i, j], u)
            c_0u = _avg(pre["temporal"][j], u)
            out.append(c_hu - c_h0 * c_0u / max(c0, 1e-30))
    return np.asarray(out)


def separability_test(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    centering: str = "anomaly",
    pixel_size: float = 30.0,
    distances: Sequence[float] = (60.0, 150.0, 300.0),
    lags: Sequence[int] = (1, 2, 3),
    n_pairs: int = 3000,
    n_bootstrap: int = 200,
    block_length: int = 30,
    alpha: float = 0.05,
    seed: int = 0,
) -> SeparabilityTest:
    """
    Test ``H0: C(h,u) = C_s(h) C_t(u)`` on a grid of ``(h, u)`` pairs.

    Follows the Li-Genton-Sherman construction: form the contrasts

    ``G(h,u) = C(h,u) - C(h,0) C(0,u) / C(0,0)``

    which vanish under separability, estimate their joint covariance by a **moving
    block bootstrap over time**, and refer ``G^T Sigma^-1 G`` to a chi-square.

    The block bootstrap is essential, not a detail: NDVI is strongly autocorrelated
    in time, so an i.i.d. resample would badly understate the variance of every
    contrast and reject separability almost regardless of the truth.
    ``block_length`` must exceed the temporal decorrelation scale.

    A caveat worth stating plainly: a temporally **white** field is trivially
    separable, since ``C(h,u) = C_s(h) delta_{u0}`` factorises. Non-rejection on a
    record with no temporal correlation is therefore uninformative about spatial
    structure, and :data:`SeparabilityTest.temporal_correlation` is reported so
    that case is visible rather than mistaken for a finding.

    :param frames: ``(T, H, W)`` field values on a uniform time grid.
    :param valid_mask: ``(T, H, W)`` validity.
    :param centering: passed through; ``"anomaly"`` is the meaningful default here.
    :param pixel_size: ground sample distance in metres.
    :param distances: separations ``h`` (metres) at which to test.
    :param lags: temporal lags ``u`` at which to test.
    :param n_pairs: pixel pairs per separation.
    :param n_bootstrap: bootstrap replicates.
    :param block_length: moving-block length in time steps.
    :param alpha: significance level.
    :param seed: RNG seed.
    :return: the :class:`SeparabilityTest`.
    """
    from scipy import stats

    frames = np.asarray(frames, dtype=np.float64)
    t = frames.shape[0]
    lags = list(lags)
    block_length = int(min(block_length, max(t // 4, max(lags) + 2)))

    pre = _lagged_product_series(
        frames, valid_mask, centering, pixel_size, distances, lags, n_pairs, seed
    )
    point = _contrasts_from_blocks(pre, [(0, t)], lags)

    rng = np.random.RandomState(seed)
    n_blocks = max(int(np.ceil(t / block_length)), 2)
    reps = []
    for _ in range(n_bootstrap):
        starts = rng.randint(0, max(t - block_length, 1), size=n_blocks)
        blocks = [(int(s), int(min(s + block_length, t))) for s in starts]
        n_g = pre["pair"].shape[3]
        gsel = rng.randint(0, n_g, size=n_g)
        g = _contrasts_from_blocks(pre, blocks, lags, gsel)
        if np.all(np.isfinite(g)):
            reps.append(g)
    if len(reps) < 20:
        raise ValueError(
            "Block bootstrap produced only {} usable replicates; the subset is too "
            "short or too sparsely valid to test separability.".format(len(reps))
        )
    boot = np.stack(reps, axis=0)
    cov = np.atleast_2d(np.cov(boot, rowvar=False))
    cov = cov + 1e-10 * max(np.trace(cov) / cov.shape[0], 1e-12) * np.eye(cov.shape[0])

    stat = float(point @ np.linalg.solve(cov, point))
    dof = point.size
    p = float(stats.chi2.sf(stat, dof))

    # Separability ratio deviation, and the temporal correlation that tells you
    # whether the test had anything to detect.
    all_lags = list(pre["lags"])
    i0 = all_lags.index(0)
    c0 = float(np.nanmean(pre["c0"]))
    temporal = np.array([np.nanmean(pre["temporal"][j]) for j in range(len(all_lags))])
    dev = 0.0
    for i in range(pre["pair"].shape[0]):
        c_h0 = float(np.nanmean(pre["pair"][i, i0]))
        for u in lags:
            j = all_lags.index(u)
            c_hu = float(np.nanmean(pre["pair"][i, j]))
            denom = c_h0 * temporal[j]
            if abs(denom) > 1e-12 * abs(c0):
                dev = max(dev, abs(c_hu * c0 / denom - 1.0))
    tcorr = float(temporal[all_lags.index(lags[0])] / max(c0, 1e-30))

    out = SeparabilityTest(
        statistic=stat,
        dof=dof,
        p_value=p,
        reject=bool(p < alpha),
        contrasts=point,
        max_abs_ratio_deviation=float(dev),
        alpha=alpha,
        temporal_correlation=tcorr,
    )
    logger.info("%s", out.verdict())
    if abs(tcorr) < 0.05:
        logger.warning(
            "Lag-%d temporal correlation is only %.3f: the field is nearly white in "
            "time, which is TRIVIALLY separable. Non-rejection here says nothing "
            "about spatial structure.", lags[0], tcorr,
        )
    return out


def _edges_around(distances: Sequence[float], pixel_size: float) -> np.ndarray:
    """
    Build bin edges bracketing each requested separation.

    :param distances: target separations in metres.
    :param pixel_size: ground sample distance.
    :return: monotone edge array.
    """
    d = np.asarray(sorted(distances), dtype=float)
    half = np.maximum(pixel_size, np.diff(np.concatenate([[0.0], d])) / 2.0)
    edges = [max(d[0] - half[0], 0.0)]
    for i, x in enumerate(d):
        edges.append(x + half[i])
    return np.asarray(edges)


# --------------------------------------------------------------------------- #
# Per-season driver
# --------------------------------------------------------------------------- #
@dataclass
class SeasonReport:
    """
    Statistics for one season within one split.

    :ivar season: season key.
    :ivar split: ``"train"`` or ``"test"``.
    :ivar statistics: the :class:`FieldStatistics`.
    :ivar covariograms: ``{centering: Covariogram}``.
    :ivar separability: the :class:`SeparabilityTest`, or ``None`` if not run.
    :ivar n_frames: frames contributing.
    """

    season: str
    split: str
    statistics: FieldStatistics
    covariograms: Dict[str, Covariogram] = field(default_factory=dict)
    separability: Optional[SeparabilityTest] = None
    n_frames: int = 0


def season_reports(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    dates: Sequence[dt.date],
    split: str,
    pixel_size: float = 30.0,
    run_separability: bool = True,
    min_frames: int = 20,
    min_abs_mean: float = 0.05,
    **kwargs,
) -> Dict[str, SeasonReport]:
    """
    Compute the four analyses for every season within one split.

    Note on the time axis: a season's frames are contiguous *within* an instance
    but the instances themselves are separated by the rest of the year. Lagged
    products are therefore taken only within contiguous runs, so a "lag 1" never
    silently spans a nine-month gap between two Kharifs.

    :param frames: ``(T, H, W)`` field values for this split.
    :param valid_mask: ``(T, H, W)`` validity.
    :param dates: ``(T,)`` dates aligned to ``frames``.
    :param split: ``"train"`` or ``"test"``, used for labelling.
    :param pixel_size: ground sample distance in metres.
    :param run_separability: run the formal test (the slow part).
    :param min_frames: skip a season with fewer frames than this.
    :param min_abs_mean: CV suppression floor, from the modality. NDVI crosses
        zero so 0.05 guards a genuinely diverging ratio; LST on an absolute
        kelvin scale never approaches zero, so the guard never fires there and
        the CV column is uninformative for a different reason (see
        :meth:`~dbwm.data.modality.ModalitySpec.cv_caveat`).
    :param kwargs: forwarded to :func:`spatiotemporal_covariance`.
    :return: ``{season: SeasonReport}``.
    """
    frames = np.asarray(frames)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    masks = season_masks(list(dates))
    out: Dict[str, SeasonReport] = {}

    for season in SEASONS:
        sel = masks[season]
        n = int(sel.sum())
        if n < min_frames:
            logger.warning(
                "%s / %s: only %d frames (< %d) -- skipping.",
                SEASON_LABELS[season], split, n, min_frames,
            )
            continue
        sub, sub_mask = frames[sel], valid_mask[sel]
        stats_ = field_statistics(sub, sub_mask, min_abs_mean=min_abs_mean)

        cgs = {}
        for centering in ("anomaly", "global"):
            try:
                cgs[centering] = spatiotemporal_covariance(
                    sub, sub_mask, centering, pixel_size, **kwargs
                )
            except ValueError as exc:
                logger.warning(
                    "%s / %s: covariogram (%s) failed: %s",
                    SEASON_LABELS[season], split, centering, exc,
                )

        sep = None
        if run_separability:
            try:
                sep = separability_test(
                    sub, sub_mask, "anomaly", pixel_size, seed=0
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    "%s / %s: separability test failed: %s",
                    SEASON_LABELS[season], split, exc,
                )

        out[season] = SeasonReport(
            season=season,
            split=split,
            statistics=stats_,
            covariograms=cgs,
            separability=sep,
            n_frames=n,
        )
        logger.info(
            "%s / %s: %d frames | mean %.3f | std %.3f | median CV %.3f "
            "(CV suppressed at %.1f%% of pixels)",
            SEASON_LABELS[season], split, n,
            stats_.summary().get("mean_of_mean", float("nan")),
            stats_.summary().get("mean_of_std", float("nan")),
            stats_.summary().get("median_cv", float("nan")),
            100.0 * stats_.cv_masked_fraction,
        )
    return out


def format_season_table(reports: Dict[str, Dict[str, SeasonReport]]) -> str:
    """
    Render the per-season summary as a fixed-width table.

    :param reports: ``{split: {season: SeasonReport}}``.
    :return: multi-line string.
    """
    head = "{:<18} {:<6} {:>7} {:>9} {:>9} {:>9} {:>10} {:>12}".format(
        "season", "split", "frames", "mean", "std", "med CV", "CV masked", "separable"
    )
    lines = [head, "-" * len(head)]
    for split, per in reports.items():
        for season in SEASONS:
            rep = per.get(season)
            if rep is None:
                continue
            s = rep.statistics.summary()
            sep = (
                "n/a"
                if rep.separability is None
                else ("no" if rep.separability.reject else "yes")
            )
            lines.append(
                "{:<18} {:<6} {:>7d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.1f}% {:>12}".format(
                    SEASON_LABELS[season], split, s.get("n_pixels", 0) and rep.n_frames,
                    s.get("mean_of_mean", float("nan")),
                    s.get("mean_of_std", float("nan")),
                    s.get("median_cv", float("nan")),
                    100.0 * rep.statistics.cv_masked_fraction,
                    sep,
                )
            )
    return "\n".join(lines)
