"""
GP-posterior state extraction: turning each raster into a latent weight.

This is the step the intuition document calls out as "a nontrivial inference step,
not a trivial extraction" (Sec. 2, Correction 3). For each date::

    Phi_{X_t} in R^{n_t x r}   rows Psi(x_i) over the VALID pixels of that date
    Lambda_{X_t} = Phi^T Phi + sigma_eps^2 I_r
    w_t       = Lambda^{-1} Phi^T y_t            (posterior mean)
    Sigma_w_t = sigma_eps^2 Lambda^{-1}          (posterior covariance)

This is exactly equation (10) of the DBK paper and Sec. 5 of the intuition
document: the GP reduces to Bayesian linear regression in the learned feature
space, which is what makes the whole thing ``O(n r^2)`` instead of ``O(n^3)``.

Why this path rather than the image encoder
-------------------------------------------
The encoder path ``w_t = phi_theta(o_t)`` must feed a fixed-size grid to a CNN, so
invalid pixels have to be zero-filled *before the network sees them* -- and ~52% of
this AOI's bounding box is invalid. The GP path instead simply omits invalid rows
from ``Phi_{X_t}``, which is the mathematically correct treatment of a missing
observation and requires no imputation at all.

It also yields a genuine per-date **posterior covariance**. Dates with fewer valid
pixels produce a larger ``Sigma_{w_t}``, and the Kalman update then trusts them
less. A scalar ``sigma_eps^2 I`` would discard that entirely.

Caching
-------
When the validity mask is date-invariant (a fixed field clip), ``Phi_X`` and the
Cholesky factor of ``Lambda_X`` are built **once**: per-date cost drops from
``O(n r^2)`` to ``O(n r)``. When the mask varies (cloud), ``Lambda_{X_t}`` must be
rebuilt per date, and the module says so rather than quietly doing the expensive
thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.gpstate")


@dataclass
class GPStateExtractor:
    """
    Turns frames into ``(w_t, Sigma_{w_t})`` through the DBK GP posterior.

    :ivar phi: ``(n_pixels, r)`` spatial features ``Psi(x)`` on the full grid.
    :ivar sigma_eps2: observation-noise variance.
    :ivar static_mask_flat: ``(n_pixels,)`` bool if the mask is date-invariant.
    :ivar offset: ``(n_pixels,)`` fixed field subtracted before the solve and added
        back on decode -- the training climatology (see
        :mod:`dbwm.gp.empirical_basis`). ``None`` means no offset, which is the v2
        behaviour. With an offset the weights describe the *anomaly*, which is both
        what the dynamics should model and far better conditioned for a linear
        operator than a state carrying a large static mean.
    """

    phi: np.ndarray
    sigma_eps2: float
    static_mask_flat: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None

    def __post_init__(self):
        self.phi = np.asarray(self.phi, dtype=np.float64)
        if self.offset is not None:
            self.offset = np.asarray(self.offset, dtype=np.float64).reshape(-1)
            if self.offset.shape[0] != self.phi.shape[0]:
                raise ValueError(
                    "offset has {} entries but phi has {} pixels".format(
                        self.offset.shape[0], self.phi.shape[0]
                    )
                )
        self._cache: Optional[Dict[str, np.ndarray]] = None
        if self.static_mask_flat is not None:
            m = np.asarray(self.static_mask_flat, dtype=bool)
            phi_v = self.phi[m]
            lam = phi_v.T @ phi_v + self.sigma_eps2 * np.eye(self.r)
            self._cache = {
                "mask": m,
                "phi": phi_v,
                "lam": lam,
                "chol": np.linalg.cholesky(lam),
            }
            logger.info(
                "Cached Phi_X (%d valid pixels x r=%d) and chol(Lambda_X): per-date "
                "cost is now O(n r) instead of O(n r^2).", int(m.sum()), self.r,
            )

    @property
    def r(self) -> int:
        """Basis dimension."""
        return self.phi.shape[1]

    def _solve(self, chol: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        Solve ``Lambda x = rhs`` given the Cholesky factor.

        :param chol: lower Cholesky factor of ``Lambda``.
        :param rhs: right-hand side.
        :return: the solution.
        """
        from scipy.linalg import cho_solve

        return cho_solve((chol, True), rhs)

    def solve_frame(
        self, values: np.ndarray, mask: np.ndarray, want_cov: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        GP-solve one frame for ``w_t`` and optionally ``Sigma_{w_t}``.

        :param values: ``(H, W)`` or ``(n_pixels,)`` frame values (normalised).
        :param mask: matching validity mask.
        :param want_cov: also return ``sigma_eps^2 Lambda^{-1}``.
        :return: ``(w, Sigma or None)``.
        """
        y = np.asarray(values, dtype=np.float64).reshape(-1)
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.sum() < 2:
            return np.zeros(self.r), (np.eye(self.r) * 1e6 if want_cov else None)

        if self._cache is not None and np.array_equal(m, self._cache["mask"]):
            phi_v, chol = self._cache["phi"], self._cache["chol"]
        else:
            phi_v = self.phi[m]
            lam = phi_v.T @ phi_v + self.sigma_eps2 * np.eye(self.r)
            chol = np.linalg.cholesky(lam)

        target = y[m] if self.offset is None else y[m] - self.offset[m]
        w = self._solve(chol, phi_v.T @ target)
        if not want_cov:
            return w, None
        cov = self.sigma_eps2 * self._solve(chol, np.eye(self.r))
        return w, 0.5 * (cov + cov.T)

    def solve_sequence(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        observed: Optional[np.ndarray] = None,
        want_cov: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        GP-solve a whole record.

        Unobserved dates get ``w_t = 0`` and an effectively infinite covariance, so
        that nothing downstream can mistake them for data; the observer skips them
        via the ``observed`` flag rather than assimilating a fabricated value.

        :param frames: ``(T, H, W)`` normalised values.
        :param masks: ``(T, H, W)`` validity.
        :param observed: ``(T,)`` bool; ``None`` treats every date as observed.
        :param want_cov: also return the per-date posterior covariances.
        :return: dict with ``weights`` ``(T, r)``, ``covariances`` ``(T, r, r)`` or
                 ``None``, ``n_valid`` ``(T,)`` and ``reconstruction_r2`` ``(T,)``.
        """
        frames = np.asarray(frames)
        masks = np.asarray(masks, dtype=bool)
        t_total = frames.shape[0]
        if observed is None:
            observed = np.ones(t_total, dtype=bool)
        observed = np.asarray(observed, dtype=bool)

        weights = np.zeros((t_total, self.r))
        covs = np.zeros((t_total, self.r, self.r)) if want_cov else None
        n_valid = np.zeros(t_total, dtype=int)
        r2 = np.full(t_total, np.nan)

        for t in range(t_total):
            if not observed[t]:
                if want_cov:
                    covs[t] = np.eye(self.r) * 1e6
                continue
            m = masks[t].reshape(-1)
            n_valid[t] = int(m.sum())
            w, cov = self.solve_frame(frames[t], masks[t], want_cov)
            weights[t] = w
            if want_cov:
                covs[t] = cov
            y = np.asarray(frames[t], dtype=np.float64).reshape(-1)[m]
            pred = self.phi[m] @ w
            if self.offset is not None:
                pred = pred + self.offset[m]
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot > 1e-30:
                r2[t] = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot

        obs = observed
        logger.info(
            "GP state extraction: %d/%d dates solved | valid pixels %d-%d | "
            "reconstruction R2 %.3f (median over observed dates).",
            int(obs.sum()), t_total,
            int(n_valid[obs].min()) if obs.any() else 0,
            int(n_valid[obs].max()) if obs.any() else 0,
            float(np.nanmedian(r2[obs])) if obs.any() else float("nan"),
        )
        med = float(np.nanmedian(r2[obs])) if obs.any() else 0.0
        if med < 0.5:
            logger.warning(
                "Median reconstruction R2 is only %.3f: the spatial basis Psi is not "
                "representing the field well, so w_t is a poor summary of the raster "
                "and every downstream dynamics result inherits that. Train Psi "
                "longer or raise r before interpreting the forecasts.", med,
            )
        return {
            "weights": weights,
            "covariances": covs,
            "n_valid": n_valid,
            "reconstruction_r2": r2,
        }

    def decode(self, w: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode a latent weight back to a field: ``f(x) = <w, Psi(x)>``.

        :param w: ``(r,)`` or ``(T, r)`` weights.
        :param mask: optional ``(n_pixels,)`` mask; masked-out pixels become NaN.
        :return: ``(n_pixels,)`` or ``(T, n_pixels)`` field values.
        """
        out = np.asarray(w) @ self.phi.T
        if self.offset is not None:
            out = out + self.offset
        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            out = np.where(m, out, np.nan)
        return out

    def decode_variance(self, cov: np.ndarray) -> np.ndarray:
        """
        Per-pixel predictive variance ``Psi(x)^T Sigma_w Psi(x) + sigma_eps^2``.

        :param cov: ``(r, r)`` state covariance.
        :return: ``(n_pixels,)`` predictive variances.
        """
        cov = np.asarray(cov, dtype=np.float64)
        return (
            np.einsum("nr,rs,ns->n", self.phi, cov, self.phi, optimize=True)
            + self.sigma_eps2
        )


def build_extractor(
    spatial_features: Callable[[np.ndarray], np.ndarray],
    coords: np.ndarray,
    sigma_eps2: float,
    static_mask: Optional[np.ndarray] = None,
    pixel_batch: int = 20000,
    extra_features: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
) -> GPStateExtractor:
    """
    Evaluate the spatial basis on the grid and assemble a :class:`GPStateExtractor`.

    ``Psi`` is evaluated in coordinate chunks so the ``(n, r)`` feature matrix is
    materialised once and never duplicated -- the property that keeps memory at
    ``O(n r)``.

    :param spatial_features: callable mapping ``(k, 2)`` coordinates to ``(k, r)``
        features -- typically ``model.apply(params, coords, method=model.spatial_features)``.
    :param coords: ``(n_pixels, 2)`` normalised pixel coordinates.
    :param sigma_eps2: observation-noise variance.
    :param static_mask: ``(H, W)`` or ``(n_pixels,)`` mask if date-invariant.
    :param pixel_batch: coordinates evaluated per chunk.
    :param extra_features: ``(n_pixels, q)`` empirical modes appended to ``Psi``
        (see :mod:`dbwm.gp.empirical_basis`).
    :param offset: ``(n_pixels,)`` climatology subtracted before the solve.
    :return: the assembled extractor.
    """
    coords = np.asarray(coords, dtype=np.float32)
    blocks = [
        np.asarray(spatial_features(coords[i : i + pixel_batch]))
        for i in range(0, coords.shape[0], pixel_batch)
    ]
    phi = np.concatenate(blocks, axis=0)
    if extra_features is not None:
        phi = np.concatenate([phi, np.asarray(extra_features, dtype=np.float64)], axis=1)
    flat = None if static_mask is None else np.asarray(static_mask, bool).reshape(-1)
    return GPStateExtractor(
        phi=phi, sigma_eps2=float(sigma_eps2), static_mask_flat=flat, offset=offset
    )
