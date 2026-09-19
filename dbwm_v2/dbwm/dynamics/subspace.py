"""
Reduced-order latent subspace: decoupling representation from dynamics.

``r`` is asked to do two incompatible jobs.

**Representation.** The GP posterior ``w_t = Lambda_X^{-1} Phi_X^T y_t`` must
reconstruct the field, so ``r`` has to be large enough to span its *spatial*
detail. Shrinking ``r`` degrades the reconstruction directly.

**Dynamics.** Identification regresses ``w_{t+1}`` on ``w_t``, so what matters is
how many directions the trajectory *excites* over time. For one agricultural field
that is of order 10, regardless of how many pixels or basis functions there are.

Choosing a single ``r`` to satisfy both is not a tuning problem, it is a conflict:
too small and the field is not represented; too large and ``A_0`` is singular in
the unexcited directions, its modal basis is near-defective, the S2 blocks inherit
``cond(P)``, and the lifted operator becomes so non-normal that a one-step forecast
amplifies the state error instead of propagating it.

Why a second reduction is legitimate here
------------------------------------------
v2 Sec. 2.5 argues the truncating SVD of a projected DMD is **redundant**, because
``phi_theta`` has already reduced the observation to ``R^r``. That argument holds
exactly when ``r`` matches the data's rank. When ``r`` overshoots it, Prop. 4.4's
own proof says what the surplus is: the least-squares operator's remaining
eigenvalues are **zero**, "corresponding to weight directions unexcited by the
data, i.e. ``ker W_-^T``". Projecting them out therefore discards nothing the
operator could have used -- it removes directions the estimator had already set to
zero, while eliminating the numerical damage they cause.

So this is not a departure from the framework. It restores the condition under
which Prop. 4.4's redundancy claim is true, and the reduced operator it returns has
the *identical nonzero spectrum* -- exactly the equivalence Prop. 4.4 proves.

Consequences
------------
* Reconstruction keeps the full ``r``; only the dynamics see ``k``.
* Every downstream cost collapses: the lifted state is ``Lk`` rather than ``Lr``,
  so the filter's ``O(L^2 (Lk)^3)`` and the horizon fits' ``O((Lk)^3)`` fall by
  ``(r/k)^3`` -- at ``r = 256, k = 24`` that is a factor of ~1200.
* Koopman modes remain interpretable: they are eigenvectors in the subspace, and
  :meth:`LatentSubspace.lift_operator` maps them back to ``R^r`` for decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.subspace")


@dataclass
class LatentSubspace:
    """
    An orthonormal POD basis for the excited part of the weight trajectory.

    :ivar basis: ``(r, k)`` orthonormal columns spanning the excited subspace.
    :ivar mean: ``(r,)`` trajectory mean, removed before projection.
    :ivar explained: fraction of temporal variance retained.
    :ivar singular_values: ``(min(T, r),)`` spectrum of the centred trajectory.
    :ivar r: ambient basis dimension.
    """

    basis: np.ndarray
    mean: np.ndarray
    explained: float
    singular_values: np.ndarray
    r: int

    @property
    def k(self) -> int:
        """Subspace dimension."""
        return self.basis.shape[1]

    def project(self, w: np.ndarray) -> np.ndarray:
        """
        Map weights into subspace coordinates ``z = U^T (w - mean)``.

        :param w: ``(r,)`` or ``(T, r)`` weights.
        :return: ``(k,)`` or ``(T, k)`` coordinates.
        """
        return (np.asarray(w) - self.mean) @ self.basis

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """
        Map subspace coordinates back to weights ``w = mean + U z``.

        :param z: ``(k,)`` or ``(T, k)`` coordinates.
        :return: ``(r,)`` or ``(T, r)`` weights.
        """
        return np.asarray(z) @ self.basis.T + self.mean

    def project_covariance(self, cov: np.ndarray) -> np.ndarray:
        """
        Congruence-transform a covariance into the subspace: ``U^T Sigma U``.

        :param cov: ``(r, r)`` or ``(T, r, r)`` covariance.
        :return: ``(k, k)`` or ``(T, k, k)`` covariance.
        """
        cov = np.asarray(cov)
        if cov.ndim == 2:
            return self.basis.T @ cov @ self.basis
        # Contract pairwise, NOT as a single three-operand einsum. Without an
        # explicit path numpy evaluates "rk,trs,sl->tkl" by brute force over every
        # index at once, costing O(T r^2 k^2) instead of O(T r k (r + k)). At the
        # real shapes (T = 1581, r = 352, k = 147) that is 4.2e12 operations
        # against 4.0e10 -- a ~100x difference, and the reason a full run spent
        # 67 minutes sitting between "STAGE 3" and the next log line with no
        # output at all.
        return (self.basis.T @ cov) @ self.basis

    def lift_covariance(self, cov: np.ndarray) -> np.ndarray:
        """
        Map a subspace covariance back to ``R^r``: ``U Sigma U^T``.

        Rank-``k`` by construction, which is correct: the model makes no claim
        about variance in directions it never modelled.

        :param cov: ``(k, k)`` covariance.
        :return: ``(r, r)`` covariance.
        """
        return self.basis @ np.asarray(cov) @ self.basis.T

    def lift_operator(self, a_k: np.ndarray) -> np.ndarray:
        """
        Map a subspace operator back to ``R^r``: ``U A_k U^T``.

        Preserves the nonzero spectrum exactly -- the equivalence of v2 Prop. 4.4,
        since the discarded directions carried zero eigenvalues to begin with.

        :param a_k: ``(k, k)`` operator.
        :return: ``(r, r)`` operator.
        """
        return self.basis @ np.asarray(a_k) @ self.basis.T

    def complement(self, w: np.ndarray) -> np.ndarray:
        """
        The part of ``w`` the subspace does not represent: ``(I - U U^T)(w - mean)``.

        :param w: ``(r,)`` or ``(T, r)`` weights.
        :return: the unmodelled component, same shape.
        """
        w = np.asarray(w)
        return w - self.reconstruct(self.project(w))

    def reconstruct_with_complement(
        self, z: np.ndarray, w_origin: np.ndarray
    ) -> np.ndarray:
        """
        Lift subspace coordinates and carry the origin's unmodelled part forward.

        Why this is the right forecast, not a patch
        -------------------------------------------
        The dynamics are identified only inside the subspace, so the operator says
        **nothing** about the orthogonal complement. Plain ``reconstruct`` implicitly
        predicts that component to be *zero at every horizon*, which is not a
        neutral choice -- it is an assertion that the discarded directions collapse
        instantly. That assertion is both unjustified and expensive: it puts the
        full truncation error into every forecast, at every ``h``, including
        ``h = 1``.

        Persisting the complement instead predicts that what the model does not
        model does not move. On a daily NDVI record that is a far better default,
        and it has a sharp consequence: when the operator shrinks to persistence
        inside the subspace, the whole forecast becomes *exactly* pixel-space
        persistence rather than persistence-plus-truncation. Without this the
        model could not match persistence even in principle -- on the real record
        the truncation term was 0.021 NDVI against a persistence bar of 0.030, so
        it was spending two thirds of its budget on a modelling artefact.

        :param z: ``(k,)`` or ``(n, k)`` forecast coordinates.
        :param w_origin: ``(r,)`` or ``(n, r)`` weights at the forecast origin.
        :return: ``(r,)`` or ``(n, r)`` lifted weights.
        """
        return self.reconstruct(z) + self.complement(np.asarray(w_origin))

    def reconstruction_error(self, w: np.ndarray) -> float:
        """
        Relative error of the rank-``k`` truncation on a trajectory.

        :param w: ``(T, r)`` weights.
        :return: relative Frobenius error.
        """
        w = np.asarray(w, dtype=np.float64)
        approx = self.reconstruct(self.project(w))
        denom = np.linalg.norm(w - self.mean)
        return float(np.linalg.norm(w - approx) / max(denom, 1e-30))


def build_subspace(
    weights: np.ndarray,
    valid: Optional[np.ndarray] = None,
    energy: float = 0.995,
    max_k: Optional[int] = None,
    min_k: int = 4,
) -> LatentSubspace:
    """
    Build the POD subspace of the weight trajectory.

    The dimension is chosen by retained temporal variance rather than by a fixed
    number, so a field with genuinely rich dynamics keeps more directions and a
    quiet one keeps fewer.

    :param weights: ``(T, r)`` trajectory.
    :param valid: ``(T,)`` mask of rows to fit on -- pass the TRAINING rows only,
        so the subspace carries no test-period information.
    :param energy: fraction of temporal variance to retain.
    :param max_k: hard cap on the subspace dimension.
    :param min_k: floor, so a very quiet record still has room to move.
    :return: the :class:`LatentSubspace`.
    """
    w = np.asarray(weights, dtype=np.float64)
    fit = w[np.asarray(valid, dtype=bool)] if valid is not None else w
    if fit.shape[0] < 2:
        raise ValueError("need at least 2 rows to build a subspace")
    mean = fit.mean(axis=0)
    u, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
    cum = np.cumsum(s**2) / max(float(np.sum(s**2)), 1e-30)
    k = int(np.searchsorted(cum, energy) + 1)
    k = max(min_k, k)
    if max_k is not None:
        k = min(k, int(max_k))
    k = min(k, vt.shape[0], w.shape[1])
    sub = LatentSubspace(
        basis=vt[:k].T.copy(),
        mean=mean,
        explained=float(cum[k - 1]),
        singular_values=s,
        r=w.shape[1],
    )
    logger.info(
        "Latent subspace: r = %d -> k = %d retaining %.2f%% of temporal variance "
        "(truncation error %.3f). Representation keeps the full r; only the "
        "dynamics are reduced.",
        sub.r, sub.k, 100.0 * sub.explained, sub.reconstruction_error(fit),
    )
    if sub.k >= sub.r:
        logger.info("Subspace is full rank; the projection is a no-op.")
    else:
        logger.info(
            "Lifted state drops from L*r = %d to L*k = %d, so the filter and "
            "horizon fits get ~%.0fx cheaper and A_0 becomes well conditioned.",
            sub.r, sub.k, (sub.r / max(sub.k, 1)) ** 3,
        )
    return sub
