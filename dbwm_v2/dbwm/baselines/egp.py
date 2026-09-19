"""
Evolving Gaussian Processes (E-GP) / Kernel Observers -- the comparison baseline.

Reference: Kingravi, Maske, Chowdhary et al., "Kernel Observers: Systems-Theoretic
Modeling and Inference of Spatiotemporally Evolving Processes", *IEEE Control
Systems Magazine*, Feb 2021, pp. 41-65; reference implementation ``funcobspy``.

This is the ``O(M^3)`` predecessor DB-WM is claimed to replace, so the comparison
is only worth publishing if the baseline is the *method as specified* rather than
a convenient caricature. The previous implementation in this file was the latter:
it hard-coded the kernel bandwidth, placed centres uniformly at random, used only
a random measurement operator, and reported a single filtered trajectory. The
paper's actual contributions -- marginal-likelihood hyperparameter learning, the
cyclic index as the sensor lower bound, and the measurement map built from the
rational canonical structure -- were all absent.

What the method is, in the paper's own terms
--------------------------------------------
Spatiotemporal evolution of a function ``f_tau`` in an RKHS is modelled as a
**linear transition on the mixing weights** of a kernel model (eq. 3):

    w_{tau+1} = A_hat w_tau + eta_tau ,     y_tau = K w_tau + zeta_tau

with ``K`` the observation matrix whose rows are ``psi(x_i)`` at the sensing
locations. Inference is a Kalman filter on ``R^M`` (Algorithm 5). The systems-
theoretic payoff is that the number of sensors needed for observability is
governed by the **cyclic index** ``ell = max_i gamma_{lambda_i}(A_hat)`` --
"essentially independent of the dimensionality ``M``" (p. 53) -- not by ``M``.

Faithful pieces implemented here
--------------------------------
============================  ==================================================
Paper element                 Where
============================  ==================================================
Dictionary-of-atoms map (2)   :meth:`EGPBaseline.features`
Hyperparameters by NLL        :meth:`EGPBaseline._fit_hyperparameters`
  + robust median over steps   (``FeatureSpaceGenerator.return_final_params``)
Per-step weights              :meth:`EGPBaseline._solve_weights` (``solve_tikhinov``)
Matrix least squares for A    :meth:`EGPBaseline.fit`, eq. (17)
Shadedness (Definition 1)     :meth:`EGPBaseline.is_shaded`
Cyclic index (Proposition 2)  :meth:`EGPBaseline.cyclic_index`
Observability rank            :meth:`EGPBaseline.observability_rank`
k-invariant subspaces (Alg 3) :func:`k_invariant_subspaces`
Sampling locations (Alg 2)    :meth:`EGPBaseline.measurement_indices`
AKO / FKO observers (Alg 5)   :meth:`EGPBaseline.rolling_forecast`
============================  ==================================================

Two deliberate deviations from ``funcobspy``, both documented rather than silent:

**The operator regression.** ``funcobspy`` builds its regressor by stacking
``weights[0:T-1]`` with a *duplicate* of ``weights[T-1]`` and regressing the full
``weights`` on it, which pairs the last state with itself and is a defect rather
than a design. The paper specifies least squares across *successive* weights, so
that is what is fitted here: ``W_+`` on ``W_-``.

**The 'rational' measurement map.** ``funcobspy`` raises ``NotImplementedError``
for it. Since the cyclic-index bound is the paper's central theoretical claim, a
comparison that only ever used random placement could not exercise it, so
Algorithms 2 and 3 are implemented here.

Complexity is left deliberately unoptimised at ``O(M^3)``: that cost *is* the
finding, and hiding it would misrepresent the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.egp")


@dataclass
class EGPConfig:
    """
    Configuration for the E-GP / kernel-observer baseline.

    Defaults follow the paper's ocean-temperature and CFD studies: ``|C| = 300``
    centres, a Gaussian RBF kernel, hyperparameters fitted by marginal likelihood.
    """

    #: ``M``, the number of kernel centres (the paper's ``|C|``; 300-600).
    n_centers: int = 300
    #: Centre placement. "kmeans" mirrors the paper's observation that centres
    #: cluster in the most dynamic regions; "grid" and "random" are ablations.
    center_placement: str = "kmeans"
    #: Gaussian RBF bandwidth ``sigma`` (initial value if it is learned).
    lengthscale: float = 0.30
    #: Observation-noise std (initial value if it is learned).
    noise: float = 0.05
    #: Fit ``(sigma, noise)`` by marginal likelihood per step, then take the
    #: median across steps -- ``FeatureSpaceGenerator`` in the reference code.
    learn_hyperparameters: bool = True
    #: Steps sub-sampled for the hyperparameter search (it is ``O(M^3)`` per
    #: evaluation, so the paper's per-step sweep is expensive on 1200 dates).
    hyper_n_steps: int = 12
    #: Candidate bandwidths, in units of the MEDIAN NEAREST-CENTRE SPACING.
    #: Absolute values are the wrong parameterisation: what matters is whether
    #: neighbouring atoms overlap, which depends on how densely ``M`` centres
    #: cover the domain. On this AOI the spacing is 0.104, so an absolute grid
    #: starting at 0.05 starts at half a spacing -- atoms that barely touch --
    #: and the marginal likelihood duly selects it, because wider bandwidths
    #: make the Gram matrix singular (cond(K^T K) goes 1.8e1 -> 9.2e14 between
    #: 0.5 and 2 spacings) and destabilise the very likelihood being maximised.
    hyper_lengthscale_grid: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)
    #: Reject bandwidth candidates whose Gram ``K^T K`` exceeds this condition
    #: number. The marginal likelihood and the operator fit pull in OPPOSITE
    #: directions here and the conflict is intrinsic to the method, not a bug:
    #: a wider kernel fits each frame better (higher likelihood) but drives
    #: ``cond(K^T K)`` from 1.8e1 at half a centre spacing to 9.2e14 at two, and
    #: the per-step weights then stop being identifiable -- so the operator
    #: regressed on them is degenerate, its eigenvalues collapse together, and
    #: the cyclic index and observability rank become meaningless. Selecting on
    #: likelihood alone picks a model whose STATE cannot be recovered, which is
    #: not a legitimate fit. The constraint is stated and its binding is logged.
    max_gram_condition: float = 1e10
    #: Candidate noise stds for the search.
    hyper_noise_grid: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)
    #: Ridge on the operator regression.
    ridge: float = 1e-2
    #: Number of sensors ``N``. ``None`` defers to :attr:`sensor_rule`.
    n_measurements: Optional[int] = None
    #: How ``N`` is chosen when ``n_measurements`` is ``None``.
    #:
    #: ``"observable"`` (default) -- the smallest ``N`` on the search ladder that
    #:   makes ``rank(O_Upsilon) = M``. This is what the paper's own Figure 9
    #:   plots: observability is reached *above* the cyclic index, not at it.
    #: ``"cyclic_index"`` -- ``N = ell`` exactly. Reproduces Proposition 2's
    #:   LOWER BOUND literally, and is kept because it is a legitimate ablation --
    #:   but it is a bound, not a recipe. On this record it gives ``ell = 1``,
    #:   which leaves 290 of 300 centres unobserved by any sensor and the pair
    #:   unobservable at rank 52/300; the "feedback" observer then corrects a
    #:   300-dimensional state from a single pixel and is indistinguishable from
    #:   the autonomous one.
    sensor_rule: str = "observable"
    #: Subtract the per-pixel TRAINING mean before fitting and add it back on
    #: decode. Not part of either method -- it is an experiment-protocol choice,
    #: and it must be the SAME for both or the comparison measures the protocol
    #: rather than the methods. DB-WM applies it via
    #: ``BasisConfig.use_climatology``; withholding it here while granting it
    #: there is the single largest unfairness in a head-to-head table.
    use_climatology: bool = True
    #: "random" (uniform over pixels) or "rational" (Algorithms 2-3: one sensor
    #: per invariant subspace, at the centroid of that subspace's centres).
    meas_type: str = "rational"
    #: Process-noise covariance scale ``Q``.
    process_noise: float = 1e-4
    #: Initial state covariance ``P_init``.
    p_init: float = 1e-4
    seed: int = 0


# --------------------------------------------------------------------------- #
# Feature map (paper eq. 2: dictionary of atoms)
# --------------------------------------------------------------------------- #
def rbf_features(
    coords: np.ndarray, centers: np.ndarray, lengthscale: float
) -> np.ndarray:
    """
    Dictionary-of-atoms feature map ``psi_i(x) = k(x, c_i)`` with a Gaussian kernel.

    This is the paper's eq. (2), and the contrast with DB-WM is exactly here: the
    map is **stationary** and **fixed** -- it depends only on the distance to a
    centre, and nothing about it is learned from how the field evolves.

    :param coords: ``(N, d)`` query locations.
    :param centers: ``(M, d)`` centres.
    :param lengthscale: Gaussian bandwidth ``sigma``.
    :return: ``(N, M)`` feature matrix.
    """
    sq = (
        np.sum(coords**2, axis=1)[:, None]
        - 2.0 * coords @ centers.T
        + np.sum(centers**2, axis=1)[None, :]
    )
    return np.exp(-0.5 * np.maximum(sq, 0.0) / max(lengthscale, 1e-8) ** 2)


def _kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    """
    Small k-means++ used to place centres where the data actually is.

    :param x: ``(N, d)`` points.
    :param k: number of centres.
    :param seed: RNG seed.
    :param iters: Lloyd iterations.
    :return: ``(k, d)`` centres.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    centers = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(
            ((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1
        )
        total = d2.sum()
        probs = d2 / total if total > 1e-30 else np.full(n, 1.0 / n)
        centers.append(x[rng.choice(n, p=probs)])
    c = np.asarray(centers)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = x[lab == j].mean(axis=0)
    return c


# --------------------------------------------------------------------------- #
# Paper Algorithm 3: k-invariant subspace clustering
# --------------------------------------------------------------------------- #
def k_invariant_subspaces(
    a: np.ndarray, k: int, seed: int = 0, max_iter: int = 20
) -> np.ndarray:
    """
    Cluster the state coordinates into invariant subspaces (paper Algorithm 3).

    The paper calls this a "soft Jordan decomposition": a data-driven ``A_hat``
    almost never has a clean block-diagonal Jordan form, so the invariant
    subspaces have to be *found* rather than read off. The metric replaces
    Euclidean distance with dynamical influence, ``A_ij``, and the objective is
    **maximised** because the entries represent coupling rather than distance.

    Following the paper, the spectrum is first flattened -- eigenvalues strictly
    inside the unit circle are zeroed, the rest are unitised and their angles set
    to ``+-pi/4`` -- so that clustering is not dominated by the fastest-growing or
    highest-frequency modes.

    :param a: ``(M, M)`` transition operator.
    :param k: number of subspaces (the cyclic index).
    :param seed: RNG seed for initial assignment.
    :param max_iter: maximum sweeps.
    :return: ``(M,)`` integer cluster label per coordinate.
    """
    m = a.shape[0]
    k = int(max(1, min(k, m)))
    vals, vecs = np.linalg.eig(a)
    keep = np.abs(vals) >= 0.999
    ang = np.where(np.angle(vals) >= 0, np.pi / 4.0, -np.pi / 4.0)
    flat = np.where(keep, np.exp(1j * ang), 0.0)
    try:
        a_bar = np.real(vecs @ np.diag(flat) @ np.linalg.inv(vecs))
    except np.linalg.LinAlgError:  # pragma: no cover - defective basis
        a_bar = np.real(a)

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=m)
    for _ in range(max_iter):
        changed = False
        for i in range(m):
            best, best_score = labels[i], -np.inf
            for c in range(k):
                members = np.nonzero(labels == c)[0]
                members = members[members != i]
                if members.size == 0:
                    score = 0.0
                else:
                    score = (
                        np.sum(a_bar[i, members] ** 2)
                        + np.sum(a_bar[members, i] ** 2)
                    ) / float(members.size)
                if score > best_score:
                    best, best_score = c, score
            if best != labels[i]:
                labels[i] = best
                changed = True
        if not changed:
            break
    return labels


class EGPBaseline:
    """
    E-GP / kernel observer with a stationary RBF dictionary.

    :ivar cfg: an :class:`EGPConfig`.
    """

    def __init__(self, cfg: Optional[EGPConfig] = None):
        self.cfg = cfg or EGPConfig()
        self.centers: Optional[np.ndarray] = None
        self.coords: Optional[np.ndarray] = None
        self.psi: Optional[np.ndarray] = None  # (N, M)
        self.a: Optional[np.ndarray] = None  # (M, M)
        self.lengthscale = self.cfg.lengthscale
        self.noise = self.cfg.noise
        self.meas_idx: Optional[np.ndarray] = None
        self.offset: np.ndarray = np.zeros(0)
        self.spacing: float = float("nan")
        self.report: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Weights and hyperparameters
    # ------------------------------------------------------------------ #
    def _solve_weights(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Tikhonov weight solve ``w = (Psi^T Psi + noise^2 I)^{-1} Psi^T y``.

        Matches ``funcobspy``'s ``solve_tikhinov`` called from ``fit_current``,
        including its use of ``noise^2`` (not a free ridge) as the regulariser.

        :param psi: ``(N, M)`` features.
        :param y: ``(N,)`` or ``(T, N)`` targets.
        :return: ``(M,)`` or ``(T, M)`` weights.
        """
        m = psi.shape[1]
        gram = psi.T @ psi + (self.noise**2 + 1e-7) * np.eye(m)
        rhs = psi.T @ (y.T if y.ndim > 1 else y)
        out = np.linalg.solve(gram, rhs)
        return out.T if y.ndim > 1 else out

    def _marginal_likelihood(
        self, coords: np.ndarray, y: np.ndarray, ell: float, noise: float
    ) -> float:
        """
        Negative log marginal likelihood of one frame under the RBF-network prior.

        Uses the primal (capacitance) form of ``funcobspy``'s
        ``negative_log_likelihood``: with ``M << N`` the ``M x M`` system is the
        cheap way to evaluate an ``N``-dimensional Gaussian density.

        :param coords: ``(N, d)`` locations.
        :param y: ``(N,)`` values.
        :param ell: bandwidth.
        :param noise: observation-noise std.
        :return: negative log marginal likelihood.
        """
        psi = rbf_features(coords, self.centers, ell)
        n, m = psi.shape
        s2 = noise**2 + 1e-8
        kp = psi.T @ psi / s2 + np.eye(m)
        try:
            chol = np.linalg.cholesky(kp)
        except np.linalg.LinAlgError:  # pragma: no cover
            return float("inf")
        # Woodbury: (Psi Psi^T + s2 I)^{-1} y  and  log|Psi Psi^T + s2 I|
        rhs = psi.T @ y / s2
        tmp = np.linalg.solve(chol, rhs)
        alpha = (y / s2) - (psi @ np.linalg.solve(chol.T, tmp)) / s2
        logdet = 2.0 * np.sum(np.log(np.diag(chol))) + n * np.log(s2)
        return float(0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi))

    def _fit_hyperparameters(self, coords: np.ndarray, y: np.ndarray) -> None:
        """
        Fit ``(sigma, noise)`` per frame, then take the median across frames.

        This is ``FeatureSpaceGenerator.fit`` followed by
        ``return_final_params``: the paper deliberately fits a *static* feature
        space, using the per-step estimates only to form one robust global choice.
        A grid replaces the reference's L-BFGS-B because the objective is
        two-dimensional and the reference itself needed hard bounds "to avoid
        blowup".

        :param coords: ``(N, d)`` locations.
        :param y: ``(T, N)`` frames.
        """
        steps = np.linspace(0, y.shape[0] - 1, min(self.cfg.hyper_n_steps, y.shape[0]))
        # The grid is specified in units of the median nearest-centre spacing,
        # so it means the same thing for any M and any domain extent.
        d = np.sort(
            np.sqrt(((self.centers[:, None] - self.centers[None]) ** 2).sum(-1)),
            axis=1,
        )[:, 1]
        self.spacing = float(np.median(d))
        grid = [f * self.spacing for f in self.cfg.hyper_lengthscale_grid]
        # Screen the grid for identifiability before scoring likelihood on it.
        feasible, cond = [], {}
        for ell in grid:
            k = rbf_features(coords, self.centers, ell)
            cond[ell] = float(np.linalg.cond(k.T @ k))
            if cond[ell] <= self.cfg.max_gram_condition:
                feasible.append(ell)
        if not feasible:
            feasible = [min(grid, key=lambda e: cond[e])]
            logger.warning(
                "Every bandwidth on the grid gives cond(K^T K) > %.1e (best "
                "%.2e); the E-GP weights are not identifiable at any of them on "
                "this record. Falling back to the best-conditioned candidate.",
                self.cfg.max_gram_condition, min(cond.values()),
            )
        elif len(feasible) < len(grid):
            logger.info(
                "Conditioning screen: %d of %d bandwidths rejected for "
                "cond(K^T K) > %.1e (widest kept: %.2f spacings, cond %.2e).",
                len(grid) - len(feasible), len(grid), self.cfg.max_gram_condition,
                max(feasible) / self.spacing, cond[max(feasible)],
            )
        grid = feasible

        picks = []
        for t in np.unique(steps.round().astype(int)):
            best, best_nll = (self.lengthscale, self.noise), np.inf
            for ell in grid:
                for nz in self.cfg.hyper_noise_grid:
                    nll = self._marginal_likelihood(coords, y[t], ell, nz)
                    if nll < best_nll:
                        best, best_nll = (ell, nz), nll
            picks.append(best)
        arr = np.asarray(picks)
        self.lengthscale = float(np.median(arr[:, 0]))
        self.noise = float(np.median(arr[:, 1]))
        rel = self.lengthscale / max(self.spacing, 1e-12)
        logger.info(
            "E-GP hyperparameters by marginal likelihood over %d frames "
            "(median): sigma = %.4f (%.2f x the %.4f centre spacing), "
            "noise = %.3f.",
            arr.shape[0], self.lengthscale, rel, self.spacing, self.noise,
        )
        # Edge-of-grid selection is a failure signal, not a result. It says the
        # optimum lies outside the searched range, and the previous absolute grid
        # hit its LOWER edge on this record (sigma = 0.05 = the grid minimum),
        # producing atoms that barely overlap and a measurement matrix that could
        # not be shaded.
        lo = min(self.cfg.hyper_lengthscale_grid)
        hi = max(g / self.spacing for g in grid)
        if rel <= lo + 1e-9 or (rel >= hi - 1e-9
                                and hi >= max(self.cfg.hyper_lengthscale_grid) - 1e-9):
            logger.warning(
                "The selected bandwidth sits at the %s edge of the search grid "
                "(%.2f spacings, grid %.2f..%.2f). The optimum is outside the "
                "range, so this value is a boundary artefact rather than a fit -- "
                "widen EGPConfig.hyper_lengthscale_grid before trusting it.",
                "lower" if rel <= lo + 1e-9 else "upper", rel, lo, hi,
            )
        nz = min(self.cfg.hyper_noise_grid), max(self.cfg.hyper_noise_grid)
        if self.noise <= nz[0] + 1e-12 or self.noise >= nz[1] - 1e-12:
            logger.warning(
                "The selected observation noise (%.3f) sits at the %s edge of its "
                "grid (%.3f..%.3f); same caveat as the bandwidth.",
                self.noise, "lower" if self.noise <= nz[0] + 1e-12 else "upper",
                nz[0], nz[1],
            )

    # ------------------------------------------------------------------ #
    # Structural diagnostics (the paper's actual contributions)
    # ------------------------------------------------------------------ #
    def is_shaded(self, tol: float = 1e-8) -> bool:
        """
        Shadedness of the observation matrix (paper Definition 1).

        ``K`` is shaded when every centre is "seen" by at least one sensor, i.e.
        no column of ``K`` is entirely (numerically) zero. Proposition 1 makes
        this sufficient for observability when ``A_hat`` has a full-rank Jordan
        decomposition with distinct eigenvalues.

        :param tol: magnitude below which an entry counts as zero.
        :return: whether ``K`` is shaded.
        """
        k = self.psi[self.meas_idx]
        return bool(np.all(np.max(np.abs(k), axis=0) > tol))

    def cyclic_index(self, tol: float = 1e-6) -> int:
        """
        Cyclic index ``ell = max_i gamma_{lambda_i}`` (paper Proposition 2).

        This is the paper's headline bound: the minimum number of sensing
        locations is the largest geometric multiplicity of an eigenvalue of
        ``A_hat``, and it is *essentially independent of* ``M``. Eigenvalues are
        grouped at tolerance ``tol`` because a data-driven operator never has
        exactly repeated eigenvalues.

        :param tol: relative tolerance for grouping eigenvalues.
        :return: the cyclic index.
        """
        vals = np.linalg.eigvals(self.a)
        scale = max(float(np.max(np.abs(vals))), 1e-12)
        unassigned, groups = list(range(len(vals))), []
        while unassigned:
            i = unassigned.pop(0)
            grp = [i]
            for j in list(unassigned):
                if abs(vals[i] - vals[j]) / scale < tol:
                    grp.append(j)
                    unassigned.remove(j)
            groups.append(grp)
        best = 1
        for grp in groups:
            lam = np.mean(vals[grp])
            gm = self.a.shape[0] - np.linalg.matrix_rank(
                self.a - lam * np.eye(self.a.shape[0]), tol=1e-8
            )
            best = max(best, int(gm))
        return max(best, 1)

    def observability_rank(self, n_steps: Optional[int] = None) -> int:
        """
        Rank of the generalised observability matrix ``O_Upsilon`` (paper p. 48).

        The system is observable iff this equals ``M``. Built over
        ``Upsilon = {0, ..., n_steps-1}``.

        :param n_steps: number of time instances; defaults to ``M``.
        :return: ``rank(O_Upsilon)``.
        """
        m = self.a.shape[0]
        n_steps = m if n_steps is None else n_steps
        k = self.psi[self.meas_idx]
        rows, power = [], np.eye(m)
        for _ in range(n_steps):
            rows.append(k @ power)
            power = power @ self.a
            if sum(r.shape[0] for r in rows) > 4 * m:
                break
        return int(np.linalg.matrix_rank(np.vstack(rows), tol=1e-8))

    def measurement_indices(self, n_meas: int) -> np.ndarray:
        """
        Choose sensing locations (paper Remark 2 / Algorithm 2).

        ``"rational"`` follows the paper: partition the centres into invariant
        subspaces with Algorithm 3, then place one sensor per subspace at the
        centroid of that subspace's centres -- which the paper notes is
        sufficient for radially symmetric kernels ("the centroid of the convex
        hull of ``C^(i)``"). ``"random"`` is the reference implementation's only
        option and is kept as the ablation.

        :param n_meas: number of sensors ``N``.
        :return: ``(N,)`` pixel indices into ``coords``.
        """
        rng = np.random.default_rng(self.cfg.seed)
        n_pix = self.coords.shape[0]
        n_meas = int(np.clip(n_meas, 1, n_pix))
        if self.cfg.meas_type == "random" or self.a is None:
            return rng.choice(n_pix, size=n_meas, replace=False)

        labels = k_invariant_subspaces(self.a, n_meas, seed=self.cfg.seed)
        idx: List[int] = []
        for c in range(n_meas):
            members = np.nonzero(labels == c)[0]
            if members.size == 0:
                continue
            centroid = self.centers[members].mean(axis=0)
            order = np.argsort(((self.coords - centroid) ** 2).sum(axis=1))
            for cand in order:
                if int(cand) not in idx:
                    idx.append(int(cand))
                    break
        while len(idx) < n_meas:  # subspaces can come back empty
            cand = int(rng.integers(n_pix))
            if cand not in idx:
                idx.append(cand)
        return np.asarray(idx[:n_meas])

    def choose_sensor_count(self) -> Dict[str, object]:
        """
        Pick ``N`` so the pair is actually observable (paper Figure 9).

        Proposition 2 gives the cyclic index ``ell`` as a **lower bound** on the
        number of sensing locations. It is not the number to use, and treating it
        as one is how this baseline was previously crippled: on the real record
        ``ell = 1``, which leaves 290 of 300 centres unseen by any sensor, gives
        ``rank(O) = 52/300``, and reduces the "feedback" observer to correcting a
        300-dimensional state from a single pixel -- which is why AKO and FKO came
        out indistinguishable (0.564 vs 0.559).

        The paper's own Figure 9(b) plots exactly this: observability under random
        placement is reached *well above* ``ell`` and approaches 1 gradually. So
        the ladder below searches upward for the smallest ``N`` that achieves full
        rank, reporting ``ell`` alongside as the theoretical bound it is.

        :return: dict with the chosen ``n``, the ladder tried, and the rank found.
        """
        m = self.a.shape[0]
        ell = self.cyclic_index()
        if self.cfg.n_measurements is not None:
            return {"n": int(self.cfg.n_measurements), "cyclic_index": ell,
                    "rule": "fixed", "ladder": []}
        if self.cfg.sensor_rule == "cyclic_index":
            return {"n": int(ell), "cyclic_index": ell,
                    "rule": "cyclic_index", "ladder": []}

        ladder, chosen = [], None
        n_pix = self.coords.shape[0]
        cand = sorted({int(c) for c in (
            ell, 2 * ell, m // 8, m // 4, m // 2, m, 2 * m
        ) if 1 <= c <= n_pix})
        for n in cand:
            idx = self.measurement_indices(n)
            saved, self.meas_idx = self.meas_idx, idx
            rank = self.observability_rank()
            self.meas_idx = saved
            ladder.append({"n": int(n), "rank": int(rank), "of": int(m)})
            if rank >= m and chosen is None:
                chosen = n
                break
        if chosen is None:
            chosen = cand[-1]
            logger.warning(
                "No sensor count up to N = %d reached full rank (best %d/%d). "
                "The E-GP state is not fully recoverable from any placement on "
                "this ladder; the feedback observer will correct only the "
                "observable subspace, and that is a property of the method on "
                "this record, not a bug.",
                chosen, max(r["rank"] for r in ladder), m,
            )
        return {"n": int(chosen), "cyclic_index": int(ell),
                "rule": "observable", "ladder": ladder}

    # ------------------------------------------------------------------ #
    def fit(self, coords: np.ndarray, y: np.ndarray) -> "EGPBaseline":
        """
        Fit the feature space, the per-step weights and the operator ``A_hat``.

        :param coords: ``(N, d)`` pixel locations (normalised).
        :param y: ``(T, N)`` training frames over those pixels.
        :return: ``self``.
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError("y must be (T, N); got shape {}".format(y.shape))

        # Per-pixel TRAINING climatology, exactly as DB-WM's decoder uses. The
        # E-GP then models the same anomaly field DB-WM models, instead of being
        # asked to represent the static plot mosaic with stationary RBF atoms
        # while its competitor gets that field for free.
        self.offset = (
            y.mean(axis=0) if self.cfg.use_climatology else np.zeros(y.shape[1])
        )
        y = y - self.offset

        if self.cfg.center_placement == "kmeans":
            self.centers = _kmeans(self.coords, self.cfg.n_centers, self.cfg.seed)
        elif self.cfg.center_placement == "grid":
            side = int(np.ceil(np.sqrt(self.cfg.n_centers)))
            lo, hi = self.coords.min(axis=0), self.coords.max(axis=0)
            axes = [np.linspace(lo[d], hi[d], side) for d in range(self.coords.shape[1])]
            self.centers = np.stack(
                [g.ravel() for g in np.meshgrid(*axes)], axis=1
            )[: self.cfg.n_centers]
        else:
            rng = np.random.default_rng(self.cfg.seed)
            self.centers = self.coords[
                rng.choice(self.coords.shape[0],
                           size=min(self.cfg.n_centers, self.coords.shape[0]),
                           replace=False)
            ]

        if self.cfg.learn_hyperparameters:
            self._fit_hyperparameters(self.coords, y)

        self.psi = rbf_features(self.coords, self.centers, self.lengthscale)
        w = self._solve_weights(self.psi, y)  # (T, M)

        # Matrix least squares across SUCCESSIVE weights (paper eq. 17). See the
        # module docstring for why this differs from funcobspy's stacking.
        m = w.shape[1]
        w_minus, w_plus = w[:-1], w[1:]
        gram = w_minus.T @ w_minus + self.cfg.ridge * np.eye(m)
        self.a = np.linalg.solve(gram, w_minus.T @ w_plus).T  # w_{t+1} = A w_t

        sensors = self.choose_sensor_count()
        ell = sensors["cyclic_index"]
        self.meas_idx = self.measurement_indices(sensors["n"])
        rank = self.observability_rank()
        self.report = {
            "n_centers": int(m),
            "lengthscale": self.lengthscale,
            "noise": self.noise,
            "cyclic_index": int(ell),
            "n_measurements": int(len(self.meas_idx)),
            "shaded": self.is_shaded(),
            "observability_rank": rank,
            "observable": bool(rank >= m),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(self.a)))),
            "meas_type": self.cfg.meas_type,
            "sensor_rule": sensors["rule"],
            "sensor_ladder": sensors["ladder"],
            "bandwidth_over_spacing": (
                float(self.lengthscale / self.spacing)
                if np.isfinite(self.spacing) else None
            ),
            "use_climatology": bool(self.cfg.use_climatology),
        }
        logger.info(
            "E-GP fit: M = %d centres, sigma = %.3f, noise = %.3f | cyclic index "
            "ell = %d -> %d sensors (%s) | shaded = %s | rank(O) = %d/%d "
            "(observable = %s) | rho(A) = %.4f",
            m, self.lengthscale, self.noise, ell, len(self.meas_idx),
            self.cfg.meas_type, self.report["shaded"], rank, m,
            self.report["observable"], self.report["spectral_radius"],
        )
        if not self.report["observable"]:
            logger.warning(
                "The E-GP pair (K, A_hat) is NOT observable at %d sensors. Per "
                "Proposition 2 the cyclic index is a LOWER bound, not a "
                "sufficient count, once eigenvalues repeat; raise "
                "cfg.n_measurements to recover the state.", len(self.meas_idx),
            )
        return self

    def encode(self, y: np.ndarray) -> np.ndarray:
        """
        Solve for the weights of one or more frames.

        :param y: ``(N,)`` or ``(T, N)`` values.
        :return: ``(M,)`` or ``(T, M)`` weights.
        """
        y = np.asarray(y, dtype=np.float64) - self.offset
        return self._solve_weights(self.psi, y)

    def decode(self, w: np.ndarray) -> np.ndarray:
        """
        Decode weights to the field, ``f = Psi w``.

        :param w: ``(M,)`` or ``(T, M)`` weights.
        :return: ``(N,)`` or ``(T, N)`` field values.
        """
        return np.asarray(w) @ self.psi.T + self.offset

    def filtered_rolling_forecast(
        self,
        frames: np.ndarray,
        origins: Sequence[int],
        horizon: int,
        observed: Optional[np.ndarray] = None,
        assimilate: str = "full",
        feedback: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run ONE kernel-observer pass over the calendar, branching a forecast at each origin.

        This is the E-GP placed in DB-WM's own protocol, and the protocol is the
        thing that has to match or the numbers are not comparable. DB-WM runs a
        single filter over all 1581 dates, assimilating every observed frame, and
        branches ``t+1..t+H`` from the live state at each origin. The previous
        implementation instead re-solved ``w`` from the origin frame alone, so it
        entered every forecast having forgotten the entire history its competitor
        had assimilated.

        ``assimilate`` controls what the filter is allowed to see, and the choice
        is the whole comparison:

        ``"full"``
            the complete frame at each observed date, through the ridge solve --
            the same information DB-WM's GP posterior receives. This is the
            like-for-like forecast comparator: identical information up to the
            origin, no NDVI after it.
        ``"sensors"``
            only the ``N`` sensing pixels, through ``K``. This is the paper's
            actual operating regime and its central economy claim; the gap
            between the two rows is precisely what the sensor budget costs.

        ``feedback`` additionally keeps correcting *during* the horizon from the
        target frame's sensors. That is filtering, not forecasting -- it reads
        part of the frame it is scored on -- and is reported separately for that
        reason.

        :param frames: ``(T, N)`` values on the fitted pixels.
        :param origins: forecast origin indices.
        :param horizon: ``H``.
        :param observed: ``(T,)`` which dates carry a frame.
        :param assimilate: ``"full"`` or ``"sensors"``.
        :param feedback: keep correcting through the horizon.
        :return: dict with ``mean`` ``(n_origins, H, N)`` and ``valid``.
        """
        frames = np.asarray(frames, dtype=np.float64)
        t_total = frames.shape[0]
        observed = (
            np.ones(t_total, bool) if observed is None else np.asarray(observed, bool)
        )
        m = self.a.shape[0]
        k_meas = self.psi[self.meas_idx]
        q = self.cfg.process_noise * np.eye(m)
        r_sensor = (self.noise**2) * np.eye(len(self.meas_idx))
        # A whole frame, solved through the ridge, is a FAR more precise
        # observation of w than a single pixel is -- its covariance is the ridge
        # posterior sigma^2 (K^T K + sigma^2 I)^{-1}, exactly the
        # sigma_eps^2 Lambda_X^{-1} that DB-WM assimilates. Using sigma^2 I here
        # instead overstates that uncertainty by orders of magnitude (K^T K is
        # large), so the filter under-trusts the data and the "full" row came out
        # WORSE than the N-sensor row -- an ordering that cannot be right when one
        # sees strictly more of the same frame.
        r_full = (self.noise**2) * np.linalg.inv(
            self.psi.T @ self.psi + (self.noise**2 + 1e-7) * np.eye(m)
        )
        r_full = 0.5 * (r_full + r_full.T)

        want = {int(o): i for i, o in enumerate(origins)}
        out = np.zeros((len(origins), horizon, frames.shape[1]))
        valid = np.zeros((len(origins), horizon), dtype=bool)

        state = np.zeros(m)
        cov = self.cfg.p_init * np.eye(m)
        for t in range(t_total):
            if t > 0:
                state = self.a @ state
                cov = self.a @ cov @ self.a.T + q
            if observed[t]:
                if assimilate == "full":
                    # Identity measurement on the weights, exactly as DB-WM
                    # assimilates its GP-solved w_t.
                    obs_w = self.encode(frames[t])
                    s_mat = cov + r_full
                    gain = np.linalg.solve(s_mat, cov).T
                    state = state + gain @ (obs_w - state)
                    cov = cov - gain @ cov
                else:
                    y_s = frames[t][self.meas_idx] - self.offset[self.meas_idx]
                    s_mat = k_meas @ cov @ k_meas.T + r_sensor
                    gain = np.linalg.solve(s_mat, k_meas @ cov).T
                    state = state + gain @ (y_s - k_meas @ state)
                    cov = cov - gain @ k_meas @ cov

            if t not in want:
                continue
            i = want[t]
            fs, fc = state.copy(), cov.copy()
            for h in range(horizon):
                fs = self.a @ fs
                fc = self.a @ fc @ self.a.T + q
                tgt = t + h + 1
                if feedback and tgt < t_total and observed[tgt]:
                    y_s = frames[tgt][self.meas_idx] - self.offset[self.meas_idx]
                    s_mat = k_meas @ fc @ k_meas.T + r_sensor
                    gain = np.linalg.solve(s_mat, k_meas @ fc).T
                    fs = fs + gain @ (y_s - k_meas @ fs)
                    fc = fc - gain @ k_meas @ fc
                if tgt < t_total:
                    out[i, h] = self.decode(fs)
                    valid[i, h] = True
        return {"mean": out, "valid": valid}

    def rolling_forecast(
        self,
        frames: np.ndarray,
        origins: Sequence[int],
        horizon: int,
        observed: Optional[np.ndarray] = None,
        feedback: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run the kernel observer from each origin and forecast ``t+1..t+H``.

        Implements Algorithm 5. With ``feedback=True`` this is the paper's
        **FKO**: at each horizon step the state is propagated by ``A_hat`` and
        then corrected using the ``N`` sensor pixels of the *target* frame -- the
        setting the paper's headline results use, where "the feedback kernel
        observer does well throughout" while the autonomous one diverges.
        With ``feedback=False`` it is the **AKO**, propagating from the origin
        with no measurements at all.

        The two are not interchangeable and the distinction matters for a fair
        comparison: FKO sees ``N`` pixels of the frame it is being scored on, so
        it is a *filtering* result, whereas DB-WM's recursive forecast sees no
        NDVI pixels after the origin. Both are returned so the thesis can report
        the comparison it means to.

        :param frames: ``(T, N)`` values on the fitted pixels.
        :param origins: forecast origin indices.
        :param horizon: ``H``.
        :param observed: ``(T,)`` which dates carry a frame.
        :param feedback: FKO when ``True``, AKO when ``False``.
        :return: dict with ``mean`` ``(n_origins, H, N)`` and ``valid``.
        """
        frames = np.asarray(frames, dtype=np.float64)
        t_total = frames.shape[0]
        observed = (
            np.ones(t_total, bool) if observed is None else np.asarray(observed, bool)
        )
        m = self.a.shape[0]
        k_meas = self.psi[self.meas_idx]
        q = self.cfg.process_noise * np.eye(m)
        r_noise = (self.noise**2) * np.eye(len(self.meas_idx))

        out = np.zeros((len(origins), horizon, frames.shape[1]))
        valid = np.zeros((len(origins), horizon), dtype=bool)
        for i, o in enumerate(origins):
            if not observed[o]:
                continue
            w_state = self.encode(frames[o])
            p = self.cfg.p_init * np.eye(m)
            for h in range(horizon):
                w_state = self.a @ w_state
                p = self.a @ p @ self.a.T + q
                tgt = o + h + 1
                if feedback and tgt < t_total and observed[tgt]:
                    s = k_meas @ p @ k_meas.T + r_noise
                    gain = np.linalg.solve(s, k_meas @ p).T
                    # The state is an ANOMALY once a climatology is in use, so
                    # the observation must be too. Differencing an absolute frame
                    # against an anomaly prediction injects the climatology into
                    # every innovation, which is a systematic bias applied at each
                    # update -- it made the feedback observer worse than the
                    # autonomous one.
                    innov = (
                        frames[tgt][self.meas_idx] - self.offset[self.meas_idx]
                        - k_meas @ w_state
                    )
                    w_state = w_state + gain @ innov
                    p = p - gain @ k_meas @ p
                if tgt < t_total:
                    out[i, h] = self.decode(w_state)
                    valid[i, h] = True
        return {"mean": out, "valid": valid}
