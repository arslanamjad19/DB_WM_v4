"""
Memory-augmented latent dynamics (v3 Sec. 2.2.1 - 2.2.4).

The v2 model asserts a Markov transition on the deep-basis weights. The
Mori-Zwanzig identity (v3 Prop. 2.5) says the *exact* projected dynamics cannot be
Markov: the finite-section residual carries a memory kernel, and truncating at
``L = 1`` forces that kernel into the white-noise term. This module implements the
order-``L`` model::

    w_{t+1} = sum_{j=0}^{L-1} A_j w_{t-j} + B_p p_t + eta_t          (DB-WM-M)

and its block-companion (Markov) realisation on ``R^{Lr}``::

    w_bar_{t+1} = A_cal w_bar_t + B_cal p_t + E_cal eta_t

with ``w_bar_t = [w_t; w_{t-1}; ...; w_{t-L+1}]`` and ``S = [I_r 0 ... 0]``.

Three structural facts drive the implementation
-----------------------------------------------
**1. The lifted spectrum is not sigma(A_0)** (Lemma 2.6). ``lambda`` is an
eigenvalue of ``A_cal`` iff ``det(lambda^L I - sum_j lambda^{L-1-j} A_j) = 0``.
So the v2 eigenvalue clipping of Algorithm 3 step 4 is simply the **wrong
operation** once ``L > 1`` -- clipping ``sigma(A_0)`` constrains something that is
not the lifted spectrum, and clipping ``sigma(A_cal)`` directly destroys the
companion structure. Stability is instead enforced on the **exact** lifted
spectrum, which Lemma 2.6 makes cheap to compute (:func:`enforce_stability`).
v3 offers ``sum_j ||A_j||_2 <= rho_max`` as a sufficient condition; it is sound but
grows conservative with ``L``, so it is used only as a reported diagnostic and
as a fallback, never as the primary constraint.

**2. Over-lagging destroys observability** (Thm 2.8). If the true order is
``L* < L`` then ``A_{L-1} = 0`` and the lifted pair is unobservable -- detectable,
since the offending modes sit at ``lambda = 0``, but the order must be *selected*,
never maximised. :func:`observability_certificate` reports both conditions.

**3. Structure is mandatory** (Sec. 2.2.4). At ``r = 256, L = 7`` an unstructured
memory is 4.6e5 parameters against ~1200 training transitions. The default is
**S2**: diagonalise ``A_0`` once into its real modal form, then give each Koopman
mode its own scalar AR(``L``). Parameter count is exactly ``rL`` -- 1,792 here --
and each mode gets its own relaxation time, which is what a canopy water/energy
balance predicts.

Real modal form
---------------
S2 needs a diagonalisation, but ``A_0`` is real with complex-conjugate eigenpairs.
Rather than let complex arithmetic leak into the Kalman filter, :func:`real_modal_form`
builds the **real** modal decomposition ``A_0 = P D P^{-1}``: a real eigenvalue
contributes a 1x1 block ``[mu]``; a pair ``a +- ib`` contributes the 2x2 rotation-
scaling block ``[[a, b], [-b, a]]`` spanned by ``(Re v, Im v)``. In the coordinates
of a 2x2 block, ``z = x - iy`` evolves as ``z <- lambda z``, so a *complex* scalar
memory coefficient acting on that mode is realised as ``[[Re a, Im a], [-Im a, Re a]]``.
Real modes cost ``L`` parameters, conjugate pairs ``2L`` for the pair -- exactly
``rL`` in total, as claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.memory")


# --------------------------------------------------------------------------- #
# Real modal form
# --------------------------------------------------------------------------- #
@dataclass
class RealModalForm:
    """
    Real modal decomposition ``A = P D P^{-1}`` with ``D`` real block-diagonal.

    :ivar p: ``(r, r)`` real modal basis (columns span the invariant subspaces).
    :ivar p_inv: ``(r, r)`` inverse of ``p``.
    :ivar block_sizes: list of 1s and 2s summing to ``r``.
    :ivar eigenvalues: ``(K,)`` complex representative eigenvalue per block (the
        one with non-negative imaginary part for a conjugate pair).
    :ivar cond: condition number of ``p`` -- large values mean ``A`` is close to
        defective and the per-mode parameterization is numerically fragile.
    """

    p: np.ndarray
    p_inv: np.ndarray
    block_sizes: List[int]
    eigenvalues: np.ndarray
    cond: float

    @property
    def n_blocks(self) -> int:
        """Number of modal blocks ``K`` (``<= r``; equal only if all modes real)."""
        return len(self.block_sizes)

    def block_slices(self) -> List[slice]:
        """Coordinate slice of each block in the modal basis."""
        out, start = [], 0
        for s in self.block_sizes:
            out.append(slice(start, start + s))
            start += s
        return out

    def to_modal(self, w: np.ndarray) -> np.ndarray:
        """
        Map weights into modal coordinates ``w~ = P^{-1} w``.

        :param w: ``(..., r)`` weights.
        :return: ``(..., r)`` modal coordinates.
        """
        return w @ self.p_inv.T

    def from_modal(self, w_modal: np.ndarray) -> np.ndarray:
        """
        Map modal coordinates back to weights ``w = P w~``.

        :param w_modal: ``(..., r)`` modal coordinates.
        :return: ``(..., r)`` weights.
        """
        return w_modal @ self.p.T


def real_modal_form(a: np.ndarray, cond_warn: float = 1e3) -> RealModalForm:
    """
    Compute the real modal decomposition ``A = P D P^{-1}``.

    :param a: ``(r, r)`` real matrix.
    :param cond_warn: warn if ``cond(P)`` exceeds this (near-defective ``A``).
    :return: the :class:`RealModalForm`.
    """
    a = np.asarray(a, dtype=np.float64)
    r = a.shape[0]
    evals, evecs = np.linalg.eig(a)

    order = np.argsort(-np.abs(evals))  # dominant modes first
    evals, evecs = evals[order], evecs[:, order]

    cols: List[np.ndarray] = []
    sizes: List[int] = []
    reps: List[complex] = []
    used = np.zeros(r, dtype=bool)

    for i in range(r):
        if used[i]:
            continue
        lam = evals[i]
        if abs(lam.imag) < 1e-12:
            cols.append(np.real(evecs[:, i]))
            sizes.append(1)
            reps.append(complex(lam.real, 0.0))
            used[i] = True
            continue
        # Locate the conjugate partner.
        partner = -1
        for j in range(i + 1, r):
            if not used[j] and abs(evals[j] - np.conj(lam)) < 1e-9 * max(1.0, abs(lam)):
                partner = j
                break
        v = evecs[:, i]
        if partner < 0:  # pragma: no cover - numerically pathological
            cols.append(np.real(v))
            sizes.append(1)
            reps.append(complex(lam.real, 0.0))
            used[i] = True
            continue
        # A(vr + i vi) = (a + ib)(vr + i vi)  =>  A[vr vi] = [vr vi] [[a, b], [-b, a]]
        if lam.imag < 0:
            lam, v = np.conj(lam), np.conj(v)
        cols.append(np.real(v))
        cols.append(np.imag(v))
        sizes.append(2)
        reps.append(complex(lam))
        used[i] = used[partner] = True

    p = np.stack(cols, axis=-1)
    # Normalise columns so P is well scaled before inversion.
    norms = np.linalg.norm(p, axis=0)
    p = p / np.where(norms > 1e-30, norms, 1.0)
    cond = float(np.linalg.cond(p))
    if cond > cond_warn:
        logger.warning(
            "Modal basis is ill-conditioned (cond(P) = %.2e): A_0 is close to "
            "defective. The S2 blocks are A_j = P diag(alpha_j) P^-1, so their "
            "norms scale with cond(P) -- even tiny per-mode coefficients then give "
            "a large sum_j ||A_j|| and a violently NON-NORMAL lifted operator, "
            "whose powers grow far beyond what rho <= 1 suggests. The usual cause "
            "is r exceeding the rank the weight trajectory excites; reduce r first, "
            "and only then consider parameterization='s1' or 's3'.",
            cond,
        )
    return RealModalForm(
        p=p,
        p_inv=np.linalg.inv(p),
        block_sizes=sizes,
        eigenvalues=np.asarray(reps, dtype=complex),
        cond=cond,
    )


def _complex_to_block(alpha: complex) -> np.ndarray:
    """
    Realise a complex scalar as the 2x2 matrix acting on ``(x, y)`` with ``z = x - iy``.

    :param alpha: complex coefficient.
    :return: ``(2, 2)`` real matrix ``[[Re, Im], [-Im, Re]]``.
    """
    return np.array(
        [[alpha.real, alpha.imag], [-alpha.imag, alpha.real]], dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# The memory operator
# --------------------------------------------------------------------------- #
@dataclass
class MemoryOperator:
    """
    The identified memory kernel ``{A_0, ..., A_{L-1}}`` and its companion form.

    :ivar blocks: ``(L, r, r)`` stack of the memory blocks ``A_j``.
    :ivar modal: the :class:`RealModalForm` of ``A_0`` (``None`` for unstructured).
    :ivar coefficients: ``(K, L)`` complex per-mode AR coefficients under S1/S2
        (``None`` otherwise). Row ``k`` is mode ``k``'s own relaxation profile.
    :ivar parameterization: which structure was used.
    """

    blocks: np.ndarray
    modal: Optional[RealModalForm] = None
    coefficients: Optional[np.ndarray] = None
    parameterization: str = "unstructured"

    @property
    def order(self) -> int:
        """Memory order ``L``."""
        return self.blocks.shape[0]

    @property
    def r(self) -> int:
        """Latent dimension ``r``."""
        return self.blocks.shape[1]

    @property
    def lifted_dim(self) -> int:
        """Lifted dimension ``L * r``."""
        return self.order * self.r

    def companion(self) -> np.ndarray:
        """
        Assemble the block-companion matrix ``A_cal in R^{Lr x Lr}`` (v3 Def. 2.4).

        :return: ``(Lr, Lr)`` companion matrix.
        """
        return companion_matrix(self.blocks)

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        One autonomous step from a memory history.

        :param history: ``(L, r)`` history ``[w_t, w_{t-1}, ..., w_{t-L+1}]``.
        :return: ``(r,)`` predicted ``w_{t+1}`` (excluding forcing and noise).
        """
        history = np.asarray(history)
        if history.shape != (self.order, self.r):
            raise ValueError(
                "history must be ({}, {}), got {}".format(self.order, self.r, history.shape)
            )
        return np.einsum("jrs,js->r", self.blocks, history)

    def norm_budget(self) -> float:
        """
        The stability budget ``sum_j ||A_j||_2`` of the Lemma 2.6 consequence.

        :return: the summed spectral norms.
        """
        return float(sum(np.linalg.norm(aj, ord=2) for aj in self.blocks))


def companion_matrix(blocks: np.ndarray) -> np.ndarray:
    """
    Build ``A_cal`` from the memory blocks (v3 Def. 2.4).

    Top block row is ``[A_0 A_1 ... A_{L-1}]``; the remaining rows are the shift
    ``I_r`` sub-diagonal that carries the history forward.

    :param blocks: ``(L, r, r)`` memory blocks.
    :return: ``(Lr, Lr)`` companion matrix.
    """
    blocks = np.asarray(blocks)
    l, r, _ = blocks.shape
    out = np.zeros((l * r, l * r), dtype=blocks.dtype)
    out[:r] = blocks.transpose(1, 0, 2).reshape(r, l * r)
    if l > 1:
        out[r:, : (l - 1) * r] = np.eye((l - 1) * r)
    return out


def selector_matrix(order: int, r: int) -> np.ndarray:
    """
    The current-block selector ``S = [I_r 0 ... 0] in R^{r x Lr}`` (v3 Def. 2.4).

    :param order: memory order ``L``.
    :param r: latent dimension.
    :return: ``(r, Lr)`` selector.
    """
    s = np.zeros((r, order * r))
    s[:, :r] = np.eye(r)
    return s


def noise_injection_matrix(order: int, r: int) -> np.ndarray:
    """
    The process-noise injection ``E = [I_r; 0; ...; 0] in R^{Lr x r}``.

    The lifted process noise ``E Q E^T`` is singular (rank ``r`` of ``Lr``), which
    makes the naive reading of v2 Thm 4.3 appear to fail. Lemma 2.7 restores it:
    ``(A_cal, E)`` is controllable for *every* memory kernel, so the pair is always
    stabilizable and ``P_inf`` still exists.

    :param order: memory order ``L``.
    :param r: latent dimension.
    :return: ``(Lr, r)`` injection matrix.
    """
    return selector_matrix(order, r).T


def lifted_input_matrix(order: int, b: np.ndarray) -> np.ndarray:
    """
    Lift the input matrix to ``B_cal = [B; 0; ...; 0] in R^{Lr x ell}``.

    :param order: memory order ``L``.
    :param b: ``(r, ell)`` input matrix.
    :return: ``(Lr, ell)`` lifted input matrix.
    """
    b = np.asarray(b)
    r, ell = b.shape
    out = np.zeros((order * r, ell), dtype=b.dtype)
    out[:r] = b
    return out


# --------------------------------------------------------------------------- #
# Lemma 2.6: the lifted spectrum
# --------------------------------------------------------------------------- #
def lifted_spectrum(op: MemoryOperator, fast: bool = True) -> np.ndarray:
    """
    Eigenvalues of ``A_cal`` via Lemma 2.6, without forming the ``Lr x Lr`` matrix.

    ``lambda in sigma(A_cal)`` iff ``det(lambda^L I_r - sum_j lambda^{L-1-j} A_j) = 0``.

    Under S1/S2 the matrix polynomial factorises per mode (Prop. 2.11): each v2
    Koopman mode ``mu_i`` splits into exactly ``L`` lifted modes, and the spread of
    that cluster *is* the mode's thermal/phenological inertia. Cost drops from
    ``O(L^3 r^3)`` to ``O(r^3 + r L^3)``.

    :param op: the identified :class:`MemoryOperator`.
    :param fast: use the per-mode factorisation when the structure allows it.
    :return: ``(Lr,)`` complex eigenvalues, sorted by decreasing modulus.
    """
    if fast and op.coefficients is not None and op.modal is not None:
        roots: List[complex] = []
        for k, size in enumerate(op.modal.block_sizes):
            alpha = op.coefficients[k]
            # z_{t+1} = sum_j alpha_j z_{t-j}  =>  lambda^L - sum_j alpha_j lambda^{L-1-j}
            poly = np.concatenate([[1.0 + 0j], -np.asarray(alpha, dtype=complex)])
            rts = np.roots(poly)
            roots.extend(rts)
            if size == 2:
                # The conjugate mode contributes the conjugate root set.
                roots.extend(np.conj(rts))
        out = np.asarray(roots, dtype=complex)
    else:
        out = np.linalg.eigvals(op.companion())
    return out[np.argsort(-np.abs(out))]


def spectral_radius_lifted(op: MemoryOperator) -> float:
    """
    Spectral radius of the lifted operator ``rho(A_cal)``.

    :param op: the memory operator.
    :return: ``max |lambda|`` over the lifted spectrum.
    """
    return float(np.max(np.abs(lifted_spectrum(op))))


def enforce_stability(
    op: MemoryOperator, rho_max: float = 1.0, max_iter: int = 40
) -> MemoryOperator:
    """
    Enforce ``rho(A_cal) <= rho_max`` on the **exact** lifted spectrum.

    Lemma 2.6 gives the lifted spectrum in closed form -- and cheaply under S1/S2,
    where the matrix polynomial factorises per mode -- so there is no need to fall
    back on a surrogate. The operator is left untouched whenever it is already
    stable, and otherwise all blocks are scaled by the largest common factor that
    brings ``rho`` to ``rho_max`` (found by bisection, since ``rho`` is continuous
    and monotone in that factor and vanishes as it goes to zero).

    Why not the norm budget
    -----------------------
    v3 offers ``sum_j ||A_j||_2 <= rho_max`` as a *sufficient* condition, and it is
    sound but increasingly conservative as ``L`` grows: seven blocks must share a
    budget of 1, so a near-conservative field needing ``||A_0|| ~ 1`` on its own has
    nothing left for the memory. Applied unconditionally it silently cripples the
    very lift it is meant to protect -- on the real record it clamped the budget to
    1.000 while the true ``rho`` was only 0.53, i.e. it shrank the operator to half
    the permitted spectral radius and inflated every semigroup defect as a result.

    Scaling is deliberately uniform across lags, so the *shape* of the memory
    kernel -- the relative weight of each lag, which is the physically meaningful
    part -- is preserved while its gain is reduced. Eigenvalue clipping is still
    never used: it constrains a spectrum that is not the lifted one (``sigma(A_0)``)
    or destroys the companion structure (``sigma(A_cal)``).

    :param op: the memory operator.
    :param rho_max: maximum allowed lifted spectral radius.
    :param max_iter: bisection iterations.
    :return: a new :class:`MemoryOperator`, scaled only if it was unstable.
    """
    rho = spectral_radius_lifted(op)
    if not np.isfinite(rho):
        logger.warning(
            "Lifted spectral radius is not finite; falling back to the norm budget."
        )
        return _scale_blocks(op, rho_max / max(op.norm_budget(), 1e-30))
    if rho <= rho_max:
        logger.info(
            "Stability: rho(A_cal) = %.4f <= rho_max = %.2f; operator left "
            "unscaled (norm budget was %.4f, which alone would have over-shrunk it).",
            rho, rho_max, op.norm_budget(),
        )
        return op

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if spectral_radius_lifted(_scale_blocks(op, mid)) > rho_max:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    scaled = _scale_blocks(op, lo)
    logger.info(
        "Stability: rho(A_cal) = %.4f > rho_max = %.2f; scaled blocks by %.4f "
        "-> rho = %.4f (Lemma 2.6 exact spectrum, NOT eigenvalue clipping).",
        rho, rho_max, lo, spectral_radius_lifted(scaled),
    )
    return scaled


def forecast_gain(op: MemoryOperator, horizon: int = 6) -> np.ndarray:
    """
    ``g_h = ||S A_cal^h||_2`` -- how much an ``h``-step forecast amplifies a state error.

    This is the finite-horizon quantity that v2 Theorem 4.2's ``rho^T`` factor
    stands for. The forecast is ``w_{t+h} = S A_cal^h w_bar_t``, so an error
    ``delta`` in the lifted state arrives at the prediction as ``S A_cal^h delta``:
    ``g_h`` is exactly the factor by which the encoding error reaches the output.

    For a persistence model (``A_0 = I``, ``A_j = 0``) ``g_h = 1`` for every ``h``,
    which is the natural reference: a forecast with ``g_h > 1`` amplifies its own
    input error, and one with ``g_h < 1`` damps it.

    :param op: the memory operator.
    :param horizon: largest ``h`` to evaluate.
    :return: ``(horizon,)`` array of gains, ``h = 1..horizon``.
    """
    a_cal = op.companion()
    sel = np.zeros((op.r, a_cal.shape[0]))
    sel[:, : op.r] = np.eye(op.r)
    gains, power = [], np.eye(a_cal.shape[0])
    for _ in range(max(horizon, 1)):
        power = power @ a_cal
        gains.append(float(np.linalg.norm(sel @ power, ord=2)))
    return np.asarray(gains)


def _persistence_blocks(r: int, order: int) -> np.ndarray:
    """
    The memory blocks of the persistence model ``w_{t+1} = w_t``.

    :param r: state dimension.
    :param order: memory order ``L``.
    :return: ``(L, r, r)`` blocks, ``A_0 = I`` and the rest zero.
    """
    blocks = np.zeros((order, r, r))
    blocks[0] = np.eye(r)
    return blocks


def _persistence_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """
    Persistence in per-mode coefficient space: ``alpha_0 = 1``, ``alpha_j = 0``.

    Exact, not an approximation. Persistence is ``A_0 = I``, and the identity is
    the identity in every basis, so ``D_0 = P^{-1} I P = I``: a real modal block
    takes the scalar 1, and a conjugate pair takes ``1 + 0i``, whose 2x2
    realisation ``[[1, 0], [0, 1]]`` is again the identity. Every deeper lag is
    zero.

    :param coeffs: ``(K, L)`` complex per-mode coefficients, for the shape.
    :return: ``(K, L)`` the persistence anchor.
    """
    anchor = np.zeros_like(coeffs)
    anchor[:, 0] = 1.0
    return anchor


def _blend_to_persistence(op: MemoryOperator, s: float) -> MemoryOperator:
    """
    Interpolate between persistence (``s = 0``) and the fitted operator (``s = 1``).

    The per-mode coefficients are blended **too**, against the same anchor. That
    is not bookkeeping -- leaving them stale silently corrupts every spectral
    quantity in the pipeline, because :func:`lifted_spectrum` takes its fast path
    from ``coefficients`` whenever they are present (Prop. 2.11) and would then
    return the spectrum of the *un-blended* fit:

    * ``rho(A_cal)`` and the ``rho <= rho_max`` rails, which bisect on it and so
      could never converge -- the reported radius simply never moved;
    * the plotted Koopman spectrum, which showed the pre-shrinkage modes;
    * :func:`observability_certificate`, whose fast path builds the Thm 2.8(ii)
      null directions from ``coefficients`` as well.

    On the real record this is exactly why ``rho`` was reported as 1.37 after a
    blend that had already brought the operator inside the unit disc.

    The blend is exact in coefficient space because ``_blocks_from_coefficients``
    is **linear** in the coefficients and persistence has the closed form above,
    so blending the coefficients and blending the blocks give the same operator.
    :func:`_scale_blocks` already scaled its coefficients for the same reason.

    :param op: the memory operator.
    :param s: blend weight.
    :return: a new :class:`MemoryOperator`.
    """
    anchor = _persistence_blocks(op.r, op.order)
    coeffs = None
    if op.coefficients is not None:
        c_anchor = _persistence_coefficients(op.coefficients)
        coeffs = c_anchor + s * (op.coefficients - c_anchor)
    return MemoryOperator(
        blocks=anchor + s * (op.blocks - anchor),
        modal=op.modal,
        coefficients=coeffs,
        parameterization=op.parameterization,
    )


def calibrate_persistence_blend(
    op: MemoryOperator,
    b_p: np.ndarray,
    weights: np.ndarray,
    forcing: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    horizon: int = 6,
    gain_max: float = 2.0,
    holdout_fraction: float = 0.2,
    grid: Optional[Sequence[float]] = None,
) -> Tuple[MemoryOperator, Dict[str, object]]:
    """
    Choose how far the operator may move from persistence, by held-out ``h``-step error.

    A hard cap ``max_h g_h <= 1`` is the safe reading of Theorem 4.2, but it is too
    blunt to be useful: persistence sits at exactly 1, so *any* operator with an
    expansive direction is rejected and the bisection collapses to persistence,
    throwing away the learned dynamics wholesale. That is a guarantee bought by
    not modelling anything.

    The blend weight ``s`` in ``A(s) = A_persist + s (A_fit - A_persist)`` is
    precisely a shrinkage parameter toward a low-variance anchor, so v3
    Theorem 2.12 applies to it directly: risk is ``(1-w)^2 V + w^2 b^2`` with a
    strict interior optimum whenever the anchor is biased and the fit is noisy.
    Rather than assume where that optimum is, it is *measured* -- ``s`` is selected
    on a held-out tail of the training split by mean ``h``-step error, exactly as
    ``nu_h`` is selected for the horizon family.

    Two rails keep this honest:

    * ``s = 0`` (persistence) is always in the grid, so the selected operator can
      never be worse on held-out data than doing nothing;
    * candidates whose gain exceeds ``gain_max`` are rejected outright, so a
      violently non-normal operator cannot win by overfitting a short holdout.

    :param op: the fitted memory operator.
    :param b_p: ``(r, ell)`` input matrix.
    :param weights: ``(T, r)`` training trajectory.
    :param forcing: ``(T, ell)`` inputs, or ``None``.
    :param valid: ``(T,)`` observation mask.
    :param horizon: horizon ``H`` scored.
    :param gain_max: hard rejection threshold on ``max_h g_h``.
    :param holdout_fraction: tail fraction of the trajectory used for selection.
    :param grid: candidate blend weights; defaults to a 0..1 sweep.
    :return: ``(blended operator, report)``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    t_total = weights.shape[0]
    valid = np.ones(t_total, bool) if valid is None else np.asarray(valid, bool)
    grid = (
        np.linspace(0.0, 1.0, 21) if grid is None else np.asarray(grid, dtype=float)
    )
    start = int(t_total * (1.0 - holdout_fraction))
    origins = [
        t for t in range(max(start, op.order - 1), t_total - horizon)
        if valid[t - op.order + 1 : t + 1].all()
    ]
    if len(origins) < 4:
        logger.warning(
            "Only %d held-out origins available to calibrate the persistence "
            "blend; keeping the hard gain cap instead.", len(origins),
        )
        return enforce_forecast_gain(op, min(gain_max, 1.0), horizon), {
            "blend": None, "reason": "insufficient_holdout",
        }

    def score(cand: MemoryOperator) -> float:
        """Mean over horizons of the RMS h-step latent error."""
        errs = []
        for t in origins:
            hist = [weights[t - j] for j in range(op.order)]
            state = list(hist)
            for h in range(horizon):
                nxt = np.einsum("jrs,js->r", cand.blocks, np.stack(state))
                if forcing is not None and b_p.shape[1] and t + h < len(forcing):
                    nxt = nxt + b_p @ forcing[t + h]
                state = [nxt] + state[:-1]
                if valid[t + h + 1]:
                    errs.append(np.sum((nxt - weights[t + h + 1]) ** 2))
        return float(np.sqrt(np.mean(errs))) if errs else float("inf")

    rows = []
    for s in grid:
        cand = _blend_to_persistence(op, float(s))
        g = float(np.max(forecast_gain(cand, horizon)))
        rows.append((float(s), g, score(cand) if g <= gain_max else float("inf")))

    feasible = [r for r in rows if np.isfinite(r[2])]
    best = min(feasible, key=lambda r: r[2])
    s_best, gain_best, err_best = best
    err_persist = next(r[2] for r in rows if r[0] == 0.0)
    err_full = rows[-1][2]

    logger.info(
        "Persistence blend selected on %d held-out origins: s = %.2f "
        "(gain %.3f, held-out h-step RMS %.4f). Reference points: s=0 "
        "persistence %.4f, s=1 raw fit %s. %d/%d candidates rejected by the "
        "gain cap %.1f.",
        len(origins), s_best, gain_best, err_best, err_persist,
        "%.4f" % err_full if np.isfinite(err_full) else "rejected (gain)",
        len(rows) - len(feasible), len(rows), gain_max,
    )
    if s_best == 0.0:
        logger.warning(
            "The learned memory kernel does not beat persistence on held-out "
            "data, so it was shrunk away entirely. Report this: it means the "
            "dynamics contribute nothing beyond the basis on this record."
        )
    return _blend_to_persistence(op, s_best), {
        "blend": s_best,
        "gain": gain_best,
        "holdout_error": err_best,
        "holdout_error_persistence": err_persist,
        "holdout_error_raw": err_full,
        "n_origins": len(origins),
        "grid": [{"s": r[0], "gain": r[1], "error": r[2]} for r in rows],
    }


def enforce_forecast_gain(
    op: MemoryOperator, gain_max: float = 1.0, horizon: int = 6, max_iter: int = 60
) -> MemoryOperator:
    """
    Enforce ``max_h ||S A_cal^h||_2 <= gain_max`` -- the Theorem 4.2 hypothesis.

    What was wrong before
    ---------------------
    v2 Theorem 4.2 opens with "let ``rho = ||A||_2`` be the spectral **norm**" and
    bounds the ``T``-step error by ``rho^T eps_enc + ...``. The pipeline was
    instead constraining the spectral **radius** ``rho(A_cal)``, and for a
    non-normal operator the two are unrelated at finite horizon: on the real
    record the radius sat at exactly 1.000 while ``||A_cal||`` was 4.13 and the
    powers peaked at 11.1. The theorem's guarantee was never actually in force,
    and the first predict step multiplied the encoding error by 4.13 -- which is,
    to three significant figures, the entire observed one-step error.

    Why the gain and not ``||A_cal||`` itself
    -----------------------------------------
    The companion matrix carries the shift identities in its sub-diagonal, so
    ``||A_cal||_2 >= 1`` for every ``L >= 2`` regardless of the dynamics: asking
    for ``||A_cal|| <= 1`` is infeasible by construction, and bisecting toward it
    would drive the blocks to zero. Those identity blocks are bookkeeping -- they
    shift the delay line, they do not act on the prediction. ``S A_cal^h`` strips
    them out and leaves exactly the map from lifted state to forecast, which is
    what the error actually travels through.

    Why the shrinkage target is persistence, not zero
    -------------------------------------------------
    Scaling all blocks toward **zero** (the previous behaviour) does not just
    reduce the gain, it destroys the forecast: it pulls every prediction toward the
    state-space origin, which after the climatology offset is the training mean
    field. That is visible in the reported triptychs as a large negative bias
    (-0.48 normalised) on top of the RMSE. Blending toward **persistence** instead
    reaches ``g_h = 1`` exactly at ``s = 0`` while predicting ``w_{t+h} = w_t``,
    which on a daily NDVI record is a strong forecast rather than a null one. The
    bisection is therefore always feasible for ``gain_max >= 1``, and the worst
    case it can degrade to is persistence.

    :param op: the memory operator.
    :param gain_max: maximum allowed ``max_h g_h``.
    :param horizon: horizon over which the gain is controlled.
    :param max_iter: bisection iterations.
    :return: a new :class:`MemoryOperator`, blended only if the bound was violated.
    """
    gains = forecast_gain(op, horizon)
    rho = spectral_radius_lifted(op)
    worst = float(np.max(gains))
    if worst <= gain_max:
        logger.info(
            "Forecast gain: max_h ||S A_cal^h|| = %.4f <= %.2f (rho = %.4f). "
            "Thm 4.2 is in force: the encoding error is not amplified over "
            "h = 1..%d. Gains %s", worst, gain_max, rho, horizon,
            np.array2string(gains, precision=3),
        )
        return op

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(np.max(forecast_gain(_blend_to_persistence(op, mid), horizon))) > gain_max:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break
    blended = _blend_to_persistence(op, lo)
    logger.info(
        "Forecast gain: max_h ||S A_cal^h|| = %.4f > %.2f (rho was %.4f -- the "
        "radius alone would have passed, which is how this went unnoticed). "
        "Blended %.1f%% toward persistence -> gain %.4f. Gains now %s",
        worst, gain_max, rho, 100.0 * (1.0 - lo),
        float(np.max(forecast_gain(blended, horizon))),
        np.array2string(forecast_gain(blended, horizon), precision=3),
    )
    return blended


def enforce_spectral_radius(
    op: MemoryOperator, rho_max: float = 1.0, max_iter: int = 60
) -> Tuple[MemoryOperator, Dict[str, float]]:
    """
    Blend toward persistence until ``rho(A_cal) <= rho_max`` (v3 Lemma 2.6).

    Why this exists alongside the forecast-gain cap
    -----------------------------------------------
    The two constraints bound *different* things and neither implies the other:

    ``max_h ||S A_cal^h|| <= gain_max``
        a **finite-horizon** statement. It is what v2 Thm 4.2 actually multiplies
        the encoding error by over ``h = 1..H``, so it is the one that governs the
        reported metrics.

    ``rho(A_cal) <= rho_max``
        an **asymptotic** statement. It is what decides whether the operator is
        stable *at all*.

    A non-normal operator can satisfy the first and violate the second: on the
    real record the 6-step gain was held under 2.0 while ``rho = 1.37``, meaning
    the unstable eigen-directions were barely visible through ``S`` over six steps
    but grow as ``1.37^h`` thereafter. That is a fragile place to be. It is not
    wrong at ``h <= 6``, but it makes the model's behaviour depend on the horizon
    never being extended and on the state never rotating into those directions --
    neither of which is a property one wants to rely on in a forecast that is
    described as robust.

    ``MemoryConfig.stability_metric = "forecast_gain"`` selected the first cap and
    left ``DynamicsConfig.rho_max`` declared but **never applied**, so this rail is
    now run as well as, not instead of, the gain cap.

    Blending toward persistence rather than scaling toward zero
    -----------------------------------------------------------
    Persistence has ``rho = 1`` exactly: its matrix polynomial is
    ``lambda^{L-1} (lambda - 1) I``, with roots at 1 and 0. So the bisection is
    always feasible for ``rho_max >= 1``, and the worst case it degrades to is
    "predict today's field" -- a strong forecast on a daily NDVI record. Scaling
    all blocks toward *zero* would instead pull every prediction to the state-space
    origin, which after the climatology offset is the training mean field, and
    shows up as a large negative bias on top of the RMSE.

    :param op: the memory operator, already gain-capped.
    :param rho_max: maximum allowed lifted spectral radius.
    :param max_iter: bisection iterations.
    :return: ``(operator, report)``; the operator is unchanged when already stable.
    """
    rho = spectral_radius_lifted(op)
    report = {"rho_before": float(rho), "rho_after": float(rho), "blend": 1.0}
    if not np.isfinite(rho):  # pragma: no cover - pathological fit
        logger.warning("Lifted spectral radius is not finite; falling back to persistence.")
        return _blend_to_persistence(op, 0.0), {
            "rho_before": float("inf"), "rho_after": 1.0, "blend": 0.0,
        }
    if rho <= rho_max:
        logger.info(
            "Spectral radius rail: rho(A_cal) = %.4f <= %.2f, operator unchanged.",
            rho, rho_max,
        )
        return op, report

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if spectral_radius_lifted(_blend_to_persistence(op, mid)) > rho_max:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break
    out = _blend_to_persistence(op, lo)
    report = {
        "rho_before": float(rho),
        "rho_after": float(spectral_radius_lifted(out)),
        "blend": float(lo),
    }
    logger.warning(
        "Spectral radius rail BOUND: rho(A_cal) = %.4f > %.2f, so the operator was "
        "unstable and would grow as %.2f^h beyond the fitted horizon. Blended a "
        "further %.1f%% toward persistence -> rho = %.4f, gains now %s. The "
        "forecast-gain cap alone did not catch this: it bounds ||S A^h|| over "
        "h = 1..H only, and a non-normal operator can hide unstable directions "
        "from S over a short horizon.",
        rho, rho_max, rho, 100.0 * (1.0 - lo), report["rho_after"],
        np.array2string(forecast_gain(out, 6), precision=3),
    )
    return out, report


def _scale_blocks(op: MemoryOperator, scale: float) -> MemoryOperator:
    """
    Scale every memory block by a common factor, preserving the kernel's shape.

    :param op: the memory operator.
    :param scale: multiplicative factor.
    :return: a new :class:`MemoryOperator`.
    """
    return MemoryOperator(
        blocks=op.blocks * scale,
        modal=op.modal,
        coefficients=None if op.coefficients is None else op.coefficients * scale,
        parameterization=op.parameterization,
    )


# --------------------------------------------------------------------------- #
# Theorem 2.8: observability / detectability of the lift
# --------------------------------------------------------------------------- #
def _modal_null_directions(op: "MemoryOperator") -> Optional[List[np.ndarray]]:
    """
    Null directions of the matrix polynomial, in closed form, for S1/S2 kernels.

    Theorem 2.8(ii) asks whether ``C v != 0`` for every root ``lambda`` of the
    matrix polynomial and every associated null vector ``v``. Computing that by
    SVD-ing ``M(lambda)`` at each of the ``Lr`` roots costs ``O(L r^4)`` -- ~3e10
    operations at ``r = 256``, minutes per call, and the memory-depth sweep calls
    it once per ``L``.

    It is also unnecessary. Under S1/S2 the polynomial factorises per mode
    (Prop. 2.11), so in modal coordinates ``M(lambda)`` is block-diagonal and only
    block ``k`` is singular at mode ``k``'s roots. The null direction is therefore
    the mode's own invariant direction -- **the same for all ``L`` of its lifted
    roots**, which is exactly what "each Koopman mode splits into a cluster of
    ``L`` delay modes" means geometrically: the cluster shares one spatial
    direction and differs only in temporal decay.

    So there are at most ``K <= r`` distinct directions to test, not ``Lr``:

    * a real 1x1 block contributes its real modal column ``P[:, k]``;
    * a conjugate pair contributes the complex direction ``P[:, k] + i P[:, k+1]``
      (with ``z = x - iy``, the weight is ``Re(z * v_c)``, so ``v_c`` spans it).

    Cost drops from ``O(L r^4)`` to ``O(r^2)``, and the result is *exact* rather
    than a numerical null-space estimate.

    :param op: the memory operator.
    :return: one direction per modal block, or ``None`` if the operator has no
             per-mode structure (unstructured / S3), where the general path applies.
    """
    if op.coefficients is None or op.modal is None:
        return None
    out: List[np.ndarray] = []
    for sl, size in zip(op.modal.block_slices(), op.modal.block_sizes):
        if size == 1:
            out.append(op.modal.p[:, sl.start].astype(complex))
        else:
            out.append(
                op.modal.p[:, sl.start] + 1j * op.modal.p[:, sl.start + 1]
            )
    return out


def is_persistence(op: "MemoryOperator", tol: float = 1e-10) -> bool:
    """
    Is this operator (numerically) the persistence model ``w_{t+1} = w_t``?

    Worth its own predicate because persistence is a *legitimate selected
    outcome* of the blend, not a degenerate accident -- and every structural
    certificate reads strangely on it. ``A_{L-1} = 0`` exactly, so Thm 2.8(i)
    reports the lifted pair as unobservable and blames over-lagging, when in
    fact the memory kernel is zero by construction because the data did not
    support one. Detecting the case lets the diagnostics say that instead of
    sending the reader off to reduce ``L``.

    :param op: the memory operator.
    :param tol: tolerance relative to ``||A_0||``.
    :return: whether the operator is persistence.
    """
    scale = max(float(np.linalg.norm(op.blocks[0], ord=2)), 1e-30)
    if op.order > 1 and float(np.max(np.abs(op.blocks[1:]))) > tol * scale:
        return False
    return bool(
        np.allclose(op.blocks[0], np.eye(op.r), atol=max(tol * scale, 1e-9))
    )


def observability_certificate(
    op: MemoryOperator,
    c: np.ndarray,
    tol: float = 1e-8,
    rank_tol: float = 1e-2,
) -> Dict[str, object]:
    """
    Check Theorem 2.8 for the lifted pair ``(C_cal, A_cal)``.

    Observable iff **(i)** ``A_{L-1}`` is nonsingular *and* **(ii)** for every root
    ``lambda`` of the matrix polynomial and every associated null vector ``v``,
    ``C v != 0``. Detectable iff (ii) holds for the roots with ``|lambda| >= 1``.

    Condition (i) is the sharp one in practice: if the true memory order is
    ``L* < L`` then ``A_{L-1} = 0``, the system is unobservable, and the excess
    lags are unrecoverable. This is why the order must be selected by the
    ``||D_h||``-vs-``L`` sweep rather than set as large as affordable.

    Note that v2 Def. 4.1 shadedness is **no longer sufficient**: it asks only that
    every basis function be activated somewhere, while (ii) asks that ``ker C``
    avoid the eigen-directions of the matrix polynomial.

    :param op: the memory operator.
    :param c: ``(m, r)`` emission acting on the current block (``I_r`` for a fully
              observed NDVI frame, or the ``3 x r`` weather emission).
    :param tol: singular-value tolerance for condition (ii).
    :param rank_tol: **relative** threshold for condition (i):
        ``A_{L-1}`` counts as singular when ``s_min(A_{L-1}) / ||A_0||_2 < rank_tol``.
        An absolute test is useless here -- with a finite sample the excess block is
        never exactly zero, only ``O(1/sqrt(T))`` -- so over-lagging shows up as a
        *ratio* two to three orders below the kernel's own scale, not as an exact
        rank drop.
    :return: dict with ``a_last_nonsingular``, ``a_last_smin``,
             ``a_last_relative_smin``, ``observable``, ``detectable``,
             ``min_ratio``, ``n_unobservable_modes``.
    """
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    a_last = op.blocks[-1]
    s_last = np.linalg.svd(a_last, compute_uv=False)
    a_last_smin = float(s_last.min()) if s_last.size else 0.0
    scale = float(np.linalg.norm(op.blocks[0], ord=2))
    a_last_rel = a_last_smin / max(scale, 1e-30)
    a_last_ok = op.order == 1 or a_last_rel >= rank_tol

    r = op.r
    worst, worst_unstable, n_bad = np.inf, np.inf, 0

    directions = _modal_null_directions(op)
    if directions is not None:
        # Fast, exact path: one direction per modal block, shared by all L of its
        # lifted roots (see _modal_null_directions). O(r^2) instead of O(L r^4).
        for k, v in enumerate(directions):
            alpha = op.coefficients[k]
            poly = np.concatenate([[1.0 + 0j], -np.asarray(alpha, dtype=complex)])
            roots = np.roots(poly)
            ratio = float(np.linalg.norm(c @ v) / max(np.linalg.norm(v), 1e-30))
            worst = min(worst, ratio)
            if ratio <= tol:
                n_bad += int(roots.size)
            if np.any(np.abs(roots) >= 1.0 - 1e-9):
                worst_unstable = min(worst_unstable, ratio)
    else:
        # General path (unstructured / S3): null space by SVD at every root.
        lam = lifted_spectrum(op, fast=False)
        for z in lam:
            # M(lambda) = lambda^L I - sum_j lambda^{L-1-j} A_j
            m = (z ** op.order) * np.eye(r, dtype=complex)
            for j in range(op.order):
                m = m - (z ** (op.order - 1 - j)) * op.blocks[j]
            _, sv, vh = np.linalg.svd(m)
            v = vh[-1].conj()  # right null vector (smallest singular direction)
            ratio = float(np.linalg.norm(c @ v) / max(np.linalg.norm(v), 1e-30))
            worst = min(worst, ratio)
            if ratio <= tol:
                n_bad += 1
            if abs(z) >= 1.0 - 1e-9:
                worst_unstable = min(worst_unstable, ratio)

    observable = bool(a_last_ok and worst > tol)
    detectable = bool(worst_unstable > tol or not np.isfinite(worst_unstable))
    # A_0 being near-singular in its own right is a DIFFERENT diagnosis from
    # A_{L-1} vanishing, and the two demand opposite remedies. Separate them.
    s0 = np.linalg.svd(op.blocks[0], compute_uv=False)
    a0_rel = float(s0.min() / max(s0.max(), 1e-30)) if s0.size else 0.0
    a0_rank_deficient = a0_rel < rank_tol

    if not a_last_ok:
        if op.order == 1 or a0_rank_deficient:
            # At L = 1, A_{L-1} IS A_0, so this cannot possibly mean over-lagging.
            # More generally, when A_0 is itself near-singular the whole kernel
            # inherits it and blaming the memory order sends you to fix the wrong
            # knob. The cause is r exceeding the rank the trajectory excites.
            logger.warning(
                "Theorem 2.8(i) FAILS because A_0 ITSELF is near-singular "
                "(s_min/s_max = %.2e), not because L over-shoots%s. r = %d exceeds "
                "the number of directions the weight trajectory excites, so the "
                "ridge fit is singular in the unexcited ones (v2 Prop. 4.4: those "
                "are ker W_-^T). Reduce r -- see the weight-rank report -- rather "
                "than reducing L, which will not help.",
                a0_rel,
                " (L = 1 cannot over-lag)" if op.order == 1 else "",
                op.r,
            )
        else:
            logger.warning(
                "Theorem 2.8(i) FAILS: A_{L-1} is effectively singular "
                "(s_min/||A_0||_2 = %.2e < %.1e) while A_0 is well conditioned "
                "(%.2e). The memory order L = %d over-shoots the true order, so the "
                "lifted system is UNOBSERVABLE -- still detectable, since the "
                "offending modes sit at lambda = 0, but the excess lags carry no "
                "recoverable state. Reduce L using the ||D_h||-vs-L sweep.",
                a_last_rel, rank_tol, a0_rel, op.order,
            )
    return {
        "a_last_nonsingular": bool(a_last_ok),
        "a_last_smin": a_last_smin,
        "a_last_relative_smin": float(a_last_rel),
        "a0_relative_smin": a0_rel,
        # Distinguishes "r too large" from "L too large" -- the two failure modes
        # look identical in a_last_relative_smin but need opposite fixes.
        "cause": (
            "a0_rank_deficient" if (not a_last_ok) and (op.order == 1 or a0_rank_deficient)
            else "over_lagged" if not a_last_ok
            else "none"
        ),
        "observable": observable,
        "detectable": detectable,
        "min_ratio": float(worst),
        "min_ratio_unstable": float(worst_unstable),
        "n_unobservable_modes": int(n_bad),
    }


# --------------------------------------------------------------------------- #
# Identification of the memory kernel
# --------------------------------------------------------------------------- #
def build_lifted_design(
    weights: np.ndarray,
    order: int,
    horizon: int = 1,
    valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the lifted regression design (v3 estimation pipeline step 1).

    Row ``i`` of the design corresponds to origin ``t = order - 1 + i`` and holds
    ``w_bar_t = [w_t, w_{t-1}, ..., w_{t-L+1}]``; the targets hold ``w_{t+h}``.

    :param weights: ``(T, r)`` weight trajectory on a **uniform** time grid.
    :param order: memory order ``L``.
    :param horizon: maximum horizon ``H`` (targets are built for ``h = 1..H``).
    :param valid: ``(T,)`` bool -- ``False`` where ``w_t`` was interpolated rather
        than observed. A window is used only if every frame it touches is valid.
    :return: ``(w_bar, targets, origins)`` with shapes ``(n, L, r)``,
             ``(H, n, r)`` and ``(n,)``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    t_total, r = weights.shape
    if order < 1:
        raise ValueError("order must be >= 1")
    if t_total < order + horizon:
        raise ValueError(
            "need at least L + H = {} steps, got {}".format(order + horizon, t_total)
        )
    origins = np.arange(order - 1, t_total - horizon)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        keep = np.asarray(
            [valid[t - order + 1 : t + horizon + 1].all() for t in origins], dtype=bool
        )
        origins = origins[keep]
        if origins.size == 0:
            raise ValueError(
                "No fully observed window of length L + H = {} exists.".format(order + horizon)
            )
    # w_bar[i, j] = w_{t - j}
    w_bar = np.stack([weights[origins - j] for j in range(order)], axis=1)
    targets = np.stack([weights[origins + h] for h in range(1, horizon + 1)], axis=0)
    return w_bar, targets, origins


def _ridge(target: np.ndarray, regressor: np.ndarray, mu: float) -> np.ndarray:
    """
    Solve ``G = target @ X^T (X X^T + mu I)^{-1}`` for real or complex data.

    :param target: ``(p, n)`` targets.
    :param regressor: ``(q, n)`` regressors.
    :param mu: ridge parameter.
    :return: ``(p, q)`` operator.
    """
    q = regressor.shape[0]
    gram = regressor @ regressor.conj().T + mu * np.eye(q, dtype=regressor.dtype)
    rhs = target @ regressor.conj().T
    return np.linalg.solve(gram.conj().T, rhs.conj().T).conj().T


def identify_memory(
    weights: np.ndarray,
    order: int,
    forcing: Optional[np.ndarray] = None,
    parameterization: str = "s2",
    ridge_mu: float = 1e-3,
    quiescent_threshold: float = 1e-8,
    reduced_rank: int = 32,
    valid: Optional[np.ndarray] = None,
    increment: bool = True,
) -> Tuple[MemoryOperator, np.ndarray]:
    """
    Identify the memory kernel ``{A_j}`` and the input matrix ``B_p`` in closed form.

    Follows Algorithm 3 Regime B (two-stage), lifted to order ``L``:

    * **Stage I** -- the memory blocks are fitted on *quiescent* transitions
      (``p_t = 0``) only, so sparse rain cannot leak into the autonomous kernel.
      With Lahore's rain climatology this retains ~85% of transitions, which is
      exactly why moving Rs/Ta/VPD into the emission (where they belong) rescued
      this regime: with always-on forcing the quiescent set would be empty.
    * **Stage II** -- ``B_p`` is fitted from the residuals on *all* transitions.

    :param weights: ``(T, r)`` weight trajectory on a uniform daily grid.
    :param order: memory order ``L``.
    :param forcing: ``(T, ell)`` input rows, or ``None`` for the autonomous case.
    :param parameterization: ``"s1"``, ``"s2"``, ``"s3"`` or ``"unstructured"``.
    :param ridge_mu: ridge parameter.
    :param quiescent_threshold: rows of ``forcing`` with L1 norm below this count
                                as quiescent.
    :param reduced_rank: rank ``q`` for ``"s3"``.
    :param valid: ``(T,)`` observation validity mask.
    :param increment: fit the **increment** ``w_{t+1} - w_t`` rather than
        ``w_{t+1}`` itself, then set ``A_0 <- A_0 + I``. This is a
        reparameterisation, not a different model -- the representable set is
        identical -- but it moves the ridge's shrinkage target from the *zero*
        operator to the *identity*, i.e. from "predict nothing" to "predict
        persistence". On a daily NDVI record where the field barely moves between
        consecutive dates, shrinking toward zero is shrinking toward a prediction
        that is worse than doing nothing, and the resulting operator has to fight
        back with large blocks -- which is where ``||A_cal|| = 4.13`` came from.
        Physically it also states the right thing: each Koopman mode gets a
        relaxation *rate* rather than an absolute level.
    :return: ``(MemoryOperator, B_p)`` with ``B_p`` of shape ``(r, ell)``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        n_bad = int((~np.isfinite(weights)).any(axis=1).sum())
        raise ValueError(
            "Weight trajectory contains non-finite values at {} of {} steps. This "
            "usually means the generating dynamics diverged (check "
            "sum_j ||A_j||_2 <= 1) or that unobserved dates were left unfilled "
            "instead of being masked via `valid`.".format(n_bad, weights.shape[0])
        )
    r = weights.shape[1]
    w_bar, targets, origins = build_lifted_design(weights, order, 1, valid)
    y = targets[0]  # (n, r) -- w_{t+1}

    if forcing is None:
        ups = np.zeros((origins.size, 0))
    else:
        ups = np.asarray(forcing, dtype=np.float64)[origins]
    ell = ups.shape[1]

    if ell:
        quiescent = np.abs(ups).sum(axis=1) <= quiescent_threshold
        if quiescent.sum() < max(order * 4, 32):
            logger.warning(
                "Only %d quiescent transitions available for Stage I (need >~ %d). "
                "The autonomous kernel is weakly determined; consider joint "
                "identification.",
                int(quiescent.sum()), max(order * 4, 32),
            )
            quiescent = np.ones(origins.size, dtype=bool)
    else:
        quiescent = np.ones(origins.size, dtype=bool)

    # ---- Stage I: memory kernel on quiescent transitions ------------------ #
    # With `increment`, the regression target is w_{t+1} - w_t; adding I back to
    # A_0 afterwards recovers an operator for w_{t+1}. Ridge then shrinks toward
    # persistence instead of toward zero (see the `increment` docstring).
    y_fit = y - w_bar[:, 0, :] if increment else y
    op = _fit_kernel(
        w_bar[quiescent], y_fit[quiescent], parameterization, ridge_mu,
        reduced_rank, r, order,
    )
    if increment:
        blocks = np.array(op.blocks, copy=True)
        blocks[0] = blocks[0] + np.eye(r)
        # The per-mode coefficients must absorb the same ``+I``, or they stop
        # describing the operator that `blocks` now represents -- and
        # `lifted_spectrum` takes its FAST path from them whenever they exist
        # (Prop. 2.11), so every spectral quantity would silently report the
        # spectrum of the *increment* G rather than of A_0 = I + G: the radius,
        # the rho <= rho_max rails that bisect on it, the plotted Koopman
        # spectrum, and the Thm 2.8(ii) null directions.
        #
        # Adding I in modal coordinates is exactly adding 1 to alpha_0, because
        # I = P I P^-1: a real block's scalar gains 1, and a conjugate pair's
        # 2x2 realisation gains [[1,0],[0,1]] since
        # _complex_to_block(a + 1) = _complex_to_block(a) + I_2.
        coeffs = None
        if op.coefficients is not None:
            coeffs = np.array(op.coefficients, copy=True)
            coeffs[:, 0] = coeffs[:, 0] + 1.0
        op = MemoryOperator(
            blocks=blocks, modal=op.modal, coefficients=coeffs,
            parameterization=op.parameterization,
        )

    # ---- Stage II: forcing directions from residuals on ALL transitions --- #
    if ell:
        resid = y - np.einsum("jrs,njs->nr", op.blocks, w_bar)  # (n, r)
        b_p = _ridge(resid.T, ups.T, ridge_mu)  # (r, ell)
    else:
        b_p = np.zeros((r, 0))
    return op, b_p


def _fit_kernel(
    w_bar: np.ndarray,
    y: np.ndarray,
    parameterization: str,
    ridge_mu: float,
    reduced_rank: int,
    r: int,
    order: int,
) -> MemoryOperator:
    """
    Fit ``{A_j}`` under the requested structure.

    :param w_bar: ``(n, L, r)`` lifted regressors.
    :param y: ``(n, r)`` one-step targets.
    :param parameterization: structure key.
    :param ridge_mu: ridge parameter.
    :param reduced_rank: rank for ``"s3"``.
    :param r: latent dimension.
    :param order: memory order.
    :return: the fitted :class:`MemoryOperator`.
    """
    key = parameterization.lower()
    if key == "unstructured":
        x = w_bar.reshape(w_bar.shape[0], order * r).T  # (Lr, n)
        g = _ridge(y.T, x, ridge_mu)  # (r, Lr)
        blocks = np.stack([g[:, j * r : (j + 1) * r] for j in range(order)], axis=0)
        return MemoryOperator(blocks=blocks, parameterization=key)

    # S1/S2/S3 all start from the unconstrained one-step operator A_0, fitted on
    # the current block alone -- the v2 model. Its real modal form supplies the
    # coordinate system the memory then lives in.
    a0 = _ridge(y.T, w_bar[:, 0].T, ridge_mu)  # (r, r)

    if key == "s3":
        return _fit_reduced_rank(w_bar, y, a0, ridge_mu, reduced_rank, r, order)

    modal = real_modal_form(a0)
    z = modal.to_modal(w_bar.reshape(-1, r)).reshape(w_bar.shape)  # (n, L, r)
    zy = modal.to_modal(y)  # (n, r)

    coeffs = np.zeros((modal.n_blocks, order), dtype=complex)
    shared: Optional[np.ndarray] = None
    if key == "s1":
        shared = _fit_shared_lag_weights(z, zy, modal, order, ridge_mu)

    for k, (sl, size) in enumerate(zip(modal.block_slices(), modal.block_sizes)):
        if key == "s1":
            # A_j = alpha_j A: one lag profile shared by every mode, scaled by mu_i.
            coeffs[k] = shared * modal.eigenvalues[k]
            continue
        if size == 1:
            x = z[:, :, sl.start].T  # (L, n) real
            alpha = _ridge(zy[:, sl.start][None, :], x, ridge_mu)[0]
            coeffs[k] = alpha.astype(complex)
        else:
            # Complex coordinate z = x - i y (see module docstring).
            zc = z[:, :, sl.start] - 1j * z[:, :, sl.start + 1]  # (n, L)
            yc = zy[:, sl.start] - 1j * zy[:, sl.start + 1]  # (n,)
            coeffs[k] = _ridge(yc[None, :], zc.T, ridge_mu)[0]

    blocks = _blocks_from_coefficients(coeffs, modal, order)
    return MemoryOperator(
        blocks=blocks, modal=modal, coefficients=coeffs, parameterization=key
    )


def _fit_shared_lag_weights(
    z: np.ndarray,
    zy: np.ndarray,
    modal: RealModalForm,
    order: int,
    ridge_mu: float,
) -> np.ndarray:
    """
    Fit the S1 scalar lag weights ``alpha_j`` in ``A_j = alpha_j A`` (v3 Prop. 2.11).

    Regressing ``z_{t+1}`` on ``{mu_i z^{(i)}_{t-j}}_j`` in modal coordinates makes
    the design a single ``L``-column problem shared by all modes.

    :param z: ``(n, L, r)`` modal regressors.
    :param zy: ``(n, r)`` modal targets.
    :param modal: the modal form.
    :param order: memory order.
    :param ridge_mu: ridge parameter.
    :return: ``(L,)`` real lag weights.
    """
    mu_full = np.zeros(modal.p.shape[0], dtype=complex)
    for k, (sl, size) in enumerate(zip(modal.block_slices(), modal.block_sizes)):
        mu_full[sl] = modal.eigenvalues[k]
    # Stack every mode's equation: target z^{(i)}_{t+1}, regressors mu_i z^{(i)}_{t-j}.
    x = (z * mu_full[None, None, :]).transpose(1, 0, 2).reshape(order, -1)  # (L, n*r)
    t = zy.reshape(1, -1)  # (1, n*r)
    return np.real(_ridge(t, x, ridge_mu)[0])


def _fit_reduced_rank(
    w_bar: np.ndarray,
    y: np.ndarray,
    a0: np.ndarray,
    ridge_mu: float,
    q: int,
    r: int,
    order: int,
) -> MemoryOperator:
    """
    Fit the S3 reduced-rank memory ``A_j = U C_j V^T`` with ``rank <= q``.

    ``A_0`` is left unconstrained (it is the v2 operator); the *memory* blocks
    ``j >= 1`` are the ones that must be regularised, and they are constrained to a
    shared ``q``-dimensional row/column space obtained from the SVD of the stacked
    unconstrained memory fit.

    :param w_bar: ``(n, L, r)`` lifted regressors.
    :param y: ``(n, r)`` targets.
    :param a0: ``(r, r)`` unconstrained one-step operator.
    :param ridge_mu: ridge parameter.
    :param q: target rank.
    :param r: latent dimension.
    :param order: memory order.
    :return: the fitted :class:`MemoryOperator`.
    """
    if order == 1:
        return MemoryOperator(blocks=a0[None], parameterization="s3")
    resid = y - w_bar[:, 0] @ a0.T  # (n, r)
    x = w_bar[:, 1:].reshape(w_bar.shape[0], (order - 1) * r).T  # ((L-1)r, n)
    g = _ridge(resid.T, x, ridge_mu)  # (r, (L-1)r)
    q = int(min(q, r, g.shape[1]))
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    g_low = (u[:, :q] * s[:q]) @ vt[:q]
    blocks = np.concatenate(
        [a0[None], np.stack([g_low[:, j * r : (j + 1) * r] for j in range(order - 1)])],
        axis=0,
    )
    return MemoryOperator(blocks=blocks, parameterization="s3")


def _blocks_from_coefficients(
    coeffs: np.ndarray, modal: RealModalForm, order: int
) -> np.ndarray:
    """
    Reassemble ``A_j = P blkdiag(alpha^{(k)}_j) P^{-1}`` from per-mode coefficients.

    :param coeffs: ``(K, L)`` complex per-mode AR coefficients.
    :param modal: the modal form.
    :param order: memory order.
    :return: ``(L, r, r)`` real memory blocks.
    """
    r = modal.p.shape[0]
    blocks = np.zeros((order, r, r))
    for j in range(order):
        d = np.zeros((r, r))
        for k, (sl, size) in enumerate(zip(modal.block_slices(), modal.block_sizes)):
            if size == 1:
                d[sl.start, sl.start] = coeffs[k, j].real
            else:
                d[sl, sl] = _complex_to_block(coeffs[k, j])
        blocks[j] = modal.p @ d @ modal.p_inv
    return blocks


def memory_profile(op: MemoryOperator) -> Dict[str, np.ndarray]:
    """
    Per-mode memory summary: the physical read-out of the S1/S2 kernel.

    For each mode this reports the AR coefficient magnitudes across lags and the
    resulting **memory half-life** -- the lag at which the kernel magnitude first
    falls below half its lag-0 value. Modes with long half-lives are the ones
    carrying phenological inertia; modes that decay immediately are effectively
    Markov and are what a ``||D_h||``-vs-``L`` sweep will show as excess order.

    :param op: an S1/S2 :class:`MemoryOperator`.
    :return: dict with ``eigenvalues``, ``magnitudes`` ``(K, L)`` and
             ``half_life`` ``(K,)``.
    :raises ValueError: if the operator carries no per-mode coefficients.
    """
    if op.coefficients is None or op.modal is None:
        raise ValueError(
            "memory_profile requires a per-mode (S1/S2) parameterization; got "
            + op.parameterization
        )
    mag = np.abs(op.coefficients)  # (K, L)
    half = np.full(mag.shape[0], float(op.order))
    for k in range(mag.shape[0]):
        ref = mag[k, 0]
        if ref <= 1e-30:
            half[k] = 0.0
            continue
        below = np.nonzero(mag[k] < 0.5 * ref)[0]
        if below.size:
            half[k] = float(below[0])
    return {
        "eigenvalues": op.modal.eigenvalues,
        "magnitudes": mag,
        "half_life": half,
    }
