"""
Direct multi-horizon prediction with semigroup shrinkage (v3 Sec. 2.2.3 - 2.2.5).

For ``h = 1..H`` the direct horizon family is (v3 Def. 2.5)::

    w_{t+h} = Theta_h w_bar_t + sum_{i<h} B^(h)_i p_{t+i} + eps^(h)_t,
    eps^(h)_t ~ N(0, Sigma_h)

where ``[B^(h)_0, ..., B^(h)_{h-1}]`` is the finite-horizon input-to-state Toeplitz
(Markov-parameter) block of subspace identification.

The honest framing
------------------
Theorem 2.10 is an *obstruction*, and this module implements it as such: if
``w_bar_t`` really were Markov, then necessarily ``Theta_h = S A^h`` and the direct
family would carry no information the iterated one lacks. So a nonzero **semigroup
defect** ``D_h = Theta_h - S A^h`` (Def. 2.6) is precisely an admission that the
order-``L`` linear Markov model is misspecified -- memory beyond ``L``,
nonlinearity, or non-stationarity. That makes ``||D_h||`` an *estimable diagnostic*
rather than an embarrassment: sweeping ``L`` and plotting ``||D_h||`` measures the
Mori-Zwanzig memory depth of the field directly (:func:`memory_depth_sweep`).

Why shrinkage rather than either extreme
----------------------------------------
The estimator (v3 Sec. 2.2.5) is::

    G_h = (W_{+h} Z^T + nu_h [S A^h, 0]) (Z Z^T + mu I + nu_h Pi)^{-1}

``nu_h -> inf`` gives the pure iterated predictor (low variance, biased by ``D_h``);
``nu_h = 0`` gives the pure direct one (unbiased, high variance at ``Lr`` parameters
against ~1200 samples). Theorem 2.12 shows the risk
``R(omega) = (1-omega)^2 V_h + omega^2 b_h^2`` is uniquely minimised in the interior,
so the shrunk estimator **strictly dominates both** whenever the defect is nonzero
and finite. That is why "shrunk" is the default and not a hedge.

What the direct family actually buys
------------------------------------
Proposition 2.13: the iterated error covariance is
``Sigma^it_h + b_h Cov(w_bar) b_h^T``, so intervals built by propagating ``Q``
under-cover by exactly the bias term. The direct residual covariance
``Sigma_h = Cov(eps^(h))`` estimates the *full* h-step error and is asymptotically
calibrated. Hence :data:`HorizonFamily.sigma` is always estimated from **direct
residuals** and never by propagating ``Q`` -- the sharpest defensible claim here is
that direct multi-horizon prediction is primarily an *uncertainty-calibration*
device, not an accuracy device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dbwm.dynamics.memory import (
    MemoryOperator,
    build_lifted_design,
    companion_matrix,
    lifted_input_matrix,
    selector_matrix,
)
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.horizons")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class HorizonFamily:
    """
    The identified direct horizon family ``{Theta_h, B^(h), Sigma_h}``.

    :ivar theta: ``(H, r, Lr)`` direct horizon operators.
    :ivar inputs: length-``H`` list; entry ``h-1`` is ``(h, r, ell)`` holding
        ``[B^(h)_0, ..., B^(h)_{h-1}]``.
    :ivar sigma: ``(H, r, r)`` **direct** residual covariances (never propagated
        from ``Q`` -- see Prop. 2.13).
    :ivar nu: ``(H,)`` selected shrinkage strengths.
    :ivar defect: ``(H,)`` semigroup defect norms ``||D_h||_F``.
    :ivar defect_relative: ``(H,)`` ``||D_h||_F / ||S A^h||_F``. Read this with
        care for a dissipative field: ``||S A^h||`` decays geometrically in ``h``,
        so the ratio inflates with the horizon even when the defect itself is flat.
    :ivar defect_normalized: ``(H,)`` the horizon-comparable measure,
        ``sqrt(tr(D_h Cov(w_bar) D_h^T) / tr(Cov(w_{t+h})))`` -- the defect's actual
        effect on predictions, as a fraction of the signal being predicted. This is
        the same quantity that appears as the bias term in Prop. 2.13, and it is the
        one to use when choosing ``L``.
    :ivar theta_iterated: ``(H, r, Lr)`` iterated operators ``S A^h``, kept so the
        iterated/direct/shrunk comparison can be run without refitting.
    :ivar estimator: which estimator produced ``theta``.
    """

    theta: np.ndarray
    inputs: List[np.ndarray]
    sigma: np.ndarray
    nu: np.ndarray
    defect: np.ndarray
    defect_relative: np.ndarray
    theta_iterated: np.ndarray
    defect_normalized: np.ndarray = None
    estimator: str = "shrunk"
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """Maximum horizon ``H``."""
        return self.theta.shape[0]

    def predict(
        self, w_bar: np.ndarray, forcing: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict ``w_{t+1..t+H}`` from one lifted state.

        :param w_bar: ``(L, r)`` history ``[w_t, ..., w_{t-L+1}]`` or ``(Lr,)``.
        :param forcing: ``(H, ell)`` future inputs ``[p_t, ..., p_{t+H-1}]``.
        :return: ``(H, r)`` predicted weights.
        """
        flat = np.asarray(w_bar).reshape(-1)
        out = np.stack([self.theta[h] @ flat for h in range(self.horizon)], axis=0)
        if forcing is not None:
            forcing = np.asarray(forcing)
            for h in range(self.horizon):
                for i in range(h + 1):
                    out[h] = out[h] + self.inputs[h][i] @ forcing[i]
        return out


# --------------------------------------------------------------------------- #
# Iterated (semigroup-consistent) values
# --------------------------------------------------------------------------- #
def iterated_family(
    op: MemoryOperator,
    b_p: np.ndarray,
    horizon: int,
    q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """
    Semigroup-consistent operators ``Theta^it_h = S A^h`` and their Toeplitz blocks.

    Per v3 Sec. 2.2.3::

        Theta^it_h    = S A^h
        B^(h),it_i    = S A^{h-1-i} B_cal
        Sigma^it_h    = sum_{i<h} S A^i E Q E^T (A^i)^T S^T

    ``Sigma^it_h`` is returned only so :func:`coverage_comparison` can demonstrate
    that it **under-covers** (Prop. 2.13). It is never used to build a reported
    predictive interval.

    :param op: the identified memory operator.
    :param b_p: ``(r, ell)`` input matrix.
    :param horizon: maximum horizon ``H``.
    :param q: ``(r, r)`` process-noise covariance, or ``None`` to skip ``Sigma^it``.
    :return: ``(theta_it, inputs_it, sigma_it)``.
    """
    a_cal = op.companion()
    s = selector_matrix(op.order, op.r)
    b_cal = lifted_input_matrix(op.order, np.asarray(b_p))
    e_cal = s.T

    powers = [np.eye(a_cal.shape[0])]
    for _ in range(horizon):
        powers.append(powers[-1] @ a_cal)

    theta_it = np.stack([s @ powers[h] for h in range(1, horizon + 1)], axis=0)
    inputs_it = [
        np.stack([s @ powers[h - 1 - i] @ b_cal for i in range(h)], axis=0)
        for h in range(1, horizon + 1)
    ]

    sigma_it = None
    if q is not None:
        q = np.asarray(q)
        sigma_it = np.zeros((horizon, op.r, op.r))
        acc = np.zeros((op.r, op.r))
        for h in range(horizon):
            m = s @ powers[h] @ e_cal  # (r, r)
            acc = acc + m @ q @ m.T
            sigma_it[h] = acc
    return theta_it, inputs_it, sigma_it


# --------------------------------------------------------------------------- #
# Design assembly
# --------------------------------------------------------------------------- #
def build_horizon_design(
    weights: np.ndarray,
    order: int,
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Build ``Z_t = [w_bar_t; p_t; ...; p_{t+H-1}]`` and the targets ``W_{+h}``.

    :param weights: ``(T, r)`` weight trajectory on a uniform grid.
    :param order: memory order ``L``.
    :param horizon: maximum horizon ``H``.
    :param forcing: ``(T, ell)`` input rows, or ``None``.
    :param valid: ``(T,)`` observation validity mask.
    :return: dict with ``w_bar`` ``(n, L, r)``, ``targets`` ``(H, n, r)``,
             ``future_forcing`` ``(n, H, ell)`` and ``origins`` ``(n,)``.
    """
    w_bar, targets, origins = build_lifted_design(weights, order, horizon, valid)
    if forcing is None:
        fut = np.zeros((origins.size, horizon, 0))
    else:
        f = np.asarray(forcing, dtype=np.float64)
        fut = np.stack([f[origins + i] for i in range(horizon)], axis=1)
    return {
        "w_bar": w_bar,
        "targets": targets,
        "future_forcing": fut,
        "origins": origins,
    }


# --------------------------------------------------------------------------- #
# The shrunk estimator
# --------------------------------------------------------------------------- #
def _fit_one_horizon(
    z: np.ndarray,
    y: np.ndarray,
    theta_target: np.ndarray,
    input_target: np.ndarray,
    nu: float,
    mu: float,
    shrink_inputs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form semigroup-shrunk ridge for a single horizon (v3 Sec. 2.2.5).

    ``G_h = (W_{+h} Z^T + nu [S A^h, 0]) (Z Z^T + mu I + nu Pi)^{-1}``.

    :param z: ``(d, n)`` stacked regressor ``[w_bar; future forcing]``.
    :param y: ``(r, n)`` targets ``w_{t+h}``.
    :param theta_target: ``(r, Lr)`` iterated shrinkage target ``S A^h``.
    :param input_target: ``(r, h*ell)`` iterated Toeplitz target, flattened.
    :param nu: shrinkage strength ``nu_h``.
    :param mu: ridge parameter.
    :param shrink_inputs: also shrink the input block toward its iterated value.
        v3 writes ``Pi = diag(I_Lr, 0)``, i.e. ``False``; ``True`` shrinks the whole
        predictor coherently toward the iterated one.
    :return: ``(Theta_h, B-block)`` of shapes ``(r, Lr)`` and ``(r, h*ell)``.
    """
    d = z.shape[0]
    n_theta = theta_target.shape[1]

    if not np.isfinite(nu):
        # nu -> inf is the pure iterated predictor. Handled in closed form rather
        # than by a large finite value, so the endpoint of the shrinkage path is
        # exactly representable and the selector can actually reach it.
        if shrink_inputs:
            return theta_target, input_target
        resid = y - theta_target @ z[:n_theta]
        zi = z[n_theta:]
        if zi.shape[0] == 0:
            return theta_target, np.zeros((y.shape[0], 0))
        gram = zi @ zi.T + mu * np.eye(zi.shape[0])
        return theta_target, np.linalg.solve(gram.T, (resid @ zi.T).T).T

    pi = np.zeros(d)
    pi[:n_theta] = 1.0
    if shrink_inputs:
        pi[n_theta:] = 1.0
    target = np.concatenate(
        [theta_target, input_target if shrink_inputs else np.zeros_like(input_target)],
        axis=1,
    )
    gram = z @ z.T + mu * np.eye(d) + nu * np.diag(pi)
    rhs = y @ z.T + nu * target
    g = np.linalg.solve(gram.T, rhs.T).T
    return g[:, :n_theta], g[:, n_theta:]


def _nu_score(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_va: np.ndarray,
    y_va: np.ndarray,
    theta_target: np.ndarray,
    input_target: np.ndarray,
    nu: float,
    mu: float,
    shrink_inputs: bool,
    criterion: str,
) -> float:
    """
    Score one candidate ``nu`` (lower is better).

    ``"gcv"`` uses in-sample generalized cross-validation with the effective
    degrees of freedom of the shrunk hat matrix; ``"holdout"`` uses held-out MSE;
    ``"innovation_likelihood"`` uses the held-out Gaussian NLL under the direct
    residual covariance, which is what the calibration claim of Prop. 2.13 actually
    cares about.

    :param z_tr: ``(d, n_tr)`` fitting regressors.
    :param y_tr: ``(r, n_tr)`` fitting targets.
    :param z_va: ``(d, n_va)`` validation regressors.
    :param y_va: ``(r, n_va)`` validation targets.
    :param theta_target: iterated ``Theta`` target.
    :param input_target: iterated input target.
    :param nu: candidate shrinkage.
    :param mu: ridge.
    :param shrink_inputs: see :func:`_fit_one_horizon`.
    :param criterion: ``"gcv"``, ``"holdout"`` or ``"innovation_likelihood"``.
    :return: the score.
    """
    th, bb = _fit_one_horizon(
        z_tr, y_tr, theta_target, input_target, nu, mu, shrink_inputs
    )
    g = np.concatenate([th, bb], axis=1)

    if criterion == "gcv":
        d, n = z_tr.shape
        pi = np.zeros(d)
        pi[: theta_target.shape[1]] = 1.0
        if shrink_inputs:
            pi[theta_target.shape[1]:] = 1.0
        if np.isfinite(nu):
            gram = z_tr @ z_tr.T + mu * np.eye(d) + nu * np.diag(pi)
            dof = float(np.trace(np.linalg.solve(gram, z_tr @ z_tr.T)))
        else:
            # Fully shrunk directions contribute no degrees of freedom.
            free = pi == 0.0
            dof = float(free.sum())
        resid = y_tr - g @ z_tr
        denom = max(n - dof, 1e-6)
        return float(n * np.sum(resid**2) / (denom**2))

    resid = y_va - g @ z_va
    if criterion == "holdout":
        return float(np.mean(resid**2))
    if criterion == "innovation_likelihood":
        cov = (resid @ resid.T) / max(resid.shape[1] - 1, 1)
        cov = cov + 1e-8 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return np.inf
        sol = np.linalg.solve(cov, resid)
        return float(0.5 * (np.sum(resid * sol) / resid.shape[1] + logdet))
    raise ValueError("unknown nu selection criterion: {}".format(criterion))


def fit_horizon_family(
    weights: np.ndarray,
    op: MemoryOperator,
    b_p: np.ndarray,
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    estimator: str = "shrunk",
    ridge_mu: float = 1e-3,
    nu: Optional[Sequence[float]] = None,
    nu_grid: Sequence[float] = (0.0, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, np.inf),
    nu_selection: str = "gcv",
    nu_holdout_fraction: float = 0.15,
    shrink_inputs: bool = False,
    q: Optional[np.ndarray] = None,
) -> HorizonFamily:
    """
    Identify ``{Theta_h, B^(h), Sigma_h}`` in closed form for ``h = 1..H``.

    Preserves the "no SGD, closed-form identification" property of v2 Algorithm 3:
    ``Theta_1`` and the companion come from :func:`~dbwm.dynamics.memory.identify_memory`,
    and the whole family follows from one linear solve per horizon.

    :param weights: ``(T, r)`` weight trajectory on a uniform grid.
    :param op: the identified memory operator (supplies the shrinkage target).
    :param b_p: ``(r, ell)`` input matrix.
    :param horizon: maximum horizon ``H``.
    :param forcing: ``(T, ell)`` input rows, or ``None``.
    :param valid: ``(T,)`` observation validity mask.
    :param estimator: ``"shrunk"`` (default), ``"direct"`` (``nu = 0``) or
                      ``"iterated"`` (``nu -> inf``).
    :param ridge_mu: ridge parameter ``mu``.
    :param nu: explicit per-horizon shrinkage, or ``None`` to select.
    :param nu_grid: candidate values for the selection sweep.
    :param nu_selection: ``"gcv"``, ``"holdout"`` or ``"innovation_likelihood"``.
    :param nu_holdout_fraction: tail fraction of the training design held out for
        selection. Held out **chronologically**, not at random: a random holdout
        with overlapping memory windows leaks.
    :param shrink_inputs: also shrink the Toeplitz input block.
    :param q: ``(r, r)`` process noise, used only for the iterated-covariance
              comparison diagnostics.
    :return: the identified :class:`HorizonFamily`.
    """
    design = build_horizon_design(weights, op.order, horizon, forcing, valid)
    w_bar = design["w_bar"]  # (n, L, r)
    targets = design["targets"]  # (H, n, r)
    fut = design["future_forcing"]  # (n, H, ell)
    n = w_bar.shape[0]
    r, order = op.r, op.order
    ell = fut.shape[2]

    theta_it, inputs_it, sigma_it = iterated_family(op, b_p, horizon, q)

    if estimator == "iterated":
        sigma = _direct_residual_covariance(
            theta_it, inputs_it, w_bar, targets, fut, horizon
        )
        return HorizonFamily(
            theta=theta_it,
            inputs=inputs_it,
            sigma=sigma,
            nu=np.full(horizon, np.inf),
            defect=np.zeros(horizon),
            defect_relative=np.zeros(horizon),
            defect_normalized=np.zeros(horizon),
            theta_iterated=theta_it,
            estimator="iterated",
            diagnostics={"n_windows": n, "sigma_iterated": sigma_it},
        )

    flat_bar = w_bar.reshape(n, order * r).T  # (Lr, n)
    n_val = max(1, int(round(nu_holdout_fraction * n))) if nu is None else 0
    n_fit = n - n_val

    theta_out = np.zeros((horizon, r, order * r))
    inputs_out: List[np.ndarray] = []
    nu_out = np.zeros(horizon)

    for h in range(1, horizon + 1):
        z = np.concatenate([flat_bar, fut[:, :h].reshape(n, h * ell).T], axis=0)
        y = targets[h - 1].T  # (r, n)
        th_target = theta_it[h - 1]
        in_target = inputs_it[h - 1].transpose(1, 0, 2).reshape(r, h * ell)

        if estimator == "direct":
            nu_h = 0.0
        elif nu is not None:
            nu_h = float(nu[h - 1])
        else:
            scores = [
                _nu_score(
                    z[:, :n_fit], y[:, :n_fit], z[:, n_fit:], y[:, n_fit:],
                    th_target, in_target, float(c), ridge_mu, shrink_inputs,
                    nu_selection,
                )
                for c in nu_grid
            ]
            nu_h = float(nu_grid[int(np.argmin(scores))])

        th, bb = _fit_one_horizon(
            z, y, th_target, in_target, nu_h, ridge_mu, shrink_inputs
        )
        theta_out[h - 1] = th
        inputs_out.append(bb.reshape(r, h, ell).transpose(1, 0, 2))
        nu_out[h - 1] = nu_h

    sigma = _direct_residual_covariance(
        theta_out, inputs_out, w_bar, targets, fut, horizon
    )
    defect = np.array(
        [np.linalg.norm(theta_out[h] - theta_it[h]) for h in range(horizon)]
    )
    scale = np.array([max(np.linalg.norm(theta_it[h]), 1e-30) for h in range(horizon)])
    defect_norm = _normalized_defect(theta_out, theta_it, w_bar, targets)

    logger.info(
        "Horizon family (%s): nu = %s", estimator,
        np.array2string(nu_out, precision=3, suppress_small=True),
    )
    logger.info(
        "Semigroup defect ||D_h||_F = %s | normalized (effect on prediction) %s",
        np.array2string(defect, precision=4, suppress_small=True),
        np.array2string(defect_norm, precision=4, suppress_small=True),
    )
    return HorizonFamily(
        theta=theta_out,
        inputs=inputs_out,
        sigma=sigma,
        nu=nu_out,
        defect=defect,
        defect_relative=defect / scale,
        defect_normalized=defect_norm,
        theta_iterated=theta_it,
        estimator=estimator,
        diagnostics={"n_windows": n, "n_holdout": n_val, "sigma_iterated": sigma_it},
    )


def _normalized_defect(
    theta: np.ndarray,
    theta_it: np.ndarray,
    w_bar: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Horizon-comparable defect: the effect of ``D_h`` on predictions, normalised by
    the signal being predicted.

    ``sqrt(tr(D_h Cov(w_bar) D_h^T) / tr(Cov(w_{t+h})))``. Unlike
    ``||D_h||_F / ||S A^h||_F`` this does not inflate with ``h`` merely because the
    operator decays, so it is the measure to use for the ``L`` sweep on a
    dissipative field.

    :param theta: ``(H, r, Lr)`` fitted horizon operators.
    :param theta_it: ``(H, r, Lr)`` iterated operators.
    :param w_bar: ``(n, L, r)`` lifted regressors.
    :param targets: ``(H, n, r)`` targets.
    :return: ``(H,)`` normalised defects.
    """
    n, order, r = w_bar.shape
    flat = w_bar.reshape(n, order * r)
    cov_bar = np.cov(flat, rowvar=False)
    horizon = theta.shape[0]
    out = np.zeros(horizon)
    for h in range(horizon):
        d_h = theta[h] - theta_it[h]
        num = float(np.trace(d_h @ cov_bar @ d_h.T))
        den = float(np.sum(np.var(targets[h], axis=0)))
        out[h] = np.sqrt(max(num, 0.0) / max(den, 1e-30))
    return out


def _direct_residual_covariance(
    theta: np.ndarray,
    inputs: List[np.ndarray],
    w_bar: np.ndarray,
    targets: np.ndarray,
    fut: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Estimate ``Sigma_h`` from the **direct** ``h``-step residuals (v3 step 6).

    Deliberately not built by propagating ``Q``: Proposition 2.13 shows the
    propagated covariance is short by exactly ``b_h Cov(w_bar) b_h^T``, so intervals
    derived from it under-cover whenever the semigroup defect is nonzero.

    :param theta: ``(H, r, Lr)`` horizon operators.
    :param inputs: per-horizon Toeplitz blocks.
    :param w_bar: ``(n, L, r)`` lifted regressors.
    :param targets: ``(H, n, r)`` targets.
    :param fut: ``(n, H, ell)`` future inputs.
    :param horizon: ``H``.
    :return: ``(H, r, r)`` residual covariances.
    """
    n, order, r = w_bar.shape
    flat = w_bar.reshape(n, order * r)
    out = np.zeros((horizon, r, r))
    for h in range(horizon):
        pred = flat @ theta[h].T
        for i in range(h + 1):
            pred = pred + fut[:, i] @ inputs[h][i].T
        resid = targets[h] - pred
        out[h] = (resid.T @ resid) / max(n - 1, 1)
    return out


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def semigroup_defect(family: HorizonFamily) -> Dict[str, np.ndarray]:
    """
    Report the semigroup defect ``D_h = Theta_h - S A^h`` (v3 Def. 2.6, ``h >= 2``).

    Under Theorem 2.10, ``D_h == 0`` for all ``h`` iff the lifted state is genuinely
    Markov of order ``L``. Its magnitude measures how far the process departs from
    that model, and by Theorem 2.12 it is also the quantity that sets the optimal
    shrinkage and the value of multi-origin fusion.

    ``h = 1`` is reported **separately** as ``structural_residual`` and is *not* part
    of the defect. v3 Def. 2.6 starts at ``h = 2`` for a reason: the top block row of
    ``A`` is ``Theta_1`` by construction, so ``S A^1 == Theta_1`` identically and no
    semigroup property is being tested at ``h = 1``. Any nonzero value there measures
    something else entirely -- the gap between the *structured* (S1/S2) kernel fitted
    on quiescent transitions and an unstructured one-step fit over all windows.
    Folding it into the defect would misattribute a parameterization choice to a
    Markov violation.

    :param family: the identified family.
    :return: dict with ``horizons`` (``2..H``), ``defect``, ``defect_relative`` and
             the scalar ``structural_residual`` / ``structural_residual_relative``.
    """
    return {
        "horizons": np.arange(2, family.horizon + 1),
        "defect": family.defect[1:],
        "defect_relative": family.defect_relative[1:],
        "defect_normalized": family.defect_normalized[1:],
        "structural_residual": float(family.defect[0]),
        "structural_residual_relative": float(family.defect_relative[0]),
    }


def memory_depth_sweep(
    weights: np.ndarray,
    orders: Sequence[int],
    horizon: int,
    forcing: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    parameterization: str = "s2",
    ridge_mu: float = 1e-3,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Sweep the memory order and record ``||D_h||`` -- the Mori-Zwanzig memory depth.

    This is the standalone physical measurement of v3 Sec. 2.2.3: under Prop. 2.5
    the defect decays as ``L`` grows past the decay length of the memory kernel, so
    the elbow of ``||D_h||`` versus ``L`` *is* the memory depth of the field. It has
    value independently of any forecasting gain, and it is the principled way to
    choose ``L`` given that Theorem 2.8 makes over-lagging actively harmful.

    :param weights: ``(T, r)`` weight trajectory.
    :param orders: memory orders to try.
    :param horizon: maximum horizon ``H``.
    :param forcing: ``(T, ell)`` inputs.
    :param valid: ``(T,)`` validity mask.
    :param parameterization: memory structure.
    :param ridge_mu: ridge parameter.
    :param kwargs: forwarded to :func:`fit_horizon_family`.
    :return: dict with ``orders``, ``defect`` ``(n_orders, H)``,
             ``defect_relative``, and ``a_last_relative_smin`` ``(n_orders,)``.
    """
    from dbwm.dynamics.memory import identify_memory, observability_certificate

    defects, rel, norm, smins = [], [], [], []
    for l in orders:
        op, b_p = identify_memory(
            weights, l, forcing, parameterization, ridge_mu, valid=valid
        )
        fam = fit_horizon_family(
            weights, op, b_p, horizon, forcing, valid,
            ridge_mu=ridge_mu, **kwargs,
        )
        cert = observability_certificate(op, np.eye(op.r))
        defects.append(fam.defect)
        rel.append(fam.defect_relative)
        norm.append(fam.defect_normalized)
        smins.append(cert["a_last_relative_smin"])
        logger.info(
            "L=%d: normalized ||D_h||=%s  s_min(A_last)/||A_0||=%.2e",
            l,
            np.array2string(fam.defect_normalized, precision=4, suppress_small=True),
            cert["a_last_relative_smin"],
        )
    return {
        "orders": np.asarray(orders),
        "defect": np.stack(defects, axis=0),
        "defect_relative": np.stack(rel, axis=0),
        "defect_normalized": np.stack(norm, axis=0),
        "a_last_relative_smin": np.asarray(smins),
    }


def coverage_comparison(
    family: HorizonFamily,
    weights: np.ndarray,
    op: MemoryOperator,
    b_p: np.ndarray,
    forcing: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Quantify Proposition 2.13: by how much does the iterated covariance under-cover?

    Compares, per horizon, the trace of the direct residual covariance
    ``Sigma_h`` against the trace of the propagated ``Sigma^it_h``. Prop. 2.13
    predicts ``Sigma_h >= Sigma^it_h`` with the gap equal to
    ``b_h Cov(w_bar) b_h^T``; this returns the measured gap alongside the predicted
    one so the identity can be checked rather than assumed.

    The comparison Prop. 2.13 makes is between the **iterated predictor's own**
    error covariance and the covariance one would quote for it -- *not* between the
    direct family's residuals and ``Sigma^it_h``. The direct predictor is the better
    predictor, so its residual covariance is naturally smaller; comparing that
    against ``Sigma^it_h`` would test nothing and comes out backwards.

    The identity that is exactly true and estimable
    ----------------------------------------------
    Writing the true direct relation ``w_{t+h} = Theta^o_h w_bar_t + eps^(h)`` with
    ``eps^(h)`` orthogonal to ``w_bar_t``, the iterated error is
    ``b_h w_bar_t + eps^(h)`` with ``b_h = Theta^o_h - S A^h``, and the cross term
    vanishes. Hence::

        Cov(iterated error) = Sigma_h^direct + b_h Cov(w_bar) b_h^T   >=  Sigma_h^direct

    The baseline in that identity is the **direct residual covariance**, not the
    ``Q``-propagated ``Sigma^it_h``. The two coincide only when the order-``L``
    linear Markov model is correctly specified -- which is precisely what is in
    question -- so this function checks the estimable form and reports the
    propagated version separately as an *additional*, distinct source of
    miscalibration (it can err in either direction, since propagating ``Q`` treats
    a state-dependent residual as white).

    :param family: the identified family (must carry ``sigma_iterated``).
    :param weights: ``(T, r)`` weight trajectory.
    :param op: the memory operator.
    :param b_p: ``(r, ell)`` input matrix.
    :param forcing: ``(T, ell)`` inputs.
    :param valid: ``(T,)`` validity mask.
    :return: dict with ``trace_direct`` (the calibrated baseline),
             ``trace_iterated_actual``, ``trace_bias_predicted``,
             ``under_coverage_ratio`` (actual / direct, ``> 1`` when the defect
             bites), ``identity_residual`` (how well
             ``actual = direct + bias`` holds), plus ``trace_iterated_nominal``
             and ``propagated_ratio`` for the ``Q``-propagated version.
    """
    sigma_it = family.diagnostics.get("sigma_iterated")
    if sigma_it is None:
        raise ValueError(
            "coverage_comparison needs the iterated covariance; refit with q=Q."
        )
    horizon = family.horizon
    design = build_horizon_design(weights, op.order, horizon, forcing, valid)
    w_bar = design["w_bar"]
    targets = design["targets"]
    fut = design["future_forcing"]
    n, order, r = w_bar.shape
    flat = w_bar.reshape(n, order * r)
    cov_bar = np.cov(flat, rowvar=False)

    _, inputs_it, _ = iterated_family(op, b_p, horizon)

    tr_dir = np.zeros(horizon)
    tr_nom = np.zeros(horizon)
    tr_act = np.zeros(horizon)
    tr_bias = np.zeros(horizon)
    for h in range(horizon):
        # The iterated predictor's ACTUAL h-step error on this record.
        pred_it = flat @ family.theta_iterated[h].T
        for i in range(h + 1):
            pred_it = pred_it + fut[:, i] @ inputs_it[h][i].T
        resid_it = targets[h] - pred_it
        tr_act[h] = np.trace((resid_it.T @ resid_it) / max(n - 1, 1))
        tr_nom[h] = np.trace(sigma_it[h])
        tr_dir[h] = np.trace(family.sigma[h])
        b_h = family.theta[h] - family.theta_iterated[h]
        tr_bias[h] = np.trace(b_h @ cov_bar @ b_h.T)

    return {
        "horizons": np.arange(1, horizon + 1),
        "trace_direct": tr_dir,
        "trace_iterated_actual": tr_act,
        "trace_bias_predicted": tr_bias,
        # The estimable Prop. 2.13 statement: the iterated predictor's real error
        # exceeds the calibrated (direct) covariance by exactly the bias term.
        "under_coverage_ratio": tr_act / np.maximum(tr_dir, 1e-30),
        "identity_residual": tr_act - (tr_dir + tr_bias),
        # The Q-propagated covariance is a second, separate miscalibration: it
        # treats a state-dependent residual as white, so it can err either way.
        "trace_iterated_nominal": tr_nom,
        "propagated_ratio": tr_act / np.maximum(tr_nom, 1e-30),
    }
