"""
Tests for the direct multi-horizon family (v3 Sec. 2.2.3 - 2.2.5).

Pins the three claims that carry the framework: Theorem 2.10 (the semigroup
obstruction), Theorem 2.12 (shrinkage dominance) and Proposition 2.13 (the
iterated covariance under-covers).
"""
import numpy as np
import pytest

from dbwm.dynamics import memory as M
from dbwm.dynamics import multihorizon as MH
from tests.test_memory import make_s2_system, simulate


def simulate_nonlinear(op, n, seed=0, strength=0.15, noise=0.4):
    """Drive the system with a mild nonlinearity so the semigroup defect is nonzero."""
    rng = np.random.RandomState(seed)
    w = np.zeros((n, op.r))
    w[: op.order] = rng.randn(op.order, op.r)
    for t in range(op.order - 1, n - 1):
        w[t + 1] = (
            op.predict(w[t::-1][: op.order])
            + strength * np.tanh(3.0 * w[t, 0])
            + noise * rng.randn(op.r)
        )
    return w


# --------------------------------------------------------------------------- #
# Iterated family
# --------------------------------------------------------------------------- #
def test_iterated_family_matches_repeated_application():
    """Theta^it_h = S A^h really does propagate h steps."""
    op = make_s2_system(order=3)
    b_p = np.zeros((op.r, 0))
    theta_it, _, _ = MH.iterated_family(op, b_p, horizon=4)
    rng = np.random.RandomState(0)
    hist = rng.randn(op.order, op.r)
    rolled = hist.copy()
    for h in range(4):
        nxt = op.predict(rolled)
        rolled = np.concatenate([nxt[None], rolled[:-1]], axis=0)
        assert np.allclose(theta_it[h] @ hist.reshape(-1), nxt, atol=1e-9)


def test_iterated_toeplitz_block_matches_forced_rollout():
    """The Markov-parameter block reproduces a forced rollout exactly."""
    op = make_s2_system(order=2)
    rng = np.random.RandomState(1)
    b_p = rng.randn(op.r, 1)
    horizon = 3
    theta_it, inputs_it, _ = MH.iterated_family(op, b_p, horizon)
    hist = rng.randn(op.order, op.r)
    p = rng.randn(horizon, 1)

    rolled = hist.copy()
    for h in range(horizon):
        nxt = op.predict(rolled) + b_p @ p[h]
        rolled = np.concatenate([nxt[None], rolled[:-1]], axis=0)
        pred = theta_it[h] @ hist.reshape(-1)
        for i in range(h + 1):
            pred = pred + inputs_it[h][i] @ p[i]
        assert np.allclose(pred, nxt, atol=1e-9)


# --------------------------------------------------------------------------- #
# Theorem 2.10: the semigroup obstruction
# --------------------------------------------------------------------------- #
def test_theorem_2_10_defect_vanishes_for_an_exactly_markov_process():
    """If w_bar really is Markov of order L, then Theta_h == S A^h."""
    truth = make_s2_system(order=3)
    w = simulate(truth, 40_000, seed=2, noise=1.0)
    op, b_p = M.identify_memory(w, 3, parameterization="s2", ridge_mu=1e-10)
    fam = MH.fit_horizon_family(
        w, op, b_p, horizon=4, estimator="direct", ridge_mu=1e-10
    )
    d = MH.semigroup_defect(fam)
    # Judged on the horizon-comparable measure. ||D_h|| / ||S A^h|| is unusable
    # here: this system is dissipative, so the denominator decays geometrically
    # and the ratio inflates with h even when the defect itself is flat.
    assert np.all(d["defect_normalized"] < 0.05)


def test_defect_grows_when_the_memory_order_is_truncated():
    """Fitting L below the true order leaves memory in the residual -> D_h > 0."""
    truth = make_s2_system(order=3)
    w = simulate(truth, 40_000, seed=3, noise=1.0)
    rel = {}
    for l in (1, 3):
        op, b_p = M.identify_memory(w, l, parameterization="s2", ridge_mu=1e-10)
        fam = MH.fit_horizon_family(
            w, op, b_p, horizon=4, estimator="direct", ridge_mu=1e-10
        )
        rel[l] = MH.semigroup_defect(fam)["defect_normalized"]
    # Under-lagging shows a materially larger defect at every horizon.
    assert np.all(rel[1] > rel[3])
    assert rel[1][-1] > 3.0 * rel[3][-1]


def test_semigroup_defect_excludes_h_equals_one():
    """D_h is defined for h >= 2; h = 1 is reported separately.

    S A^1 == Theta_1 identically by construction, so nothing about the semigroup
    property is tested at h = 1. Any value there measures the S1/S2 structural
    constraint instead, and folding it in would misattribute a parameterization
    choice to a Markov violation.
    """
    truth = make_s2_system(order=2)
    w = simulate(truth, 5_000, seed=4)
    op, b_p = M.identify_memory(w, 2, parameterization="s2")
    fam = MH.fit_horizon_family(w, op, b_p, horizon=3, estimator="direct")
    d = MH.semigroup_defect(fam)
    assert list(d["horizons"]) == [2, 3]
    assert d["defect"].size == 2
    assert "structural_residual" in d


def test_memory_depth_sweep_locates_the_true_order():
    """The ||D_h||-vs-L elbow sits at the generating memory order."""
    truth = make_s2_system(order=3)
    w = simulate(truth, 20_000, seed=5, noise=1.0)
    sweep = MH.memory_depth_sweep(
        w, orders=(1, 2, 3, 4), horizon=3, estimator="direct", ridge_mu=1e-10
    )
    last_h = sweep["defect_normalized"][:, -1]
    assert last_h[0] > last_h[1] > last_h[2]        # improving up to the true order
    assert last_h[3] >= last_h[2] * 0.8             # no real gain beyond it
    # And the Thm 2.8(i) ratio collapses once L over-shoots.
    assert sweep["a_last_relative_smin"][2] > 10.0 * sweep["a_last_relative_smin"][3]


# --------------------------------------------------------------------------- #
# Theorem 2.12: shrinkage dominance
# --------------------------------------------------------------------------- #
def test_shrinkage_endpoints_are_exactly_representable():
    """nu = 0 gives the direct fit; nu = inf gives the iterated one, exactly."""
    truth = make_s2_system(order=2)
    w = simulate_nonlinear(truth, 800, seed=6)
    op, b_p = M.identify_memory(w, 2, parameterization="s2")

    direct = MH.fit_horizon_family(w, op, b_p, 3, estimator="direct")
    iterated = MH.fit_horizon_family(w, op, b_p, 3, estimator="iterated")
    at_zero = MH.fit_horizon_family(w, op, b_p, 3, nu=[0.0] * 3)
    at_inf = MH.fit_horizon_family(w, op, b_p, 3, nu=[np.inf] * 3)

    assert np.allclose(at_zero.theta, direct.theta, atol=1e-8)
    assert np.allclose(at_inf.theta, iterated.theta, atol=1e-8)
    assert np.allclose(at_inf.theta, iterated.theta_iterated, atol=1e-8)


def test_theorem_2_12_shrunk_dominates_both_extremes_in_expected_risk():
    """Shrinkage lowers the AVERAGE held-out risk relative to both endpoints.

    Theorem 2.12 is a statement about *risk* -- an expectation over training draws
    -- so it is tested as one, by averaging over independent training sets. On any
    single draw the empirical nu-selector can pick a slightly suboptimal value and
    lose to an endpoint by a hair; asserting per-draw dominance would be asserting
    something the theorem does not claim.
    """
    truth = make_s2_system(order=2)
    horizon, order = 4, 2
    test = simulate_nonlinear(truth, 6_000, seed=99)
    design = MH.build_horizon_design(test, order, horizon)
    n = design["w_bar"].shape[0]
    flat = design["w_bar"].reshape(n, order * truth.r)

    risk = {est: [] for est in ("direct", "iterated", "shrunk")}
    for seed in range(6):
        train = simulate_nonlinear(truth, 500, seed=100 + seed)
        op, b_p = M.identify_memory(train, order, parameterization="s2", ridge_mu=1e-3)
        for est in risk:
            fam = MH.fit_horizon_family(
                train, op, b_p, horizon, estimator=est, ridge_mu=1e-3,
                nu_selection="holdout",
            )
            risk[est].append(
                np.array(
                    [
                        np.mean((design["targets"][h] - flat @ fam.theta[h].T) ** 2)
                        for h in range(horizon)
                    ]
                )
            )
    mean = {est: np.mean(np.stack(v), axis=0) for est, v in risk.items()}
    assert np.all(mean["shrunk"] <= mean["direct"] + 1e-9)
    assert np.all(mean["shrunk"] <= mean["iterated"] + 1e-9)
    # Shrinkage buys something real against the high-variance endpoint.
    assert np.any(mean["shrunk"] < mean["direct"])


@pytest.mark.parametrize("criterion", ["gcv", "holdout", "innovation_likelihood"])
def test_nu_selection_criteria_all_run_and_stay_in_grid(criterion):
    truth = make_s2_system(order=2)
    w = simulate_nonlinear(truth, 500, seed=8)
    op, b_p = M.identify_memory(w, 2, parameterization="s2")
    grid = (0.0, 1.0, 1e2, np.inf)
    fam = MH.fit_horizon_family(
        w, op, b_p, 3, nu_grid=grid, nu_selection=criterion
    )
    assert all(v in grid for v in fam.nu)


def test_unknown_nu_criterion_raises():
    truth = make_s2_system(order=2)
    w = simulate(truth, 400, seed=9)
    op, b_p = M.identify_memory(w, 2, parameterization="s2")
    with pytest.raises(ValueError, match="unknown nu selection criterion"):
        MH.fit_horizon_family(w, op, b_p, 2, nu_selection="nonsense")


# --------------------------------------------------------------------------- #
# Proposition 2.13: the iterated covariance under-covers
# --------------------------------------------------------------------------- #
def test_prop_2_13_iterated_covariance_under_covers():
    """The ITERATED predictor's real error exceeds the Sigma^it_h quoted for it.

    The comparison is between the iterated predictor's actual error covariance and
    the covariance one would nominally report for it by propagating Q -- NOT
    between the direct family's residuals and Sigma^it_h. The direct predictor is
    the better predictor, so its residual covariance is naturally smaller, and
    comparing that against Sigma^it_h would test nothing.

    This is the headline empirical claim: in an observer-corrected latent world
    model, the direct family is primarily an uncertainty-CALIBRATION device.
    """
    truth = make_s2_system(order=2)
    w = simulate_nonlinear(truth, 4_000, seed=10, strength=0.6, noise=0.4)
    op, b_p = M.identify_memory(w, 2, parameterization="s2", ridge_mu=1e-6)
    hist = np.stack([w[1:-1], w[:-2]], axis=1)
    q = np.cov((w[2:] - np.einsum("jrs,njs->nr", op.blocks, hist)).T)
    fam = MH.fit_horizon_family(
        w, op, b_p, 4, estimator="direct", ridge_mu=1e-6, q=q
    )
    cmp = MH.coverage_comparison(fam, w, op, b_p)

    # The iterated predictor's real error exceeds the calibrated (direct) baseline.
    assert np.all(cmp["under_coverage_ratio"] > 1.0)
    assert np.all(cmp["trace_bias_predicted"] > 0.0)
    # The exact decomposition holds: actual = direct + b_h Cov(w_bar) b_h^T.
    rel = np.abs(cmp["identity_residual"]) / cmp["trace_iterated_actual"]
    assert np.all(rel < 0.02), f"Prop 2.13 identity residual too large: {rel}"


def test_sigma_is_estimated_from_direct_residuals():
    """Sigma_h must reproduce the empirical h-step residual covariance."""
    truth = make_s2_system(order=2)
    w = simulate(truth, 3_000, seed=11)
    op, b_p = M.identify_memory(w, 2, parameterization="s2", ridge_mu=1e-8)
    fam = MH.fit_horizon_family(w, op, b_p, 3, estimator="direct", ridge_mu=1e-8)
    design = MH.build_horizon_design(w, 2, 3)
    n = design["w_bar"].shape[0]
    flat = design["w_bar"].reshape(n, 2 * truth.r)
    for h in range(3):
        resid = design["targets"][h] - flat @ fam.theta[h].T
        emp = (resid.T @ resid) / (n - 1)
        assert np.allclose(fam.sigma[h], emp, atol=1e-8)
    # Variance must grow with the horizon -- an honest multi-day forecast.
    traces = [np.trace(fam.sigma[h]) for h in range(3)]
    assert traces[0] < traces[1] < traces[2]


def test_coverage_comparison_requires_the_iterated_covariance():
    truth = make_s2_system(order=2)
    w = simulate(truth, 500, seed=12)
    op, b_p = M.identify_memory(w, 2, parameterization="s2")
    fam = MH.fit_horizon_family(w, op, b_p, 2, estimator="direct")  # no q passed
    with pytest.raises(ValueError, match="iterated covariance"):
        MH.coverage_comparison(fam, w, op, b_p)


def test_predict_matches_the_stored_operators():
    truth = make_s2_system(order=2)
    rng = np.random.RandomState(13)
    w = simulate(truth, 1_000, seed=13)
    forcing = rng.randn(1_000, 1)
    op, b_p = M.identify_memory(w, 2, forcing, parameterization="s2")
    fam = MH.fit_horizon_family(w, op, b_p, 3, forcing, estimator="direct")
    hist = rng.randn(2, truth.r)
    fut = rng.randn(3, 1)
    out = fam.predict(hist, fut)
    for h in range(3):
        expect = fam.theta[h] @ hist.reshape(-1)
        for i in range(h + 1):
            expect = expect + fam.inputs[h][i] @ fut[i]
        assert np.allclose(out[h], expect, atol=1e-12)
