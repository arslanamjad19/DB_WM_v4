"""
Tests for the weather emission ``C`` and the v3 step-0 whiteness pretest.

The emission is what makes a Kalman *update* possible at forecast steps t+1..t+6,
where no NDVI frame exists by construction. Its value therefore has to be measured,
not assumed -- hence the R^2 and information-gain tests.
"""
import numpy as np
import pytest

from dbwm.dynamics import diagnostics as D
from dbwm.dynamics import emission as E
from dbwm.dynamics import memory as M
from tests.test_memory import make_s2_system, simulate


# --------------------------------------------------------------------------- #
# Emission fit
# --------------------------------------------------------------------------- #
def test_emission_recovers_a_known_c():
    rng = np.random.RandomState(0)
    r, n, m = 12, 3000, 3
    w = rng.randn(n, r)
    c_true = rng.randn(m, r)
    y = w @ c_true.T + 0.1 * rng.randn(n, m)
    model = E.fit_emission(w, y, ["rs", "ta", "vpd"], ridge=1e-8, holdout_fraction=0.0)
    assert np.allclose(model.c, c_true, atol=0.02)
    assert np.all(model.r2 > 0.99)


def test_noise_covariance_is_full_and_captures_cross_correlation():
    """Rs/Ta/VPD residuals co-vary, and a diagonal R throws that structure away.

    Note the direction: under strong positive correlation the correct full-R filter
    extracts *more* information than the diagonal approximation, because
    differencing two highly correlated channels cancels the shared disturbance and
    exposes the state. The reason to fit the full matrix is that it is the true
    one, not that it errs conveniently.
    """
    rng = np.random.RandomState(1)
    r, n = 8, 2000
    w = rng.randn(n, r)
    c_true = rng.randn(3, r)
    shared = rng.randn(n, 1)  # one common disturbance across all three channels
    y = w @ c_true.T + shared * np.array([[1.0, 0.9, 0.8]]) + 0.1 * rng.randn(n, 3)

    full = E.fit_emission(w, y, ["rs", "ta", "vpd"], holdout_fraction=0.0)
    diag = E.fit_emission(
        w, y, ["rs", "ta", "vpd"], full_covariance=False, holdout_fraction=0.0
    )
    off = full.r_cov[~np.eye(3, dtype=bool)]
    assert np.max(np.abs(off)) > 0.3
    assert np.allclose(diag.r_cov[~np.eye(3, dtype=bool)], 0.0)
    # The two give materially different gains, so the choice is not cosmetic.
    p = np.eye(r)
    g_full = E.information_gain(full, p)["trace_reduction"]
    g_diag = E.information_gain(diag, p)["trace_reduction"]
    assert not np.isclose(g_full, g_diag, rtol=0.01)
    # Here the correlation is strongly positive, so the correct full-R filter is
    # the one that extracts more.
    assert g_full > g_diag


def test_information_gain_matches_the_analytic_rank_m_bound():
    """A near-noiseless rank-m sensor removes ~m units of trace from P = I."""
    rng = np.random.RandomState(2)
    r, n, m = 20, 4000, 3
    w = rng.randn(n, r)
    c_true = np.zeros((m, r))
    c_true[:, :5] = rng.randn(m, 5)
    y = w @ c_true.T + 0.05 * rng.randn(n, m)
    model = E.fit_emission(w, y, list("abc"), ridge=1e-8, holdout_fraction=0.0)
    gain = E.information_gain(model, np.eye(r))
    assert gain["trace_reduction"] == pytest.approx(m, abs=0.1)
    assert 0.0 < gain["fraction_removed"] < 1.0


def test_uninformative_sensor_is_disabled_not_just_flagged(caplog):
    """
    A weather channel uncorrelated with the state must be **dropped**, not warned about.

    Flagging alone was the old behaviour and it was not enough: the observer went
    on to assimilate the sensor anyway, applying a Kalman update built from a
    meaningless innovation at every one of ~1581 filter steps. That does not
    merely fail to help -- it drives the state the forecast starts from, which
    surfaces as a whole-field bias in the output maps rather than as extra noise.
    """
    rng = np.random.RandomState(3)
    w = rng.randn(1500, 10)
    y = rng.randn(1500, 3)  # pure noise: no relationship to the state at all
    with caplog.at_level("WARNING"):
        model = E.fit_emission(w, y, ["rs", "ta", "vpd"], holdout_fraction=0.2)
    assert model.usable is False
    assert any("DISABLED" in rec.message for rec in caplog.records)
    assert np.all(model.r2_holdout < 0.1)


def test_informative_sensor_stays_usable():
    """
    The gate must not fire on a sensor that genuinely tracks the state.

    Without this the fix would be indistinguishable from deleting the weather
    update outright, and the framework's two-role weather design (precipitation
    into ``B_p``, Rs/Ta/VPD into ``C``) would be silently half-removed.
    """
    rng = np.random.RandomState(11)
    w = rng.randn(800, 6)
    c = rng.randn(3, 6)
    y = w @ c.T + 0.05 * rng.randn(800, 3)
    model = E.fit_emission(w, y, ["rs", "ta", "vpd"], holdout_fraction=0.2)
    assert model.usable is True
    assert np.all(model.r2_holdout > 0.5)


def test_gate_threshold_is_respected():
    """``min_holdout_r2`` raises the bar a channel has to clear."""
    rng = np.random.RandomState(12)
    w = rng.randn(600, 5)
    c = rng.randn(3, 5)
    y = w @ c.T + 3.0 * rng.randn(600, 3)  # weak but real signal
    lax = E.fit_emission(w, y, list("abc"), holdout_fraction=0.2, min_holdout_r2=0.0)
    strict = E.fit_emission(w, y, list("abc"), holdout_fraction=0.2, min_holdout_r2=0.99)
    assert lax.usable is True
    assert strict.usable is False


def test_holdout_r2_is_chronological_and_honest():
    """In-sample R^2 can be inflated by overfitting; the holdout must expose it."""
    rng = np.random.RandomState(4)
    n, r = 200, 150  # r comparable to n -> in-sample fit is nearly perfect
    w = rng.randn(n, r)
    y = rng.randn(n, 3)
    model = E.fit_emission(w, y, list("abc"), ridge=1e-6, holdout_fraction=0.25)
    assert np.mean(model.r2) > np.mean(model.r2_holdout)


def test_innovation_uses_anomalies_without_a_seasonal_term():
    """The climatology is already removed upstream, so no offset appears here."""
    rng = np.random.RandomState(5)
    w = rng.randn(500, 6)
    c = rng.randn(3, 6)
    y = w @ c.T
    model = E.fit_emission(w, y, list("abc"), ridge=1e-10, holdout_fraction=0.0)
    assert np.allclose(model.innovation(w[0], y[0]), 0.0, atol=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        E.fit_emission(np.zeros((10, 4)), np.zeros((9, 3)), list("abc"))


# --------------------------------------------------------------------------- #
# v3 step 0: the whiteness pretest
# --------------------------------------------------------------------------- #
def test_pretest_does_not_reject_a_genuinely_markov_process():
    """L = 1 truly suffices -> the lift would be pure variance inflation."""
    rng = np.random.RandomState(6)
    r, n = 24, 1200
    a = np.eye(r) * 0.7 + 0.05 * rng.randn(r, r)
    w = np.zeros((n, r))
    for t in range(n - 1):
        w[t + 1] = a @ w[t] + rng.randn(r)
    op, _ = M.identify_memory(w, 1, parameterization="unstructured", ridge_mu=1e-6)
    out = D.whiteness_pretest(D.one_step_residuals(w, op.blocks[0]), lags=12)
    assert not out["reject"]
    assert out["fraction_significant"] < 0.15
    assert "FAIL TO REJECT" in out["verdict"]


def test_pretest_rejects_when_memory_is_present():
    """Order-3 dynamics fitted at L = 1 leave coloured residuals -> Prop. 2.5 bites."""
    truth = make_s2_system(order=3)
    w = simulate(truth, 1500, seed=7)
    op, _ = M.identify_memory(w, 1, parameterization="unstructured", ridge_mu=1e-6)
    out = D.whiteness_pretest(D.one_step_residuals(w, op.blocks[0]), lags=12)
    assert out["reject"]
    assert out["fraction_significant"] > 0.5
    assert "REJECT whiteness" in out["verdict"]


def test_projection_dim_is_reduced_to_keep_chi2_valid():
    """d^2 K must stay well below T or the chi-square approximation is meaningless.

    At r = 256 and K = 12 a full multivariate portmanteau would have 786,432
    degrees of freedom against ~1200 samples.
    """
    rng = np.random.RandomState(8)
    resid = rng.randn(400, 64)
    out = D.whiteness_pretest(resid, lags=12, projection_dim=16)
    assert out["projection_dim"] < 16
    assert out["projection_dim"] ** 2 * 12 <= 400 // 4


def test_benjamini_hochberg_controls_fdr_under_the_null():
    """Uniform p-values should yield roughly alpha-level rejections, not more."""
    rng = np.random.RandomState(9)
    p = rng.random_sample(2000)
    rejected = D.benjamini_hochberg(p, alpha=0.05)
    assert rejected.mean() < 0.05


def test_benjamini_hochberg_is_a_step_up_procedure():
    """Everything below the largest passing rank is rejected, not just the passing ones."""
    p = np.array([0.001, 0.02, 0.9, 0.95])
    out = D.benjamini_hochberg(p, alpha=0.05)
    assert out.tolist() == [True, True, False, False]


def test_one_step_residuals_subtract_the_forcing():
    """Leaving B_p p_t in the residual would fake autocorrelation from the rain."""
    rng = np.random.RandomState(10)
    n, r = 300, 5
    a = np.eye(r) * 0.5
    b = rng.randn(r, 1)
    forcing = rng.randn(n, 1)
    w = np.zeros((n, r))
    for t in range(n - 1):
        w[t + 1] = a @ w[t] + b @ forcing[t]
    resid = D.one_step_residuals(w, a, b, forcing)
    assert np.allclose(resid, 0.0, atol=1e-10)


def test_residual_autocorrelation_band_and_shape():
    rng = np.random.RandomState(11)
    out = D.residual_autocorrelation(rng.randn(1000, 8), lags=10)
    assert out["mean_abs_acf"].shape == (10,)
    assert out["band"] == pytest.approx(1.96 / np.sqrt(1000))
    # White noise sits at or below the band on average.
    assert np.median(out["mean_abs_acf"]) < 2 * out["band"]


def test_hosking_requires_enough_samples():
    with pytest.raises(ValueError, match="Need T >"):
        D.ljung_box_hosking(np.zeros((10, 4)), lags=12)
