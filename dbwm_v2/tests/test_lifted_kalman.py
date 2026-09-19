"""
Tests for the memory-lifted, two-sensor Kalman observer (v3 Sec. 2.2.6).

Covers the mechanics (predict/update, missing frames, fixed-lag smoothing) and the
claims that matter for the thesis: that the daily rank-``m`` weather sensor really
does correct the t+1..t+6 forecast, and that the two forecast modes trade off the
way the theory says they should.
"""
import numpy as np
import pytest

from dbwm.dynamics.emission import fit_emission
from dbwm.dynamics.memory import identify_memory
from dbwm.dynamics.multihorizon import fit_horizon_family
from dbwm.inference import lifted_kalman as K
from tests.test_memory import make_s2_system, simulate


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def build_system(n=3000, order=3, seed=1, noise=0.5, weather_strength=0.8):
    """Identify a full system from a synthetic record: memory, B_p, Q, C, R."""
    truth = make_s2_system(order=order)
    r = truth.r
    w = simulate(truth, n, seed=seed, noise=noise)
    rng = np.random.RandomState(seed)
    forcing = np.where(
        rng.random_sample((n, 2)) < 0.2, rng.gamma(2.0, 1.0, (n, 2)), 0.0
    )
    c_true = rng.randn(3, r) * weather_strength
    weather = w @ c_true.T + 0.3 * rng.randn(n, 3)

    op, b_p = identify_memory(
        w, order, forcing, parameterization="s2", ridge_mu=1e-6
    )
    em = fit_emission(w, weather, ["rs", "ta", "vpd"], holdout_fraction=0.0)
    hist = np.stack([w[order - 1 - j : n - 1 - j] for j in range(order)], axis=1)
    resid = w[order:] - np.einsum("jrs,njs->nr", op.blocks, hist)
    if b_p.shape[1]:
        resid = resid - forcing[order - 1 : n - 1] @ b_p.T
    q = np.cov(resid.T)
    sysm = K.LiftedSystem(op=op, b_p=b_p, q=q, c_w=em.c, r_w=em.r_cov)
    return sysm, w, forcing, weather


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_predict_matches_the_dense_companion():
    """The companion-sparsity shortcut must equal the dense computation exactly."""
    sysm, _, _, _ = build_system(n=600, order=3)
    rng = np.random.RandomState(0)
    n = sysm.lifted_dim
    state = rng.randn(n)
    a = rng.randn(n, n)
    cov = a @ a.T
    u = rng.randn(sysm.b_p.shape[1])

    s_fast, p_fast = K.predict(sysm, state, cov, u)
    a_cal = sysm.op.companion()
    b_cal = np.zeros((n, sysm.b_p.shape[1]))
    b_cal[: sysm.r] = sysm.b_p
    s_dense = a_cal @ state + b_cal @ u
    p_dense = a_cal @ cov @ a_cal.T + sysm.process_noise()
    assert np.allclose(s_fast, s_dense, atol=1e-9)
    assert np.allclose(p_fast, 0.5 * (p_dense + p_dense.T), atol=1e-9)


def test_update_matches_the_textbook_form():
    """P - K S K^T equals P - K C P, but stays symmetric by construction."""
    rng = np.random.RandomState(1)
    lr, r, m = 12, 4, 3
    a = rng.randn(lr, lr)
    cov = a @ a.T
    state = rng.randn(lr)
    c = rng.randn(m, r)
    noise = np.eye(m) * 0.3
    y = rng.randn(m)

    s_post, p_post, nu = K.update(state, cov, c, y, noise)
    c_full = np.zeros((m, lr))
    c_full[:, :r] = c
    s = c_full @ cov @ c_full.T + noise
    gain = cov @ c_full.T @ np.linalg.inv(s)
    assert np.allclose(s_post, state + gain @ (y - c_full @ state), atol=1e-9)
    assert np.allclose(p_post, cov - gain @ c_full @ cov, atol=1e-8)
    assert np.allclose(p_post, p_post.T, atol=1e-12)
    assert np.allclose(nu, y - c @ state[:r], atol=1e-12)


def test_update_reduces_uncertainty():
    rng = np.random.RandomState(2)
    lr, r, m = 9, 3, 2
    a = rng.randn(lr, lr)
    cov = a @ a.T + np.eye(lr)
    _, p_post, _ = K.update(rng.randn(lr), cov, rng.randn(m, r), rng.randn(m), np.eye(m))
    assert np.trace(p_post) < np.trace(cov)
    assert np.all(np.linalg.eigvalsh(p_post) > -1e-9)


def test_lemma_2_7_lifted_process_noise_is_singular_but_reachable():
    """E Q E^T has rank r of Lr -- singular, yet the pair stays stabilizable."""
    sysm, _, _, _ = build_system(n=400, order=3)
    qn = sysm.process_noise()
    assert qn.shape == (sysm.lifted_dim, sysm.lifted_dim)
    assert np.linalg.matrix_rank(qn, tol=1e-8) == sysm.r
    assert np.allclose(qn[sysm.r :, :], 0.0)


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_filter_recovers_the_state_and_beats_the_prior():
    sysm, w, forcing, weather = build_system(n=2000)
    observed = np.ones(len(w), dtype=bool)
    res = K.filter_sequence(sysm, w, observed, forcing, weather, sigma_eps2=1e-2)
    prior_rmse = np.sqrt(np.mean((res.priors[10:] - w[10:]) ** 2))
    post_rmse = np.sqrt(np.mean((res.current()[10:] - w[10:]) ** 2))
    assert post_rmse < 0.25 * prior_rmse
    assert res.n_ndvi_updates == len(w)
    assert res.n_weather_updates == len(w)


def test_missing_frames_are_skipped_not_faked():
    """A gap must reduce the NDVI update count and degrade the state honestly."""
    sysm, w, forcing, weather = build_system(n=2000)
    observed = np.ones(len(w), dtype=bool)
    observed[1000:1010] = False
    res = K.filter_sequence(sysm, w, observed, forcing, weather, sigma_eps2=1e-2)
    assert res.n_ndvi_updates == len(w) - 10
    gap = np.sqrt(np.mean((res.current()[1000:1010] - w[1000:1010]) ** 2))
    clean = np.sqrt(np.mean((res.current()[1200:1210] - w[1200:1210]) ** 2))
    assert gap > clean  # no pretending the gap was observed


def test_weather_only_filtering_still_constrains_the_state():
    """During an NDVI blackout the rank-m sensor must do measurable work."""
    sysm, w, forcing, weather = build_system(n=2000)
    observed = np.ones(len(w), dtype=bool)
    observed[800:900] = False
    with_w = K.filter_sequence(sysm, w, observed, forcing, weather, sigma_eps2=1e-2)
    without_w = K.filter_sequence(sysm, w, observed, forcing, None, sigma_eps2=1e-2)
    e_with = np.sqrt(np.mean((with_w.current()[820:900] - w[820:900]) ** 2))
    e_without = np.sqrt(np.mean((without_w.current()[820:900] - w[820:900]) ** 2))
    assert e_with < e_without


def test_prop_2_9_lower_blocks_are_fixed_lag_smoothed_estimates():
    """The lifted filter retro-corrects past weights -- not mere bookkeeping.

    The lag-j block at time t is w_{t-j|t}, which conditions on j MORE observations
    than the filtered estimate w_{t-j|t-j} did. It must therefore be at least as
    accurate. This is the mechanism that fills revisit and cloud gaps.
    """
    sysm, w, forcing, weather = build_system(n=2000, order=3, noise=0.8)
    observed = np.ones(len(w), dtype=bool)
    res = K.filter_sequence(sysm, w, observed, forcing, weather, sigma_eps2=0.5)
    filt = res.current()
    lag2 = res.smoothed_lag(2)
    # Compare w_{t-2|t} against w_{t-2|t-2} on the same targets.
    err_smoothed = np.sqrt(np.mean((lag2[10:] - w[8:-2]) ** 2))
    err_filtered = np.sqrt(np.mean((filt[8:-2] - w[8:-2]) ** 2))
    assert err_smoothed <= err_filtered * 1.001


def test_smoothed_lag_bounds_checked():
    sysm, w, _, _ = build_system(n=300, order=3)
    res = K.filter_sequence(sysm, w, np.ones(len(w), dtype=bool))
    with pytest.raises(ValueError, match="lag must be in"):
        res.smoothed_lag(3)


def test_per_date_observation_covariance_is_respected():
    """A date with few valid pixels must be trusted less than a well-observed one.

    Using a scalar sigma_eps^2 I would discard exactly this information, and the
    valid-pixel count genuinely varies across the NDVI archive.
    """
    sysm, w, forcing, weather = build_system(n=800)
    observed = np.ones(len(w), dtype=bool)
    r = sysm.r
    tight = np.tile(1e-4 * np.eye(r), (len(w), 1, 1))
    loose = np.tile(1e2 * np.eye(r), (len(w), 1, 1))
    res_tight = K.filter_sequence(sysm, w, observed, forcing, None, tight)
    res_loose = K.filter_sequence(sysm, w, observed, forcing, None, loose)
    e_tight = np.sqrt(np.mean((res_tight.current()[10:] - w[10:]) ** 2))
    e_loose = np.sqrt(np.mean((res_loose.current()[10:] - w[10:]) ** 2))
    assert e_tight < e_loose


# --------------------------------------------------------------------------- #
# Forecasting: the t+1..t+6 protocol
# --------------------------------------------------------------------------- #
def test_forecast_error_grows_then_saturates_without_corrections():
    """Uncorrected multi-step error accumulates -- v2 Thm 4.2 / v3 Thm 2.14.

    Growth is unbounded only in the rho_max = 1 regime that v2 adopts for a
    near-conservative field. This synthetic system is dissipative (rho < 1), so the
    error rises and then SATURATES at the stationary level rather than growing
    monotonically; asserting strict monotonicity would be asserting the wrong
    theorem. What must hold is that h = 1 is the best step and that the error rises
    materially before it plateaus.
    """
    sysm, w, forcing, weather = build_system(n=3000)
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(2000, 2900, 10)
    out = K.rolling_forecast(
        sysm, w, observed, origins, 6, forcing, None, mode="recursive"
    )
    err = np.array(
        [
            np.sqrt(
                np.mean(
                    (out["mean"][:, h] - np.stack([w[o + h + 1] for o in origins])) ** 2
                )
            )
            for h in range(6)
        ]
    )
    assert err[0] == err.min()          # one step ahead is always the easiest
    assert err[2] > 1.15 * err[0]       # error genuinely accumulates
    # Saturation, not divergence: later steps stay near the plateau.
    assert np.all(err[2:] > 0.9 * err[2])
    assert err.max() < 2.0 * err[0]


def test_weather_update_corrects_every_horizon_step():
    """The rank-m sensor must reduce error at EVERY step from t+1 to t+6.

    This is the core of the requested design: during a forecast no NDVI frame
    exists, so the weather sensor is the only correction available.
    """
    sysm, w, forcing, weather = build_system(n=3000)
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(2000, 2900, 5)
    scores = {}
    for use_weather in (False, True):
        out = K.rolling_forecast(
            sysm, w, observed, origins, 6, forcing,
            weather if use_weather else None, mode="recursive",
        )
        scores[use_weather] = np.array(
            [
                np.sqrt(
                    np.mean(
                        (out["mean"][:, h] - np.stack([w[o + h + 1] for o in origins])) ** 2
                    )
                )
                for h in range(6)
            ]
        )
    assert np.all(scores[True] < scores[False])
    # And the correction must be substantial, not cosmetic.
    assert np.mean(scores[True] / scores[False]) < 0.8


def test_trace_reduction_is_reported_and_positive():
    """The uncertainty actually removed by each weather update is measured."""
    sysm, w, forcing, weather = build_system(n=1500)
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(1000, 1400, 20)
    out = K.rolling_forecast(
        sysm, w, observed, origins, 6, forcing, weather, mode="recursive"
    )
    assert np.all(out["trace_reduction"] >= 0.0)
    assert np.mean(out["trace_reduction"]) > 0.0


def test_recursive_accumulates_corrections_better_than_direct():
    """Six compounding rank-m updates beat one, when the sensor is informative.

    Neither mode dominates a priori: recursive accumulates corrections but carries
    the iterated bias b_h, while direct is unbiased and calibrated but corrects
    once. With a strong sensor the accumulation wins at long horizons -- which is
    a measured outcome, not an assumption.
    """
    sysm, w, forcing, weather = build_system(n=3000)
    op, b_p = sysm.op, sysm.b_p
    family = fit_horizon_family(
        w, op, b_p, 6, forcing, ridge_mu=1e-6, nu_selection="holdout"
    )
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(2000, 2900, 5)
    scores = {}
    for mode in ("recursive", "direct"):
        out = K.rolling_forecast(
            sysm, w, observed, origins, 6, forcing, weather,
            family=family, mode=mode,
        )
        scores[mode] = np.array(
            [
                np.sqrt(
                    np.mean(
                        (out["mean"][:, h] - np.stack([w[o + h + 1] for o in origins])) ** 2
                    )
                )
                for h in range(6)
            ]
        )
    assert scores["recursive"][-1] < scores["direct"][-1]
    # They must agree at h = 1, where both are one step from the same origin.
    assert scores["recursive"][0] == pytest.approx(scores["direct"][0], rel=0.05)


def test_direct_mode_requires_a_horizon_family():
    sysm, w, forcing, weather = build_system(n=500)
    with pytest.raises(ValueError, match="requires a HorizonFamily"):
        K.forecast_from_origin(
            sysm, np.zeros(sysm.lifted_dim), np.eye(sysm.lifted_dim), 3, mode="direct"
        )


def test_unknown_forecast_mode_raises():
    sysm, w, _, _ = build_system(n=500)
    with pytest.raises(ValueError, match="mode must be"):
        K.forecast_from_origin(
            sysm, np.zeros(sysm.lifted_dim), np.eye(sysm.lifted_dim), 3, mode="sideways"
        )


def test_forecast_never_uses_the_held_out_ndvi_frame():
    """A forecast branch must be identical whether or not future frames exist.

    If future NDVI leaked into the branch, corrupting it would change the result.
    """
    sysm, w, forcing, weather = build_system(n=1500)
    observed = np.ones(len(w), dtype=bool)
    origins = np.array([1000, 1100])
    base = K.rolling_forecast(
        sysm, w, observed, origins, 6, forcing, weather, mode="recursive"
    )
    corrupted = w.copy()
    corrupted[1001:1007] += 1e3  # destroy the frames a leak would read
    corrupted[1101:1107] += 1e3
    leaked = K.rolling_forecast(
        sysm, corrupted, observed, origins, 6, forcing, weather, mode="recursive"
    )
    assert np.allclose(base["mean"], leaked["mean"], atol=1e-8)


def test_rolling_forecast_marks_out_of_record_horizons_invalid():
    sysm, w, forcing, weather = build_system(n=600)
    observed = np.ones(len(w), dtype=bool)
    origins = np.array([len(w) - 4])
    out = K.rolling_forecast(
        sysm, w, observed, origins, 6, forcing, weather, mode="recursive"
    )
    assert out["valid"][0].sum() == 3
    assert not out["valid"][0, 3:].any()


# --------------------------------------------------------------------------- #
# Single-pass multi-variant forecasting
# --------------------------------------------------------------------------- #
def test_multi_variant_matches_separate_calls():
    """One filter pass must give bit-identical branches to N separate passes.

    Calling rolling_forecast once per variant repeats the identical 1581-step
    filter -- ~9 minutes each at r=256 -- for no information gain. This pins that
    the cheap path is not an approximation.
    """
    from dbwm.inference.lifted_kalman import rolling_forecast_multi

    sysm, w, forcing, weather = build_system(n=1500)
    family = fit_horizon_family(
        w, sysm.op, sysm.b_p, 4, forcing, ridge_mu=1e-6, nu_selection="holdout"
    )
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(1000, 1200, 10)
    variants = [
        {"name": "recursive", "mode": "recursive", "use_weather": True},
        {"name": "direct", "mode": "direct", "use_weather": True},
    ]
    multi = rolling_forecast_multi(
        sysm, w, observed, origins, 4, variants, forcing, weather, family=family
    )
    for spec in variants:
        single = K.rolling_forecast(
            sysm, w, observed, origins, 4, forcing, weather,
            family=family, mode=spec["mode"],
        )
        assert np.allclose(multi[spec["name"]]["mean"], single["mean"], atol=1e-10)
        assert np.allclose(
            multi[spec["name"]]["cov_trace"], single["cov_trace"], atol=1e-10
        )


def test_no_weather_variant_toggles_only_the_branch():
    """The controlled baseline must keep the origin state, changing one thing.

    `rolling_forecast(weather=None)` also strips weather from the FILTER, so its
    baseline starts from a degraded origin state -- that confounds "what does the
    t+1..t+6 correction buy?" with "what is the sensor worth end to end?".
    """
    from dbwm.inference.lifted_kalman import rolling_forecast_multi

    sysm, w, forcing, weather = build_system(n=1500)
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(1000, 1200, 10)
    variants = [
        {"name": "on", "mode": "recursive", "use_weather": True},
        {"name": "off", "mode": "recursive", "use_weather": False},
    ]
    multi = rolling_forecast_multi(
        sysm, w, observed, origins, 4, variants, forcing, weather
    )
    stripped = K.rolling_forecast(
        sysm, w, observed, origins, 4, forcing, None, mode="recursive"
    )
    # Branch-only toggle differs from the fully-stripped run: the origin state is
    # better in the controlled version because the filter still saw the weather.
    assert not np.allclose(multi["off"]["mean"], stripped["mean"])

    def rmse(mean):
        return np.array([
            np.sqrt(np.mean((mean[:, h] - np.stack([w[o + h + 1] for o in origins])) ** 2))
            for h in range(4)
        ])

    # With the origin held fixed, the correction must still help at every horizon.
    assert np.all(rmse(multi["on"]["mean"]) < rmse(multi["off"]["mean"]))


def test_keep_cov_off_by_default_but_trace_still_reported():
    """4.4 GB of covariances at r=256 that nothing reads is not worth allocating."""
    sysm, w, forcing, weather = build_system(n=900)
    observed = np.ones(len(w), dtype=bool)
    origins = np.arange(700, 800, 10)
    off = K.rolling_forecast(sysm, w, observed, origins, 4, forcing, weather)
    on = K.rolling_forecast(
        sysm, w, observed, origins, 4, forcing, weather, keep_cov=True
    )
    assert "cov" not in off
    assert "cov" in on
    assert off["cov_trace"].shape == (origins.size, 4)
    assert np.allclose(
        off["cov_trace"], np.trace(on["cov"], axis1=2, axis2=3), atol=1e-12
    )
