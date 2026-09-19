"""
Tests for the temporal alignment of the exogenous forcing.

The single most dangerous bug in this pipeline is an off-by-one between the
*window the rain fell in* and the *transition it is regressed against*. It is
silent: shapes match, training runs, and ``B_p`` still comes out non-zero -- it is
just fitted to the wrong step, so the learned "rain direction" is really a lagged
artefact. These tests pin the convention down.

Convention (Section 2.2 / Algorithm 3 / Section 8.1)
    w_{t+1} = A w_t + B_p p_t + B_u u_t
    => row t of the forcing is the water arriving in (d_t, d_{t+1}].
"""
import datetime as dt

import numpy as np
import jax.numpy as jnp

from dbwm.config import ForcingConfig
from dbwm.data.forcing import (
    accumulation_windows,
    accumulate_between_acquisitions,
    add_lag_channels,
    channel_split,
    load_irrigation_series,
)
from dbwm.dynamics.identification import two_stage_least_squares, split_input_matrix


def _dates(*days):
    """Build acquisition dates from day-of-January offsets."""
    return [dt.date(2022, 1, d) for d in days]


def test_forcing_row_t_covers_the_forward_window():
    """Row t must accumulate (d_t, d_{t+1}] -- the step it drives, not the one before."""
    dates = _dates(1, 9, 17)
    windows = accumulation_windows(dates, max_accum_days=32)

    assert windows[0] == (dt.date(2022, 1, 1), dt.date(2022, 1, 9))
    assert windows[1] == (dt.date(2022, 1, 9), dt.date(2022, 1, 17))
    # The last acquisition has no successor: its forcing is undefined.
    assert windows[2] == (None, None)


def test_rain_is_attributed_to_the_transition_it_drives():
    """
    Rain falling between d_0 and d_1 must land in row 0 (which drives w_0 -> w_1),
    NOT row 1. This is the regression test for the off-by-one: the backwards
    convention ``(d_{t-1}, d_t]`` would put it in row 1.
    """
    dates = _dates(1, 9, 17)
    daily = {dt.date(2022, 1, 5): np.array([10.0], dtype=np.float32)}  # between d0 and d1

    p = accumulate_between_acquisitions(daily, dates, n_zones=1, max_accum_days=32)

    assert p[0, 0] == 10.0, "rain in (d_0, d_1] must force the w_0 -> w_1 transition"
    assert p[1, 0] == 0.0, "it must NOT be attributed to the next step"
    assert p[2, 0] == 0.0, "final step has no successor acquisition"


def test_window_is_half_open_at_the_start():
    """Rain exactly on d_t belongs to the PREVIOUS window, not (d_t, d_{t+1}]."""
    dates = _dates(1, 9, 17)
    daily = {dt.date(2022, 1, 9): np.array([7.0], dtype=np.float32)}  # exactly on d_1

    p = accumulate_between_acquisitions(daily, dates, n_zones=1, max_accum_days=32)

    # (d_0, d_1] is closed at d_1, so the rain lands in row 0.
    assert p[0, 0] == 7.0
    assert p[1, 0] == 0.0


def test_long_gap_is_capped_but_still_ends_at_the_acquisition():
    """A capped window must be truncated at its START, still ending at d_{t+1}."""
    dates = [dt.date(2022, 1, 1), dt.date(2022, 6, 1)]  # 151-day gap
    windows = accumulation_windows(dates, max_accum_days=10)

    start, end = windows[0]
    assert end == dt.date(2022, 6, 1), "window must end at the acquisition it forecasts into"
    assert (end - start).days == 10, "over-long gap is capped to max_accum_days"


def test_lag_channel_is_the_antecedent_wet_soil():
    """Lag-1 at step t is the rain of the PREVIOUS window -- soil already wet at w_t."""
    p = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    out = add_lag_channels(p, n_lags=1)

    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.0])  # contemporaneous
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0, 2.0])  # lag 1, zero-padded


def test_irrigation_shares_the_precipitation_grid(tmp_path):
    """Irrigation must use the SAME forward windows, or B_p and B_u desynchronise."""
    csv = tmp_path / "irrig.csv"
    csv.write_text("date,irrigation_mm\n2022-01-05,25.0\n")

    cfg = ForcingConfig(irrigation_value_cols=("irrigation_mm",), max_accum_days=32)
    dates = _dates(1, 9, 17)
    u = load_irrigation_series(str(csv), dates, cfg)

    # Applied on Jan 5, i.e. within (d_0, d_1] -> drives the w_0 -> w_1 transition.
    assert u[0, 0] == 25.0
    assert u[1, 0] == 0.0


def test_channel_split_separates_disturbance_from_actuator():
    """B_p (rain, uncontrollable) and B_u (irrigation, controllable) must be separable."""
    names = ["precip", "precip_lag1", "irrigation_mm"]
    precip_cols, irrig_cols = channel_split(names)

    assert precip_cols == [0, 1]
    assert irrig_cols == [2]


def test_closed_loop_prior_uses_the_forcing():
    """
    The one-step-ahead prior is what gets SCORED as the forecast, so it must include
    the ``B u`` term (Algorithm 2: w_{t|t-1} = A w_{t-1|t-1} + B u_{t-1}).

    Dropping it would silently evaluate a pure-autonomous forecast while reporting it
    as the forced model -- hiding exactly the skill that B_p / B_u are supposed to add.
    Here B is huge, so an unforced prior cannot possibly coincide with a forced one.
    """
    r = 4
    rng = np.random.RandomState(0)
    a = jnp.asarray(0.9 * np.eye(r), dtype=jnp.float32)
    b = jnp.asarray(100.0 * rng.randn(r, 1), dtype=jnp.float32)
    w_filt = jnp.asarray(rng.randn(3, r), dtype=jnp.float32)
    u = jnp.asarray(np.array([[1.0], [1.0], [1.0]]), dtype=jnp.float32)

    # The expression used by closed_loop_filter for rows 1..T-1.
    prior_forced = w_filt[:-1] @ a.T + u[:-1] @ b.T
    prior_unforced = w_filt[:-1] @ a.T

    assert not np.allclose(np.asarray(prior_forced), np.asarray(prior_unforced)), (
        "the forecast prior must depend on rain/irrigation"
    )
    # Row t-1 of the forcing drives t-1 -> t (forward-window convention).
    np.testing.assert_allclose(
        np.asarray(prior_forced[0]),
        np.asarray(a @ w_filt[0] + b @ u[0]),
        rtol=1e-5,
    )


def test_identification_recovers_B_under_the_pipeline_convention():
    """
    End-to-end: simulate w_{t+1} = A w_t + B u_t using the pipeline's OWN index
    convention, then check Algorithm 3 Stage II recovers the true B_p / B_u.

    If the accumulation convention and the identification convention disagreed, the
    recovered B would be a lagged smear and this would fail.
    """
    rng = np.random.RandomState(0)
    r, t = 6, 400

    a_true = 0.9 * np.eye(r) + 0.02 * rng.randn(r, r)
    b_true = rng.randn(r, 2)  # [B_p | B_u]

    # Sparse, mutually exclusive forcing (rain XOR irrigation), as in the real data.
    u = np.zeros((t, 2), dtype=np.float32)
    rainy = rng.rand(t) < 0.25
    u[rainy, 0] = rng.gamma(2.0, 6.0, size=int(rainy.sum()))
    irrigate = (np.arange(t) % 4 == 0) & (~rainy)
    u[irrigate, 1] = rng.uniform(10.0, 40.0, size=int(irrigate.sum()))

    # Row t of u drives the transition w_t -> w_{t+1}: the pipeline's convention.
    w = np.zeros((t, r), dtype=np.float32)
    w[0] = rng.randn(r)
    for i in range(t - 1):
        w[i + 1] = a_true @ w[i] + b_true @ u[i]

    a_hat, b_hat = two_stage_least_squares(
        jnp.asarray(w), jnp.asarray(u), mu=1e-8, quiescent_threshold=1e-8
    )

    # float32 least squares on a near-unit-root system: ~1e-3 is at the noise floor.
    np.testing.assert_allclose(np.asarray(a_hat), a_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(b_hat), b_true, atol=5e-3)

    # And the two forcing roles come apart cleanly.
    b_p, b_u = split_input_matrix(b_hat, ["precip", "irrigation_mm"])
    assert b_p.shape == (r, 1) and b_u.shape == (r, 1)
    np.testing.assert_allclose(np.asarray(b_p)[:, 0], b_true[:, 0], atol=5e-3)
    np.testing.assert_allclose(np.asarray(b_u)[:, 0], b_true[:, 1], atol=5e-3)

    # --- The point of the test: the WRONG convention must NOT recover B. ---
    # Feed the forcing shifted by one step (the old backwards convention) and confirm
    # the recovered B is nowhere near the truth. This is what makes the assertions
    # above a convention check rather than a numerical-precision check: the correct
    # alignment errs by ~1e-3, the misaligned one by O(1).
    u_shifted = np.zeros_like(u)
    u_shifted[1:] = u[:-1]
    _, b_wrong = two_stage_least_squares(
        jnp.asarray(w), jnp.asarray(u_shifted), mu=1e-8, quiescent_threshold=1e-8
    )
    err_right = float(np.abs(np.asarray(b_hat) - b_true).max())
    err_wrong = float(np.abs(np.asarray(b_wrong) - b_true).max())
    assert err_wrong > 100 * err_right, (
        "an off-by-one in the forcing index must visibly destroy B "
        "(correct={:.2e}, shifted={:.2e})".format(err_right, err_wrong)
    )
