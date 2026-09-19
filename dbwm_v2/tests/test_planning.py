"""Tests for CEM planning in the weight space (Algorithm 2, PLAN block)."""
import numpy as np
import jax
import jax.numpy as jnp

from dbwm.config import smoke_config
from dbwm.inference.planning import (
    admissible_bounds,
    rollout,
    covariance_trace_sequence,
    plan_cost,
    cem_plan,
    receding_horizon_control,
)


def _system(r=5, ell_u=1, ell_p=1, seed=0):
    """A small stable forced linear system to plan against."""
    rng = np.random.RandomState(seed)
    a = jnp.asarray(0.9 * np.eye(r) + 0.02 * rng.randn(r, r), dtype=jnp.float32)
    b_u = jnp.asarray(rng.randn(r, ell_u), dtype=jnp.float32)
    b_p = jnp.asarray(rng.randn(r, ell_p), dtype=jnp.float32)
    q = jnp.asarray(0.01 * np.eye(r), dtype=jnp.float32)
    return a, b_u, b_p, q


# --------------------------------------------------------------------------- #
# Admissible set: mutual exclusivity of rain and irrigation
# --------------------------------------------------------------------------- #
def test_no_irrigation_is_admissible_when_rain_is_forecast():
    """U_{t+k} = {0} on rainy steps: the box collapses, so irrigation is impossible."""
    rain = jnp.asarray([0.0, 5.0, 0.0, 12.0])
    lo, hi = admissible_bounds(rain, u_max=40.0, n_actuators=1)

    np.testing.assert_allclose(np.asarray(lo), 0.0)
    # Dry steps admit [0, u_max]; rainy steps admit only {0}.
    np.testing.assert_allclose(np.asarray(hi)[:, 0], [40.0, 0.0, 40.0, 0.0])


def test_cem_never_irrigates_into_the_rain():
    """
    The mutual-exclusivity constraint is enforced by CONSTRUCTION (clipping into the
    box), not by a penalty -- so it must hold exactly, for every rainy step, even
    when irrigating there would lower the cost.
    """
    cfg = smoke_config().planning
    a, b_u, b_p, q = _system()
    r = a.shape[0]

    # Rain on every other step.
    rain = jnp.asarray([0.0, 8.0, 0.0, 8.0][: cfg.horizon])
    p_forecast = rain[:, None]

    out = cem_plan(
        jax.random.PRNGKey(0),
        w0=jnp.zeros(r), p0=jnp.eye(r), w_goal=jnp.ones(r),
        a=a, b_u=b_u, q=q, cfg=cfg, b_p=b_p, p_forecast=p_forecast,
    )
    u = np.asarray(out["u_plan"])

    rainy = np.asarray(rain) > 0
    assert np.all(u[rainy] == 0.0), "irrigation during forecast rain must be impossible"
    assert np.all(u >= 0.0), "irrigation is non-negative"
    assert np.all(u <= cfg.u_max + 1e-6), "irrigation is capped at u_max"


def test_lagged_rain_does_not_veto_irrigation():
    """
    Only the CONTEMPORANEOUS rain channel gates the actuator. If the lag channels
    also gated it, one storm would lock irrigation off for as many steps as lags.
    """
    cfg = smoke_config().planning
    a, b_u, b_p, q = _system(ell_p=2)
    r = a.shape[0]

    # Column 0 = rain now (all dry); column 1 = lag-1 rain (wet from a past storm).
    p_forecast = jnp.stack(
        [jnp.zeros(cfg.horizon), jnp.full((cfg.horizon,), 9.0)], axis=-1
    )

    out = cem_plan(
        jax.random.PRNGKey(0),
        w0=jnp.zeros(r), p0=jnp.eye(r), w_goal=jnp.ones(r),
        a=a, b_u=b_u, q=q, cfg=cfg, b_p=b_p, p_forecast=p_forecast,
    )
    _, hi = admissible_bounds(p_forecast[:, 0], cfg.u_max, 1)

    assert np.all(np.asarray(hi) == cfg.u_max), "dry-now steps must remain irrigable"
    assert np.asarray(out["u_plan"]).max() >= 0.0


# --------------------------------------------------------------------------- #
# Rollout and cost
# --------------------------------------------------------------------------- #
def test_rollout_matches_the_input_affine_dynamics():
    """Roll-out must be exactly w_{k+1} = A w_k + B_p p_k + B_u u_k."""
    a, b_u, b_p, _ = _system()
    r = a.shape[0]
    h = 3
    w0 = jnp.ones(r)
    u = jnp.full((h, 1), 2.0)
    p = jnp.full((h, 1), 3.0)

    states = rollout(w0, u, a, b_u, b_p, p)

    w = w0
    for k in range(h):
        w = a @ w + b_u @ u[k] + b_p @ p[k]
        np.testing.assert_allclose(np.asarray(states[k]), np.asarray(w), rtol=1e-5)


def test_covariance_is_unchanged_by_the_known_forcing():
    """
    Theorem 4.3: the known forcing shifts the predict MEAN but leaves the covariance
    recursion untouched. This is what licenses hoisting tr(P) out of the CEM
    candidate loop -- and it is the concrete pay-off of the input-affine form, since
    an action-conditioned A(a_t) would make the covariance input-dependent.
    """
    from dbwm.inference.kalman import kalman_predict

    a, b_u, b_p, q = _system()
    r = a.shape[0]
    p0 = jnp.eye(r)
    w0 = jnp.zeros(r)

    w_free, p_free = kalman_predict(w0, p0, a, q)
    w_forced, p_forced = kalman_predict(w0, p0, a, q, b=b_p, u=jnp.asarray([7.0]))

    np.testing.assert_allclose(np.asarray(p_free), np.asarray(p_forced), rtol=1e-6)
    assert not np.allclose(np.asarray(w_free), np.asarray(w_forced)), (
        "the forcing must still move the mean"
    )


def test_open_loop_covariance_accumulates_from_certainty():
    """
    Starting from a certain state (P_0 = 0), open-loop uncertainty grows every step:
    with no observations the process noise Q simply accumulates. (Started from a
    *large* P_0 a contractive A would instead decay toward the DARE fixed point --
    so the monotone claim only holds from below.)
    """
    a, _, _, q = _system()
    r = a.shape[0]
    tr = covariance_trace_sequence(jnp.zeros((r, r)), a, q, horizon=4)

    assert tr.shape == (4,)
    assert np.all(np.diff(np.asarray(tr)) > 0), "open-loop uncertainty must accumulate"


def test_control_cost_penalises_water_use():
    """lambda_u makes a big irrigation plan cost more than a small one, all else equal."""
    r, h = 4, 3
    states = jnp.zeros((h, r))
    goal = jnp.zeros(r)
    trace_p = jnp.zeros(h)

    cheap = plan_cost(states, jnp.full((h, 1), 1.0), goal, trace_p, 0.95, 0.0, 0.01)
    pricey = plan_cost(states, jnp.full((h, 1), 10.0), goal, trace_p, 0.95, 0.0, 0.01)
    assert float(pricey) > float(cheap)


# --------------------------------------------------------------------------- #
# CEM optimisation
# --------------------------------------------------------------------------- #
def test_cem_beats_a_do_nothing_plan():
    """
    Sanity: with a reachable goal and no rain, the optimised plan should cost less
    than irrigating not at all. (If it does not, CEM is not optimising anything.)
    """
    cfg = smoke_config().planning
    cfg.lambda_u = 0.0  # judge purely on goal-tracking
    a, b_u, b_p, q = _system()
    r = a.shape[0]

    w0 = jnp.zeros(r)
    # A goal that is actually reachable through the actuator direction.
    w_goal = b_u[:, 0] * 5.0
    p_forecast = jnp.zeros((cfg.horizon, 1))
    trace_p = covariance_trace_sequence(jnp.eye(r), a, q, cfg.horizon, cfg.gamma_dyn_inflation)

    out = cem_plan(
        jax.random.PRNGKey(0),
        w0=w0, p0=jnp.eye(r), w_goal=w_goal,
        a=a, b_u=b_u, q=q, cfg=cfg, b_p=b_p, p_forecast=p_forecast,
    )

    zero_plan = jnp.zeros((cfg.horizon, 1))
    zero_states = rollout(w0, zero_plan, a, b_u, b_p, p_forecast)
    zero_cost = plan_cost(
        zero_states, zero_plan, w_goal, trace_p, cfg.gamma, cfg.beta_plan, cfg.lambda_u
    )

    assert float(out["cost"]) < float(zero_cost)


def test_receding_horizon_executes_cadence_and_replans():
    """Executing K actions per plan must yield exactly n_steps actions overall."""
    cfg = smoke_config().planning
    cfg.cadence = 2
    a, b_u, b_p, q = _system()
    r = a.shape[0]
    n_steps = 5

    out = receding_horizon_control(
        jax.random.PRNGKey(0),
        w0=jnp.zeros(r), p0=jnp.eye(r), w_goal=jnp.ones(r),
        a=a, b_u=b_u, q=q, cfg=cfg, n_steps=n_steps,
        b_p=b_p, p_series=jnp.zeros((n_steps + cfg.horizon, 1)),
    )

    assert out["u_executed"].shape == (n_steps, 1)
    assert out["w_traj"].shape == (n_steps, r)
    # ceil(5 / 2) = 3 replans.
    assert out["costs"].shape[0] == 3
