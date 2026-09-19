"""
Cross-Entropy Method (CEM) planning in the deep-basis weight space -- the PLAN
block of Algorithm 2, and item 4 of the World-Model isomorphism (Section 7.1).

This is the *control* half of the framework, and it exists only because the
dynamics are **input-affine** (Remark 2.1). The planner optimises the irrigation
sequence ``u_{t:t+H}`` against the rollout

    w_{t+k+1} = A w_{t+k} + B_p p_{t+k} + B_u u_{t+k},

in which ``p`` is a *known forecast* (an uncontrollable disturbance we must live
with) and ``u`` is the *decision*. Had we adopted the rejected action-conditioned
form ``A(a_t) w_t``, there would be no fixed ``B_u`` to plan against and no single
spectrum to reason about.

Mutual exclusivity as a box constraint
--------------------------------------
A farmer does not irrigate into the rain. Algorithm 2 encodes this not as a
discrete switch but as a *continuous box*:

    U_{t+k} = {0}          if p_{t+k} > 0    (rain forecast -> no irrigation)
            = [0, u_max]   if p_{t+k} = 0

which is just ``0 <= u <= u_max * 1[no rain]``. Clipping CEM samples into this box
keeps the search space convex and differentiable-friendly, and makes the
constraint impossible to violate by construction rather than by penalty.

Only the **contemporaneous** rain channel gates irrigation. The lagged precipitation
channels describe soil that is *already* wet; they belong in the rollout (they force
the state) but they must not veto irrigation, or a single storm would lock the
actuator off for as many steps as there are lags.

Cost and the covariance term
----------------------------
    C = sum_k gamma^k [ ||w_{t+k+1} - w_g||^2 + beta_plan * tr(P_{t+k+1})
                        + lambda_u * ||u_{t+k}||^2 ]

Worth being explicit about ``tr(P)``: by Theorem 4.3 the known forcing shifts the
predict *mean* but leaves the covariance recursion untouched, so ``P_{t+k}`` does
**not** depend on ``u`` -- the uncertainty term is a control-independent offset and
cannot change the argmin. It is retained because it makes the reported cost the
true expected cost (and it would start to bite the moment one adds control-dependent
process noise). We compute it once, outside the candidate loop, rather than
recomputing an ``O(r^3)`` recursion per candidate.

Receding horizon: plan ``H`` steps, execute the first ``K`` (``cadence``), replan.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from dbwm.inference.kalman import kalman_predict


def admissible_bounds(
    rain_forecast: jnp.ndarray, u_max: float, n_actuators: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The admissible irrigation box ``U_{t+k}`` of Algorithm 2.

    :param rain_forecast: ``(H,)`` forecast rain magnitude for each planning step
                          (the *contemporaneous* precipitation channel only -- lags
                          must not gate the actuator).
    :param u_max: maximum irrigation magnitude per step.
    :param n_actuators: ``ell_u``, number of irrigation channels.
    :return: ``(lo, hi)`` each ``(H, ell_u)``. ``hi`` collapses to 0 on rainy steps,
             making irrigation-during-rain unrepresentable rather than merely costly.
    """
    dry = (rain_forecast <= 0.0).astype(jnp.float32)  # (H,)
    hi = (u_max * dry)[:, None] * jnp.ones((1, n_actuators))  # (H, ell_u)
    lo = jnp.zeros_like(hi)
    return lo, hi


def rollout(
    w0: jnp.ndarray,
    u_plan: jnp.ndarray,
    a: jnp.ndarray,
    b_u: jnp.ndarray,
    b_p: Optional[jnp.ndarray] = None,
    p_forecast: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Roll the input-affine dynamics forward under a candidate irrigation plan.

    :param w0: ``(r,)`` current state estimate ``w_{t|t}``.
    :param u_plan: ``(H, ell_u)`` candidate irrigation sequence (the decision).
    :param a: ``(r, r)`` transition matrix.
    :param b_u: ``(r, ell_u)`` irrigation directions.
    :param b_p: ``(r, ell_p)`` precipitation directions or ``None``.
    :param p_forecast: ``(H, ell_p)`` known precipitation forecast or ``None``.
    :return: ``(H, r)`` predicted states ``w_{t+1}, ..., w_{t+H}``.
    """

    def step(w, k):
        w_next = a @ w + b_u @ u_plan[k]
        if b_p is not None and p_forecast is not None:
            w_next = w_next + b_p @ p_forecast[k]
        return w_next, w_next

    _, states = jax.lax.scan(step, w0, jnp.arange(u_plan.shape[0]))
    return states


def covariance_trace_sequence(
    p0: jnp.ndarray, a: jnp.ndarray, q: jnp.ndarray, horizon: int, gamma_dyn: float = 0.0
) -> jnp.ndarray:
    """
    Propagate ``tr(P_{t+k})`` over the planning horizon.

    Control-independent by Theorem 4.3 (the known forcing moves the mean, not the
    covariance), so this is computed **once** per replan instead of once per CEM
    candidate -- turning an ``O(n_cand * H * r^3)`` cost into ``O(H * r^3)``.

    :param p0: ``(r, r)`` current state covariance ``P_{t|t}``.
    :param a: ``(r, r)`` transition matrix.
    :param q: ``(r, r)`` process-noise covariance.
    :param horizon: ``H``.
    :param gamma_dyn: dynamics-residual covariance inflation.
    :return: ``(H,)`` traces ``tr(P_{t+1}), ..., tr(P_{t+H})``.
    """

    def step(p, _):
        _, p_next = kalman_predict(
            jnp.zeros(a.shape[0]), p, a, q, gamma_dyn=gamma_dyn
        )
        return p_next, jnp.trace(p_next)

    _, traces = jax.lax.scan(step, p0, jnp.arange(horizon))
    return traces


def plan_cost(
    w_states: jnp.ndarray,
    u_plan: jnp.ndarray,
    w_goal: jnp.ndarray,
    trace_p: jnp.ndarray,
    gamma: float,
    beta_plan: float,
    lambda_u: float,
) -> jnp.ndarray:
    """
    Discounted planning cost of Algorithm 2::

        C = sum_k gamma^k [ ||w_{t+k+1} - w_g||^2 + beta_plan tr(P_{t+k+1})
                            + lambda_u ||u_{t+k}||^2 ]

    :param w_states: ``(H, r)`` rolled-out states.
    :param u_plan: ``(H, ell_u)`` candidate irrigation sequence.
    :param w_goal: ``(r,)`` goal state ``w_g = phi_theta(o_g)``.
    :param trace_p: ``(H,)`` propagated covariance traces.
    :param gamma: discount factor.
    :param beta_plan: weight on the uncertainty term.
    :param lambda_u: weight on the control (water-use) term.
    :return: scalar cost.
    """
    h = u_plan.shape[0]
    disc = gamma ** jnp.arange(h)  # (H,)
    goal_err = jnp.sum((w_states - w_goal[None, :]) ** 2, axis=-1)  # (H,)
    ctrl = jnp.sum(u_plan**2, axis=-1)  # (H,)
    return jnp.sum(disc * (goal_err + beta_plan * trace_p + lambda_u * ctrl))


def cem_plan(
    key: jax.Array,
    w0: jnp.ndarray,
    p0: jnp.ndarray,
    w_goal: jnp.ndarray,
    a: jnp.ndarray,
    b_u: jnp.ndarray,
    q: jnp.ndarray,
    cfg,
    b_p: Optional[jnp.ndarray] = None,
    p_forecast: Optional[jnp.ndarray] = None,
    rain_forecast: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Optimise the irrigation sequence ``u_{t:t+H}`` by the Cross-Entropy Method.

    Each iteration samples ``n_candidates`` plans from a diagonal Gaussian, **clips
    them into the admissible box** (so the no-irrigation-during-rain rule holds
    exactly, for every sample, at every iteration), scores them by :func:`plan_cost`,
    and refits the Gaussian to the elite set.

    :param key: PRNG key.
    :param w0: ``(r,)`` current state estimate.
    :param p0: ``(r, r)`` current state covariance.
    :param w_goal: ``(r,)`` goal state.
    :param a: ``(r, r)`` transition matrix.
    :param b_u: ``(r, ell_u)`` irrigation directions.
    :param q: ``(r, r)`` process-noise covariance.
    :param cfg: a :class:`~dbwm.config.PlanningConfig`.
    :param b_p: ``(r, ell_p)`` precipitation directions or ``None``.
    :param p_forecast: ``(H, ell_p)`` known precipitation forecast or ``None``.
    :param rain_forecast: ``(H,)`` contemporaneous rain magnitude used to gate the
                          actuator. Defaults to the first column of ``p_forecast``
                          (the lag-0 channel), or all-dry if no forecast is given.
    :return: dict with ``u_plan`` ``(H, ell_u)``, ``cost`` (scalar), ``w_states``
             ``(H, r)`` under the winning plan, and ``trace_p`` ``(H,)``.
    """
    h = cfg.horizon
    ell_u = b_u.shape[1]

    if rain_forecast is None:
        rain_forecast = (
            p_forecast[:, 0] if p_forecast is not None else jnp.zeros(h)
        )
    lo, hi = admissible_bounds(rain_forecast, cfg.u_max, ell_u)

    # Control-independent (Theorem 4.3): compute once, not per candidate.
    trace_p = covariance_trace_sequence(p0, a, q, h, cfg.gamma_dyn_inflation)

    def score(u_plan):
        states = rollout(w0, u_plan, a, b_u, b_p, p_forecast)
        cost = plan_cost(
            states, u_plan, w_goal, trace_p, cfg.gamma, cfg.beta_plan, cfg.lambda_u
        )
        return cost, states

    score_batch = jax.vmap(score)

    # Start centred in the admissible box, so the initial population is feasible
    # and (on rainy steps, where hi == lo == 0) already pinned to zero.
    mu = (lo + hi) / 2.0  # (H, ell_u)
    sigma = jnp.maximum((hi - lo) / 2.0, cfg.sigma_min)  # (H, ell_u)

    for _ in range(cfg.n_iters):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (cfg.n_candidates, h, ell_u))
        cand = mu[None] + sigma[None] * noise
        cand = jnp.clip(cand, lo[None], hi[None])  # enforce U_{t+k} exactly

        costs, _ = score_batch(cand)  # (n_candidates,)
        elite_idx = jnp.argsort(costs)[: cfg.n_elite]
        elites = cand[elite_idx]  # (n_elite, H, ell_u)

        mu_new = jnp.mean(elites, axis=0)
        sigma_new = jnp.std(elites, axis=0)
        # Smooth the update (standard CEM damping) to avoid premature collapse.
        mu = cfg.alpha_smooth * mu + (1.0 - cfg.alpha_smooth) * mu_new
        sigma = jnp.maximum(
            cfg.alpha_smooth * sigma + (1.0 - cfg.alpha_smooth) * sigma_new,
            cfg.sigma_min,
        )

    u_plan = jnp.clip(mu, lo, hi)
    cost, states = score(u_plan)
    return {"u_plan": u_plan, "cost": cost, "w_states": states, "trace_p": trace_p}


def receding_horizon_control(
    key: jax.Array,
    w0: jnp.ndarray,
    p0: jnp.ndarray,
    w_goal: jnp.ndarray,
    a: jnp.ndarray,
    b_u: jnp.ndarray,
    q: jnp.ndarray,
    cfg,
    n_steps: int,
    b_p: Optional[jnp.ndarray] = None,
    p_series: Optional[jnp.ndarray] = None,
    sigma_eps2: float = 1e-2,
    phi_obs_seq: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Closed-loop receding-horizon control: plan ``H``, execute ``cadence`` actions,
    re-estimate, replan (Algorithm 2, "Execute first K irrigation actions, then
    replan").

    If encoded observations ``phi_obs_seq`` are supplied, each executed step is
    followed by a Kalman UPDATE, so the plan is recomputed from a *corrected* state
    -- this is what keeps the controller's error bounded by ``P_inf`` (Theorem 4.3)
    instead of drifting with the open-loop bound of Theorem 4.2.

    :param key: PRNG key.
    :param w0: ``(r,)`` initial state estimate.
    :param p0: ``(r, r)`` initial covariance.
    :param w_goal: ``(r,)`` goal state.
    :param a: ``(r, r)`` transition matrix.
    :param b_u: ``(r, ell_u)`` irrigation directions.
    :param q: ``(r, r)`` process-noise covariance.
    :param cfg: a :class:`~dbwm.config.PlanningConfig`.
    :param n_steps: total number of steps to control.
    :param b_p: ``(r, ell_p)`` precipitation directions or ``None``.
    :param p_series: ``(n_steps + H, ell_p)`` precipitation forecasts or ``None``.
    :param sigma_eps2: measurement-noise variance (for the Kalman update).
    :param phi_obs_seq: ``(n_steps, r)`` encoded observations, or ``None`` for a
                        pure open-loop (no-correction) run.
    :return: dict with ``u_executed`` ``(n_steps, ell_u)``, ``w_traj``
             ``(n_steps, r)`` and ``costs`` ``(n_plans,)``.
    """
    from dbwm.inference.kalman import kalman_update

    h, ell_u = cfg.horizon, b_u.shape[1]
    w, p = w0, p0
    u_exec, w_traj, costs = [], [], []

    step = 0
    while step < n_steps:
        if p_series is not None:
            # Pad the forecast window if we run off the end of the series.
            window = p_series[step : step + h]
            if window.shape[0] < h:
                pad = jnp.zeros((h - window.shape[0], p_series.shape[1]))
                window = jnp.concatenate([window, pad], axis=0)
        else:
            window = None

        key, subkey = jax.random.split(key)
        plan = cem_plan(
            subkey, w, p, w_goal, a, b_u, q, cfg,
            b_p=b_p, p_forecast=window,
        )
        costs.append(plan["cost"])

        # Execute the first `cadence` actions, then replan.
        k_exec = min(cfg.cadence, n_steps - step)
        for k in range(k_exec):
            u_k = plan["u_plan"][k]
            p_k = None if window is None else window[k]
            w, p = kalman_predict(
                w, p, a, q,
                b=b_p if p_k is not None else None,
                u=p_k,
                gamma_dyn=cfg.gamma_dyn_inflation,
            )
            w = w + b_u @ u_k  # the chosen actuation
            if phi_obs_seq is not None and step + k < phi_obs_seq.shape[0]:
                w, p = kalman_update(w, p, phi_obs_seq[step + k], sigma_eps2)
            u_exec.append(u_k)
            w_traj.append(w)
        step += k_exec

    return {
        "u_executed": jnp.stack(u_exec, axis=0),
        "w_traj": jnp.stack(w_traj, axis=0),
        "costs": jnp.stack(costs, axis=0),
    }
