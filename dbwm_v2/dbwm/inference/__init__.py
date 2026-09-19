"""Inference: weight-space Kalman observer, map forecasting and CEM planning."""
from dbwm.inference.kalman import (
    kalman_predict,
    kalman_update,
    filter_sequence,
    open_loop_forecast,
)
from dbwm.inference.observer import (
    encode_sequence,
    decode_maps,
    variance_map,
    closed_loop_filter,
    open_loop_forecast_maps,
)
from dbwm.inference.planning import (
    admissible_bounds,
    rollout,
    covariance_trace_sequence,
    plan_cost,
    cem_plan,
    receding_horizon_control,
)

__all__ = [
    "kalman_predict",
    "kalman_update",
    "filter_sequence",
    "open_loop_forecast",
    "encode_sequence",
    "decode_maps",
    "variance_map",
    "closed_loop_filter",
    "open_loop_forecast_maps",
    # Algorithm 2, PLAN block.
    "admissible_bounds",
    "rollout",
    "covariance_trace_sequence",
    "plan_cost",
    "cem_plan",
    "receding_horizon_control",
]
