"""
Tests for the forecast-gain control and the increment parameterisation.

These two changes exist because the pipeline was enforcing the spectral *radius*
while v2 Theorem 4.2 is stated with the spectral *norm*. On the real record the
radius sat at exactly 1.000 while the operator amplified its input error 4.13x per
step, and that factor accounted for the entire observed one-step error. The tests
below pin the properties that make the guarantee real rather than nominal.
"""
from __future__ import annotations

import numpy as np

from dbwm.dynamics.memory import (
    MemoryOperator, _blend_to_persistence, _persistence_blocks,
    calibrate_persistence_blend, enforce_forecast_gain, forecast_gain,
    identify_memory,
)


def test_persistence_has_unit_gain_at_every_horizon():
    """
    The reference point of the whole scheme.

    ``w_{t+h} = w_t`` neither amplifies nor damps the state error, so its gain is
    exactly 1. Everything else is measured against this.
    """
    op = MemoryOperator(blocks=_persistence_blocks(6, 4))
    assert np.allclose(forecast_gain(op, 8), 1.0)


def test_gain_sees_amplification_the_spectral_radius_misses():
    """
    A non-normal operator can have ``rho <= 1`` and still blow up over a horizon.

    This is the exact failure mode observed on the real record, reproduced in
    miniature: the radius passes, the gain does not.
    """
    r = 6
    blocks = np.zeros((2, r, r))
    upper = np.eye(r) * 0.5 + np.triu(np.full((r, r), 6.0), 1)
    blocks[0] = upper
    op = MemoryOperator(blocks=blocks)
    rho = float(np.max(np.abs(np.linalg.eigvals(op.companion()))))
    assert rho <= 1.0 + 1e-9
    assert np.max(forecast_gain(op, 6)) > 5.0


def test_enforce_gain_reaches_the_cap():
    """Blending must actually bring the gain under the requested cap."""
    rng = np.random.default_rng(1)
    op = MemoryOperator(blocks=rng.normal(size=(3, 5, 5)))
    fixed = enforce_forecast_gain(op, gain_max=1.2, horizon=6)
    assert np.max(forecast_gain(fixed, 6)) <= 1.2 + 1e-6


def test_enforce_gain_leaves_a_compliant_operator_untouched():
    """A model already inside the budget must not be shrunk."""
    op = MemoryOperator(blocks=_persistence_blocks(4, 2))
    out = enforce_forecast_gain(op, gain_max=1.0, horizon=4)
    assert np.allclose(out.blocks, op.blocks)


def test_blending_shrinks_toward_persistence_not_toward_zero():
    """
    The anchor matters, and it is the reason the reported forecasts were biased.

    Shrinking toward ZERO drags every prediction to the state-space origin, which
    after the climatology offset is the training mean field -- visible in the
    triptychs as a large negative bias on top of the RMSE. Shrinking toward
    PERSISTENCE keeps predicting ``w_t``. At ``s = 0`` the operator must be exactly
    persistence, not exactly zero.
    """
    rng = np.random.default_rng(2)
    op = MemoryOperator(blocks=rng.normal(size=(3, 4, 4)))
    at_zero = _blend_to_persistence(op, 0.0)
    assert np.allclose(at_zero.blocks[0], np.eye(4))
    assert np.allclose(at_zero.blocks[1:], 0.0)
    w = rng.normal(size=4)
    stacked = np.stack([w] + [rng.normal(size=4) for _ in range(2)])
    assert np.allclose(np.einsum("jrs,js->r", at_zero.blocks, stacked), w)


def test_blend_at_one_returns_the_fitted_operator():
    """``s = 1`` must be the unmodified fit, so the sweep spans both endpoints."""
    rng = np.random.default_rng(3)
    op = MemoryOperator(blocks=rng.normal(size=(2, 5, 5)))
    assert np.allclose(_blend_to_persistence(op, 1.0).blocks, op.blocks)


def test_calibration_never_loses_to_persistence_on_holdout():
    """
    ``s = 0`` is always a candidate, so the selection cannot end up worse than
    doing nothing on the data it was selected on.
    """
    rng = np.random.default_rng(4)
    t, r = 400, 3
    w = np.zeros((t, r))
    for i in range(1, t):  # a slow random walk: persistence is strong here
        w[i] = 0.98 * w[i - 1] + 0.05 * rng.normal(size=r)
    op, b_p = identify_memory(w, 2, None, "unstructured", 1e-3)
    _, rep = calibrate_persistence_blend(op, b_p, w, None, None, 4, gain_max=3.0)
    assert rep["holdout_error"] <= rep["holdout_error_persistence"] + 1e-9


def test_calibration_recovers_real_dynamics_when_they_exist():
    """
    When the data genuinely moves, the blend must NOT collapse to persistence.

    A rotation is the clean case: persistence is a poor model of it, so a
    correctly working selector has to keep most of the fitted operator.
    """
    rng = np.random.default_rng(5)
    theta = 0.35
    a = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = 600
    w = np.zeros((t, 2))
    w[0] = [1.0, 0.0]
    for i in range(1, t):
        w[i] = a @ w[i - 1] + 0.002 * rng.normal(size=2)
    op, b_p = identify_memory(w, 1, None, "unstructured", 1e-6)
    _, rep = calibrate_persistence_blend(op, b_p, w, None, None, 4, gain_max=3.0)
    assert rep["blend"] > 0.5, rep
    assert rep["holdout_error"] < 0.5 * rep["holdout_error_persistence"]


def test_gain_cap_rejects_candidates():
    """The hard rail must actually exclude over-gained candidates."""
    rng = np.random.default_rng(6)
    t, r = 300, 3
    w = np.cumsum(rng.normal(size=(t, r)) * 0.05, axis=0)
    op, b_p = identify_memory(w, 2, None, "unstructured", 1e-4)
    _, rep = calibrate_persistence_blend(op, b_p, w, None, None, 6, gain_max=1.05)
    assert rep["gain"] <= 1.05 + 1e-6


def test_increment_parameterisation_shrinks_toward_persistence():
    """
    With a strong ridge the increment fit must approach the identity, and the
    absolute fit must approach zero. That difference is the entire point: one
    degenerates to "predict today's field", the other to "predict nothing".
    """
    rng = np.random.default_rng(7)
    t, r = 200, 4
    w = np.cumsum(rng.normal(size=(t, r)) * 0.1, axis=0)
    inc, _ = identify_memory(w, 1, None, "unstructured", ridge_mu=1e8, increment=True)
    abs_, _ = identify_memory(w, 1, None, "unstructured", ridge_mu=1e8, increment=False)
    assert np.linalg.norm(inc.blocks[0] - np.eye(r)) < 1e-3
    assert np.linalg.norm(abs_.blocks[0]) < 1e-3


def test_increment_and_absolute_span_the_same_model_class():
    """
    With no regularisation the two parameterisations must give the SAME operator.

    They differ only in where the ridge pulls, so an unregularised fit has to
    agree -- otherwise the reparameterisation is changing the model rather than
    the prior, and the comparison between them would be meaningless.
    """
    rng = np.random.default_rng(8)
    t, r = 300, 3
    w = np.cumsum(rng.normal(size=(t, r)) * 0.1, axis=0)
    inc, _ = identify_memory(w, 2, None, "unstructured", ridge_mu=1e-12, increment=True)
    abs_, _ = identify_memory(w, 2, None, "unstructured", ridge_mu=1e-12, increment=False)
    assert np.allclose(inc.blocks, abs_.blocks, atol=1e-5)
