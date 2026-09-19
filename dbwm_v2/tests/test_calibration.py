"""
Tests for the predictive-interval calibration and the two error-attribution reports.

These cover the three defects the 2026-04-23 figures exposed, each of which was
invisible in the headline metrics:

**The interval omitted its dominant variance term.** ``uncertainty_band`` summed
the propagated latent variance and ``sigma_eps^2`` but not the *representation*
residual -- the part of the field the basis provably cannot reach. Reported
sigma came out at exactly half the realised error and coverage at 58% against a
nominal 95%.

**rho was unconstrained.** ``stability_metric = "forecast_gain"`` put
``enforce_stability`` in an ``else`` branch, so ``DynamicsConfig.rho_max = 1.0``
was declared and applied nowhere. A non-normal operator then satisfied the
6-step gain cap while sitting at ``rho = 1.37``, i.e. unstable.

**"Worst pixel = 0.30" was not actionable.** A representation failure and an
unpredicted harvest look identical in that number and only one is fixable.
"""
import numpy as np
import pytest

from dbwm.dynamics.memory import (
    MemoryOperator, enforce_spectral_radius, forecast_gain, is_persistence,
    spectral_radius_lifted,
)
from dbwm.evaluation.error_budget import bias_attribution, worst_pixel_attribution
from dbwm.evaluation.geotiff_export import representation_variance, uncertainty_band


class _Extractor:
    """Minimal GP state extractor: a linear basis with an optional offset."""

    def __init__(self, phi, sigma_eps2=1e-4, offset=None):
        self.phi = np.asarray(phi, dtype=float)
        self.sigma_eps2 = sigma_eps2
        self.offset = offset

    def decode(self, w):
        out = np.asarray(w) @ self.phi.T
        return out if self.offset is None else out + self.offset


# --------------------------------------------------------------------------- #
# The missing variance term
# --------------------------------------------------------------------------- #
def test_representation_variance_finds_what_the_basis_cannot_reach():
    """
    A field component orthogonal to the basis must appear in the variance.

    That component is exactly the forecast floor: it is present the moment the
    field is encoded, no dynamics are involved, and it is what the predictive
    band was silently leaving out.
    """
    rng = np.random.default_rng(0)
    n_pix, r, t = 40, 3, 25
    phi = np.linalg.qr(rng.normal(size=(n_pix, r)))[0]
    # A direction the basis cannot express, concentrated on a few pixels.
    hidden = np.zeros(n_pix)
    hidden[7] = 1.0
    hidden -= phi @ (phi.T @ hidden)
    hidden /= np.linalg.norm(hidden)

    w = rng.normal(size=(t, r))
    frames = (w @ phi.T + 0.4 * rng.normal(size=(t, 1)) * hidden).reshape(t, 5, 8)
    valid = np.ones(n_pix, dtype=bool)
    ex = _Extractor(phi)

    got = representation_variance(
        ex, w, frames, valid, np.ones(t, dtype=bool), scale=2.0
    )
    # The unrepresentable direction shows up, scaled into physical units.
    assert got["variance"][7] > 10 * np.median(got["variance"])
    assert got["rms"] > 0
    # A basis that spans the field leaves nothing behind.
    clean = (w @ phi.T).reshape(t, 5, 8)
    exact = representation_variance(
        ex, w, clean, valid, np.ones(t, dtype=bool), scale=2.0
    )
    assert exact["rms"] < 1e-9


def test_uncertainty_band_adds_the_representation_term():
    """The band must grow by exactly the representation variance, in quadrature."""
    phi = np.eye(6)
    ex = _Extractor(phi, sigma_eps2=0.01)
    base = uncertainty_band(0.6, ex, None, scale=2.0)
    extra = np.full(6, 0.09)                       # a physical-unit variance
    with_repr = uncertainty_band(0.6, ex, None, scale=2.0, repr_var=extra)
    assert np.allclose(with_repr**2, base**2 + extra)
    assert (with_repr > base).all()


def test_omitting_the_representation_term_reproduces_the_observed_under_coverage():
    """
    The 58% coverage is fully explained by the missing term plus the bias.

    This pins the diagnosis rather than the fix: with the reported half-width
    (0.0638), bias (+0.0424) and ubRMSE (0.0653) from the real ``t+1`` figure, a
    Gaussian error model predicts 57.6% against the 58.4% actually printed. So
    the shortfall is arithmetic, not a subtle modelling effect -- and restoring
    the term is what closes it.
    """
    from math import erf, sqrt

    phi_cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    half, bias, ub = 0.0638, 0.0424, 0.0653
    predicted = phi_cdf((half - bias) / ub) - phi_cdf((-half - bias) / ub)
    assert 0.56 < predicted < 0.60          # reported: 0.584

    # With the band widened to the true error scale AND the bias removed, the
    # same arithmetic lands on nominal.
    fixed = phi_cdf(1.959964) - phi_cdf(-1.959964)
    assert fixed == pytest.approx(0.95, abs=1e-3)


# --------------------------------------------------------------------------- #
# The spectral-radius rail
# --------------------------------------------------------------------------- #
def _op(a0, order=2):
    """A memory operator with the given A_0 and zero memory blocks."""
    r = a0.shape[0]
    blocks = np.zeros((order, r, r))
    blocks[0] = a0
    return MemoryOperator(blocks=blocks, parameterization="unstructured")


def test_rho_rail_leaves_a_stable_operator_alone():
    """A rail that fires on a stable operator would be shrinking for nothing."""
    op = _op(np.diag([0.8, 0.5]))
    out, rep = enforce_spectral_radius(op, 1.0)
    assert out is op
    assert rep["blend"] == 1.0
    assert rep["rho_after"] == pytest.approx(rep["rho_before"])


def test_rho_rail_pulls_an_unstable_operator_back_to_the_cap():
    """
    An operator with rho > 1 must be blended until rho <= rho_max.

    This is the 1.37 case: the model is stable over the fitted horizon only
    because the gain cap holds ||S A^h||, while the operator itself diverges.
    """
    op = _op(np.diag([1.4, 0.9]))
    assert spectral_radius_lifted(op) > 1.0
    out, rep = enforce_spectral_radius(out_op := op, 1.0)
    assert rep["rho_before"] > 1.0
    assert rep["rho_after"] <= 1.0 + 1e-6
    assert 0.0 <= rep["blend"] < 1.0
    assert spectral_radius_lifted(out) <= 1.0 + 1e-6


def test_rho_rail_is_always_feasible_because_persistence_sits_at_one():
    """
    Persistence has rho = 1 exactly, so the bisection can never fail.

    Its matrix polynomial is ``lambda^{L-1}(lambda - 1) I``: roots at 1 and 0.
    That is what makes blending toward persistence a safe rail, where scaling
    toward zero would instead pull every forecast to the training mean field.
    """
    op = _op(np.diag([50.0, 40.0]), order=3)
    out, rep = enforce_spectral_radius(op, 1.0)
    assert rep["rho_after"] <= 1.0 + 1e-6
    # Driven this hard the rail lands on (or extremely near) persistence itself.
    assert rep["blend"] < 0.05
    assert float(np.max(forecast_gain(out, 6))) <= 1.0 + 1e-6


def test_gain_cap_alone_does_not_bound_rho():
    """
    The two constraints are independent -- which is the whole reason for the rail.

    A non-normal operator can keep ``||S A^h||`` small over a short horizon while
    ``rho > 1``, so a pipeline that checks only the gain reports a stable-looking
    model that diverges just past the horizon it was scored on.
    """
    # Upper-triangular: rho = max|diagonal| = 1.3, but S A^h sees the (0,0) entry.
    a0 = np.array([[0.2, 9.0], [0.0, 1.3]])
    op = _op(a0, order=1)
    assert spectral_radius_lifted(op) == pytest.approx(1.3)
    gains_6 = float(np.max(forecast_gain(op, 6)))
    gains_20 = float(np.max(forecast_gain(op, 20)))
    # Bounded early, unbounded later: exactly the failure mode.
    assert gains_20 > 3 * gains_6
    out, rep = enforce_spectral_radius(op, 1.0)
    assert rep["rho_after"] <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Bias attribution
# --------------------------------------------------------------------------- #
def test_bias_attribution_separates_the_basis_from_the_dynamics():
    """
    A pure decode offset must be charged to the basis, not to the operator.

    Chasing a basis-level offset by tuning the dynamics is the specific mistake
    this report exists to prevent.
    """
    rng = np.random.default_rng(1)
    n_pix, r, t = 30, 4, 20
    phi = np.linalg.qr(rng.normal(size=(n_pix, r)))[0]
    w = rng.normal(size=(t, r))
    truth = w @ phi.T
    # Decode 0.05 (normalised) too high everywhere: a pure encode/decode bias.
    ex = _Extractor(phi, offset=np.full(n_pix, 0.05))
    frames = truth.reshape(t, 5, 6)

    got = bias_attribution(
        ex, w, frames, np.ones(n_pix, dtype=bool), np.ones(t, dtype=bool),
        scale=2.0, forecast_bias=[0.10, 0.10, 0.10],
    )
    assert got["encode_bias"] == pytest.approx(0.05 * 2.0, abs=1e-9)
    # The forecast bias is 0.10; 0.10 of it comes from the basis, 0.0 from
    # the filter and the operator.
    assert got["dynamics_bias"] == pytest.approx(0.0, abs=1e-9)
    assert got["drift_per_step"] == pytest.approx(0.0, abs=1e-9)


def test_bias_attribution_detects_a_drifting_operator():
    """A bias that grows with lead is the operator, and is reported as such."""
    rng = np.random.default_rng(2)
    n_pix, r, t = 20, 3, 15
    phi = np.linalg.qr(rng.normal(size=(n_pix, r)))[0]
    w = rng.normal(size=(t, r))
    ex = _Extractor(phi)
    frames = (w @ phi.T).reshape(t, 4, 5)
    got = bias_attribution(
        ex, w, frames, np.ones(n_pix, dtype=bool), np.ones(t, dtype=bool),
        scale=1.0, forecast_bias=[0.01, 0.02, 0.03, 0.04],
    )
    assert got["encode_bias"] == pytest.approx(0.0, abs=1e-9)
    assert got["drift_per_step"] == pytest.approx(0.01, abs=1e-6)
    assert got["dynamics_bias"] == pytest.approx(0.01, abs=1e-6)


# --------------------------------------------------------------------------- #
# Worst-pixel attribution
# --------------------------------------------------------------------------- #
def _stack(err_per_lead, shape=(4, 4), pixel=(1, 2), n_org=6):
    """Build an error stack whose one hot pixel has the given per-lead RMSE."""
    out, org = {}, {}
    for lead, val in err_per_lead.items():
        e = np.zeros((n_org, *shape))
        e[:, pixel[0], pixel[1]] = val
        out[lead] = e
        org[lead] = np.arange(n_org)
    return out, org


def test_worst_pixel_flat_error_on_a_still_field_is_representation():
    """
    Error constant across leads, truth barely moving: the basis cannot render it.

    This is the fixable case -- more empirical modes or a larger ``r``.
    """
    errs, orgs = _stack({1: 0.30, 2: 0.30, 3: 0.30})
    frames = np.zeros((20, 4, 4))          # the ground never moves
    got = worst_pixel_attribution(
        errs, frames, orgs, np.ones(16, dtype=bool), scale=1.0, n_worst=1
    )
    row = got["pixels"][0]
    assert (row["row"], row["col"]) == (1, 2)
    assert row["verdict"].startswith("representation")
    assert abs(row["growth_per_lead"]) < 1e-9


def test_worst_pixel_matching_ground_motion_is_an_unpredicted_event():
    """
    When the truth moves as much as the error, no linear operator could have known.

    A harvest is a discrete decision by a farmer; it is not a function of NDVI
    history and weather, so this is *not* reducible by tuning L, rho or the
    shrinkage. Saying so is the point -- promising a fix here would be dishonest.
    """
    errs, orgs = _stack({1: 0.30, 2: 0.30, 3: 0.30})
    frames = np.zeros((20, 4, 4))
    # The pixel's truth jumps by 0.4 between consecutive dates, across the whole
    # span the origins reach -- so every scored pair straddles a real change and
    # the mean observed motion (0.4) exceeds the error (0.30).
    frames[:, 1, 2] = 0.4 * (np.arange(20) % 2)
    got = worst_pixel_attribution(
        errs, frames, orgs, np.ones(16, dtype=bool), scale=1.0, n_worst=1
    )
    row = got["pixels"][0]
    assert row["observed_change_by_lead"][0] > row["rmse_by_lead"][0]
    assert row["verdict"].startswith("unpredicted event")


def test_worst_pixel_growing_error_is_charged_to_the_dynamics():
    """An error that grows with lead is the operator's, not the basis's."""
    errs, orgs = _stack({1: 0.05, 2: 0.15, 3: 0.30})
    frames = np.zeros((20, 4, 4))
    got = worst_pixel_attribution(
        errs, frames, orgs, np.ones(16, dtype=bool), scale=1.0, n_worst=1
    )
    row = got["pixels"][0]
    assert row["verdict"].startswith("dynamics")
    assert row["growth_per_lead"] > 0.1


def test_worst_pixel_attribution_ignores_fully_masked_pixels():
    """A pixel that is NaN in every origin must never be ranked as the worst."""
    errs, orgs = _stack({1: 0.05, 2: 0.05})
    errs[1][:, 0, 0] = np.nan
    errs[2][:, 0, 0] = np.nan
    frames = np.zeros((20, 4, 4))
    got = worst_pixel_attribution(
        errs, frames, orgs, np.ones(16, dtype=bool), scale=1.0, n_worst=2
    )
    assert all((r["row"], r["col"]) != (0, 0) for r in got["pixels"])
