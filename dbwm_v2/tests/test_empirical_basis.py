"""
Tests for the empirical completion of the spatial basis.

This module exists because the learned basis plateaued at ``R^2 = 0.958`` on the
real record, and v2 Theorem 4.2 multiplies that residual by the operator gain --
making it, not the dynamics, the term that set the observed forecast floor. The
properties worth pinning are therefore the ones that decide whether the floor
really comes down and whether it comes down *honestly*.
"""
from __future__ import annotations

import numpy as np
import pytest

from dbwm.gp.empirical_basis import build_augmented_basis, _reconstruction_r2


def _mosaic(t=60, h=16, w=16, n_plots=8, seed=0):
    """
    A miniature agricultural mosaic: piecewise-constant plots on their own cycles.

    Smooth bases are bad at exactly this, which is the situation the module is
    for; a smooth synthetic field would not exercise it.

    :return: ``(frames (T,H,W), valid (H*W,), smooth psi (H*W, r))``
    """
    rng = np.random.default_rng(seed)
    plot = (np.arange(h * w) % n_plots).reshape(h, w)
    phase = rng.uniform(0, 2 * np.pi, n_plots)
    amp = rng.uniform(0.2, 0.6, n_plots)
    base = rng.uniform(-0.5, 0.5, n_plots)
    doy = np.arange(t)
    series = base[None] + amp[None] * np.sin(2 * np.pi * doy[:, None] / 30.0 + phase[None])
    frames = series[:, plot.reshape(-1)].reshape(t, h, w)

    valid = np.ones(h * w, dtype=bool)
    valid[: h * w // 8] = False  # a clip that leaves part of the box invalid
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel() / w, yy.ravel() / h], axis=1)
    # A deliberately SMOOTH basis: low-order Fourier features in the coordinates.
    cols = [np.ones(h * w)]
    for k in range(1, 5):
        for c in (coords[:, 0], coords[:, 1]):
            cols += [np.sin(2 * np.pi * k * c), np.cos(2 * np.pi * k * c)]
    return frames, valid, np.stack(cols, axis=1)


def test_augmentation_raises_reconstruction_r2():
    """The whole point: the floor must come down, and measurably."""
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[: 40] = True
    aug = build_augmented_basis(psi, frames, valid, train, n_modes=32)
    assert aug.r2_after > aug.r2_before
    assert aug.r2_after > 0.99


def test_climatology_alone_helps_on_a_static_mosaic():
    """
    A static plot layout should be largely removed by the per-pixel mean alone.

    This is the cheap half of the fix and it is worth isolating: if the offset
    were not pulling its weight, the EOF block would be silently doing all the
    work and ``q`` could be cut.
    """
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    with_clim = build_augmented_basis(psi, frames, valid, train, n_modes=0,
                                      use_climatology=True)
    without = build_augmented_basis(psi, frames, valid, train, n_modes=0,
                                    use_climatology=False)
    assert with_clim.r2_after > without.r2_after


def test_modes_are_fitted_on_training_dates_only():
    """
    No leakage: perturbing only the TEST dates must not change the basis.

    A basis fitted on the whole record would improve every forecast metric for a
    reason that has nothing to do with the dynamics, and would do so invisibly.
    """
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    a = build_augmented_basis(psi, frames, valid, train, n_modes=16)

    tampered = frames.copy()
    tampered[40:] += 5.0  # wreck the test period only
    b = build_augmented_basis(psi, tampered, valid, train, n_modes=16)

    assert np.allclose(a.phi, b.phi)
    assert np.allclose(a.offset, b.offset)


def test_invalid_pixels_carry_no_basis():
    """Pixels outside the clip must get zero features and zero offset."""
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    aug = build_augmented_basis(psi, frames, valid, train, n_modes=16)
    assert np.allclose(aug.phi[~valid, aug.r_learned:], 0.0)
    assert np.allclose(aug.offset[~valid], 0.0)


def test_empirical_block_is_orthogonal_to_the_learned_block():
    """
    The modes must span what ``Psi`` *cannot* reach, not duplicate it.

    Fitting EOFs to the raw anomaly instead would re-learn directions already
    covered, wasting modes and leaving ``Lambda_X`` near-singular through
    collinear columns.
    """
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    aug = build_augmented_basis(psi, frames, valid, train, n_modes=16)
    learned = aug.phi[valid, : aug.r_learned]
    empirical = aug.phi[valid, aug.r_learned:]
    cross = learned.T @ empirical
    rel = np.linalg.norm(cross) / (
        np.linalg.norm(learned) * np.linalg.norm(empirical)
    )
    assert rel < 0.05, rel


def test_zero_modes_is_a_no_op_on_the_feature_block():
    """``n_modes=0`` must leave the learned basis exactly as it was."""
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    aug = build_augmented_basis(psi, frames, valid, train, n_modes=0)
    assert aug.q_empirical == 0
    assert aug.phi.shape[1] == psi.shape[1]
    assert np.allclose(aug.phi[valid], psi[valid])


def test_energy_cap_limits_the_number_of_modes():
    """A low energy target must buy fewer modes than a high one."""
    frames, valid, psi = _mosaic()
    train = np.zeros(frames.shape[0], bool)
    train[:40] = True
    lo = build_augmented_basis(psi, frames, valid, train, n_modes=64, energy=0.5)
    hi = build_augmented_basis(psi, frames, valid, train, n_modes=64, energy=0.9999)
    assert lo.q_empirical <= hi.q_empirical


def test_r2_helper_matches_a_hand_computation():
    """The scoring helper must agree with the definition it claims to use."""
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(30, 4))
    y = rng.normal(size=(5, 30))
    offset = np.zeros(30)
    got = _reconstruction_r2(phi, y, offset, ridge=1e-6)

    lam = phi.T @ phi + 1e-6 * np.eye(4)
    w = np.linalg.solve(lam, phi.T @ y.T).T
    pred = w @ phi.T
    ss_res = np.sum((y - pred) ** 2, axis=1)
    ss_tot = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    assert np.isclose(got, float(np.median(1 - ss_res / ss_tot)))


def test_too_few_frames_is_refused():
    """One training frame cannot define a covariance; say so rather than guess."""
    frames, valid, psi = _mosaic(t=4)
    train = np.zeros(4, bool)
    train[0] = True
    with pytest.raises(ValueError, match="at least 2"):
        build_augmented_basis(psi, frames, valid, train, n_modes=4)
