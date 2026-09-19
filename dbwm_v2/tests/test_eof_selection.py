"""
Tests for held-out selection of the empirical (EOF) mode count.

The empirical block is a **data-driven** basis: its modes are the singular
vectors of the training residual, so it reconstructs the frames it was fitted on
almost exactly by construction. Two consequences drive everything here.

**The fitted residual is not the forecast floor.** On the real record the basis
reconstructed training dates at 0.0016 NDVI and test dates at 0.044 -- a 28x gap.
With the operator shrunk to persistence the forecast *is* the origin's
reconstruction, so that 0.044 entered every horizon additively:
``ubRMSE(h) = sqrt(persistence(h)^2 + 0.0437^2)`` reproduced all six measured
values to within 0.0005 NDVI.

**The energy criterion cannot see it.** Cumulative training-residual variance is
monotone in ``q``, so it can only ever be stopped by the ``--eof-modes`` cap, and
it has no opinion on whether the modes describe the field or those particular
frames. Only a held-out score can tell a mode-starved basis (raise the cap) from
an over-fitted one (lower it) -- and the two call for opposite actions.
"""
import numpy as np
import pytest

from dbwm.gp.empirical_basis import build_augmented_basis


def _smooth_basis(h, w, r):
    """An orthonormal-ish smooth coordinate basis, standing in for Psi."""
    yy, xx = np.mgrid[0:h, 0:w]
    cols = []
    for k in range(1, r // 2 + 1):
        cols.append(np.sin(2 * np.pi * k * xx / w) * np.cos(2 * np.pi * k * yy / h))
        cols.append(np.cos(2 * np.pi * k * xx / w) * np.sin(2 * np.pi * k * yy / h))
    phi = np.stack(cols[:r]).reshape(r, h * w).T
    return phi / np.linalg.norm(phi, axis=0, keepdims=True)


def _mosaic_field(t=300, h=24, w=22, r=16, drift=True, seed=0):
    """
    Frames = smooth signal + a piecewise-constant parcel mosaic + noise.

    ``drift`` re-draws the parcel levels in blocks, which is what crop rotation
    does: the mosaic *geometry* is static but which parcels are green changes, so
    modes fitted on early frames only partly span later ones.
    """
    rng = np.random.default_rng(seed)
    phi = _smooth_basis(h, w, r)
    yy, xx = np.mgrid[0:h, 0:w]
    plots = ((xx // 5) * 4 + (yy // 6)).reshape(-1)
    amp = np.cumsum(rng.normal(0, 0.05, (t, r)), axis=0)
    out = np.zeros((t, h * w))
    for i in range(t):
        block = i // 50 if drift else 0
        lvl = np.random.default_rng(900 + block).normal(0, 0.6, plots.max() + 1)
        out[i] = amp[i] @ phi.T + lvl[plots] + rng.normal(0, 0.02, h * w)
    return phi, out.reshape(t, h, w), np.ones(h * w, dtype=bool), np.ones(t, dtype=bool)


def test_holdout_selection_reports_both_sides_of_the_gap():
    """
    The basis must report fitted AND held-out reconstruction, never just fitted.

    Reporting only the fitted number is how a 28x generalisation gap stayed
    invisible while every log line said ``reconstruction R2 = 1.000``.
    """
    phi, frames, valid, train = _mosaic_field()
    aug = build_augmented_basis(phi, frames, valid, train, n_modes=64, ridge=1e-3)
    assert np.isfinite(aug.rmse_fit) and np.isfinite(aug.rmse_holdout)
    assert aug.rmse_holdout >= aug.rmse_fit
    assert aug.generalisation_gap >= 1.0
    assert aug.holdout_variance is not None
    assert aug.holdout_variance.shape == (frames[0].size,)
    # The per-pixel variance must be consistent with the scalar RMSE it summarises.
    assert np.sqrt(aug.holdout_variance[valid].mean()) == pytest.approx(
        aug.rmse_holdout, rel=1e-6
    )


def test_holdout_variance_exceeds_the_in_sample_residual():
    """
    The band must be fed the held-out residual, not the fitted one.

    The modes ARE the training residual's SVD, so the in-sample residual is near
    zero by construction and using it restates exactly the under-dispersion the
    representation term exists to remove.
    """
    phi, frames, valid, train = _mosaic_field()
    aug = build_augmented_basis(phi, frames, valid, train, n_modes=64, ridge=1e-3)

    # Reconstruct in-sample through the SAME basis, the way the band used to.
    y = frames.reshape(frames.shape[0], -1)[:, valid]
    phi_v, off = aug.phi[valid], aug.offset[valid]
    lam = phi_v.T @ phi_v + 1e-3 * np.eye(phi_v.shape[1])
    wts = np.linalg.solve(lam, phi_v.T @ (y - off).T).T
    in_sample = float(np.sqrt(np.mean((y - (wts @ phi_v.T + off)) ** 2)))
    assert aug.rmse_holdout > in_sample


def test_selection_walks_a_grid_and_records_it():
    """The q-versus-held-out-RMSE curve is the evidence, so it must be kept."""
    phi, frames, valid, train = _mosaic_field()
    aug = build_augmented_basis(phi, frames, valid, train, n_modes=48, ridge=1e-3)
    grid = aug.selection["grid"]
    assert aug.selection["rule"] == "holdout"
    assert len(grid) >= 4
    qs = [row["q"] for row in grid]
    assert qs == sorted(qs) and 0 in qs
    # The selected q is the argmin of the held-out column, by construction.
    best = min(grid, key=lambda row: row["rmse_holdout"])
    assert best["q"] == aug.q_empirical
    # Fitted error falls monotonically in q; held-out need not. That asymmetry is
    # the entire reason the energy rule is the wrong criterion.
    fits = [row["rmse_fit"] for row in grid]
    assert fits == sorted(fits, reverse=True)


def test_energy_rule_answers_a_training_only_question():
    """
    ``energy`` stops at a fixed share of the TRAINING residual, or at the cap.

    Either way the stopping rule never looks at data the modes were not fitted
    to, so it cannot distinguish a mode-starved basis from an over-fitted one --
    which is the whole decision. Raising the threshold monotonically raises q
    until the cap binds, and the held-out cost of that is never consulted.
    """
    phi, frames, valid, train = _mosaic_field()
    q_by_energy = [
        build_augmented_basis(phi, frames, valid, train, n_modes=96,
                              energy=e, ridge=1e-3, select="energy").q_empirical
        for e in (0.90, 0.99, 0.999)
    ]
    assert q_by_energy == sorted(q_by_energy)      # monotone in the threshold
    # A cap below that binds instead, and nothing else can stop it.
    tight = build_augmented_basis(phi, frames, valid, train, n_modes=8,
                                  energy=0.999, ridge=1e-3, select="energy")
    assert tight.q_empirical == 8
    # Even on the ablation path the held-out cost is still measured and reported.
    assert np.isfinite(tight.rmse_holdout)


def test_holdout_selection_never_loses_out_of_sample():
    """
    Whatever q it lands on, held-out selection is not worse out of sample.

    It optimises the held-out score directly, so it either matches the energy
    rule's q or finds one that generalises better. That is the guarantee; which
    direction it moves depends on whether the cap or the optimum binds, and the
    log says which.
    """
    phi, frames, valid, train = _mosaic_field(drift=True, seed=3)
    cap = 96
    chosen = build_augmented_basis(
        phi, frames, valid, train, n_modes=cap, ridge=1e-3, select="holdout"
    )
    capped = build_augmented_basis(
        phi, frames, valid, train, n_modes=cap, ridge=1e-3, select="energy"
    )
    assert chosen.rmse_holdout <= capped.rmse_holdout + 1e-12
    assert chosen.selection["rule"] == "holdout"


def test_zero_modes_is_a_legitimate_selection():
    """
    When Psi already spans the field the empirical block must be allowed to vanish.

    A rule that always appends modes would add unexcited directions that wreck
    the operator's conditioning for no representation gain.
    """
    rng = np.random.default_rng(5)
    h, w, r, t = 20, 18, 12, 200
    phi = _smooth_basis(h, w, r)
    # Frames lie EXACTLY in span(Psi): nothing is left for EOFs to explain.
    frames = (rng.normal(size=(t, r)) @ phi.T).reshape(t, h, w)
    aug = build_augmented_basis(
        phi, frames, np.ones(h * w, dtype=bool), np.ones(t, dtype=bool),
        n_modes=32, ridge=1e-8,
    )
    assert aug.q_empirical == 0
    assert aug.rmse_holdout < 1e-6


def test_selection_split_is_chronological():
    """
    The held-out tail must be the END of the training window, not a random subset.

    Random frames from the same weeks are near-duplicates of their neighbours, so
    a random split reports an in-sample number under a held-out name.
    """
    rng = np.random.default_rng(7)
    h, w, r, t = 16, 14, 8, 200
    phi = _smooth_basis(h, w, r)
    frames = (rng.normal(size=(t, r)) @ phi.T).reshape(t, h, w)
    # Make the LAST 20% of dates structurally different. A chronological split
    # puts exactly those in the holdout, so the score must degrade; a random
    # split would dilute them across both sides and barely move.
    frames[int(0.8 * t):] += 5.0 * rng.normal(size=(h, w))
    aug = build_augmented_basis(
        phi, frames, np.ones(h * w, dtype=bool), np.ones(t, dtype=bool),
        n_modes=16, ridge=1e-6, holdout_fraction=0.2,
    )
    assert aug.rmse_holdout > 10 * aug.rmse_fit
