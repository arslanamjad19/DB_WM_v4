"""
Tests for the memory lift (v3 Sec. 2.2.1 - 2.2.4).

Each test pins one of the structural results, including the *negative* ones:
Lemma 2.6 (the lifted spectrum is not sigma(A_0)), Lemma 2.7 (process-noise
reachability is automatic), and Theorem 2.8 (over-lagging destroys observability).
"""
import numpy as np
import pytest

from dbwm.dynamics import memory as M
from dbwm.dynamics.memory import _blocks_from_coefficients


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_s2_system(order=3, seed=0, rotate=False):
    """
    Build a stable S2 (per-mode AR) system with a known memory kernel.

    ``rotate=True`` conjugates by a random orthogonal matrix so the modal
    directions are **not** axis-aligned. That matters for the shadedness test:
    with an axis-aligned basis, projecting out a mode also zeroes a whole column
    of ``C``, which conflates Thm 2.8(ii) with plain non-shadedness.
    """
    a0 = np.array(
        [
            [0.55, 0.30, 0.0, 0.0],
            [-0.30, 0.55, 0.0, 0.0],
            [0.0, 0.0, 0.60, 0.0],
            [0.0, 0.0, 0.0, -0.45],
        ]
    )
    if rotate:
        q, _ = np.linalg.qr(np.random.RandomState(seed).randn(4, 4))
        a0 = q @ a0 @ q.T
    modal = M.real_modal_form(a0)
    k = modal.n_blocks
    coeffs = np.zeros((k, order), dtype=complex)
    coeffs[:, 0] = modal.eigenvalues
    for j in range(1, order):
        coeffs[:, j] = (0.25**j) * modal.eigenvalues
    blocks = _blocks_from_coefficients(coeffs, modal, order)
    return M.MemoryOperator(
        blocks=blocks, modal=modal, coefficients=coeffs, parameterization="s2"
    )


def simulate(op, n, seed=0, noise=1.0):
    """Drive a memory operator with white noise."""
    rng = np.random.RandomState(seed)
    w = np.zeros((n, op.r))
    w[: op.order] = rng.randn(op.order, op.r)
    for t in range(op.order - 1, n - 1):
        w[t + 1] = op.predict(w[t::-1][: op.order]) + noise * rng.randn(op.r)
    return w


# --------------------------------------------------------------------------- #
# Real modal form
# --------------------------------------------------------------------------- #
def test_real_modal_form_reconstructs_the_matrix():
    """A = P D P^-1 exactly, with D real block-diagonal (no complex leakage)."""
    op = make_s2_system(order=1)
    a0 = op.blocks[0]
    modal = M.real_modal_form(a0)
    d = modal.p_inv @ a0 @ modal.p
    assert np.allclose(modal.p @ d @ modal.p_inv, a0, atol=1e-10)
    assert np.isrealobj(modal.p)
    # Block-diagonal: no coupling between different modal blocks.
    for i, si in enumerate(modal.block_slices()):
        for j, sj in enumerate(modal.block_slices()):
            if i != j:
                assert np.allclose(d[si, sj], 0.0, atol=1e-9)


def test_conjugate_pairs_become_2x2_rotation_blocks():
    """A complex pair a +- ib gives one 2x2 block, not two 1x1s."""
    modal = M.real_modal_form(make_s2_system(order=1).blocks[0])
    assert sorted(modal.block_sizes) == [1, 1, 2]
    assert sum(modal.block_sizes) == 4


def test_s2_parameter_count_is_exactly_rL():
    """Real modes cost L each, conjugate pairs 2L for the pair -- rL in total."""
    order = 5
    op = make_s2_system(order=order)
    assert sum(op.modal.block_sizes) * order == op.r * order


# --------------------------------------------------------------------------- #
# Companion realisation
# --------------------------------------------------------------------------- #
def test_companion_reproduces_the_recursion():
    """A_cal applied to w_bar_t yields [w_{t+1}; w_t; ...] -- the shift is right."""
    op = make_s2_system(order=4)
    rng = np.random.RandomState(3)
    hist = rng.randn(op.order, op.r)
    nxt = op.companion() @ hist.reshape(-1)
    assert np.allclose(nxt[: op.r], op.predict(hist), atol=1e-12)
    # Lower blocks are the shifted history.
    assert np.allclose(nxt[op.r :], hist[:-1].reshape(-1), atol=1e-12)


def test_selector_and_injection_matrices():
    """S picks the current block; E injects noise into it only."""
    s = M.selector_matrix(4, 3)
    e = M.noise_injection_matrix(4, 3)
    assert s.shape == (3, 12) and e.shape == (12, 3)
    assert np.allclose(s @ e, np.eye(3))
    assert np.allclose(e[3:], 0.0)


def test_lemma_2_7_process_noise_reachability_is_automatic():
    """(A_cal, E) is controllable for EVERY memory kernel.

    This is what rescues the stabilizability hypothesis of v2 Thm 4.3: the lifted
    process noise E Q E^T is singular (rank r of Lr), so a naive reading suggests
    P_inf may not exist. Lemma 2.7 says the pair is reachable regardless.
    """
    rng = np.random.RandomState(7)
    for order in (1, 2, 5):
        for trial in range(3):
            blocks = rng.randn(order, 4, 4) * 0.2
            op = M.MemoryOperator(blocks=blocks)
            a_cal, e = op.companion(), M.noise_injection_matrix(order, 4)
            ctrb = np.concatenate(
                [np.linalg.matrix_power(a_cal, k) @ e for k in range(order)], axis=1
            )
            assert np.linalg.matrix_rank(ctrb, tol=1e-8) == order * 4


# --------------------------------------------------------------------------- #
# Lemma 2.6: the lifted spectrum
# --------------------------------------------------------------------------- #
def test_lemma_2_6_matrix_polynomial_matches_companion_eigenvalues():
    """The per-mode factorisation reproduces sigma(A_cal) exactly."""
    op = make_s2_system(order=4)
    fast = np.sort(np.abs(M.lifted_spectrum(op, fast=True)))
    slow = np.sort(np.abs(M.lifted_spectrum(op, fast=False)))
    assert fast.size == op.lifted_dim
    assert np.allclose(fast, slow, atol=1e-9)


def test_lifted_spectrum_is_not_sigma_of_a0():
    """rho(A_cal) differs from rho(A_0) -- clipping sigma(A_0) is the wrong knob.

    This is the concrete reason the v2 eigenvalue clipping of Algorithm 3 step 4
    cannot be carried over once L > 1.
    """
    op = make_s2_system(order=4)
    rho_lifted = M.spectral_radius_lifted(op)
    rho_a0 = float(np.max(np.abs(np.linalg.eigvals(op.blocks[0]))))
    assert not np.isclose(rho_lifted, rho_a0, atol=1e-3)


def test_prop_2_11_each_mode_splits_into_L_lifted_modes():
    """A memory kernel redistributes each Koopman mode into a cluster of L modes."""
    order = 4
    op = make_s2_system(order=order)
    lam = M.lifted_spectrum(op, fast=True)
    assert lam.size == op.r * order


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def test_enforce_stability_targets_the_exact_lifted_spectrum():
    """An unstable operator must land AT rho_max, not far below it.

    Scaling by the norm budget would satisfy stability but overshoot badly: it is
    a sufficient condition, not a tight one, and it gets worse as L grows.
    """
    rng = np.random.RandomState(11)
    for order in (1, 3, 7):
        op = M.MemoryOperator(blocks=rng.randn(order, 6, 6) * 0.9)
        assert M.spectral_radius_lifted(op) > 1.0
        scaled = M.enforce_stability(op, rho_max=1.0)
        assert M.spectral_radius_lifted(scaled) == pytest.approx(1.0, abs=1e-3)
        # Uniform scaling: relative lag weights -- the physically meaningful part
        # of the kernel -- are untouched.
        ratios = [
            np.linalg.norm(scaled.blocks[j]) / np.linalg.norm(op.blocks[j])
            for j in range(order)
        ]
        assert np.allclose(ratios, ratios[0])


def test_stable_operator_is_never_shrunk_even_when_over_budget():
    """The norm budget must not be allowed to cripple an already-stable kernel.

    Alternating-sign memory partially cancels, so sum_j ||A_j||_2 = 1.3 while the
    true rho is only sqrt(0.4) = 0.632. Scaling by the budget here would shrink a
    perfectly stable operator to rho = 0.55 for no reason -- which on the real
    record clamped the budget to 1.000 at a true rho of 0.53 and inflated every
    semigroup defect.
    """
    op = M.MemoryOperator(blocks=np.stack([np.eye(6) * 0.9, -np.eye(6) * 0.4]))
    assert op.norm_budget() > 1.0
    assert M.spectral_radius_lifted(op) == pytest.approx(np.sqrt(0.4), abs=1e-6)
    out = M.enforce_stability(op, 1.0)
    assert out is op or np.allclose(out.blocks, op.blocks)


def test_enforce_stability_is_a_noop_when_already_stable():
    op = make_s2_system(order=3)
    assert M.spectral_radius_lifted(op) <= 1.0
    assert M.enforce_stability(op, 1.0) is op


# --------------------------------------------------------------------------- #
# Identification
# --------------------------------------------------------------------------- #
def test_unstructured_identification_is_consistent():
    """The LS fit converges at the sqrt(T) rate."""
    rng = np.random.RandomState(5)
    blocks = rng.randn(3, 5, 5) * 0.2
    truth = M.enforce_stability(M.MemoryOperator(blocks=blocks), 0.95)
    errs = []
    for n in (5_000, 50_000):
        w = simulate(truth, n, seed=1)
        op, _ = M.identify_memory(
            w, 3, parameterization="unstructured", ridge_mu=1e-10
        )
        errs.append(
            np.linalg.norm(op.blocks - truth.blocks) / np.linalg.norm(truth.blocks)
        )
    assert errs[1] < errs[0] / 2.0  # 10x data -> ~3.2x better


def test_s2_identification_recovers_a_known_kernel():
    truth = make_s2_system(order=3)
    w = simulate(truth, 60_000, seed=2)
    op, _ = M.identify_memory(w, 3, parameterization="s2", ridge_mu=1e-10)
    rel = np.linalg.norm(op.blocks - truth.blocks) / np.linalg.norm(truth.blocks)
    assert rel < 0.05


def test_stage_one_uses_quiescent_transitions_only():
    """B_p is recovered even though A is fitted only on rain-free steps."""
    truth = make_s2_system(order=2)
    rng = np.random.RandomState(4)
    n = 40_000
    b_true = rng.randn(truth.r, 1) * 2.0
    forcing = np.where(rng.random_sample((n, 1)) < 0.15, rng.gamma(2.0, 1.0, (n, 1)), 0.0)
    w = np.zeros((n, truth.r))
    w[: truth.order] = rng.randn(truth.order, truth.r)
    for t in range(truth.order - 1, n - 1):
        w[t + 1] = (
            truth.predict(w[t::-1][: truth.order])
            + (b_true @ forcing[t])
            + rng.randn(truth.r)
        )
    op, b_hat = M.identify_memory(
        w, 2, forcing, parameterization="s2", ridge_mu=1e-8
    )
    assert np.linalg.norm(b_hat - b_true) / np.linalg.norm(b_true) < 0.1


def test_identify_rejects_non_finite_weights():
    w = np.ones((50, 4))
    w[10, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        M.identify_memory(w, 2)


def test_build_lifted_design_orders_history_correctly():
    """w_bar[i, j] must be w_{t-j}, newest first."""
    w = np.arange(20, dtype=float)[:, None] * np.ones((1, 2))
    w_bar, targets, origins = M.build_lifted_design(w, order=3, horizon=2)
    t0 = origins[0]
    assert t0 == 2
    assert w_bar[0, 0, 0] == t0
    assert w_bar[0, 1, 0] == t0 - 1
    assert w_bar[0, 2, 0] == t0 - 2
    assert targets[0][0, 0] == t0 + 1
    assert targets[1][0, 0] == t0 + 2


def test_build_lifted_design_skips_windows_touching_unobserved_frames():
    """A window is used only if every frame it touches was actually observed."""
    w = np.random.RandomState(0).randn(40, 3)
    valid = np.ones(40, dtype=bool)
    valid[10] = False
    _, _, origins = M.build_lifted_design(w, order=3, horizon=2, valid=valid)
    # Origins 8..12 all touch the missing frame at index 10.
    assert not set(range(8, 13)) & set(origins.tolist())
    assert 7 in origins and 13 in origins


# --------------------------------------------------------------------------- #
# Theorem 2.8: observability of the lift
# --------------------------------------------------------------------------- #
def test_theorem_2_8_over_lagging_destroys_observability():
    """Fitting past the true order makes A_{L-1} vanish -> unobservable.

    This is the hard constraint that makes memory order something to *select*
    rather than maximise, and it is the sharp practical content of Thm 2.8(i).
    The system stays DETECTABLE because the offending modes sit at lambda = 0.
    """
    truth = make_s2_system(order=3)
    w = simulate(truth, 60_000, seed=6)
    ratios = {}
    for l in (2, 3, 4, 6):
        op, _ = M.identify_memory(w, l, parameterization="s2", ridge_mu=1e-10)
        cert = M.observability_certificate(op, np.eye(truth.r))
        ratios[l] = cert["a_last_relative_smin"]
        if l <= 3:
            assert cert["observable"], f"L={l} should be observable"
        else:
            assert not cert["observable"], f"L={l} over-lags and must fail Thm 2.8(i)"
            assert cert["detectable"], "excess modes sit at lambda=0, so detectable"
    # The drop at the true order is orders of magnitude, not marginal.
    assert ratios[3] / ratios[4] > 20.0


def test_observability_requires_more_than_shadedness():
    """A C whose kernel contains a modal direction fails Thm 2.8(ii).

    v2 Def. 4.1 shadedness only asks that each basis function be activated
    somewhere; the lift additionally requires ker(C) to avoid the eigen-directions
    of the matrix polynomial.
    """
    op = make_s2_system(order=2, rotate=True)
    modal = op.modal
    # Project out one real modal direction. Because the modal basis is rotated
    # away from the coordinate axes, every column of C stays activated -- so C is
    # shaded in the sense of v2 Def. 4.1 -- yet ker(C) contains a modal direction.
    v = modal.p[:, modal.block_slices()[-1].start]
    v = v / np.linalg.norm(v)
    c_blind = np.eye(op.r) - np.outer(v, v)
    assert np.all(np.abs(c_blind).max(axis=0) > 1e-6)  # shaded: every column activated
    assert np.linalg.matrix_rank(c_blind, tol=1e-8) == op.r - 1  # yet rank-deficient
    cert_blind = M.observability_certificate(op, c_blind)
    cert_full = M.observability_certificate(op, np.eye(op.r))
    assert cert_full["min_ratio"] > cert_blind["min_ratio"]
    assert cert_blind["n_unobservable_modes"] >= 1


def test_memory_profile_reports_per_mode_relaxation():
    """Each mode gets its own memory half-life -- the physical read-out of S2."""
    op = make_s2_system(order=5)
    prof = M.memory_profile(op)
    assert prof["magnitudes"].shape == (op.modal.n_blocks, 5)
    # Coefficients decay by 0.25 per lag, so magnitude halves by lag 1.
    assert np.all(prof["half_life"] == 1.0)


def test_memory_profile_requires_per_mode_structure():
    op = M.MemoryOperator(blocks=np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="per-mode"):
        M.memory_profile(op)


def test_modal_null_directions_agree_with_the_svd_path():
    """The closed-form certificate must match the general one, not just be faster.

    Under S1/S2 all L lifted roots of a Koopman mode share one spatial direction
    (Prop. 2.11 mode-splitting), so the null vectors are the modal basis columns.
    That drops the cost from O(L r^4) -- 88 s at r=256 -- to O(r^2), but only if it
    is the same answer.
    """
    for order in (2, 3, 5):
        op = make_s2_system(order=order, rotate=True)
        assert M._modal_null_directions(op) is not None
        fast = M.observability_certificate(op, np.eye(op.r))
        general = M.observability_certificate(
            M.MemoryOperator(blocks=op.blocks), np.eye(op.r)  # no coeffs -> SVD path
        )
        assert fast["observable"] == general["observable"]
        assert fast["detectable"] == general["detectable"]
        assert fast["min_ratio"] == pytest.approx(general["min_ratio"], abs=1e-6)


def test_modal_null_directions_absent_without_per_mode_structure():
    """Unstructured / S3 kernels must fall through to the general SVD path."""
    assert M._modal_null_directions(M.MemoryOperator(blocks=np.zeros((3, 4, 4)))) is None


def test_blind_emission_still_detected_by_the_fast_path():
    """The speedup must not cost the ability to see a Thm 2.8(ii) violation."""
    op = make_s2_system(order=2, rotate=True)
    v = op.modal.p[:, op.modal.block_slices()[-1].start]
    v = v / np.linalg.norm(v)
    c_blind = np.eye(op.r) - np.outer(v, v)
    cert = M.observability_certificate(op, c_blind)
    assert cert["min_ratio"] < M.observability_certificate(op, np.eye(op.r))["min_ratio"]
    assert cert["n_unobservable_modes"] >= 1


# --------------------------------------------------------------------------- #
# The coefficients and the blocks must never describe different operators
# --------------------------------------------------------------------------- #
#
# ``lifted_spectrum`` takes a FAST path from ``coefficients`` whenever they are
# present (Prop. 2.11), and falls back to eigenvalues of the dense companion
# otherwise. So any transformation that edits ``blocks`` without editing
# ``coefficients`` splits the operator in two: the one that forecasts (blocks)
# and the one that gets certified and plotted (coefficients). Nothing raises --
# every reported spectral quantity is simply about a different matrix.
#
# Two such transformations shipped, and together they are why a blended,
# increment-parameterised operator reported rho = 1.37 while the operator
# actually in use sat at 0.98:
#
#   * ``identify_memory(increment=True)`` -- the default -- added ``I`` to
#     ``A_0`` and left the coefficients describing the increment ``G``;
#   * ``_blend_to_persistence`` blended the blocks and carried the coefficients
#     through untouched, so the rho-bisections could never move their own
#     objective.
def _fitted_s2(order=3, seed=5, increment=True):
    """An S2 operator identified from a synthetic record."""
    truth = make_s2_system(order=order)
    w = simulate(truth, 1500, seed=seed, noise=0.4)
    op, _ = M.identify_memory(
        w, order, None, parameterization="s2", ridge_mu=1e-6, increment=increment
    )
    return op


@pytest.mark.parametrize("increment", [True, False])
def test_fast_spectrum_matches_the_dense_companion(increment):
    """
    Prop. 2.11's per-mode factorisation must agree with the dense eigenvalues.

    They are two routes to the same object; a disagreement means the
    coefficients no longer describe the blocks.
    """
    op = _fitted_s2(increment=increment)
    fast = np.sort_complex(M.lifted_spectrum(op, fast=True))
    dense = np.sort_complex(np.linalg.eigvals(op.companion()))
    assert np.allclose(np.sort(np.abs(fast)), np.sort(np.abs(dense)), atol=1e-8)
    assert M.spectral_radius_lifted(op) == pytest.approx(
        float(np.max(np.abs(dense))), abs=1e-8
    )


def test_increment_reparameterisation_keeps_coefficients_and_blocks_in_step():
    """
    After ``A_0 <- A_0 + I`` the coefficients must still rebuild the blocks.

    Note the two ``increment`` settings do *not* differ by exactly ``I``: with
    ``increment=True`` the regression target is ``w_{t+1} - w_t``, so the ridge
    shrinks a different quantity and the two fits are genuinely different
    operators. What must hold is the internal invariant -- that the operator's
    two representations agree with each other.
    """
    inc = _fitted_s2(increment=True)
    rebuilt = _blocks_from_coefficients(inc.coefficients, inc.modal, inc.order)
    assert np.allclose(rebuilt, inc.blocks, atol=1e-8)


def test_adding_one_to_alpha0_is_adding_the_identity_to_a0():
    """
    The algebraic step the increment path relies on, checked directly.

    ``I = P I P^-1`` in every basis, so a real modal block's scalar gains 1 and a
    conjugate pair's realisation gains ``I_2`` -- which is why absorbing the
    ``+I`` into ``alpha_0`` is exact rather than an approximation.
    """
    op = _fitted_s2(increment=False)
    bumped = np.array(op.coefficients, copy=True)
    bumped[:, 0] = bumped[:, 0] + 1.0
    blocks = _blocks_from_coefficients(bumped, op.modal, op.order)
    assert np.allclose(blocks[0], op.blocks[0] + np.eye(op.r), atol=1e-8)
    assert np.allclose(blocks[1:], op.blocks[1:], atol=1e-8)


@pytest.mark.parametrize("s", [1.0, 0.75, 0.5, 0.25, 0.0])
def test_blend_keeps_coefficients_consistent_with_blocks(s):
    """
    A blended operator's two representations must stay the same operator.

    Without this the rho rails bisect on a value their own blending never
    changes, and the plotted Koopman spectrum shows the pre-shrinkage modes.
    """
    op = _fitted_s2()
    blended = M._blend_to_persistence(op, s)
    rebuilt = _blocks_from_coefficients(
        blended.coefficients, blended.modal, blended.order
    )
    assert np.allclose(rebuilt, blended.blocks, atol=1e-8)
    fast = float(np.max(np.abs(M.lifted_spectrum(blended, fast=True))))
    dense = float(np.max(np.abs(np.linalg.eigvals(blended.companion()))))
    assert fast == pytest.approx(dense, abs=1e-8)


def test_persistence_blend_endpoint_has_radius_exactly_one():
    """
    ``s = 0`` is persistence, whose lifted spectrum is ``{1, 0, ..., 0}``.

    This is what makes the rho rail always feasible for ``rho_max >= 1``: the
    bisection can never run out of room.
    """
    op = _fitted_s2()
    persist = M._blend_to_persistence(op, 0.0)
    assert M.is_persistence(persist)
    assert M.spectral_radius_lifted(persist) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(M.forecast_gain(persist, 6), 1.0, atol=1e-9)
