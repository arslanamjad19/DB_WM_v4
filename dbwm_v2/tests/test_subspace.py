"""
Tests for the reduced-order latent subspace.

The subspace exists to resolve a conflict: ``r`` must be large to REPRESENT the
field and small to identify well-conditioned DYNAMICS. v2 Prop. 4.4 licenses the
second reduction -- the directions removed are ``ker W_-^T``, on which the
least-squares operator already had zero eigenvalues.
"""
import numpy as np
import pytest

from dbwm.dynamics.conditioning import transient_amplification, weight_rank
from dbwm.dynamics.memory import identify_memory
from dbwm.dynamics.subspace import LatentSubspace, build_subspace


def low_rank_trajectory(T=800, r=64, k_true=5, seed=0):
    """A trajectory embedded in r dims but excited in only k_true of them."""
    rng = np.random.RandomState(seed)
    z = np.zeros((T, k_true))
    for t in range(2, T):
        z[t] = 1.3 * z[t - 1] - 0.45 * z[t - 2] + 0.1 * rng.randn(k_true)
    return z @ rng.randn(k_true, r) + 0.001 * rng.randn(T, r)


def test_subspace_recovers_the_true_dimension():
    w = low_rank_trajectory(k_true=5)
    sub = build_subspace(w, energy=0.995, min_k=1)
    assert 4 <= sub.k <= 8          # the true 5, with a little slack
    assert sub.explained >= 0.995


def test_projection_round_trip_is_near_lossless_on_excited_directions():
    w = low_rank_trajectory(k_true=6)
    sub = build_subspace(w, energy=0.999, min_k=1)
    assert sub.reconstruction_error(w) < 0.05
    z = sub.project(w)
    assert z.shape == (w.shape[0], sub.k)
    assert sub.reconstruct(z).shape == w.shape


def test_basis_is_orthonormal():
    sub = build_subspace(low_rank_trajectory(), energy=0.99, min_k=1)
    assert np.allclose(sub.basis.T @ sub.basis, np.eye(sub.k), atol=1e-10)


def test_subspace_is_fitted_on_training_rows_only():
    """A subspace fitted on everything would leak test-period structure."""
    w = low_rank_trajectory(T=600)
    train = np.zeros(600, dtype=bool)
    train[:400] = True
    sub = build_subspace(w, train, energy=0.99, min_k=1)
    assert np.allclose(sub.mean, w[:400].mean(axis=0))


def test_projection_fixes_the_conditioning_of_A0():
    """The whole point: r >> excited rank gives a singular A_0; the subspace does not.

    A near-singular A_0 is the head of the chain that ends in a non-normal operator
    whose one-step forecast is worse than climatology.
    """
    w = low_rank_trajectory(T=900, r=64, k_true=5)
    full, _ = identify_memory(w, 1, parameterization="unstructured", ridge_mu=1e-3)
    sv_full = np.linalg.svd(full.blocks[0], compute_uv=False)
    cond_full = sv_full[0] / max(sv_full[-1], 1e-30)

    sub = build_subspace(w, energy=0.995, min_k=1)
    red, _ = identify_memory(
        sub.project(w), 1, parameterization="unstructured", ridge_mu=1e-3
    )
    sv_red = np.linalg.svd(red.blocks[0], compute_uv=False)
    cond_red = sv_red[0] / max(sv_red[-1], 1e-30)

    assert cond_full > 1e3
    assert cond_red < cond_full / 100.0


def test_projection_reduces_non_normal_transient_growth():
    """rho <= 1 does not bound the powers; the subspace is what actually tames them."""
    w = low_rank_trajectory(T=900, r=64, k_true=5)
    full, _ = identify_memory(w, 3, parameterization="s2", ridge_mu=1e-3)
    sub = build_subspace(w, energy=0.995, min_k=1)
    red, _ = identify_memory(sub.project(w), 3, parameterization="s2", ridge_mu=1e-3)
    assert (
        transient_amplification(red.companion())["non_normality"]
        < transient_amplification(full.companion())["non_normality"]
    )


def test_lift_operator_preserves_the_nonzero_spectrum():
    """v2 Prop. 4.4's equivalence: the discarded directions carried zero eigenvalues."""
    rng = np.random.RandomState(3)
    sub = build_subspace(low_rank_trajectory(r=40, k_true=6), energy=0.999, min_k=1)
    a_k = rng.randn(sub.k, sub.k) * 0.3
    lifted = sub.lift_operator(a_k)
    ev_k = np.sort_complex(np.linalg.eigvals(a_k))
    ev_r = np.linalg.eigvals(lifted)
    top = np.sort_complex(ev_r[np.argsort(-np.abs(ev_r))[: sub.k]])
    assert np.allclose(np.sort(np.abs(ev_k)), np.sort(np.abs(top)), atol=1e-8)
    assert np.allclose(np.sort(np.abs(ev_r))[: sub.r - sub.k], 0.0, atol=1e-8)


def test_covariance_projection_and_lift_are_consistent():
    sub = build_subspace(low_rank_trajectory(r=32, k_true=4), energy=0.99, min_k=1)
    rng = np.random.RandomState(4)
    a = rng.randn(sub.k, sub.k)
    cov_k = a @ a.T
    assert np.allclose(sub.project_covariance(sub.lift_covariance(cov_k)), cov_k, atol=1e-9)
    stack = np.stack([cov_k, cov_k * 2])
    lifted = np.stack([sub.lift_covariance(c) for c in stack])
    assert np.allclose(sub.project_covariance(lifted), stack, atol=1e-9)


def test_rank_report_flags_over_parameterisation():
    w = low_rank_trajectory(T=800, r=128, k_true=5)
    rep = weight_rank(w)
    assert rep.rank_99 < 15
    assert rep.over_parameterised
    assert rep.suggested_r() < 64


def test_rank_report_accepts_a_well_sized_r():
    rng = np.random.RandomState(5)
    w = rng.randn(500, 8)          # full-rank: every direction excited
    rep = weight_rank(w)
    assert not rep.over_parameterised


def test_complement_is_orthogonal_to_the_subspace():
    """
    ``complement(w)`` must be exactly what the subspace cannot represent.

    :return: ``None``
    """
    import numpy as np
    from dbwm.dynamics.subspace import build_subspace

    rng = np.random.default_rng(0)
    w = rng.normal(size=(80, 12))
    sub = build_subspace(w, None, energy=0.8, min_k=2)
    comp = sub.complement(w)
    assert np.allclose(comp @ sub.basis, 0.0, atol=1e-8)


def test_reconstruct_with_complement_is_exact_at_the_origin():
    """
    Lifting the origin's own projection must return the origin exactly.

    This is what makes a persistence operator reproduce *pixel-space*
    persistence rather than persistence-plus-truncation: with ``Theta = I`` the
    forecast is ``reconstruct(project(w_t)) + complement(w_t) = w_t``.
    """
    import numpy as np
    from dbwm.dynamics.subspace import build_subspace

    rng = np.random.default_rng(1)
    w = rng.normal(size=(60, 10))
    sub = build_subspace(w, None, energy=0.7, min_k=2)
    assert sub.k < sub.r  # the truncation must be real for the test to mean anything
    for i in (0, 5, 17):
        got = sub.reconstruct_with_complement(sub.project(w[i]), w[i])
        assert np.allclose(got, w[i], atol=1e-10)


def test_plain_reconstruct_loses_what_the_complement_carries():
    """
    The two differ by exactly the truncation error -- the term this removes.

    :return: ``None``
    """
    import numpy as np
    from dbwm.dynamics.subspace import build_subspace

    rng = np.random.default_rng(2)
    w = rng.normal(size=(60, 10))
    sub = build_subspace(w, None, energy=0.7, min_k=2)
    z = sub.project(w[3])
    plain = sub.reconstruct(z)
    carried = sub.reconstruct_with_complement(z, w[3])
    assert np.linalg.norm(carried - w[3]) < 1e-10
    assert np.linalg.norm(plain - w[3]) > 1e-6
