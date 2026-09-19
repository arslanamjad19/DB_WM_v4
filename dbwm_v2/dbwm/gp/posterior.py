"""
Scalable Gaussian-process posterior via the low-rank Deep Basis Kernel.

Everything here exploits ``K_XX = Phi_X Phi_X^T`` (rank <= r) through the
Woodbury identity so the cost is ``O(n r^2)`` time / ``O(n r)`` space rather than
``O(n^3)`` (Section 3.1, Proposition 3.1).

Key quantities (with ``Phi_X in R^{n x r}``, targets ``y in R^n``,
``sigma_eps2`` the observation noise):

    Lambda_X = Phi_X^T Phi_X + sigma_eps2 I_r            in R^{r x r}
    w_hat    = Lambda_X^{-1} Phi_X^T y                   (GP-solved weights)
    mu_f(x*) = phi(x*)^T Lambda_X^{-1} Phi_X^T y         (posterior mean)
    var_f(x*)= sigma_eps2 * phi(x*)^T Lambda_X^{-1} phi(x*)   (posterior var)

The Gram block ``G = Phi_X^T Phi_X`` and the projection ``Phi_X^T y`` are
accumulated in *pixel batches* so the full ``n x r`` matrix is never
materialised -- the property that lets ``n`` reach 10^5-10^6 pixels.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def gram_matrix(phi_x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the r x r Gram block ``Phi_X^T Phi_X``.

    :param phi_x: ``(n, r)`` feature matrix.
    :return: ``(r, r)`` Gram matrix. Cost ``O(n r^2)``.
    """
    return phi_x.T @ phi_x


def accumulate_gram(
    phi_x_batches, n_basis: int
) -> Tuple[jnp.ndarray, int]:
    """
    Accumulate ``Phi_X^T Phi_X`` over an iterable of pixel-batch feature
    matrices without ever holding the full ``(n, r)`` matrix in memory.

    :param phi_x_batches: iterable yielding ``(b_i, r)`` feature blocks.
    :param n_basis: number of basis functions ``r``.
    :return: ``(G, n)`` where ``G`` is the ``(r, r)`` Gram sum and ``n`` the
             total number of pixels accumulated.
    """
    g = jnp.zeros((n_basis, n_basis))
    n = 0
    for block in phi_x_batches:
        g = g + block.T @ block
        n += block.shape[0]
    return g, n


def lambda_matrix(gram: jnp.ndarray, sigma_eps2: float) -> jnp.ndarray:
    """
    Form ``Lambda_X = Phi_X^T Phi_X + sigma_eps2 I_r``.

    :param gram: ``(r, r)`` Gram matrix.
    :param sigma_eps2: observation-noise variance.
    :return: ``(r, r)`` regularised Gram matrix.
    """
    r = gram.shape[0]
    return gram + sigma_eps2 * jnp.eye(r)


def solve_weights(
    phi_x: jnp.ndarray, y: jnp.ndarray, sigma_eps2: float
) -> jnp.ndarray:
    """
    GP-solve the basis weights ``w_hat = Lambda_X^{-1} Phi_X^T y``.

    This is the encoding ``y -> w`` used to derive ground-truth latent states for
    system identification (Algorithm 3 input) and for the consistency loss.

    :param phi_x: ``(n, r)`` feature matrix.
    :param y: ``(n,)`` or ``(n, 1)`` targets.
    :param sigma_eps2: observation-noise variance.
    :return: ``(r,)`` solved weight vector. Cost ``O(n r^2 + r^3)``.
    """
    y = y.reshape(-1)
    lam = lambda_matrix(gram_matrix(phi_x), sigma_eps2)
    rhs = phi_x.T @ y  # (r,)
    return jnp.linalg.solve(lam, rhs)


def posterior_mean_var(
    phi_star: jnp.ndarray,
    lam: jnp.ndarray,
    phi_x_t_y: jnp.ndarray,
    sigma_eps2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Exact DBK GP posterior mean and variance at query features ``phi_star``.

    :param phi_star: ``(m, r)`` query feature matrix (e.g. test pixels).
    :param lam: ``(r, r)`` precomputed ``Lambda_X``.
    :param phi_x_t_y: ``(r,)`` precomputed projection ``Phi_X^T y``.
    :param sigma_eps2: observation-noise variance.
    :return: ``(mean, var)`` each ``(m,)``. Cost ``O(m r^2)``.
    """
    lam_inv_proj = jnp.linalg.solve(lam, phi_x_t_y)  # (r,)
    mean = phi_star @ lam_inv_proj  # (m,)
    lam_inv_phi = jnp.linalg.solve(lam, phi_star.T)  # (r, m)
    var = sigma_eps2 * jnp.sum(phi_star * lam_inv_phi.T, axis=-1)  # (m,)
    return mean, var


def log_marginal_likelihood(
    phi_x: jnp.ndarray, y: jnp.ndarray, sigma_eps2: float
) -> jnp.ndarray:
    """
    Low-rank GP log marginal likelihood via Woodbury + matrix-determinant lemma.

    ``log N(y; 0, Phi Phi^T + sigma^2 I)`` computed in ``O(n r^2 + r^3)`` using

        log|Sigma| = (n - r) log sigma^2 + log|Lambda_X|
        y^T Sigma^{-1} y = (y^T y - (Phi^T y)^T Lambda_X^{-1} (Phi^T y)) / sigma^2

    Provided for completeness / the E-GP comparison; the DB-WM trains with the
    dPPGP objective (Remark 3.1) rather than this MML.

    :param phi_x: ``(n, r)`` feature matrix.
    :param y: ``(n,)`` targets.
    :param sigma_eps2: observation-noise variance.
    :return: scalar log marginal likelihood.
    """
    y = y.reshape(-1)
    n, r = phi_x.shape
    lam = lambda_matrix(gram_matrix(phi_x), sigma_eps2)
    proj = phi_x.T @ y  # (r,)
    chol = jnp.linalg.cholesky(lam)
    logdet_lam = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    logdet = (n - r) * jnp.log(sigma_eps2) + logdet_lam
    lam_inv_proj = jax.scipy.linalg.cho_solve((chol, True), proj)
    quad = (y @ y - proj @ lam_inv_proj) / sigma_eps2
    return -0.5 * (quad + logdet + n * jnp.log(2.0 * jnp.pi))
