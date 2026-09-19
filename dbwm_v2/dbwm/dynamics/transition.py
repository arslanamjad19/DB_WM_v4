"""
Latent linear dynamics (Section 2.2): w_{t+1} = A w_t + B u_t^raw + eta_t.

:class:`LatentDynamics` is a Flax module holding the differentiable transition
operator ``A in R^{r x r}`` and the input matrix ``B = [B_p | B_u] in R^{r x ell}``.
For the pure-temporal LST/NDVI application ``use_forcing=False`` so ``B`` is absent
and ``w_{t+1} = A w_t`` (the "Set B_p = B_u = 0" special case of Section 8.1).

``A`` is initialised to identity-plus-small-perturbation (Algorithm 1), which is
the right inductive bias for a near-conservative thermal field (rho ~ 1).
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import flax.linen as nn


class LatentDynamics(nn.Module):
    """
    Differentiable input-affine dynamics in the deep-basis weight space.

    :ivar r: latent dimension.
    :ivar use_forcing: whether to include the input matrix ``B``.
    :ivar input_dim: forcing dimension ``ell`` (= 2 for ``[p_t, u_t]``).
    """

    r: int
    use_forcing: bool = False
    input_dim: int = 2

    def setup(self):
        """Create ``A`` (init: I + small noise) and, if used, ``B`` (zero-init)."""
        self.a = self.param(
            "A",
            lambda key, shape: jnp.eye(shape[0])
            + 1e-3 * nn.initializers.normal(1.0)(key, shape),
            (self.r, self.r),
        )
        if self.use_forcing:
            self.b = self.param(
                "B", nn.initializers.zeros, (self.r, self.input_dim)
            )

    def __call__(self, w: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advance the latent state one step.

        :param w: ``(..., r)`` current weights.
        :param u: ``(..., ell)`` raw forcing ``[p_t, u_t]`` or ``None``.
        :return: ``(..., r)`` predicted next weights.
        """
        w_next = w @ self.a.T  # (..., r)
        if self.use_forcing and u is not None:
            w_next = w_next + u @ self.b.T
        return w_next

    def transition_matrix(self) -> jnp.ndarray:
        """Return ``A`` (for spectral regularisation / Koopman analysis)."""
        return self.a

    def input_matrix(self) -> jnp.ndarray:
        """Return ``B`` (zeros if forcing disabled)."""
        if self.use_forcing:
            return self.b
        return jnp.zeros((self.r, self.input_dim))


def spectral_radius(a: jnp.ndarray) -> jnp.ndarray:
    """
    Spectral radius ``rho(A) = max_i |lambda_i|`` of a square matrix.

    :param a: ``(r, r)`` matrix.
    :return: scalar spectral radius.
    """
    return jnp.max(jnp.abs(jnp.linalg.eigvals(a)))


def spectral_norm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Spectral norm ``||A||_2`` (largest singular value) used in Theorem 4.2.

    :param a: ``(r, r)`` matrix.
    :return: scalar spectral norm.
    """
    return jnp.linalg.norm(a, ord=2)


def clip_spectral_radius(a: jnp.ndarray, rho_max: float) -> jnp.ndarray:
    """
    Clip the eigenvalues of ``A`` so its spectral radius is ``<= rho_max``.

    Implements the closed-form spectral safety of Algorithm 3 step 4 /
    v3 Section 5.2: ``A = P diag(clip(lambda)) P^{-1}``. Uses a complex
    eigendecomposition and returns the real part (``A`` is real).

    :param a: ``(r, r)`` real matrix.
    :param rho_max: maximum allowed eigenvalue magnitude.
    :return: ``(r, r)`` matrix with spectral radius ``<= rho_max``.
    """
    evals, evecs = jnp.linalg.eig(a)
    mag = jnp.abs(evals)
    scale = jnp.where(mag > rho_max, rho_max / (mag + 1e-12), 1.0)
    evals_clipped = evals * scale
    a_clipped = evecs @ jnp.diag(evals_clipped) @ jnp.linalg.inv(evecs)
    return jnp.real(a_clipped)
