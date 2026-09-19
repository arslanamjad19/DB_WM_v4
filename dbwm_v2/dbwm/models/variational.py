"""
Variational posterior q(w) = N(w; m, L L^T) for the dPPGP objective.

Implements the decoupled Parametric Predictive GP predictive moments
(Section 3.2):

    mu_f(o)   = <m, phi(o)>
    var_f(o)  = || L^T phi(o) ||^2

together with the KL divergence ``D_KL( N(m, LL^T) || N(0, I_r) )`` used as the
anti-collapse regulariser (Remark 3.1). ``L`` is parameterised as a lower-
triangular factor so ``LL^T`` is always PSD.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn


class VariationalPosterior(nn.Module):
    """
    Gaussian variational posterior over the basis weights ``w in R^r``.

    Per Algorithm 1 the parameters are initialised ``m = 0`` and
    ``L = (1/sqrt(r)) I_r``.

    :ivar r: latent dimension.
    """

    r: int

    def setup(self):
        """Create the variational parameters ``m`` and the raw factor ``L_raw``."""
        self.m = self.param("m", nn.initializers.zeros, (self.r,))
        self.l_raw = self.param(
            "L_raw",
            lambda key, shape: (1.0 / jnp.sqrt(self.r)) * jnp.eye(shape[0]),
            (self.r, self.r),
        )

    def _factor(self) -> jnp.ndarray:
        """Return the lower-triangular Cholesky-like factor ``L``."""
        return jnp.tril(self.l_raw)

    def __call__(self, phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return predictive mean and variance for feature(s) ``phi``.

        :param phi: ``(..., r)`` feature vector(s) (pixel or frame features).
        :return: ``(mu, var)`` predictive moments, each ``(...,)``.
        """
        l_tril = self._factor()
        mu = phi @ self.m  # (...,)
        # var = || L^T phi ||^2 = sum_k (phi . L[:, k])^2
        lt_phi = phi @ l_tril  # (..., r): row j = phi_j^T L
        var = jnp.sum(lt_phi**2, axis=-1)
        return mu, var

    def kl_to_standard_normal(self) -> jnp.ndarray:
        """
        Compute ``D_KL( N(m, LL^T) || N(0, I_r) )``.

        ``= 0.5 (tr(LL^T) + m^T m - r - log|LL^T|)``.

        :return: scalar KL divergence.
        """
        l_tril = self._factor()
        cov_trace = jnp.sum(l_tril**2)
        # log|LL^T| = 2 sum log|diag(L)| (L lower-triangular).
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(l_tril)) + 1e-12))
        return 0.5 * (cov_trace + self.m @ self.m - self.r - logdet)

    def get_factor(self) -> jnp.ndarray:
        """Return the lower-triangular factor ``L`` (for diagnostics)."""
        return self._factor()
