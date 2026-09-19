"""
Expansion layers (Stage 2 of the deep basis map): expand : R^h -> R^r.

These lift an ``h``-dimensional backbone feature to the ``r`` basis functions
phi_theta(o) in R^r whose inner product defines the Deep Basis Kernel
``k(o, o') = <phi(o), phi(o')>`` (Definition 2.2).

Variants
--------
* :class:`SwiGLUExpansion` -- the primary DB-WM variant. Uses the *correct*
  GLU-variant SwiGLU formula (Shazeer 2020, "GLU Variants Improve
  Transformer"), which the framework's Section 2.4 wrote in an abbreviated
  single-projection form. The correct form needs TWO linear projections::

      SwiGLU(x) = Swish_beta(x W1 + b1)  (elementwise *)  (x W2 + b2)
      Swish_beta(z) = z * sigmoid(beta * z)        # beta = 1 -> SiLU
      phi(o) = diag(s) . SwiGLU(g_theta(o))

  ``s in R^r`` is a learnable per-basis scale. SwiGLU is nonzero a.e., giving
  automatic shadedness (Proposition 4.1).

* :class:`GELUExpansion` -- a single-projection GELU variant (provided as an
  additional ablation / sanity baseline).

* :class:`RBFExpansion` -- the sparse-DKL variant (DB-WM-RBF). Learnable
  inducing points ``Z = {z_i} in R^h`` and an RBF base kernel give
  ``phi(o) = K_ZZ^{-1/2} k_Z(g_theta(o))`` (Section 2.4), i.e. the whitened
  RBF features. This realises the inducing-point deep kernel and is left
  mathematically unchanged from the framework, per the user's instruction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


def swish(z, beta: float = 1.0):
    """
    Swish / SiLU activation ``z * sigmoid(beta * z)``.

    :param z: input array.
    :param beta: gating temperature (``beta = 1`` recovers SiLU).
    :return: activated array.
    """
    return z * jax.nn.sigmoid(beta * z)


class SwiGLUExpansion(nn.Module):
    """
    Corrected SwiGLU expansion ``phi(o) = diag(s) . SwiGLU(W1 g + b1, W2 g + b2)``.

    :ivar r: number of output basis functions.
    :ivar beta: Swish gating temperature.
    """

    r: int
    beta: float = 1.0

    @nn.compact
    def __call__(self, g):
        """
        Expand backbone features to ``r`` basis activations.

        :param g: ``(B, h)`` backbone features.
        :return: ``(B, r)`` basis features phi(o).
        """
        gate = nn.Dense(self.r, name="W_gate")(g)  # x W1 + b1
        value = nn.Dense(self.r, name="W_value")(g)  # x W2 + b2
        s = self.param("scale", nn.initializers.ones, (self.r,))
        return s * (swish(gate, self.beta) * value)


class GELUExpansion(nn.Module):
    """
    Single-projection GELU expansion ``phi(o) = diag(s) . GELU(W g + b)``.

    :ivar r: number of output basis functions.
    """

    r: int

    @nn.compact
    def __call__(self, g):
        """
        Expand backbone features to ``r`` basis activations.

        :param g: ``(B, h)`` backbone features.
        :return: ``(B, r)`` basis features phi(o).
        """
        z = nn.Dense(self.r, name="W_expand")(g)
        s = self.param("scale", nn.initializers.ones, (self.r,))
        return s * nn.gelu(z)


class RBFExpansion(nn.Module):
    """
    Sparse-DKL RBF expansion ``phi(o) = K_ZZ^{-1/2} k_Z(g_theta(o))``.

    Learnable inducing points ``Z in R^{r x h}`` live in the backbone-feature
    space. ``k_Z(g)_i = exp(-||g - z_i||^2 / (2 l^2))`` and the whitening factor
    ``K_ZZ^{-1/2}`` (computed from an eigendecomposition of the r x r inducing
    Gram matrix) decorrelates the features so the kernel is well-conditioned.

    :ivar r: number of inducing points (= output dimension).
    :ivar lengthscale: RBF base-kernel lengthscale ``l``.
    """

    r: int
    lengthscale: float = 1.0

    @nn.compact
    def __call__(self, g):
        """
        Expand backbone features to ``r`` whitened RBF basis activations.

        :param g: ``(B, h)`` backbone features.
        :return: ``(B, r)`` basis features phi(o).
        """
        h = g.shape[-1]
        z = self.param("inducing", nn.initializers.normal(1.0), (self.r, h))
        log_ell = self.param(
            "log_lengthscale",
            lambda key, shape: jnp.full(shape, jnp.log(self.lengthscale)),
            (),
        )
        # Dimension-aware bandwidth: in an h-dim feature space squared distances
        # grow ~O(h), so without the 1/h scaling exp(-||g-z||^2 / 2l^2) saturates
        # to ~0 and the RBF features collapse to an input-independent constant.
        ell2 = jnp.exp(2.0 * log_ell) * h

        def rbf(a, b):
            # a: (B, h), b: (M, h) -> (B, M)
            sq = (
                jnp.sum(a**2, axis=-1, keepdims=True)
                - 2.0 * a @ b.T
                + jnp.sum(b**2, axis=-1)[None, :]
            )
            return jnp.exp(-0.5 * sq / ell2)

        k_zz = rbf(z, z) + 1e-3 * jnp.eye(self.r)  # (r, r), jittered for stability
        k_z = rbf(g, z)  # (B, r)

        # Whitening: phi(o) = L^{-1} k_z, where K_ZZ = L L^T (Cholesky). This is
        # a valid whitening factor (L^{-T} L^{-1} = K_ZZ^{-1}) giving the sparse
        # Nystrom kernel <phi(o), phi(o')> = k_z^T K_ZZ^{-1} k_z'. Cholesky +
        # triangular solve is numerically stable and differentiable, unlike the
        # eigendecomposition-based K_ZZ^{-1/2} whose gradients blow up for
        # near-degenerate / tiny eigenvalues.
        chol = jnp.linalg.cholesky(k_zz)  # (r, r) lower
        phi = jax.scipy.linalg.solve_triangular(chol, k_z.T, lower=True)  # (r, B)
        return phi.T


def build_expansion(cfg) -> nn.Module:
    """
    Construct an expansion module from a :class:`~dbwm.config.BasisConfig`.

    :param cfg: basis configuration.
    :return: an un-initialised Flax expansion module.
    """
    if cfg.expansion == "swiglu":
        return SwiGLUExpansion(r=cfg.r)
    elif cfg.expansion == "gelu":
        return GELUExpansion(r=cfg.r)
    elif cfg.expansion == "rbf":
        return RBFExpansion(r=cfg.r, lengthscale=cfg.rbf_lengthscale)
    raise ValueError("Unknown expansion: {}".format(cfg.expansion))
