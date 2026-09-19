"""
Spatial Deep Basis Kernel  Psi_xi : R^2 -> R^r.

This is the GP / decoder side of the DB-WM. While the image backbone encodes a
whole frame ``o_t`` into the latent weight ``w_t in R^r``, the spatial basis maps
a *pixel coordinate* ``x = (lon, lat)`` to its feature vector ``Psi(x) in R^r`` so
that the reconstructed field is the deep-basis GP mean

    f_t(x) = <w_t, Psi(x)>            (Section 2.2)

and the kernel between two locations is ``k(x, x') = <Psi(x), Psi(x')>``.

Because the LST/NDVI grid is *fixed across all dates*, ``Psi`` is evaluated once
on the H*W pixel grid to give ``Phi_X in R^{n x r}``; the Gram block
``Phi_X^T Phi_X`` (r x r) is then cached and reused for every time step -- this is
what makes encoding/decoding ``O(nr)`` and identification ``O(nr^2)`` rather than
``O(n^3)`` (Section 7 of v3).

Coordinates are lifted by a random Fourier positional encoding (so the MLP can
represent high-frequency spatial structure), then an MLP, then the *same*
expansion family (SwiGLU / RBF / GELU) chosen for the model so the kernel variant
is consistent across encoder and decoder.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from dbwm.models.expansion import build_expansion


class FourierFeatures(nn.Module):
    """
    Random Fourier positional encoding for 2-D coordinates.

    ``gamma(x) = [sin(2 pi B x), cos(2 pi B x)]`` with a fixed random ``B`` drawn
    once and stored as a (non-trained) parameter so encode/decode are consistent
    across calls and checkpoints.

    :ivar n_features: number of Fourier features (output dim is ``2 * n_features``).
    :ivar scale: standard deviation of the random frequencies ``B``.
    """

    n_features: int = 128
    scale: float = 10.0

    @nn.compact
    def __call__(self, x):
        """
        Encode coordinates.

        :param x: ``(N, 2)`` coordinate array in ``[-1, 1]^2``.
        :return: ``(N, 2 * n_features)`` Fourier features.
        """
        b = self.param(
            "B",
            lambda key, shape: self.scale * jax.random.normal(key, shape),
            (x.shape[-1], self.n_features),
        )
        proj = 2.0 * jnp.pi * x @ b
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


class SpatialBasis(nn.Module):
    """
    Deep spatial basis ``Psi_xi : R^2 -> R^r`` defining the DBK over locations.

    :ivar cfg: a :class:`~dbwm.config.BasisConfig`.
    """

    cfg: object

    @nn.compact
    def __call__(self, coords):
        """
        Evaluate the spatial basis at pixel coordinates.

        :param coords: ``(N, 2)`` coordinates in ``[-1, 1]^2``.
        :return: ``(N, r)`` basis features Phi_X (one row per pixel).
        """
        cfg = self.cfg
        h = FourierFeatures(cfg.fourier_features, cfg.fourier_scale)(coords)
        for i in range(cfg.spatial_n_layers):
            h = nn.Dense(cfg.spatial_hidden_dim, name=f"spatial_dense_{i}")(h)
            h = nn.gelu(h)
        # Reuse the chosen expansion family so the kernel variant is consistent.
        return build_expansion(cfg)(h)
