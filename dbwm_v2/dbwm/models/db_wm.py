"""
The Deep Basis World Model (DB-WM) -- the full Flax module tying together the
encoder, spatial decoder/basis, latent dynamics and variational posterior.

Architecture (image-backbone / JEPA-style world model, Section 7 + Section 8.1):

    o_t  --[ backbone g_theta ]-->  h_t  --[ expansion ]-->  w_t = phi_theta(o_t) in R^r
                                                              |
                          dynamics:  w_{t+1} = A w_t + B u_t  |
                                                              v
    field:  f_t(x) = < w_t, Psi(x) >      (Psi = spatial DBK basis on the grid)

* **Encoder** ``phi_theta`` (backbone + expansion) maps a whole frame to the
  latent weight ``w_t`` -- the world-model state the dynamics evolve.
* **Spatial basis** ``Psi`` evaluated on the fixed pixel grid gives
  ``Phi_X in R^{n x r}``; decoding a state to a map is ``Phi_X w`` (``O(nr)``),
  and ``Phi_X^T Phi_X`` is cached once (``O(nr^2)``). This is the GP side that
  yields calibrated per-pixel variance and the dPPGP objective.
* **Consistency**: a loss term ties ``phi_theta(o_t)`` to the GP-solved weights
  ``Lambda_X^{-1} Phi_X^T y_t`` so the encoder state and the GP state agree.

The module exposes granular methods (``encode``, ``spatial_features``,
``decode``, ``step``, ``predict_pixels``) which are invoked through
``model.apply(params, ..., method=...)`` from the trainer / inference code.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn

from dbwm.models.backbones import build_backbone
from dbwm.models.expansion import build_expansion
from dbwm.models.spatial_basis import SpatialBasis
from dbwm.models.variational import VariationalPosterior
from dbwm.dynamics.transition import LatentDynamics


class DBWM(nn.Module):
    """
    Full Deep Basis World Model.

    :ivar cfg: an :class:`~dbwm.config.ExperimentConfig`.
    """

    cfg: object

    def setup(self):
        """Instantiate all sub-modules and the learnable observation noise."""
        cfg = self.cfg
        self.backbone = build_backbone(cfg.backbone)
        self.expansion = build_expansion(cfg.basis)
        self.spatial = SpatialBasis(cfg.basis)
        self.variational = VariationalPosterior(cfg.basis.r)
        self.dynamics = LatentDynamics(
            r=cfg.basis.r,
            use_forcing=cfg.dynamics.use_forcing,
            input_dim=cfg.dynamics.input_dim,
        )
        # log sigma_eps^2 so the noise stays positive under unconstrained opt.
        self.log_sigma_eps2 = self.param(
            "log_sigma_eps2",
            lambda key: jnp.array(jnp.log(cfg.basis.sigma_eps2_init), dtype=jnp.float32),
        )

    # --------------------------------------------------------------------- #
    # Core maps
    # --------------------------------------------------------------------- #
    def encode(self, o, train: bool = True) -> jnp.ndarray:
        """
        Encode a batch of frames to latent weights ``w = phi_theta(o)``.

        :param o: ``(B, H, W, C)`` images.
        :param train: training flag (passed to the backbone).
        :return: ``(B, r)`` latent weights.
        """
        h = self.backbone(o, train=train)
        return self.expansion(h)

    def spatial_features(self, coords) -> jnp.ndarray:
        """
        Evaluate the spatial DBK basis ``Psi`` at pixel coordinates.

        :param coords: ``(N, 2)`` coordinates in ``[-1, 1]^2``.
        :return: ``(N, r)`` feature matrix ``Phi_X``.
        """
        return self.spatial(coords)

    def sigma_eps2(self) -> jnp.ndarray:
        """Return the (positive) observation-noise variance ``sigma_eps^2``."""
        return jnp.exp(self.log_sigma_eps2)

    def decode(self, w, coords) -> jnp.ndarray:
        """
        Decode latent weight(s) to field values at ``coords``: ``f(x)=<w, Psi(x)>``.

        :param w: ``(r,)`` or ``(B, r)`` latent weights.
        :param coords: ``(N, 2)`` query coordinates.
        :return: ``(N,)`` or ``(B, N)`` field values.
        """
        phi = self.spatial(coords)  # (N, r)
        return w @ phi.T

    def step(self, w, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advance latent state one step ``w_{t+1} = A w_t + B u_t``.

        :param w: ``(..., r)`` current weights.
        :param u: ``(..., ell)`` forcing or ``None``.
        :return: ``(..., r)`` next weights.
        """
        return self.dynamics(w, u)

    def transition(self) -> jnp.ndarray:
        """Return the transition matrix ``A`` (for losses / Koopman analysis)."""
        return self.dynamics.transition_matrix()

    def input_mat(self) -> jnp.ndarray:
        """Return the input matrix ``B`` (zeros if forcing disabled)."""
        return self.dynamics.input_matrix()

    def variational_var(self, coords) -> jnp.ndarray:
        """
        Per-pixel predictive variance ``||L^T Psi(x)||^2 + sigma_eps^2``.

        :param coords: ``(N, 2)`` query pixels.
        :return: ``(N,)`` predictive variances (state-independent).
        """
        phi = self.spatial(coords)
        _, var_q = self.variational(phi)
        return var_q + self.sigma_eps2()

    def feature_norms(self, coords) -> jnp.ndarray:
        """
        Squared spatial-feature norms ``||Psi(x)||^2`` (for the trace regulariser).

        :param coords: ``(N, 2)`` query pixels.
        :return: ``(N,)`` squared norms.
        """
        phi = self.spatial(coords)
        return jnp.sum(phi**2, axis=-1)

    def kl(self) -> jnp.ndarray:
        """Return the variational KL to the standard normal prior."""
        return self.variational.kl_to_standard_normal()

    # --------------------------------------------------------------------- #
    # GP / dPPGP predictive moments
    # --------------------------------------------------------------------- #
    def predict_pixels(
        self, w, coords
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict per-pixel field mean and (variational) variance for a state.

        Combines the deterministic state mean ``<w, Psi(x)>`` with the dPPGP
        predictive variance ``||L^T Psi(x)||^2 + sigma_eps^2`` (Section 3.2).

        :param w: ``(r,)`` latent weight (single frame state).
        :param coords: ``(N, 2)`` query pixels.
        :return: ``(mean, var)`` each ``(N,)``.
        """
        phi = self.spatial(coords)  # (N, r)
        mean = phi @ w  # (N,)
        _, var_q = self.variational(phi)  # (N,)
        return mean, var_q + self.sigma_eps2()

    def __call__(self, o, coords, u: Optional[jnp.ndarray] = None, train: bool = True):
        """
        Full forward pass exercising **every** sub-module, so a single
        ``model.init`` call materialises the complete parameter tree (encoder,
        spatial basis, variational posterior, dynamics, noise). The trainer and
        inference code then invoke the granular methods via ``method=``.

        :param o: ``(B, H, W, C)`` images.
        :param coords: ``(N, 2)`` query pixels.
        :param u: ``(B, ell)`` forcing or ``None``.
        :param train: training flag.
        :return: dict with ``w``, ``recon``, ``w_next``, ``pix_mean``,
                 ``pix_var``, ``kl``.
        """
        w = self.encode(o, train=train)  # (B, r)
        recon = self.decode(w, coords)  # (B, N)
        w_next = self.step(w, u)  # (B, r) -- touches dynamics A (and B)
        pix_mean, pix_var = self.predict_pixels(w[0], coords)  # touches variational
        kl = self.variational.kl_to_standard_normal()
        return {
            "w": w,
            "recon": recon,
            "w_next": w_next,
            "pix_mean": pix_mean,
            "pix_var": pix_var,
            "kl": kl,
        }
