"""
Image backbones g_theta : R^{H x W x C} -> R^h (Stage 1 of the deep basis map).

Two ablations, both implemented natively in Flax so the whole DB-WM is a single
JAX computation graph (end-to-end gradients, jit/vmap friendly, no PyTorch in the
training loop):

* :class:`ResNetBackbone`  -- a ResNet-18 (the convolutional ablation).
* :class:`DeiTBackbone`    -- a DeiT-Tiny style Vision Transformer
  (https://huggingface.co/docs/transformers/en/model_doc/deit). DeiT *is*
  architecturally a ViT trained with a distillation token; we include the
  optional distillation token and average the [CLS] + [DIST] tokens for the
  pooled feature. Pretrained HF weights can be ported offline (see README);
  the default is from-scratch training, which keeps the stack pure-JAX.

Both return an ``h``-dimensional pooled feature vector per image. Input images
are channels-last ``(B, H, W, C)`` (the Flax convention).
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn


def _num_groups(channels: int, target: int = 32) -> int:
    """
    Pick a GroupNorm group count that divides ``channels``.

    GroupNorm (rather than BatchNorm) is used throughout the ResNet so the model
    carries *no* running statistics -- there is no ``batch_stats`` mutable
    collection to thread through the jit'd training step, which keeps the whole
    DB-WM a single pure-functional graph.

    :param channels: number of feature channels.
    :param target: preferred maximum number of groups.
    :return: a divisor of ``channels`` no greater than ``target``.
    """
    g = min(target, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


# --------------------------------------------------------------------------- #
# ResNet-18
# --------------------------------------------------------------------------- #
class ResNetBlock(nn.Module):
    """A basic (non-bottleneck) ResNet residual block with two 3x3 convs."""

    filters: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Apply the residual block.

        :param x: ``(B, H, W, C)`` input feature map.
        :param train: unused (GroupNorm has no train/eval distinction); kept for
                      a uniform backbone call signature.
        :return: ``(B, H', W', filters)`` output feature map.
        """
        residual = x
        y = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False)(x)
        y = nn.GroupNorm(num_groups=_num_groups(self.filters))(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), (1, 1), padding="SAME", use_bias=False)(y)
        y = nn.GroupNorm(
            num_groups=_num_groups(self.filters), scale_init=nn.initializers.zeros
        )(y)

        if residual.shape != y.shape:
            residual = nn.Conv(
                self.filters, (1, 1), self.strides, padding="SAME", use_bias=False
            )(residual)
            residual = nn.GroupNorm(num_groups=_num_groups(self.filters))(residual)
        return nn.relu(residual + y)


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone producing an ``h``-dimensional pooled feature.

    :ivar hidden_dim: output feature dimension ``h``.
    :ivar stage_sizes: number of blocks per stage (ResNet-18: ``(2,2,2,2)``).
    :ivar width: base channel width (64 for standard ResNet-18).
    """

    hidden_dim: int = 256
    stage_sizes: Tuple[int, ...] = (2, 2, 2, 2)
    width: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Encode a batch of images.

        :param x: ``(B, H, W, C)`` images.
        :param train: training flag for batch-norm.
        :return: ``(B, hidden_dim)`` pooled features.
        """
        y = nn.Conv(self.width, (7, 7), (2, 2), padding="SAME", use_bias=False)(x)
        y = nn.GroupNorm(num_groups=_num_groups(self.width))(y)
        y = nn.relu(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding="SAME")

        for stage, n_blocks in enumerate(self.stage_sizes):
            filters = self.width * (2**stage)
            for block in range(n_blocks):
                strides = (2, 2) if (block == 0 and stage > 0) else (1, 1)
                y = ResNetBlock(filters, strides)(y, train=train)

        y = jnp.mean(y, axis=(1, 2))  # global average pool -> (B, filters)
        y = nn.Dense(self.hidden_dim)(y)
        return y


# --------------------------------------------------------------------------- #
# DeiT / ViT
# --------------------------------------------------------------------------- #
class TransformerEncoderBlock(nn.Module):
    """A pre-norm Transformer encoder block (MHSA + MLP)."""

    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Apply one Transformer block.

        :param x: ``(B, N, D)`` token sequence.
        :param train: training flag (unused; no dropout by default).
        :return: ``(B, N, D)`` updated tokens.
        """
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        hidden = int(self.embed_dim * self.mlp_ratio)
        h = nn.Dense(hidden)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return x + h


class DeiTBackbone(nn.Module):
    """
    DeiT-Tiny style Vision Transformer backbone.

    Patchify -> linear embed -> prepend [CLS] (+ optional [DIST]) tokens ->
    add learnable positional embeddings -> Transformer encoder -> pool the
    classification (and distillation) tokens -> project to ``hidden_dim``.

    :ivar hidden_dim: output feature dimension ``h``.
    :ivar patch_size: square patch side length.
    :ivar embed_dim: transformer token dimension (DeiT-Tiny: 192).
    :ivar depth: number of Transformer blocks.
    :ivar num_heads: attention heads.
    :ivar mlp_ratio: MLP expansion ratio.
    :ivar use_distill_token: include the DeiT distillation token.
    """

    hidden_dim: int = 256
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    use_distill_token: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Encode a batch of images with the ViT/DeiT backbone.

        :param x: ``(B, H, W, C)`` images. ``H, W`` must be divisible by
                  ``patch_size``.
        :param train: training flag.
        :return: ``(B, hidden_dim)`` pooled features.
        """
        b, h, w, c = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            "Image size must be divisible by patch_size."
        )
        # Patch embedding via a strided conv (equivalent to linear patch proj).
        patches = nn.Conv(
            self.embed_dim,
            (self.patch_size, self.patch_size),
            (self.patch_size, self.patch_size),
            padding="VALID",
            name="patch_embed",
        )(x)
        n_h, n_w = patches.shape[1], patches.shape[2]
        tokens = patches.reshape(b, n_h * n_w, self.embed_dim)
        n_patches = n_h * n_w

        cls = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, self.embed_dim))
        cls = jnp.broadcast_to(cls, (b, 1, self.embed_dim))
        prefix = [cls]
        n_special = 1
        if self.use_distill_token:
            dist = self.param(
                "dist_token", nn.initializers.normal(0.02), (1, 1, self.embed_dim)
            )
            prefix.append(jnp.broadcast_to(dist, (b, 1, self.embed_dim)))
            n_special = 2
        tokens = jnp.concatenate(prefix + [tokens], axis=1)

        pos = self.param(
            "pos_embed",
            nn.initializers.normal(0.02),
            (1, n_patches + n_special, self.embed_dim),
        )
        tokens = tokens + pos

        for i in range(self.depth):
            tokens = TransformerEncoderBlock(
                self.embed_dim, self.num_heads, self.mlp_ratio, name=f"block_{i}"
            )(tokens, train=train)

        tokens = nn.LayerNorm()(tokens)
        # Pool the special (CLS + optional DIST) tokens, per DeiT inference.
        pooled = jnp.mean(tokens[:, :n_special, :], axis=1)
        return nn.Dense(self.hidden_dim)(pooled)


def build_backbone(cfg) -> nn.Module:
    """
    Construct a backbone module from a :class:`~dbwm.config.BackboneConfig`.

    :param cfg: backbone configuration.
    :return: an un-initialised Flax backbone module.
    """
    if cfg.kind == "resnet":
        return ResNetBackbone(
            hidden_dim=cfg.hidden_dim,
            stage_sizes=cfg.resnet_stage_sizes,
            width=cfg.resnet_width,
        )
    elif cfg.kind == "deit":
        return DeiTBackbone(
            hidden_dim=cfg.hidden_dim,
            patch_size=cfg.deit_patch_size,
            embed_dim=cfg.deit_embed_dim,
            depth=cfg.deit_depth,
            num_heads=cfg.deit_num_heads,
            mlp_ratio=cfg.deit_mlp_ratio,
            use_distill_token=cfg.deit_use_distill_token,
        )
    raise ValueError("Unknown backbone kind: {}".format(cfg.kind))
