"""
DB-WM training loop (Algorithm 1) with scheduled-sampling teacher forcing.

Implements the end-to-end schedule (v3 Sec. 4.6 (b)): the encoder ``phi_theta``,
spatial basis ``Psi``, variational posterior ``(m, L)``, observation noise
``sigma_eps^2`` *and* the differentiable transition operator ``A`` (and ``B`` if
forcing is enabled) are all updated by SGD on the composite loss of
:mod:`dbwm.training.losses`.

A complementary closed-form path (regularized matrix least squares, Algorithm 3) is
provided by :func:`identify_dynamics_closed_form` for the two-phase schedule (a):
train the basis first, freeze it, then solve ``[A B]`` in closed form. Per Section
2.5 / Proposition 4.4 this is *not* a DMD -- the deep basis has already performed
the reduction to ``R^r``, so no truncating SVD is involved.

Data are served as ``(b, T, H, W, C)`` trajectory-segment mini-batches by
:class:`TrajectoryBatcher`; pixel coordinates for the dPPGP term are subsampled
each step so per-iteration cost stays ``O(b T (r^2 + c_phi))`` and never
materialises the full ``n x r`` feature matrix.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import struct

from dbwm.models.db_wm import DBWM
from dbwm.data.geotiff_dataset import SpatiotemporalDataset, make_pixel_grid
from dbwm.dynamics.identification import identify, persistence_of_excitation
from dbwm.dynamics.transition import spectral_radius
from dbwm.training import losses as L
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.train")


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
class TrajectoryBatcher:
    """
    Sample fixed-length contiguous trajectory segments from a dataset.

    :ivar ds: the source :class:`SpatiotemporalDataset`.
    :ivar segment_length: ``T``.
    :ivar batch_size: ``b``.
    :ivar seed: RNG seed for segment-start sampling.
    """

    def __init__(self, ds: SpatiotemporalDataset, segment_length: int, batch_size: int, seed: int = 0):
        self.ds = ds
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.max_start = ds.n_frames - segment_length
        if self.max_start < 0:
            raise ValueError(
                "segment_length {} exceeds dataset length {}.".format(
                    segment_length, ds.n_frames
                )
            )

    def next_batch(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Draw one mini-batch of trajectory segments.

        :return: ``(frames, forcing)`` with ``frames`` ``(b, T, H, W, C)`` and
                 ``forcing`` ``(b, T, ell)`` or ``None``.
        """
        starts = self.rng.randint(0, self.max_start + 1, size=self.batch_size)
        f = np.stack(
            [self.ds.frames[s : s + self.segment_length] for s in starts], axis=0
        )
        if self.ds.forcing is None:
            u = None
        else:
            u = np.stack(
                [self.ds.forcing[s : s + self.segment_length] for s in starts], axis=0
            )
        return f.astype(np.float32), (None if u is None else u.astype(np.float32))


@struct.dataclass
class TrainState:
    """Bundle of training state (a pytree, so it can be a jit argument)."""

    params: dict
    opt_state: optax.OptState
    step: jnp.ndarray


# --------------------------------------------------------------------------- #
# Loss assembly
# --------------------------------------------------------------------------- #
def _encode_segment(model, params, frames, train):
    """
    Encode a ``(b, T, H, W, C)`` batch into ``(b, T, r)`` weight trajectories.

    :return: ``(b, T, r)`` encoded weights.
    """
    b, t = frames.shape[0], frames.shape[1]
    flat = frames.reshape((b * t,) + frames.shape[2:])
    w = model.apply(params, flat, train=train, method=model.encode)
    return w.reshape(b, t, -1)


def compute_loss(
    params,
    model: DBWM,
    frames: jnp.ndarray,
    forcing: Optional[jnp.ndarray],
    coords: jnp.ndarray,
    targets: jnp.ndarray,
    cfg,
    train: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute the full composite DB-WM loss for one mini-batch.

    :param params: model parameters.
    :param model: the :class:`DBWM` module.
    :param frames: ``(b, T, H, W, C)`` images.
    :param forcing: ``(b, T, ell)`` forcing or ``None``.
    :param coords: ``(N, 2)`` subsampled pixel coordinates for the dPPGP term.
    :param targets: ``(b, T, N)`` ground-truth pixel values at ``coords``.
    :param cfg: an :class:`~dbwm.config.ExperimentConfig`.
    :param train: training flag.
    :return: ``(loss, metrics)``.
    """
    tcfg, dcfg = cfg.training, cfg.dynamics
    w_batch = _encode_segment(model, params, frames, train)  # (b, T, r)
    a = model.apply(params, method=model.transition)  # (r, r)
    sigma2 = model.apply(params, method=model.sigma_eps2)

    if dcfg.use_forcing and forcing is not None:
        b_mat = model.apply(params, method=model.input_mat)
    else:
        b_mat, forcing = None, None

    # --- L_dyn: teacher-forced rollout in weight space ---
    l_dyn = L.dynamics_loss(
        a, w_batch, b_mat, forcing, tcfg.tf_threshold, tcfg.teacher_forcing
    )

    # --- Pixel predictions for the data + dPPGP terms ---
    phi_pix = model.apply(params, coords, method=model.spatial_features)  # (N, r)
    pred = jnp.einsum("btr,nr->btn", w_batch, phi_pix)  # (b, T, N) means
    var_pix = model.apply(params, coords, method=model.variational_var)  # (N,)

    # L_recon: plain MSE data fit (ties encoder state to pixels).
    l_recon = jnp.mean((pred - targets) ** 2)

    # L_dPPGP: pixel NLL + trace + (beta/n) KL.
    l_nll = L.gaussian_nll(targets, pred, var_pix[None, None, :])
    fnorms = model.apply(params, coords, method=model.feature_norms)
    l_trace = L.trace_regularizer(fnorms, sigma2)
    kl = model.apply(params, method=model.kl)
    n_eff = float(targets.shape[0] * targets.shape[1] * targets.shape[2])
    l_dppgp = l_nll + tcfg.alpha_trace * l_trace + (tcfg.beta_kl / n_eff) * kl

    # L_spec: spectral-radius safety.
    l_spec = L.spectral_penalty(a, dcfg.rho_max)

    loss = (
        tcfg.lambda_consistency * l_recon
        + tcfg.lambda_dynamics * l_dyn
        + tcfg.lambda_dppgp * l_dppgp
        + tcfg.lambda_spectral * l_spec
    )
    metrics = {
        "loss": loss,
        "l_recon": l_recon,
        "l_dyn": l_dyn,
        "l_nll": l_nll,
        "l_trace": l_trace,
        "kl": kl,
        "l_spec": l_spec,
        "sigma_eps2": sigma2,
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Train step (jit'd)
# --------------------------------------------------------------------------- #
def make_train_step(model: DBWM, optimizer: optax.GradientTransformation, cfg):
    """
    Build a jit-compiled training step closing over the model, optimizer, cfg.

    :return: ``train_step(state, frames, forcing, coords, targets) -> (state, metrics)``.
    """

    @partial(jax.jit, static_argnums=())
    def train_step(state: TrainState, frames, forcing, coords, targets):
        def loss_fn(p):
            return compute_loss(
                p, model, frames, forcing, coords, targets, cfg, train=True
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        new_state = TrainState(params=params, opt_state=opt_state, step=state.step + 1)
        return new_state, metrics

    return train_step


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def init_train_state(model: DBWM, cfg, sample_frame, coords) -> Tuple[TrainState, optax.GradientTransformation]:
    """
    Initialise parameters and the Adam(W) optimiser with gradient clipping.

    :param model: the :class:`DBWM` module.
    :param cfg: experiment config.
    :param sample_frame: a single ``(H, W, C)`` frame for shape inference.
    :param coords: ``(N, 2)`` coordinate sample for init.
    :return: ``(state, optimizer)``.
    """
    key = jax.random.PRNGKey(cfg.training.seed)
    o = jnp.asarray(sample_frame)[None]  # (1, H, W, C)
    params = model.init(key, o, coords, train=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip_norm),
        optax.adamw(cfg.training.learning_rate, weight_decay=cfg.training.weight_decay),
    )
    opt_state = optimizer.init(params)
    return (
        TrainState(params=params, opt_state=opt_state, step=jnp.array(0)),
        optimizer,
    )


def _gather_targets(frames: np.ndarray, pixel_idx: np.ndarray) -> np.ndarray:
    """
    Gather ground-truth pixel values at ``pixel_idx`` for a frame batch.

    :param frames: ``(b, T, H, W, C)`` (single channel used).
    :param pixel_idx: ``(N,)`` flat pixel indices into ``H*W``.
    :return: ``(b, T, N)`` targets.
    """
    b, t, h, w, c = frames.shape
    flat = frames[..., 0].reshape(b, t, h * w)
    return flat[:, :, pixel_idx]


def train(model: DBWM, train_ds: SpatiotemporalDataset, cfg) -> TrainState:
    """
    Run the DB-WM training loop and return the trained state.

    :param model: the :class:`DBWM` module.
    :param train_ds: training :class:`SpatiotemporalDataset`.
    :param cfg: experiment config.
    :return: the final :class:`TrainState`.
    """
    h, w, c = train_ds.image_shape
    grid = make_pixel_grid(h, w)  # (H*W, 2)
    n_pix = grid.shape[0]
    rng = np.random.RandomState(cfg.training.seed)

    init_idx = rng.choice(n_pix, size=min(cfg.training.dppgp_pixel_samples, n_pix), replace=False)
    state, optimizer = init_train_state(
        model, cfg, train_ds.frames[0], jnp.asarray(grid[init_idx])
    )
    train_step = make_train_step(model, optimizer, cfg)
    batcher = TrajectoryBatcher(
        train_ds, cfg.training.segment_length, cfg.training.batch_size, cfg.training.seed
    )

    steps_per_epoch = max(1, train_ds.n_frames // cfg.training.batch_size)
    logger.info(
        "Training %s: %d epochs x %d steps (r=%d, backbone=%s, expansion=%s).",
        cfg.name, cfg.training.n_epochs, steps_per_epoch, cfg.basis.r,
        cfg.backbone.kind, cfg.basis.expansion,
    )
    for epoch in range(cfg.training.n_epochs):
        last = {}
        for _ in range(steps_per_epoch):
            frames, forcing = batcher.next_batch()
            pix = rng.choice(n_pix, size=min(cfg.training.dppgp_pixel_samples, n_pix), replace=False)
            coords = jnp.asarray(grid[pix])
            targets = jnp.asarray(_gather_targets(frames, pix))
            f = jnp.asarray(frames)
            u = None if forcing is None else jnp.asarray(forcing)
            state, last = train_step(state, f, u, coords, targets)
        if epoch % cfg.training.log_every == 0 or epoch == cfg.training.n_epochs - 1:
            logger.info(
                "epoch %3d | loss %.4f | dyn %.4f | nll %.4f | recon %.4f | spec %.2e | s2 %.3e",
                epoch, float(last["loss"]), float(last["l_dyn"]), float(last["l_nll"]),
                float(last["l_recon"]), float(last["l_spec"]), float(last["sigma_eps2"]),
            )
    return state


def identify_dynamics_closed_form(
    model: DBWM, params, ds: SpatiotemporalDataset, cfg
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Two-phase schedule (a): encode all dates with the frozen basis, then solve
    ``[A B]`` and ``Q`` in closed form by regularized matrix least squares
    (Algorithm 3).

    Also reports persistence of excitation (Remark 6.1). This is the condition under
    which ``A`` and ``B`` are *separately* identifiable at all: if the forcing is
    collinear with an autonomous mode -- monsoon rain aligned with the seasonal
    swing is the canonical LST failure -- the regression cannot attribute the
    variance and ``B_p`` absorbs part of the thermal dynamics.

    :param model: trained :class:`DBWM`.
    :param params: trained parameters.
    :param ds: dataset to identify dynamics on (usually the training split).
    :param cfg: experiment config.
    :return: ``(A, B, Q)``.
    """
    frames = jnp.asarray(ds.frames)  # (T, H, W, C)
    w = model.apply(params, frames, train=False, method=model.encode)  # (T, r)
    forcing = None if ds.forcing is None else jnp.asarray(ds.forcing)

    if forcing is not None and cfg.dynamics.use_forcing:
        pe = persistence_of_excitation(w, forcing)
        if not pe["satisfied"]:
            logger.warning(
                "Persistence of excitation NOT satisfied: rank(Omega)=%d < r+ell=%d "
                "(deficiency %d, cond=%.2e). A and B are not separately identifiable "
                "from this data; the ridge is doing the disambiguating, not the data.",
                pe["rank"], pe["required"], pe["deficiency"], pe["cond"],
            )
        else:
            logger.info(
                "Persistence of excitation satisfied (rank %d = r+ell, cond=%.2e).",
                pe["rank"], pe["cond"],
            )

    a, b, q = identify(w, forcing, cfg.dynamics)
    logger.info(
        "Closed-form least-squares identification complete (%s): A %s, B %s, Q %s.",
        cfg.dynamics.identification, a.shape, b.shape, q.shape,
    )
    return a, b, q
