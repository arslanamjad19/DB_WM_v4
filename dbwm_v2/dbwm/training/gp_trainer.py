"""
Training the spatial deep basis ``Psi`` on the GP path (two-phase schedule (a)).

v2 Remark 6.1 offers two schedules. The v4 NDVI pipeline uses the **two-phase**
one: train ``Psi`` first with the dPPGP objective, freeze it, encode every date
through the GP posterior, then identify the dynamics in closed form. That ordering
is deliberate -- v3 Open Item 2 notes that training the basis *jointly* with a
multi-horizon loss rewards features that are predictable rather than informative,
an anti-collapse pressure dPPGP was never designed for. Freezing ``Psi`` sidesteps
it entirely.

The objective (Definition 3.1, GP-path form)
--------------------------------------------
There is no image encoder here, so the loss contains no encoder-consistency term.
Each step samples a batch of dates and splits that date's valid pixels into a
**context** set and a disjoint **target** set::

    w_t      = (Phi_C^T Phi_C + sigma_eps^2 I)^{-1} Phi_C^T y_C     (GP posterior)
    L_pred   = mean over target pixels of -log N(y_T ; <w_t, Psi(x_T)>, var + sigma^2)
    L_trace  = mean over pixels of (k_b - ||Psi(x)||^2) / (2 sigma_eps^2)
    L_kl     = (beta / n) D_KL( N(m, LL^T) || N(0, I_r) )

Scoring on **held-out** pixels is what makes this a predictive objective rather
than a reconstruction one: a basis can always fit the pixels it was solved on, so
an in-sample score would reward memorisation and say nothing about whether ``Psi``
generalises across the field.

The trace regulariser and the KL are the anti-collapse machinery of Remark 3.1:
without them the low-rank kernel is free to collapse onto a single direction, which
would make every downstream Koopman mode meaningless.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax import struct

from dbwm.models.spatial_basis import SpatialBasis
from dbwm.models.variational import VariationalPosterior
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.gptrain")


class SpatialGPModule(nn.Module):
    """
    The GP-path model: spatial basis + variational posterior + noise.

    Deliberately excludes the image backbone. On the GP path the state comes from
    the pixel observations through the GP posterior, so an encoder would be an
    unused parameter block competing for gradient signal.

    :ivar cfg: a :class:`~dbwm.config.BasisConfig`.
    """

    cfg: object

    def setup(self):
        """Instantiate the basis, the variational posterior and the noise."""
        self.spatial = SpatialBasis(self.cfg)
        self.variational = VariationalPosterior(self.cfg.r)
        self.log_sigma_eps2 = self.param(
            "log_sigma_eps2",
            lambda key: jnp.array(
                jnp.log(self.cfg.sigma_eps2_init), dtype=jnp.float32
            ),
        )
        # Latent operator used ONLY by the teacher-forced rollout term. Initialised
        # to the identity, i.e. persistence: the rollout starts from the strongest
        # trivial forecast and can only be pushed away from it by evidence. It is
        # not the operator the pipeline forecasts with -- that one is identified in
        # closed form on the frozen basis (v2 Algorithm 3). Its job here is to give
        # the rollout something to propagate so that gradients reach Psi.
        self.rollout_a = self.param(
            "rollout_a", lambda key: jnp.eye(self.cfg.r, dtype=jnp.float32)
        )

    def features(self, coords):
        """
        Evaluate ``Psi`` at coordinates.

        :param coords: ``(N, 2)`` coordinates.
        :return: ``(N, r)`` features.
        """
        return self.spatial(coords)

    def sigma_eps2(self):
        """Positive observation-noise variance."""
        return jnp.exp(self.log_sigma_eps2)

    def predictive_variance(self, phi):
        """
        dPPGP predictive variance ``||L^T Psi(x)||^2``.

        :param phi: ``(N, r)`` features.
        :return: ``(N,)`` variances.
        """
        _, var = self.variational(phi)
        return var

    def kl(self):
        """Variational KL to the standard normal prior."""
        return self.variational.kl_to_standard_normal()

    def operator(self):
        """The rollout operator ``A`` (identity at initialisation)."""
        return self.rollout_a

    def __call__(self, coords):
        """
        Touch every sub-module so ``init`` materialises the full parameter tree.

        :param coords: ``(N, 2)`` coordinates.
        :return: dict of intermediate quantities.
        """
        phi = self.features(coords)
        return {
            "phi": phi,
            "var": self.predictive_variance(phi),
            "kl": self.kl(),
            "sigma_eps2": self.sigma_eps2(),
            "operator": self.operator(),
        }


def teacher_forced_rollout(
    a: jnp.ndarray,
    w_seq: jnp.ndarray,
    phi_tgt: jnp.ndarray,
    y_tgt: jnp.ndarray,
    offset_tgt: jnp.ndarray,
    threshold: float,
    use_teacher_forcing: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Roll the latent dynamics across a window of dates, forcing on **pixel** RMSE.

    This is the scheduled-sampling rule as specified: predict
    ``w_hat_{t+1} = A w_t``, decode it to the field, and compare against the true
    frame. If the decoded RMSE exceeds ``threshold``, feed the ground-truth state
    into the next step instead of the prediction; otherwise let the model run on
    its own output.

    Why the threshold is on decoded pixels and not on ``||w||``
    -----------------------------------------------------------
    The specified 0.02 is an NDVI number, and latent distance is not commensurate
    with it: ``||w_hat - w||`` depends on the arbitrary scaling of the basis, so
    the same threshold would mean different things for different ``Psi`` and would
    silently change meaning as ``Psi`` trains. Decoding first makes the criterion
    exactly the reported metric, so "forced whenever the forecast is worse than
    0.02 NDVI" is literally true.

    The branch selector is ``stop_gradient``'d, so forcing switches the *value*
    fed forward without contributing a gradient through the comparison itself --
    the recursion stays differentiable and the scan stays ``jit``-able.

    :param a: ``(r, r)`` transition operator.
    :param w_seq: ``(K+1, r)`` ground-truth GP states across the window.
    :param phi_tgt: ``(n_t, r)`` basis at the held-out target pixels.
    :param y_tgt: ``(K+1, n_t)`` true values at those pixels.
    :param offset_tgt: ``(n_t,)`` climatology at those pixels.
    :param threshold: pixel RMSE above which ground truth is fed forward, in the
        same (normalised) units as ``y_tgt``.
    :param use_teacher_forcing: disable to get a pure free-running rollout.
    :return: ``(per_step_mse, forced_fraction)``.
    """

    def step(carry, idx):
        w_state = carry
        w_pred = a @ w_state
        pred = phi_tgt @ w_pred + offset_tgt
        err = pred - y_tgt[idx + 1]
        mse = jnp.mean(err**2)
        rmse = jnp.sqrt(mse)
        w_true_next = w_seq[idx + 1]
        if use_teacher_forcing:
            feed_truth = jax.lax.stop_gradient(rmse > threshold)
            next_state = jnp.where(feed_truth, w_true_next, w_pred)
            forced = feed_truth.astype(jnp.float32)
        else:
            next_state = w_pred
            forced = jnp.array(0.0, dtype=jnp.float32)
        return next_state, (mse, forced)

    _, (mses, forced) = jax.lax.scan(
        step, w_seq[0], jnp.arange(w_seq.shape[0] - 1)
    )
    return mses, forced


@struct.dataclass
class GPTrainState:
    """Optimiser state bundle (a pytree, so it can cross a ``jit`` boundary)."""

    params: dict
    opt_state: optax.OptState
    step: jnp.ndarray


def _gp_solve(phi_c: jnp.ndarray, y_c: jnp.ndarray, sigma2: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiable GP posterior mean ``w = (Phi^T Phi + sigma^2 I)^{-1} Phi^T y``.

    :param phi_c: ``(n_c, r)`` context features.
    :param y_c: ``(n_c,)`` context targets.
    :param sigma2: scalar noise variance.
    :return: ``(r,)`` posterior mean weight.
    """
    r = phi_c.shape[-1]
    lam = phi_c.T @ phi_c + sigma2 * jnp.eye(r)
    return jnp.linalg.solve(lam, phi_c.T @ y_c)


def dppgp_loss(
    params,
    model: SpatialGPModule,
    coords_ctx: jnp.ndarray,
    y_ctx: jnp.ndarray,
    coords_tgt: jnp.ndarray,
    y_tgt: jnp.ndarray,
    alpha_trace: float,
    beta_kl: float,
    n_total: float,
    y_roll: Optional[jnp.ndarray] = None,
    lambda_rollout: float = 0.0,
    tf_threshold: float = 0.02,
    use_teacher_forcing: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    dPPGP objective on the GP path, scored on held-out target pixels.

    :param params: model parameters.
    :param model: the :class:`SpatialGPModule`.
    :param coords_ctx: ``(n_c, 2)`` context coordinates.
    :param y_ctx: ``(b, n_c)`` context values, one row per date.
    :param coords_tgt: ``(n_t, 2)`` target coordinates (disjoint from context).
    :param y_tgt: ``(b, n_t)`` target values.
    :param alpha_trace: trace-regulariser weight.
    :param beta_kl: KL weight (scaled by ``1/n_total``).
    :param n_total: dataset size used to scale the KL.
    :param y_roll: ``(K+1, n_c + n_t)`` values on a window of CONSECUTIVE dates,
        context columns first, used for the teacher-forced rollout term. ``None``
        disables it.
    :param lambda_rollout: weight on the rollout term.
    :param tf_threshold: pixel-RMSE threshold above which ground truth is fed
        forward, in normalised units.
    :param use_teacher_forcing: enable scheduled sampling in the rollout.
    :return: ``(loss, metrics)``.
    """
    phi_c = model.apply(params, coords_ctx, method=model.features)  # (n_c, r)
    phi_t = model.apply(params, coords_tgt, method=model.features)  # (n_t, r)
    sigma2 = model.apply(params, method=model.sigma_eps2)

    w = jax.vmap(lambda y: _gp_solve(phi_c, y, sigma2))(y_ctx)  # (b, r)
    mu = w @ phi_t.T  # (b, n_t)
    var_q = model.apply(params, phi_t, method=model.predictive_variance)  # (n_t,)
    var = jnp.clip(var_q[None, :] + sigma2, 1e-8, None)

    nll = 0.5 * jnp.mean((y_tgt - mu) ** 2 / var + jnp.log(var) + jnp.log(2.0 * jnp.pi))
    norms = jnp.sum(phi_t**2, axis=-1)
    trace = jnp.mean(jnp.max(norms) - norms) / (2.0 * sigma2)
    kl = model.apply(params, method=model.kl)

    loss = nll + alpha_trace * trace + (beta_kl / n_total) * kl

    # --- teacher-forced rollout (v2 Remark 6.1 schedule (b)) ----------------- #
    # This is what makes Psi *dynamics-aware*: without it the basis is optimised
    # purely for one-frame reconstruction and has no reason to prefer features
    # that propagate. v3 Open Item 2 warns that a multi-horizon loss can reward
    # merely PREDICTABLE features, so the term is weighted rather than dominant
    # and the effective rank of Phi is reported so collapse would be visible.
    roll_mse = jnp.array(0.0)
    forced_frac = jnp.array(0.0)
    if y_roll is not None and lambda_rollout > 0.0:
        n_c = y_ctx.shape[-1]
        w_roll = jax.vmap(lambda y: _gp_solve(phi_c, y[:n_c], sigma2))(y_roll)
        a_op = model.apply(params, method=model.operator)
        mses, forced = teacher_forced_rollout(
            a_op, w_roll, phi_t, y_roll[:, n_c:], jnp.zeros(phi_t.shape[0]),
            tf_threshold, use_teacher_forcing,
        )
        roll_mse = jnp.mean(mses)
        forced_frac = jnp.mean(forced)
        loss = loss + lambda_rollout * roll_mse

    mse = jnp.mean((y_tgt - mu) ** 2)
    return loss, {
        "loss": loss,
        "nll": nll,
        "trace": trace,
        "kl": kl,
        "mse": mse,
        "sigma_eps2": sigma2,
        "rollout_mse": roll_mse,
        "forced_fraction": forced_frac,
    }


def train_spatial_basis(
    cfg,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    observed: np.ndarray,
    coords: np.ndarray,
    n_epochs: Optional[int] = None,
    batch_dates: int = 8,
    n_context: int = 1024,
    n_target: int = 1024,
    seed: int = 0,
    rollout_steps: int = 0,
    lambda_rollout: float = 0.0,
    tf_threshold: float = 0.02,
    use_teacher_forcing: bool = True,
    norm_std: float = 1.0,
) -> Tuple[SpatialGPModule, dict, Dict[str, float]]:
    """
    Train ``Psi`` with the dPPGP objective on the training split.

    :param cfg: an :class:`~dbwm.config.ExperimentConfig`.
    :param frames: ``(T, H, W)`` normalised training frames.
    :param valid_mask: ``(T, H, W)`` validity.
    :param observed: ``(T,)`` which dates carry a frame.
    :param coords: ``(H*W, 2)`` normalised pixel coordinates.
    :param n_epochs: override ``cfg.training.n_epochs``.
    :param batch_dates: dates per step.
    :param n_context: context pixels used to solve ``w_t``.
    :param n_target: held-out pixels the loss is scored on.
    :param seed: RNG seed.
    :param rollout_steps: ``K``, the number of teacher-forced rollout steps. 0
        disables the term and recovers the pure two-phase schedule (a).
    :param lambda_rollout: weight on the rollout term.
    :param tf_threshold: teacher-forcing threshold in **physical** units (NDVI).
        Converted internally by ``norm_std`` so the specified 0.02 means 0.02 NDVI
        regardless of how the frames were normalised.
    :param use_teacher_forcing: enable scheduled sampling.
    :param norm_std: the normalisation std used on the frames, for the conversion
        above.
    :return: ``(model, params, final_metrics)``.
    """
    frames = np.asarray(frames)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    coords = np.asarray(coords, dtype=np.float32)

    usable = np.nonzero(observed)[0]
    if usable.size == 0:
        raise ValueError("No observed frames to train the spatial basis on.")
    flat = frames.reshape(frames.shape[0], -1)
    mask_flat = valid_mask.reshape(valid_mask.shape[0], -1)
    coverage = mask_flat[usable].mean(axis=0)
    always_valid = np.nonzero(coverage >= 1.0)[0]
    if always_valid.size < 64:
        always_valid = np.nonzero(coverage > 0.9)[0]
    if always_valid.size < 64:
        # Say which of the two causes it is, because they need opposite fixes and
        # the old message ("cannot be trained on this AOI") pointed at neither.
        # The usual cause is not a bad AOI at all: it is one frame that exists but
        # carries no valid pixels, marked observed, which zeroes the intersection
        # for every other date. The loader now drops those for LST, so reaching
        # here means something else.
        empty = int((mask_flat[usable].sum(axis=1) == 0).sum())
        ever = int((coverage > 0).sum())
        raise ValueError(
            "Fewer than 64 pixels are reliably valid, so the spatial basis cannot "
            "be trained.\n"
            "  observed frames        : {n}\n"
            "  frames with NO valid px: {empty}\n"
            "  pixels ever valid      : {ever}\n"
            "  pixels valid >90% dates: {p90}\n"
            "If 'frames with no valid px' is nonzero, those rasters are the "
            "cause and should be dropped rather than counted as observations "
            "(automatic for --modality lst; check --keep-empty-frames is not "
            "set). If 'pixels ever valid' is itself tiny, the AOI or the "
            "no-data sentinel is wrong. If pixels are *ever* valid but rarely "
            "so, a validity gate is too tight -- widen --lst-valid-range.".format(
                n=usable.size, empty=empty, ever=ever,
                p90=int((coverage > 0.9).sum()),
            )
        )

    model = SpatialGPModule(cfg.basis)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.asarray(coords[:16]))
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip_norm),
        optax.adamw(cfg.training.learning_rate, weight_decay=cfg.training.weight_decay),
    )
    state = GPTrainState(
        params=params, opt_state=optimizer.init(params), step=jnp.array(0)
    )

    n_total = float(always_valid.size * usable.size)

    tau_norm = float(tf_threshold) / max(float(norm_std), 1e-12)

    @jax.jit
    def step_fn(st, c_ctx, y_ctx, c_tgt, y_tgt, y_roll):
        def loss_fn(p):
            return dppgp_loss(
                p, model, c_ctx, y_ctx, c_tgt, y_tgt,
                cfg.training.alpha_trace, cfg.training.beta_kl, n_total,
                y_roll=y_roll, lambda_rollout=lambda_rollout,
                tf_threshold=tau_norm, use_teacher_forcing=use_teacher_forcing,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(st.params)
        updates, opt_state = optimizer.update(grads, st.opt_state, st.params)
        new_params = optax.apply_updates(st.params, updates)
        return GPTrainState(new_params, opt_state, st.step + 1), metrics

    epochs = cfg.training.n_epochs if n_epochs is None else n_epochs
    steps_per_epoch = max(1, usable.size // max(batch_dates, 1))
    rng = np.random.RandomState(seed)
    n_ctx = min(n_context, max(always_valid.size // 2, 8))
    n_tgt = min(n_target, always_valid.size - n_ctx)
    logger.info(
        "Training Psi (dPPGP, GP path): %d epochs x %d steps | r=%d | "
        "%d context / %d target pixels per step (target pixels are HELD OUT).",
        epochs, steps_per_epoch, cfg.basis.r, n_ctx, n_tgt,
    )

    # Windows of CONSECUTIVE observed dates, for the rollout term. A window with a
    # gap in it would roll the operator across a hole and score the result as if
    # it were a one-day step, which silently mis-states the dynamics.
    windows = []
    if rollout_steps > 0 and lambda_rollout > 0.0:
        obs = np.zeros(frames.shape[0], dtype=bool)
        obs[usable] = True
        for t0 in range(frames.shape[0] - rollout_steps):
            if obs[t0 : t0 + rollout_steps + 1].all():
                windows.append(t0)
        windows = np.asarray(windows, dtype=int)
        if windows.size == 0:
            logger.warning(
                "No gap-free window of %d consecutive observed dates: the "
                "teacher-forced rollout term is disabled for this record.",
                rollout_steps + 1,
            )
            lambda_rollout = 0.0
        else:
            logger.info(
                "Teacher forcing ON: %d rollout steps, %d gap-free windows, "
                "threshold %.4f NDVI (= %.4f normalised), weight %.3g. Ground "
                "truth is fed forward whenever the decoded RMSE exceeds it.",
                rollout_steps, windows.size, tf_threshold, tau_norm,
                lambda_rollout,
            )

    last: Dict[str, float] = {}
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            dates = rng.choice(usable, size=min(batch_dates, usable.size), replace=False)
            pix = rng.permutation(always_valid)
            ctx, tgt = pix[:n_ctx], pix[n_ctx : n_ctx + n_tgt]
            y_roll = None
            if lambda_rollout > 0.0 and len(windows):
                t0 = int(rng.choice(windows))
                span = np.arange(t0, t0 + rollout_steps + 1)
                y_roll = jnp.asarray(
                    np.concatenate(
                        [flat[span][:, ctx], flat[span][:, tgt]], axis=1
                    )
                )
            state, metrics = step_fn(
                state,
                jnp.asarray(coords[ctx]),
                jnp.asarray(flat[dates][:, ctx]),
                jnp.asarray(coords[tgt]),
                jnp.asarray(flat[dates][:, tgt]),
                y_roll,
            )
            last = metrics
        if epoch % max(cfg.training.log_every, 1) == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %3d | loss %.4f | held-out NLL %.4f | held-out MSE %.5f | "
                "trace %.3e | KL %.1f | sigma_eps2 %.3e | rollout MSE %.5f | "
                "forced %.0f%%",
                epoch, float(last["loss"]), float(last["nll"]), float(last["mse"]),
                float(last["trace"]), float(last["kl"]), float(last["sigma_eps2"]),
                float(last.get("rollout_mse", 0.0)),
                100.0 * float(last.get("forced_fraction", 0.0)),
            )
    return model, state.params, {k: float(v) for k, v in last.items()}
