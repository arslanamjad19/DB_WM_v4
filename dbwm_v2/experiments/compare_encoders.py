"""
Which state map should v4 use: the GP posterior, or an image encoder?

v4 takes the state from the GP posterior, ``w_t = Lambda_X^{-1} Phi_X^T y_t``,
while v2 took it from an image backbone, ``w_t = phi_theta(o_t)``. The choice was
argued in :mod:`dbwm.gp.state` on theoretical grounds -- the encoder needs a fixed
dense grid, so the ~52% of this bounding box that lies outside the field clip has
to be zero-filled *before the network sees it*, whereas the GP simply omits those
rows -- but it was never **measured**. This script measures it.

All three paths share the same calendar, the same split, the same augmented
spatial basis, the same dynamics identification and the same scoring function, so
the only thing that varies is how ``w_t`` is produced:

===================  =========================================================
``gp``               ``w_t = Lambda_X^{-1} Phi_X^T y_t`` over VALID pixels only
``deit``             ``w_t = phi_theta(o_t)``, DeiT-Tiny ViT backbone
``resnet``           ``w_t = phi_theta(o_t)``, ResNet backbone
===================  =========================================================

What to look at, in order
------------------------
**Reconstruction R^2 first, forecast RMSE second.** v2 Theorem 4.2 bounds the
``h``-step error by ``rho^h eps_enc + ...``, so the encoding error enters every
horizon multiplied by the operator gain. On the real record this term was the
*entire* one-step error, which means an encoder that reconstructs worse cannot be
rescued by better dynamics and one that reconstructs better starts ahead
regardless of them. A comparison that reported only forecast RMSE would confuse
the two effects.

**Then the invalid-pixel cost.** The table reports ``R^2`` restricted to valid
pixels for every path, so the encoder is not penalised for the nodata it was
forced to ingest -- only for what that ingestion does to the state it produces.

Usage
-----
    python -m experiments.compare_encoders --smoke
    python -m experiments.compare_encoders --ndvi-dir <dir> --weather-csv <csv> \\
        --encoders gp deit resnet --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from dbwm.platform import preflight

_PREFLIGHT = preflight()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from dbwm.config import default_config, smoke_config  # noqa: E402
from dbwm.data.seasons import purge_boundary_windows  # noqa: E402
from dbwm.dynamics.memory import (  # noqa: E402
    calibrate_persistence_blend, forecast_gain, identify_memory,
)
from dbwm.dynamics.subspace import build_subspace  # noqa: E402
from dbwm.gp.empirical_basis import build_augmented_basis  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.models.db_wm import DBWM  # noqa: E402
from dbwm.training.gp_trainer import (  # noqa: E402
    teacher_forced_rollout, train_spatial_basis,
)
from experiments._v4_common import (  # noqa: E402
    add_data_args, add_model_args, apply_model_args, load_inputs,
)

logger = configure_logger(level="INFO", name="dbwm.encoders")


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="GP posterior vs image encoders (v4)")
    add_data_args(p)
    p.add_argument("--encoders", nargs="+", default=["gp", "deit", "resnet"],
                   choices=["gp", "deit", "resnet"])
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--memory-order", type=int, default=None)
    p.add_argument("--results-dir", default="./_v4_encoders")
    p.add_argument("--forecast-stride", type=int, default=3)
    p.add_argument("--subspace-energy", type=float, default=0.999)
    add_model_args(p)
    return p.parse_args()


def _pad_to_patch(frames: np.ndarray, patch: int) -> np.ndarray:
    """
    Zero-pad a frame stack so both spatial sides are divisible by ``patch``.

    The ViT requires it. Note what this costs and why it is reported: padding is
    a *second* dose of fabricated pixels on top of the nodata fill, and both are
    inputs the GP path never has to invent.

    :param frames: ``(T, H, W)`` frames.
    :param patch: patch side length.
    :return: ``(T, H', W', 1)`` padded frames with a channel axis.
    """
    t, h, w = frames.shape
    ph, pw = (-h) % patch, (-w) % patch
    out = np.pad(frames, ((0, 0), (0, ph), (0, pw)), mode="constant")
    return out[..., None]


def train_encoder_path(cfg, ds, train_sel, kind: str, args) -> Dict[str, object]:
    """
    Train an image-encoder state map ``w_t = phi_theta(o_t)`` plus its basis.

    The objective is the reconstruction the GP path gets for free -- decode
    ``w_t`` through ``Psi`` and score it on the frame's valid pixels -- optionally
    with the same teacher-forced rollout term, so the two paths differ in the
    state map and nothing else.

    :param cfg: the experiment config.
    :param ds: the calendar dataset.
    :param train_sel: ``(T,)`` training mask.
    :param kind: ``"deit"`` or ``"resnet"``.
    :param args: parsed arguments.
    :return: dict with the trained model, params and the padded frame stack.
    """
    cfg.backbone.kind = kind
    model = DBWM(cfg)

    patch = cfg.backbone.deit_patch_size if kind == "deit" else 1
    imgs = _pad_to_patch(np.asarray(ds.frames, dtype=np.float32), max(patch, 1))
    coords = np.asarray(ds.coords, dtype=np.float32)
    mask_flat = ds.common_mask()
    valid_idx = np.nonzero(mask_flat)[0]
    flat = np.asarray(ds.frames, dtype=np.float32).reshape(ds.n_steps, -1)

    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.asarray(imgs[:1]), jnp.asarray(coords[:8]))
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip_norm),
        optax.adamw(cfg.training.learning_rate, weight_decay=cfg.training.weight_decay),
    )
    opt_state = optimizer.init(params)

    tau = cfg.teacher_forcing_threshold() / max(float(ds.std[0]), 1e-12)
    n_roll = cfg.training.rollout_steps
    lam_roll = cfg.training.lambda_rollout

    def loss_fn(p, obs, y_pix, pix_coords, roll_obs, roll_y):
        """Reconstruction on valid pixels, plus the teacher-forced rollout."""
        w = model.apply(p, obs, method=model.encode)
        phi = model.apply(p, pix_coords, method=model.spatial_features)
        pred = w @ phi.T
        recon = jnp.mean((pred - y_pix) ** 2)
        loss, roll_mse, forced = recon, jnp.array(0.0), jnp.array(0.0)
        if roll_obs is not None:
            w_seq = model.apply(p, roll_obs, method=model.encode)
            a_op = model.apply(p, method=model.transition)
            mses, frac = teacher_forced_rollout(
                a_op, w_seq, phi, roll_y, jnp.zeros(phi.shape[0]), tau, True
            )
            roll_mse = jnp.mean(mses)
            forced = jnp.mean(frac)
            loss = loss + lam_roll * roll_mse
        return loss, {"recon": recon, "rollout": roll_mse, "forced": forced}

    @jax.jit
    def step(p, o_state, obs, y_pix, pix_coords, roll_obs, roll_y):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, obs, y_pix, pix_coords, roll_obs, roll_y
        )
        updates, o_state = optimizer.update(grads, o_state, p)
        return optax.apply_updates(p, updates), o_state, aux

    usable = np.nonzero(train_sel & ds.observed)[0]
    obs_all = np.zeros(ds.n_steps, dtype=bool)
    obs_all[usable] = True
    windows = [
        t for t in range(ds.n_steps - n_roll) if obs_all[t : t + n_roll + 1].all()
    ] if n_roll > 0 else []

    rng = np.random.RandomState(args.seed)
    n_pix = min(2048, valid_idx.size)
    epochs = cfg.training.n_epochs
    steps_per_epoch = max(1, usable.size // 8)
    logger.info(
        "[%s] training the encoder path: %d epochs x %d steps | r=%d | image "
        "%dx%d (padded from %dx%d) | %d pixels scored per step | teacher "
        "forcing %s",
        kind, epochs, steps_per_epoch, cfg.basis.r, imgs.shape[1], imgs.shape[2],
        ds.shape[0], ds.shape[1], n_pix,
        "off" if lam_roll == 0 else "on (tau = %.4f %s)" % (
            cfg.teacher_forcing_threshold(), cfg.modality_spec().units),
    )

    last = {}
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            dates = rng.choice(usable, size=min(8, usable.size), replace=False)
            pix = rng.choice(valid_idx, size=n_pix, replace=False)
            roll_obs = roll_y = None
            if windows:
                t0 = int(rng.choice(windows))
                span = np.arange(t0, t0 + n_roll + 1)
                roll_obs = jnp.asarray(imgs[span])
                roll_y = jnp.asarray(flat[span][:, pix])
            params, opt_state, last = step(
                params, opt_state, jnp.asarray(imgs[dates]),
                jnp.asarray(flat[dates][:, pix]), jnp.asarray(coords[pix]),
                roll_obs, roll_y,
            )
        if epoch % max(cfg.training.log_every, 1) == 0 or epoch == epochs - 1:
            logger.info(
                "[%s] epoch %3d | recon MSE %.5f | rollout MSE %.5f | forced %.0f%%",
                kind, epoch, float(last["recon"]), float(last["rollout"]),
                100.0 * float(last["forced"]),
            )
    return {"model": model, "params": params, "images": imgs}


def encode_all(kind, trained, model_gp, params_gp, cfg, ds, train_sel, args):
    """
    Produce ``(weights, extractor)`` for one encoder path.

    :param kind: ``"gp"``, ``"deit"`` or ``"resnet"``.
    :param trained: output of :func:`train_encoder_path` (``None`` for ``"gp"``).
    :param model_gp: the GP-path module.
    :param params_gp: its parameters.
    :param cfg: config.
    :param ds: dataset.
    :param train_sel: training mask.
    :param args: parsed arguments.
    :return: ``(weights (T, r), extractor, aug)``.
    """
    mask_flat = ds.common_mask()
    if kind == "gp":
        model, params = model_gp, params_gp
        feats = lambda c: model.apply(params, jnp.asarray(c), method=model.features)
        sigma2 = float(model.apply(params, method=model.sigma_eps2))
    else:
        model, params = trained["model"], trained["params"]
        feats = lambda c: model.apply(
            params, jnp.asarray(c), method=model.spatial_features
        )
        sigma2 = float(model.apply(params, method=model.sigma_eps2))

    psi = np.concatenate(
        [np.asarray(feats(ds.coords[i:i + 20000]))
         for i in range(0, ds.coords.shape[0], 20000)], axis=0)
    aug = build_augmented_basis(
        psi, ds.frames, mask_flat, train_sel & ds.observed,
        n_modes=cfg.basis.eof_modes, energy=cfg.basis.eof_energy,
        ridge=sigma2, use_climatology=cfg.basis.use_climatology,
    )
    extractor = build_extractor(
        feats, ds.coords, sigma2, ds.static_mask,
        extra_features=aug.phi[:, aug.r_learned:] if aug.q_empirical else None,
        offset=aug.offset if cfg.basis.use_climatology else None,
    )
    if kind == "gp":
        gp = extractor.solve_sequence(
            ds.frames, ds.valid_mask, ds.observed, want_cov=False
        )
        return gp["weights"], extractor, aug

    # Encoder path: the state is the network's output, but it must live in the
    # SAME coordinates as the augmented basis, so the empirical block's weights
    # are solved for while the learned block's come from the encoder. Anything
    # else would compare two different decoders rather than two state maps.
    imgs = trained["images"]
    w_enc = np.concatenate(
        [np.asarray(model.apply(params, jnp.asarray(imgs[i:i + 32]),
                                method=model.encode, train=False))
         for i in range(0, imgs.shape[0], 32)], axis=0)
    if aug.q_empirical:
        resid = (
            np.asarray(ds.frames).reshape(ds.n_steps, -1)[:, mask_flat]
            - aug.offset[mask_flat]
            - w_enc @ aug.phi[mask_flat, : aug.r_learned].T
        )
        emp = aug.phi[mask_flat, aug.r_learned:]
        lam = emp.T @ emp + sigma2 * np.eye(emp.shape[1])
        w_emp = np.linalg.solve(lam, emp.T @ resid.T).T
        w_enc = np.concatenate([w_enc, w_emp], axis=1)
    return w_enc, extractor, aug


def main():
    """Train each encoder path, score it identically, and tabulate."""
    args = build_args()
    cfg = smoke_config() if args.smoke else default_config()
    cfg.data.modality = args.modality
    if args.ndvi_dir:
        cfg.data.ndvi_dir = args.ndvi_dir
    if args.weather_csv:
        cfg.weather.csv_path = args.weather_csv
    if args.r:
        cfg.basis.r = args.r
    if args.epochs:
        cfg.training.n_epochs = args.epochs
    if args.horizon:
        cfg.horizons.horizon = args.horizon
    if args.memory_order:
        cfg.memory.order = args.memory_order
    apply_model_args(cfg, args)
    cfg.sync_input_dim()
    os.makedirs(args.results_dir, exist_ok=True)

    inp = load_inputs(cfg, args, logger)
    ds, train_sel = inp.ds, inp.train_sel
    mask_flat = ds.common_mask()
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]
    scale = float(ds.std[0])

    logger.info("=" * 78)
    logger.info("Training the shared GP-path basis (used by every encoder's decoder)")
    model_gp, params_gp, _ = train_spatial_basis(
        cfg, ds.frames[train_sel], ds.valid_mask[train_sel], ds.observed[train_sel],
        ds.coords, seed=args.seed,
        rollout_steps=cfg.training.rollout_steps,
        lambda_rollout=cfg.training.lambda_rollout,
        tf_threshold=cfg.teacher_forcing_threshold(),
        norm_std=scale,
    )

    _, scorable = purge_boundary_windows(
        inp.train_idx, inp.test_idx, cfg.memory.order
    )
    origins = np.array([
        t for t in scorable[:: max(args.forecast_stride, 1)]
        if t + cfg.horizons.horizon < ds.n_steps and ds.observed[t]
    ])
    logger.info("Scoring every encoder on %d test origins.", origins.size)

    table: Dict[str, Dict] = {}
    for kind in args.encoders:
        logger.info("=" * 78)
        logger.info("ENCODER: %s", kind)
        trained = None
        if kind != "gp":
            trained = train_encoder_path(cfg, ds, train_sel, kind, args)
        w, extractor, aug = encode_all(
            kind, trained, model_gp, params_gp, cfg, ds, train_sel, args
        )

        recon = extractor.decode(w)[:, mask_flat]
        obs = ds.observed
        ss_res = np.sum((recon[obs] - flat[obs]) ** 2, axis=1)
        ss_tot = np.sum(
            (flat[obs] - flat[obs].mean(axis=1, keepdims=True)) ** 2, axis=1
        )
        r2 = float(np.median(1.0 - ss_res / np.maximum(ss_tot, 1e-30)))
        recon_rmse = float(np.sqrt(np.mean((recon[obs] - flat[obs]) ** 2)) * scale)

        sub = build_subspace(w, ds.observed & train_sel, args.subspace_energy)
        if sub.k >= sub.r:
            sub = None
        w_dyn = sub.project(w) if sub is not None else w
        op, b_p = identify_memory(
            w_dyn[train_sel], cfg.memory.order, inp.forcing[train_sel],
            cfg.memory.parameterization, cfg.dynamics.ridge_mu,
            cfg.dynamics.quiescent_threshold, cfg.memory.reduced_rank,
            valid=ds.observed[train_sel], increment=cfg.memory.increment,
        )
        op, blend = calibrate_persistence_blend(
            op, b_p, w_dyn[train_sel], inp.forcing[train_sel],
            ds.observed[train_sel], cfg.horizons.horizon,
            gain_max=cfg.memory.gain_max,
        )

        per_h = np.full(cfg.horizons.horizon, np.nan)
        for h in range(cfg.horizons.horizon):
            preds, truths = [], []
            for t in origins:
                if not ds.observed[t - cfg.memory.order + 1 : t + 1].all():
                    continue
                state = [w_dyn[t - j] for j in range(cfg.memory.order)]
                for step_i in range(h + 1):
                    nxt = np.einsum("jrs,js->r", op.blocks, np.stack(state))
                    if b_p.shape[1]:
                        nxt = nxt + b_p @ inp.forcing[t + step_i]
                    state = [nxt] + state[:-1]
                tgt = t + h + 1
                if tgt < ds.n_steps and ds.observed[tgt]:
                    wv = sub.reconstruct(state[0]) if sub is not None else state[0]
                    preds.append(extractor.decode(wv)[mask_flat])
                    truths.append(flat[tgt])
            if preds:
                per_h[h] = float(
                    np.sqrt(np.mean((np.stack(preds) - np.stack(truths)) ** 2)) * scale
                )

        table[kind] = {
            "reconstruction_r2": r2,
            "reconstruction_rmse": recon_rmse,
            "r_total": int(aug.r),
            "k_dynamics": int(sub.k) if sub is not None else int(w.shape[1]),
            "persistence_blend": blend.get("blend"),
            "max_forecast_gain": float(np.max(forecast_gain(op, cfg.horizons.horizon))),
            "pixel_rmse": per_h.tolist(),
            "pixel_rmse_h1": float(per_h[0]),
        }
        logger.info(
            "[%s] reconstruction R2 %.4f (RMSE %.4f %s) | blend s=%s | "
            "max gain %.3f | pixel RMSE %s",
            kind, r2, recon_rmse, cfg.modality_spec().units,
            table[kind]["persistence_blend"],
            table[kind]["max_forecast_gain"], np.round(per_h, 4),
        )

    logger.info("=" * 78)
    header = "{:<10} {:>10} {:>12} {:>8} {:>10}  {}".format(
        "encoder", "recon R2", "recon RMSE", "blend", "max gain", "pixel RMSE by h"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for kind, row in sorted(table.items(), key=lambda kv: kv[1]["pixel_rmse_h1"]):
        logger.info(
            "{:<10} {:>10.4f} {:>12.4f} {:>8} {:>10.3f}  {}".format(
                kind, row["reconstruction_r2"], row["reconstruction_rmse"],
                str(row["persistence_blend"]), row["max_forecast_gain"],
                np.round(row["pixel_rmse"], 4),
            )
        )
    with open(os.path.join(args.results_dir, "encoders.json"), "w") as fh:
        json.dump(table, fh, indent=2, default=float)
    logger.info("Written to %s", os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
