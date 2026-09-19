"""
Train a single DB-WM v2 model (one backbone x expansion configuration).

Usage
-----
    python -m experiments.train_dbwm --backbone resnet --expansion swiglu \
        --modality lst --epochs 50

    # CPU smoke run on synthetic data (no Drive needed):
    python -m experiments.train_dbwm --smoke

Saves the trained parameters and the closed-form-identified dynamics ``(A,B,Q)``
to ``<ckpt_dir>/<name>.pkl``.
"""
from __future__ import annotations

import os
import argparse
import pickle

import numpy as np

from dbwm.config import default_config, smoke_config
from dbwm.models import DBWM
from dbwm.data.geotiff_dataset import load_dataset
from dbwm.training.trainer import train, identify_dynamics_closed_form
from dbwm.dynamics.guarantees import (
    check_observability,
    check_controllability,
    cyclic_index,
    error_bound_report,
    empirical_error_terms,
    steady_state_error,
)
from dbwm.dynamics.identification import split_input_matrix, koopman_modes
from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.exp.train")


def build_config(args):
    """Build an :class:`ExperimentConfig` from CLI args."""
    cfg = smoke_config() if args.smoke else default_config()
    cfg.backbone.kind = args.backbone
    cfg.basis.expansion = args.expansion
    cfg.data.modality = args.modality
    if not args.smoke:
        cfg.training.n_epochs = args.epochs
        cfg.basis.r = args.r
        if args.lst_dir:
            cfg.data.lst_dir = args.lst_dir
        if args.ndvi_dir:
            cfg.data.ndvi_dir = args.ndvi_dir
        cfg.data.use_synthetic = args.synthetic
    if args.no_forcing:
        cfg.forcing.use_precip = False
        cfg.forcing.use_irrigation = False
    if args.forcing_path:
        cfg.data.forcing_path = args.forcing_path
    cfg.sync_input_dim()  # ell must agree with the channels actually requested
    cfg.name = "dbwm_{}_{}_{}".format(args.modality, args.backbone, args.expansion)
    return cfg


def report_guarantees(model, params, train_ds, a, b, q, cfg):
    """
    Post-identification certificates (Section 4). Remark 4.1 is what licenses this:
    the guarantees depend only on the resulting ``(A, Phi_X, B)``, never on how they
    were fitted -- so they are checked *after the fact*, on whatever operator came out.
    """
    import jax.numpy as jnp
    from dbwm.dynamics.transition import spectral_radius, spectral_norm

    w = model.apply(params, jnp.asarray(train_ds.frames), train=False, method=model.encode)

    logger.info("--- Section 4 guarantees ---")
    rho, nrm = float(spectral_radius(a)), float(spectral_norm(a))
    logger.info("spectral radius %.4f | spectral norm %.4f", rho, nrm)
    if nrm > 1.05 * max(rho, 1e-9):
        # rho(A) <= ||A||_2 always; a large gap means A is non-normal, so clipping
        # the radius to rho_max does NOT preclude transient growth (Theorem 4.2).
        logger.warning(
            "A is non-normal (norm/radius = %.2f): radius-clipping does not bound "
            "||A||_2, so multi-step forecasts can transiently amplify.", nrm / max(rho, 1e-9)
        )

    # Observability (Theorem 4.1): the encoder supplies the full r-vector each step,
    # so Phi_X = I_r in the visual setting (Section 2.3).
    obs = check_observability(jnp.eye(a.shape[0]), a)
    logger.info(
        "observable=%s (rank %d/%d) | shaded=%s | distinct eigenvalues=%s | cyclic index=%d",
        obs["observable"], obs["rank"], obs["required"], obs["shaded"],
        obs["distinct_eigenvalues"], cyclic_index(a),
    )

    # Theorem 4.2: eps_dyn measured UNDER the known forcing.
    b_arg = b if (cfg.dynamics.use_forcing and train_ds.forcing is not None) else None
    f_arg = jnp.asarray(train_ds.forcing) if b_arg is not None else None
    terms = empirical_error_terms(w, w, a, b_arg, f_arg)
    bound = error_bound_report(a, terms["eps_enc"], terms["eps_dyn"], cfg.inference.horizon)
    logger.info(
        "open-loop bound (H=%d): %.4f | regime: %s | eps_dyn=%.4f",
        bound["horizon"], bound["bound"], bound["regime"], bound["eps_dyn"],
    )

    # Theorem 4.3: horizon-INDEPENDENT error once the observer corrects.
    sigma2 = float(model.apply(params, method=model.sigma_eps2))
    ss = steady_state_error(a, q, sigma2)
    logger.info("steady-state observer error: tr(P_inf)=%.4f (horizon-independent)", ss["trace"])

    # Controllability (Proposition 4.3): irrigation ONLY -- rain is a disturbance.
    if b_arg is not None and train_ds.forcing_names:
        b_p, b_u = split_input_matrix(b, train_ds.forcing_names)
        logger.info("B split: B_p %s (disturbance) | B_u %s (actuator)", b_p.shape, b_u.shape)
        if b_u.shape[1] > 0:
            ctrl = check_controllability(a, b_u)
            logger.info(
                "controllable=%s (Gramian rank %d/%d). A single irrigation scalar "
                "cannot steer an r=%d state everywhere; the rank is how much IS reachable.",
                ctrl["controllable"], ctrl["rank"], ctrl["required"], a.shape[0],
            )

    modes = koopman_modes(a)
    top = int(jnp.argmax(jnp.abs(modes["eigenvalues"])))
    logger.info("dominant Koopman mode: |lambda|=%.4f", float(jnp.abs(modes["eigenvalues"][top])))


def main():
    """Parse args, train, identify dynamics, and checkpoint."""
    ap = argparse.ArgumentParser(description="Train a DB-WM v2 model.")
    ap.add_argument("--backbone", default="resnet", choices=["resnet", "deit"])
    ap.add_argument("--expansion", default="swiglu", choices=["swiglu", "rbf", "gelu"])
    ap.add_argument("--modality", default="lst", choices=["lst", "ndvi"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--r", type=int, default=512)
    ap.add_argument("--lst-dir", dest="lst_dir", default=None)
    ap.add_argument("--ndvi-dir", dest="ndvi_dir", default=None)
    ap.add_argument(
        "--forcing-path", dest="forcing_path", default=None,
        help="cached forcing .npz from experiments/preprocess_forcing.py",
    )
    ap.add_argument(
        "--no-forcing", dest="no_forcing", action="store_true",
        help="pure-temporal ablation: B_p = B_u = 0 (Section 8.1)",
    )
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke run")
    args = ap.parse_args()

    cfg = build_config(args)
    logger.info("Experiment: %s", cfg.name)

    # The forcing config MUST be passed: without it ds.forcing is None, the trainer
    # never sees u_t^raw, and Algorithm 3 returns B = 0 -- i.e. the precipitation and
    # irrigation channels would be silently ablated while still being reported.
    forcing_cfg = cfg.forcing if cfg.dynamics.use_forcing else None
    train_ds, test_ds = load_dataset(cfg.data, forcing_cfg)
    logger.info(
        "Forcing: ell=%d channels %s",
        train_ds.ell, train_ds.forcing_names or "none (pure-temporal)",
    )

    model = DBWM(cfg)
    state = train(model, train_ds, cfg)
    a, b, q = identify_dynamics_closed_form(model, state.params, train_ds, cfg)

    report_guarantees(model, state.params, train_ds, a, b, q, cfg)

    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.ckpt_dir, cfg.name + ".pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "params": state.params,
                "A": np.asarray(a),
                "B": np.asarray(b),
                "Q": np.asarray(q),
                "config": cfg.to_dict(),
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                # Needed to split B into [B_p | B_u] at inference/planning time.
                "forcing_names": train_ds.forcing_names,
                "forcing_scale": train_ds.forcing_scale,
            },
            f,
        )
    logger.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
