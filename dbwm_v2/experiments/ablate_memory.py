"""
Ablation over the memory order ``L``: how many past days does the forecast need?

This is the experiment v3 Sec. 2.2 asks for and v2 has no analogue of. The v2
model asserts a **Markov** transition on the deep-basis weights
(``w_{t+1} = A w_t + B p_t + eta_t``, v2 Sec. 2.2), which is the ``L = 1``
truncation. v3 Prop. 2.5 shows via Mori-Zwanzig that the exact projected dynamics
*cannot* be Markov -- the finite-section residual carries a memory kernel
``{A_j}_{j>=1}`` -- so ``L`` is a real modelling choice, and this script measures
what each additional day of context buys.

Why ``L`` must be selected rather than maximised
------------------------------------------------
Three separate statements bound it from opposite sides, and a run that reported
only forecast error would see none of them:

**Theorem 2.8(i) (v3).** The lifted pair ``(C, A_cal)`` is observable only if
``A_{L-1}`` is nonsingular. If the true order is ``L* < L`` then ``A_{L-1} -> 0``,
the lift becomes unobservable, and the excess lags carry no recoverable state.
Over-lagging is therefore *actively harmful*, not merely wasteful. The certificate
``s_min(A_{L-1}) / ||A_0||_2`` is reported per order and is the sharp diagnostic.

**Section 2.2.4 (v3).** An unstructured order-``L`` kernel has ``L r^2``
parameters -- at ``r = 256, L = 7`` that is 4.6e5 against ~1,200 usable daily
transitions. Structure is mandatory, so every order here is fitted under the same
parameterization (S2 by default: each Koopman mode gets its own scalar AR(``L``),
exactly ``rL`` parameters), and the comparison is between orders rather than
between parameter counts of wildly different scale.

**Definition 2.6 / Theorem 2.10 (v3).** The semigroup defect
``D_h = Theta_h - S A_cal^h`` is nonzero exactly to the extent that an order-``L``
linear Markov model is misspecified. Its decay as ``L`` grows *is* the
Mori-Zwanzig memory depth of the field -- a physical measurement with standalone
value, independent of any forecasting gain.

What is held fixed, and why that is the whole point
----------------------------------------------------
The spatial basis ``Psi``, the empirical (EOF) block, the GP posterior ``w_t``,
the POD subspace, the weather emission ``C`` and the evaluation origins are
computed **once** and shared by every order. Only the memory identification
onward varies.

That is not an optimisation, it is the experiment. v2 Theorem 4.2 bounds the
``h``-step error by ``rho^h eps_enc + ...``, so the *representation* error enters
every horizon multiplied by the operator gain; on the real record it was the
entire one-step error. Retraining ``Psi`` per order would let that term move
between cells and the resulting table would be measuring basis-training variance
under a column heading that said "memory order".

Evaluation origins are purged with ``max(L)`` for **every** order, so all orders
score the identical set of dates. Purging per order would give ``L = 2`` five more
origins than ``L = 7`` and the columns would not be comparable.

Usage
-----
    python -m experiments.ablate_memory --smoke
    python -m experiments.ablate_memory \\
        --ndvi-dir <dir> --weather-csv <csv> \\
        --orders 2 4 6 7 --r 256 --horizon 6 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from dbwm.platform import ensure_working_backend, preflight

_PREFLIGHT = preflight()

import jax.numpy as jnp  # noqa: E402

from dbwm.config import default_config, smoke_config  # noqa: E402
from dbwm.data.seasons import purge_boundary_windows  # noqa: E402
from dbwm.dynamics.conditioning import log_rank_report, weight_rank  # noqa: E402
from dbwm.dynamics.diagnostics import (  # noqa: E402
    one_step_residuals, whiteness_pretest,
)
from dbwm.dynamics.emission import fit_emission  # noqa: E402
from dbwm.dynamics.memory import (  # noqa: E402
    build_lifted_design, calibrate_persistence_blend, forecast_gain,
    identify_memory, is_persistence, observability_certificate,
    spectral_radius_lifted,
)
from dbwm.dynamics.multihorizon import fit_horizon_family  # noqa: E402
from dbwm.dynamics.subspace import build_subspace  # noqa: E402
from dbwm.evaluation import v4_plots  # noqa: E402
from dbwm.evaluation.error_budget import persistence_rmse  # noqa: E402
from dbwm.evaluation.metrics import field_error_metrics  # noqa: E402
from dbwm.gp.empirical_basis import build_augmented_basis  # noqa: E402
from dbwm.gp.state import build_extractor  # noqa: E402
from dbwm.inference.lifted_kalman import (  # noqa: E402
    LiftedSystem, rolling_forecast_multi,
)
from dbwm.log_utils import configure_logger  # noqa: E402
from dbwm.training.gp_trainer import train_spatial_basis  # noqa: E402
from experiments._v4_common import (  # noqa: E402
    add_data_args, add_model_args, apply_model_args, load_inputs,
)

logger = configure_logger(level="INFO", name="dbwm.ablate")

#: The orders the thesis reports. ``7`` is the pipeline default (one week of
#: context); ``2`` is the smallest order that is genuinely non-Markov. ``1`` is
#: accepted too and recovers the v2 model exactly, which is the reference the
#: whole memory extension is measured against.
DEFAULT_ORDERS = (2, 4, 6, 7)


def build_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Ablate the memory order L (how many past days the forecast uses)."
    )
    add_data_args(p)
    p.add_argument(
        "--orders", type=int, nargs="+", default=list(DEFAULT_ORDERS),
        help="memory orders to compare (default: 2 4 6 7). L = 1 is the v2 "
             "Markov model and is a legitimate entry -- it is the baseline the "
             "memory lift has to beat.",
    )
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None, help="H")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--parameterization", default=None,
                   choices=["s1", "s2", "s3", "unstructured"],
                   help="held FIXED across orders; see the module docstring")
    p.add_argument("--estimator", default=None,
                   choices=["shrunk", "direct", "iterated"])
    p.add_argument("--forecast-mode", default="recursive",
                   choices=["recursive", "direct", "both"])
    p.add_argument("--results-dir", default="./_v4_ablate_memory")
    p.add_argument("--forecast-stride", type=int, default=1)
    p.add_argument("--subspace-energy", type=float, default=0.999)
    p.add_argument("--max-k", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--require-accelerator", action="store_true")
    add_model_args(p)
    return p.parse_args()


def _config(args):
    """
    Build the shared config, with everything except ``L`` pinned.

    :param args: parsed arguments.
    :return: the config.
    """
    cfg = smoke_config() if args.smoke else default_config()
    cfg.data.modality = args.modality
    if args.ndvi_dir:
        cfg.data.ndvi_dir = args.ndvi_dir
    if args.lst_dir:
        cfg.data.lst_dir = args.lst_dir
    if args.weather_csv:
        cfg.weather.csv_path = args.weather_csv
    if args.r:
        cfg.basis.r = args.r
    if args.horizon:
        cfg.horizons.horizon = args.horizon
    if args.epochs:
        cfg.training.n_epochs = args.epochs
    if args.parameterization:
        cfg.memory.parameterization = args.parameterization
    if args.estimator:
        cfg.horizons.estimator = args.estimator
    for attr, field in (("archive_start", "archive_start"),
                        ("archive_end", "archive_end"),
                        ("split_date", "split_date")):
        if getattr(args, attr):
            setattr(cfg.seasons, field, getattr(args, attr))
    apply_model_args(cfg, args)
    return cfg.sync_input_dim()


def _process_noise(weights, op, b_p, forcing, valid, jitter):
    """
    Residual process-noise covariance ``Q`` (v2 Algorithm 3 step 5).

    :param weights: ``(T, k)`` training trajectory.
    :param op: the memory operator.
    :param b_p: ``(k, ell)`` input matrix.
    :param forcing: ``(T, ell)`` inputs.
    :param valid: ``(T,)`` observation validity.
    :param jitter: diagonal conditioning.
    :return: ``(k, k)`` covariance.
    """
    w_bar, targets, origins = build_lifted_design(weights, op.order, 1, valid)
    pred = np.einsum("jrs,njs->nr", op.blocks, w_bar)
    if b_p.shape[1]:
        pred = pred + forcing[origins] @ b_p.T
    resid = targets[0] - pred
    q = (resid.T @ resid) / max(resid.shape[0] - 1, 1)
    return q + jitter * np.eye(op.r)


def score_variant(fc, ds, extractor, origins, horizon, subspace, mask_flat, w_true):
    """
    Score one forecast variant in **pixel** space, per horizon.

    Pixel space rather than latent space, because a latent RMSE is not what the
    thesis reports and the two diverge when the basis is ill-conditioned. The
    metric set matches ``run_ndvi_v4`` exactly so an ablation row can be laid
    beside a full-pipeline row.

    :param fc: one variant's output from ``rolling_forecast_multi``.
    :param ds: the calendar dataset.
    :param extractor: the GP state extractor.
    :param origins: forecast origins.
    :param horizon: ``H``.
    :param subspace: the latent subspace, or ``None``.
    :param mask_flat: ``(n_pixels,)`` validity.
    :param w_true: ``(T, r)`` GP-solved weights, for the complement carry.
    :return: ``{"pixel_<metric>": (H,), "n": (H,)}``.
    """
    flat = ds.frames.reshape(ds.n_steps, -1)[:, mask_flat]
    scale = float(ds.std[0])
    keys = ("rmse", "bias", "ubrmse", "mae")
    acc = {k: np.full(horizon, np.nan) for k in keys}
    counts = np.zeros(horizon, dtype=int)
    for h in range(horizon):
        ok = np.asarray(fc["valid"], dtype=bool)[:, h]
        org = np.asarray(origins)[ok]
        tgt = org + h + 1
        keep = (tgt < ds.n_steps)
        keep[keep] &= ds.observed[tgt[keep]]
        if not keep.any():
            continue
        wv = np.asarray(fc["mean"])[ok, h][keep]
        if subspace is not None:
            wv = subspace.reconstruct_with_complement(wv, w_true[org[keep]])
        pred = extractor.decode(wv)[:, mask_flat]
        m = field_error_metrics(pred, flat[tgt[keep]], scale=scale)
        for k in keys:
            acc[k][h] = m[k]
        counts[h] = int(keep.sum())
    out = {"pixel_{}".format(k): acc[k] for k in keys}
    out["n"] = counts
    return out


def run_order(order, cfg, ds, inp, w_dyn, cov_dyn, emission, subspace, extractor,
              w, origins, mask_flat, sigma_eps2, modes):
    """
    Identify, certify and score one memory order.

    Everything upstream of the memory kernel is passed in already computed, so
    this function is exactly the part of the pipeline that ``L`` changes.

    :param order: the memory order ``L``.
    :param cfg: the shared config.
    :param ds: the calendar dataset.
    :param inp: the :class:`~experiments._v4_common.Inputs` bundle.
    :param w_dyn: ``(T, k)`` dynamics-space weights.
    :param cov_dyn: ``(T, k, k)`` per-date posterior covariances.
    :param emission: the shared weather emission.
    :param subspace: the latent subspace, or ``None``.
    :param extractor: the GP state extractor.
    :param w: ``(T, r)`` full-``r`` weights.
    :param origins: the shared evaluation origins.
    :param mask_flat: ``(n_pixels,)`` validity.
    :param sigma_eps2: observation-noise variance.
    :param modes: forecast modes to score.
    :return: one result row.
    """
    train = inp.train_sel
    obs_train = ds.observed[train]
    logger.info("=" * 78)
    logger.info("L = %d  (%d past days of context, lifted dim %d x %d)",
                order, order, order, w_dyn.shape[1])

    op, b_p = identify_memory(
        w_dyn[train], order, inp.forcing[train], cfg.memory.parameterization,
        cfg.dynamics.ridge_mu, cfg.dynamics.quiescent_threshold,
        cfg.memory.reduced_rank, valid=obs_train, increment=cfg.memory.increment,
    )
    op, blend = calibrate_persistence_blend(
        op, b_p, w_dyn[train], inp.forcing[train], obs_train,
        cfg.horizons.horizon, gain_max=cfg.memory.gain_max,
    )
    q = _process_noise(w_dyn[train], op, b_p, inp.forcing[train], obs_train,
                       cfg.dynamics.process_noise_jitter)
    cert = observability_certificate(op, np.eye(op.r))
    gains = forecast_gain(op, cfg.horizons.horizon)
    family = fit_horizon_family(
        w_dyn[train], op, b_p, cfg.horizons.horizon, inp.forcing[train],
        obs_train, cfg.horizons.estimator, cfg.horizons.ridge_mu,
        nu_grid=cfg.horizons.nu_grid, nu_selection=cfg.horizons.nu_selection, q=q,
    )

    system = LiftedSystem(
        op=op, b_p=b_p, q=q,
        c_w=emission.c if emission.usable else None,
        r_w=emission.r_cov if emission.usable else None,
        gamma_dyn=cfg.inference.gamma_dyn_inflation,
    )
    variants = [{"name": m, "mode": m, "use_weather": True} for m in modes]
    forecasts = rolling_forecast_multi(
        system, w_dyn, ds.observed, origins, cfg.horizons.horizon, variants,
        inp.forcing, inp.measurement, cov_dyn, sigma_eps2, family=family,
    )
    scored = {
        name: score_variant(fc, ds, extractor, origins, cfg.horizons.horizon,
                            subspace, mask_flat, w)
        for name, fc in forecasts.items()
    }

    primary = scored[modes[0]]
    row: Dict[str, object] = {
        "order": int(order),
        "r": int(op.r),
        "blend": None if blend.get("blend") is None else float(blend["blend"]),
        "is_persistence": bool(is_persistence(op)),
        "rho_lifted": float(spectral_radius_lifted(op)),
        "max_forecast_gain": float(np.max(gains)),
        "norm_budget": float(op.norm_budget()),
        "a_last_relative_smin": float(cert["a_last_relative_smin"]),
        "a0_relative_smin": float(cert["a0_relative_smin"]),
        "observable": bool(cert["observable"]),
        "detectable": bool(cert["detectable"]),
        "observability_cause": cert.get("cause"),
        "defect": np.asarray(family.defect).tolist(),
        "defect_normalized": np.asarray(family.defect_normalized).tolist(),
        "nu": np.asarray(family.nu).tolist(),
        "n_parameters": int(_parameter_count(op, cfg.memory.reduced_rank)),
        "variants": {
            k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            for k, v in scored.items()
        },
    }
    # Kept as arrays for the plotting layer; the JSON copy above is the record.
    for key in ("pixel_ubrmse", "pixel_mae", "pixel_rmse", "pixel_bias"):
        row[key] = np.asarray(primary[key])

    logger.info(
        "L = %-2d | s = %-5s | rho %.4f | max gain %.3f | "
        "s_min(A_{L-1})/||A_0|| = %.2e (%s)",
        order, "n/a" if row["blend"] is None else "{:.2f}".format(row["blend"]),
        row["rho_lifted"], row["max_forecast_gain"],
        row["a_last_relative_smin"],
        "observable" if cert["observable"] else "NOT observable: " + str(cert["cause"]),
    )
    fmt = lambda a: np.array2string(np.asarray(a), precision=4, suppress_small=True)
    logger.info("L = %-2d | ubRMSE %s", order, fmt(primary["pixel_ubrmse"]))
    logger.info("L = %-2d | MAE    %s", order, fmt(primary["pixel_mae"]))
    logger.info("L = %-2d | normalised ||D_h|| %s", order,
                fmt(family.defect_normalized))
    if is_persistence(op):
        logger.warning(
            "L = %d was shrunk all the way to persistence (s = 0): the memory "
            "kernel did not beat 'predict today's field' on held-out data. Its "
            "Thm 2.8(i) failure is then vacuous -- A_{L-1} = 0 by construction, "
            "not by over-lagging -- and its row measures the anchor, not the "
            "order.", order,
        )
    return row


def _parameter_count(op, reduced_rank: Optional[int] = None) -> int:
    """
    Free parameters in the memory kernel, under its parameterization.

    Reported because v3 Sec. 2.2.4's argument is about *counts*: an unstructured
    order-7 kernel at ``r = 256`` is ``L r^2 = 4.6e5`` parameters against ~1,200
    usable transitions, while S2 is ``rL = 1,792``. A table of scores without the
    counts beside them invites reading a structured win as an order effect.

    The counts follow the structures as v3 states them:

    =============  ============================================================
    S1             ``r^2 + L`` -- one operator plus a lag profile shared by all
                   modes (``A_j = alpha_j A``)
    S2             ``r L`` -- each mode's own scalar AR(``L``). A real modal
                   block contributes ``L`` reals; a conjugate pair contributes
                   ``2L`` for the *pair*, so the total is ``rL`` either way.
    S3             ``r^2 + q L r`` -- unconstrained ``A_0`` plus a rank-``q``
                   memory block
    unstructured   ``L r^2``
    =============  ============================================================

    :param op: the memory operator.
    :param reduced_rank: the rank ``q`` used by S3, if applicable.
    :return: the number of free parameters.
    """
    r, order = op.r, op.order
    key = (op.parameterization or "unstructured").lower()
    if key == "s1":
        return int(r * r + order)
    if key == "s2" and op.modal is not None:
        return int(sum(op.modal.block_sizes) * order)
    if key == "s3":
        q = int(min(reduced_rank or r, r))
        return int(r * r + q * order * r)
    return int(order * r * r)


def main():
    """Share every upstream stage, then vary only the memory order."""
    args = build_args()
    orders = sorted({int(o) for o in args.orders if int(o) >= 1})
    if not orders:
        raise SystemExit("--orders must contain at least one order >= 1")
    cfg = _config(args)
    os.makedirs(args.results_dir, exist_ok=True)

    backend = ensure_working_backend(allow_cpu_fallback=not args.require_accelerator)
    logger.info("Backend: %s | orders %s | horizon %d",
                backend["backend"], orders, cfg.horizons.horizon)

    inp = load_inputs(cfg, args, logger)
    ds, train = inp.ds, inp.train_sel

    # ---- Shared: basis, empirical block, GP posterior, subspace ---------- #
    logger.info("Training the shared spatial basis once (r = %d, %d epochs).",
                cfg.basis.r, cfg.training.n_epochs)
    model, params, _ = train_spatial_basis(
        cfg, ds.frames[train], ds.valid_mask[train], ds.observed[train],
        ds.coords, seed=args.seed,
        rollout_steps=cfg.training.rollout_steps,
        lambda_rollout=cfg.training.lambda_rollout,
        tf_threshold=cfg.teacher_forcing_threshold(),
        use_teacher_forcing=cfg.training.teacher_forcing,
        norm_std=float(ds.std[0]),
    )
    sigma_eps2 = float(model.apply(params, method=model.sigma_eps2))
    mask_flat = ds.common_mask()
    psi = np.concatenate(
        [np.asarray(model.apply(params, jnp.asarray(ds.coords[i:i + 20000]),
                                method=model.features))
         for i in range(0, ds.coords.shape[0], 20000)], axis=0)
    aug = build_augmented_basis(
        psi, ds.frames, mask_flat, train & ds.observed,
        n_modes=cfg.basis.eof_modes, energy=cfg.basis.eof_energy,
        ridge=sigma_eps2, use_climatology=cfg.basis.use_climatology,
        select=cfg.basis.eof_select,
        holdout_fraction=cfg.basis.eof_holdout_fraction,
    )
    extractor = build_extractor(
        lambda c: model.apply(params, jnp.asarray(c), method=model.features),
        ds.coords, sigma_eps2, ds.static_mask,
        extra_features=aug.phi[:, aug.r_learned:] if aug.q_empirical else None,
        offset=aug.offset if cfg.basis.use_climatology else None,
    )
    gp = extractor.solve_sequence(ds.frames, ds.valid_mask, ds.observed, want_cov=True)
    w, w_cov = gp["weights"], gp["covariances"]
    log_rank_report(weight_rank(w, ds.observed & train))

    subspace = None
    if args.subspace_energy and args.subspace_energy > 0:
        subspace = build_subspace(w, ds.observed & train, args.subspace_energy,
                                  args.max_k)
        if subspace.k >= subspace.r:
            subspace = None
    w_dyn = subspace.project(w) if subspace is not None else w
    cov_dyn = subspace.project_covariance(w_cov) if subspace is not None else w_cov

    # ---- Shared: the whiteness pretest (v3 step 0, defined at L = 1) ----- #
    op1, b1 = identify_memory(
        w_dyn[train], 1, inp.forcing[train], "unstructured",
        cfg.dynamics.ridge_mu, valid=ds.observed[train],
    )
    pretest = whiteness_pretest(
        one_step_residuals(w_dyn[train], op1.blocks[0], b1, inp.forcing[train]),
        cfg.memory.ljung_box_lags, cfg.memory.ljung_box_alpha,
        cfg.memory.ljung_box_projection, args.seed,
    )
    logger.info(
        "Whiteness pretest on the L=1 innovations: %s. %s",
        pretest["verdict"],
        "Rejected -- Prop. 2.5 bites, so an L > 1 kernel is empirically motivated."
        if pretest["reject"] else
        "NOT rejected -- per Cor. 2.5.1 the memory kernel has nothing left to "
        "explain on this record and every row below is fitting noise. This is the "
        "pivotal experiment and it came out negative; report it as such.",
    )

    # ---- Shared: weather emission (acts on the current block only) ------- #
    obs_train = train & ds.observed
    emission = fit_emission(
        w_dyn[obs_train], inp.measurement[obs_train],
        list(inp.weather.measurement_names), cfg.weather.emission_ridge,
        cfg.weather.full_noise_covariance, scale=inp.weather.measurement_scale,
    )

    # ---- Shared: evaluation origins, purged with the LARGEST order ------- #
    # Every order must score the identical dates. Purging per order would hand
    # L = 2 five more origins than L = 7, and the columns would then differ by
    # their evaluation set as well as by their model.
    _, scorable = purge_boundary_windows(inp.train_idx, inp.test_idx, max(orders))
    origins = np.array([
        t for t in scorable[:: max(args.forecast_stride, 1)]
        if t + cfg.horizons.horizon < ds.n_steps and ds.observed[t]
    ])
    if origins.size == 0:
        raise SystemExit("No scorable test origins for L = {}.".format(max(orders)))
    logger.info(
        "Shared evaluation set: %d origins (%s .. %s), purged with L = %d so "
        "every order is scored on identical dates.",
        origins.size, ds.dates[int(origins[0])], ds.dates[int(origins[-1])],
        max(orders),
    )
    pers = persistence_rmse(
        ds.frames, mask_flat, ds.observed, origins, cfg.horizons.horizon,
        float(ds.std[0]),
    )
    logger.info("persistence RMSE by horizon (the bar to beat): %s",
                np.array2string(pers, precision=4))

    modes = (["recursive", "direct"] if args.forecast_mode == "both"
             else [args.forecast_mode])

    rows: List[Dict[str, object]] = []
    for order in orders:
        try:
            rows.append(run_order(
                order, cfg, ds, inp, w_dyn, cov_dyn, emission, subspace,
                extractor, w, origins, mask_flat, sigma_eps2, modes,
            ))
        except Exception as exc:  # pragma: no cover - a singular fit must not
            logger.exception("L = %d failed: %s", order, exc)  # kill the sweep
            rows.append({"order": int(order), "error": str(exc)})

    _report(rows, pers, orders, cfg.modality_spec().units)
    _write(args.results_dir, cfg, rows, pers, origins, ds, pretest, orders)
    if not args.no_plots:
        fig = v4_plots.plot_memory_ablation(
            rows, os.path.join(args.results_dir, "memory_ablation.png"),
            metric="pixel_ubrmse", persistence=pers,
            units=cfg.modality_spec().units,
        )
        if fig:
            logger.info("Figure written to %s", os.path.abspath(fig))
    logger.info("Done. Results in %s", os.path.abspath(args.results_dir))


def _report(rows, persistence, orders, units: str = "NDVI") -> None:
    """
    Print the ablation table, then say what it licenses and what it does not.

    :param rows: per-order result rows.
    :param units: modality unit symbol, for the table heading.
    :param persistence: ``(H,)`` persistence RMSE.
    :param orders: the orders attempted.
    """
    ok = [r for r in rows if "error" not in r]
    if not ok:
        logger.error("Every order failed; nothing to report.")
        return
    horizon = len(np.asarray(ok[0]["pixel_ubrmse"]))
    logger.info("=" * 78)
    logger.info(
        "MEMORY-ORDER ABLATION -- pixel ubRMSE by horizon (%s)", units
    )
    header = "  L   params    s    gain   " + "  ".join(
        "t+{}".format(h + 1).rjust(7) for h in range(horizon)
    )
    logger.info("%s", header)
    for r in sorted(ok, key=lambda d: d["order"]):
        logger.info(
            "%3d %8d  %5s  %5.2f   %s",
            r["order"], r["n_parameters"],
            "n/a" if r["blend"] is None else "{:.2f}".format(r["blend"]),
            r["max_forecast_gain"],
            "  ".join("{:7.4f}".format(v) for v in np.asarray(r["pixel_ubrmse"])),
        )
    logger.info(
        "%3s %8s  %5s  %5.2f   %s", "--", "", "", 1.00,
        "  ".join("{:7.4f}".format(v) for v in np.asarray(persistence)),
    )
    logger.info("      (last row: persistence, which has gain exactly 1)")

    best = min(ok, key=lambda r: float(np.nanmean(np.asarray(r["pixel_ubrmse"]))))
    beats = [
        r for r in ok
        if np.nanmean(np.asarray(r["pixel_ubrmse"])) < np.nanmean(persistence)
    ]
    logger.info("-" * 78)
    logger.info(
        "Best horizon-mean ubRMSE: L = %d at %.4f. %d of %d orders beat "
        "persistence (%.4f).",
        best["order"], float(np.nanmean(np.asarray(best["pixel_ubrmse"]))),
        len(beats), len(ok), float(np.nanmean(persistence)),
    )
    # Thm 2.8(i) can fail for two reasons that look identical in
    # a_last_relative_smin and demand OPPOSITE fixes. Reporting them together
    # would send the reader to reduce L when the problem is r, or the reverse.
    over = [r["order"] for r in ok
            if not r["observable"] and not r["is_persistence"]
            and r.get("observability_cause") == "over_lagged"]
    rank_def = [r["order"] for r in ok
                if not r["observable"] and not r["is_persistence"]
                and r.get("observability_cause") == "a0_rank_deficient"]
    if over:
        logger.warning(
            "Thm 2.8(i) fails by OVER-LAGGING at L = %s: A_{L-1} is effectively "
            "singular while A_0 is well conditioned, so the lift is unobservable "
            "and the deepest lags carry no recoverable state. Those orders "
            "over-shoot the record's memory depth regardless of their score -- "
            "this is the ablation's own verdict on how deep the memory goes.",
            over,
        )
    if rank_def:
        logger.warning(
            "Thm 2.8(i) fails at L = %s because A_0 is ITSELF near-singular, "
            "NOT because those orders over-lag -- it would fail at L = 1 too. "
            "r = %d exceeds the number of directions the weight trajectory "
            "excites, so the ridge fit is singular in the unexcited ones (v2 "
            "Prop. 4.4: ker W_-^T). The fix is a smaller r or a tighter "
            "--subspace-energy; reducing L will not help, and the memory-order "
            "column of this table cannot be read as a memory-depth measurement "
            "until it is fixed.",
            rank_def, ok[0].get("r", 0) or 0,
        )
    shrunk = [r["order"] for r in ok if r["is_persistence"]]
    if shrunk:
        logger.warning(
            "L = %s were shrunk to persistence on held-out data (s = 0). Their "
            "rows measure the shrinkage anchor, not the memory order: read them "
            "as 'the kernel did not earn its place', not as 'this order is "
            "good'.", shrunk,
        )
    if len(ok) > 1:
        spread = (
            max(float(np.nanmean(np.asarray(r["pixel_ubrmse"]))) for r in ok)
            - min(float(np.nanmean(np.asarray(r["pixel_ubrmse"]))) for r in ok)
        )
        logger.info(
            "Spread across orders: %.4f %s (%.1f%% of the best score). A spread "
            "below the run-to-run noise of the basis fit means the record does "
            "not resolve the memory order, which is itself the finding.",
            spread, units,
            100.0 * spread / max(float(np.nanmean(np.asarray(best["pixel_ubrmse"]))), 1e-12),
        )
    logger.info("=" * 78)


def _write(results_dir, cfg, rows, persistence, origins, ds, pretest, orders) -> str:
    """
    Persist the ablation as JSON.

    :param results_dir: output directory.
    :param cfg: the shared config.
    :param rows: per-order rows.
    :param persistence: ``(H,)`` persistence RMSE.
    :param origins: the shared evaluation origins.
    :param ds: the calendar dataset.
    :param pretest: the whiteness pretest result.
    :param orders: the orders attempted.
    :return: the path written.
    """
    payload = {
        "config": cfg.to_dict(),
        "orders": list(orders),
        "held_fixed": [
            "spatial basis Psi", "empirical (EOF) block", "GP posterior w_t",
            "POD subspace", "weather emission C/R", "evaluation origins",
            "parameterization ({})".format(cfg.memory.parameterization),
        ],
        "origins": {
            "n": int(origins.size),
            "first": str(ds.dates[int(origins[0])]),
            "last": str(ds.dates[int(origins[-1])]),
            "purged_with_order": int(max(orders)),
        },
        "whiteness_pretest": {
            "reject": bool(pretest["reject"]),
            "verdict": pretest["verdict"],
            "hosking_p": pretest["hosking"]["p_value"],
        },
        "persistence_rmse_by_horizon": np.asarray(persistence).tolist(),
        "rows": [
            {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items()}
            for r in rows
        ],
    }
    path = os.path.join(results_dir, "memory_ablation.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


if __name__ == "__main__":
    main()
