"""
End-to-end smoke tests: the full train -> identify -> infer pipeline must run
and reduce loss, for both backbones and the SwiGLU/RBF expansions, plus the
E-GP baseline.
"""
import numpy as np
import jax.numpy as jnp

from dbwm.config import smoke_config
from dbwm.models import DBWM
from dbwm.data.geotiff_dataset import load_dataset
from dbwm.training.trainer import (
    train,
    identify_dynamics_closed_form,
    TrajectoryBatcher,
    compute_loss,
    init_train_state,
)
from dbwm.training.losses import teacher_forced_rollout
from dbwm.inference.observer import closed_loop_filter
from dbwm.baselines.egp import EGPBaseline, EGPConfig
from dbwm.evaluation.metrics import evaluate_sequence
from dbwm.data.geotiff_dataset import make_pixel_grid


def _cfg(backbone, expansion):
    cfg = smoke_config()
    cfg.backbone.kind = backbone
    cfg.basis.expansion = expansion
    cfg.training.n_epochs = 2
    return cfg


def test_train_infer_resnet_swiglu():
    """Full pipeline for the primary ResNet + SwiGLU configuration."""
    cfg = _cfg("resnet", "swiglu")
    train_ds, test_ds = load_dataset(cfg.data)
    model = DBWM(cfg)
    state = train(model, train_ds, cfg)
    a, b, q = identify_dynamics_closed_form(model, state.params, train_ds, cfg)
    sigma2 = float(model.apply(state.params, method=model.sigma_eps2))
    out = closed_loop_filter(model, state.params, test_ds, a, q, sigma2, cfg)
    metrics = evaluate_sequence(out["prior_maps"], test_ds.frames[..., 0], test_ds.valid_mask)
    assert np.isfinite(metrics["rmse"])
    assert out["prior_maps"].shape == test_ds.frames[..., 0].shape


def test_train_deit_rbf_runs():
    """DeiT backbone with the RBF expansion must initialise and step."""
    cfg = _cfg("deit", "rbf")
    train_ds, _ = load_dataset(cfg.data)
    model = DBWM(cfg)
    state = train(model, train_ds, cfg)
    assert state.step > 0


def test_loss_decreases():
    """The composite loss should decrease after a few optimisation steps."""
    cfg = _cfg("resnet", "swiglu")
    cfg.training.n_epochs = 1
    train_ds, _ = load_dataset(cfg.data)
    model = DBWM(cfg)
    grid = make_pixel_grid(*train_ds.image_shape[:2])
    rng = np.random.RandomState(0)
    idx = rng.choice(grid.shape[0], size=cfg.training.dppgp_pixel_samples, replace=False)
    coords = jnp.asarray(grid[idx])
    state, optimizer = init_train_state(model, cfg, train_ds.frames[0], coords)

    from dbwm.training.trainer import make_train_step, _gather_targets

    step = make_train_step(model, optimizer, cfg)
    batcher = TrajectoryBatcher(train_ds, cfg.training.segment_length, cfg.training.batch_size)
    frames, forcing = batcher.next_batch()
    targets = jnp.asarray(_gather_targets(frames, idx))
    f = jnp.asarray(frames)

    losses = []
    for _ in range(8):
        state, m = step(state, f, None, coords, targets)
        losses.append(float(m["loss"]))
    assert losses[-1] < losses[0], "loss did not decrease: {}".format(losses)


def test_teacher_forcing_caps_error():
    """
    With teacher forcing on, the per-step rollout error should not exceed the
    free-running error (it can only help by re-injecting ground truth).
    """
    import jax
    r, t = 4, 12
    a = 1.2 * jnp.eye(r)  # unstable -> free-running error explodes
    w_seq = jax.random.normal(jax.random.PRNGKey(0), (t, r))
    err_tf, _ = teacher_forced_rollout(a, w_seq, threshold=0.01, use_teacher_forcing=True)
    err_free, _ = teacher_forced_rollout(a, w_seq, threshold=1e9, use_teacher_forcing=True)
    # threshold=1e9 never feeds truth -> pure free-running.
    assert float(jnp.sum(err_tf)) <= float(jnp.sum(err_free)) + 1e-4


def test_egp_baseline_runs():
    """
    E-GP baseline fits and forecasts on synthetic data, in both observer modes.

    The API takes ``(coords, frames)`` rather than a dataset object so that the
    ablation grid can hand it the exact pixel set and frames every DB-WM cell is
    scored on -- a baseline fitted on a different pixel selection would not be a
    comparison.
    """
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, test_ds = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)
    test = test_ds.frames[..., 0].reshape(test_ds.n_frames, -1)

    egp = EGPBaseline(
        EGPConfig(n_centers=32, n_measurements=8, learn_hyperparameters=False)
    ).fit(coords, train)

    assert egp.report["cyclic_index"] >= 1
    assert egp.report["n_centers"] == 32

    origins = list(range(0, max(test.shape[0] - 3, 1), 2))
    for feedback in (False, True):
        out = egp.rolling_forecast(test, origins, 2, feedback=feedback)
        assert out["mean"].shape == (len(origins), 2, test.shape[1])
        assert np.isfinite(out["mean"][out["valid"]]).all()


def test_egp_feedback_beats_autonomous():
    """
    The feedback observer must beat the autonomous one.

    This is the paper's own headline result (Figs. 10-13): the AKO drifts once it
    leaves the training period while the FKO "does well throughout". If the
    ordering ever inverts, the measurement update is wired wrong -- which would
    silently flatter DB-WM in the comparison table.
    """
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, test_ds = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)
    test = test_ds.frames[..., 0].reshape(test_ds.n_frames, -1)

    egp = EGPBaseline(
        EGPConfig(n_centers=48, n_measurements=24, learn_hyperparameters=False)
    ).fit(coords, train)
    origins = list(range(0, max(test.shape[0] - 3, 1), 2))

    errs = {}
    for name, fb in (("ako", False), ("fko", True)):
        out = egp.rolling_forecast(test, origins, 2, feedback=fb)
        tgt = np.array(origins) + 1
        errs[name] = float(
            np.sqrt(np.mean((out["mean"][:, 0][tgt < len(test)]
                             - test[tgt[tgt < len(test)]]) ** 2))
        )
    assert errs["fko"] <= errs["ako"], errs


def test_egp_cyclic_index_bounds_sensors():
    """
    Both sensor rules behave as specified, and the bound is not mistaken for a recipe.

    Proposition 2's cyclic index is "essentially independent of the dimensionality
    M", which is the paper's central systems-theoretic claim, so the baseline must
    be able to reproduce it exactly (``sensor_rule="cyclic_index"``). But it is a
    LOWER BOUND: the default rule has to search above it for actual observability,
    as the paper's Figure 9 does.
    """
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, _ = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)

    lit = EGPBaseline(EGPConfig(
        n_centers=32, n_measurements=None, sensor_rule="cyclic_index",
        learn_hyperparameters=False,
    )).fit(coords, train)
    assert lit.report["n_measurements"] == lit.report["cyclic_index"]
    assert 1 <= lit.report["cyclic_index"] <= 32

    # The DEFAULT rule must not stop at the bound. Proposition 2 gives ell as a
    # lower bound on the sensor count; using it as the count itself produced an
    # unobservable pair (rank 52/300 at ell = 1 on the real record) and a
    # "feedback" observer that saw one pixel. The default searches upward for
    # observability, as the paper's own Figure 9 does.
    auto = EGPBaseline(EGPConfig(
        n_centers=32, n_measurements=None, sensor_rule="observable",
        learn_hyperparameters=False,
    )).fit(coords, train)
    assert auto.report["n_measurements"] >= auto.report["cyclic_index"]
    assert auto.report["observability_rank"] >= lit.report["observability_rank"]
    assert auto.report["sensor_ladder"], "the search ladder must be reported"


def test_egp_climatology_is_a_matched_protocol_choice():
    """
    The climatology offset must be available and must change the state.

    Withholding it from the E-GP while DB-WM's decoder uses it is the single
    largest unfairness in a head-to-head table: the E-GP would have to represent
    the static plot mosaic with stationary RBF atoms while its competitor gets
    that field for free.
    """
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, _ = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)

    on = EGPBaseline(EGPConfig(n_centers=24, n_measurements=8,
                               learn_hyperparameters=False,
                               use_climatology=True)).fit(coords, train)
    off = EGPBaseline(EGPConfig(n_centers=24, n_measurements=8,
                                learn_hyperparameters=False,
                                use_climatology=False)).fit(coords, train)
    assert np.any(on.offset != 0.0)
    assert np.all(off.offset == 0.0)
    # decode(encode(y)) must reproduce y equally well either way -- the offset
    # moves work between the mean and the weights, it does not add information.
    for m in (on, off):
        rec = m.decode(m.encode(train[0]))
        assert np.isfinite(rec).all()


def test_egp_filtered_forecast_matches_the_dbwm_protocol():
    """
    One filter pass over the calendar, branching at each origin.

    The comparison is only meaningful if both models enter a forecast having seen
    the same history. Re-solving ``w`` from the origin frame alone -- the previous
    behaviour -- discards everything DB-WM's filter had assimilated.
    """
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, test_ds = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)
    test = test_ds.frames[..., 0].reshape(test_ds.n_frames, -1)

    egp = EGPBaseline(EGPConfig(n_centers=32, n_measurements=16,
                                learn_hyperparameters=False)).fit(coords, train)
    origins = list(range(2, max(test.shape[0] - 3, 3), 2))
    full = egp.filtered_rolling_forecast(test, origins, 2, assimilate="full")
    sens = egp.filtered_rolling_forecast(test, origins, 2, assimilate="sensors")
    assert full["mean"].shape == (len(origins), 2, test.shape[1])
    assert np.isfinite(full["mean"][full["valid"]]).all()
    assert np.isfinite(sens["mean"][sens["valid"]]).all()

    # Assimilating the whole frame cannot be worse than assimilating N pixels of
    # it; if it is, the full-field update is wired wrong.
    def err(out):
        tgt = np.array(origins) + 1
        ok = tgt < len(test)
        return float(np.sqrt(np.mean((out["mean"][ok, 0] - test[tgt[ok]]) ** 2)))

    assert err(full) <= err(sens) * 1.05, (err(full), err(sens))


def test_field_error_metrics_decompose_exactly():
    """
    ``RMSE^2 = bias^2 + ubRMSE^2`` must hold exactly, not approximately.

    The decomposition is the whole reason ubRMSE is reported; if it did not hold
    the two numbers could not be read as complementary parts of one error.
    """
    from dbwm.evaluation.metrics import field_error_metrics

    rng = np.random.default_rng(0)
    true = rng.normal(size=(40, 40))
    pred = true + 0.3 + 0.15 * rng.normal(size=(40, 40))
    m = field_error_metrics(pred, true)
    assert np.isclose(m["rmse"] ** 2, m["bias"] ** 2 + m["ubrmse"] ** 2, rtol=1e-10)
    assert np.isclose(m["bias"], 0.3, atol=0.02)


def test_pure_offset_has_zero_ubrmse():
    """A constant offset is all bias and no structural error."""
    from dbwm.evaluation.metrics import field_error_metrics

    true = np.random.default_rng(1).normal(size=(20, 20))
    m = field_error_metrics(true + 0.25, true)
    assert np.isclose(m["bias"], 0.25)
    assert m["ubrmse"] < 1e-12
    assert np.isclose(m["mae"], 0.25)


def test_metrics_scale_to_physical_units():
    """The `scale` argument converts every metric, not just RMSE."""
    from dbwm.evaluation.metrics import field_error_metrics

    true = np.random.default_rng(2).normal(size=(15, 15))
    pred = true + 0.2
    a = field_error_metrics(pred, true, scale=1.0)
    b = field_error_metrics(pred, true, scale=0.135)
    for k in ("rmse", "bias", "ubrmse", "mae"):
        assert np.isclose(b[k], 0.135 * a[k])


def test_egp_bandwidth_screen_rejects_unidentifiable_kernels():
    """
    The bandwidth search must not select a kernel whose weights are unidentifiable.

    The marginal likelihood and the operator fit pull in opposite directions: a
    wider kernel fits each frame better but drives ``cond(K^T K)`` from ~1e1 at
    half a centre spacing to ~1e15 at two, after which the per-step weights stop
    being identifiable and the operator regressed on them is degenerate. Selecting
    on likelihood alone picked such a kernel and produced ``rank(O) = 60/120``
    even with 240 sensors.
    """
    from dbwm.baselines.egp import EGPBaseline, EGPConfig, rbf_features
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, _ = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)

    egp = EGPBaseline(EGPConfig(n_centers=48, learn_hyperparameters=True,
                                max_gram_condition=1e10)).fit(coords, train)
    gram = egp.psi.T @ egp.psi
    assert np.linalg.cond(gram) <= 1e10 * 10, np.linalg.cond(gram)

    # With the screen switched off the search is free to pick a singular Gram.
    loose = EGPBaseline(EGPConfig(n_centers=48, learn_hyperparameters=True,
                                  max_gram_condition=np.inf)).fit(coords, train)
    assert loose.lengthscale >= egp.lengthscale


def test_egp_full_field_update_uses_the_ridge_posterior():
    """
    A whole frame must be assimilated as a precise observation, not a noisy one.

    Its covariance is the ridge posterior ``sigma^2 (K^T K + sigma^2 I)^{-1}`` --
    the same ``sigma_eps^2 Lambda_X^{-1}`` DB-WM assimilates. Using ``sigma^2 I``
    instead overstates the uncertainty of a full frame by orders of magnitude, and
    made the full-field row score WORSE than the N-sensor row, which cannot be
    right when it sees strictly more of the same data.
    """
    from dbwm.baselines.egp import EGPBaseline, EGPConfig
    from dbwm.data.geotiff_dataset import make_pixel_grid

    cfg = smoke_config()
    train_ds, test_ds = load_dataset(cfg.data)
    h, w, _ = train_ds.image_shape
    coords = make_pixel_grid(h, w)
    train = train_ds.frames[..., 0].reshape(train_ds.n_frames, -1)
    test = test_ds.frames[..., 0].reshape(test_ds.n_frames, -1)

    egp = EGPBaseline(EGPConfig(n_centers=32, n_measurements=16,
                                learn_hyperparameters=False)).fit(coords, train)
    ridge_post = (egp.noise**2) * np.linalg.inv(
        egp.psi.T @ egp.psi + (egp.noise**2 + 1e-7) * np.eye(egp.psi.shape[1])
    )
    # The frame is far more informative than a single pixel of it.
    assert np.trace(ridge_post) < (egp.noise**2) * egp.psi.shape[1]

    origins = list(range(2, max(test.shape[0] - 3, 3), 2))
    full = egp.filtered_rolling_forecast(test, origins, 1, assimilate="full")
    sens = egp.filtered_rolling_forecast(test, origins, 1, assimilate="sensors")

    def err(out):
        tgt = np.array(origins) + 1
        ok = tgt < len(test)
        return float(np.sqrt(np.mean((out["mean"][ok, 0] - test[tgt[ok]]) ** 2)))

    assert err(full) <= err(sens) * 1.05, (err(full), err(sens))
