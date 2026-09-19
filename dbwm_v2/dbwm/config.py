"""
Configuration dataclasses for the DB-WM v2 pipeline.

All tunable knobs of the framework live here so that experiment scripts and
tests can construct a single :class:`ExperimentConfig` and pass it around.

The design mirrors ``DB_WM_v2_framework.md`` Section 8.1 (Satellite Imagery)
for the LST/NDVI application:

* backbone ablations  -> ``BackboneConfig.kind in {"resnet", "deit"}``
* expansion variants  -> ``BasisConfig.expansion in {"swiglu", "rbf"}``
* input-affine forcing -> ``w_{t+1} = A w_t + B_p p_t + B_u u_t`` (Remark 2.1;
  the action-conditioned ``A(a_t)`` form is deliberately rejected). Setting
  ``DynamicsConfig.use_forcing = False`` recovers the pure-temporal special case
  ``B_p = B_u = 0``.

LST and NDVI are trained as **separate models** (separate runs / checkpoints)
selected by ``DataConfig.modality``; each builds its own forcing matrix against
its own acquisition-date grid (the two sensors' dates generally differ).

Google Drive paths (Colab-mounted) are the defaults for the data sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# Forcing (precipitation p_t + irrigation u_t  ->  Upsilon, and hence B=[B_p B_u])
# --------------------------------------------------------------------------- #
@dataclass
class ForcingConfig:
    """
    Configuration for the exogenous/control inputs ``u_t^raw = [p_t, u_t]``.

    * **Precipitation ``p_t``** (uncontrollable *exogenous disturbance*): supplied
      as GeoTIFF rasters (CHIRPS / GPM IMERG / ERA5). They are geometrically
      aligned pixel-by-pixel onto the LST/NDVI reference grid, accumulated (mm)
      over the window between consecutive acquisition dates, then reduced to
      ``n_precip_zones`` scalars (1 = AOI mean; >1 = zonal means).
    * **Irrigation ``u_t``** (controllable *actuator*): supplied as a time series
      (CSV), resampled onto the same acquisition-date grid, giving
      ``n_irrig_zones`` scalars (applied depth mm / valve 0-1 / canal discharge).

    The resulting regressor is ``Upsilon in R^{ell x (T-1)}`` with
    ``ell = n_precip_zones * (1 + precip_lags) + n_irrig_zones``, and
    ``B = [B_p B_u] in R^{r x ell}`` is *learned* by Algorithm 3 Stage II:
    ``[B_p B_u] = dW Upsilon^T (Upsilon Upsilon^T + mu I_ell)^{-1}``.
    """

    use_precip: bool = True
    use_irrigation: bool = True

    # --- Precipitation rasters ---
    precip_dir: str = "/content/drive/MyDrive/Precipitation_v3"
    # Resampling for the coarse (CHIRPS 0.05 deg / IMERG 0.1 deg) -> 30 m warp.
    # "bilinear" smoothly downscales the coarse field; "nearest" preserves the
    # native coarse blocks; "average" is area-weighted (use when *upscaling*).
    precip_resampling: str = "bilinear"
    # Number of spatial zones for p_t: 1 -> single AOI-mean scalar (J_p = 1).
    n_precip_zones: int = 1
    # Extra lagged precipitation channels (wet-soil memory). 1 -> append p_{t-1}.
    precip_lags: int = 1
    # Accumulate rain over the window (prev_date, date]. If an acquisition gap is
    # longer than this cap (days), the window is truncated (guards huge gaps).
    max_accum_days: int = 32

    # --- Irrigation time series ---
    irrigation_csv: str = "/content/drive/MyDrive/Irrigation_v3/irrigation_schedule.csv"
    # Column holding the date, and the column(s) holding the magnitude(s).
    irrigation_date_col: str = "date"
    irrigation_value_cols: Tuple[str, ...] = ("irrigation_mm",)
    # How to map the daily/irregular irrigation series onto the acquisition grid:
    # "sum" (accumulated depth over the step) or "mean" (average rate).
    irrigation_aggregation: str = "sum"

    # --- Conditioning ---
    # Scale each forcing channel by its training std. NOTE: scale-ONLY, never
    # mean-centred -- centring would move p_t = 0 off zero and destroy the
    # "quiescent transition" (p_t = u_t = 0) detection that Stage I relies on.
    scale_forcing: bool = True

    def ell(self) -> int:
        """Return the forcing dimension ``ell`` implied by this config."""
        n = 0
        if self.use_precip:
            n += self.n_precip_zones * (1 + self.precip_lags)
        if self.use_irrigation:
            n += len(self.irrigation_value_cols)
        return n


# --------------------------------------------------------------------------- #
# Weather (v4): precipitation -> input B_p ; Rs/Ta/VPD -> measurement C
# --------------------------------------------------------------------------- #
@dataclass
class WeatherConfig:
    """
    Point weather from ``collect_historical_weather.py``, split by *role*.

    The two roles are structurally different and must never be mixed:

    * **Precipitation** is an *input*. It enters the predict step through ``B_p``
      (v2 Sec. 2.2) because rain drives the vegetation state.
    * **Rs / Ta / VPD** are *measurements*. They enter the Kalman update through
      the emission ``y_t^w = C w_t + d(doy_t) + nu_t`` with ``C in R^{m x r}``,
      because they are observed quantities correlated with the state, not
      actuators of it.

    A channel that is simultaneously a known input and a measurement makes the
    innovation correlated with the input, which costs the filter its
    minimum-variance property and invalidates the calibration claims of v3
    Proposition 2.13. Hence the strict split.

    The AOI (~3.75 x 4.05 km) sits inside a single ERA5/IFS cell, so each channel
    is one scalar per day applied uniformly across the field -- registration, not
    rainfall downscaling.
    """

    #: Daily weather CSV. Column names are resolved through
    #: ``dbwm.data.weather.COLUMN_ALIASES``, so both the Open-Meteo collector
    #: schema (``precip_mm``/``ta_mean_c``/...) and the Sayedanwala archive schema
    #: (``Precip_mm``/``Ta_C``/``Sw_rad_mj_m2``/``VPD_kpa``, ``Date`` as M/D/YYYY)
    #: load unchanged. Config always refers to CANONICAL names.
    csv_path: str = (
        "/content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P"
        "/sayedanwala_historical_weather_2022_2026.csv"
    )

    # --- Input block (precipitation) ---
    #: Lagged precipitation channels (wet-soil memory). ell = 1 + precip_lags.
    precip_lags: int = 1
    #: Scale-ONLY (never centre): centring moves p_t = 0 off zero and destroys the
    #: quiescent-transition detection that Algorithm 3 Stage I relies on.
    scale_forcing: bool = True

    # --- Measurement block (Rs, Ta, VPD) ---
    #: Row order of ``C`` and ``R``, in canonical names. The Sayedanwala archive
    #: supplies one daily value per channel, so the ``ta_max_c`` / ``vpd_max_kpa``
    #: alternates are unavailable there; they resolve only if the file has them.
    measurement_cols: Tuple[str, ...] = (
        "swrad_mj_m2", "ta_mean_c", "vpd_mean_kpa",
    )
    #: Annual harmonics in the day-of-year climatology ``d(doy)`` carried as a
    #: known offset in the emission equation. Ta and Rs are near-perfect annual
    #: sinusoids, and so is NDVI's dominant autonomous Koopman mode; leaving the
    #: cycle in the raw channel is the collinearity failure of v2 Remark 6.1.
    n_harmonics: int = 2
    #: Fit a full m x m measurement-noise covariance R from the emission residuals.
    #: Rs/Ta/VPD residuals are strongly cross-correlated, so a diagonal R misstates
    #: the noise model and gives a non-optimal Kalman gain. The direction of that
    #: error is not fixed: under strong positive correlation the correct full-R
    #: filter extracts MORE information, because differencing correlated channels
    #: cancels the shared disturbance.
    full_noise_covariance: bool = True
    #: Ridge for the least-squares fit of C.
    emission_ridge: float = 1e-3

    def ell(self) -> int:
        """Input dimension ``ell`` implied by this config."""
        return 1 + self.precip_lags

    def n_measurements(self) -> int:
        """Measurement dimension ``m`` (rows of ``C``)."""
        return len(self.measurement_cols)


# --------------------------------------------------------------------------- #
# Seasons (Kharif / Rabi / Zaid) and the train/test protocol
# --------------------------------------------------------------------------- #
@dataclass
class SeasonConfig:
    """
    Cropping-season calendar and the chronological split.

    Kharif 1 Jun-30 Sep, Rabi 1 Oct-end Feb (next year), Zaid 1 Mar-31 May.

    The default cut realises 3.00 Kharif / 3.39 Rabi / 3.49 Zaid in training over
    the 2022-01-01..2026-04-30 archive -- the thesis target of "~3 / 3.5 / 3.5" to
    within 0.11 of a season -- while keeping the test set strictly in the future.
    A season-blocked split would balance the seasons exactly but, with the memory
    lift, lets training windows straddle held-out blocks; that is leakage.
    """

    #: First and last date of the modelling window (inclusive).
    archive_start: str = "2022-01-01"
    archive_end: str = "2026-04-30"
    #: First date of the TEST split; train is [archive_start, split_date).
    split_date: str = "2025-04-15"
    #: Intended training season counts, checked at load time.
    target_kharif: float = 3.0
    target_rabi: float = 3.5
    target_zaid: float = 3.5
    #: Allowed deviation per season, in units of seasons.
    target_tolerance: float = 0.15
    #: Emit the per-season spatiotemporal statistics after evaluation.
    report_statistics: bool = True


# --------------------------------------------------------------------------- #
# Memory lift (v3 Def. 2.3 / 2.4)
# --------------------------------------------------------------------------- #
@dataclass
class MemoryConfig:
    """
    Memory-augmented dynamics ``w_{t+1} = sum_{j=0}^{L-1} A_j w_{t-j} + B_p p_t``.

    The lifted state is ``w_bar_t = [w_t; ...; w_{t-L+1}] in R^{Lr}`` with the
    block-companion realisation ``A_cal`` (v3 Def. 2.4). ``L = 1`` recovers v2.

    **Structure is mandatory, not optional** (v3 Sec. 2.2.4): unstructured memory
    at ``L=7, r=256`` is 4.6e5 parameters against ~1200 training transitions, and
    the design matrix is rank-deficient by construction.

    **Over-lagging is not free** (v3 Thm 2.8): if the true order is ``L* < L`` then
    ``A_{L-1} = 0`` and the lifted pair is *unobservable* (still detectable -- the
    offending modes sit at ``lambda = 0``). The order must be selected by the
    ``||D_h||``-vs-``L`` sweep, never maximised.
    """

    #: Memory order L. L=7 gives the requested (x_t, x_{t-1}, ..., x_{t-6}).
    order: int = 7
    #: Structured parameterization (v3 Sec. 2.2.4).
    #: "s2" per-mode scalar AR(L) in the Koopman eigenbasis (rL params) -- default;
    #: "s1" scalar lag weighting A_j = alpha_j A (r^2 + L);
    #: "s3" reduced-rank memory A_j = U C_j V^T (O(Lqr));
    #: "unstructured" for tests / small r only.
    parameterization: str = "s2"
    #: Rank q for the "s3" parameterization.
    reduced_rank: int = 32
    #: Stability is enforced by sum_j ||A_j||_2 <= rho_max (v3 Lemma 2.6
    #: consequence). Clipping sigma(A_0) is the WRONG operation once L > 1: the
    #: lifted spectrum is the root set of the matrix polynomial, not sigma(A_0).
    enforce_stability: bool = True
    #: Sweep L in [1, order] and report ||D_h||(L) -- the Mori-Zwanzig memory depth.
    sweep_memory_order: bool = True
    #: Ljung-Box lags for the v3 step-0 whiteness pretest on the L=1 residuals.
    ljung_box_lags: int = 12
    #: Significance level for the pretest.
    ljung_box_alpha: float = 0.05
    #: Random projection dim for the multivariate portmanteau statistic (r is far
    #: too large for a full r^2-per-lag statistic at T ~ 1200).
    ljung_box_projection: int = 16
    #: Fit the increment w_{t+1} - w_t and add I back to A_0. Same model class,
    #: but the ridge then shrinks toward PERSISTENCE instead of toward zero. On a
    #: daily NDVI record, shrinking toward zero shrinks toward a prediction worse
    #: than doing nothing, which is what forced the large blocks behind the
    #: measured ||A_cal|| = 4.13.
    increment: bool = True
    #: Which stability quantity to enforce.
    #: "forecast_gain" -- max_h ||S A_cal^h||_2 <= gain_max. This is the factor by
    #:   which an h-step forecast amplifies the encoding error, i.e. what v2
    #:   Thm 4.2's rho^T actually stands for. Persistence sits at exactly 1.
    #: "spectrum" -- rho(A_cal) <= rho_max (v3 Lemma 2.6). Correct for asymptotics,
    #:   but it bounds nothing at h = 1..6 when the operator is non-normal: the
    #:   real record had rho = 1.000 with gains of 4.13 and rising.
    stability_metric: str = "forecast_gain"
    #: HARD rejection threshold on max_h ||S A_cal^h||_2 during blend selection.
    #: Not a target: the blend weight itself is chosen by held-out h-step error
    #: (v3 Thm 2.12), and this only rules out violently non-normal candidates
    #: that could win a short holdout by luck. 1.0 exactly would admit nothing
    #: but persistence, since persistence already sits at gain 1.
    gain_max: float = 2.0


# --------------------------------------------------------------------------- #
# Multi-horizon prediction (v3 Def. 2.5 / Sec. 2.2.5)
# --------------------------------------------------------------------------- #
@dataclass
class HorizonConfig:
    """
    Direct horizon family ``w_{t+h} = Theta_h w_bar_t + Toeplitz(B^(h)) p + eps^(h)``.

    Identified in closed form by semigroup-shrunk ridge (v3 Sec. 2.2.5)::

        G_h = (W_{+h} Z^T + nu_h [S A^h, 0]) (Z Z^T + mu I + nu_h Pi)^{-1}

    ``nu_h -> inf`` recovers the pure iterated predictor, ``nu_h = 0`` the pure
    direct one. Theorem 2.12 proves the shrunk estimator *strictly dominates both*
    at every finite T with nonzero finite semigroup defect, which is why "shrunk"
    is the default rather than a compromise.
    """

    #: Maximum horizon H. 6 gives the requested t+1 .. t+6.
    horizon: int = 6
    #: Estimator: "shrunk" (default, Thm 2.12), "direct" (nu=0), "iterated" (nu=inf).
    estimator: str = "shrunk"
    #: Ridge mu on the whole G_h block.
    ridge_mu: float = 1e-3
    #: Shrinkage nu_h. None -> selected per horizon by the criterion below.
    nu: Optional[Tuple[float, ...]] = None
    #: "gcv" or "innovation_likelihood", evaluated on a held-out tail of TRAIN.
    nu_selection: str = "gcv"
    #: Candidate nu values (log-spaced) for the selection sweep.
    nu_grid: Tuple[float, ...] = (
        0.0, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, float("inf"),
    )
    #: Fraction of the training split held out for nu selection.
    nu_holdout_fraction: float = 0.15
    #: Constrain Theta_h to the same per-mode structure as the S2 memory. With
    #: r=256, L=7 an unstructured Theta_h is 459k parameters per horizon against
    #: ~1200 samples; the shrinkage alone keeps it defined but not well determined.
    structured: bool = True
    #: Estimate Sigma_h from the DIRECT h-step residuals (v3 step 6). Never build
    #: it by propagating Q: Prop. 2.13 shows the propagated version under-covers by
    #: exactly the bias term b_h Cov(w_bar) b_h^T.
    direct_residual_covariance: bool = True
    #: Report the semigroup defect ||D_h||_F = ||Theta_h - S A^h||_F.
    report_defect: bool = True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Configuration for the LST / NDVI GeoTIFF datasets."""

    # Google Drive folder roots (Colab: drive.mount('/content/drive')).
    drive_root: str = "/content/drive/MyDrive"
    lst_dir: str = "/content/drive/MyDrive/LST_Downscaled_v3"
    ndvi_dir: str = "/content/drive/MyDrive/NDVI_Downscaled_30m/NDVI_downscaled_30m"

    # Which modality to train/evaluate on for a given run. LST and NDVI are
    # SEPARATE models (separate checkpoints), each with its own date grid.
    #
    # The modality also selects the physical conventions in
    # ``dbwm.data.modality``: units (index vs kelvin), display range, colour ramp,
    # teacher-forcing threshold and the raster validity gate. Those are not
    # presentation details -- an NDVI teacher-forcing threshold of 0.02 read as
    # 0.02 K forces every rollout step forever, and an NDVI [0, 1] colour range
    # saturates every pixel of a 310 K field.
    modality: str = "ndvi"  # {"lst", "ndvi"}

    # Mask pixels falling outside the modality's PLAUSIBILITY range
    # (``ModalitySpec.physical_range``; None for NDVI, 250-350 K for LST). This is
    # deliberately NOT the display range: set to the latter it discarded 8.4% of
    # the real LST archive, because 47 degC bare soil is ordinary data and an
    # unremarkable colour at the same time. Set False to ablate the gate and see
    # what the extrapolated pixels do to the normalisation statistics.
    apply_physical_range: bool = True

    # Override the modality's plausibility range, e.g. (240.0, 360.0) for a
    # product that legitimately ranges wider. None keeps the modality default.
    valid_range_override: Optional[Tuple[float, float]] = None

    # Treat a raster that exists but carries no usable pixels as an UNOBSERVED
    # date. Defaults to the modality's setting (on for LST, off for NDVI); set
    # False to make an empty frame an error rather than a skipped date.
    drop_empty_frames: Optional[bool] = None

    # How the latent weight w_t is obtained from a frame:
    #   "gp"      -- w_t = Lambda_{X_t}^{-1} Phi_{X_t}^T y_t over the VALID pixels
    #                of that date (Intuition doc Sec. 3/5). Handles the ~48% valid
    #                mask natively by dropping invalid rows, and makes the model
    #                literally a GP. This is the v4 default.
    #   "encoder" -- w_t = phi_theta(o_t) via the ResNet/DeiT image backbone (v2).
    #                Retained as an ablation; it must zero-fill invalid pixels
    #                before the CNN sees them.
    state_path: str = "gp"  # {"gp", "encoder"}

    # Where the exogenous inputs come from:
    #   "weather_csv" -- point Open-Meteo record (v4 default; see WeatherConfig)
    #   "rasters"     -- legacy gridded precipitation + irrigation CSV (ForcingConfig)
    forcing_source: str = "weather_csv"  # {"weather_csv", "rasters"}

    # Reindex the frame stack onto a complete daily calendar, marking dates with
    # no GeoTIFF as unobserved. The memory lift assumes w_{t-1}..w_{t-L+1} are
    # CONSECUTIVE days; compressing to observed dates would make a "lag-6"
    # sometimes span 7-8 real days and bias the memory-depth measurement.
    calendar_reindex: bool = True

    # Optional precomputed forcing table (written by experiments/preprocess_forcing.py).
    # If set and present, it is loaded directly instead of re-warping the rasters.
    forcing_path: str | None = None

    # Spatial grid. The Sayedanwala AOI tiles are natively 135 x 125 at 30 m
    # (EPSG:32643), so the default is the NATIVE size: resampling up to 256^2
    # would invent 3.7x more pixels than the sensor recorded. Set explicitly to
    # resample; None means "use whatever the first tile has".
    image_height: Optional[int] = 135
    image_width: Optional[int] = 125
    n_channels: int = 1  # single-band LST or NDVI

    # Temporal split: 85% train / 15% test, chronological (no shuffle) so the
    # test set is strictly future relative to training (honest forecasting).
    train_fraction: float = 0.85
    chronological_split: bool = True

    # Normalisation. Values are computed from the training split if None.
    normalize: bool = True
    nodata_value: float = -9999.0

    # Set True to synthesise a small spatiotemporal dataset in memory (used by
    # tests and for smoke-running the pipeline without the real Drive data).
    use_synthetic: bool = False
    synthetic_n_frames: int = 60


# --------------------------------------------------------------------------- #
# Backbone (Stage 1: g_theta : R^{C x H x W} -> R^h)
# --------------------------------------------------------------------------- #
@dataclass
class BackboneConfig:
    """Configuration for the image backbone g_theta."""

    kind: str = "resnet"  # {"resnet", "deit"}
    hidden_dim: int = 256  # h in the framework (Section 8.1: h = 256)

    # ResNet specifics.
    resnet_stage_sizes: Tuple[int, ...] = (2, 2, 2, 2)  # ResNet-18
    resnet_width: int = 64

    # DeiT / ViT specifics.
    deit_patch_size: int = 16
    deit_depth: int = 6
    deit_num_heads: int = 6
    deit_embed_dim: int = 192  # DeiT-Tiny embed dim
    deit_mlp_ratio: float = 4.0
    deit_use_distill_token: bool = True


# --------------------------------------------------------------------------- #
# Basis / Deep Basis Kernel (Stage 2: expand : R^h -> R^r)
# --------------------------------------------------------------------------- #
@dataclass
class BasisConfig:
    """Configuration for the deep basis map phi_theta."""

    # Number of basis functions. Section 8.1 suggests r in [512, 1024] for a
    # 4,500 km^2 scene, but the Sayedanwala AOI has only ~8,167 valid pixels and
    # ~1,200 training transitions: r=512 with L=7 puts the lifted state at 3,584
    # dimensions, which the data cannot determine. r=256 keeps r << n_pixels and
    # keeps the lifted state at 1,792.
    r: int = 256
    expansion: str = "swiglu"  # {"swiglu", "rbf", "gelu"}

    # Spatial DBK basis Psi : R^2 -> R^r used as the GP / decoder side
    # (reconstructs the field via f_t(x) = <w_t, Psi(x)>). Defines the kernel
    # k(x, x') = <Psi(x), Psi(x')>.
    spatial_hidden_dim: int = 256
    spatial_n_layers: int = 2
    fourier_features: int = 128  # random Fourier positional encoding for coords
    fourier_scale: float = 10.0

    # RBF expansion: number of inducing points (= r) and base kernel lengthscale.
    rbf_lengthscale: float = 1.0

    # Observation noise sigma_eps^2 (initial value; learnable).
    sigma_eps2_init: float = 1e-2

    # --- Empirical completion of the basis (dbwm/gp/empirical_basis.py) --- #
    # A smooth coordinate MLP cannot represent the sharp plot boundaries of an
    # agricultural mosaic; on the real record it plateaued at reconstruction
    # R^2 = 0.958, and v2 Thm 4.2 multiplies that residual by the operator gain,
    # so it -- not the dynamics -- set the observed forecast floor. These two
    # blocks close it: a static per-pixel climatology, and EOFs of exactly what
    # Psi cannot express. Both are fitted on TRAINING dates only.
    use_climatology: bool = True
    eof_modes: int = 96          # cap on q; 0 disables the empirical block
    eof_energy: float = 0.999    # fraction of the RESIDUAL variance to retain
    #: How q is chosen. "holdout" scores a chronological tail of training the
    #: modes were not fitted to; "energy" uses eof_energy and is monotone in q,
    #: so it can only ever be stopped by the eof_modes cap.
    eof_select: str = "holdout"
    eof_holdout_fraction: float = 0.2


# --------------------------------------------------------------------------- #
# Dynamics (Section 2.2)
# --------------------------------------------------------------------------- #
@dataclass
class DynamicsConfig:
    """
    Configuration for the input-affine latent dynamics (Section 2.2, Remark 2.1)

        w_{t+1} = A w_t + B u_t^raw + eta_t,   B = [B_p B_u] in R^{r x ell}.

    ``A`` is a single, input-INDEPENDENT operator; the action-conditioned form
    ``A(a_t)`` is rejected (Remark 2.1) because it destroys the autonomous/forcing
    disambiguation and the well-defined Koopman spectrum.
    """

    # Forcing. Set False to recover the pure-temporal special case B_p = B_u = 0.
    use_forcing: bool = True
    # ell: forcing dimension. Keep in sync with ForcingConfig.ell(); the loaders
    # set this automatically from the data actually found.
    input_dim: int = 3  # default: [p_t, p_{t-1}, u_t]

    # Identification regime (Algorithm 3). Regularized matrix least squares --
    # NOT a DMD: the deep basis already performed the reduction to R^r, so the
    # truncating SVD of a projected DMD is redundant (Proposition 4.4).
    identification: str = "two_stage"  # {"two_stage", "joint"}
    ridge_mu: float = 1e-3
    quiescent_threshold: float = 1e-8  # |u_raw| below this == quiescent step

    # Spectral safety (L_spec / Algorithm 3 step 4). rho_max = 1 for the
    # near-conservative LST thermal field.
    rho_max: float = 1.0
    clip_eigenvalues: bool = True  # closed-form eigenvalue clipping to rho_max

    # Process-noise conditioning jitter nu (Algorithm 3 step 5).
    process_noise_jitter: float = 1e-6


# --------------------------------------------------------------------------- #
# Training (Algorithm 1 + teacher forcing spec)
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """Configuration for the dPPGP + dynamics training loop."""

    n_epochs: int = 50
    batch_size: int = 8  # b: number of trajectory segments per mini-batch
    segment_length: int = 16  # T: length of each rolled-out trajectory segment
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 0

    # Loss weights (Definition 3.1).
    lambda_dynamics: float = 1.0  # lambda_2
    lambda_dppgp: float = 1.0  # lambda_1
    lambda_spectral: float = 1.0
    lambda_consistency: float = 0.1  # ties encoder output to GP-solved weights
    alpha_trace: float = 1.0  # trace regulariser weight
    beta_kl: float = 1.0  # KL weight (scaled by 1/n internally)

    # Teacher forcing (user spec, see DB_WM_v2_framework discussion + v3 Sec. 8).
    # During a rollout, if the one-step loss exceeds ``tf_threshold`` the
    # ground-truth weight is fed to the dynamics for the next step instead of
    # the model's own prediction (scheduled sampling / teacher forcing).
    teacher_forcing: bool = True
    tf_threshold: float = 0.01
    # --- v4 GP-path teacher forcing (Remark 6.1 schedule (b)) --------------- #
    # The v4 basis is trained by dPPGP on single frames, so by default nothing is
    # rolled out and there is nothing to force. Setting rollout_steps > 0 adds a
    # K-step rollout of the latent state through a trainable operator, decoded to
    # pixels and scored against the true frames; whenever that decoded RMSE
    # exceeds tf_threshold_ndvi the GROUND TRUTH state is fed forward instead of
    # the prediction. This is what makes Psi dynamics-aware rather than merely
    # reconstruction-optimal.
    rollout_steps: int = 3
    lambda_rollout: float = 1.0
    #: Threshold in PHYSICAL units (NDVI), converted internally by the frame
    #: normalisation so 0.02 always means 0.02 NDVI.
    #:
    #: Modality-specific, and the LST value is NOT a rescaling of this one -- see
    #: ``tf_threshold_lst``. Resolve through
    #: :meth:`ExperimentConfig.teacher_forcing_threshold` rather than reading
    #: either field directly.
    tf_threshold_ndvi: float = 0.02
    #: The same threshold for LST, in **kelvin**. 0.5 K: the accuracy a downscaled
    #: thermal product is typically quoted at, and about the same fraction of this
    #: field's spatial standard deviation (~1.7 K) that 0.02 is of NDVI's (~0.055),
    #: which is what keeps the *fraction of forced steps* comparable between the
    #: two runs. Leaving the NDVI number in place for LST would mean 0.02 K --
    #: a fortieth of the field's own variability -- so every step would be forced
    #: forever while the training log still reported teacher forcing as "on".
    tf_threshold_lst: float = 0.5

    # Number of pixels sampled per frame for the dPPGP pixel-likelihood term
    # (keeps the per-iteration cost O(b * r^2), never materialising n x r).
    dppgp_pixel_samples: int = 2048

    # Checkpointing.
    ckpt_dir: str = "/content/drive/MyDrive/LST_NDVI_Thesis/checkpoints"
    log_every: int = 10


# --------------------------------------------------------------------------- #
# Inference (Algorithm 2: visual basis observer)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Configuration for the Kalman-filter inference / forecasting."""

    horizon: int = 7  # open-loop forecast horizon H (days)
    use_kalman_update: bool = True  # closed-loop correction when obs available
    gamma_dyn_inflation: float = 0.1  # Gamma_dyn covariance inflation factor
    results_dir: str = "/content/drive/MyDrive/LST_NDVI_Thesis/results"


# --------------------------------------------------------------------------- #
# Planning (Algorithm 2, PLAN block: CEM over irrigation)
# --------------------------------------------------------------------------- #
@dataclass
class PlanningConfig:
    """
    Configuration for CEM planning over the *controllable* irrigation channel.

    Only relevant when irrigation is a **decision** rather than an observed record.
    In the retrospective LST/NDVI study irrigation is observed (it comes from canal
    records), so identification uses it as a regressor and this config is unused;
    it is what turns the trained DB-WM into a controller (Section 7.1, item 4).

    Precipitation is never a decision variable -- it is an uncontrollable
    disturbance entering the rollout as a known forecast (Proposition 4.3).
    """

    horizon: int = 7  # H: planning horizon (steps)
    cadence: int = 1  # K: execute the first K actions, then replan
    n_candidates: int = 300  # CEM population (Section 8.3: 300)
    n_iters: int = 30  # CEM iterations (Section 8.3: 30)
    n_elite: int = 30  # elite set size

    # Cost weights: C = sum_k gamma^k [||w - w_g||^2 + beta_plan tr(P) + lambda_u u^2]
    gamma: float = 0.95  # temporal discount
    beta_plan: float = 0.1  # uncertainty penalty (control-independent; Thm 4.3)
    lambda_u: float = 0.01  # water-use penalty

    # Admissible set U_{t+k} = {0} if rain else [0, u_max]  (mm applied per step).
    u_max: float = 40.0

    # CEM sampling.
    sigma_min: float = 1e-3  # variance floor (prevents premature collapse)
    alpha_smooth: float = 0.1  # damping of the (mu, sigma) update
    gamma_dyn_inflation: float = 0.1  # covariance inflation during rollout
    seed: int = 0


# --------------------------------------------------------------------------- #
# Top-level experiment config
# --------------------------------------------------------------------------- #
#: Run names that :meth:`ExperimentConfig.apply_modality_defaults` is allowed to
#: rewrite, i.e. the ones nobody chose deliberately.
_DEFAULT_NAMES = frozenset({"dbwm_ndvi_gp_swiglu", "dbwm_v4", "dbwm_smoke"})


@dataclass
class ExperimentConfig:
    """Bundles every sub-config for a single experiment run."""

    data: DataConfig = field(default_factory=DataConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    seasons: SeasonConfig = field(default_factory=SeasonConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    horizons: HorizonConfig = field(default_factory=HorizonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)

    name: str = "dbwm_ndvi_gp_swiglu"

    def sync_input_dim(self) -> "ExperimentConfig":
        """
        Make ``DynamicsConfig.input_dim`` (ell) agree with the input channels
        actually requested, and disable forcing entirely if ell == 0.

        Which config supplies ``ell`` depends on the forcing source: the v4 point
        weather path (``DataConfig.forcing_source == "weather_csv"``) takes it from
        :class:`WeatherConfig`, while the legacy precipitation-raster path takes it
        from :class:`ForcingConfig`.

        :return: ``self`` (for chaining).
        """
        ell = (
            self.weather.ell()
            if self.data.forcing_source == "weather_csv"
            else self.forcing.ell()
        )
        if ell == 0:
            self.dynamics.use_forcing = False
            self.dynamics.input_dim = 1  # placeholder; B is never used
        else:
            self.dynamics.input_dim = ell
        return self

    def apply_modality_defaults(self) -> "ExperimentConfig":
        """
        Rename the run when the modality changed but the name did not.

        ``name`` seeds every artefact path -- the checkpoint stem, the figure
        filenames, the summary key. An LST run left on the NDVI default writes
        ``dbwm_ndvi_gp_swiglu_v4.pkl`` containing an LST model, and the two
        modalities then overwrite each other in a shared ``--ckpt-dir``. Only the
        untouched default is rewritten, so an explicit ``--name`` always wins.

        :return: ``self`` (for chaining).
        """
        if self.name in _DEFAULT_NAMES and self.data.modality != "ndvi":
            self.name = "dbwm_{}_gp_swiglu".format(self.data.modality)
        return self

    def modality_spec(self):
        """
        The physical conventions of this run's modality.

        :return: the :class:`~dbwm.data.modality.ModalitySpec`.
        """
        from dbwm.data.modality import get_modality

        return get_modality(self.data.modality)

    def data_dir(self) -> str:
        """
        The raster folder for this run's modality.

        Single source of truth for ``cfg.data.ndvi_dir if modality == "ndvi" else
        cfg.data.lst_dir``, which was written out by hand in six entry points.

        :return: the directory path.
        """
        return (
            self.data.ndvi_dir if self.data.modality == "ndvi" else self.data.lst_dir
        )

    def teacher_forcing_threshold(self) -> float:
        """
        Scheduled-sampling threshold in this modality's **physical** units.

        The trainer divides it by the frame normalisation, so "forced whenever the
        decoded rollout is worse than this" is literally true in NDVI index units
        or in kelvin as appropriate.

        :return: the threshold.
        """
        return (
            self.training.tf_threshold_ndvi
            if self.data.modality == "ndvi"
            else self.training.tf_threshold_lst
        )

    def physical_range(self):
        """
        The raster **plausibility** gate for this modality, or ``None``.

        Not the display range (``ModalitySpec.vlim``). A value outside the display
        range clips to the end colour; a value outside this one is discarded as
        not-data, which is a far stronger claim and needs a far wider bound.

        :return: ``(lo, hi)`` in physical units, or ``None`` when disabled or when
            the modality declares no range.
        """
        if not self.data.apply_physical_range:
            return None
        if self.data.valid_range_override is not None:
            lo, hi = self.data.valid_range_override
            return (float(lo), float(hi))
        return self.modality_spec().physical_range

    def drop_empty_frames(self) -> bool:
        """
        Whether an all-no-data raster counts as an unobserved date.

        :return: the config override if set, else the modality default.
        """
        if self.data.drop_empty_frames is not None:
            return bool(self.data.drop_empty_frames)
        return bool(self.modality_spec().drop_empty_frames)

    def lifted_dim(self) -> int:
        """Dimension ``L * r`` of the memory-lifted state ``w_bar_t``."""
        return self.memory.order * self.basis.r

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict view of the config (for logging / JSON dump)."""
        return asdict(self)


def default_config() -> ExperimentConfig:
    """Return the default DB-WM v2 experiment configuration."""
    return ExperimentConfig().sync_input_dim()


def smoke_config() -> ExperimentConfig:
    """
    Return a tiny configuration for tests and CPU smoke runs.

    Small images, small r, synthetic data (including synthetic sparse forcing so
    the B_p / B_u identification path is exercised), and a couple of epochs so
    the whole pipeline executes in seconds.
    """
    cfg = ExperimentConfig(name="dbwm_smoke")
    cfg.data.use_synthetic = True
    cfg.data.synthetic_n_frames = 240  # >= L + H + enough transitions to identify
    cfg.data.image_height = 32
    cfg.data.image_width = 32
    cfg.basis.r = 32
    # Small but still >1 so the memory/multi-horizon code paths are exercised.
    cfg.memory.order = 3
    cfg.memory.sweep_memory_order = False
    cfg.memory.ljung_box_projection = 4
    cfg.horizons.horizon = 3
    cfg.horizons.nu_grid = (0.0, 1.0, 1e2, float("inf"))
    cfg.basis.fourier_features = 16
    cfg.basis.spatial_hidden_dim = 32
    cfg.backbone.hidden_dim = 32
    cfg.backbone.resnet_width = 8
    cfg.backbone.deit_embed_dim = 32
    cfg.backbone.deit_depth = 2
    cfg.backbone.deit_num_heads = 2
    cfg.backbone.deit_patch_size = 8
    cfg.training.n_epochs = 2
    cfg.training.batch_size = 2
    cfg.training.segment_length = 6
    cfg.training.dppgp_pixel_samples = 128
    cfg.inference.horizon = 3
    # Tiny CEM so the planning path is exercised in seconds.
    cfg.planning.horizon = 4
    cfg.planning.n_candidates = 32
    cfg.planning.n_iters = 3
    cfg.planning.n_elite = 8
    # Synthetic weather: precipitation + 1 lag -> ell = 2; Rs/Ta/VPD -> m = 3.
    cfg.weather.precip_lags = 1
    cfg.forcing.n_precip_zones = 1
    cfg.forcing.precip_lags = 1
    cfg.dynamics.use_forcing = True
    # Local output paths so smoke runs don't touch the Colab Drive mount.
    cfg.training.ckpt_dir = "./_smoke_out/checkpoints"
    cfg.inference.results_dir = "./_smoke_out/results"
    return cfg.sync_input_dim()
