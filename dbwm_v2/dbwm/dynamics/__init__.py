"""
Latent dynamics: the input-affine transition, the closed-form least-squares
(finite-section Koopman) identification of ``[A B]``, and the v3 memory /
multi-horizon extension.

The v4 pipeline for NDVI runs:

1. :mod:`~dbwm.dynamics.diagnostics` -- the pivotal whiteness pretest (v3 step 0).
2. :mod:`~dbwm.dynamics.memory` -- the order-``L`` memory kernel and its companion.
3. :mod:`~dbwm.dynamics.emission` -- the rank-``m`` weather sensor ``C``.
4. :mod:`~dbwm.dynamics.multihorizon` -- the semigroup-shrunk horizon family.
"""
from dbwm.dynamics.transition import (
    LatentDynamics,
    spectral_radius,
    spectral_norm,
    clip_spectral_radius,
)
from dbwm.dynamics.identification import (
    least_squares_operator,
    joint_least_squares,
    two_stage_least_squares,
    reduced_order_operator,
    verify_proposition_4_4,
    persistence_of_excitation,
    controllability_gramian,
    process_noise_cov,
    identify,
    koopman_modes,
    split_input_matrix,
    forcing_response_maps,
)
from dbwm.dynamics.memory import (
    RealModalForm,
    MemoryOperator,
    real_modal_form,
    companion_matrix,
    selector_matrix,
    noise_injection_matrix,
    lifted_input_matrix,
    lifted_spectrum,
    spectral_radius_lifted,
    enforce_stability,
    enforce_forecast_gain,
    calibrate_persistence_blend,
    forecast_gain,
    observability_certificate,
    is_persistence,
    build_lifted_design,
    identify_memory,
    memory_profile,
)
from dbwm.dynamics.multihorizon import (
    HorizonFamily,
    iterated_family,
    build_horizon_design,
    fit_horizon_family,
    semigroup_defect,
    memory_depth_sweep,
    coverage_comparison,
)
from dbwm.dynamics.emission import (
    EmissionModel,
    fit_emission,
    information_gain,
)
from dbwm.dynamics.diagnostics import (
    one_step_residuals,
    ljung_box_univariate,
    ljung_box_hosking,
    benjamini_hochberg,
    whiteness_pretest,
    residual_autocorrelation,
)
from dbwm.dynamics.conditioning import (
    RankReport,
    weight_rank,
    log_rank_report,
    transient_amplification,
    log_transient,
)
from dbwm.dynamics.guarantees import (
    is_shaded,
    observability_matrix,
    check_observability,
    cyclic_index,
    check_controllability,
    open_loop_error_bound,
    empirical_error_terms,
    error_bound_report,
    solve_dare,
    steady_state_error,
)

__all__ = [
    "LatentDynamics",
    "spectral_radius",
    "spectral_norm",
    "clip_spectral_radius",
    "least_squares_operator",
    "joint_least_squares",
    "two_stage_least_squares",
    "reduced_order_operator",
    "verify_proposition_4_4",
    "persistence_of_excitation",
    "controllability_gramian",
    "process_noise_cov",
    "identify",
    "koopman_modes",
    "split_input_matrix",
    "forcing_response_maps",
    # v3 memory lift (Sec. 2.2.1-2.2.4).
    "RealModalForm",
    "MemoryOperator",
    "real_modal_form",
    "companion_matrix",
    "selector_matrix",
    "noise_injection_matrix",
    "lifted_input_matrix",
    "lifted_spectrum",
    "spectral_radius_lifted",
    "enforce_stability",
    "enforce_forecast_gain",
    "calibrate_persistence_blend",
    "forecast_gain",
    "observability_certificate",
    "is_persistence",
    "build_lifted_design",
    "identify_memory",
    "memory_profile",
    # v3 multi-horizon family (Sec. 2.2.3-2.2.5).
    "HorizonFamily",
    "iterated_family",
    "build_horizon_design",
    "fit_horizon_family",
    "semigroup_defect",
    "memory_depth_sweep",
    "coverage_comparison",
    # Weather emission (the rank-m daily sensor).
    "EmissionModel",
    "fit_emission",
    "information_gain",
    # v3 step-0 whiteness pretest.
    "one_step_residuals",
    "ljung_box_univariate",
    "ljung_box_hosking",
    "benjamini_hochberg",
    "whiteness_pretest",
    "residual_autocorrelation",
    # Rank / conditioning diagnostics (why rho <= 1 can still forecast badly).
    "RankReport",
    "weight_rank",
    "log_rank_report",
    "transient_amplification",
    "log_transient",
    # Section 4 structural guarantees.
    "is_shaded",
    "observability_matrix",
    "check_observability",
    "cyclic_index",
    "check_controllability",
    "open_loop_error_bound",
    "empirical_error_terms",
    "error_bound_report",
    "solve_dare",
    "steady_state_error",
]
