"""
Deprecated module: the identification is no longer described as a DMD/DMDc.

Section 2.5 and Proposition 4.4 of the framework establish that, because the deep
basis ``phi_theta`` has already reduced the state to ``R^r``, the truncating SVD of
a projected DMD is **redundant** -- the reduced-order operator is recovered exactly
by regularized matrix least squares. The identification is therefore stated as a
least-squares / finite-section (Galerkin) Koopman fit.

The implementation now lives in :mod:`dbwm.dynamics.identification`. This shim
re-exports the old names so existing scripts keep working:

    edmd            -> least_squares_operator
    dmdc_joint      -> joint_least_squares
    dmdc_two_stage  -> two_stage_least_squares
"""
from __future__ import annotations

import warnings

from dbwm.dynamics.identification import (  # noqa: F401
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
    forcing_response_maps,
)

warnings.warn(
    "dbwm.dynamics.dmdc is deprecated; the identification is a least-squares / "
    "finite-section Koopman fit (Section 2.5, Proposition 4.4). Import from "
    "dbwm.dynamics.identification instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backwards-compatible aliases.
edmd = least_squares_operator
dmdc_joint = joint_least_squares
dmdc_two_stage = two_stage_least_squares

__all__ = [
    "edmd",
    "dmdc_joint",
    "dmdc_two_stage",
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
    "forcing_response_maps",
]
