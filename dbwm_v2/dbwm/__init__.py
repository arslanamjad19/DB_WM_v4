"""
DB-WM v2: Deep Basis World Models for scalable Gaussian-process dynamics from
visual (satellite) observations.

JAX / Flax / Optax implementation of the framework described in
``DB_WM_v2_framework.md`` (reconciled with the v3 input-affine corrections),
specialised to 30 m LST / NDVI GeoTIFF forecasting over Lahore.
"""
# Neutralise a mismatched accelerator plugin BEFORE any submodule imports JAX.
# This must run first: once `import jax` has happened, the broken plugin is
# already registered and no in-process override can escape it. Opt out with
# DBWM_SKIP_PREFLIGHT=1 if you are managing JAX_PLATFORMS yourself.
import os as _os

if not _os.environ.get("DBWM_SKIP_PREFLIGHT"):
    from dbwm.platform import preflight as _preflight

    _preflight()

from dbwm.config import (
    ExperimentConfig,
    DataConfig,
    BackboneConfig,
    BasisConfig,
    DynamicsConfig,
    TrainingConfig,
    InferenceConfig,
    default_config,
    smoke_config,
)

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "BackboneConfig",
    "BasisConfig",
    "DynamicsConfig",
    "TrainingConfig",
    "InferenceConfig",
    "default_config",
    "smoke_config",
]

__version__ = "2.0.0"
