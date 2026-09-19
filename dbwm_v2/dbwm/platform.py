"""
JAX backend preflight: fail fast and legibly, or fall back to CPU.

Colab images frequently ship a CUDA plugin built against a *different* JAX
release than the installed ``jaxlib``. JAX then registers the plugin, fails to
initialise it, and the mismatch surfaces later as an ABI error on the first real
operation -- typically::

    jax_cuda12_plugin version 0.7.2 is installed, but it is not compatible
    with the installed jaxlib version 0.10.2

    AttributeError: module 'jaxlib.xla_client' has no attribute
    'register_custom_type_id_handler'

    JaxRuntimeError: INVALID_ARGUMENT: Unexpected PJRT_FFI_UserData_Add_Args
    size: expected 48, got 40. The plugin is likely built with a later version
    than the framework.

Three properties make this worth handling rather than letting it crash:

* It surfaces at the **first JAX op**, which in this pipeline is inside stage 2,
  after data loading and weather assembly have already run. The user waits, then
  gets a 40-frame traceback pointing at ``jax.random.PRNGKey`` -- which is not
  where the problem is.
* It is purely environmental. Nothing in this codebase can be wrong in a way that
  produces it.
* Only stage 2 needs JAX at all; stages 3-10 are NumPy. So falling back to CPU
  costs the smoke run essentially nothing, and costs a full run only the basis
  training time. Aborting would be the worse trade.

:func:`ensure_working_backend` therefore probes the backend **before** any real
work, and on failure diagnoses the cause, retries on CPU, and continues -- or, if
even CPU is broken, raises with instructions instead of a stack trace.
"""
from __future__ import annotations

import os
import sys as _sys
from typing import Dict

from dbwm.log_utils import configure_logger

logger = configure_logger(level="INFO", name="dbwm.platform")

#: Substrings that identify an accelerator plugin / framework version mismatch
#: rather than a genuine JAX bug or an out-of-memory condition.
_MISMATCH_SIGNATURES = (
    "pjrt",
    "plugin",
    "register_custom_type_id_handler",
    "not compatible with the installed jaxlib",
    "unexpected pjrt_ffi_userdata_add_args",
    "built with a later version than the framework",
)


def _installed_versions() -> Dict[str, str]:
    """
    Report the versions of the packages involved in a plugin mismatch.

    :return: ``{package: version}``; missing packages map to ``"(absent)"``.
    """
    import importlib.metadata as md

    out = {}
    for pkg in (
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
        "jax-cuda11-plugin", "flax", "optax",
    ):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "(absent)"
    return out


def _looks_like_version_mismatch(exc: BaseException) -> bool:
    """
    Decide whether a backend failure is a plugin/framework version mismatch.

    :param exc: the exception raised while initialising or using the backend.
    :return: ``True`` if the message matches a known mismatch signature.
    """
    text = "{}: {}".format(type(exc).__name__, exc).lower()
    return any(sig in text for sig in _MISMATCH_SIGNATURES)


def _remedy_message(versions: Dict[str, str]) -> str:
    """
    Build actionable install instructions for a mismatched CUDA plugin.

    :param versions: output of :func:`_installed_versions`.
    :return: a multi-line remedy string.
    """
    jax_v = versions.get("jax", "?")
    return "\n".join(
        [
            "Installed versions:",
            *["    {:22s} {}".format(k, v) for k, v in versions.items()],
            "",
            "The CUDA plugin and jaxlib must come from the SAME JAX release.",
            "Pick one:",
            "",
            "  (a) Keep the GPU -- reinstall a matched set:",
            "        pip install -U 'jax[cuda12]=={}'".format(jax_v),
            "      then RESTART the runtime (Colab caches the loaded plugin).",
            "",
            "  (b) Drop the GPU -- remove the stale plugin entirely:",
            "        pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt",
            "",
            "  (c) Ignore the GPU for this run only, no install needed:",
            "        JAX_PLATFORMS=cpu python -m experiments.run_ndvi_v4 --smoke",
            "",
            "Only stage 2 (basis training) uses JAX in this pipeline; stages 3-10",
            "are NumPy and run on CPU regardless. For --smoke the GPU buys nothing,",
            "so (c) is the fastest way to get moving.",
        ]
    )


def plugin_version_mismatch(versions: Dict[str, str] | None = None) -> Dict[str, object]:
    """
    Detect a CUDA-plugin / jaxlib version mismatch **without importing JAX**.

    This is the decisive check, and it is import-free on purpose. Once
    ``import jax`` has run, ``discover_pjrt_plugins()`` has already loaded and
    registered the plugin; from that point nothing in-process can undo it. Setting
    ``jax_platforms`` afterwards does not help, because creating *any* backend --
    CPU included -- goes through the same PJRT layer whose ABI the stale plugin
    has broken. That is precisely why an in-process fallback fails with the
    identical ``PJRT_FFI_UserData_Add_Args`` error it was trying to escape.

    JAX ships ``jax-cuda12-plugin`` and ``jax-cuda12-pjrt`` in lockstep with
    ``jaxlib``, so an exact version equality test is a reliable predictor.

    :param versions: pre-fetched versions, or ``None`` to read them.
    :return: dict with ``mismatch`` (bool), ``versions``, and ``offenders``.
    """
    versions = versions or _installed_versions()
    jaxlib = versions.get("jaxlib", "(absent)")
    offenders = {}
    if jaxlib != "(absent)":
        for pkg in ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda11-plugin"):
            v = versions.get(pkg, "(absent)")
            if v != "(absent)" and v != jaxlib:
                offenders[pkg] = v
    return {
        "mismatch": bool(offenders),
        "versions": versions,
        "offenders": offenders,
        "jaxlib": jaxlib,
    }


def preflight(force_cpu: bool = False) -> Dict[str, object]:
    """
    Neutralise a broken accelerator plugin **before JAX is imported**.

    Call this at the top of an entry point, ahead of every ``import jax`` and
    ahead of any module that imports JAX. If a plugin/jaxlib mismatch is found,
    ``JAX_PLATFORMS=cpu`` is set in the environment so JAX never initialises the
    broken backend in the first place. Doing it here rather than after import is
    the whole point: after import it is too late.

    :param force_cpu: pin CPU unconditionally, regardless of what is installed.
    :return: dict with ``applied`` (bool), ``reason``, and the mismatch report.
    """
    if os.environ.get("JAX_PLATFORMS"):
        return {"applied": False, "reason": "JAX_PLATFORMS already set",
                "report": {}, "already": os.environ["JAX_PLATFORMS"]}

    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        logger.info("Preflight: JAX pinned to CPU by request.")
        return {"applied": True, "reason": "forced", "report": {}}

    if "jax" in _sys.modules:
        logger.debug(
            "Preflight ran after JAX was already imported; the environment "
            "override can no longer take effect in this process."
        )
        return {"applied": False, "reason": "jax already imported", "report": {}}

    report = plugin_version_mismatch()
    if not report["mismatch"]:
        return {"applied": False, "reason": "versions consistent", "report": report}

    os.environ["JAX_PLATFORMS"] = "cpu"
    logger.warning(
        "Preflight: accelerator plugin does not match jaxlib %s -- %s.\n"
        "Pinning JAX_PLATFORMS=cpu so the broken backend is never initialised.\n%s",
        report["jaxlib"],
        ", ".join(f"{k} {v}" for k, v in report["offenders"].items()),
        _remedy_message(report["versions"]),
    )
    return {"applied": True, "reason": "plugin/jaxlib version mismatch",
            "report": report}


def _probe() -> str:

    """
    Force backend initialisation with a trivial operation.

    :return: the backend name.
    :raises Exception: whatever the backend raises.
    """
    import jax
    import jax.numpy as jnp

    # A real op, not just `jax.devices()`: the ABI mismatch only surfaces when
    # the plugin is actually asked to execute something.
    _ = jnp.asarray([0.0]) + 1.0
    return jax.default_backend()


def ensure_working_backend(allow_cpu_fallback: bool = True) -> Dict[str, object]:
    """
    Verify that JAX can actually execute, falling back to CPU if it cannot.

    :param allow_cpu_fallback: if ``True``, a broken accelerator downgrades the run
        to CPU with a warning; if ``False``, the original error is raised.
    :return: dict with ``backend``, ``fell_back`` and ``versions``.
    :raises RuntimeError: if JAX cannot execute even on CPU, or if
        ``allow_cpu_fallback`` is ``False`` and the accelerator is broken.
    """
    try:
        backend = _probe()
        logger.info("JAX backend OK: %s", backend)
        return {"backend": backend, "fell_back": False, "versions": {}}
    except Exception as exc:  # noqa: BLE001 - any backend failure is in scope
        versions = _installed_versions()
        mismatch = _looks_like_version_mismatch(exc)
        headline = (
            "JAX accelerator backend is broken: the installed CUDA plugin does not "
            "match jaxlib."
            if mismatch
            else "JAX accelerator backend failed to initialise."
        )
        if not allow_cpu_fallback:
            raise RuntimeError(
                "{}\n\n{}\n\nOriginal error: {}".format(
                    headline, _remedy_message(versions), exc
                )
            ) from exc

        logger.warning("%s\n%s", headline, _remedy_message(versions))
        logger.warning("Falling back to CPU for this run.")

        try:
            import jax
            from jax._src import xla_bridge

            jax.config.update("jax_platforms", "cpu")
            os.environ["JAX_PLATFORMS"] = "cpu"
            getter = getattr(xla_bridge, "get_backend", None)
            if getter is not None and hasattr(getter, "cache_clear"):
                getter.cache_clear()
            backend = _probe()
        except Exception as inner:  # noqa: BLE001
            raise RuntimeError(
                "JAX cannot execute, and the in-process CPU fallback did not take "
                "effect (the broken plugin was already loaded). Set the environment "
                "variable BEFORE starting python:\n\n"
                "    JAX_PLATFORMS=cpu python -m experiments.run_ndvi_v4 --smoke\n\n"
                "{}\n\nOriginal error: {}\nFallback error: {}".format(
                    _remedy_message(versions), exc, inner
                )
            ) from inner

        logger.info("CPU fallback succeeded; continuing on %s.", backend)
        return {"backend": backend, "fell_back": True, "versions": versions}


def describe_execution_plan(backend: str) -> str:
    """
    Summarise which pipeline stages will use the accelerator.

    Stated explicitly because the answer is counter-intuitive: an accelerator
    speeds up exactly one of the ten stages, and *not* the most expensive one.

    :param backend: the active JAX backend name.
    :return: a multi-line description.
    """
    accel = backend not in ("cpu", "interpreter")
    return "\n".join(
        [
            "Execution plan:",
            "  stage 2     train Psi (dPPGP)      JAX/Flax -> {}".format(
                backend.upper() if accel else "CPU"
            ),
            "  stages 3-10 GP posterior, memory, emission, horizons,",
            "              filter/forecast, statistics   NumPy -> CPU always",
            "",
            "  Stage 9 (filter + forecast) dominates wall-clock at r=256 and is",
            "  NumPy, so an accelerator does not shorten it.",
        ]
    )
