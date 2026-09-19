"""
Tests for the JAX backend preflight.

The failure this guards against is environmental (a Colab image shipping a CUDA
plugin from a different JAX release) but it lands *inside stage 2*, so without a
preflight the user waits through data loading only to get a traceback pointing at
``jax.random.PRNGKey`` -- which is not where the problem is.
"""
import pytest

from dbwm import platform as P


COLAB_ERROR = (
    "INVALID_ARGUMENT: Unexpected PJRT_FFI_UserData_Add_Args size: expected 48, "
    "got 40. The plugin is likely built with a later version than the framework. "
    "This plugin is built with PJRT API version 0.76."
)
PLUGIN_ATTR_ERROR = (
    "module 'jaxlib.xla_client' has no attribute "
    "'register_custom_type_id_handler'"
)


def test_healthy_backend_is_reported_without_fallback():
    out = P.ensure_working_backend()
    assert out["backend"] in ("cpu", "gpu", "tpu", "cuda", "rocm", "METAL")
    assert out["fell_back"] is False


@pytest.mark.parametrize("msg", [COLAB_ERROR, PLUGIN_ATTR_ERROR,
                                 "jax_cuda12_plugin version 0.7.2 is installed, but "
                                 "it is not compatible with the installed jaxlib "
                                 "version 0.10.2"])
def test_version_mismatch_is_recognised(msg):
    assert P._looks_like_version_mismatch(RuntimeError(msg))


@pytest.mark.parametrize("msg", ["RESOURCE_EXHAUSTED: out of memory",
                                 "NOT_FOUND: no visible devices",
                                 "some unrelated failure"])
def test_genuine_failures_are_not_misread_as_version_mismatch(msg):
    """A real OOM must not be mislabelled, or the remedy text would mislead."""
    assert not P._looks_like_version_mismatch(RuntimeError(msg))


def test_broken_accelerator_falls_back_to_cpu(monkeypatch, caplog):
    """The pipeline must continue on CPU rather than abort.

    Only stage 2 uses JAX; stages 3-10 are NumPy. Aborting would cost the whole
    run to save nothing.
    """
    real = P._probe
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError(COLAB_ERROR)
        return real()

    monkeypatch.setattr(P, "_probe", flaky)
    with caplog.at_level("WARNING"):
        out = P.ensure_working_backend(allow_cpu_fallback=True)
    assert out["fell_back"] is True
    assert out["backend"] == "cpu"
    text = "\n".join(r.getMessage() for r in caplog.records)
    # The remedy must be actionable, not just a complaint.
    assert "jax[cuda12]" in text
    assert "JAX_PLATFORMS=cpu" in text
    assert "pip uninstall" in text


def test_require_accelerator_refuses_to_downgrade(monkeypatch):
    """--require-accelerator exists so a GPU run cannot silently become a CPU run."""
    monkeypatch.setattr(P, "_probe", lambda: (_ for _ in ()).throw(RuntimeError(COLAB_ERROR)))
    with pytest.raises(RuntimeError) as exc:
        P.ensure_working_backend(allow_cpu_fallback=False)
    assert "does not match jaxlib" in str(exc.value)
    assert "jax[cuda12]" in str(exc.value)


def test_unrecoverable_failure_names_the_env_var(monkeypatch):
    """If the fallback cannot take effect in-process, say what to do instead."""
    monkeypatch.setattr(
        P, "_probe", lambda: (_ for _ in ()).throw(RuntimeError("totally broken"))
    )
    with pytest.raises(RuntimeError) as exc:
        P.ensure_working_backend(allow_cpu_fallback=True)
    assert "JAX_PLATFORMS=cpu" in str(exc.value)


def test_version_report_covers_the_relevant_packages():
    v = P._installed_versions()
    assert {"jax", "jaxlib", "jax-cuda12-plugin"} <= set(v)
    assert v["jax"] != "(absent)"


def test_execution_plan_is_honest_about_gpu_coverage():
    """The plan must say plainly that the dominant stage is not accelerated."""
    plan = P.describe_execution_plan("gpu")
    assert "stage 2" in plan.lower()
    assert "NumPy -> CPU always" in plan
    assert "Stage 9" in plan


# --------------------------------------------------------------------------- #
# The decisive fix: import-free detection, applied BEFORE jax loads
# --------------------------------------------------------------------------- #
COLAB_VERSIONS = {
    "jax": "0.10.2", "jaxlib": "0.10.2",
    "jax-cuda12-plugin": "0.7.2", "jax-cuda12-pjrt": "0.7.2",
    "jax-cuda11-plugin": "(absent)", "flax": "0.12.7", "optax": "0.2.8",
}
HEALTHY_VERSIONS = dict(COLAB_VERSIONS,
                        **{"jax-cuda12-plugin": "0.10.2", "jax-cuda12-pjrt": "0.10.2"})
CPU_ONLY_VERSIONS = dict(COLAB_VERSIONS,
                         **{"jax-cuda12-plugin": "(absent)", "jax-cuda12-pjrt": "(absent)"})


def test_mismatch_detected_from_versions_alone():
    """The check must work WITHOUT importing jax -- after import it is too late.

    Once `import jax` has run, discover_pjrt_plugins() has registered the broken
    plugin and every backend (CPU included) goes through the same broken PJRT
    layer. That is why the in-process fallback failed with the very error it was
    trying to escape.
    """
    r = P.plugin_version_mismatch(COLAB_VERSIONS)
    assert r["mismatch"] is True
    assert r["offenders"] == {"jax-cuda12-plugin": "0.7.2", "jax-cuda12-pjrt": "0.7.2"}


def test_matched_plugin_is_not_flagged():
    assert P.plugin_version_mismatch(HEALTHY_VERSIONS)["mismatch"] is False


def test_absent_plugin_is_not_flagged():
    """A CPU-only install must not be mistaken for a broken one."""
    assert P.plugin_version_mismatch(CPU_ONLY_VERSIONS)["mismatch"] is False


def test_preflight_pins_cpu_on_mismatch(monkeypatch, caplog):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(P, "_installed_versions", lambda: COLAB_VERSIONS)
    monkeypatch.setattr(P._sys, "modules", {})  # pretend jax not yet imported
    with caplog.at_level("WARNING"):
        out = P.preflight()
    assert out["applied"] is True
    assert P.os.environ["JAX_PLATFORMS"] == "cpu"
    assert "does not match jaxlib" in "\n".join(r.getMessage() for r in caplog.records)


def test_preflight_leaves_healthy_environment_alone(monkeypatch):
    """No accelerator must be given up when nothing is actually wrong."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(P, "_installed_versions", lambda: HEALTHY_VERSIONS)
    monkeypatch.setattr(P._sys, "modules", {})
    out = P.preflight()
    assert out["applied"] is False
    assert P.os.environ.get("JAX_PLATFORMS") is None


def test_preflight_respects_a_user_set_platform(monkeypatch):
    """An explicit JAX_PLATFORMS must never be overridden."""
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setattr(P, "_installed_versions", lambda: COLAB_VERSIONS)
    out = P.preflight()
    assert out["applied"] is False
    assert P.os.environ["JAX_PLATFORMS"] == "cuda"


def test_preflight_is_a_noop_once_jax_is_imported(monkeypatch):
    """After import the env override cannot help, and pretending otherwise misleads."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(P, "_installed_versions", lambda: COLAB_VERSIONS)
    monkeypatch.setattr(P._sys, "modules", {"jax": object()})
    out = P.preflight()
    assert out["applied"] is False
    assert out["reason"] == "jax already imported"


def test_force_cpu_pins_regardless(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(P._sys, "modules", {})
    assert P.preflight(force_cpu=True)["applied"] is True
    assert P.os.environ["JAX_PLATFORMS"] == "cpu"


def test_driver_calls_preflight_before_importing_jax():
    """Ordering is the whole fix, so pin it against accidental reordering."""
    src = open("experiments/run_ndvi_v4.py").read()
    pf = src.index("_PREFLIGHT = preflight()")
    jx = src.index("import jax.numpy as jnp")
    assert pf < jx, "preflight() must run before any jax import"


def test_package_init_preflights_before_jax_submodules():
    """`import dbwm.<anything>` in a notebook must be protected too."""
    src = open("dbwm/__init__.py").read()
    assert "preflight" in src
    assert "DBWM_SKIP_PREFLIGHT" in src
