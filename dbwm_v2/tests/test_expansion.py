"""Tests for the expansion layers, especially the corrected SwiGLU formula."""
import jax
import jax.numpy as jnp

from dbwm.models.expansion import (
    SwiGLUExpansion,
    RBFExpansion,
    GELUExpansion,
    swish,
)


def test_swish_matches_silu():
    """swish(z, beta=1) must equal jax.nn.silu (z * sigmoid(z))."""
    z = jnp.linspace(-5, 5, 50)
    assert jnp.allclose(swish(z, 1.0), jax.nn.silu(z), atol=1e-6)


def test_swiglu_two_projection_formula():
    """
    SwiGLU output must equal the *correct* GLU-variant form
    Swish(W1 g + b1) * (W2 g + b2) * scale, NOT a single projection.
    """
    r, h = 8, 4
    mod = SwiGLUExpansion(r=r)
    g = jax.random.normal(jax.random.PRNGKey(1), (3, h))
    params = mod.init(jax.random.PRNGKey(0), g)
    out = mod.apply(params, g)

    p = params["params"]
    gate = g @ p["W_gate"]["kernel"] + p["W_gate"]["bias"]
    value = g @ p["W_value"]["kernel"] + p["W_value"]["bias"]
    expected = p["scale"] * (jax.nn.silu(gate) * value)
    assert out.shape == (3, r)
    assert jnp.allclose(out, expected, atol=1e-5)


def test_swiglu_nonzero_almost_everywhere():
    """SwiGLU features should be shaded (nonzero) for generic inputs (Prop 4.1)."""
    mod = SwiGLUExpansion(r=16)
    g = jax.random.normal(jax.random.PRNGKey(2), (10, 6))
    params = mod.init(jax.random.PRNGKey(0), g)
    out = mod.apply(params, g)
    # Every basis column activated by at least one input.
    col_active = jnp.any(jnp.abs(out) > 1e-8, axis=0)
    assert bool(jnp.all(col_active))


def test_rbf_expansion_shape_and_finite():
    """RBF expansion returns (B, r) finite features."""
    mod = RBFExpansion(r=12, lengthscale=1.0)
    g = jax.random.normal(jax.random.PRNGKey(3), (5, 7))
    params = mod.init(jax.random.PRNGKey(0), g)
    out = mod.apply(params, g)
    assert out.shape == (5, 12)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_gelu_expansion_shape():
    """GELU expansion returns (B, r)."""
    mod = GELUExpansion(r=9)
    g = jax.random.normal(jax.random.PRNGKey(4), (4, 5))
    params = mod.init(jax.random.PRNGKey(0), g)
    out = mod.apply(params, g)
    assert out.shape == (4, 9)
