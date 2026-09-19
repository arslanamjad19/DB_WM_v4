"""Packaging for the DB-WM v2 library."""
from setuptools import setup, find_packages

setup(
    name="dbwm_v2",
    version="2.0.0",
    author="Arslan Amjad",
    description="Deep Basis World Models (DB-WM v2): scalable GP dynamics for "
    "LST/NDVI spatiotemporal forecasting, in JAX/Flax.",
    packages=find_packages(exclude=["tests", "tests.*", "experiments"]),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4",
        "flax>=0.8",
        "optax>=0.2",
        "numpy",
        "scipy",
        "rasterio",
        "matplotlib",
        "pandas",
        "scikit-image",
        "tqdm",
    ],
)
