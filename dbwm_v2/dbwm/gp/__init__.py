"""Scalable low-rank Gaussian-process posterior (Woodbury / Deep Basis Kernel)."""
from dbwm.gp.posterior import (
    gram_matrix,
    accumulate_gram,
    lambda_matrix,
    solve_weights,
    posterior_mean_var,
    log_marginal_likelihood,
)

__all__ = [
    "gram_matrix",
    "accumulate_gram",
    "lambda_matrix",
    "solve_weights",
    "posterior_mean_var",
    "log_marginal_likelihood",
]
