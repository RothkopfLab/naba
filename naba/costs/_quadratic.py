from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpyro.distributions as dist
from numpyro.distributions import LogNormal

from naba.parameters import Parameter
from naba.costs._base import CostFunction


class QuadraticCost(CostFunction):
    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        return (state - response) ** 2

    def optimal_estimate(self, posterior_dist: LogNormal) -> ArrayLike:
        return jnp.exp(posterior_dist.loc + 0.5 * posterior_dist.scale ** 2)

    def optimal_action(self, posterior_dist: LogNormal, sigma_r: ArrayLike = 0.) -> ArrayLike:
        return self.optimal_estimate(posterior_dist) * jnp.exp(- 3 / 2 * sigma_r ** 2)


class EffortParams(NamedTuple):
    beta: Parameter = 1.


class QuadraticCostQuadraticEffort(CostFunction):
    param_type = EffortParams
    param_priors = param_type(beta=dist.Uniform(0.5, 1.))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        return self.params.beta * (state - response) ** 2 + (1. - self.params.beta) * response ** 2

    def optimal_estimate(self, posterior_dist: LogNormal) -> ArrayLike:
        return posterior_dist.mean  # jnp.exp(posterior_dist.loc + 0.5 * posterior_dist.scale ** 2)

    def optimal_action(self, posterior_dist: LogNormal, sigma_r: ArrayLike = 0.) -> ArrayLike:
        return self.params.beta * self.optimal_estimate(posterior_dist) * jnp.exp(- 3 / 2 * sigma_r ** 2)
