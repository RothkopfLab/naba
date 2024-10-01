import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpyro.distributions as dist
from numpyro.distributions import LogNormal

from fechner.costs._base import CostFunction
from fechner.costs._quadratic import EffortParams


class AbsoluteCost(CostFunction):
    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        return jnp.abs(state - response)

    def optimal_estimate(self, posterior_dist: LogNormal) -> ArrayLike:
        return jnp.exp(posterior_dist.loc)


class AbsoluteCostQuadraticEffort(CostFunction):
    param_type = EffortParams
    param_priors = param_type(beta=dist.Uniform(.5, 1.))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        return self.params.beta * jnp.abs(state - response) + (1. - self.params.beta) * response ** 2

    def optimal_estimate(self, posterior_dist: LogNormal) -> ArrayLike:
        return jnp.exp(posterior_dist.loc)
        