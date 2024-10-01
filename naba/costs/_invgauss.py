from typing import NamedTuple

from jaxtyping import ArrayLike
import jax.numpy as jnp
import numpyro.distributions as dist

from fechner.costs._base import CostFunction
from fechner.parameters import Parameter


class InvGaussParams(NamedTuple):
    gamma: Parameter = 1.


class InvertedGaussian(CostFunction):
    param_type = InvGaussParams
    param_priors = param_type(gamma=dist.LogNormal(jnp.log(2.5), .5))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        return 1. - jnp.exp(-(state - response) ** 2 / (2 * self.params.gamma ** 2))
