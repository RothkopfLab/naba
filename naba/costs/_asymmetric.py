from typing import NamedTuple
import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike
import numpyro.distributions as dist

from naba.costs import CostFunction
from naba.parameters import Parameter


class AsymmetricParams(NamedTuple):
    alpha: Parameter = 0.5


class AsymmetricQuadratic(CostFunction):
    param_type = AsymmetricParams
    param_priors = param_type(alpha=dist.Uniform(.1, .9))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        d = (response - state)
        return 2 * jnp.abs(self.params.alpha - jnp.heaviside(d, 1.)) * d ** 2
