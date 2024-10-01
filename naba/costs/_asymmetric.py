from typing import NamedTuple
import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike
import numpyro.distributions as dist

from naba.costs import CostFunction
from naba.parameters import Parameter


class LinexParams(NamedTuple):
    a: Parameter = 1.


class Linex(CostFunction):
    param_type = LinexParams
    param_priors = param_type(a=dist.LogNormal(jnp.log(.5), 1.))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        a = self.params.a
        b = 2 / a ** 2
        return b * (jnp.exp(a * (response - state)) - a * (response - state) - 1.)


class AsymmetricParams(NamedTuple):
    alpha: Parameter = 0.5


class AsymmetricQuadratic(CostFunction):
    param_type = AsymmetricParams
    param_priors = param_type(alpha=dist.Uniform(.1, .9))

    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        d = (response - state)
        return 2 * jnp.abs(self.params.alpha - jnp.heaviside(d, 1.)) * d ** 2
