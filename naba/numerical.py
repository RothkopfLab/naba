from typing import Tuple
import jax.numpy as jnp
from jax import random
from jaxtyping import ArrayLike
from numpyro.distributions import LogNormal
from jax.scipy.special import erfinv
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar
from jax.scipy.optimize import minimize

from naba.costs import CostFunction


def _weighted_cost(cost_function, r, s, posterior_dist: LogNormal, response_dist: LogNormal, ):
    """ Calculates the cost function weighted by the densities for response r and stimulus s
    given the posterior and response distribution. """
    return jnp.exp(response_dist.log_prob(r) + posterior_dist.log_prob(s)) * cost_function(s, r)


def quantile(d: LogNormal, q: ArrayLike):
    return jnp.exp(d.loc + np.sqrt(2) * d.scale * erfinv(2 * q - 1))


def _integrate_cost_function(cost_function: CostFunction, posterior_dist: LogNormal, response_dist: LogNormal,
                             quantile_bounds: Tuple[float, float] = (0.001, 0.999)):
    """ Integrates the cost function over the action and perceptual space. """

    act_bounds = quantile(response_dist, jnp.array(quantile_bounds))
    per_bounds = quantile(posterior_dist, jnp.array(quantile_bounds))

    integral = dblquad(lambda r, s: _weighted_cost(cost_function, r, s, posterior_dist, response_dist),
                       *per_bounds, *act_bounds)[0]

    return integral


def solve(cost: CostFunction, posterior: LogNormal, sigma_r: float,
          quantile_bounds: Tuple[float, float] = (0.001, 0.999)) -> float:
    # minimize the expected cost function under the posterior and response distribution
    # by finding the optimal action.
    # The optimal action is found in log space
    result = minimize_scalar(lambda a: _integrate_cost_function(cost,
                                                                posterior,
                                                                LogNormal(loc=a, scale=sigma_r),
                                                                quantile_bounds=quantile_bounds))
    return np.exp(result.x)


def solve_mc(cost: CostFunction, posterior: LogNormal, sigma_r: float,
             num_samples: int = 10_000, key: random.PRNGKey = random.PRNGKey(0)) -> ArrayLike:
    act_key, per_key = random.split(key)
    untrans_respone_samples = random.normal(act_key, shape=(num_samples,))
    posterior_samples = posterior.sample(per_key, sample_shape=(num_samples,))

    # minimize the expected cost function under the posterior and response distribution
    # by finding the optimal action.
    # The optimal action is found in log space
    result = minimize(lambda a: cost(posterior_samples, jnp.exp(a + sigma_r * untrans_respone_samples)).mean(),
                      x0=jnp.log(jnp.array([posterior.mean])), method="BFGS")
    return jnp.exp(result.x[0])
