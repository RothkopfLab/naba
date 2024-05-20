from typing import NamedTuple, Tuple, Type, TypeVar
import jax
from jax import random, jit
from jaxtyping import Array
import jax.numpy as jnp
import numpyro.distributions as dist

from naba.parameters import SensorimotorParams
from naba.costs import CostFunction
from naba.numerical import solve_mc
from naba import lognormal


def random_split_like_tree(rng_key: random.PRNGKey, target=None, treedef=None, is_leaf=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target, is_leaf=is_leaf)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


D = TypeVar("D", bound=NamedTuple)
C = TypeVar("C", bound=CostFunction)


def sample_from_dists(key: random.PRNGKey,
                      dists: D,
                      sample_shape: Tuple = ()) -> D:
    """
    Sample from a bunch of numpyro distributions.

    Args:
        key: jax.random.PRNGKey
        dists: OrderedDict mapping from names to numpyro.distributions.Distribution
        sample_shape: Tuple of ints specifying the shape of the samples

    Returns:
        OrderedDict with the same keys dists, but with samples from the distributions as values
    """
    is_leaf = lambda x: isinstance(x, dist.Distribution)
    keys = random_split_like_tree(key, dists, is_leaf=is_leaf)
    return jax.tree_map(lambda d, k: d.sample(key=k, sample_shape=sample_shape), dists, keys, is_leaf=is_leaf)


def generate_data(key: random.PRNGKey,
                  priors: SensorimotorParams,
                  cost_fn: Type[C],
                  batch_size: int = 32) -> Tuple[Array, SensorimotorParams, NamedTuple]:
    # split random keys
    sensorimotor_key, cost_key, mu_key, sigma_key, s_key, m_key = jax.random.split(key, 6)

    # sample parameters from prior
    sensorimotor_params = sample_from_dists(sensorimotor_key, priors, (batch_size,))

    # sample cost parameters from prior
    cost_params = sample_from_dists(cost_key, cost_fn.param_priors, (batch_size,))

    # sample a set of stimuli
    s = priors.mu_0.sample(s_key, sample_shape=(batch_size,))

    # sample sensory observations from the stimuli
    m = dist.LogNormal(jnp.log(s), sensorimotor_params.sigma).sample(m_key)

    return m, sensorimotor_params, cost_params


def generate_eval_data(key: random.PRNGKey,
                       priors: SensorimotorParams,
                       cost_fn: Type[CostFunction],
                       num_samples: int, analytical=True):
    # generate a bunch of observations and parameters
    m, sensorimotor_params, cost_params = generate_data(key, priors, cost_fn, batch_size=num_samples)

    try:  # try computing the analytical solution
        assert analytical  # only try the analytical solution if analytical == True
        a = cost_fn(params=cost_params).optimal_action(
            posterior_dist=lognormal.posterior(m=m, sigma=sensorimotor_params.sigma,
                                               sigma_0=sensorimotor_params.sigma_0,
                                               mu_0=sensorimotor_params.mu_0),
            sigma_r=sensorimotor_params.sigma_r)
    except (NotImplementedError, AssertionError) as error:  # if analytical solution not implemented or we don't want it
        if isinstance(error, NotImplementedError):  # warn the user if they specified analytical but it doesn't work
            print("Resorting to numerical approximation for evaluation data generation.")
        # do the numerical (Monte Carlo) approximation for each of the data points
        a = []

        @jit
        def solve_fn(m, sigma, sigma_0, mu_0, sigma_r, cost_params, key):
            return solve_mc(cost=cost_fn(params=cost_params),
                            posterior=lognormal.posterior(m, sigma=sigma, sigma_0=sigma_0, mu_0=mu_0),
                            sigma_r=sigma_r, key=key)

        # tried vmapping as well, for some reason the manual for loop with jit is faster
        # solve_fn = jit(vmap(solve_fn))
        # a = solve_fn(m, sensorimotor_params, cost_params)

        for i in range(num_samples):
            key, subkey = random.split(key)
            a.append(solve_fn(m=m[i], sigma=sensorimotor_params.sigma[i], sigma_0=sensorimotor_params.sigma_0[i],
                              mu_0=sensorimotor_params.mu_0[i], sigma_r=sensorimotor_params.sigma_r[i],
                              cost_params=cost_fn.param_type(
                                  **{name: values[i] for name, values in cost_params._asdict().items()}), key=subkey))

        a = jnp.array(a)

    return m, sensorimotor_params, cost_params, a
