import jax.numpy as jnp
from jax.scipy.stats import norm
from numpyro import distributions as dist

from naba.parameters import SensorimotorParams


def lognormal_from_quantiles(x1, x2, p1=0.05, p2=0.95):
    """ Compute the parameters of a log-normal distribution, such that F(x1) = p1 and F(x2) = p2

    Args:
        x1: lower value
        x2: upper value
        p1: lower probability
        p2: upper probability

    Returns:
        mu, sigma (parameters of the log-normal distribution
    """
    sigma = (jnp.log(x2) - jnp.log(x1)) / (norm.ppf(p2) - norm.ppf(p1))
    mu = (jnp.log(x2) * norm.ppf(p2) - jnp.log(x1) * norm.ppf(p1)) / (norm.ppf(p2) - norm.ppf(p1))
    return mu, sigma


priors_default = SensorimotorParams(
    sigma=dist.Uniform(low=0.01, high=0.5),
    sigma_0=dist.Uniform(low=0.01, high=0.5),
    mu_0=dist.Uniform(low=0.1, high=7.),
    sigma_r=dist.Uniform(low=0.01, high=0.5),
)

# priors used in MCMC
priors_inference = SensorimotorParams(
    sigma=dist.HalfNormal(.25),
    sigma_0=dist.HalfNormal(.25),
    mu_0=dist.Uniform(low=0.1, high=5.),
    sigma_r=dist.HalfNormal(.25)
)

# smaller upper bounds for noises to have evaluation on data sets not dominated by noise
priors_evaluation = SensorimotorParams(
    sigma=dist.Uniform(low=0.1, high=0.25),
    sigma_0=dist.Uniform(low=0.1, high=0.25),
    mu_0=dist.Uniform(low=2.0, high=5.),
    sigma_r=dist.Uniform(low=0.1, high=0.25),
)
