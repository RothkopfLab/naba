from jax import numpy as jnp
from numpyro import distributions as dist


def moment(mu, sigma, n):
    return jnp.exp(n * jnp.log(mu) + n ** 2 * sigma ** 2 / 2)


def posterior(m, mu_0, sigma, sigma_0):
    # posterior params
    sigma_post = (1 / sigma_0 ** 2 + 1 / sigma ** 2) ** (-1 / 2)

    w_m = sigma_post ** 2 / sigma ** 2
    w_0 = sigma_post ** 2 / sigma_0 ** 2

    return dist.LogNormal(loc=w_0 * jnp.log(mu_0) + w_m * jnp.log(m), scale=sigma_post)
