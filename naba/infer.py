from typing import Callable
from jaxtyping import Array
from jax import random, numpy as jnp
from numpyro import optim
from numpyro.infer import MCMC, NUTS, init_to_mean, SVI, Trace_ELBO, autoguide
from numpyro.infer.reparam import NeuTraReparam


def run_mcmc(key: random.PRNGKey, s: Array, r: Array, model: Callable, num_warmup: int, num_samples: int,
             init_strategy: Callable = init_to_mean, num_chains: int = 4, method="nuts", **kwargs) -> MCMC:
    """

    Args:
        key: jax.random.PRGNKey
        s: stimuli
        r: responses
        model: numpyro model function
        init_strategy: initialization strategy for MCMC
        num_warmup: warmup samples
        num_samples: samples to draw
        num_chains: number of parallel chains (make sure to set device count beforehand)
        method: nuts or neutra
        **kwargs: keyword arguments for the model, could e.g. be fixed parameters (which will not be sampled)

    Returns:
        MCMC object
    """

    if method.lower() == "nuts":
        nuts_kernel = NUTS(model, init_strategy=init_strategy)

    elif method.lower() == "neutra":
        guide = autoguide.AutoIAFNormal(model)

        svi = SVI(model, guide, optim.Adam(3e-3), Trace_ELBO())

        key, subkey = random.split(key)
        svi_result = svi.run(rng_key=subkey, num_steps=10_000, stable_update=True, s=s, r=r, **kwargs)

        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(model)
        nuts_kernel = NUTS(neutra_model, init_strategy=init_strategy)
    else:
        raise ValueError("Method must be nuts or neutra")

    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

    mcmc.run(key, s=s, r=r, **kwargs)
    return mcmc


def percentile_interval(x, p=0.95, axis=None):
    return tuple(jnp.quantile(x, jnp.array([(1 - p) / 2, 1 - (1 - p) / 2]), axis=axis))
