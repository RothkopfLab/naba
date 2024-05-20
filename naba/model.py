from collections import OrderedDict

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer.reparam import TransformReparam

from naba.costs import QuadraticCostQuadraticEffort
from naba import lognormal
from naba.parameters import SensorimotorParams
from naba.priors import priors_inference


def model(s, r=None, cost=QuadraticCostQuadraticEffort, priors: SensorimotorParams = None, **fixed_params):
    if priors is None:
        priors = priors_inference

    N = s.shape[0]

    # sensorimotor parameters
    sensorimotor_params = SensorimotorParams(
        **{name: numpyro.sample(name, d) if name not in fixed_params.keys() else fixed_params[name] for name, d in
           priors._asdict().items()})

    # cost params
    cost_params = cost.param_type(
        **{name: numpyro.sample(name, d) if name not in fixed_params.keys() else fixed_params[name] for name, d in
           cost.param_priors._asdict().items()})

    with numpyro.plate("N", N):
        with numpyro.handlers.reparam(config={"m": TransformReparam()}):
            m = numpyro.sample("m", dist.TransformedDistribution(
                dist.Normal(loc=0., scale=1.),
                transforms=[dist.transforms.AffineTransform(jnp.log(s), sensorimotor_params.sigma),
                            dist.transforms.ExpTransform()]))
        post = lognormal.posterior(m, mu_0=sensorimotor_params.mu_0,
                                   sigma_0=sensorimotor_params.sigma_0,
                                   sigma=sensorimotor_params.sigma)

        # a = numpyro.deterministic("a",
        a = cost(cost_params).optimal_action(posterior_dist=post, sigma_r=sensorimotor_params.sigma_r)

        r = numpyro.sample("r", dist.LogNormal(jnp.log(a), sensorimotor_params.sigma_r), obs=r)

    return r
