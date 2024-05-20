import jax
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer.reparam import TransformReparam

from naba.parameters import SensorimotorParams
from naba.priors import priors_inference


def model(s, nn, cost, r=None, priors: SensorimotorParams = None, **fixed_params):
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

        # the above is the same as this, but makes use of TransformReparam
        # m = numpyro.sample("m", dist.LogNormal(jnp.log(s), sigma))

        a = jax.vmap(nn, in_axes=(0, None, None))(m, sensorimotor_params, cost_params)

        r = numpyro.sample("r", dist.LogNormal(jnp.log(a), sensorimotor_params.sigma_r), obs=r)

    return r
