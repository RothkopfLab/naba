import argparse

import jax
from jax import random, numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

import matplotlib.pyplot as plt
import matplotlib as mpl
import arviz as az

numpyro.set_host_device_count(4)

from naba.nn import model, load
from naba import model as ground_truth
from naba.infer import run_mcmc
from naba import costs
from naba.parameters import SensorimotorParams

import plot_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--cost", type=str, default="QuadraticCost",
                        help="Cost function type")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


args = parse_args()

# setup plotting
plot_specs.update()
mpl.rcParams.update()

# setup random key
model_key, train_key = random.split(jax.random.PRNGKey(args.seed), 2)

# set cost function type
cost_name = args.cost
cost_fn = getattr(costs, cost_name)

# load pre-trained network
nn = load(f"checkpoints/{cost_name}-ActionNetwork.pkl")

# random seed
key = random.PRNGKey(0)

# set ground truth parameters
true_params = dict(mu_0=1.5, sigma_0=.15, sigma=0.2, sigma_r=0.1)
cost_params = dict()

# generate random stimuli
N = 60
key, subkey = random.split(key)
s = dist.Uniform(.1, 5.).sample(subkey, sample_shape=(N,))

# generate responses
prior_predictive = Predictive(ground_truth.model, num_samples=1)
ppc = prior_predictive(subkey, s=s, cost=cost_fn, **true_params, **cost_params)
r = ppc["r"]

# parameters held fixed during inference
fixed_params = ["sigma"]

# parameters to infer
var_names = list(set(SensorimotorParams._fields) - set(fixed_params)) + cost_fn.param_names()

# run inference using neural net
key, subkey = random.split(key)
mcmc = run_mcmc(subkey, s, r, model, num_warmup=2_500, num_samples=5_000,
                nn=nn,
                cost=cost_fn,
                **{k: true_params[k] for k in fixed_params},
                )
idata = az.from_numpyro(mcmc)
print(az.summary(idata, var_names=var_names))

# run inference using ground truth model
key, subkey = random.split(key)
mcmc_true = run_mcmc(subkey, s, r, ground_truth.model, num_warmup=2_500, num_samples=5_000,
                     cost=cost_fn,
                     **{k: true_params[k] for k in fixed_params},
                     )
idata_true = az.from_numpyro(mcmc_true)
print(az.summary(idata_true, var_names=var_names))

# setup plot labels
var_name_map = {"mu_0": r"$\mu_0$", "sigma_0": r"$\sigma_0$", "sigma_r": r"$\sigma_r$", "sigma": r"$\sigma$"}
labeller = az.labels.MapLabeller(var_name_map=var_name_map)

# visualize posterior for ground truth
ax = az.plot_pair(idata_true, var_names=var_names, kind="kde",
                  kde_kwargs={"hdi_probs": [0.94],
                              "contour": True,
                              "contour_kwargs": {"colors": "C1"},
                              "contourf_kwargs": {"alpha": 0.}
                              }, labeller=labeller,
                  marginals=True, marginal_kwargs={"color": "C1"},
                  figsize=(2.7, 2.),
                  reference_values={var_name_map[k]: v for k, v in {**true_params, **cost_params}.items() if
                                    k not in fixed_params},
                  reference_values_kwargs={"ms": 3.},
                  textsize=6,
                  )
# visualize posterior for neural network
ax = az.plot_pair(idata, var_names=var_names, kind="kde",
                  kde_kwargs={"hdi_probs": [0.94],
                              "contour": True,
                              "contour_kwargs": {"colors": "C0"},
                              "contourf_kwargs": {"alpha": 0.}
                              },
                  labeller=labeller, ax=ax,
                  marginals=True, marginal_kwargs={"color": "C0"}, textsize=6)
plt.show()

# posterior predictive
s_range = jnp.linspace(0.1, 5.0)
posterior_predictive = Predictive(model, {k: v for k, v in mcmc.get_samples().items() if k in var_names})
r_pred = posterior_predictive(subkey, s=s_range, nn=nn, cost=cost_fn, **{k: true_params[k] for k in fixed_params})["r"]

# plot data and posterior predictives
f, ax = plt.subplots(figsize=(2., 1.))
plt.scatter(s, r, s=2, color="k")

plt.plot(jnp.linspace(0, s.max()), jnp.linspace(0, s.max()), label="$r = s$", linestyle="--", color="k")
plt.axhline(true_params["mu_0"], linestyle=":", color="gray", label=r"$\mu_0$")

plt.plot(s_range, jnp.mean(r_pred, axis=0), label="$p(r^* | s, r)$")

plt.fill_between(s_range, *jnp.percentile(r_pred, jnp.array([3, 97]), axis=0), alpha=0.25)

plt.xlabel("$s$")
plt.ylabel("$r$")
plt.legend(fontsize=6)
plt.show()
