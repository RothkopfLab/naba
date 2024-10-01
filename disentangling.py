import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import jax
from jax import random, numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, init_to_median
from scipy.stats import pearsonr

numpyro.set_host_device_count(4)

from naba.priors import priors_inference
from naba.parameters import SensorimotorParams
from naba.costs import QuadraticCostQuadraticEffort
from naba.infer import run_mcmc
from naba.nn import load

from plot_specs import var_name_map
import plot_specs

plot_specs.update()

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

colors = plt.cm.tab20c((4. / 3 * np.arange(20 * 3 / 4)).astype(int))


def multiple_noises_model(s, nn, cost, r=None, priors: SensorimotorParams = None, **fixed_params):
    if priors is None:
        priors = priors_inference
    M, N = s.shape

    mu_0 = numpyro.sample("mu_0", priors.mu_0) if fixed_params.get("mu_0") is None else fixed_params["mu_0"]
    sigma_0 = numpyro.sample("sigma_0", priors.sigma_0) if fixed_params.get("sigma_0") is None else fixed_params[
        "sigma_0"]
    sigma_r = numpyro.sample("sigma_r", priors.sigma_r) if fixed_params.get("sigma_r") is None else fixed_params[
        "sigma_r"]
    if fixed_params.get("sigma") is None:
        sigma = numpyro.sample("sigma", priors.sigma, sample_shape=(M,))
    else:
        sigma = fixed_params["sigma"]

    # cost params
    cost_params = cost.param_type(
        **{name: numpyro.sample(name, d) if name not in fixed_params.keys() else fixed_params[name] for name, d in
           cost.param_priors._asdict().items()})

    m = numpyro.sample("m", dist.LogNormal(loc=jnp.log(s), scale=sigma[:, jnp.newaxis]))

    nn_vec = jax.vmap(
        lambda m, sigma: nn(m, SensorimotorParams(mu_0=mu_0, sigma_0=sigma_0, sigma_r=sigma_r, sigma=sigma),
                            cost_params=cost_params))
    nn_mat = jax.vmap(nn_vec, in_axes=(1, None))

    a = nn_mat(m, sigma).T

    r = numpyro.sample("r", dist.LogNormal(jnp.log(a), sigma_r), obs=r)
    return r


if __name__ == '__main__':
    key = random.PRNGKey(12)

    cost_fn = QuadraticCostQuadraticEffort
    nn = load(f"checkpoints/QuadraticCostQuadraticEffort-ActionNetwork.pkl")

    N = 45

    noise_levels = [0.1, 0.2]
    M = len(noise_levels)


    # same stimuli for both experiments
    key, subkey = random.split(key)
    s = dist.Uniform(.25, 5.).sample(subkey, sample_shape=(M, N))

    base_true_params = dict(mu_0=1.5, sigma_0=.2, sigma_r=0.15)

    true_params_options = {"multi_noise": dict(**base_true_params, sigma=jnp.array(noise_levels)),
                           "single_noise_low": dict(**base_true_params, sigma=jnp.array([noise_levels[0]] * M)),
                           "single_noise_high": dict(**base_true_params, sigma=jnp.array([noise_levels[1]] * M)),
                           }
    cost_params = dict(beta=0.9)

    inference_datas = {}
    for option, true_params in true_params_options.items():
        # sample some responses (i.e. a single sample from the prior predictive of r)
        key, subkey = random.split(key)
        prior_predictive = Predictive(multiple_noises_model, num_samples=10_000)
        ppc = prior_predictive(subkey, s=s, nn=nn, cost=cost_fn, **true_params, **cost_params)
        r = ppc["r"][0]

        # visualize behavior
        if option == "multi_noise":
            f, ax = plt.subplots(figsize=(4.65 / 2, 4.65 / 2 / GOLDEN_RATIO))

            for i in range(M):
                # plt.plot(jnp.sort(s[i]), ppc["r"].mean(axis=0)[i][jnp.argsort(s[i])], color=colors[i])
                ax.scatter(s[i], r[i], color=colors[i], s=2)

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$r$")
            f.show()

        key, subkey = random.split(key)

        mcmc = run_mcmc(subkey, s, r, multiple_noises_model,
                        num_warmup=2_500, num_samples=5_000,
                        nn=nn,
                        cost=cost_fn,
                        init_strategy=init_to_median,
                        )

        idata = az.from_numpyro(mcmc)
        inference_datas[option] = idata

        print(az.summary(idata, var_names=["mu_0", "sigma_0", "sigma_r", "sigma", "beta"]))

    labeller = az.labels.MapLabeller(var_name_map=var_name_map)

    var_names = ["mu_0", "beta"]
    for i, (option, idata) in enumerate(inference_datas.items()):
        print(option)
        print(pearsonr(idata.posterior["mu_0"].values.flatten(),
                       idata.posterior["beta"].values.flatten()))
        if i == 0:
            ax = az.plot_pair(idata, var_names=var_names, kind="kde",
                              kde_kwargs={"hdi_probs": [0.94], "contour_kwargs": {"colors": "purple", "zorder": 100},
                                          "contourf_kwargs": {"alpha": 0., "colors": "white"}},
                              labeller=labeller, marginals=True, marginal_kwargs={"color": "purple"},
                              reference_values={var_name_map[k]: v for k, v in {**base_true_params,
                                                                                **cost_params}.items() if
                                                k in var_names},
                              reference_values_kwargs={"ms": 2., "color": "k"},
                              textsize=6,
                              figsize=(4.65 / 2, 4.65 / 2 / GOLDEN_RATIO),
                              )
        else:
            ax = az.plot_pair(idata, var_names=var_names, kind="kde",
                              kde_kwargs={"hdi_probs": [0.94],
                                          "contour_kwargs": {"colors": colors[i - 1], "linestyle": ":"},
                                          "contourf_kwargs": {"alpha": 0., "colors": "white"}},
                              labeller=labeller, ax=ax,
                              marginals=True, marginal_kwargs={"color": colors[i - 1]}, textsize=6,
                              )

    plt.tight_layout()
    plt.show()