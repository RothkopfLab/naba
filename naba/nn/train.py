from typing import OrderedDict, Type, Optional, Tuple, NamedTuple

import equinox as eqx
import jax
import optax
from jax import random, numpy as jnp
from jaxtyping import Array, PyTree, ArrayLike
from numpyro import distributions as dist

from naba import lognormal
from naba.costs import CostFunction
from naba.data import generate_data, generate_eval_data
from naba.nn import ActionNetwork
from naba.parameters import SensorimotorParams


@eqx.filter_value_and_grad
def unsupervised_training_loss(model: ActionNetwork,
                               key: random.PRNGKey,
                               cost_fn: Type[CostFunction],
                               priors: SensorimotorParams,
                               batch_size: int = 32,
                               num_monte_carlo_samples: int = 100):
    """ Unsupervised loss for training an action network.

    Args:
        model: neural network model
        key: jax random key
        cost_fn: cost function type (class)
        priors: OrderedDict of prior distributions
        batch_size: training data batch size
        num_monte_carlo_samples: number of samples to approximate the expectations

    Returns:
        loss (expected posterior loss)
    """
    # generate random keys for stimulus and response distributions
    s_key, r_key, data_key = random.split(key, 3)

    # generate observations and parameters
    m, sensorimotor_params, cost_params = generate_data(data_key, priors=priors, cost_fn=cost_fn,
                                                        batch_size=batch_size)

    # compute predicted action of current model
    pred_a = jax.vmap(model)(m, sensorimotor_params, cost_params)  # vectorise the model over a batch of data

    # generate log-normal posterior samples
    s = lognormal.posterior(m, sigma=sensorimotor_params.sigma,
                            sigma_0=sensorimotor_params.sigma_0,
                            mu_0=sensorimotor_params.mu_0).sample(s_key, sample_shape=(num_monte_carlo_samples,))

    # generate response distribution samples
    r = dist.LogNormal(loc=jnp.log(pred_a), scale=sensorimotor_params.sigma_r).sample(r_key, sample_shape=(
        num_monte_carlo_samples,))

    # compute average cost across samples and batch
    return cost_fn(params=cost_params)(s, r).mean()


@eqx.filter_jit
def test_loss(model: ActionNetwork, m: ArrayLike, sensorimotor_params: SensorimotorParams,
              cost_params: NamedTuple, a: ArrayLike):
    """ Evaluate test loss for an action network

    Args:
        model: neural network model
        m: sensory measurement
        theta: subject's model parameters
        a: optimal action

    Returns:
        RMSE
    """
    # compute predicted action of current model
    pred_a = jax.vmap(model)(m, sensorimotor_params, cost_params)

    # compute RMSE
    return jnp.sqrt(((a - pred_a) ** 2).mean())


def train(
        key: random.PRNGKey,
        model: ActionNetwork,
        priors: SensorimotorParams,
        cost_fn: Type[CostFunction],
        optim: optax.GradientTransformation,
        steps: int,
        print_every: int,
        batch_size: int = 32,
        num_monte_carlo_samples: int = 128,
        evaluate: bool = True,
        eval_size: int = 1_000,
        eval_data: Optional[Tuple[Array, Array, Array]] = None
) -> ActionNetwork:
    """ Train an action network

    Args:
        key: random.PRNGKey
        model: neural network model
        priors: prior distributions for the subject's parameters
        cost_fn: cost function type (class)
        optim: optax optimizer
        steps: number of training steps
        print_every: print the training and evaluation loss every .. steps
        batch_size: batch size for training data
        num_monte_carlo_samples: number of Monte Carlo samples for evaluating the expectations in unsupervised loss
        evaluate: whether to compute the evaluation loss
        eval_size: number of samples to evaluate on (only works if test_data is not given)
        eval_data: pre-computed test data (speeds up initialization for loss with no analytical solutions)

    Returns:
        trained ActionNetwork
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if evaluate:
        if eval_data is None:
            key, subkey = random.split(key)
            eval_data = generate_eval_data(subkey, priors=priors, cost_fn=cost_fn, num_samples=eval_size)

        m_test, sensorimotor_params_test, cost_params_test, a_test = eval_data

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
            model: ActionNetwork,
            key: random.PRNGKey,
            opt_state: PyTree,
    ):
        loss_value, grads = unsupervised_training_loss(model, key=key,
                                                       cost_fn=cost_fn,
                                                       batch_size=batch_size,
                                                       priors=priors,
                                                       num_monte_carlo_samples=num_monte_carlo_samples)
        # loss_value, grads = supervised_training_loss(model, m=m, theta=theta)

        # perform parameter update using gradients
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for step in range(steps):

        # split random key
        key, loss_key = random.split(key, 2)

        # perform step TODO: we might want to have early stopping
        model, opt_state, train_loss = make_step(model, loss_key, opt_state)

        # print evaluation loss
        if (step % print_every) == 0 or (step == steps - 1):
            print(
                f"{step=}, train_loss={train_loss.item()}, ",
                f"eval_loss={test_loss(model, m_test, sensorimotor_params_test, cost_params_test, a_test).item()}" if evaluate else None
            )
    return model
