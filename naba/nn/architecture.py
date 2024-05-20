from typing import Callable, Type, Tuple, NamedTuple

import equinox as eqx
import jax
from jax import random, numpy as jnp
from jaxtyping import ArrayLike

from naba.costs import CostFunction
from naba.parameters import SensorimotorParams


class ActionNetwork(eqx.Module):
    """ Neural network architecture for optimal actions.
    """
    layers: list
    activation: Callable

    def __init__(self, key: random.PRNGKey,
                 cost_fn: Type[CostFunction],
                 num_hidden_units: Tuple[int] = (16, 64, 16, 8),
                 activation: Callable = jax.nn.swish):
        """

        Args:
            key: jax.random.PRNGKey
            cost_fn: CostFunction type
            num_hidden_units: tuple of ints specifying the number of hidden units
            activation: activation function for hidden layers
        """
        num_hidden_layers = len(num_hidden_units)
        keys = random.split(key, num_hidden_layers + 1)

        num_params = 4 + cost_fn.num_params()

        # These contain trainable parameters.
        layers = [eqx.nn.Linear(num_params, num_hidden_units[0], key=keys[0])]
        for i in range(num_hidden_layers - 1):
            # the input size is always the output of the previous layer
            layers.append(eqx.nn.Linear(num_hidden_units[i], num_hidden_units[i + 1], key=keys[i + 1]))

        # the final layer has 3 output neurons, because we use it to parameterize a specific function of m
        layers.append(eqx.nn.Linear(num_hidden_units[num_hidden_layers - 1], 3, key=keys[-1]))

        self.layers = layers

        self.activation = activation

    def __call__(self, m: ArrayLike, sensorimotor_params: SensorimotorParams, cost_params: NamedTuple):
        """ Compute the optimal action for a given sensory measurement and set of parameters.

        The final layer has a specific functional form (see below).

        Args:
            m: sensory measurment
            theta: array of parameters

        Returns:
            optimal action (same shape as m)
        """

        theta = jnp.stack((*sensorimotor_params, *cost_params))

        # in the first layer, we simply feed in the parameters
        x = self.activation(self.layers[0](theta))

        # in the second until second-to-last layer, we feed in the output from the previous layer and the parameters
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))

        # in the last layer, we have no activation function, but instead use the custom functional form below
        x = self.layers[-1](x)

        # finally, we assume that the output is of a particular functional form
        # a * m ^ b + c
        # with a > 0 and b < 1
        # This is based on the functional form we have analytically derived for certain costs
        return jax.nn.softplus(jax.nn.softplus(x[0]) * m ** jax.nn.sigmoid(x[1]) + x[2])
