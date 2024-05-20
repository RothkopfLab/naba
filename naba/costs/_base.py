from typing import NamedTuple
from abc import ABC, abstractmethod
from jax.tree_util import register_pytree_node_class
from numpyro.distributions import LogNormal
from jaxtyping import ArrayLike


class CostParams(NamedTuple):
    pass


@register_pytree_node_class
class CostFunction(ABC):
    """ Abstract base class for the cost function.

    Each concrete cost function must implement at leas the __call__ method, which is the cost function itself.

    It can optionally implement the optimal_estimate under a log-normal posterior
    and the optimal_action uncer a log-normal posterior and log-normal action variability .

    If __call__ depends on parameters, the parameter names and priors need to be defined as class attributes.

    The functions tree_flatten and tree_unflatten provide some jax magic so that CostFunction can be registered as
    a PyTree and we can differentiate wrt the parameters self.params (which are treated as children in the PyTree)

    """
    param_type = CostParams
    param_priors = param_type()

    def __init__(self, params=None):
        self.params = params if params is not None else self.param_type()

    @abstractmethod
    def __call__(self, state: ArrayLike, response: ArrayLike) -> ArrayLike:
        pass

    @classmethod
    def num_params(cls) -> int:
        return len(cls.param_type._fields)

    @classmethod
    def param_names(cls) -> list[str]:
        return list(cls.param_type._fields)

    def optimal_estimate(self, posterior_dist: LogNormal) -> ArrayLike:
        raise NotImplementedError

    def optimal_action(self, posterior_dist: LogNormal, sigma_r: ArrayLike = 0.) -> ArrayLike:
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.params)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
