from typing import NamedTuple
from jaxtyping import ArrayLike
from numpyro.distributions import Distribution

Parameter = ArrayLike | Distribution


class SensorimotorParams(NamedTuple):
    sigma: Parameter
    sigma_0: Parameter
    mu_0: Parameter
    sigma_r: Parameter

    def param_names(self):
        return list(self._fields)


if __name__ == '__main__':
    p = SensorimotorParams(sigma=1., sigma_0=1., mu_0=1., sigma_r=1.)
