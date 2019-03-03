from dataclasses import dataclass
from typing import *

import numpy as np
from numpy.testing import assert_allclose

from .types import Probability, RNG

X = TypeVar('X')


@dataclass
class DiscreteDistribution(Generic[X]):
    """ A discrete distribution over values of type X. """
    prob2value: Tuple[Tuple[X, Probability], ...]

    def __post_init__(self):
        self.probabilities = tuple(p for x, p in self.prob2value)
        self.values = tuple(x for x, p in self.prob2value)
        assert_allclose(np.sum(self.probabilities), 1.0)

    def sample(self, rng: RNG) -> X:
        """ Samples one of the values from the distribution. """

        return rng.choice(self.values, p=self.probabilities)

    def mean(self):
        return np.average(self.values, weights=self.probabilities)

    def __repr__(self):
        return '{' +', '.join(f'{x} with probability {p}' for x, p in zip(self.values, self.probabilities)) +'}'

def choose_pair(n) -> Tuple[int, int]:
    i = np.random.randint(0, n)
    j = i
    while j == i:
        j = np.random.randint(0, n)
    return i, j
