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
        return '{' + ', '.join(f'{x} with probability {p}' for x, p in zip(self.values, self.probabilities)) + '}'

    @staticmethod
    def uniform(values):
        n = len(values)
        p = 1.0 / n
        p2v = []
        for v in values:
            p2v.append((v, p))
        return DiscreteDistribution(tuple(p2v))

    @staticmethod
    def dirac(value):
        p2v = ((value, 1.0),)
        return DiscreteDistribution(p2v)


def choose_pair(n, rng: RNG) -> Tuple[int, int]:
    i = rng.randint(0, n)
    j = i
    while j == i:
        j = rng.randint(0, n)
    return i, j
