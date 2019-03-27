from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from .types import RNG, KarmaValue


class InitialKarmaScenario(metaclass=ABCMeta):
    """ How to choose the initial Karma for the agent. """

    @abstractmethod
    def choose_initial_karma_for_agent(self, i: int, n: int, rng: RNG) -> KarmaValue:
        pass


@dataclass
class SameKarma(InitialKarmaScenario):
    k: KarmaValue

    def choose_initial_karma_for_agent(self, i: int, n: int, rng: RNG) -> KarmaValue:
        return self.k

    def __repr__(self):
        return f"SameKarma: All same Karma starting at {self.k}."
import numpy as np

@dataclass
class RandomKarma(InitialKarmaScenario):
    """ Random Karma """

    l: KarmaValue
    h: KarmaValue

    def __post_init__(self):
        assert self.l <= self.h

    def choose_initial_karma_for_agent(self, i: int, n: int, rng: RNG) -> KarmaValue:
        # return np.random.randint(self.h, self.l)
        options = list(range(self.l, self.h+1))
        return rng.choice(options)
        # randi int(rng.uniform(self.h, self.l))

    def __repr__(self):
        return f"RandomKarma: Random Karma between {self.l} and {self.h}."
