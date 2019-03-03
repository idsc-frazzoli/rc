from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from .distribution import DiscreteDistribution
from .types import RNG


class UrgencyDistributionScenario(metaclass=ABCMeta):

    @abstractmethod
    def choose_distribution_for_agent(self, i: int, n: int, rng: RNG) -> DiscreteDistribution:
        pass


@dataclass
class ConstantUrgencyDistribution(UrgencyDistributionScenario):
    d: DiscreteDistribution

    """ Constant distribution for everybody. """

    def choose_distribution_for_agent(self, i: int, n: int, rng: RNG) -> DiscreteDistribution:
        return self.d


    def __repr__(self):
        return 'ConstantUrgencyDistribution: all agents have the same urgency distribution, which is:\n%r' % self.d
