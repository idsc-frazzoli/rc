from abc import ABCMeta, abstractmethod
from typing import *

import numpy as np

from .types import MessageValue, RNG, KarmaValue, CostValue, UrgencyValue


class WhoGoes(metaclass=ABCMeta):

    @abstractmethod
    def who_goes(self,
                 karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue],
                 messages: Tuple[MessageValue],
                 urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        """ Return the index of the agent that should go first. """


class RandomWhoGoes(WhoGoes):
    """ Random choice of who goes. """

    def who_goes(self, karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue], messages: Tuple[MessageValue], urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        # sample random numbers
        n = len(messages)
        # choose the highest
        r = rng.uniform(size=n)
        return int(np.argmax(r))

    def __repr__(self):
        return 'RandomWhoGoes: choosing who goes randomly'


class MaxMessageGoes(WhoGoes):
    """ The one with the highest message goes. """

    def who_goes(self,
                 karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue],
                 messages: Tuple[MessageValue],
                 urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        # add small epsilon to avoid ties / bias
        messages = perturb(messages, rng=rng)
        return int(np.argmax(messages))

    def __repr__(self):
        return 'MaxMessageGoes: who bid the max goes'

class MaxUrgencyGoes(WhoGoes):
    """ The one with the highest message goes. """

    def who_goes(self, karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue],
                 messages: Tuple[MessageValue],
                 urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        # add small epsilon to avoid ties / bias
        urgencies = perturb(urgencies, rng=rng)
        return int(np.argmax(urgencies))

    def __repr__(self):
        return 'MaxUrgencyGoes: centralized chooses the one with max urgency'


class MaxCostGoes(WhoGoes):
    """ The one with the highest message goes. """

    def who_goes(self, karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue],
                 messages: Tuple[MessageValue],
                 urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        # add small epsilon to avoid ties / bias
        costs = perturb(costs, rng=rng)
        return int(np.argmax(costs))

    def __repr__(self):
        return 'MaxCostGoes: centralized chooses the one with max accumulated cost'


def perturb(x, rng, epsilon=0.01):
    """ add a very small random value so that we avoid ties """
    x = np.array(x)
    n = x.shape[0]
    extra = rng.uniform(0, epsilon, n)
    return extra + x


class MaxGoesIfHasKarma(WhoGoes):
    """ The one with the highest message goes if it could pay the Karma. """

    def who_goes(self, karmas: Tuple[KarmaValue],
                 costs: Tuple[CostValue],
                 messages: Tuple[MessageValue],
                 urgencies: Tuple[UrgencyValue],
                 rng: RNG) -> int:
        # you cannot bid more than the karma
        actual_messages = np.minimum(karmas, messages)

        # add small epsilon to avoid ties / bias
        actual_messages = perturb(actual_messages, rng=rng)

        return int(np.argmax(actual_messages))

    def __repr__(self):
        return 'MaxGoesIfHasKarma: you can bid only the karma you have. Who bid the max goes.'
