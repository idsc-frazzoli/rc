from abc import ABCMeta, abstractmethod
from typing import *

from .types import KarmaValue, MessageValue, RNG


class KarmaUpdatePolicy(metaclass=ABCMeta):
    @abstractmethod
    def update(
        self,
        karma: Tuple[KarmaValue],
        messages: Tuple[MessageValue],
        who_goes: int,
        rng: RNG,
    ) -> Tuple[KarmaValue]:
        pass


#
# class SimpleKarmaUpdatePolicy(KarmaUpdatePolicy):
#     """ +1 if you wait, -1 if you go. """
#
#     def update(self,
#                karma: Tuple[KarmaValue],
#                messages: Tuple[MessageValue],
#                who_goes: int,
#                rng: RNG) ->  Tuple[KarmaValue]:
#
#         new_karma = list(karma)
#         for i in range(len(karma)):
#             if i == who_goes:
#                 delta = -1
#             else:
#                 delta = 1
#             new_karma[i] += delta
#
#         return tuple(new_karma)
#
#
#     def __repr__(self):
#         return 'SimpleKarmaUpdatePolicy: lose 1 if you go, gain 1 if you wait.'
#

# class SimpleKarmaUpdatePolicy_but_floor0(KarmaUpdatePolicy):
#     """ +1 if you wait, -1 if you go. """
#
#     def update(self,
#                karma: Tuple[KarmaValue],
#                messages: Tuple[MessageValue],
#                who_goes: int,
#                rng: RNG) ->  Tuple[KarmaValue]:
#         if a == who_goes:
#             delta = -1
#         else:
#             delta = 1
#
#         c = max(0, current_karma + delta)
#         return c
#
#
#     def __repr__(self):
#         return 'SimpleKarmaUpdatePolicy_but_floor0: lose 1 if you go, gain 1 if you wait. Lower bound of zero. '

import numpy as np


class BoundedKarma(KarmaUpdatePolicy):
    def __init__(self, max_carma):
        self.max_carma = max_carma

    def update(
        self,
        karma: Tuple[KarmaValue],
        messages: Tuple[MessageValue],
        who_goes: int,
        rng: RNG,
    ) -> Tuple[KarmaValue]:

        # cannot bid more than the karma
        messages = np.minimum(messages, karma)
        assert len(messages) == 2

        new_carma = np.array(karma, dtype="int")

        # for each agent
        for i in range(len(karma)):
            j = 0 if i == 1 else 1

            if i == who_goes:
                # he is the winner
                # he will lose up to max_lose
                max_lose = self.max_carma - karma[j]
                to_lose = min(messages[i], max_lose)
                new_carma[i] -= to_lose
            else:
                # he is the loser
                max_win = self.max_carma - karma[i]
                to_win = min(messages[j], max_win)
                new_carma[i] += to_win

        # the karma should be preserved
        assert np.sum(karma) == np.sum(new_carma), (karma, new_carma)
        return tuple(list(new_carma))

    def __repr__(self):
        return f"{type(self).__name__}: max_carma = {self.max_carma} "
