from abc import ABCMeta, abstractmethod
from typing import *

from .types import KarmaValue, MessageValue, RNG


class KarmaUpdatePolicy(metaclass=ABCMeta):

    @abstractmethod
    def update(self,
               current_karma: KarmaValue,
               a: int, # your index
               messages: Tuple[MessageValue],
               who_goes: int, # index of who goes
               rng: RNG) -> KarmaValue:
        pass


class SimpleKarmaUpdatePolicy(KarmaUpdatePolicy):
    """ +1 if you wait, -1 if you go. """

    def update(self, current_karma: KarmaValue, a: int, messages: Tuple[MessageValue], who_goes: int,
               rng: RNG) -> KarmaValue:
        if a == who_goes:
            delta = -1
        else:
            delta = 1

        return current_karma + delta


    def __repr__(self):
        return 'SimpleKarmaUpdatePolicy: lose 1 if you go, gain 1 if you wait.'


class SimpleKarmaUpdatePolicy_but_floor0(KarmaUpdatePolicy):
    """ +1 if you wait, -1 if you go. """

    def update(self, current_karma: KarmaValue, a: int, messages: Tuple[MessageValue], who_goes: int,
               rng: RNG) -> KarmaValue:
        if a == who_goes:
            delta = -1
        else:
            delta = 1

        c = max(0, current_karma + delta)
        return c


    def __repr__(self):
        return 'SimpleKarmaUpdatePolicy_but_floor0: lose 1 if you go, gain 1 if you wait. Lower bound of zero. '
