from abc import ABCMeta, abstractmethod
from typing import *

from .types import KarmaValue, MessageValue, RNG


class KarmaUpdatePolicy(metaclass=ABCMeta):

    @abstractmethod
    def update(self, current_karma: KarmaValue, a: int, messages: Tuple[MessageValue], who_goes: int,
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
