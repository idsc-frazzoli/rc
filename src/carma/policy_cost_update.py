from abc import ABCMeta, abstractmethod
from typing import *

from .types import CostValue, UrgencyValue, MessageValue, RNG


class CostUpdatePolicy(metaclass=ABCMeta):

    @abstractmethod
    def update(self,
               cost: CostValue,
               urgency: UrgencyValue,
               a: int,
               messages: Tuple[MessageValue],
               who_goes: int,
               rng: RNG) -> CostValue:
        pass


class SimpleCostUpdatePolicy(CostUpdatePolicy):
    def update(self,
               cost: CostValue,
               urgency: UrgencyValue,
               a: int,
               messages: Tuple[MessageValue],
               who_goes: int,
               rng: RNG) -> CostValue:
        if a == who_goes:
            delta = 0
        else:
            delta = 1
        return cost + delta

    def __repr__(self):
        return f"SimpleCostUpdatePolicy: 0 for the first, 1 for the others."

