from typing import *
from decimal import Decimal
import numpy as np


RNG = np.random.RandomState

CostValue = NewType("CostValue", Decimal)
KarmaValue = NewType("Karma", Decimal)
UrgencyValue = NewType("UrgencyValue", Decimal)
Probability = NewType("Probability", Decimal)
MessageValue = NewType("MessageValue", Decimal)
