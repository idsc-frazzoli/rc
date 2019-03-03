from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from .distribution import DiscreteDistribution
from .types import KarmaValue, CostValue, UrgencyValue, MessageValue, RNG


class AgentPolicy(metaclass=ABCMeta):

    @abstractmethod
    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> MessageValue:
        pass

class RandomAgentPolicy(AgentPolicy):
    """ Random agent policy """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> MessageValue:
        # random message
        return rng.uniform(0, 1)

    def __repr__(self):
        return 'RandomAgentPolicy: bid a random number'


class BidUrgency(AgentPolicy):
    """ Bids the current urgency (truthful) """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> MessageValue:
        return current_urgency

    def __repr__(self):
        return 'BidUrgency: bid the true urgency'

#####

class AgentPolicyScenario(metaclass=ABCMeta):

    @abstractmethod
    def choose_policy_for_agent(self, i: int, n: int, rng: RNG) -> AgentPolicy:
        pass




@dataclass
class FixedPolicy(AgentPolicyScenario):
    policy: AgentPolicy

    def choose_policy_for_agent(self, i: int, n: int, rng: RNG) -> AgentPolicy:
        return self.policy


    def __repr__(self):
        return 'FixedPolicy: all agents have the same policy, which is:\n%r' % self.policy

