from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from .distribution import DiscreteDistribution
from .types import KarmaValue, CostValue, UrgencyValue, MessageValue, RNG


class Globals:
    max_carma = 12
    valid_karma_values = list(range(max_carma + 1))


class AgentPolicy(metaclass=ABCMeta):

    @abstractmethod
    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:
        pass


class RandomAgentPolicy(AgentPolicy):
    """ Random agent policy """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:
        # random message
        return DiscreteDistribution[MessageValue].uniform(Globals.valid_karma_values)

    def __repr__(self):
        return f'{type(self).__name__}: bid a random number'


class Bid1(AgentPolicy):
    """ Random agent policy """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:
        # random message
        return DiscreteDistribution[MessageValue].dirac(1)

    def __repr__(self):
        return f'{type(self).__name__}: bid 1 '


class Bid1IfUrgent(AgentPolicy):
    """ Random agent policy """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:
        if current_urgency > 0:
            return DiscreteDistribution[MessageValue].dirac(1)
        else:
            return DiscreteDistribution[MessageValue].dirac(0)

    def __repr__(self):
        return f'{type(self).__name__}: bid 1 '


class BidUrgency(AgentPolicy):
    """ Bids the current urgency (truthful) """

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:
        return DiscreteDistribution[MessageValue].dirac(current_urgency)

    def __repr__(self):
        return f'{type(self).__name__}: bid the true urgency'


class PureStrategy(AgentPolicy):
    """ Equilibrium for alpha - 0.8 """

    def __init__(self, policy):
        self.policy = policy

    def generate_message(self,
                         current_carma: KarmaValue,
                         cost_accumulated: CostValue,
                         current_urgency: UrgencyValue,
                         urgency_distribution: DiscreteDistribution, rng: RNG) -> DiscreteDistribution[MessageValue]:

        if current_urgency > 0:
            return DiscreteDistribution[MessageValue].dirac(self.policy[current_carma])
        else:
            return DiscreteDistribution[MessageValue].dirac(0)

    def __repr__(self):
        return f'{type(self).__name__}: bid according to policy {self.policy}'


#
# equilibria_ws = {
#     0.00: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#     0.30: [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6],
#     0.50: [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
#     0.70: [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
#     # 0.75:[0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
#     0.80: [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
#     0.85: [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
#     0.90: [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
#     # 0.95:[0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
#     0.98: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
# }

# equilibria with self-effect activated
equilibria = {
    0.00: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    0.30: [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6],
    0.50: [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    0.70: [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    0.75: [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
    0.80: [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    0.85: [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
    0.90: [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    0.95: [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    0.98: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
}


class ComputedEquilibrium(PureStrategy):
    def __init__(self, alpha):
        self.alpha = alpha
        policy = equilibria[alpha]
        PureStrategy.__init__(self, policy)

    def __repr__(self):
        return f'{type(self).__name__}: alpha = {self.alpha}'


class GoodGuees(PureStrategy):
    def __init__(self):
        # policy =[0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        policy = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        PureStrategy.__init__(self, policy)


#
# class Equilibrium050(PureStrategy):
#     def __init__(self):
#         # policy = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 12]
#         policy = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium070(PureStrategy):
#     def __init__(self):
#         # policy = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 10]
#         # ???
#         policy = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
#
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium075(PureStrategy):
#     def __init__(self):
#         # policy = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
#         policy = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium080(PureStrategy):
#     def __init__(self):
#         policy = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5]
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium085(PureStrategy):
#     def __init__(self):
#         policy = [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
#
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium090(PureStrategy):
#     def __init__(self):
#         policy = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
#
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium095(PureStrategy):
#     def __init__(self):
#         policy = [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3]
#
#         PureStrategy.__init__(self, policy)
#
#
# class Equilibrium098(PureStrategy):
#     def __init__(self):
#         policy = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
#
#         PureStrategy.__init__(self, policy)


#
#
# class BidAccordingToCurrent(AgentPolicy):
#     """ Bids the current urgency (truthful) """
#
#     def generate_message(self,
#                          current_carma: KarmaValue,
#                          cost_accumulated: CostValue,
#                          current_urgency: UrgencyValue,
#                          urgency_distribution: DiscreteDistribution, rng: RNG) -> MessageValue:
#
#
#         return current_urgency
#
#     def __repr__(self):
#         return 'BidUrgency: bid the true urgency'

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
        return f'{type(self).__name__}: all agents have the same policy, which is:\n%r' % self.policy
