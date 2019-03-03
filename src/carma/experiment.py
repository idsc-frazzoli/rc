from dataclasses import dataclass

from .policy_agent import AgentPolicyScenario
from .policy_cost_update import CostUpdatePolicy
from .policy_initial import InitialKarmaScenario
from .policy_karma_update import KarmaUpdatePolicy
from .policy_urgency import UrgencyDistributionScenario
from .policy_who_goes import WhoGoes


@dataclass
class Experiment:
    desc: str
    num_agents: int
    num_days: int
    average_encounters_per_day_per_agent: float
    agent_policy_scenario: AgentPolicyScenario
    urgency_distribution_scenario: UrgencyDistributionScenario
    initial_karma_scenario: InitialKarmaScenario
    who_goes: WhoGoes
    cost_update_policy: CostUpdatePolicy
    karma_update_policy: KarmaUpdatePolicy

