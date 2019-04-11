# Real = Decimal
from decimal import Decimal

import matplotlib

from carma.iterative import plot_transitions
from reprep import Report
from reprep.plot_utils import y_axis_set
from .experiment import Experiment
from .policy_agent import *
from .policy_cost_update import *
from .policy_initial import *
from .policy_karma_update import *
from .policy_urgency import *
from .policy_who_goes import *
from .simulation import run_experiment, compute_karma_distribution2
from .types import Probability
from .statistics import stats_avg_cum_cost, stats_avg_cum_cost_avg, \
stats_avg_cum_cost_std, stats_std_final_karma
from .plotting import make_figures
import seaborn

# constant = DiscreteDistribution(((UrgencyValue(4), Probability(1)),))

highlow = DiscreteDistribution(((UrgencyValue(0), Probability(0.5)),
                                (UrgencyValue(3), Probability(0.5))))

# highlowmean = DiscreteDistribution(((UrgencyValue(0), Probability(0.3)),
#                                     (UrgencyValue(1.5), Probability(0.4)),
#                                     (UrgencyValue(3), Probability(0.3))))
experiments = {}
num_agents = 200
num_days = 1000
average_encounters_per_day_per_agent = 0.1

initial_karma = RandomKarma(0, Globals.max_carma)  # lower, upper

common = dict(num_agents=num_agents,
              num_days=num_days,
              average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
              initial_karma_scenario=initial_karma, cost_update_policy=PayUrgency(),
              karma_update_policy=BoundedKarma(Globals.max_carma),
              urgency_distribution_scenario=ConstantUrgencyDistribution(highlow))

experiments['guess1'] = Experiment(desc="A guess about optimal strategy.",
                                   agent_policy_scenario=FixedPolicy(GoodGuees()),
                                   who_goes=MaxGoesIfHasKarma(), **common)

for alpha in equilibria:
    name = 'equilibrium%.2f' % alpha
    experiments[name] = Experiment(desc="Mixed equilibrium for alpha = %.2f" % alpha,
                                   agent_policy_scenario=FixedPolicy(ComputedEquilibrium(alpha)),
                                   who_goes=MaxGoesIfHasKarma(),
                                   **common
                                   )

for alpha in equilibria_pure:
    name = 'pure%.2f' % alpha
    experiments[name] = Experiment(desc="Pure equilibrium for alpha = %.2f" % alpha,
                                   agent_policy_scenario=FixedPolicy(ComputedEquilibriumPure(alpha)),
                                   who_goes=MaxGoesIfHasKarma(),
                                   **common
                                   )

experiments['centralized-urgency'] = Experiment(desc="Centralized controller chooses the one with highest urgency.",
                                                agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
                                                who_goes=MaxUrgencyGoes(),
                                                **common)

experiments['centralized-urgency-then-cost'] = Experiment(
    desc="Centralized controller chooses the one with highest urgency and if ties the one with the maximum cost.",
    agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
    who_goes=MaxUrgencyThenCost(),
    **common)

experiments['baseline-random'] = Experiment(desc="Random choice of who goes",
                                            agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
                                            who_goes=RandomWhoGoes(), **common)

experiments['bid1-always'] = Experiment(desc="The agents always bid 1",
                                        agent_policy_scenario=FixedPolicy(Bid1()),
                                        who_goes=MaxGoesIfHasKarma(),
                                        **common)
experiments['bid1-if-urgent'] = Experiment(desc="The agents bid 1 if they are urgent",
                                           agent_policy_scenario=FixedPolicy(Bid1IfUrgent()),
                                           who_goes=MaxGoesIfHasKarma(),
                                           **common)
experiments['bid-urgency'] = Experiment(desc="The agents bid their urgency",
                                        agent_policy_scenario=FixedPolicy(BidUrgency()),
                                        who_goes=MaxGoesIfHasKarma(), **common)

experiments['centralized-cost'] = Experiment(
        desc="Centralized controller chooses the agent with the highest accumulated cost.",
        agent_policy_scenario=FixedPolicy(BidUrgency()),
        who_goes=MaxCostGoes(),
        **common)

prec = 5

statistics = [
    # stats_avg_cost,
    stats_avg_cum_cost,
    stats_avg_cum_cost_avg,
    stats_avg_cum_cost_std,
    # stats_std_final_cost_avg,
    stats_std_final_karma
]

import argparse
def carma1_main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '--no-reports', action='store_true', default=False)
    parser.add_argument('--experiment', type=str, default=None)
    parsed = parser.parse_args()

    do_reports = not parsed.no_reports

    od = './out-experiments'
    fn0 = os.path.join(od, 'summary.html')
    r0 = Report('all-experiments')
    rows = []
    data = []
    cols = [x.__doc__ for x in statistics]

    if parsed.experiment is None:
        todo = list(experiments)
    else:
        todo = parsed.experiment.split(',')

    for exp_name in todo:
        exp = experiments[exp_name]

    for exp_name in todo:
        print(f'running experiment {exp_name}')
        exp = experiments[exp_name]
        rows.append(exp_name)

        history = run_experiment(exp, seed=42)

        dn = os.path.join(od, exp_name)
        if not os.path.exists(dn):
            os.makedirs(dn)

        datae = []
        for s in statistics:
            val = s(exp, history, prec)
            datae.append(val)
        data.append(datae)

        print('Creating reports...')
        if do_reports:
            r = make_figures(exp_name, exp, history)
        else:
            r = Report()
        r.table('stats', data=data, cols=cols, rows=rows)

        r.nid = exp_name
        fn = os.path.join(dn, 'partial.html')
        r.to_html(fn)
        print(f'Report written to {fn}')

        r0.add_child(r)
        r0.to_html(fn0)

    r0.table('stats', data=data, cols=cols, rows=rows)
    print(f'Complete report written to {fn0}')
    r0.to_html(fn0)


import os




if __name__ == '__main__':
    carma1_main()
