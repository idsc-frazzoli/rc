# Real = Decimal
from decimal import Decimal

from matplotlib import rcParams

from reprep import Report
from reprep.plot_utils import y_axis_set
from .experiment import Experiment
from .policy_agent import *
from .policy_cost_update import *
from .policy_initial import *
from .policy_karma_update import *
from .policy_urgency import *
from .policy_who_goes import *
from .simulation import run_experiment
from .types import Probability

constant = DiscreteDistribution(((UrgencyValue(4), Probability(1)),))

highlow = DiscreteDistribution(((UrgencyValue(3), Probability(0.5)),
                                (UrgencyValue(5), Probability(0.5))))
experiments = {}
num_agents = 10
num_days = 10
average_encounters_per_day_per_agent = 100

desc = """

Random choice of who goes.

Urgency: high/low.

"""
experiments['baseline-random'] = Experiment(desc=desc, num_agents=num_agents,
                                            num_days=num_days,
                                            average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
                                            agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
                                            urgency_distribution_scenario=ConstantUrgencyDistribution(highlow),
                                            initial_karma_scenario=RandomKarma(0.0, 0.0),
                                            who_goes=RandomWhoGoes(),
                                            cost_update_policy=SimpleCostUpdatePolicy(),
                                            karma_update_policy=SimpleKarmaUpdatePolicy())

desc = """

Centralized controller chooses the one with highest urgency.

Urgency: high/low.

"""
experiments['centralized-urgency'] = Experiment(desc=desc, num_agents=num_agents,
                                                num_days=num_days,
                                                average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
                                                agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
                                                urgency_distribution_scenario=ConstantUrgencyDistribution(highlow),
                                                initial_karma_scenario=RandomKarma(0.0, 0.0),
                                                who_goes=MaxUrgencyGoes(),
                                                cost_update_policy=SimpleCostUpdatePolicy(),
                                                karma_update_policy=SimpleKarmaUpdatePolicy())

desc = """

Centralized controller chooses the agent with the highest accumulated cost.

Urgency: high/low.

"""
experiments['centralized-cost'] = Experiment(desc=desc, num_agents=num_agents,
                                             num_days=num_days,
                                             average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
                                             agent_policy_scenario=FixedPolicy(BidUrgency()),
                                             urgency_distribution_scenario=ConstantUrgencyDistribution(highlow),
                                             initial_karma_scenario=RandomKarma(0.0, 0.0),
                                             who_goes=MaxCostGoes(),
                                             cost_update_policy=SimpleCostUpdatePolicy(),
                                             karma_update_policy=SimpleKarmaUpdatePolicy())

desc = """

The agents can bid with karma (if they have it)

Urgency: high/low.

"""
experiments['karma-bid-floor'] = Experiment(desc=desc, num_agents=num_agents,
                                            num_days=num_days,
                                            average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
                                            agent_policy_scenario=FixedPolicy(BidUrgency()),
                                            urgency_distribution_scenario=ConstantUrgencyDistribution(highlow),
                                            initial_karma_scenario=RandomKarma(0.0, 0.0),
                                            who_goes=MaxGoesIfHasKarma(),
                                            cost_update_policy=SimpleCostUpdatePolicy(),
                                            karma_update_policy=SimpleKarmaUpdatePolicy())

prec = 3

def stats_avg_cost(exp, history):
    """ Mean average cost. """
    return Decimal(np.mean(np.mean(history[:, :]['cost']))).__round__(prec)


def stats_std_final_cost_avg(exp, history):
    """ STD of final average cost distribution. """
    last = history[-1, :]['cost_average']
    return Decimal(np.std(last)).__round__(prec)


def stats_std_final_karma(exp, history):
    """ STD of final karma distribution. """
    last = history[-1, :]['karma']
    return Decimal(np.std(last)).__round__(prec)


statistics = [
    stats_avg_cost,
    stats_std_final_cost_avg,
    stats_std_final_karma
]


def carma1_main():
    od = './out-experiments'
    fn0 = os.path.join(od, 'index.html')
    r0 = Report('all-experiments')
    rows = []
    data = []

    for exp_name, exp in experiments.items():
        rows.append(exp_name)

        history = run_experiment(exp)

        dn = os.path.join(od, exp_name)
        if not os.path.exists(dn):
            os.makedirs(dn)

        r = make_figures(exp_name, exp, history)
        fn = os.path.join(dn, 'index.html')
        r.to_html(fn)
        print(f'Report written to {fn}')

        r.nid = exp_name
        r0.add_child(r)
        r0.to_html(fn0)

        datae = []
        for s in statistics:
            val = s(exp, history)
            datae.append(val)
        data.append(datae)

    cols = [x.__doc__ for x in statistics]
    r0.table('stats', data=data, cols=cols, rows=rows)
    print(f'Complete report written to {fn0}')
    r0.to_html(fn0)


import os


def make_figures(name: str, exp: Experiment, history) -> Report:
    r = Report(name)

    data = ""
    for k in exp.__annotations__:
        v = getattr(exp, k)
        if hasattr(v, '__desc__'):
            data += f'{k}:: {v.__desc__}'
        else:
            data += f'\n{k}: {v}\n'

    r.text('description', str(data))
    # Experiment(num_agents=num_agents,
    #            num_days=num_days,
    #            average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
    #            agent_policy_scenario=FixedPolicy(BidUrgency()),
    #            urgency_distribution_scenario=ConstantUrgencyDistribution(highlow),
    #            initial_karma_scenario=RandomKarma(0.0, 0.0),
    #            who_goes=MaxGoesIfHasKarma(),
    #            cost_update_policy=SimpleCostUpdatePolicy(),
    #            karma_update_policy=SimpleKarmaUpdatePolicy())

    rcParams['backend'] = 'agg'

    style = dict(alpha=0.5, linewidth=0.3)
    K, nagents = history.shape
    time = np.array(range(K))
    sub = time[::1]

    f = r.figure(cols=2)
    caption = 'Cumulative cost'
    with f.plot('cost_cumulative', caption=caption) as pylab:
        cost = history[sub, :]['cost']
        pylab.plot(time[sub], cost, **style)
        pylab.title('cost')
        pylab.ylabel('cost')
        pylab.xlabel('time')

    caption = 'Average cost (cumulative divided by time). Shown for the latter part of trajectory'
    with f.plot('cost_average', caption=caption) as pylab:

        cost = history[sub, :]['cost_average']
        last = history[-1, :]['cost_average']
        m = np.median(last)

        # m1, m2 = np.percentile(one, q=[3,97])

        pylab.plot(time[sub], cost, **style)

        y_axis_set(pylab, m / 2, m * 2)
        pylab.title('average cost')
        pylab.ylabel('average cost')
        pylab.xlabel('time')

    with f.plot('karma') as pylab:
        karma = history[sub, :]['karma']
        pylab.plot(time[sub], karma, **style)
        pylab.title('karma')
        pylab.xlabel('time')
        pylab.ylabel('karma')

    sub = time > (len(time) / 4)

    caption = """ Cost vs karma phase space. """
    with f.plot('cost-karma', caption=caption) as pylab:
        for i in range(nagents):
            cost_i = history[sub, i]['cost']
            karma_i = history[sub, i]['karma']
            pylab.plot(cost_i, karma_i, **style)

        pylab.title('cost/karma')

        pylab.xlabel('cost')
        pylab.ylabel('karma')

    caption = """ Agerage cost vs karma phase space. """
    with f.plot('cost_average-karma', caption=caption) as pylab:
        for i in range(nagents):
            cost_i = history[sub, i]['cost_average']
            karma_i = history[sub, i]['karma']
            pylab.plot(cost_i, karma_i, **style)

        pylab.title('cost_average/karma')

        pylab.xlabel('cost_average')
        pylab.ylabel('karma')

    return r


if __name__ == '__main__':
    carma1_main()
