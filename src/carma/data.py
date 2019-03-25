# Real = Decimal
from decimal import Decimal

from matplotlib import rcParams

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
from .simulation import run_experiment
from .types import Probability

# constant = DiscreteDistribution(((UrgencyValue(4), Probability(1)),))

highlow = DiscreteDistribution(((UrgencyValue(0), Probability(0.5)),
                                (UrgencyValue(3), Probability(0.5))))

# highlowmean = DiscreteDistribution(((UrgencyValue(0), Probability(0.3)),
#                                     (UrgencyValue(1.5), Probability(0.4)),
#                                     (UrgencyValue(3), Probability(0.3))))
experiments = {}
num_agents = 200
num_days = 2000
average_encounters_per_day_per_agent = 0.1

initial_karma = RandomKarma(0, Globals.max_carma)  # lower, upper

common = dict(num_agents=num_agents,
              num_days=num_days,
              average_encounters_per_day_per_agent=average_encounters_per_day_per_agent,
              initial_karma_scenario=initial_karma, cost_update_policy=PayUrgency(),
              karma_update_policy=BoundedKarma(Globals.max_carma),
              urgency_distribution_scenario=ConstantUrgencyDistribution(highlow))

experiments['equilibrium-0.75'] = Experiment(desc="Equilibrium for alpha = 0.75",
                                             agent_policy_scenario=FixedPolicy(Equilibrium075()),
                                             who_goes=MaxGoesIfHasKarma(),
                                             **common
                                             )
experiments['equilibrium-0.80'] = Experiment(desc="Equilibrium for alpha = 0.8",
                                             agent_policy_scenario=FixedPolicy(Equilibrium080()),
                                             who_goes=MaxGoesIfHasKarma(),
                                             **common
                                             )
experiments['equilibrium-0.85'] = Experiment(desc="Equilibrium for alpha = 0.85",
                                             agent_policy_scenario=FixedPolicy(Equilibrium085()),
                                             who_goes=MaxGoesIfHasKarma(),
                                             **common
                                             )

experiments['centralized-urgency'] = Experiment(desc="Centralized controller chooses the one with highest urgency.",
                                                agent_policy_scenario=FixedPolicy(RandomAgentPolicy()),
                                                who_goes=MaxUrgencyGoes(),
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


def compute_transitions_matrix_and_policy_for_urgency_nonzero(history):
    NK = Globals.max_carma + 1
    urgencies = set(history['urgency'].flatten())
    print(urgencies)

    P = np.zeros((NK, NK))
    policy = np.zeros((NK, NK))

    ntimes, nagents = history['karma'].shape
    for t in range(1, ntimes):
        for i in range(nagents):
            u = history[t, i]['urgency']
            if u == 0:
                continue
            k1 = history[t - 1, i]['karma']
            k2 = history[t, i]['karma']
            part = history[t, i]['participated']
            message = history[t, i]['message']
            if part:
                assert 0 <= message <= NK, history[t, i]
                P[k1, k2] += 1.0
                policy[k1, message] += 1

    for k in Globals.valid_karma_values:
        P[k, :] = normalize_dist(P[k, :])

    for k in Globals.valid_karma_values:
        policy[k, :] = normalize_dist(policy[k, :])

    return P, policy


def normalize_dist(p):
    s = np.sum(p)
    return p / s if s > 0 else p


def compute_karma_distribution(karma: np.ndarray):
    """

    :param karma[times, agents]:
    :return:
    """
    ntimes, nagents = karma.shape
    max_karma = Globals.max_carma
    NK = max_karma + 1
    cdf = np.zeros((ntimes, NK))

    for i in range(ntimes):
        karma_day = karma[i, :]
        h, bin_edges = np.histogram(karma_day, bins=NK, density=True)
        # h = h / (1.0 / np.sum(h))
        cdf[i, :] = h
    return cdf


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

    rcParams['backend'] = 'agg'

    style = dict(alpha=0.5, linewidth=0.3)
    K, nagents = history.shape
    time = np.array(range(K))
    sub = time[::1]

    f = r.figure(cols=4)
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

    cdf = compute_karma_distribution(history[:, :]['karma'])
    INTERVAL_STAT = 200
    karma_stationary = np.mean(cdf[-INTERVAL_STAT:, :], axis=0)

    transitions, policy = compute_transitions_matrix_and_policy_for_urgency_nonzero(history)

    with f.plot('policy', caption='Policy for high urgency') as pylab:
        plot_transitions(pylab, policy)

    with f.plot('transitions', caption='Transitions for high urgency') as pylab:
        plot_transitions(pylab, transitions)

    print(cdf.shape)
    with f.plot('karma') as pylab:

        cdf_plot = np.kron(cdf, np.ones((1, 40)))

        pylab.imshow(cdf_plot.T)

        # pylab.plot(time[sub], karma, '.', **style)
        pylab.title('karma')
        pylab.xlabel('time')
        pylab.ylabel('karma')
        pylab.gca().invert_yaxis()
        # TODO: turn off y axis

    with f.plot('karma_last') as pylab:
        # n = 10
        # for t in range(-n, -1):
        #     k = cdf[t, :]
        pylab.bar(Globals.valid_karma_values, karma_stationary)
        pylab.title('karma stationary')
        pylab.xlabel('karma')
        pylab.ylabel('p(karma)')

    sub = time > (len(time) / 4)

    caption = """ Cost vs karma phase space. """
    with f.plot('cost-karma', caption=caption) as pylab:
        for i in range(nagents):
            cost_i = history[sub, i]['cost']
            karma_i = history[sub, i]['karma']
            pylab.plot(cost_i, karma_i, '.', **style)

        pylab.title('cost/karma')

        pylab.xlabel('cost')
        pylab.ylabel('karma')

    caption = """ Agerage cost vs karma phase space. """
    with f.plot('cost_average-karma', caption=caption) as pylab:
        for i in range(nagents):
            cost_i = history[sub, i]['cost_average']
            karma_i = history[sub, i]['karma']
            pylab.plot(cost_i, karma_i, '.', **style)

        pylab.title('cost_average/karma')

        pylab.xlabel('cost_average')
        pylab.ylabel('karma')

    return r


if __name__ == '__main__':
    carma1_main()
