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


def stats_avg_cum_cost(exp, history):
    """ Final cumulative cost. """
    last = history[-1, :]['cost']
    return Decimal(np.mean(last)).__round__(prec)


def stats_avg_cum_cost_avg(exp, history):
    """ Final cost / number of encounters each """
    last = history[-1, :]['cost_average']
    return Decimal(np.mean(last)).__round__(prec)


def stats_avg_cum_cost_std(exp, history):
    """ stddev """
    last = history[-1, :]['cost_average']
    return Decimal(np.std(last)).__round__(prec)


def stats_std_final_karma(exp, history):
    """ STD of final karma distribution. """
    last = history[-1, :]['karma']
    return Decimal(np.std(last)).__round__(prec)


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
            val = s(exp, history)
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


def compute_transitions_matrix_and_policy_for_urgency_nonzero(history):
    NK = Globals.max_carma + 1
    urgencies = set(history['urgency'].flatten())
    print(urgencies)

    P = np.zeros((NK, NK))
    policy = np.zeros((NK, NK))

    ntimes, nagents = history['karma'].shape
    for i in range(nagents):
        yes = np.logical_and(history[:, i]['urgency'] > 0,
                             history[:, i]['participated'])
        yes = yes.flatten()
        # print(yes)
        # print (np.argwhere(yes).flatten())
        interesting = np.argwhere(yes).flatten()
        for t in interesting[1:]:
            k1 = history[t - 1, i]['karma']
            k2 = history[t, i]['karma']
            # part = history[t, i]['participated']
            message = history[t, i]['message']

            # assert 0 <= message <= NK, history[t, i]
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
        # h, bin_edges = np.histogram(karma_day, bins=NK, density=True)
        h = compute_karma_distribution2(karma_day)
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

    matplotlib.use('cairo')
    # RepRepDefaults.default_image_format = MIME_SVG
    # RepRepDefaults.save_extra_png = False

    style = dict(alpha=0.5, linewidth=0.3)
    K, nagents = history.shape
    time = np.array(range(K))
    sub = time[::1]

    f = r.figure(cols=4)

    # caption = 'avg number of encounters'
    # with f.plot('avg_encounters', caption=caption) as pylab:
    #     mean_encounters = np.mean(history['encounters'].astype('float64'), axis=1)
    #     pylab.plot(mean_encounters, **style)
    #     pylab.title('mean_encounters')
    #     pylab.ylabel('encounters')
    #     pylab.xlabel('time')
    #
    # with f.plot('avg_encounters_first', caption=caption) as pylab:
    #     mean_encounters = np.mean(history['encounters_first'].astype('float64'), axis=1)
    #     pylab.plot(mean_encounters, **style)
    #     pylab.title('encounters first')
    #     pylab.ylabel('encounters')
    #     pylab.xlabel('time')

    if False:
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
    from .simulation import  compute_karma_distribution2
    cdf = compute_karma_distribution(history[:, :]['karma'])
    INTERVAL_STAT = 200
    karma_first = compute_karma_distribution2(history[0, :]['karma'])
    karma_last = compute_karma_distribution2(history[-1, :]['karma'])

    karma_stationary = np.mean(cdf[-INTERVAL_STAT:, :], axis=0)

    transitions, policy = compute_transitions_matrix_and_policy_for_urgency_nonzero(history)

    with f.plot('policy', caption='Policy for high urgency') as pylab:
        plot_transitions(pylab, policy)

    with f.plot('transitions', caption='Transitions for high urgency') as pylab:
        plot_transitions(pylab, transitions)

    with f.plot('num_encounters', caption='Number of encounters') as pylab:
        pylab.hist(history[-1, :]['encounters'], density='True')
        pylab.xlabel('num encounters')




    mean_karma = np.mean(history['karma'], axis=1)
    std_karma = np.std(history['karma'], axis=1)
    with f.plot('total_karma') as pylab:
        pylab.plot(mean_karma, 'b-', **style)
    with f.plot('std_karma') as pylab:
        pylab.plot(std_karma, 'b-', **style) 


    with f.plot('karma') as pylab:

        cdf_plot = np.kron(cdf, np.ones((1, 40)))

        pylab.imshow(cdf_plot.T)

        # pylab.plot(time[sub], karma, '.', **style)
        pylab.title('karma')
        pylab.xlabel('time')
        pylab.ylabel('karma')
        pylab.gca().invert_yaxis()
        # TODO: turn off y axis

    f = r.figure('karma-dist', caption='Karma distribution')
    with f.plot('karma_initial') as pylab:
        # n = 10
        # for t in range(-n, -1):
        #     k = cdf[t, :]
        pylab.bar(Globals.valid_karma_values, karma_first)
        pylab.title('karma at time 0')
        pylab.xlabel('karma')
        pylab.ylabel('p(karma)')

    with f.plot('karma_last') as pylab:
        # n = 10
        # for t in range(-n, -1):
        #     k = cdf[t, :]
        pylab.bar(Globals.valid_karma_values, karma_last)
        pylab.title('final karma')
        pylab.xlabel('karma')
        pylab.ylabel('p(karma)')

    with f.plot('karma_stat') as pylab:
        # n = 10
        # for t in range(-n, -1):
        #     k = cdf[t, :]
        pylab.bar(Globals.valid_karma_values, karma_stationary)
        pylab.title('karma stationary')
        pylab.xlabel('karma')
        pylab.ylabel('p(karma)')



    if False:
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
