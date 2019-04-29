# All statistics to implement
from decimal import Decimal
import numpy as np
from .policy_agent import Globals
from .simulation import compute_karma_distribution2

def stats_avg_cum_cost(exp, history, prec=5):
    """ Final cumulative cost. """
    last = history[-1, :]['cost']
    return Decimal(np.mean(last)).__round__(prec)


def stats_avg_cum_cost_avg(exp, history, prec=5):
    """ Final cost / number of encounters each """
    last = history[-1, :]['cost_average']
    return Decimal(np.mean(last)).__round__(prec)


def stats_avg_cum_cost_std(exp, history, prec=5):
    """ stddev """
    last = history[-1, :]['cost_average']
    return Decimal(np.std(last)).__round__(prec)


def stats_std_final_karma(exp, history, prec=5):
    """ STD of final karma distribution. """
    last = history[-1, :]['karma']
    return Decimal(np.std(last)).__round__(prec)

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


# 1) average cost slo


# 2) accumulayrf
