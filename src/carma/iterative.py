import dataclasses
import itertools
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import *

import matplotlib
import numpy as np
import yaml
from numpy.testing import assert_allclose

from reprep import Report, posneg, RepRepDefaults, MIME_PNG


class Assignment(Enum):
    FirstPrice = 1
    SecondPrice = 2
    HalfPrice = 3


@dataclass
class Model:
    description: str
    max_karma: int
    prob_high: float
    alpha: float
    urgency0: float

    assignment: Assignment

    p_win_if_bid_more: float = 1.0
    mix: float = 0.0
    ties_won_by_higher: bool = False

    prob_low: Optional[float] = None

    valid_karma_values: Optional[np.ndarray] = None
    distinct_karma_values: Optional[int] = None

    def __post_init__(self):
        self.distinct_karma_values = self.max_karma + 1
        self.prob_low = 1.0 - self.prob_high
        self.valid_karma_values = list(range(0, self.max_karma + 1))
        assert max(self.valid_karma_values) == self.max_karma

    def initialize_caches(self):
        N = self.distinct_karma_values
        self.probability_of_winning_ = np.zeros((N, N, N, N), dtype='float64')
        self.next_karma_if_win = np.zeros((N, N, N, N), dtype='int')
        self.next_karma_if_lose = np.zeros((N, N, N, N), dtype='int')

        print('initializing caches')
        for k_i, m_i, k_j, m_j in itertools.product(range(N), range(N), range(N), range(N)):
            self.probability_of_winning_[k_i, m_i, k_j, m_j] = probability_of_winning(self, k_i, m_i, k_j, m_j)

            w, l = delta_karma_if_i_wins_or_lose(self, k_i, m_i, k_j, m_j)
            self.next_karma_if_win[k_i, m_i, k_j, m_j] = w
            self.next_karma_if_lose[k_i, m_i, k_j, m_j] = l
        print('done initializing caches')


@dataclass
class Optimization:
    # energy_factor: Optional[float]
    # energy_factor_delta: Optional[float]
    # energy_factor_max: Optional[float]
    energy_factor_schedule: Tuple[float, ...]

    num_iterations: int
    inertia: float
    diff_threshold: float
    consider_self_effect: bool

    regularize_utility_monotone: bool
    regularize_marginal_utility_monotone: bool


@dataclass
class Simulation:
    model: Model
    opt: Optimization


assertions = False


def assert_pd(x):
    if not assertions:
        return
    assert np.all(x >= 0), x
    assert np.allclose(np.sum(x), 1), x


def assert_good_transitions(M):
    if not assertions:
        return

    n1, n2 = M.shape
    assert n1 == n2
    for i in range(n1):
        cond = M[i, :]
        assert_allclose(np.sum(cond), 1.0)


def assert_good_policy(p):
    if not assertions:
        return

    n1, n2 = p.shape
    assert n1 == n2
    for i in range(n1):
        pol = p[i, :]
        assert_allclose(np.sum(pol), 1.0)
        for j in range(n2):
            if j > i:
                assert pol[j] == 0


def assert_valid_karma_value(model: Model, p):
    if not assertions:
        return

    assert 0 <= p <= model.max_karma, p


@dataclass
class Iteration:
    policy: np.ndarray
    policy_inst: np.ndarray
    stationary_karma_pd: np.ndarray
    stationary_karma_pd_inst: np.ndarray
    utility: np.ndarray
    energy_factor: float

    debug_utilities: Optional[Any] = None
    transitions: Optional[Any] = None

    diff: float = 0
    average_karma: float = None
    global_utility: float = None
    social_utility: float = None
    expected_cost_today_per_karma: np.ndarray = None

    def __post_init__(self):
        N = self.stationary_karma_pd.size
        assert self.policy.shape == (N, N)
        assert_good_policy(self.policy)
        assert_pd(self.stationary_karma_pd)

        if self.transitions is not None:
            n = self.transitions.shape[0]
            # print(np.sum(self.transitions, axis=1))
            for i in range(n):
                tp = np.sum(self.transitions[i, :])
                assert np.allclose(tp, 1.0), tp


from typing import *


def probability_of_winning(model: Model, k_i, m_i, k_j, m_j) -> float:
    """ Returns the probability of agent i winning
        and 1- that probability."""
    if m_i == m_j:
        if model.ties_won_by_higher:
            if k_i > k_j:
                pwin, plose = 1.0, 0.0
            elif k_i < k_j:
                pwin, plose = 0.0, 1.0
            elif k_i == k_j:
                pwin, plose = 0.5, 0.5
            else:
                assert False
        else:
            pwin, plose = 0.5, 0.5
    elif m_i < m_j:
        plose = model.p_win_if_bid_more
        pwin = 1 - plose
    elif m_i > m_j:
        pwin = model.p_win_if_bid_more
        plose = 1 - pwin
    else:
        assert False

    # assert_allclose(pwin + plose, 1.0)

    return pwin


def delta_karma_if_i_wins_or_lose(model: Model, k_i, m_i, k_j, m_j) -> Tuple[float, float]:
    """ Returns delta karma for agent i if it wins or loses."""

    if model.assignment == Assignment.FirstPrice:

        # conservation of karma: I can only lose up to what the other can win
        next_karma_if_wins = k_i - min(m_i, model.max_karma - k_j)
        next_karma_if_loses = min(k_i + m_j, model.max_karma)


    elif model.assignment == Assignment.SecondPrice:
        # second price: I pay m_j
        next_karma_if_wins = k_i - min(m_j, model.max_karma - k_j)
        # I gain m_i
        next_karma_if_loses = min(k_i + m_i, model.max_karma)
    elif model.assignment == Assignment.HalfPrice:
        # second price: I pay m_j
        half_price = int(np.ceil(0.5 * m_i + 0.5 * m_j))
        next_karma_if_wins = k_i - min(half_price, model.max_karma - k_j)
        # I gain m_i
        next_karma_if_loses = min(k_i + half_price, model.max_karma)
    else:
        assert False

    return next_karma_if_wins, next_karma_if_loses


def consider_bidding(model: Model, stationary_karma_pd, utility, policy, k_i, m_i,
                     consider_self_effect) -> \
        Tuple[float, float]:
    """
        What would happen if we bid m_i when we are k_i and high?

    returns expected_utility_of_m_i, expected_cost_today_of_m_i
    """
    expected_utility_of_m_i = 0.0
    expected_cost_today_of_m_i = 0.0
    # I can bid up to m_i
    assert 0 <= m_i <= k_i
    next_karma_if_win = model.next_karma_if_win
    next_karma_if_lose = model.next_karma_if_lose
    probability_of_winning_ = model.probability_of_winning_
    alpha = model.alpha
    urgency0 = model.urgency0
    # for each karma of the other
    for k_j in model.valid_karma_values:
        # probability that they have this karma
        p_k_j = stationary_karma_pd[k_j]

        if p_k_j == 0:
            continue

        # first, account for type "low"
        # they bid 0
        m_j = 0
        pwin_if_low = probability_of_winning_[k_i, m_i, k_j, m_j]
        plose_if_low = 1.0 - pwin_if_low

        next_karma_if_low_and_win = next_karma_if_win[k_i, m_i, k_j, m_j]
        next_karma_if_low_and_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]

        utility_if_low_and_lose = alpha * utility[next_karma_if_low_and_lose]
        utility_if_low_and_win = alpha * utility[next_karma_if_low_and_win]

        P = p_k_j * model.prob_low
        expected_cost_today_of_m_i += P * plose_if_low * (-urgency0)

        expected_utility_of_m_i += P * (
                plose_if_low * (-urgency0 + utility_if_low_and_lose) +
                pwin_if_low * (0 + utility_if_low_and_win)
        )

        # now account for type "high"

        # all possible karmas
        for m_j in range(0, k_j + 1):

            if k_j == k_i and consider_self_effect:
                p_m_j_given_k_j = 1 if m_j == m_i else 0
            else:
                # with this probability
                p_m_j_given_k_j = policy[k_j, m_j]
            if p_m_j_given_k_j == 0:
                continue

            pwin = probability_of_winning_[k_i, m_i, k_j, m_j]
            plose = 1.0 - pwin
            next_karma_if_high_and_win = next_karma_if_win[k_i, m_i, k_j, m_j]
            next_karma_if_high_and_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]
            utility_if_high_and_win = alpha * utility[next_karma_if_high_and_win]
            utility_if_high_and_lose = alpha * utility[next_karma_if_high_and_lose]

            P = p_k_j * model.prob_high * p_m_j_given_k_j
            expected_utility_of_m_i += P * \
                                       (plose * (- urgency0 + utility_if_high_and_lose) +
                                        pwin * (0 + utility_if_high_and_win))
            expected_cost_today_of_m_i += P * (plose * (-urgency0) + pwin * 0)

    return expected_utility_of_m_i, expected_cost_today_of_m_i


def find_best_action2(x):
    cur = 0
    for i, v in enumerate(x):
        if v > x[cur]:
            cur = i
        if v < x[cur]:
            break
    return cur


def policy_given_utilities(model: Model, expected_utilities, energy_factor: float) -> np.ndarray:
    N = model.distinct_karma_values
    U = np.copy(expected_utilities)
    U[np.isnan(U)] = -np.inf

    # if True:
    #     best_action = np.argmax(U)
    #     u_best = U[best_action]
    #     U[best_action] = -np.inf
    #     second_best = np.argmax(U)
    #     u_second_best = U[second_best]
    #     delta = (u_best - u_second_best) / np.abs(u_best)
    #
    #     p_best = 1.0
    #     p_second = np.exp(-delta)
    #     policy = np.zeros(N, dtype='float64')
    #     policy[best_action] = p_best
    #     policy[second_best] = p_second
    #
    # else:

    np.seterr(under='ignore')

    # best_action = np.argmax(U)
    best_action = find_best_action2(U)
    policy_sharp = np.zeros(N, dtype='float64')
    policy_sharp[best_action] = 1.0

    policy_sharp = policy_sharp / np.sum(policy_sharp)

    policy_smooth = np.exp(expected_utilities)
    policy_smooth[np.isnan(expected_utilities)] = 0.0

    policy_smooth = policy_smooth / np.sum(policy_smooth)
    # policy_smooth = remove_underflow(policy_smooth)

    q = float(energy_factor)
    assert 0 <= q <= 1, q

    policy = policy_smooth * (1 - q) + q * policy_sharp

    policy = policy / np.sum(policy)

    assert_pd(policy)
    return policy


def normalize_affine(x0):
    # print(x)
    x = x0 - np.nanmin(x0)
    # print(x)
    m = np.nanmax(x)
    if m == 0:
        return x0
    # print(m)
    return x / m


# def remove_underflow(dist, min_pd=0.001):
#     """ Removes the values of a p.d. that would create underflow later. """
#     dist = np.copy(dist)
#     dist[dist < min_pd] = 0
#     dist = dist / np.sum(dist)
#     return dist


def compute_expectation(p, values):
    N = p.size
    assert N == values.size
    for i in range(N):
        if np.isnan(values[i]):
            assert p[i] == 0
    values = np.copy(values)
    values[np.isnan(values)] = 0
    return np.dot(p, values)


def iterate(sim: Simulation, it: Iteration, energy_factor: float, it_ef: int, consider_self_effect: bool) -> Iteration:
    # need to find for each karma
    N = sim.model.distinct_karma_values
    policy2 = np.zeros((N, N), dtype='float64')
    debug_utilities = np.zeros((N, N), dtype='float64')
    debug_utilities.fill(np.nan)
    expected_cost_per_karma = np.zeros(sim.model.distinct_karma_values, 'float64')
    expected_cost_today_per_karma = np.zeros(sim.model.distinct_karma_values, 'float64')

    for k_i in sim.model.valid_karma_values:
        expected_utilities = np.zeros(N, dtype='float64')
        expected_utilities.fill(np.nan)
        expected_cost_today = np.zeros(N, dtype='float64')
        expected_cost_today.fill(np.nan)
        for m_i in range(0, k_i + 1):
            eu, ec = consider_bidding(sim.model, stationary_karma_pd=it.stationary_karma_pd,
                                      utility=it.utility, policy=it.policy, k_i=k_i, m_i=m_i,
                                      consider_self_effect=consider_self_effect)
            #expected_cost_today[m_i] = ec
            expected_cost_today[m_i] = ec / 2
            expected_utilities[m_i] = eu

        # # FIXME: trying to find off-by-one error bu
        # if k_i == sim.model.max_karma:
        #     expected_cost_today[-1] = expected_cost_today[-2]
        #     expected_utilities[-1] = expected_utilities[-2]

        policy_k_i = policy_given_utilities(model=sim.model, expected_utilities=expected_utilities,
                                            energy_factor=energy_factor)
        assert_pd(policy_k_i)

        expected_cost_per_karma[k_i] = compute_expectation(policy_k_i, expected_utilities)
        expected_cost_today_per_karma[k_i] = compute_expectation(policy_k_i, expected_cost_today)

        debug_utilities[k_i, :] = expected_utilities

        policy2[k_i, :] = policy_k_i
    # update randomly
    # p_update = 1.0
    # stay_constant = np.random.uniform(0, 1, sim.model.distinct_karma_values) > p_update
    # policy2[stay_constant, :] = it.policy[stay_constant, :]
    #
    #
    # if False:
    #     policy0 = np.copy(policy2)
    #     for i in exp.valid_karma_values:
    #         if i > 0 and i < exp.max_karma:
    #             policy2[i, :] = 0.2 * policy0[i - 1, :] + 0.6 * policy0[i, :] + 0.2 * policy0[i + 1, :]
    #         elif i == exp.max_karma:
    #             policy2[i, :] = 0.4 * policy0[i - 1, :] + 0.6 * policy0[i, :]
    # FIXME: fixing bug
    # policy2[exp.max_karma, :] = policy2[exp.max_karma-1, :]

    policy2_inst = policy2
    q = sim.opt.inertia

    # if this is the first iteration of a new ef, do not use inertia
    # if it_ef == 0:
    #     q = 1

    policy2 = q * policy2_inst + (1 - q) * it.policy

    # with timeit('compute_transitions'):
    transitions = compute_transitions(sim.model, policy2, it.stationary_karma_pd)

    # with timeit('solveStationary'):
    stationary_karma_pd2 = solveStationary(transitions)

    # print(stationary_karma_pd2)

    # for i in exp.valid_karma_values:
    #     stationary_karma_pd2[i] = (exp.max_karma/2.0) - i
    # stationary_karma_pd2.fill(1.0)
    # stationary_karma_pd2 = stationary_karma_pd2 / np.sum(stationary_karma_pd2)

    utility2 = solveStationaryUtility(sim.opt, sim.model, transitions, expected_cost_today_per_karma)
    # make a delta adjustment

    # utility2 = q * utility2 + (1 - q) * it.utility
    # stationary_karma_pd2_final = q * stationary_karma_pd2 + (1 - q) * it.stationary_karma_pd

    global_utility = float(np.dot(stationary_karma_pd2, utility2))
    average_karma = float(np.dot(stationary_karma_pd2, sim.model.valid_karma_values))
    social_utility = float(np.dot(stationary_karma_pd2, expected_cost_today_per_karma))

    # r = 0
    # policy2 = get_random_policy(exp) * r + (1 - r) * policy2
    return Iteration(policy=policy2,
                     policy_inst=policy2_inst,
                     stationary_karma_pd=stationary_karma_pd2,
                     stationary_karma_pd_inst=stationary_karma_pd2,
                     debug_utilities=debug_utilities, utility=utility2,
                     expected_cost_today_per_karma=expected_cost_today_per_karma,
                     social_utility=social_utility,
                     transitions=transitions, energy_factor=energy_factor,
                     average_karma=average_karma,
                     global_utility=global_utility)


def compute_transitions(model: Model, policy, stationary_karma_pd):
    assert_pd(stationary_karma_pd)

    transitions_high = compute_transitions_high(model, policy, stationary_karma_pd)

    transitions_low = compute_transitions_low(model, policy, stationary_karma_pd)

    transitions0 = model.prob_high * transitions_high + model.prob_low * transitions_low

    if model.mix > 0:
        r = 1 - model.mix
        transitions = r * transitions0 + (1 - r) * get_transitions_mix(model)
    else:
        transitions = transitions0
    assert_good_transitions(transitions)

    return transitions


def get_transitions_mix(exp):
    N = exp.distinct_karma_values
    transitions = np.zeros(shape=(N, N), dtype='float64')
    for i in exp.valid_karma_values:
        if i == 0:
            transitions[i, i] = 0.5
            transitions[i, i + 1] = 0.5
        elif i == max(exp.valid_karma_values):
            transitions[i, i] = 0.5
            transitions[i, i - 1] = 0.5
        else:
            transitions[i, i + 1] = 0.2
            transitions[i, i] = 0.6
            transitions[i, i - 1] = 0.2
    return transitions


def compute_transitions_low(model: Model, policy, stationary_karma_pd):
    assert_pd(stationary_karma_pd)
    # print(stationary_karma_pd)
    N = model.distinct_karma_values
    transitions = np.zeros(shape=(N, N), dtype='float64')

    probability_of_winning_ = model.probability_of_winning_
    next_karma_if_win = model.next_karma_if_win
    next_karma_if_lose = model.next_karma_if_lose
    prob_high = model.prob_high
    prob_low = model.prob_low

    for k_i in model.valid_karma_values:
        assert_pd(policy[k_i, :])
        # when it's low, I always bid 0
        m_i = 0

        # print('p_m_i', p_m_i)
        for k_j in model.valid_karma_values:
            p_k_j = stationary_karma_pd[k_j]
            if p_k_j == 0:
                continue

            # first account for type low
            m_j = 0
            pwin = probability_of_winning_[k_i, m_i, k_j, m_j]
            plose = 1.0 - pwin

            k_if_win = next_karma_if_win[k_i, m_i, k_j, m_j]
            k_if_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]

            B = prob_low * p_k_j
            transitions[k_i, k_if_win] += B * pwin
            transitions[k_i, k_if_lose] += B * plose

            # now account for type high
            assert_pd(policy[k_j, :])
            # all possible karmas
            for m_j in range(0, k_j + 1):
                p_m_j_given_k_j = policy[k_j, m_j]
                if p_m_j_given_k_j == 0:
                    continue

                pwin = probability_of_winning_[k_i, m_i, k_j, m_j]
                plose = 1.0 - pwin

                k_if_win = next_karma_if_win[k_i, m_i, k_j, m_j]
                k_if_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]

                C = prob_high * p_k_j * p_m_j_given_k_j
                transitions[k_i, k_if_win] += C * pwin
                transitions[k_i, k_if_lose] += C * plose

        assert_allclose(np.sum(transitions[k_i, :]), 1.0)

    assert_good_transitions(transitions)
    return transitions


def compute_transitions_high(model: Model, policy, stationary_karma_pd):
    assert_pd(stationary_karma_pd)
    # print(stationary_karma_pd)
    N = model.distinct_karma_values
    transitions = np.zeros(shape=(N, N), dtype='float64')

    probability_of_winning_ = model.probability_of_winning_
    next_karma_if_win = model.next_karma_if_win
    next_karma_if_lose = model.next_karma_if_lose
    prob_high = model.prob_high
    prob_low = model.prob_low

    for k_i in model.valid_karma_values:
        assert_pd(policy[k_i, :])

        # print(f'policy k_i {k_i} = {policy[k_i, :]}')
        for m_i in range(0, k_i + 1):
            p_m_i = policy[k_i, m_i]
            if p_m_i == 0:
                continue
            # print('p_m_i', p_m_i)
            for k_j in model.valid_karma_values:

                p_k_j = stationary_karma_pd[k_j]
                if p_k_j == 0:
                    continue

                # first account if the other is type low
                m_j = 0
                pwin = probability_of_winning_[k_i, m_i, k_j, m_j]
                plose = 1.0 - pwin
                k_if_win = next_karma_if_win[k_i, m_i, k_j, m_j]
                k_if_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]

                B = p_m_i * p_k_j * prob_low
                transitions[k_i, k_if_win] += B * pwin
                transitions[k_i, k_if_lose] += B * plose

                # all possible karmas
                for m_j in range(0, k_j + 1):
                    p_m_j_given_k_j = policy[k_j, m_j]
                    k_if_win = next_karma_if_win[k_i, m_i, k_j, m_j]
                    k_if_lose = next_karma_if_lose[k_i, m_i, k_j, m_j]

                    pwin = probability_of_winning_[k_i, m_i, k_j, m_j]
                    plose = 1.0 - pwin

                    C = p_m_i * p_k_j * prob_high * p_m_j_given_k_j
                    transitions[k_i, k_if_win] += C * pwin
                    transitions[k_i, k_if_lose] += C * plose

        assert_allclose(np.sum(transitions[k_i, :]), 1.0)
    assert_good_transitions(transitions)
    return transitions


def solveStationaryUtility(opt: Optimization, model: Model, transitions, expected_cost_today_per_karma):
    u = np.zeros(model.distinct_karma_values, 'float64')
    # print('expected: %s' % expected_cost_today_per_karma)
    for i in range(100):
        u_prev = np.copy(u)
        for k_i in model.valid_karma_values:
            u[k_i] = expected_cost_today_per_karma[k_i] + model.alpha * np.dot(transitions[k_i, :], u_prev)
        diff = np.max(np.abs(u_prev - u))
        # print('diff %10.10f' % diff)
        if diff < 0.00000001:
            break
    if opt.regularize_utility_monotone:
        u = np.array(sorted(list(u)))

        if opt.regularize_marginal_utility_monotone:
            d = np.diff(u)
            d2 = sorted(list(d), reverse=True)
            u2 = np.cumsum([u[0]] + list(d2))
            u = u2
    return u


def get_random_policy(model: Model):
    N = model.distinct_karma_values
    policy = np.zeros((N, N), dtype='float64')
    for i in model.valid_karma_values:
        n_possible = i + 1
        for m_i in range(0, i + 1):
            policy[i, m_i] = 1.0 / n_possible
    return policy


def get_max_policy(model: Model):
    N = model.distinct_karma_values
    policy = np.zeros((N, N), dtype='float64')
    for i in model.valid_karma_values:
        policy[i, i] = 1.0
    return policy


def get_policy_0(model: Model):
    N = model.distinct_karma_values
    policy = np.zeros((N, N), dtype='float64')
    policy[:, 0] = 1.0
    return policy


def initialize(model: Model, energy_factor) -> Iteration:
    # policy = get_random_policy(exp)
    # policy = get_max_policy(model)
    policy = get_policy_0(model)

    # utility = np.ones(exp.distinct_karma_values, dtype='float64')
    # utility starts as identity
    utility = np.array(model.valid_karma_values, dtype='float64')
    stationary_karma = np.zeros(model.distinct_karma_values, dtype='float64')
    stationary_karma.fill(1.0 / model.distinct_karma_values)

    # global_utility = float(np.dot(stationary_karma, utility2))
    average_karma = float(np.dot(stationary_karma, model.valid_karma_values))
    # stationary_karma[10] = 1.0
    it = Iteration(policy=policy,
                   policy_inst=policy,
                   stationary_karma_pd=stationary_karma,
                   stationary_karma_pd_inst=stationary_karma,
                   utility=utility,
                   energy_factor=energy_factor,
                   average_karma=average_karma)
    return it


# rcParams['backend'] = 'agg'


def policy_diff(p1, p2):
    return np.linalg.norm(p1 - p2)


import cv2


def get_policy_as_cv2_image(policy):
    Z = 16
    blowup = np.kron(np.flipud(policy.T), np.ones((Z, Z)))
    rgb = prepare_for_plot(blowup)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def get_distribution_as_cv2_image(p, H=128):
    p = p / np.max(p)
    N = p.size
    D = 16
    image = np.zeros((H, N * D, 3), dtype='uint8')
    image.fill(255)
    for i in range(N):
        a = i * D
        b = i * D + D
        r = int((1 - p[i]) * H)

        image[r:, a:b, 0] = 240
        image[r:, a:b, 1] = 10
        image[r:, a:b, 2] = 10

    return image


def display_image(window_name, it_next, energy_factor, it_ef):
    bgr_policy_inst = get_policy_as_cv2_image(it_next.policy_inst)
    scale = 0.6
    color = (0, 0, 0)
    tl = (10, 30)
    cv2.putText(bgr_policy_inst, 'inst.', tl,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    bgr_policy = get_policy_as_cv2_image(it_next.policy)
    cv2.putText(bgr_policy, 'smoothed', tl,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    bgr_dist = get_distribution_as_cv2_image(it_next.stationary_karma_pd, H=bgr_policy.shape[0])

    cv2.putText(bgr_dist, 'stationary karma', tl,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    bgr = np.hstack((bgr_policy, bgr_policy_inst, bgr_dist))
    cv2.imshow(window_name, bgr)
    cv2.setWindowTitle(window_name, 'energy_factor = %s it = %d ' % (energy_factor, it_ef))
    cv2.waitKey(1)


def run_experiment(exp_name: str, sim: Simulation, plot_incremental=True,
                   animate=False, plot_incremental_interval=100) -> List[Iteration]:
    it = initialize(sim.model, sim.opt.energy_factor_schedule[0])
    its = [it]
    done = False
    sim.model.initialize_caches()

    def plot_last():
        last = its[-1]
        ef = '%.3f' % float(energy_factor)
        name = f'{exp_name} it {len(its)} ef {ef}'
        r = make_figures2(name, sim, history=its)
        # r = Report('%d' % i)
        # f = r.figure('final', cols=2)
        # with f.plot('policy_last') as pylab:
        #     pylab.imshow(np.flipud(it.policy.T))
        #     pylab.xlabel('karma')
        #     pylab.ylabel('message')
        fn = "incremental.html"
        r.to_html(fn)
        print(f'Report written to {fn}')

    # def plot_thread():
    #     while not done:
    #         plot_last()
    window_name = 'policy'

    if animate:
        cv2.startWindowThread()

    matplotlib.use('cairo')
    RepRepDefaults.default_image_format = MIME_PNG
    # RepRepDefaults.save_extra_png = False
    # threading.Thread(target=plot_thread).start()


    try:
        it = 0
        for energy_factor in sim.opt.energy_factor_schedule:

            for it_ef in range(sim.opt.num_iterations):
                it_next = iterate(sim, its[-1], energy_factor=energy_factor, it_ef=it_ef,
                                  consider_self_effect=sim.opt.consider_self_effect)

                it_next.diff = policy_diff(its[-1].policy, it_next.policy)

                if animate and it_ef % 1 == 0:

                    if it < 30:
                        time.sleep(0.3)
                    display_image(window_name, it_next, energy_factor, it_ef)

                    if it == 0:
                        print('switch to the other window')
                        time.sleep(5)

                its.append(it_next)

                print(f'iteration %4d %5d ef %.3f delta policy %.4f' % (it, it_ef, energy_factor, it_next.diff))

                if plot_incremental:
                    if it > 0 and (it % plot_incremental_interval == 0):
                        plot_last()

                go_next_ef = it_next.diff < sim.opt.diff_threshold

                if go_next_ef:
                    break

                it += 1
                # # if energy_factor is None:
                # #     break
                # if energy_factor + sim.opt.energy_factor_delta >= sim.opt.energy_factor_max:
                #     break
                #
                # energy_factor += sim.opt.energy_factor_delta
                # it_since_ef = 0

                #     energy_factor = None

    except KeyboardInterrupt:
        print('OK, now drawing.')
    finally:
        if plot_incremental:
            plot_last()

    return its


import os


def solveStationary(A):
    n = A.shape[0]
    x = np.ones(n, dtype='float64')
    x.fill(1.0 / n)
    # print(A)
    # print('x %s' % x)
    for i in range(200):
        x_new = np.dot(A.T, x)
        x_new = x_new / np.sum(x)
        diff = np.max(np.abs(x - x_new))

        if diff < 0.00001:
            # print(i, diff)
            break
        # print('diff %s' % diff)
        # print('x_new %s' % x_new)
        x = x_new

    # k = [1, 2, 3, 2, 1]
    # x = np.convolve(x, k, mode='same')
    # x = np.convolve(x, k, mode='same')
    # x = np.ones(x.size)
    x = x / np.sum(x)
    return x


def iterative_main():
    statistics = []
    od = './out-iterative'
    fn0 = os.path.join(od, 'summary.html')
    r0 = Report('all-experiments')
    rows = []
    data = []

    experiments = {}

    models = {}
    max_carma_low, max_carma_mid, max_carma_high = 8, 12, 16

    common = dict(assignment=Assignment.FirstPrice)

    alpha_mid = 0.8
    p_low, p_mid, p_high = 0.4, 0.5, 0.6
    urgency_low, urgency_mid, urgency_high = 2.0, 3.0, 4.0

    alphas = (0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.00)
    for alpha in alphas:
        name = 'M-α%03d-p°-u°-k°' % (alpha * 100)
        models[name] = Model(description=f"α = {alpha}",
                             max_karma=max_carma_mid, urgency0=urgency_mid,
                             alpha=alpha, prob_high=p_mid,
                             **common)

    models['M-α°-p°-u°-k⁻'] = Model(description="fewer karma levels", max_karma=max_carma_low, urgency0=urgency_mid,
                                    alpha=alpha_mid, prob_high=p_mid,
                                    **common)
    models['M-α°-p°-u°-k⁺'] = Model(description="more karma levels", max_karma=max_carma_high, urgency0=urgency_mid,
                                    alpha=alpha_mid, prob_high=p_mid,
                                    **common)

    models['M-α°-p°-u°-k°'] = Model(description="baseline", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=alpha_mid,
                                    prob_high=p_mid,
                                    **common)

    models['M-α°-p°-u°-k°-sp'] = Model(description="second price", max_karma=max_carma_mid, urgency0=urgency_mid,
                                       alpha=alpha_mid,
                                       prob_high=p_mid,
                                       assignment=Assignment.SecondPrice)

    models['M-α°-p⁻-u°-k°'] = Model(description="p(high) larger", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=alpha_mid, prob_high=p_low,
                                    **common)

    models['M-α°-p⁺-u°-k°'] = Model(description="p(high) smaller", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=alpha_mid, prob_high=p_high,
                                    **common)
    models['M-α°-p°-u⁻-k°'] = Model(description="u smaller", max_karma=max_carma_mid, urgency0=urgency_low,
                                    alpha=alpha_mid, prob_high=p_mid,
                                    **common)
    models['M-α°-p°-u⁺-k°'] = Model(description="u larger", max_karma=max_carma_mid, urgency0=urgency_high,
                                    alpha=alpha_mid, prob_high=p_mid,
                                    **common)

    opts = {}
    # opts['o1-noreg-noself'] = Optimization(num_iterations=200,
    #                                        inertia=0.05,
    #                                        energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
    #                                        diff_threshold=0.01,
    #                                        consider_self_effect=False,
    #                                        regularize_utility_monotone=True,
    #                                        regularize_marginal_utility_monotone=False
    #                                        )
    opts['o2-reg-noself'] = Optimization(num_iterations=200,
                                         #inertia=0.05,
                                         inertia=1.0,
                                         #energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
                                         energy_factor_schedule=(1, 1, 1, 1, 1, 1, 1, 1, 1),
                                         #diff_threshold=0.01,
                                         diff_threshold=0.001,
                                         consider_self_effect=False,
                                         regularize_utility_monotone=True,
                                         regularize_marginal_utility_monotone=True
                                         )
    opts['tmp'] = Optimization(num_iterations=200,
                                         inertia=0.07,
                                         energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
                                         diff_threshold=0.01,
                                         consider_self_effect=False,
                                         regularize_utility_monotone=True,
                                         regularize_marginal_utility_monotone=True
                                         )

    # opts['o3-reg-self'] = Optimization(num_iterations=300,
    #                                    inertia=0.3,
    #                                    energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
    #                                    # energy_factor_schedule=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
    #                                    diff_threshold=0.01,
    #                                    consider_self_effect=True,
    #                                    regularize_utility_monotone=True,
    #                                    regularize_marginal_utility_monotone=True,
    #                                    )
    # opts['o4-noreg-self'] = Optimization(num_iterations=200,
    #                                      inertia=0.05,
    #                                      energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
    #                                      diff_threshold=0.01,
    #                                      consider_self_effect=False,
    #                                      regularize_utility_monotone=False,
    #                                      regularize_marginal_utility_monotone=False
    #                                      )
    for model_name, model in models.items():
        for name_opt, opt in opts.items():
            name_exp = f'{model_name}-{name_opt}'
            experiments[name_exp] = Simulation(model=model, opt=opt)

    args = sys.argv[1:]
    if not args:
        print('Give as argument "all" or one or more of the following: %s' % list(experiments))
        sys.exit(-1)

    if args == ['all']:
        todo = list(experiments)
    else:
        todo = args

    for exp_name in todo:
        if not exp_name in experiments:
            msg = 'Cannot find %s' % exp_name
            raise Exception(msg)

    for exp_name in todo:
        print('Experiment %s' % exp_name)
        exp = experiments[exp_name]
        print(dataclasses.asdict(exp))
        rows.append(exp_name)

        animate = 'DISPLAY' in os.environ
        history = run_experiment(exp_name, exp, plot_incremental=False,
                                 animate=animate)

        dn = os.path.join(od, exp_name)
        if not os.path.exists(dn):
            os.makedirs(dn)

        r = make_figures2(exp_name, exp, history)
        fn = os.path.join(dn, 'summary.html')
        r.to_html(fn)
        print(f'Report written to {fn}')

        fn = os.path.join(dn, 'summary.yaml')
        policy_mixed = policy_mean(exp.model, history[-1].policy)
        policy_pure = policy_best(exp.model, history[-1].policy)
        summary = {'results': {'policy_mixed': policy_mixed,
                               'policy_pure': policy_pure,
                               'policy_complete': history[-1].policy.tolist(),
                               'policy_complete_inst': history[-1].policy_inst.tolist()}}
        summary['sim'] = dataclasses.asdict(exp)
        s = yaml.dump(summary)
        with open(fn, 'w') as f:
            f.write(s)
        print('data written to %s' % fn)

        r.nid = exp_name
        r0.add_child(r)
        r0.to_html(fn0)

        datae = []
        for s in statistics:
            val = s(exp, history)
            datae.append(val)
        data.append(datae)

    # cols = [x.__doc__ for x in statistics]
    # r0.table('stats', data=data, cols=cols, rows=rows)
    print(f'Complete report written to {fn0}')
    r0.to_html(fn0)


def policy_mean(model: Model, policy):
    bests = []
    for k in model.valid_karma_values:
        m_mean = np.dot(policy[k, :], model.valid_karma_values)
        # best = np.argmax(policy[k, :])
        # bests.append(best)
        bests.append(float(m_mean))
    return bests


def policy_best(model: Model, policy):
    bests = []
    for k in model.valid_karma_values:
        # m_mean = np.dot(policy[k, :], model.valid_karma_values)
        best = np.argmax(policy[k, :])
        bests.append(int(best))
        # bests.append(float(m_mean))
    return bests


def policy_as_string(model: Model, policy):
    def s(p):
        return ", ".join(['%.1f' % _ for _ in p])

    p1 = policy_mean(model, policy)
    p2 = policy_best(model, policy)

    return s(p1) + ' ' + s(p2)


def make_figures2(name: str, sim: Simulation, history: List[Iteration]) -> Report:
    r = Report('figures')

    data = ""
    for k in sim.__annotations__:
        v = getattr(sim, k)
        if hasattr(v, '__desc__'):
            data += f'{k}:: {v.__desc__}'
        else:
            data += f'\n{k}: {v}\n'

    r.text('description', str(data))

    # from matplotlib import rcParams
    # rcParams['backend'] = 'agg'

    # style = dict(alpha=0.5, linewidth=0.3)

    f = r.figure('final', cols=4)

    last = history[-1]
    caption = """ Policy visualized as intensity (more red: agent more likely to choose message)
%s """ % policy_as_string(sim.model, last.policy)
    # print('policy: %s' % last.policy)
    with f.plot('policy_last', caption=caption) as pylab:

        pylab.imshow(prepare_for_plot(last.policy.T))
        pylab.xlabel('karma')
        pylab.ylabel('message')
        pylab.gca().invert_yaxis()
        pylab.title(f'Policy [{name}]')

    caption = """ Policy visualized as intensity (more red: agent more likely to choose message)
    %s """ % policy_as_string(sim.model, last.policy_inst)
    with f.plot('policy_last_inst', caption=caption) as pylab:
        pylab.imshow(prepare_for_plot(last.policy_inst.T))
        pylab.xlabel('karma')
        pylab.ylabel('message')
        pylab.gca().invert_yaxis()
        pylab.title(f'Policy inst [{name}]')

    caption = """ Utilities visualized in false colors. Yellow = better."""
    if last.debug_utilities is not None:
        with f.plot('utilities', caption=caption) as pylab:
            # print(last.debug_utilities)
            pylab.imshow(last.debug_utilities.T)
            pylab.gca().invert_yaxis()
            pylab.xlabel('karma')
            pylab.ylabel('message')
            pylab.title(f'Utilities [{name}]')

    if last.debug_utilities is not None:
        caption = 'Utility of sending a message for each type of agent'
        with f.plot('utilities_single', caption=caption) as pylab:
            for i in sim.model.valid_karma_values:
                label = 'k=%d' % i if i in [0, sim.model.max_karma] else None
                pylab.plot(last.debug_utilities[i, :], '*-', label=label)

            pylab.ylabel('utility')
            pylab.xlabel('message')
            pylab.legend()
            pylab.title(f'Utilities [{name}]')

    caption = 'Policy visualized as plots; probability of sending each message for each type of agent.'
    with f.plot('policy_as_plots', caption=caption) as pylab:

        for i in sim.model.valid_karma_values:
            label = 'k=%d' % i if i in [0, sim.model.max_karma] else None
            pylab.plot(last.policy[i, :], '*-', label=label)
        pylab.xlabel('message')
        pylab.ylabel('p(message)')
        pylab.legend()
        pylab.title(f'Policy [{name}]')

    if last.transitions is not None:
        caption = 'Transition matrix'
        with f.plot('transitions', caption=caption) as pylab:
            plot_transitions(pylab, last.transitions)
            pylab.title(f'Transitions [{name}]')

    caption = """ Karma stationary distribution. Computed as the equilibrium given the transition matrix. """
    with f.plot('karma_dist_last', caption=caption) as pylab:
        pylab.bar(sim.model.valid_karma_values, last.stationary_karma_pd)
        pylab.xlabel('karma')
        pylab.ylabel('probability')
        pylab.title(f'Stationary karma [{name}]')

    # with f.plot('karma_dist', caption=caption) as pylab:
    #     pylab.imshow(prepare_for_plot(make1d(last.stationary_karma_pd)))
    #     pylab.xlabel('karma')
    #     pylab.title('karma stationary distribution')
    #     pylab.title(f'Stationary karma [{name}]')

    caption = """ Expected utility as a function of karma possessed.  """
    with f.plot('utility', caption=caption) as pylab:
        pylab.plot(last.utility, '*-')
        pylab.xlabel('karma possessed')
        pylab.ylabel('expected utility')
        pylab.title(f'Expected utility [{name}]')

    if last.expected_cost_today_per_karma is not None:
        caption = """ Expected cost today as a function of karma possessed.  """
        with f.plot('cost_today', caption=caption) as pylab:
            pylab.plot(last.expected_cost_today_per_karma, '*-')
            pylab.xlabel('karma possessed')
            pylab.ylabel('expected cost today')
            pylab.title(f'Expected cost today [{name}]')

    # with f.plot('utility_bar', caption=caption) as pylab:
    #     pylab.imshow((make1d(last.utility)))
    #     pylab.xlabel('karma')
    #     pylab.title(f'Utility [{name}]')

    caption = """ Marginal value of having one more unit of karma. """
    with f.plot('utility_marginal', caption=caption) as pylab:
        marginal = np.diff(last.utility)
        pylab.plot(marginal, '*-')
        pylab.xlabel('karma possessed')
        pylab.ylabel('marginal utility of one unit of karma')
        pylab.title(f'Marginal utility of karma [{name}]')

    f = r.figure('history', cols=4)
    style = dict(alpha=0.5, linewidth=0.3)

    with f.plot('delta_policy') as pylab:
        iterations = range(len(history))
        x = [_.diff for _ in history]
        pylab.plot(iterations, x, '-', **style)
        pylab.plot(iterations, x, 'r.', markersize=0.1)
        pylab.xlabel('iterations')
        pylab.ylabel('delta policy')

    with f.plot('global_utility', caption="Global utility (discounted)") as pylab:
        iterations = range(len(history))
        x = [_.global_utility for _ in history]
        pylab.plot(iterations, x, '-', **style)
        pylab.plot(iterations, x, 'r.', markersize=0.1)
        pylab.xlabel('iterations')
        pylab.ylabel('global utility')

    with f.plot('social_utility', caption="Social utility (not discounted)") as pylab:
        iterations = range(len(history))
        x = [_.social_utility for _ in history]
        pylab.plot(iterations, x, '-', **style)
        pylab.plot(iterations, x, 'r.', markersize=0.1)
        pylab.xlabel('iterations')
        pylab.ylabel('social utility')

    with f.plot('social_vs_global') as pylab:
        x = [_.social_utility for _ in history][1:]
        y = [_.global_utility for _ in history][1:]
        pylab.plot(x, y, '-', **style)
        pylab.plot(x, y, 'r.', markersize=0.1)
        pylab.xlabel('social utility')
        pylab.ylabel('global utility')
    # with f.plot('social_vs_global2') as pylab:
    #     x = [_.social_utility for _ in history][1:]
    #     y = [_.global_utility for _ in history][1:]
    #     z = np.linspace(0, 1, len(x))
    #     colorline(x, y, z=z)
    #     pylab.xlabel('social utility')
    #     pylab.ylabel('global utility')


    with f.plot('global_utility_vs_delta') as pylab:
        x = [_.diff for _ in history][1:]
        y = [_.global_utility for _ in history][1:]
        pylab.plot(x, y, '-', **style)
        pylab.plot(x, y, 'r.', markersize=0.1)
        pylab.xlabel('delta policy (stability)')
        pylab.ylabel('global utility')

    with f.plot('average_karma', caption="Average karma. Should be constant. Quantifies numerical errors") as pylab:
        iterations = range(len(history))
        x = [_.average_karma for _ in history]
        pylab.plot(iterations, x, '-', **style)
        pylab.plot(iterations, x, 'r.', markersize=0.1)
        pylab.xlabel('iterations')
        pylab.ylabel('average karma')

    crucial = [0]
    for i in range(len(history) - 1):
        if history[i + 1].energy_factor != history[i].energy_factor:
            crucial.append(i)
    crucial.append(len(history) - 1)

    f = r.figure('snapshots', cols=len(crucial))
    for i in crucial:
        it = history[i]
        name = 'it%04d' % i
        s = policy_as_string(sim.model, it.policy)
        with f.plot(name + '-p', caption='ef = %.2f; %s' % (it.energy_factor, s)) as pylab:
            pylab.imshow(prepare_for_plot(it.policy.T))
            pylab.xlabel('karma')
            pylab.ylabel('message')
            pylab.gca().invert_yaxis()
            pylab.title(f'policy at it = {i} ef = {it.energy_factor}')

    for i in crucial:
        it = history[i]
        name = 'it%04d' % i
        s = policy_as_string(sim.model, it.policy)
        with f.plot(name + '-pi', caption='ef = %.2f; %s' % (it.energy_factor, s)) as pylab:
            pylab.imshow(prepare_for_plot(it.policy_inst.T))
            pylab.xlabel('karma')
            pylab.ylabel('message')
            pylab.gca().invert_yaxis()
            pylab.title(f'policy at it = {i} ef = {it.energy_factor}')

    for i in crucial:
        it = history[i]
        name = 'it%04d' % i

        with f.plot(name + '-u', caption='ef = %.2f' % it.energy_factor) as pylab:
            if it.debug_utilities is not None:
                for i in sim.model.valid_karma_values:
                    pylab.plot(it.debug_utilities[i, :], '*-', )
            else:
                pylab.plot(0, 0)

        pylab.ylabel('utility')
        pylab.xlabel('message')

        pylab.title(f'utilities at it = {i} ef = {it.energy_factor}')

    for i in crucial:
        it = history[i]
        name = 'it%04d' % i

        with f.plot(name + '-k', caption='ef = %.2f' % it.energy_factor) as pylab:
            pylab.bar(sim.model.valid_karma_values, it.stationary_karma_pd)
            pylab.xlabel('karma')
            pylab.ylabel('probability')
            pylab.title(f'stationary karma at it = {i} ef = {it.energy_factor}')

    # history = history[:10]
    # f = r.figure('history', cols=2)
    #
    # caption = """ Policy """
    # with f.plot('policy_history', caption=caption) as pylab:
    #     for it in history:
    #         pylab.plot(it.policy, '-')
    #     pylab.xlabel('karma')
    #     pylab.ylabel('message')
    # caption = """ Utility """
    # with f.plot('utility_history', caption=caption) as pylab:
    #     for it in history:
    #         pylab.plot(it.utility, '-')
    #     pylab.xlabel('karma')
    #     pylab.ylabel('utility')
    # caption = """ Karma stationary dist """
    # with f.plot('karma_dist', caption=caption) as pylab:
    #     for it in history:
    #         pylab.plot(it.stationary_karma_pd, '*-')
    #     pylab.xlabel('karma')
    #     pylab.ylabel('probability')

    #     for i in range(nagents):
    #         cost_i = history[sub, i]['cost_average']
    #         karma_i = history[sub, i]['karma']
    #         pylab.plot(cost_i, karma_i, **style)
    #
    #     pylab.title('cost_average/karma')
    #
    #     pylab.xlabel('cost_average')
    #     pylab.ylabel('karma')

    return r


def plot_transitions(pylab, transitions):
    pylab.imshow(prepare_for_plot(transitions.T))
    pylab.gca().invert_yaxis()
    pylab.xlabel('karma ')
    pylab.ylabel('karma next')


def prepare_for_plot(M):
    M = M.copy()
    M[M == 0] = np.nan
    return posneg(M)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

if __name__ == '__main__':
    iterative_main()
