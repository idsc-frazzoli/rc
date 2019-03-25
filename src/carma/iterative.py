import itertools
import sys
from dataclasses import dataclass
from enum import Enum
from typing import *

import numpy as np
from matplotlib import rcParams
from numpy.testing import assert_allclose

from reprep import Report, posneg


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
    stationary_karma_pd: np.ndarray
    stationary_karma_pd_raw: np.ndarray
    utility: np.ndarray
    energy_factor: float

    debug_utilities: Optional[Any] = None
    transitions: Optional[Any] = None

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

    assert_allclose(pwin + plose, 1.0)

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
                     consider_self_effect) -> Tuple[
    np.ndarray, np.ndarray]:
    """
        What would happen if we bid m_i when we are k_i and high?

    returns expected_utility_of_m_i, expected_cost_today_of_m_i
    """
    expected_utility_of_m_i = 0.0
    expected_cost_today_of_m_i = 0.0
    # I can bid up to m_i
    assert 0 <= m_i <= k_i


    # for each karma of the other
    for k_j in model.valid_karma_values:
        # probability that they have this karma
        p_k_j = stationary_karma_pd[k_j]

        if p_k_j == 0:
            continue

        # first, account for type "low"
        # they bid 0
        m_j = 0
        pwin_if_low = model.probability_of_winning_[k_i, m_i, k_j, m_j]
        plose_if_low = 1.0 - pwin_if_low

        next_karma_if_low_and_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
        next_karma_if_low_and_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]

        utility_if_low_and_lose = model.alpha * utility[next_karma_if_low_and_lose]
        utility_if_low_and_win = model.alpha * utility[next_karma_if_low_and_win]

        P = p_k_j * model.prob_low
        expected_cost_today_of_m_i += P * plose_if_low * (-model.urgency0)

        expected_utility_of_m_i += P * (
                plose_if_low * (-model.urgency0 + utility_if_low_and_lose) +
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

            pwin = model.probability_of_winning_[k_i, m_i, k_j, m_j]
            plose = 1.0 - pwin
            next_karma_if_high_and_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
            next_karma_if_high_and_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]
            utility_if_high_and_win = model.alpha * utility[next_karma_if_high_and_win]
            utility_if_high_and_lose = model.alpha * utility[next_karma_if_high_and_lose]

            P = p_k_j * model.prob_high * p_m_j_given_k_j
            expected_utility_of_m_i += P * \
                                       (plose * (- model.urgency0 + utility_if_high_and_lose) +
                                        pwin * (0 + utility_if_high_and_win))
            expected_cost_today_of_m_i += P * (plose * (-model.urgency0) + pwin * 0)

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
    policy_smooth = remove_underflow(policy_smooth)

    q = float(energy_factor)
    assert 0 <= q <= 1, q

    policy = policy_smooth * (1 - q) + q * policy_sharp

    #
    # if energy_factor is None:
    #
    # else:
    #     expected_utilities_norm = normalize_affine(expected_utilities)
    #     f= energy_factor * expected_utilities_norm
    #     f = np.maximum(f, -500)
    #     try:
    #         policy = np.exp(f)
    #     except FloatingPointError:
    #         print(expected_utilities)
    #         print(expected_utilities_norm)
    #         print(f)
    #         raise
    #     policy[np.isnan(expected_utilities)] = 0.0

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


def remove_underflow(dist, min_pd=0.001):
    """ Removes the values of a p.d. that would create underflow later. """
    dist = np.copy(dist)
    dist[dist < min_pd] = 0
    dist = dist / np.sum(dist)
    return dist


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
            expected_cost_today[m_i] = ec
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

    # with timeit('compute_transitions'):
    transitions = compute_transitions(sim.model, policy2, it.stationary_karma_pd)

    # with timeit('solveStationary'):
    stationary_karma_pd2 = solveStationary(transitions)
    # print(stationary_karma_pd2)

    # for i in exp.valid_karma_values:
    #     stationary_karma_pd2[i] = (exp.max_karma/2.0) - i
    # stationary_karma_pd2.fill(1.0)
    # stationary_karma_pd2 = stationary_karma_pd2 / np.sum(stationary_karma_pd2)

    utility2 = solveStationaryUtility(sim.model, transitions, expected_cost_today_per_karma)
    # make a delta adjustment

    q = sim.opt.inertia

    # if this is the first iteration of a new ef, do not use inertia
    if it_ef == 0:
        q = 1

    policy2 = q * policy2 + (1 - q) * it.policy
    # utility2 = q * utility2 + (1 - q) * it.utility
    # stationary_karma_pd2_final = q * stationary_karma_pd2 + (1 - q) * it.stationary_karma_pd

    # r = 0
    # policy2 = get_random_policy(exp) * r + (1 - r) * policy2
    return Iteration(policy2, stationary_karma_pd=stationary_karma_pd2,
                     stationary_karma_pd_raw=stationary_karma_pd2,
                     debug_utilities=debug_utilities, utility=utility2,
                     transitions=transitions, energy_factor=energy_factor)


def compute_transitions(model: Model, policy, stationary_karma_pd):
    assert_pd(stationary_karma_pd)

    # print('high')
    transitions_high = compute_transitions_high(model, policy, stationary_karma_pd)
    # print('low')
    transitions_low = compute_transitions_low(model, policy, stationary_karma_pd)

    r = 1 - model.mix
    transitions = r * (model.prob_high * transitions_high + model.prob_low * transitions_low) + (
            1 - r) * get_transitions_mix(model)
    assert_good_transitions(transitions)

    for i in model.valid_karma_values:
        transitions[i, :] = remove_underflow(transitions[i, :])

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
            pwin = model.probability_of_winning_[k_i, m_i, k_j, m_j]
            plose = 1.0 - pwin

            k_if_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
            k_if_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]

            B = model.prob_low * p_k_j
            transitions[k_i, k_if_win] += B * pwin
            transitions[k_i, k_if_lose] += B * plose

            # now account for type high
            assert_pd(policy[k_j, :])
            # all possible karmas
            for m_j in range(0, k_j + 1):
                p_m_j_given_k_j = policy[k_j, m_j]
                if p_m_j_given_k_j == 0:
                    continue

                pwin = model.probability_of_winning_[k_i, m_i, k_j, m_j]
                plose = 1.0 - pwin

                k_if_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
                k_if_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]

                C = model.prob_high * p_k_j * p_m_j_given_k_j
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
                pwin = model.probability_of_winning_[k_i, m_i, k_j, m_j]
                plose = 1.0 - pwin
                k_if_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
                k_if_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]

                B = p_m_i * p_k_j * model.prob_low
                transitions[k_i, k_if_win] += B * pwin
                transitions[k_i, k_if_lose] += B * plose

                # now account for type high
                assert_pd(policy[k_j, :])
                # all possible karmas
                for m_j in range(0, k_j + 1):
                    p_m_j_given_k_j = policy[k_j, m_j]
                    k_if_win = model.next_karma_if_win[k_i, m_i, k_j, m_j]
                    k_if_lose = model.next_karma_if_lose[k_i, m_i, k_j, m_j]

                    pwin = model.probability_of_winning_[k_i, m_i, k_j, m_j]
                    plose = 1.0 - pwin

                    C = p_m_i * p_k_j * model.prob_high * p_m_j_given_k_j
                    transitions[k_i, k_if_win] += C * pwin
                    transitions[k_i, k_if_lose] += C * plose

        assert_allclose(np.sum(transitions[k_i, :]), 1.0)
    assert_good_transitions(transitions)
    return transitions


def solveStationaryUtility(model: Model, transitions, expected_cost_today_per_karma):
    u = np.zeros(model.distinct_karma_values, 'float64')
    # print('expected: %s' % expected_cost_today_per_karma)
    for i in range(100):
        u_prev = np.copy(u)
        for k_i in model.valid_karma_values:
            u[k_i] = expected_cost_today_per_karma[k_i] + model.alpha * np.dot(transitions[k_i, :], u_prev)

    if True:
        u = np.array(sorted(list(u)))

        if True:
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
    # stationary_karma[10] = 1.0
    it = Iteration(policy, stationary_karma_pd=stationary_karma,
                   stationary_karma_pd_raw=stationary_karma,
                   utility=utility, energy_factor=energy_factor)
    return it


rcParams['backend'] = 'agg'


def policy_diff(p1, p2):
    return np.linalg.norm(p1 - p2)


def run_experiment(exp_name, sim: Simulation, plot_incremental=False, plot_incremental_interval=100) -> List[Iteration]:
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

    # threading.Thread(target=plot_thread).start()
    try:
        it = 0
        for energy_factor in sim.opt.energy_factor_schedule:

            for it_ef in range(sim.opt.num_iterations):
                it_next = iterate(sim, its[-1], energy_factor=energy_factor, it_ef=it_ef,
                                  consider_self_effect=sim.opt.consider_self_effect)

                diff = policy_diff(its[-1].policy, it_next.policy)

                its.append(it_next)

                print(f'iteration %4d %5d ef %.3f delta policy %.4f' % (it, it_ef, energy_factor, diff))

                if plot_incremental:
                    if it > 0 and (it % plot_incremental_interval == 0):
                        plot_last()

                go_next_ef = diff < sim.opt.diff_threshold

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
        if diff < 0.0001:
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
    alpha_low4, alpha_low3, alpha_low2, alpha_low, alpha_mid, alpha_high, alpha_high2, alpha_high3, alpha_high4 = 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.98
    p_low, p_mid, p_high = 0.4, 0.5, 0.6
    urgency_low, urgency_mid, urgency_high = 2.0, 3.0, 4.0

    models['M-α⁻⁻⁻⁻-p°-u°-k°'] = Model(description="α small 4", max_karma=max_carma_mid, urgency0=urgency_mid,
                                       alpha=alpha_low4, prob_high=p_mid,
                                       **common)
    models['M-α0-p°-u°-k°'] = Model(description="α zero", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=0, prob_high=p_mid,
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

    models['M-α⁻-p°-u°-k°'] = Model(description="α smaller", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=alpha_low, prob_high=p_mid,
                                    **common)
    models['M-α⁻⁻-p°-u°-k°'] = Model(description="α small small", max_karma=max_carma_mid, urgency0=urgency_mid,
                                     alpha=alpha_low2, prob_high=p_mid,
                                     **common)

    models['M-α⁻⁻⁻-p°-u°-k°'] = Model(description="α small small small", max_karma=max_carma_mid, urgency0=urgency_mid,
                                      alpha=alpha_low3, prob_high=p_mid,
                                      **common)

    models['M-α⁺-p°-u°-k°'] = Model(description="α larger", max_karma=max_carma_mid, urgency0=urgency_mid,
                                    alpha=alpha_high, prob_high=p_mid,
                                    **common)
    models['M-α⁺⁺-p°-u°-k°'] = Model(description="α large 2", max_karma=max_carma_mid, urgency0=urgency_mid,
                                     alpha=alpha_high2, prob_high=p_mid,
                                     **common)
    models['M-α⁺⁺⁺-p°-u°-k°'] = Model(description="α large 3", max_karma=max_carma_mid, urgency0=urgency_mid,
                                      alpha=alpha_high3, prob_high=p_mid,
                                      **common)
    models['M-α⁺⁺⁺⁺-p°-u°-k°'] = Model(description="α large 4", max_karma=max_carma_mid, urgency0=urgency_mid,
                                       alpha=alpha_high4, prob_high=p_mid,
                                       **common)

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
    opts['o2'] = Optimization(num_iterations=200,
                              # inertia=0.25, # if 1 then it is faster
                              inertia=0.05,  # if 1 then it is faster
                              energy_factor_schedule=(0.30, 0.45, 0.60, 0.65, 0.7, 0.8, 0.9, 0.95, 1),
                              # energy_factor_schedule=( 0.9, 0.95, 1),
                              # energy_factor=Decimal(0),
                              # energy_factor_delta=Decimal(0.15),
                              # energy_factor_max=0.9,
                              diff_threshold=0.01,
                              consider_self_effect=False)
    for name, model in models.items():
        for name_opt, opt in opts.items():
            name_exp = f'{name}-{name_opt}'
            experiments[name_exp] = Simulation(model=model, opt=opt)

    todo = list(experiments)
    args = sys.argv[1:]
    if args:
        todo = args

    for exp_name in todo:
        print('Experiment %s' % exp_name)
        exp = experiments[exp_name]
        rows.append(exp_name)

        history = run_experiment(exp_name, exp, plot_incremental=True)

        dn = os.path.join(od, exp_name)
        if not os.path.exists(dn):
            os.makedirs(dn)

        r = make_figures2(exp_name, exp, history)
        fn = os.path.join(dn, 'summary.html')
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

    # cols = [x.__doc__ for x in statistics]
    # r0.table('stats', data=data, cols=cols, rows=rows)
    print(f'Complete report written to {fn0}')
    r0.to_html(fn0)


def policy_as_string(policy):
    bests = []
    for k in range(policy.shape[0]):
        best = np.argmax(policy[k, :])
        bests.append(best)

    return "policy=%s" % bests


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

    rcParams['backend'] = 'agg'

    # style = dict(alpha=0.5, linewidth=0.3)

    f = r.figure('final', cols=3)

    last = history[-1]
    caption = """ Policy visualized as intensity (more red: agent more likely to choose message)
%s """ % policy_as_string(last.policy)
    # print('policy: %s' % last.policy)
    with f.plot('policy_last', caption=caption) as pylab:

        pylab.imshow(prepare_for_plot(last.policy.T))
        pylab.xlabel('karma')
        pylab.ylabel('message')
        pylab.gca().invert_yaxis()
        pylab.title(f'Policy [{name}]')

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
        # pylab.plot(last.stationary_karma_pd, '*-', label='moving average')
        # pylab.plot(last.stationary_karma_pd_raw, '*-', label='raw')
        pylab.xlabel('karma')
        pylab.ylabel('probability')
        pylab.legend()
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

    crucial = [0]
    for i in range(len(history) - 1):
        if history[i + 1].energy_factor != history[i].energy_factor:
            crucial.append(i)
    crucial.append(len(history) - 1)

    f = r.figure('snapshots', cols=len(crucial))
    for i in crucial:
        it = history[i]
        name = 'it%04d' % i
        s = policy_as_string(it.policy)
        with f.plot(name + '-p', caption='ef = %.2f; %s' % (it.energy_factor, s)) as pylab:
            pylab.imshow(prepare_for_plot(it.policy.T))
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
                pylab.plot(0,0)

        pylab.ylabel('utility')
        pylab.xlabel('message')

        pylab.title(f'utilities at it = {i} ef = {it.energy_factor}')

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


if __name__ == '__main__':
    iterative_main()
