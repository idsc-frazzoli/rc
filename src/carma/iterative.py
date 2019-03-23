from dataclasses import dataclass
from typing import *

import numpy as np
from matplotlib import rcParams
from numpy.testing import assert_allclose

from reprep import Report, posneg


@dataclass
class ItExperiment:
    num_iterations: int
    max_carma: int
    prob_high: float
    alpha: float
    urgency0: float
    inertia: float
    valid_carma_values: Optional[np.ndarray] = None
    distinct_karma_values: Optional[int] = None
    prob_low: Optional[float] = None

    def __post_init__(self):
        self.valid_carma_values = list(range(0, self.max_carma + 1))
        assert max(self.valid_carma_values) == self.max_carma
        self.distinct_karma_values = self.max_carma + 1
        self.prob_low = 1.0 - self.prob_high


def assert_pd(x):
    assert np.all(x >= 0), x
    assert np.allclose(np.sum(x), 1), x


def assert_good_transitions(M):
    n1, n2 = M.shape
    assert n1 == n2
    for i in range(n1):
        cond = M[i, :]
        assert_allclose(np.sum(cond), 1.0)
    #
    # def expected_karma(p):
    #     return np.dot(p, list(range(n1)))
    #
    # p0 = np.zeros(n1, 'float64')
    # p0[3] = 1.0
    #
    # p1 = np.dot(M.T, p0)
    # print(f'p0: {p0}')
    # print(f'p1: {p1}')
    # print(f'ex0: {expected_karma(p0)}')
    # print(f'ex1: {expected_karma(p1)}')


def assert_good_policy(p):
    n1, n2 = p.shape
    assert n1 == n2
    for i in range(n1):
        pol = p[i, :]
        assert_allclose(np.sum(pol), 1.0)
        for j in range(n2):
            if j > i:
                assert pol[j] == 0


def assert_valid_karma_value(exp: ItExperiment, p):
    assert 0 <= p <= exp.max_carma, p


@dataclass
class Iteration:
    policy: np.ndarray
    stationary_carma_pd: np.ndarray

    utility: np.ndarray
    stationary_carma_cdf: Optional[Any] = None

    debug_utilities: Optional[Any] = None
    transitions: Optional[Any] = None

    def __post_init__(self):
        N = self.stationary_carma_pd.size
        assert self.policy.shape == (N, N)
        assert_good_policy(self.policy)
        assert_pd(self.stationary_carma_pd)

        self.stationary_carma_cdf = np.cumsum(self.stationary_carma_pd)
        # print(self.stationary_carma_cdf)
        if self.transitions is not None:
            n = self.transitions.shape[0]
            # print(np.sum(self.transitions, axis=1))
            for i in range(n):
                tp = np.sum(self.transitions[i, :])
                assert np.allclose(tp, 1.0), tp



def consider_bidding(exp, stationary_carma_pd, utility, policy, k_i, m_i) -> Tuple[np.ndarray, np.ndarray]:
    """
        What would happen if we bid m_i when we are k_i and high?

    returns expected_utility_of_m_i, expected_cost_today_of_m_i
    """
    expected_utility_of_m_i = 0.0
    expected_cost_today_of_m_i = 0.0
    # I can bid up to m_i
    assert 0 <= m_i <= k_i

    # for each karma of the other
    for k_j in exp.valid_carma_values:
        # probability that they have this karma
        p_k_j = stationary_carma_pd[k_j]
        if p_k_j == 0:
            continue

        # first, account for type "low"
        # they bid 0
        if m_i == 0:
            pwin_if_low, plose_if_low = 0.5, 0.5
        else:
            pwin_if_low, plose_if_low = 1.0, 0.0

        next_karma_if_low_and_lose = k_i
        # conservation of karma: I can only lose up to what the other can win
        karma_lost = min(m_i, exp.max_carma - k_j)
        next_karma_if_low_and_win = k_i - karma_lost
        utility_if_low_and_lose = exp.alpha*utility[next_karma_if_low_and_lose]
        utility_if_low_and_win =  exp.alpha*utility[next_karma_if_low_and_win]

        P =  p_k_j * exp.prob_low
        expected_cost_today_of_m_i += P * plose_if_low * (-exp.urgency0)

        expected_utility_of_m_i += P * (
                plose_if_low * (-exp.urgency0 + utility_if_low_and_lose) +
                pwin_if_low * (0+ utility_if_low_and_win)
        )


        # now account for type "high"

        # all possible karmas
        for m_j in range(0, k_j + 1):
            # with this probability
            p_m_j_given_k_j = policy[k_j, m_j]
            if p_m_j_given_k_j == 0:
                continue

            if m_i > m_j:
                # we win
                pwin, plose = 1.0, 0.0
            elif m_i < m_j:
                # we lose
                pwin, plose = 0.0, 1.0
            elif m_i == m_j:
                # half half
                pwin, plose = 0.5, 0.5
            else:
                assert False

            # I can gain up to max_carma
            next_karma_if_high_and_lose = min(k_i + m_j, exp.max_carma)
            # I can lose only up to what he can gain
            karma_lost = min(m_i, exp.max_carma - k_j)
            next_karma_if_high_and_win = k_i - karma_lost
            utility_if_high_and_lose =  exp.alpha*utility[next_karma_if_high_and_lose]
            utility_if_high_and_win =  exp.alpha*utility[next_karma_if_high_and_win]

            expected_utility_of_m_i += p_k_j * exp.prob_high * p_m_j_given_k_j * \
                                       ((utility_if_high_and_lose - exp.urgency0) * plose +
                                        utility_if_high_and_win * pwin)
            expected_cost_today_of_m_i += p_k_j * exp.prob_high * p_m_j_given_k_j * plose * (-exp.urgency0)

    return expected_utility_of_m_i, expected_cost_today_of_m_i


def iterate(exp: ItExperiment, it: Iteration) -> Iteration:
    # need to find for each karma
    N = exp.distinct_karma_values
    policy2 = np.zeros((N, N), dtype='float64')
    debug_utilities = []

    expected_cost_per_karma = np.zeros(exp.distinct_karma_values, 'float64')
    expected_cost_today_per_karma = np.zeros(exp.distinct_karma_values, 'float64')
    for k_i in exp.valid_carma_values:
        expected_utilities = np.zeros(k_i + 1, dtype='float64')
        expected_cost_today = np.zeros(k_i + 1, dtype='float64')
        for m_i in range(0, k_i + 1):
            eu, ec = consider_bidding(exp, stationary_carma_pd=it.stationary_carma_pd,
                                      utility=it.utility, policy=it.policy, k_i=k_i, m_i=m_i)
            expected_cost_today[m_i] = ec
            expected_utilities[m_i] = eu

        best_policy = np.argmax(expected_utilities)

        if k_i in [exp.max_carma, exp.max_carma-1]:
            print(f'for {k_i} we have  best = {best_policy}\n  u: {expected_utilities}\n ct: {expected_cost_today};' )

        expected_cost_per_karma[k_i] = expected_utilities[best_policy]
        expected_cost_today_per_karma[k_i] = expected_cost_today[best_policy]

        policy2[k_i, best_policy] = 1.0
        debug_utilities.append(expected_utilities)

    # FIXME: fixing bug
    # policy2[exp.max_carma, :] = policy2[exp.max_carma-1, :]

    transitions = compute_transitions(exp, policy2, it.stationary_carma_pd)
    stationary_carma_pd2 = solveStationary(transitions)
    # print(stationary_carma_pd2)
    utility2 = solveStationaryUtility(exp, transitions, expected_cost_today_per_karma)
    # make a delta adjustment

    q = exp.inertia
    policy2 = q * policy2 + (1 - q) * it.policy

    # r = 0
    # policy2 = get_random_policy(exp) * r + (1 - r) * policy2
    return Iteration(policy2, stationary_carma_pd2, debug_utilities=debug_utilities, utility=utility2,
                     transitions=transitions)


def compute_transitions(exp: ItExperiment, policy, stationary_carma_pd):
    assert_pd(stationary_carma_pd)

    # print('high')
    transitions_high = compute_transitions_high(exp, policy, stationary_carma_pd)
    # print('low')
    transitions_low = compute_transitions_low(exp, policy, stationary_carma_pd)

    transitions = exp.prob_high * transitions_high + exp.prob_low * transitions_low
    assert_good_transitions(transitions)
    return transitions


def compute_transitions_low(exp: ItExperiment, policy, stationary_carma_pd):
    assert_pd(stationary_carma_pd)
    # print(stationary_carma_pd)
    N = exp.distinct_karma_values
    transitions = np.zeros(shape=(N, N), dtype='float64')

    for k_i in exp.valid_carma_values:
        assert_pd(policy[k_i, :])
        # when it's low, I always bid 0
        # m_i = 0

        # print('p_m_i', p_m_i)
        for k_j in exp.valid_carma_values:
            p_k_j = stationary_carma_pd[k_j]
            if p_k_j == 0:
                continue

            # first account for type low
            # we don't change the karma as they bid 0
            pwin, plose = 0.5, 0.5
            B = exp.prob_low * p_k_j
            k_if_win = k_if_lose = k_i
            transitions[k_i, k_if_win] += B * pwin
            transitions[k_i, k_if_lose] += B * plose

            # now account for type high
            assert_pd(policy[k_j, :])
            # all possible karmas
            for m_j in range(0, k_j + 1):
                p_m_j_given_k_j = policy[k_j, m_j]
                if p_m_j_given_k_j == 0:
                    continue

                if 0 == m_j:
                    pwin, plose = 0.5, 0.5
                elif 0 < m_j:
                    pwin, plose = 0.0, 1.0
                else:
                    assert False

                # I didn't bid anything
                k_if_win = k_i - 0
                # I gain what they bid
                k_if_lose = min(k_i + m_j, exp.max_carma)

                C = exp.prob_high * p_k_j * p_m_j_given_k_j
                transitions[k_i, k_if_win] += C * pwin
                transitions[k_i, k_if_lose] += C * plose

        assert_allclose(np.sum(transitions[k_i, :]), 1.0)

    assert_good_transitions(transitions)
    return transitions


def compute_transitions_high(exp: ItExperiment, policy, stationary_carma_pd):
    assert_pd(stationary_carma_pd)
    # print(stationary_carma_pd)
    N = exp.distinct_karma_values
    transitions = np.zeros(shape=(N, N), dtype='float64')

    for k_i in exp.valid_carma_values:
        assert_pd(policy[k_i, :])

        # print(f'policy k_i {k_i} = {policy[k_i, :]}')
        for m_i in range(0, k_i + 1):
            p_m_i = policy[k_i, m_i]
            if p_m_i == 0:
                continue
            # print('p_m_i', p_m_i)
            for k_j in exp.valid_carma_values:

                p_k_j = stationary_carma_pd[k_j]
                if p_k_j == 0:
                    continue

                # first account if the other is type low
                if m_i == 0:
                    pwin, plose = 0.5, 0.5
                else:
                    pwin, plose = 1.0, 0.0

                karma_lost = min(m_i, exp.max_carma - k_j)
                k_if_win = k_i - karma_lost
                k_if_lose = k_i + 0

                B = p_m_i * p_k_j * exp.prob_low
                transitions[k_i, k_if_win] += B * pwin
                transitions[k_i, k_if_lose] += B * plose

                # now account for type high
                assert_pd(policy[k_j, :])
                # all possible karmas
                for m_j in range(0, k_j + 1):
                    p_m_j_given_k_j = policy[k_j, m_j]

                    # karma_lost = min(m_i, exp.max_carma - k_j)
                    # k_if_win = k_i - karma_lost
                    k_if_lose = min(k_i + m_j, exp.max_carma)

                    if m_i == m_j:
                        pwin, plose = 0.5, 0.5

                    elif m_i > m_j:
                        pwin, plose = 1.0, 0.0

                    elif m_i < m_j:
                        pwin, plose = 0.0, 1.0
                    else:
                        assert False

                    C = p_m_i * p_k_j * exp.prob_high * p_m_j_given_k_j
                    transitions[k_i, k_if_win] += C * pwin
                    transitions[k_i, k_if_lose] += C * plose

        assert_allclose(np.sum(transitions[k_i, :]), 1.0)
    assert_good_transitions(transitions)
    return transitions


def solveStationaryUtility(exp: ItExperiment, transitions, expected_cost_today_per_karma):
    u = np.zeros(exp.distinct_karma_values, 'float64')
    # print('expected: %s' % expected_cost_today_per_karma)
    for i in range(100):
        u_prev = np.copy(u)
        for k_i in exp.valid_carma_values:
            u[k_i] = expected_cost_today_per_karma[k_i] + exp.alpha * np.dot(transitions[k_i, :], u_prev)

    if False:
        u = np.array(sorted(list(u)))
    return u


def get_random_policy(exp):
    N = exp.distinct_karma_values
    policy = np.zeros((N, N), dtype='float64')
    for i in exp.valid_carma_values:
        n_possible = i + 1
        for m_i in range(0, i + 1):
            policy[i, m_i] = 1.0 / n_possible
    return policy


def initialize(exp: ItExperiment) -> Iteration:
    policy = get_random_policy(exp)

    # utility = np.ones(exp.distinct_karma_values, dtype='float64')
    # utility starts as identity
    utility = np.array(exp.valid_carma_values)
    stationary_carma = np.zeros(exp.distinct_karma_values, dtype='float64')
    stationary_carma.fill(1.0 / exp.distinct_karma_values)
    # stationary_carma[10] = 1.0
    it = Iteration(policy, stationary_carma, utility)
    return it


rcParams['backend'] = 'agg'


def run_experiment(exp) -> List[Iteration]:
    it = initialize(exp)

    its = [it]
    try:
        for i in range(exp.num_iterations):
            if i % 5 == 0:
                r = make_figures2('it%s' % i, exp, history=[its[-1]])
                # r = Report('%d' % i)
                # f = r.figure('final', cols=2)
                # with f.plot('policy_last') as pylab:
                #     pylab.imshow(np.flipud(it.policy.T))
                #     pylab.xlabel('carma')
                #     pylab.ylabel('message')
                fn = "incremental.html"
                r.to_html(fn)
                print(f'Report written to {fn}')

            it_next = iterate(exp, its[-1])
            its.append(it_next)
            # from reprep.graphics.filter_posneg import posneg_hinton
            # rgb = posneg_hinton(it.policy)
            # new_im = Image.fromarray(rgb)
            # new_im.save("policy.png")



    except KeyboardInterrupt:
        print('OK, now drawing.')
        pass

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
    return x
    # """ x = xA where x is the answer
    # x - xA = 0
    # x( I - A ) = 0 and sum(x) = 1
    # """
    # n = A.shape[0]
    # a = np.eye( n ) - A
    # a = np.vstack( (a.T, np.ones( n )) )
    # b = np.matrix( [0] * n + [ 1 ] ).T
    # return np.linalg.lstsq( a, b )[0]


def iterative_main():
    statistics = []
    od = './out-iterative'
    fn0 = os.path.join(od, 'index.html')
    r0 = Report('all-experiments')
    rows = []
    data = []

    experiments = {}
    # experiments['one'] = ItExperiment(num_iterations=1000000, max_carma=50, alpha=0.8, urgency0=3.0, prob_high=0.25)
    # experiments['one'] = ItExperiment(num_iterations=1000000, max_carma=16, alpha=0.8, urgency0=3.0, prob_high=0.5)
    # experiments['one'] = ItExperiment(num_iterations=1000000, max_carma=32, alpha=0.4, urgency0=3.0, prob_high=0.5)
    experiments['one'] = ItExperiment(num_iterations=1000000, max_carma=16, alpha=0.9, urgency0=3.0, prob_high=0.1, inertia=0.1)

    for exp_name, exp in experiments.items():
        rows.append(exp_name)

        history = run_experiment(exp)

        dn = os.path.join(od, exp_name)
        if not os.path.exists(dn):
            os.makedirs(dn)

        r = make_figures2(exp_name, exp, history)
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

    # cols = [x.__doc__ for x in statistics]
    # r0.table('stats', data=data, cols=cols, rows=rows)
    print(f'Complete report written to {fn0}')
    r0.to_html(fn0)


def make_figures2(name: str, exp: ItExperiment, history: List[Iteration]) -> Report:
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

    # style = dict(alpha=0.5, linewidth=0.3)

    def prepare_for_plot(M):
        M = M.copy()
        M[M == 0] = np.nan
        return posneg(M)

    f = r.figure('final', cols=2)
    #
    last = history[-1]
    caption = """ Policy """
    with f.plot('policy_last', caption=caption) as pylab:
        # pylab.plot(exp.valid_carma_values, exp.valid_carma_values, '--')
        pylab.imshow(prepare_for_plot(last.policy.T))
        pylab.xlabel('carma')
        pylab.ylabel('message')
        pylab.gca().invert_yaxis()

    if last.transitions is not None:
        with f.plot('transitions', caption="transitions") as pylab:
            pylab.imshow(prepare_for_plot(last.transitions.T))
            pylab.gca().invert_yaxis()

    caption = """ Utility """
    with f.plot('utility_last', caption=caption) as pylab:

        pylab.plot(last.utility, '-')
        pylab.xlabel('carma')
        pylab.ylabel('utility')

    caption = """ Karma stationary dist """
    with f.plot('karma_dist_last', caption=caption) as pylab:
        pylab.plot(last.stationary_carma_pd, '-')
        pylab.xlabel('carma')
        pylab.ylabel('probability')

    history = history[:10]
    f = r.figure('history', cols=2)
    #
    # caption = """ Policy """
    # with f.plot('policy_history', caption=caption) as pylab:
    #     for it in history:
    #         pylab.plot(it.policy, '-')
    #     pylab.xlabel('carma')
    #     pylab.ylabel('message')
    caption = """ Utility """
    with f.plot('utility_history', caption=caption) as pylab:
        for it in history:
            pylab.plot(it.utility, '-')
        pylab.xlabel('carma')
        pylab.ylabel('utility')
    caption = """ Karma stationary dist """
    with f.plot('karma_dist', caption=caption) as pylab:
        for it in history:
            pylab.plot(it.stationary_carma_pd, '*-')
        pylab.xlabel('carma')
        pylab.ylabel('probability')

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


if __name__ == '__main__':
    iterative_main()
