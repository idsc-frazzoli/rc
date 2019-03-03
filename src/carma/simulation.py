from typing import *

import numpy as np

from .distribution import choose_pair
from .experiment import Experiment
from .types import RNG


# Real = Decimal


def run_experiment(exp: Experiment, seed: Optional[int] = None):
    rng = RNG(seed=seed)
    n = exp.num_agents
    # select urgency distributions for each agent
    urgency_distribution = tuple(exp.urgency_distribution_scenario.choose_distribution_for_agent(i=i, n=n, rng=rng)
                                 for i in range(n))

    # select policy for each agent
    initial_karma = tuple(exp.initial_karma_scenario.choose_initial_karma_for_agent(i=i, n=n, rng=rng)
                          for i in range(n))

    # select initial karma
    agent_policy = tuple(exp.agent_policy_scenario.choose_policy_for_agent(i=i, n=n, rng=rng)
                         for i in range(n))

    current_karma = list(initial_karma)
    accumulated_cost = [0 for _ in range(n)]
    encounters = [0 for _ in range(n)]
    encounters_first = [0 for _ in range(n)]
    encounters_notfirst = [0 for _ in range(n)]

    encounters_per_day = int(exp.average_encounters_per_day_per_agent * exp.num_agents / 2)
    total_encounters = encounters_per_day * exp.num_days
    urgency = None

    history = np.zeros((total_encounters, n), dtype=[('karma', float),
                                                     ('cost', float),
                                                     ('urgency', float),
                                                     ('cost_average', float),
                                                     ('encounters', int),
                                                     ('encounters_first', int),
                                                     ('encounters_notfirst', int)])
    run_experiment.h = 0  # current moment in history

    def save():
        h = run_experiment.h
        for i in range(n):
            history[h, i]['karma'] = current_karma[i]
            history[h, i]['cost'] = accumulated_cost[i]
            history[h, i]['cost_average'] = accumulated_cost[i] / (run_experiment.h + 1)
            history[h, i]['urgency'] = urgency[i]
            history[h, i]['encounters'] = encounters[i]
            history[h, i]['encounters_first'] = encounters_first[i]
            history[h, i]['encounters_notfirst'] = encounters_notfirst[i]
        run_experiment.h += 1

    # iterate over days
    for day in range(exp.num_days):
        # choose urgency for each agent for today
        urgency = [urgency_distribution[i].sample(rng=rng) for i in range(n)]

        # iterate over encounters during the day
        for encounter in range(encounters_per_day):
            # choose pair of agents who meet
            i1, i2 = choose_pair(n)

            agents = (i1, i2)

            # ask each to generate a message
            messages = tuple(agent_policy[i].generate_message(current_carma=current_karma[i],
                                                              current_urgency=urgency[i],
                                                              cost_accumulated=accumulated_cost[i],
                                                              urgency_distribution=urgency_distribution[i],
                                                              rng=rng)
                             for i in agents)

            karmas = tuple(current_karma[i] for i in agents)
            costs = tuple(accumulated_cost[i] for i in agents)
            urgencies = tuple(urgency[i] for i in agents)

            who_goes = exp.who_goes.who_goes(messages=messages, karmas=karmas, costs=costs, urgencies=urgencies,
                                             rng=rng)
            # now update each of them
            for a in range(len(agents)):

                new_karma_i = exp.karma_update_policy.update(karmas[a], messages=messages, who_goes=who_goes,
                                                             a=a, rng=rng)
                new_cost_i = exp.cost_update_policy.update(costs[a], urgency=urgencies, messages=messages,
                                                           who_goes=who_goes,
                                                           a=a, rng=rng)

                assert new_cost_i >= costs[a]
                i = agents[a]
                current_karma[i] = new_karma_i
                accumulated_cost[i] = new_cost_i
                encounters[i] += 1

                if a == who_goes:
                    encounters_first[i] += 1
                else:
                    encounters_notfirst[i] += 1

            save()

    return history
