from decimal import Decimal
from typing import *

import numpy as np
from tqdm import tqdm

from .distribution import choose_pair, DiscreteDistribution
from .experiment import Experiment
from .policy_agent import AgentPolicy, Globals
from .types import RNG, KarmaValue, CostValue, MessageValue


def compute_karma_distribution2(x: List[float]):
    bins = np.linspace(-0.5, Globals.max_carma + 0.5, Globals.max_carma + 2)
    cd, _ = np.histogram(x, bins, density=True)

    return cd


def nice_karma_dist(x: List[float]):
    cd = compute_karma_distribution2(x)
    mean = np.dot(Globals.valid_karma_values, cd)
    f = lambda _: round(Decimal(_), 3)
    cd = list(map(f, cd))
    dist = " ".join(str(_) for _ in cd)
    return "mean: %.1f  dist: %s  # %s sum %s " % (mean, dist, len(x), sum(x))


def run_experiment(exp: Experiment, seed: Optional[int] = None):
    rng_policy = RNG(seed=seed)
    rng_sim = RNG(seed=seed)
    n = exp.num_agents
    # select urgency distributions for each agent
    urgency_distribution = tuple(
        exp.urgency_distribution_scenario.choose_distribution_for_agent(
            i=i, n=n, rng=rng_sim
        )
        for i in range(n)
    )

    # select initial karma
    initial_karma = tuple(
        exp.initial_karma_scenario.choose_initial_karma_for_agent(i=i, n=n, rng=rng_sim)
        for i in range(n)
    )

    # select policy for each agent
    agent_policy: Tuple[AgentPolicy] = tuple(
        exp.agent_policy_scenario.choose_policy_for_agent(i=i, n=n, rng=rng_policy)
        for i in range(n)
    )

    current_karma: List[KarmaValue] = list(initial_karma)
    accumulated_cost: List[CostValue] = [0 for _ in range(n)]
    encounters = [0 for _ in range(n)]
    encounters_first = [0 for _ in range(n)]
    encounters_notfirst = [0 for _ in range(n)]

    encounters_per_day = int(
        exp.average_encounters_per_day_per_agent * exp.num_agents / 2
    )
    total_encounters = encounters_per_day * exp.num_days
    urgency = None

    history = np.zeros(
        (total_encounters, n),
        dtype=[
            ("karma", int),
            ("cost", float),
            ("urgency", float),
            ("cost_average", float),
            ("message", int),
            ("participated", bool),
            ("encounters", int),
            ("encounters_first", int),
            ("encounters_notfirst", int),
        ],
    )
    history["message"].fill(-100)
    history["participated"].fill(False)
    run_experiment.h = 0  # current moment in history
    h_karma = history["karma"]
    h_cost = history["cost"]
    h_cost_average = history["cost_average"]
    h_urgency = history["urgency"]
    h_encounters = history["encounters"]
    h_encounters_first = history["encounters_first"]
    h_encounters_notfirst = history["encounters_notfirst"]

    def save():
        h = run_experiment.h
        for i in range(n):
            h_karma[h, i] = current_karma[i]
            h_cost[h, i] = accumulated_cost[i]
            h_cost_average[h, i] = accumulated_cost[i] / max(encounters[i], 1)
            h_urgency[h, i] = urgency[i]
            h_encounters[h, i] = encounters[i]
            h_encounters_first[h, i] = encounters_first[i]
            h_encounters_notfirst[h, i] = encounters_notfirst[i]
        run_experiment.h += 1

    def reset_costs():
        # reset the costs after warmup
        for i in range(n):
            accumulated_cost[i] = 0
            encounters[i] = 0
            encounters_first[i] = 0
            encounters_notfirst[i] = 0

    warm_up_days = int(0.3 * exp.num_days)

    print("initial karma dist: %s" % nice_karma_dist(current_karma))
    warm_up_finished_at = None
    # iterate over days
    for day in tqdm(range(exp.num_days)):

        if day == warm_up_days:
            msg = (
                "Warm up finished after %d days.  Resetting costs, but keeping karmas."
                % day
            )
            print(msg)
            reset_costs()
            print("warmup karma dist: %s" % nice_karma_dist(current_karma))
            warm_up_finished_at = run_experiment.h
        # choose urgency for each agent for today
        urgency = [urgency_distribution[i].sample(rng=rng_sim) for i in range(n)]

        # iterate over encounters during the day
        for encounter in range(encounters_per_day):
            # choose pair of agents who meet
            i1, i2 = choose_pair(n, rng_sim)

            agents = (i1, i2)

            if day in [0, exp.num_days - 1] and encounter == 0:
                print(f"rng check: encounter {day}/{encounter} chose {agents}")

            # ask each to generate a message
            messages_dist: Tuple[DiscreteDistribution[MessageValue]] = tuple(
                agent_policy[i].generate_message(
                    current_carma=current_karma[i],
                    current_urgency=urgency[i],
                    cost_accumulated=accumulated_cost[i],
                    urgency_distribution=urgency_distribution[i],
                    rng=rng_policy,
                )
                for i in agents
            )
            messages: Tuple[MessageValue] = [
                _.sample(rng_policy) for _ in messages_dist
            ]

            karmas = tuple(current_karma[i] for i in agents)
            costs = tuple(accumulated_cost[i] for i in agents)
            urgencies = tuple(urgency[i] for i in agents)

            who_goes = exp.who_goes.who_goes(
                messages=messages,
                karmas=karmas,
                costs=costs,
                urgencies=urgencies,
                rng=rng_policy,
            )

            new_karmas = exp.karma_update_policy.update(
                karmas, messages=messages, who_goes=who_goes, rng=rng_policy
            )

            # now update each of them
            for a in range(len(agents)):
                new_cost_i = exp.cost_update_policy.update(
                    costs[a],
                    urgency=urgencies,
                    messages=messages,
                    who_goes=who_goes,
                    a=a,
                    rng=rng_policy,
                )

                # print('karma %s bid %s' % (karmas[a], messages[a]))
                assert new_cost_i >= costs[a]
                i = agents[a]

                current_karma[i] = new_karmas[a]
                accumulated_cost[i] = new_cost_i
                encounters[i] += 1

                history[run_experiment.h, i]["message"] = messages[a]
                history[run_experiment.h, i]["participated"] = True

                if a == who_goes:
                    encounters_first[i] += 1
                else:
                    encounters_notfirst[i] += 1

            save()

    res = history[warm_up_finished_at:, :]

    return res
