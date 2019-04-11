import numpy as np
from reprep import Report
from carma.iterative import plot_transitions
from .policy_agent import Globals
from .experiment import Experiment
from .simulation import run_experiment, compute_karma_distribution2
from .statistics import compute_karma_distribution, compute_transitions_matrix_and_policy_for_urgency_nonzero
import matplotlib
import seaborn

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
