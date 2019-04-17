import matplotlib
import numpy as np

# columns = ['name', 'cumulative', 'mean_cost', 'std_cost']
from carma import equilibria
from reprep import Report, RepRepDefaults, MIME_SVG

results_old = """
equilibrium0.00	56.70000	0.56691	0.11746	5.89286
#equilibrium0.30	40.00500	0.40005	0.09061	3.91992
#equilibrium0.50	39.09000	0.39044	0.09372	3.27655
equilibrium0.70	38.10000	0.38094	0.09101	2.94207
equilibrium0.80	38.19000	0.38154	0.09051	2.97250
equilibrium0.85	38.19000	0.38189	0.08782	2.92160
#equilibrium0.90	40.17000	0.40194	0.10406	2.62408
#equilibrium0.98	40.92000	0.40882	0.10589	2.95225
centralized-urgency	37.06500	0.36982	0.09607	4.73770
baseline-random	74.74500	0.74737	0.12319	4.68997
bid1-always	74.64000	0.74589	0.11070	3.40966
bid1-if-urgent	40.45500	0.40421	0.10228	3.56452
bid-urgency	44.67000	0.44657	0.10486	4.06027
centralized-cost	74.82000	0.75503	0.07508	4.41993
"""
results = """
equilibrium0.00	258.01500	0.51576	0.05561	5.49052
equilibrium0.80	192.70500	0.38530	0.04100	3.29329
equilibrium1.00	265.69500	0.53130	0.05680	1.98388
equilibrium0.20	236.71500	0.47322	0.05213	5.25792
equilibrium0.05	275.11500	0.54992	0.05461	5.80222
equilibrium0.30	229.83000	0.45950	0.05003	5.09468
equilibrium0.90	192.54000	0.38501	0.04404	2.63169
equilibrium0.70	198.63000	0.39715	0.04212	3.68317
equilibrium0.85	191.41500	0.38280	0.04121	2.79925
equilibrium0.10	239.80500	0.47943	0.05034	5.36803
equilibrium0.35	220.23000	0.44030	0.04757	4.96948
equilibrium0.65	202.48500	0.40479	0.04425	4.25626
equilibrium0.95	336.34500	0.67261	0.06092	0.57946
equilibrium0.25	231.48000	0.46277	0.04909	5.13476
equilibrium0.50	207.48000	0.41476	0.04502	4.77973
equilibrium0.60	204.99000	0.40982	0.04260	4.51838
equilibrium0.40	215.37000	0.43063	0.04797	4.85343
equilibrium0.45	210.51000	0.42089	0.04643	4.77240
equilibrium0.75	196.63500	0.39310	0.04196	3.59802
equilibrium0.15	236.23500	0.47224	0.05069	5.28827
equilibrium0.55	205.51500	0.41086	0.04256	4.57228
centralized-urgency	186.58500	0.37321	0.03943	4.57228
centralized-urgency-then-cost	186.58500	0.37374	0.01532	4.83278
baseline-random	374.73000	0.74942	0.05679	4.72819
bid1-always	374.02500	0.74795	0.04901	3.58131
bid1-if-urgent	202.93500	0.40579	0.04772	3.70618
bid-urgency	229.26000	0.45838	0.04988	4.02191
centralized-cost	372.76500	0.74673	0.02982	4.55475
"""
results ="""
guess1	233.52000	0.46694	0.05840	2.32718
equilibrium0.00	258.01500	0.51576	0.05561	5.49052
equilibrium0.80	192.70500	0.38530	0.04100	3.29329
equilibrium1.00	265.69500	0.53130	0.05680	1.98388
equilibrium0.20	236.71500	0.47322	0.05213	5.25792
equilibrium0.05	275.11500	0.54992	0.05461	5.80222
equilibrium0.30	229.83000	0.45950	0.05003	5.09468
equilibrium0.90	192.54000	0.38501	0.04404	2.63169
equilibrium0.70	198.63000	0.39715	0.04212	3.68317
equilibrium0.85	191.41500	0.38280	0.04121	2.79925
equilibrium0.10	239.80500	0.47943	0.05034	5.36803
equilibrium0.35	220.23000	0.44030	0.04757	4.96948
equilibrium0.65	202.48500	0.40479	0.04425	4.25626
equilibrium0.95	336.34500	0.67261	0.06092	0.57946
equilibrium0.25	231.48000	0.46277	0.04909	5.13476
equilibrium0.50	207.48000	0.41476	0.04502	4.77973
equilibrium0.60	204.99000	0.40982	0.04260	4.51838
equilibrium0.40	215.37000	0.43063	0.04797	4.85343
equilibrium0.45	210.51000	0.42089	0.04643	4.77240
equilibrium0.75	196.63500	0.39310	0.04196	3.59802
equilibrium0.15	236.23500	0.47224	0.05069	5.28827
equilibrium0.55	205.51500	0.41086	0.04256	4.57228
pure0.00	289.29000	0.57832	0.05321	5.93597
pure0.30	200.67000	0.40126	0.04310	3.84132
pure0.50	195.79500	0.39143	0.04174	3.29329
pure0.70	191.62500	0.38317	0.04144	2.89064
pure0.80	191.44500	0.38280	0.03906	2.80994
pure0.85	191.64000	0.38321	0.04207	2.74149
pure0.90	201.55500	0.40293	0.04607	2.50914
pure0.98	206.37000	0.41263	0.05006	2.96239
centralized-urgency	186.58500	0.37321	0.03943	4.57228
centralized-urgency-then-cost	186.58500	0.37374	0.01532	4.83278
baseline-random	374.73000	0.74942	0.05679	4.72819
bid1-always	374.02500	0.74795	0.04901	3.58131
bid1-if-urgent	202.93500	0.40579	0.04772	3.70618
bid-urgency	229.26000	0.45838	0.04988	4.02191
centralized-cost	372.76500	0.74673	0.02982	4.55475
"""

results="""
baseline-random	26.22300	0.74988	0.21747	4.72777
bid-urgency	15.29550	0.43669	0.17924	4.15160
bid1-always	26.24850	0.75051	0.20704	3.62419
bid1-if-urgent	13.97850	0.39927	0.17482	3.64510
centralized-cost	26.03100	0.76142	0.13462	4.44790
centralized-urgency	13.12500	0.37487	0.16499	4.70508
centralized-urgency-then-cost	13.12500	0.38175	0.08954	4.74729
equilibrium0.00	17.61000	0.50231	0.18793	5.54903
equilibrium0.05	18.01500	0.51447	0.19090	5.73731
equilibrium0.10	16.37250	0.46735	0.17951	5.34535
equilibrium0.15	16.07550	0.45904	0.17798	5.34086
equilibrium0.20	16.14000	0.46092	0.17970	5.24193
equilibrium0.25	15.81000	0.45106	0.17426	5.15148
equilibrium0.30	15.72900	0.44821	0.17248	5.12423
equilibrium0.35	15.14550	0.43170	0.16843	5.00977
equilibrium0.40	14.79150	0.42197	0.16474	4.99728
equilibrium0.45	14.55600	0.41556	0.16425	4.88935
equilibrium0.50	14.32050	0.40837	0.16070	4.76432
equilibrium0.55	14.19300	0.40505	0.15836	4.65948
equilibrium0.60	14.11500	0.40207	0.15610	4.56561
equilibrium0.65	13.99350	0.39879	0.15544	4.32166
equilibrium0.70	13.73250	0.39208	0.15247	3.76507
equilibrium0.75	13.62900	0.38872	0.15092	3.69740
equilibrium0.80	13.40100	0.38257	0.15145	3.24157
equilibrium0.85	13.32300	0.38039	0.15564	2.99596
equilibrium0.90	13.33050	0.38036	0.15901	2.71400
equilibrium0.95	20.23050	0.57808	0.21861	1.14751
equilibrium1.00	17.82150	0.50951	0.20686	2.20812
#guess1	19.56300	0.55943	0.21728	1.25848
pure0.00	19.36950	0.55337	0.20043	5.89430
pure0.30	13.82400	0.39442	0.15351	3.96192
pure0.50	13.54050	0.38626	0.15875	3.40232
pure0.70	13.34700	0.38066	0.15823	3.12358
pure0.80	13.35150	0.38102	0.15600	2.99980
pure0.85	13.32450	0.38064	0.15594	2.94088
pure0.90	13.78500	0.39298	0.16864	2.59380
pure0.98	14.09550	0.40246	0.17375	3.01808
"""
dtype = [('name', '<S32'), ('cumulative', float), ('mean_cost', float), ('std_cost', float)]
lines = [_ for _ in results.strip().split('\n') if not _.startswith('#')]
lines = sorted(lines)
n = len(lines)

matplotlib.use('agg')
RepRepDefaults.default_image_format = MIME_SVG
RepRepDefaults.save_extra_png = True
RepRepDefaults.save_extra_pdf = True
data = np.zeros(shape=n, dtype=dtype)
for i, x in enumerate(lines):
    tokens = x.split()
    data[i]['name'] = tokens[0]
    data[i]['cumulative'] = float(tokens[1])
    data[i]['mean_cost'] = float(tokens[2])
    data[i]['std_cost'] = float(tokens[3])

# print(data)

def color_for_alpha(alpha):
    return alpha, np.sin(alpha*np.pi) ,0.5

r = Report()
with r.plot('plot') as pylab:
    x, y = data[0]['mean_cost'], data[0]['std_cost']
    pylab.plot(x, y, '*')

    fig, ax = pylab.subplots()
    for i in range(n):
        name = data[i]['name'].decode('utf-8')
        x, y = data[i]['mean_cost'], data[i]['std_cost']
        ise = 'eq' in name
        ispure = 'pure' in name
        if ispure:
            marker = 's'
            alpha = float(name.replace('pure', ''))
            color = color_for_alpha(alpha)
            label = ' α = %.2f (pure)' % alpha
            pylab.plot(x, y, marker, label=label, color=color,
                       markeredgecolor=(0,0,0), markersize=4)
        elif ise:
            marker = 'o'
            alpha = float(name.replace('equilibrium',''))
            color = color_for_alpha(alpha)
            label = ' α = %.2f (mixed)' % alpha
            pylab.plot(x, y, marker, label=label, color=color)
        else:
            marker = '*'
            pylab.plot(x, y, marker, label=name)

            ax.annotate(name, (x + 0.01, y), ha='left', va='bottom', rotation=0, size=6)

    # ax.annotate('most efficient\nmost fair', (0.35, 0.18))

    pylab.xlabel('inefficiency (mean of cost)')
    pylab.ylabel('unfairness (std-dev of cost)')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)

    # pylab.axis((0.3, 0.5, 0.32, 0.37))
#
# d = {0.00: [0.0, 1.0, 2.0, 2.9, 3.9, 4.9, 5.8, 6.8, 7.8, 8.8, 9.7, 10.7, 11.7],
#      0.80: [0.0, 1.0, 1.0, 1.0, 1.9, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.1],
#      1.00: [0.0, 0.0, 0.3, 0.3, 0.4, 0.5, 0.7, 0.9, 0.9, 0.9, 1.0, 1.0, 1.3],
#      0.20: [0.0, 1.0, 2.0, 2.3, 3.0, 4.0, 4.7, 5.4, 6.1, 7.1, 8.1, 9.1, 11.8],
#      0.05: [0.0, 1.0, 2.0, 3.0, 3.8, 4.3, 5.0, 6.0, 7.0, 8.0, 9.2, 10.1, 12.0],
#      0.30: [0.0, 1.0, 2.0, 2.0, 3.0, 3.9, 4.4, 5.4, 6.1, 6.8, 8.0, 9.2, 11.6],
#      0.90: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.9, 2.0, 2.0, 2.0, 2.0, 3.0],
#      0.70: [0.0, 1.0, 1.0, 2.0, 2.2, 2.9, 3.1, 3.9, 4.0, 4.5, 5.0, 5.3, 6.3],
#      0.85: [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.1, 3.0, 3.1, 3.1, 3.1, 4.1],
#      0.10: [0.0, 1.0, 2.0, 2.9, 3.0, 4.0, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 11.7],
#      0.35: [0.0, 1.0, 1.6, 2.0, 3.0, 3.8, 4.2, 4.7, 5.5, 6.3, 7.3, 8.9, 11.4],
#      0.65: [0.0, 1.0, 1.0, 2.0, 2.7, 3.3, 3.9, 4.2, 4.7, 5.4, 6.3, 7.5, 9.2],
#      0.95: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#      0.25: [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.4, 5.3, 6.1, 6.7, 7.7, 9.1, 11.7],
#      0.50: [0.0, 1.0, 1.0, 2.0, 2.9, 3.6, 4.0, 4.6, 5.4, 6.0, 7.0, 8.4, 10.8],
#      0.60: [0.0, 1.0, 1.0, 2.0, 2.9, 3.5, 3.9, 4.5, 5.0, 5.6, 6.6, 8.0, 10.3],
#      0.40: [0.0, 1.0, 1.0, 2.0, 3.0, 3.9, 4.1, 4.7, 5.3, 6.2, 6.9, 8.5, 11.5],
#      0.45: [0.0, 1.0, 1.0, 2.0, 3.0, 3.8, 4.1, 4.7, 5.4, 6.2, 6.8, 8.4, 11.1],
#      0.75: [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.7, 5.1, 6.3],
#      0.15: [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.9, 5.9, 6.9, 7.9, 8.9, 9.8, 11.7],
#      0.55: [0.0, 1.0, 1.0, 2.0, 2.8, 3.5, 4.0, 4.6, 5.3, 5.7, 6.7, 7.9, 10.6]}

with r.plot('equilibria') as pylab:
    for alpha in sorted(equilibria):
        policy = equilibria[alpha]
        color = color_for_alpha(alpha)
        pylab.plot(policy, '-*', label=' α = %.2f (mixed)' % alpha, color=color)

    pylab.xlabel('karma')
    pylab.ylabel('expected value of message ')
    pylab.title('Visualization of Nash Equilibria (mixed strategies)')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)

#
# d = {0.00: [0.0, 1.0, 2.0, 2.9, 3.9, 4.9, 5.8, 6.8, 7.8, 8.8, 9.7, 10.7, 11.7],
#      0.80: [0.0, 1.0, 1.0, 1.0, 1.9, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.1],
#      1.00: [0.0, 0.0, 0.3, 0.3, 0.4, 0.5, 0.7, 0.9, 0.9, 0.9, 1.0, 1.0, 1.3],
#      0.20: [0.0, 1.0, 2.0, 2.3, 3.0, 4.0, 4.7, 5.4, 6.1, 7.1, 8.1, 9.1, 11.8],
#      0.05: [0.0, 1.0, 2.0, 3.0, 3.8, 4.3, 5.0, 6.0, 7.0, 8.0, 9.2, 10.1, 12.0],
#      0.30: [0.0, 1.0, 2.0, 2.0, 3.0, 3.9, 4.4, 5.4, 6.1, 6.8, 8.0, 9.2, 11.6],
#      0.90: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.9, 2.0, 2.0, 2.0, 2.0, 3.0],
#      0.70: [0.0, 1.0, 1.0, 2.0, 2.2, 2.9, 3.1, 3.9, 4.0, 4.5, 5.0, 5.3, 6.3],
#      0.85: [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.1, 3.0, 3.1, 3.1, 3.1, 4.1],
#      0.10: [0.0, 1.0, 2.0, 2.9, 3.0, 4.0, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 11.7],
#      0.35: [0.0, 1.0, 1.6, 2.0, 3.0, 3.8, 4.2, 4.7, 5.5, 6.3, 7.3, 8.9, 11.4],
#      0.65: [0.0, 1.0, 1.0, 2.0, 2.7, 3.3, 3.9, 4.2, 4.7, 5.4, 6.3, 7.5, 9.2],
#      0.95: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#      0.25: [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.4, 5.3, 6.1, 6.7, 7.7, 9.1, 11.7],
#      0.50: [0.0, 1.0, 1.0, 2.0, 2.9, 3.6, 4.0, 4.6, 5.4, 6.0, 7.0, 8.4, 10.8],
#      0.60: [0.0, 1.0, 1.0, 2.0, 2.9, 3.5, 3.9, 4.5, 5.0, 5.6, 6.6, 8.0, 10.3],
#      0.40: [0.0, 1.0, 1.0, 2.0, 3.0, 3.9, 4.1, 4.7, 5.3, 6.2, 6.9, 8.5, 11.5],
#      0.45: [0.0, 1.0, 1.0, 2.0, 3.0, 3.8, 4.1, 4.7, 5.4, 6.2, 6.8, 8.4, 11.1],
#      0.75: [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.7, 5.1, 6.3],
#      0.15: [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.9, 5.9, 6.9, 7.9, 8.9, 9.8, 11.7],
#      0.55: [0.0, 1.0, 1.0, 2.0, 2.8, 3.5, 4.0, 4.6, 5.3, 5.7, 6.7, 7.9, 10.6]}

with r.plot('equilibria_pure') as pylab:
    from .policy_agent import equilibria_pure
    for alpha in sorted(equilibria_pure):
        policy = equilibria_pure[alpha]
        color = color_for_alpha(alpha)
        pylab.plot(policy, '-*', label=' α = %.2f (pure)' % alpha, color=color)

    pylab.xlabel('karma')
    pylab.ylabel('expected value of message ')
    pylab.title('Visualization of Nash Equilibria (mixed strategies)')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)

fn = 'quick.html'
r.to_html(fn)
print(f'written to {fn}')
