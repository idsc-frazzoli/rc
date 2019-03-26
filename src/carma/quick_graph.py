import matplotlib
import numpy as np

# columns = ['name', 'cumulative', 'mean_cost', 'std_cost']
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
#equilibrium0.98	42.06000	0.42145	0.12323	3.05381
#equilibrium0.95	39.45000	0.39592	0.12001	3.00928
equilibrium0.90	41.65500	0.41742	0.12341	2.93526
equilibrium0.85	38.85000	0.38931	0.10826	3.01260
equilibrium0.80	38.70000	0.38786	0.10921	3.12022
equilibrium0.75	38.98500	0.39116	0.10848	3.33703
#equilibrium0.70	38.89500	0.38959	0.11530	3.22580
equilibrium0.50	39.76500	0.39851	0.11165	3.49654
equilibrium0.30	41.17500	0.41258	0.10839	3.93774
equilibrium0.00	58.99500	0.59024	0.13936	5.91741
centralized-urgency-then-cost	36.94500	0.37277	0.04553	4.81204
centralized-urgency	36.94500	0.37079	0.11604	4.69742
centralized-cost	70.80000	0.71516	0.07671	4.74824
bid1-if-urgent	41.44500	0.41503	0.12463	3.73976
bid1-always	74.53500	0.74545	0.13249	3.58131
bid-urgency	47.13000	0.47172	0.12561	4.15401
baseline-random	75.09000	0.75146	0.14678	4.40633
"""
dtype = [('name', '<S32'), ('cumulative', float), ('mean_cost', float), ('std_cost', float)]
lines = [_ for _ in results.strip().split('\n') if not _.startswith('#')]
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

print(data)

r = Report()
with r.plot('plot') as pylab:
    x, y = data[0]['mean_cost'], data[0]['std_cost']
    pylab.plot(x, y, '*')

    fig, ax = pylab.subplots()
    for i in range(n):
        name = data[i]['name'].decode('utf-8')
        x,y = data[i]['mean_cost'], data[i]['std_cost']
        ise = 'eq' in name

        marker = 'o' if ise else '*'
        pylab.plot(x, y, marker, label=name)
        if not ise:
            ax.annotate(name, (x + 0.01, y), ha='left', va='bottom', rotation=0,size=6)

    # ax.annotate('most efficient\nmost fair', (0.35, 0.18))

    pylab.xlabel('inefficiency (mean of cost)')
    pylab.ylabel('unfairness (std-dev of cost)')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)

    # pylab.axis((0.3, 0.5, 0.32, 0.37))

with r.plot('tmp') as pylab:
    x = [0.0, 0.0, 0.3, 0.3, 0.4, 0.5, 0.7, 0.9, 0.9, 0.9, 1.0, 1.0, 1.3]
    pylab.plot(x, '.')
    
fn = 'quick.html'
r.to_html(fn)
print(f'written to {fn}')
