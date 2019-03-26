import numpy as np
# columns = ['name', 'cumulative', 'mean_cost', 'std_cost']
from reprep import Report

results = """
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
dtype = [('name', '<S32'), ('cumulative', float), ('mean_cost', float), ('std_cost', float)]
lines = [_ for _ in results.strip().split('\n') if not _.startswith('#')]
n = len(lines)


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
    pylab.plot(0,0,'.')
    fig, ax = pylab.subplots()
    for i in range(n):
        name = data[i]['name'].decode('utf-8')
        x, y = data[i]['mean_cost'], data[i]['std_cost']
        pylab.plot(x,y, '*')
        ax.annotate(name, (x+0.01, y), ha='left', va='bottom', rotation=25)

    # ax.annotate('most efficient\nmost fair', (0.35, 0.18))
    # pylab.axis((0.3,1, 0.17, 0.3 ))
    pylab.xlabel('inefficiency (mean of cost)')
    pylab.ylabel('unfairness (std-dev of cost)')


fn = 'quick.html'
r.to_html(fn)
print(f'written to {fn}')
