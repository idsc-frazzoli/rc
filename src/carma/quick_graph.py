import numpy as np
# columns = ['name', 'cumulative', 'mean_cost', 'std_cost']
from reprep import Report

results = """
equilibrium0.00	10.87500	0.54270	0.25865	5.71802
equilibrium0.30	8.19000	0.41012	0.21964	3.74643
equilibrium0.50	8.04000	0.40081	0.22670	3.39349
equilibrium0.70	7.80000	0.39053	0.22132	3.07665
equilibrium0.80	7.75500	0.39016	0.22416	2.73236
equilibrium0.85	7.78500	0.38947	0.22017	2.88891
equilibrium0.90	8.35500	0.42115	0.26749	2.54475
equilibrium0.98	8.59500	0.43314	0.26237	2.91647
#equilibrium-0.75	7.99500	0.39915	0.21175	3.58968
#equilibrium-0.80	7.84500	0.39205	0.21635	3.05218
#equilibrium-0.85	7.71000	0.38767	0.22139	2.83651
centralized-urgency	7.42500	0.36973	0.22744	4.66538
baseline-random	15.01500	0.75770	0.27696	4.54816
bid1-always	14.91000	0.74640	0.25904	3.62019
bid1-if-urgent	8.25000	0.41398	0.24552	3.62295
bid-urgency	9.06000	0.45596	0.24621	4.04917
centralized-cost	15.57000	0.80270	0.18655	4.27969
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

    ax.annotate('most efficient\nmost fair', (0.35, 0.18))
    pylab.axis((0.3,1, 0.17, 0.3 ))
    pylab.xlabel('inefficiency (mean of cost)')
    pylab.ylabel('unfairness (std-dev of cost)')


fn = 'quick.html'
r.to_html(fn)
print(f'written to {fn}')
