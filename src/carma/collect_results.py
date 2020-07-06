from decimal import Decimal

import yaml
from conf_tools import locate_files

d = "."
files = locate_files(d, "*.yaml")
print(files)

expect = {
    "max_karma": 12,
    "urgency0": 3.0,
}
optimal = {}
for fn in files:
    with open(fn) as f:
        data = f.read()
        data = yaml.load(data)
        print(data)

    model = data["sim"]["model"]

    for k, v in expect.items():
        if model.get(k) != v:
            continue
    alpha = model["alpha"]
    policy = data["results"]["policy"]
    print(alpha, policy)

    optimal[round(Decimal(alpha), 2)] = [round(Decimal(_), 1) for _ in policy]
print(optimal)
