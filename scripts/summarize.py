"""Mean ± sd (over seeds) of NDCG@10 and HR@10 from a run_variant.sh results file."""
import statistics
import sys
from collections import OrderedDict

rows = OrderedDict()
with open(sys.argv[1]) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 4:
            continue
        dataset, seed, ndcg, hr = parts
        rows.setdefault(dataset, []).append((seed, float(ndcg), float(hr)))


def fmt(values):
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f} ± {sd:.3f}"


print(f"{'dataset':<34}{'seeds':<20}{'NDCG@10':<18}HR@10")
for dataset, runs in rows.items():
    seeds = ",".join(s for s, _, _ in runs)
    print(f"{dataset:<34}{seeds:<20}{fmt([n for _, n, _ in runs]):<18}{fmt([h for _, _, h in runs])}")
