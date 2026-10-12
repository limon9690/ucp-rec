"""Generate a small synthetic dataset in PrepRec's raw CSV format.

Used only to check that the environment and pipeline run end to end
(preprocessing -> training -> zero-shot evaluation) without the real data.
Numbers produced on it mean nothing.

Output: data/<name>/<name>.csv with columns item,user,rate,time (unix seconds).
"""

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="toy")
parser.add_argument("--users", type=int, default=400)
parser.add_argument("--items", type=int, default=500)
parser.add_argument("--per_user", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rng = np.random.RandomState(args.seed)
start = 1514764800  # 2018-01-01
span = 2 * 365 * 24 * 3600  # two years, enough coarse periods for the popularity tables


base = 1.0 / np.arange(1, args.items + 1) ** 0.8
rows = []
for u in range(args.users):
    n = rng.randint(args.per_user // 2, args.per_user * 2)
    times = np.sort(rng.randint(start, start + span, size=n))
    for t in times:
        phase = (t - start) / span
        weights = np.roll(base, int(phase * args.items / 4))
        item = rng.choice(args.items, p=weights / weights.sum())
        rows.append((item, u, 5, int(t)))

out_dir = os.path.join(os.path.dirname(__file__), "..", "data", args.name)
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f"{args.name}.csv")
with open(path, "w") as f:
    f.write("item,user,rate,time\n")
    for r in rows:
        f.write("%d,%d,%d,%d\n" % r)
print(f"wrote {len(rows)} interactions to {os.path.normpath(path)}")
