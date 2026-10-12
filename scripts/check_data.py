"""Check preprocessed datasets against thesis Table 3.3 and record checksums.

Usage: python scripts/check_data.py [dataset ...]
       python scripts/check_data.py --verify   # compare against data/MANIFEST.md5

The first run writes data/MANIFEST.md5 (commit it). Later runs with --verify
confirm the evaluation data is byte-identical, which is what makes results
comparable across machines.
"""

import argparse
import hashlib
import os
import pickle

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

EXPECTED = {
    "amazon/amazon_office": (39999, 25898),
    "amazon/amazon_tool": (39999, 70903),
    "douban/douban_movie": (39999, 40148),
    "douban/douban_music": (20539, 10121),
    "epinions/epinions": (30738, 20355),
}
CHECKSUMMED = [
    "_intwtime.csv",
    "_userneg.pickle",
    "_wtembed.txt",
    "_week_embed2.txt",
    "_week_wt_embed_adj.txt",
]
MANIFEST = os.path.join(ROOT, "MANIFEST.md5")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=list(EXPECTED))
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()

sums, ok = {}, True
for ds in args.datasets:
    base = os.path.join(ROOT, ds)
    if not os.path.exists(base + "_intwtime.csv"):
        print(f"MISSING  {ds} (not preprocessed)")
        ok = False
        continue
    df = pd.read_csv(base + "_intwtime.csv", header=None)
    users, items = df[0].nunique(), df[1].nunique()
    negs = pickle.load(open(base + "_userneg.pickle", "rb"))
    neg_sizes = {len(v) for v in negs.values()}
    want = EXPECTED.get(ds)
    match = want is None or (users, items) == want
    ok &= match
    tag = "OK      " if match else "MISMATCH"
    exp = f"  (thesis: {want[0]:,} / {want[1]:,})" if want else ""
    print(
        f"{tag} {ds:<24} users {users:,}  items {items:,}{exp}  negatives/user {sorted(neg_sizes)}"
    )
    for suf in CHECKSUMMED:
        p = base + suf
        if os.path.exists(p):
            sums[ds + suf] = md5(p)

if args.verify:
    recorded = dict(reversed(l.split()) for l in open(MANIFEST) if l.strip())
    for k, v in sums.items():
        if k in recorded and recorded[k] != v:
            print(f"CHANGED  {k}")
            ok = False
    print("checksums match manifest" if ok else "checksums differ from manifest")
else:
    with open(MANIFEST, "w") as f:
        for k in sorted(sums):
            f.write(f"{sums[k]}  {k}\n")
    print(f"wrote {len(sums)} checksums to data/MANIFEST.md5")

raise SystemExit(0 if ok else 1)
