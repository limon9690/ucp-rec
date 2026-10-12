"""Convert popularity tables written as text by the old preprocessing to .npy.

Usage: python scripts/txt_to_npy.py data/amazon [--keep]

Converts every *_{wtembed,week_embed2,week_wt_embed_adj,week_curr_raw,rawpop}.txt
in the folder, smallest first, deleting each .txt once its .npy is written
(unless --keep), so it works on a nearly full disk. Large files are parsed in
chunks straight into the .npy, so memory stays low. Values are parsed with
exact float round-tripping, identical to np.loadtxt.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from array_io import save_array  # noqa: E402

NAMES = ("wtembed", "week_embed2", "week_wt_embed_adj", "week_curr_raw", "rawpop")
CHUNK = int(os.environ.get("CHUNK", 200_000))

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("--keep", action="store_true", help="keep the .txt files")
args = parser.parse_args()

files = [f for n in NAMES for f in glob.glob(os.path.join(args.folder, f"*_{n}.txt"))]
for path in sorted(files, key=os.path.getsize):
    npy = path[:-4] + ".npy"
    size_gb = os.path.getsize(path) / 2**30
    print(f"{path} ({size_gb:.1f} GB) -> .npy", flush=True)
    # rawpop is a single row; loadtxt squeezes it to 1-D, so keep loadtxt for it
    if size_gb < float(os.environ.get("SMALL_GB", 0.2)) or path.endswith("_rawpop.txt"):
        save_array(path, np.loadtxt(path))
    else:
        with open(path) as f:
            ncols = len(f.readline().split())
        with open(path, "rb") as f:
            nrows = sum(1 for _ in f)
        out = np.lib.format.open_memmap(npy + ".part", mode="w+", dtype=np.float64, shape=(nrows, ncols))
        row = 0
        for chunk in pd.read_csv(path, sep=" ", header=None, dtype=np.float64,
                                 float_precision="round_trip", chunksize=CHUNK):
            out[row:row + len(chunk)] = chunk.values
            row += len(chunk)
        assert row == nrows, (row, nrows)
        out.flush()
        del out
        os.replace(npy + ".part", npy)
    if not args.keep:
        os.remove(path)
print("done")
