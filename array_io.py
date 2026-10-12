"""Save/load the popularity tables as .npy instead of text.

Upstream wrote these with np.savetxt (11-12 GB per file on the Amazon sets,
slow to parse at the start of every run). .npy holds the same float64 values
exactly in about a third of the space and loads in seconds.

Paths are still given with their upstream ".txt" names; the .npy sits next to
it. load_array falls back to the .txt file if no .npy exists, so old
preprocessed data keeps working.
"""
import os

import numpy as np


def _npy(path):
    return path[:-4] + ".npy" if path.endswith(".txt") else path + ".npy"


def save_array(path, arr):
    # match what np.loadtxt(np.savetxt(...)) would give back: float64 and
    # size-1 dimensions squeezed (a one-row table loads as a 1-D array)
    arr = np.atleast_1d(np.squeeze(np.asarray(arr, dtype=np.float64)))
    np.save(_npy(path), arr)


def load_array(path):
    npy = _npy(path)
    if os.path.exists(npy):
        return np.load(npy)
    return np.loadtxt(path)
