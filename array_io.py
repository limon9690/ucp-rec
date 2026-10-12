"""Save/load the popularity tables as .npy instead of text."""

import os

import numpy as np


def _npy(path):
    return path[:-4] + ".npy" if path.endswith(".txt") else path + ".npy"


def save_array(path, arr):
    arr = np.atleast_1d(np.squeeze(np.asarray(arr, dtype=np.float64)))
    np.save(_npy(path), arr)


def load_array(path):
    npy = _npy(path)
    if os.path.exists(npy):
        return np.load(npy)
    return np.loadtxt(path)
