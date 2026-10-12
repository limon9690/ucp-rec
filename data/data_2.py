import numpy as np
import pandas as pd
from datetime import *
from collections import Counter
from scipy.stats import rankdata
import argparse
import pickle
import sys
import os.path as _osp

sys.path.append(_osp.join(_osp.dirname(_osp.abspath(__file__)), ".."))
from array_io import save_array, load_array
from operator import itemgetter
from multiprocessing import Pool

from data import pop_embed


def uniform_negs_per_user(group):
    user, items, itemnum, seed = group
    rng = np.random.RandomState(seed + int(user))
    valid_items = sorted(set(range(1, itemnum + 1)) - set(items))
    selected_items = rng.choice(valid_items, size=100, replace=False)
    return user + 1, list(selected_items)


def popularity_negs_per_user(group):
    user, items, itemnum, lastpop, seed = group
    rng = np.random.RandomState(seed + int(user))
    valid_items = sorted(set(range(1, itemnum + 1)) - set(items))
    filtered_pop = lastpop[~np.isin(np.arange(len(lastpop)), np.array(valid_items) - 1)]
    selected_items = rng.choice(
        valid_items, size=100, replace=False, p=filtered_pop / np.sum(filtered_pop)
    )
    return user + 1, list(selected_items)


def process_user(user_data):
    last_u, userneg, t2_size = user_data
    period = by_period.get(last_u["time6"])
    counter = (
        Counter()
        if period is None
        else Counter(period["item"][period["time"] < last_u["time"]])
    )
    arr = np.array(userneg) - 1
    arr = np.insert(arr, 0, last_u["item"])
    counts = np.array(itemgetter(*arr)(counter))
    urow = otmpw[int(last_u["time6"]) - 1].copy()
    urow[arr] += counts
    percs = 100 * rankdata(urow, "average") / len(urow)
    return np.array([pop_embed(perc, t2_size) for perc in percs[arr]]).T


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./douban/douban_music", type=str)
    parser.add_argument(
        "--mode",
        default="uniform",
        type=str,
        choices=["uniform", "popularity"],
        help="uniform or popularity negative sampling",
    )
    parser.add_argument("--t2_size", default=5, type=int, help="number of fine periods")
    parser.add_argument(
        "--userneg",
        action="store_true",
        help="conduct user negative sampling for evaluation",
    )
    parser.add_argument(
        "--seed", default=2023, type=int, help="seed for evaluation negatives"
    )
    parser.add_argument(
        "--week_adj",
        action="store_true",
        help="get most recent fine popularities (prior to interaction period) for evaluation",
    )
    args = parser.parse_args()
    dataset = args.dataset

    ao = pd.read_csv(f"{dataset}_intwtime.csv", header=None)
    ao.columns = ["user", "item", "time4", "time6", "time"]

    if args.userneg:
        usernum = ao.user.nunique()
        itemnum = ao.item.nunique()
        usersneg = {}
        grouped = ao.groupby("user")
        with Pool(processes=8) as pool:
            if args.mode == "uniform":
                groups = [
                    (u, group.item.values, itemnum, args.seed) for u, group in grouped
                ]
                usersneg = dict(pool.map(uniform_negs_per_user, groups))
            elif args.mode == "popularity":
                lastpop = load_array(f"{dataset}_rawpop.txt")
                groups = [
                    (u, group.item.values, itemnum, lastpop, args.seed)
                    for u, group in grouped
                ]
                usersneg = dict(pool.map(popularity_negs_per_user, groups))
        name = "userneg" if args.mode == "uniform" else "popneg"
        with open(f"{dataset}_{name}.pickle", "wb") as handle:
            pickle.dump(usersneg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.week_adj:
        users = sorted(ao.user.unique())
        with open(f"{dataset}_userneg.pickle", "rb") as handle:
            usernegs = pickle.load(handle)
        last = ao.groupby("user").last()
        otmpw = load_array(f"{dataset}_week_curr_raw.txt")
        by_period = {
            t: {"time": g["time"].values, "item": g["item"].values}
            for t, g in ao.groupby("time6")
        }
        with Pool(processes=8) as pool:
            user_data = [(last.iloc[u], usernegs[u + 1], args.t2_size) for u in users]
            results = pool.map(process_user, user_data)
        df = np.concatenate(results)
        save_array(f"{dataset}_week_wt_embed_adj.txt", df)
