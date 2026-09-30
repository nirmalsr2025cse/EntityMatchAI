"""Leaderboard metric: macro F0.5 per S1 entity, singletons included."""
import numpy as np


def f05_entity(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map: dict, true_map: dict) -> float:
    # true_map has ALL S1 ids (singletons -> empty set); pred_map may miss keys
    return float(np.mean([f05_entity(set(pred_map.get(s, ())), set(t))
                          for s, t in true_map.items()]))