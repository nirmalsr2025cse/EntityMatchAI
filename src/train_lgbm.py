
"""LightGBM version of train.py. Same labels, same GroupKFold, same threshold search,
so its macro F0.5 is directly comparable with the Random Forest.

Usage (repo root):
  python -m src.train_lgbm --features data/features.parquet \
      --truth dataset/sample_5pct/train_ground_truth.tsv --out models/lgbm.joblib
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from src.decide import decide
from src.evaluate import macro_f05
from src.features import BASIC, FEATURES


def load_truth(path):
    truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth file is missing columns: {sorted(missing)}")
    return {
        row.source1_entity_id: {
            item.strip()
            for item in row.matched_entity_ids.split(",")
            if item.strip()
        }
        for row in truth.itertuples(index=False)
    }


def label_pairs(feats, true_map):
    y = np.fromiter(
        (cand in true_map.get(s1, set())
         for s1, cand in zip(feats["s1"], feats["cand"])),
        dtype=np.int8,
        count=len(feats),
    )
    gt_pairs = {
        (s1, cand)
        for s1, candidates in true_map.items()
        for cand in candidates
    }
    return y, gt_pairs


def blocking_recall(feats, gt_pairs):
    if not gt_pairs:
        return 1.0
    found = set(zip(feats["s1"], feats["cand"]))
    return len(found & gt_pairs) / len(gt_pairs)


def decide_no_assign(pairs, t):
    keep = pairs[pairs.p >= t]
    if keep.empty:
        return {}
    return keep.groupby("s1").cand.apply(set).to_dict()


def best_threshold(pairs, true_map, decision_fn=decide_no_assign, grid=None):
    thresholds = np.arange(0.05, 1.0, 0.01) if grid is None else grid
    best_t, best_score = None, -1.0
    for threshold in thresholds:
        score = macro_f05(decision_fn(pairs, float(threshold)), true_map)
        if score > best_score:
            best_t, best_score = float(threshold), score
    return best_t, best_score


def report_at(pairs, true_map, threshold):
    pred_map = decide(pairs, threshold)
    true_pairs = {
        (s1, cand)
        for s1, candidates in true_map.items()
        for cand in candidates
    }
    pred_pairs = {
        (s1, cand)
        for s1, candidates in pred_map.items()
        for cand in candidates
    }
    true_positive = len(true_pairs & pred_pairs)
    precision = true_positive / len(pred_pairs) if pred_pairs else 0.0
    recall = true_positive / len(true_pairs) if true_pairs else 1.0
    return {
        "macro_f05": macro_f05(pred_map, true_map),
        "precision": precision,
        "recall": recall,
    }


def make_lgbm(n_estimators=400, seed=42):
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, class_weight="balanced",
                          random_state=seed, n_jobs=-1, verbose=-1)


def oof_predict(feats, y, cols, n_splits=5, n_estimators=400, seed=42):
    X, groups = feats[cols].to_numpy(), feats["s1"].to_numpy()
    oof = np.zeros(len(feats))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_lgbm(n_estimators, seed).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def run(feats, true_map, n_splits=5, n_estimators=400, grid=None):
    present_s1 = set(feats["s1"])
    eval_truth = {s1: matches for s1, matches in true_map.items() if s1 in present_s1}
    y, gt_pairs = label_pairs(feats, eval_truth)
    res = {"n_pairs": len(feats), "n_pos": int(y.sum()), "n_gt_pairs": len(gt_pairs),
           "blocking_recall": blocking_recall(feats, gt_pairs),
           "avg_cands_per_s1": len(feats) / max(feats["s1"].nunique(), 1)}

    base = feats[["s1", "cand"]].assign(p=feats["block_sim_combo"].to_numpy())
    res["a_baseline_t"], res["a_baseline"] = best_threshold(base, eval_truth, grid=grid)

    pb = feats[["s1", "cand"]].assign(p=oof_predict(feats, y, BASIC, n_splits, n_estimators))
    res["b_lgbm_basic_t"], res["b_lgbm_basic"] = best_threshold(
        pb, eval_truth, decide_no_assign, grid
    )

    pf = feats[["s1", "cand"]].assign(p=oof_predict(feats, y, FEATURES, n_splits, n_estimators))
    res["c_lgbm_full_t"], res["c_lgbm_full"] = best_threshold(
        pf, eval_truth, decide_no_assign, grid
    )
    t, score = best_threshold(pf, eval_truth, decide, grid)
    res["d_final_t"], res["d_final"] = t, score
    res.update({f"final_{k}": v for k, v in report_at(pf, eval_truth, t).items()})

    model = make_lgbm(n_estimators).fit(feats[FEATURES].to_numpy(), y)
    res["importances"] = dict(sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1]))
    return res, model, pf, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="models/lgbm.joblib")
    ap.add_argument("--oof-out", default="data/oof_lgbm.parquet")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    feats = pd.read_parquet(a.features)
    true_map = load_truth(a.truth)
    print(f"pairs {len(feats):,} | S1 in truth {len(true_map):,} | "
          f"singletons {sum(1 for v in true_map.values() if not v):,}")
    res, model, pf, t = run(feats, true_map, a.n_splits, a.n_estimators)

    imp = res.pop("importances")
    print("\n=== LIGHTGBM RESULTS ===")
    for k, v in res.items():
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")
    print("\ntop 10 feature importances (split counts):")
    for k, v in list(imp.items())[:10]:
        print(f"  {k:26s} {v}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.oof_out) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": t, "features": FEATURES}, a.out)
    pf.to_parquet(a.oof_out, index=False)
    print(f"\nsaved model -> {a.out} (threshold {t:.2f}) and OOF probabilities -> {a.oof_out}")


if __name__ == "__main__":
    main()