"""Load the friend's parquet files, check columns, build features, save them.

Usage (from repo root):
  python run_features.py --pairs pairs.parquet --records records.parquet --out features.parquet
Optional: --n 1000 to test on the first 1000 pairs only.
"""
import argparse
import time

import pandas as pd

from src.features import FEATURES, build_features

NEED_PAIRS = {"s1", "cand", "block_rank", "block_sim_name", "block_sim_combo"}
NEED_REC = {"id", "source", "country", "name_full", "name_core", "addr_norm",
            "postal", "house_no", "landmark"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--n", type=int, default=0, help="use only the first N pairs")
    a = ap.parse_args()

    pairs, records = pd.read_parquet(a.pairs), pd.read_parquet(a.records)
    print("pairs:", pairs.shape, "records:", records.shape)
    mp, mr = NEED_PAIRS - set(pairs.columns), NEED_REC - set(records.columns)
    if mp or mr:
        raise SystemExit(f"Missing columns -> pairs: {sorted(mp)} records: {sorted(mr)}\n"
                         f"pairs has {list(pairs.columns)}\nrecords has {list(records.columns)}\n"
                         "Rename them with df.rename(columns={...}) or ask your friend.")
    print("source values:", records["source"].value_counts().to_dict())
    print("pair ids missing from records: s1 =", int((~pairs["s1"].isin(records["id"])).sum()),
          "cand =", int((~pairs["cand"].isin(records["id"])).sum()))
    if a.n:
        pairs = pairs.head(a.n)

    t = time.time()
    feats = build_features(pairs, records)
    print(f"built {feats.shape} in {time.time() - t:.1f}s")
    x = feats[FEATURES]
    print("\nconstant columns:", [c for c in FEATURES if x[c].nunique() <= 1])
    print("share of -1 per feature:\n", (x == -1).mean().round(3).to_string())
    print("\nis_s3 values:", x["is_s3"].value_counts().to_dict())
    feats.to_parquet(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
