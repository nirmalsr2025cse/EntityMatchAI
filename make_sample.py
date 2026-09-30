"""
Build a consistent sample of the training data (ground-truth aware).

Idea:
  1. Randomly pick `frac` of the Source 1 entities.
  2. Keep ALL Source 2/3 records that the ground truth says match those entities.
  3. Add random "orphan" S2/S3 records (records that match no S1 at all) at the
     same `frac` rate, so the match-to-noise ratio stays like the real data.
  4. Write the sample with the same file names, so all code just needs a different folder.

Usage:
  python make_sample.py --frac 0.05 --seed 42 --src dataset/train --out dataset/sample_5pct
  python make_sample.py --frac 0.01 --seed 42 --src dataset/train --out dataset/sample_1pct

Same seed + same input files => you and your teammate get the identical sample,
so nobody needs to share sample files.

If the row counts of the sample files look off versus `wc -l`, check quoting:
names may contain a double quote character. If so, add quoting=3 (csv.QUOTE_NONE)
to the read_csv calls here AND in the main pipeline.
"""
import argparse
import os

import numpy as np
import pandas as pd


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kw)


def split_ids(cell):
    return [x.strip() for x in cell.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/train")
    ap.add_argument("--out", default="dataset/sample")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=500_000)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- Source 1: pick the sampled entities -------------------------------
    s1 = read_tsv(f"{a.src}/train_source1.tsv")
    s1_mask = rng.random(len(s1)) < a.frac
    s1_sample = s1[s1_mask]
    sel = set(s1_sample.entity_id)
    s1_sample.to_csv(f"{a.out}/train_source1.tsv", sep="\t", index=False)
    del s1

    # ---- Ground truth: matches of the sampled entities + all matched ids ---
    gt = read_tsv(f"{a.src}/train_ground_truth.tsv")
    matched_sel, all_matched = set(), set()
    for s1_id, cell in zip(gt.source1_entity_id, gt.matched_entity_ids):
        ids = split_ids(cell)
        all_matched.update(ids)
        if s1_id in sel:
            matched_sel.update(ids)
    gt_sample = gt[gt.source1_entity_id.isin(sel)]
    gt_sample.to_csv(f"{a.out}/train_ground_truth.tsv", sep="\t", index=False)
    del gt

    # ---- Source 2 / 3: stream in chunks ------------------------------------
    found_all = set()
    for name in ("train_source2.tsv", "train_source3.tsv"):
        out_path = f"{a.out}/{name}"
        first, found, kept, orphans = True, set(), 0, 0
        for ch in read_tsv(f"{a.src}/{name}", chunksize=a.chunk):
            ids = ch.entity_id
            is_match = ids.isin(matched_sel)
            is_orphan = ~ids.isin(all_matched)
            pick_orphan = is_orphan & (rng.random(len(ch)) < a.frac)
            keep = is_match | pick_orphan
            out = ch[keep]
            out.to_csv(out_path, sep="\t", index=False,
                       mode="w" if first else "a", header=first)
            first = False
            found.update(ids[is_match])
            kept += len(out)
            orphans += int(pick_orphan.sum())
        found_all.update(found)
        print(f"{name}: kept {kept:,} rows ({orphans:,} random orphans)")

    # ---- Sanity report ------------------------------------------------------
    n_single = int((gt_sample.matched_entity_ids.str.strip() == "").sum())
    print(f"S1 sampled: {len(sel):,}  |  singletons: {n_single:,} "
          f"({n_single / max(len(sel), 1):.1%})")
    print(f"Ground-truth matched ids for sampled S1: {len(matched_sel):,}")
    missing = len(matched_sel - found_all)
    print(f"Matched ids NOT found in the S2/S3 files: {missing:,} "
          "(should be 0; if not, the ground truth references records missing from the files)")
    print(f"Sample written to: {a.out}")


if __name__ == "__main__":
    main()
