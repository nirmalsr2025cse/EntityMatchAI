"""Pair features (section 5.4). Missing value -> -1, never NaN.

build_features(pairs, records) -> DataFrame[s1, cand, <FEATURES...>]
Deliberately NOT used: entity_id text, country (one-hot or match).
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

NAME = ["name_ratio", "name_token_sort", "name_token_set", "name_jaro_winkler",
        "name_jaccard", "name_first_token_match", "name_tfidf_word_cos",
        "name_idf_overlap", "name_len_ratio"]
ADDR = ["addr_ratio", "addr_token_set", "addr_jaccard", "addr_tfidf_cos", "addr_len_ratio"]
STRUCT = ["postal_match", "house_no_match", "numeric_token_jaccard", "addr_missing_either"]
BLOCK = ["block_rank", "block_sim_name", "block_sim_combo"]
SRC = ["is_s3"]
FEATURES = NAME + ADDR + STRUCT + BLOCK + SRC
BASIC = ["name_ratio", "name_token_sort", "name_token_set", "name_jaro_winkler",
         "addr_ratio", "addr_token_set", "postal_match"]

_NUM = re.compile(r"\b\d+[a-z]?\b")


def _toks(s):
    return set(s.split())


def _jaccard(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _len_ratio(a, b):
    if not a or not b:
        return -1.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _row_cos(X, i, j):
    return np.asarray(X[i].multiply(X[j]).sum(axis=1)).ravel()


def build_features(pairs: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rec = records.drop_duplicates("id").set_index("id")
    pos = {rid: k for k, rid in enumerate(rec.index)}
    i = pairs["s1"].map(pos).to_numpy()
    j = pairs["cand"].map(pos).to_numpy()
    if np.isnan(i.astype(float)).any() or np.isnan(j.astype(float)).any():
        raise KeyError("some pair ids are not in records")

    col = lambda c: rec[c].fillna("").to_numpy()
    nc, nf = col("name_core"), col("name_full")
    ad, pc, hn = col("addr_norm"), col("postal"), col("house_no")
    A_nc, B_nc = nc[i], nc[j]
    A_nf, B_nf = nf[i], nf[j]
    A_ad, B_ad = ad[i], ad[j]

    out = pd.DataFrame({"s1": pairs["s1"].to_numpy(), "cand": pairs["cand"].to_numpy()})

    # ---------------- name ----------------
    name_ok = np.array([bool(a) and bool(b) for a, b in zip(A_nf, B_nf)], dtype=bool)
    def nm(fn):
        return np.array([fn(a, b) / 100.0 if ok else -1.0
                         for a, b, ok in zip(A_nf, B_nf, name_ok)])
    out["name_ratio"] = nm(fuzz.ratio)
    out["name_token_sort"] = nm(fuzz.token_sort_ratio)
    out["name_token_set"] = nm(fuzz.token_set_ratio)
    out["name_jaro_winkler"] = np.array(
        [JaroWinkler.normalized_similarity(a, b) if ok else -1.0
         for a, b, ok in zip(A_nf, B_nf, name_ok)])
    ta, tb = [_toks(x) for x in A_nc], [_toks(x) for x in B_nc]
    out["name_jaccard"] = [_jaccard(a, b) for a, b in zip(ta, tb)]
    out["name_first_token_match"] = [
        -1.0 if not (x and y) else float(x.split()[0] == y.split()[0])
        for x, y in zip(A_nc, B_nc)]
    out["name_len_ratio"] = [_len_ratio(a, b) for a, b in zip(A_nf, B_nf)]

    vn = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
    try:
        X = vn.fit_transform(rec["name_core"].fillna("").to_numpy())
        cos = _row_cos(X, i, j)
        idf = dict(zip(vn.get_feature_names_out(), vn.idf_))
    except ValueError:  # empty vocabulary
        cos, idf = np.zeros(len(out)), {}
    out["name_tfidf_word_cos"] = np.where(name_ok, cos, -1.0)
    ov = []
    for a, b in zip(ta, tb):
        if not a or not b:
            ov.append(-1.0); continue
        den = sum(idf.get(t, 1.0) for t in a | b)
        ov.append(sum(idf.get(t, 1.0) for t in a & b) / den if den else -1.0)
    out["name_idf_overlap"] = ov

    # ---------------- address ----------------
    addr_ok = np.array([bool(a) and bool(b) for a, b in zip(A_ad, B_ad)], dtype=bool)
    def ax(fn):
        return np.array([fn(a, b) / 100.0 if ok else -1.0
                         for a, b, ok in zip(A_ad, B_ad, addr_ok)])
    out["addr_ratio"] = ax(fuzz.ratio)
    out["addr_token_set"] = ax(fuzz.token_set_ratio)
    out["addr_jaccard"] = [_jaccard(_toks(a), _toks(b)) for a, b in zip(A_ad, B_ad)]
    out["addr_len_ratio"] = [_len_ratio(a, b) for a, b in zip(A_ad, B_ad)]
    va = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
    try:
        XA = va.fit_transform(rec["addr_norm"].fillna("").to_numpy())
        acos = _row_cos(XA, i, j)
    except ValueError:
        acos = np.zeros(len(out))
    out["addr_tfidf_cos"] = np.where(addr_ok, acos, -1.0)

    # ---------------- structured address ----------------
    def eq(a, b):
        return -1.0 if not (a and b) else float(a == b)
    out["postal_match"] = [eq(pc[x], pc[y]) for x, y in zip(i, j)]
    out["house_no_match"] = [eq(hn[x], hn[y]) for x, y in zip(i, j)]
    out["numeric_token_jaccard"] = [
        _jaccard(set(_NUM.findall(a)), set(_NUM.findall(b))) for a, b in zip(A_ad, B_ad)]
    out["addr_missing_either"] = (~addr_ok).astype(float)

    # ---------------- blocking + source ----------------
    for c in BLOCK:
        out[c] = pd.to_numeric(pairs[c], errors="coerce").fillna(-1).to_numpy()
    src = rec["source"].to_numpy()[j]
    out["is_s3"] = np.array([float(str(s).lower() in ("s3", "3", "source3")) for s in src])

    out[FEATURES] = out[FEATURES].astype(float).fillna(-1.0)
    return out[["s1", "cand"] + FEATURES]