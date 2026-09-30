import numpy as np
import pandas as pd
from src.features import build_features, FEATURES

REC = pd.DataFrame([
    # id, source, country, name_full, name_core, addr_norm, postal, house_no, landmark
    ["S1-1", "S1", "us", "acme corporation", "acme", "12 main street springfield 62704", "62704", "12", ""],
    ["S2-1", "S2", "us", "acme corp", "acme", "12 main st springfield 62704", "62704", "12", ""],
    ["S3-1", "S3", "us", "zenith holdings llc", "zenith holdings", "99 oak ave", "", "99", ""],
    ["S2-2", "S2", "us", "acme corporation", "acme", "", "", "", ""],
    ["S1-2", "S1", "in", "sharma traders private limited", "sharma traders", "45 mg road near sbi atm bangalore 560001", "560001", "45", "sbi atm"],
    ["S2-3", "S2", "in", "sharma trader pvt ltd", "sharma trader", "45 mg rd bangalore 560001", "560001", "45", ""],
    ["S2-4", "S2", "in", "sharma traders", "sharma traders", "7 mg road bangalore 560002", "560002", "7", ""],
    ["S2-5", "S2", "us", "acme corporatoin", "acme corporatoin", "12 main street springfield 62704", "62704", "12", ""],
    ["S2-6", "S2", "us", "corporation acme", "acme corporation", "12 main street springfield 62704", "62704", "12", ""],
    ["S2-7", "S2", "us", "globex services", "globex services", "12 main street springfield 62704", "62704", "12", ""],
    ["S2-8", "S2", "us", "", "", "12 main street springfield 62704", "62704", "12", ""],
], columns=["id", "source", "country", "name_full", "name_core", "addr_norm", "postal", "house_no", "landmark"])

PAIRS = pd.DataFrame({
    "s1":   ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2", "S1-2",
             "S1-1", "S1-1", "S1-1", "S1-1"],
    "cand": ["S2-1", "S3-1", "S2-2", "S2-3", "S2-4", "S3-1",
             "S2-5", "S2-6", "S2-7", "S2-8"],
    "block_rank": [0, 3, 1, 0, 1, 9, 2, 2, 4, 5],
    "block_sim_name": [.9, .1, .8, .85, .8, .05, .7, .9, .1, .0],
    "block_sim_combo": [.95, .1, .5, .9, .7, .05, .75, .95, .2, .0],
})

def test_shape_columns_no_nan():
    f = build_features(PAIRS, REC)
    assert list(f.columns) == ["s1", "cand"] + FEATURES
    assert len(f) == len(PAIRS)
    assert not f[FEATURES].isna().any().any()
    assert (f[["s1", "cand"]].values == PAIRS[["s1", "cand"]].values).all()

def test_true_match_scores_higher_than_wrong():
    f = build_features(PAIRS, REC).set_index(["s1", "cand"])
    good, bad = f.loc[("S1-1", "S2-1")], f.loc[("S1-1", "S3-1")]
    assert good.name_token_set > bad.name_token_set
    assert good.name_jaro_winkler > bad.name_jaro_winkler
    assert good.postal_match == 1 and good.house_no_match == 1

def test_missing_gives_minus_one():
    f = build_features(PAIRS, REC).set_index(["s1", "cand"])
    r = f.loc[("S1-1", "S2-2")]          # candidate has no address / postal / house_no
    assert r.postal_match == -1 and r.house_no_match == -1
    assert r.addr_ratio == -1 and r.addr_tfidf_cos == -1 and r.addr_missing_either == 1

def test_postal_mismatch_is_zero_and_source_flag():
    f = build_features(PAIRS, REC).set_index(["s1", "cand"])
    assert f.loc[("S1-2", "S2-4")].postal_match == 0
    assert f.loc[("S1-2", "S2-4")].is_s3 == 0 and f.loc[("S1-1", "S3-1")].is_s3 == 1

def test_empty_pairs():
    f = build_features(PAIRS.iloc[:0], REC)
    assert len(f) == 0 and list(f.columns) == ["s1", "cand"] + FEATURES
