import numpy as np
from src.train import label_pairs
from src.train_lgbm import make_lgbm, oof_predict, run
from src.features import FEATURES
from tests.test_train import synth


def test_oof_covers_every_pair_and_is_probability():
    f, tm = synth()
    y, _ = label_pairs(f, tm)
    oof = oof_predict(f, y, FEATURES, n_splits=3, n_estimators=20)
    assert len(oof) == len(f) and ((oof >= 0) & (oof <= 1)).all()
    assert oof[y == 1].mean() > oof[y == 0].mean()

def test_run_learns_signal_and_returns_all_stages():
    f, tm = synth()
    f["block_sim_combo"] = f["name_ratio"]
    res, model, pf, t = run(f, tm, n_splits=3, n_estimators=20)
    for k in ("a_baseline", "b_lgbm_basic", "c_lgbm_full", "d_final", "blocking_recall"):
        assert k in res
    assert res["d_final"] > 0.9 and 0.3 <= t <= 0.98

def test_model_has_expected_settings():
    m = make_lgbm(10)
    assert m.get_params()["class_weight"] == "balanced" and m.get_params()["random_state"] == 42