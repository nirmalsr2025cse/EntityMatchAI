import pandas as pd
from src.evaluate import f05_entity, macro_f05
from src.decide import decide


def test_singleton_empty_is_one():
    assert f05_entity(set(), set()) == 1.0

def test_singleton_predicted_anything_is_zero():
    assert f05_entity({"S2-1"}, set()) == 0.0

def test_nonsingleton_empty_is_zero():
    assert f05_entity(set(), {"S2-1"}) == 0.0

def test_perfect_and_disjoint():
    assert f05_entity({"a", "b"}, {"a", "b"}) == 1.0
    assert f05_entity({"x"}, {"a"}) == 0.0

def test_precision_weighted_more():
    # P=1,R=.5 -> 0.8333 ; P=.5,R=1 -> 0.5556
    assert abs(f05_entity({"a"}, {"a", "b"}) - 1.25 * 0.5 / (0.25 + 0.5)) < 1e-9
    assert f05_entity({"a"}, {"a", "b"}) > f05_entity({"a", "x"}, {"a"})

def test_macro_missing_keys_and_singletons():
    true = {"s1": {"a"}, "s2": set(), "s3": set()}
    assert abs(macro_f05({"s1": {"a"}}, true) - 1.0) < 1e-9        # s2, s3 missing -> empty -> 1.0
    assert abs(macro_f05({"s1": {"a"}, "s2": {"z"}}, true) - 2 / 3) < 1e-9

def test_decide_threshold_and_one_s1_per_cand():
    pairs = pd.DataFrame({"s1": ["A", "B", "A", "B"], "cand": ["c1", "c1", "c2", "c3"],
                          "p": [0.9, 0.8, 0.6, 0.2]})
    out = decide(pairs, 0.5)
    assert out == {"A": {"c1", "c2"}}          # c1 -> A only; c3 below threshold; B absent
