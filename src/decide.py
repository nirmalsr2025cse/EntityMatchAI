"""Decision layer: threshold, then each S2/S3 record goes to its best S1 only."""


def decide(pairs, t):
    """pairs: DataFrame[s1, cand, p]. Returns {s1: set(cand)} (S1 without matches are absent)."""
    keep = pairs[pairs.p >= t].sort_values("p", ascending=False)
    keep = keep.drop_duplicates("cand")
    return keep.groupby("s1").cand.apply(set).to_dict()