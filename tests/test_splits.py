import numpy as np

from cbp.data.splits import SPLIT_NAMES, assert_identity_disjoint, make_identity_splits


def test_identity_splits_are_disjoint_and_complete():
    ids = np.repeat([f"id{i}" for i in range(50)], 3)
    fractions = {"repr": 0.30, "fit": 0.25, "dev": 0.15, "cal": 0.15, "test": 0.15}
    splits = make_identity_splits(ids, fractions, seed=7)
    assert len(splits.labels) == len(ids)
    assert set(np.unique(splits.labels)) == set(SPLIT_NAMES)
    assert_identity_disjoint(ids, splits.labels)
    assigned = sum(len(v) for v in splits.split_to_identities.values())
    assert assigned == len(np.unique(ids))


def test_leakage_is_detected():
    ids = np.array(["a", "a", "b", "b"])
    labels = np.array(["repr", "test", "repr", "repr"])
    try:
        assert_identity_disjoint(ids, labels)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity leakage was not detected")
