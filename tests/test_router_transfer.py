import numpy as np

from cbp.router.transfer import router_transfer_metrics


def test_policy_disagreement_can_be_smaller_than_label_error():
    teacher = np.array([0, 0, 1, 1, 2, 2])
    deployed = np.array([0, 1, 1, 1, 2, 0])
    beta = np.array([1.0, 1.0, 0.0])
    metrics = router_transfer_metrics(teacher, deployed, beta)
    assert metrics["label_error"] > 0
    assert metrics["policy_disagreement"] < metrics["label_error"]
    assert metrics["policy_l1"] >= 0
    assert metrics["policy_l2_sq"] >= 0
