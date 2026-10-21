import numpy as np

from cbp.synthetic.counterexample import counterexample_diagnostics, make_cycle_counterexample


def test_cycle_counterexample_preserves_means_but_breaks_influence():
    example = make_cycle_counterexample(16)
    diag = counterexample_diagnostics(example, identity_budget=100)
    assert diag["coverage_gap"] < 1e-12
    assert diag["utility_gap"] < 1e-12
    assert diag["risk_gap"] < 1e-12
    assert diag["A_action_norm"] < 1e-10
    assert diag["A_projected_norm"] > 1e-6

    # The obstruction is asymptotic: the uncompressed action has zero
    # first-order influence (Theta(n^-2)), while the projected action has
    # nonzero first-order influence (Theta(n^-1)). At moderate n the
    # degenerate constant can still make the projected variance smaller.
    # Check the claimed eventual variance inflation in a sufficiently large
    # identity-budget regime rather than assuming it already holds at n=100.
    asymptotic = counterexample_diagnostics(example, identity_budget=10_000)
    assert asymptotic["variance_projected"] > asymptotic["variance_action"]


def test_counterexample_requires_valid_size():
    for n in (4, 10, 12):
        if n == 12:
            # 12 is valid: multiple of four and >=8.
            make_cycle_counterexample(n)
        else:
            try:
                make_cycle_counterexample(n)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for n={n}")
