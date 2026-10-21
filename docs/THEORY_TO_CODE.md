# Theory-to-Code Map

This map prevents a component name from being mistaken for a proven guarantee.
Equation numbers refer to the current merged CBP proposal where applicable.

| Scientific object | Implementation | Scope / check |
|---|---|---|
| Hard matcher and centered kernel `H=M-alpha` | `cbp/statistics/finite_state.py` | Exact on finite state space |
| First-order influence operator `A` | `build_evidence_operators` | Weighted `L2(p)` adjoint tested |
| Degenerate operator `V` | `build_evidence_operators` | Pointwise squared kernel, not operator composition |
| Identity-budget operator `Q_n` | `EvidenceOperators.q` | Weighted PSD projection for numerical stability |
| Cell basis `Z_k` | `indicator_matrix` | Hard disjoint cells |
| Cell masses/rewards/risks `p,u,ell` | `moments.py`, `identity_kernel.py` | State exact or identity-cluster plug-in |
| Cell covariance `B=Z*Q_n Z` | `aggregate_cell_moments` / identity kernel | Identity-budget specific |
| Policy Eq. (37) | `cbp/policy/solver.py` | CVXPY SOCP; SLSQP fallback checked for feasibility |
| Decoded utility/risk/evidence profile | `cbp/policy/quality.py` | Cell-level, context/population dependent |
| Context quality score Eq. (4) | `context_quality_score` | Requires valid SOCP duals and positive evidence norm |
| Mean/evidence partition features | `cbp/representation/partitions.py` | Construction heuristic, not theorem by itself |
| Boundary/sign-guided refinement | `refinement.py` | Greedy empirical implementation assumption |
| Router transfer diagnostics T5 | `cbp/router/transfer.py` | Measures label and policy differences; does not ensure training success |
| Sample-split discipline T6 | `cbp/experiment.py` | Conditional on identity-disjoint manifests and frozen candidates |
| Independent certificate | `cbp/certification/certificate.py` | Conservative valid baseline, distinct from fitted `Q_n` objective |
| P1 hard-cosine obstruction | `cbp/synthetic/counterexample.py` | Synthetic existence result only |
| E1 held-out phenomenon | `cbp/analysis/phenomenon.py` | Diagnostic permutation test; nuisance controls remain dataset-specific |
| E4 factorial challenge | `factorial_design` in `report.json` | Hard/continuous/dense x mean/evidence |

## Why two operator implementations exist

### Finite-state construction operator

`finite_state.py` implements the exact algebra on a discrete observable state
space. It is used to:

- test formulas exactly;
- construct teacher features/partitions;
- compute projection diagnostics;
- instantiate counterexamples.

It should not be silently interpreted as the exact raw-image population
operator.

### Identity-cluster fitted operator

`identity_kernel.py` operates directly on independent identity clusters and
actual cell labels. It is used to:

- refit deployed-router moments;
- estimate the covariance price for a planned identity budget;
- solve the operational policy.

It is a plug-in estimator, not an independent confidence certificate.

## Why certification is separate

The policy objective may use an estimated covariance penalty to prefer more
stable decisions. A valid deployment claim additionally requires independent
calibration. Therefore:

```text
representation/policy fitting on repr+fit+dev
                    !=
independent certificate on cal
                    !=
final descriptive evaluation on test
```

Conflating these three layers is a methodological error.
