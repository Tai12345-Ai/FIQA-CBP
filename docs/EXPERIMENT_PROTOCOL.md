# CBP Experimental Protocol

This document operationalizes the E0-E9 program in the current CBP proposal.
It is a falsification protocol: each experiment has an explicit failure
interpretation. Passing a software test is not equivalent to passing a
scientific gate.

## 1. Pre-registration block

Freeze before primary execution:

- population and dataset versions;
- identity inclusion/exclusion criteria;
- FR checkpoint hashes and preprocessing;
- operating contexts `(matcher, threshold, target FMR, coverage band, n, rho)`;
- identity split manifest;
- observable state construction and candidate family;
- candidate count and hyperparameter budget;
- calibration failure budget and multiplicity allocation;
- primary endpoints and minimum useful margins;
- seed list and compute budget;
- fallback behavior for unsupported contexts/cells.

Any change after calibration requires a new independent calibration set or a
valid predeclared sequential procedure.

## 2. Independent units and splits

All splits are by identity:

| Split | Role | Prohibited use |
|---|---|---|
| `repr` | state construction and teacher candidate construction | final moment fitting or certification |
| `fit` | estimate deployed moments and solve policies | selecting candidates after calibration |
| `dev` | candidate selection, router early stopping, diagnostics | final certificate |
| `cal` | independent certification only | model/hyperparameter changes |
| `test` | final locked report | any selection or retuning |

Pairs that share an identity are dependent. Do not bootstrap pairs as iid.
Identity-level resampling must reconstruct all relevant pair observations.

## 3. E0 — Mathematical and certificate validation

### E0.1 Finite-state operator

For small discrete state spaces, enumerate all identity samples and compare:

- empirical/exact variance of the order-2 U-statistic;
- the quadratic form `<a, Q_n a>`;
- weighted self-adjointness and PSD tolerance.

**Failure:** any systematic moment mismatch or negative quadratic form outside
numerical tolerance blocks real-data claims.

### E0.2 Mean-preserving obstruction

Use the hard-cosine cycle construction. Verify that compression preserves:

- coverage;
- genuine utility;
- mean excess risk;

while changing first-order influence/evidence variance.

**Failure:** implementation does not instantiate the claimed obstruction.

### E0.3 Certificate coverage

Under a known synthetic population, repeat independent calibration experiments
and measure false-pass/undercoverage. The nominal error budget must be respected
within Monte Carlo uncertainty.

**Failure:** undercoverage blocks use of the certificate.

## 4. E1 — Real-face phenomenon

### Hypothesis

States that are matched in mean genuine utility and mean impostor risk can
retain different identity-dependent evidence responses on unseen identities.

### Design

1. Build observable states on `D_repr`.
2. On `D_dev`, estimate state mass and genuine behavior with identity-balanced
   weights.
3. Match states in standardized `(g, r)` space.
4. Compare declared evidence feature vectors.
5. Use a fixed permutation diagnostic and report effect size, not only p-value.
6. Repeat across at least two frozen FR checkpoints.

### Controls

When available, stratify or regress out:

- embedding norm;
- pose;
- blur/resolution;
- occlusion;
- score distribution;
- state occupancy;
- demographic composition.

The current code implements the core mean-matched permutation diagnostic. The
additional nuisance controls require dataset metadata and must be added to the
analysis table rather than claimed implicitly.

**Falsifier:** evidence difference disappears on held-out identities or is fully
explained by declared mean/basic-quality controls.

## 5. E2 — Compression and downstream value

Compare at matched `K`, data, candidate budget, solver, context and coverage:

- random cells;
- embedding k-means;
- mean-only cells;
- CBP evidence-aware cells.

Primary quantities:

- genuine yield;
- retained TAR;
- achieved FMR;
- coverage;
- identity budget needed for a declared feasible yield;
- certificate pass/non-pass.

Projection/cell-purity metrics are diagnostics only.

**Falsifier:** CBP improves operator diagnostics but not verification value at
matched FMR/coverage.

## 6. E3 — Unseen-identity generalization

All representation construction, moment fitting, candidate selection,
calibration and test identities must be disjoint. Report:

- teacher representation behavior on `repr`;
- candidate behavior on `dev`;
- actual deployed-router behavior after refit;
- final locked test behavior.

**Falsifier:** gains appear only on construction/fitting identities.

## 7. E4 — Decisive strong-baseline factorial

The canonical runner reports:

```text
{hard CBP, shared continuous bottleneck, dense state policy}
    x
{mean-risk objective, evidence-aware objective}
```

All arms receive the same frozen state features, moments, contexts and declared
training/tuning budget. This separates:

- objective contribution;
- representation restriction;
- optimization/training contribution;
- hard-cell necessity.

**Interpretation:**

- If evidence-aware dense/continuous beats mean-risk versions but hard CBP does
  not beat them, the evidence principle may survive while hard-partition
  superiority fails.
- If evidence-aware arms do not improve over mean-risk arms, the practical
  motivation for `Q_n` weakens.
- If hard CBP wins only with more tuning/capacity, the comparison is confounded.

## 8. E5 — Certification

Freeze representation, router, policy, thresholds, contexts and randomization
rules before `D_cal`.

Report for each context:

- calibration identity count;
- point estimates;
- coverage interval;
- impostor numerator upper bound;
- FMR upper bound;
- genuine-yield lower bound;
- `PASS` or `INSUFFICIENT_EVIDENCE`;
- all failed candidates/contexts.

Sweep planned identity budget `n` using `scripts/run_budget_sweep.py`, but do not
reinterpret a planned-budget sweep as additional independent data.

**Falsifier:** useful coverage/yield cannot be certified at any declared
realistic operating context.

## 9. E6 — Context reversal

Compare:

- one shared scalar ranking;
- context-conditioned scalar rankings;
- context-specific CBP policies over one shared representation.

A reversal is relevant only when it changes feasible actions near a decision
boundary. Rank disagreement away from the policy boundary is insufficient.

**Falsifier:** a common scalar ranking matches context-specific feasible value.

## 10. E7 — Capacity and training controls

Main track: frozen FR backbone.

Optional matched fine-tuning track:

- same trainable FR blocks across methods;
- same optimizer, steps, augmentations and parameter budget;
- identity-preservation objective;
- hard matcher outcomes recomputed after training;
- all moments/certificates refit from scratch.

Include an extra-capacity sham branch if architecture capacity differs.

**Falsifier:** gains disappear when parameters, updates and tuning budget are
equalized.

## 11. E8 — Identity-budget transfer

For several planned budgets `n`, recompute:

- first-order component;
- degenerate component;
- fitted policy;
- feasible value.

Ablate the first-order and degenerate covariance components separately.

**Falsifier:** one generic regularizer explains all budget behavior and the
claimed two-component geometry has no measurable consequence.

## 12. E9 — Router fidelity

Evaluate teacher versus deployed router on identity-disjoint data:

- cell accuracy;
- policy L1/L2 difference;
- policy-action disagreement;
- coverage/risk/yield difference;
- T5-style transfer terms;
- actual-router moment refit.

Label accuracy is secondary: confusing cells with identical policy action may
be harmless; small label error near a policy boundary may be harmful.

**Falsifier:** theoretical diagnostics exist only for oracle teacher cells and
collapse after induction.

## 13. Statistical reporting

- Separate training-seed variability from population uncertainty.
- Use the same identities for paired method comparisons.
- Use identity-cluster inference. The implementation provides a paired Gaussian
  multiplier bootstrap from identity influence functions; it does not resample
  image pairs as iid observations.
- Do not call overlapping calibration subsets independent repetitions.
- Report absolute and relative changes separately.
- Report mean and dispersion across seeds/checkpoints.
- Do not call a result statistically significant without a declared test/CI.
- Apply multiplicity correction across contexts/candidates when selected using
  the same calibration data.
- Treat ROC interpolation on test as descriptive unless the threshold was
  independently frozen.

## 14. Gate decisions

### Gate A — Phenomenon

Pass only with held-out evidence differences after declared controls.

### Gate B — Decision relevance

Pass only when evidence-aware representation changes feasible verification
value at matched FMR and coverage; improved error bars alone are insufficient.

### Gate C — Strong baseline

Pass only when hard CBP is not dominated by every feasible dense/shared
continuous same-objective comparator under matched information and budget.

### Gate D — Nonvacuity

Pass only when an independent certificate returns useful coverage/yield at a
realistic context.

A failed gate narrows or rejects the corresponding claim. Do not add auxiliary
modules merely to hide a failed central test.
