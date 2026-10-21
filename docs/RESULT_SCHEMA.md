# Result Schema

The canonical runner writes `report.json`. All arrays are serialized as JSON
lists; non-finite floating values become `null`.

## Top-level fields

### `scientific_status`

Implementation status and explicit warning that a code run does not validate
scientific hypotheses.

### `config`

Fully resolved experiment configuration.

### `environment`

Python, OS and package versions.

### `input_hashes`

SHA-256 of the embedding cache and YAML configuration.

### `split_report`

Number of samples and independent identities per split, plus removed singleton
or under-supported captures.

### `thresholds`

Per context:

- frozen cosine threshold;
- number of sampled construction impostor scores;
- empirical construction FMR;
- target FMR.

The score-pair count is not treated as independent sample size.

### `state_space`

State count, embedding dimension and identity-balanced state masses.

### `phenomenon_E1`

Held-out mean-matched evidence diagnostic:

- observed evidence distance;
- mean-match distance;
- permutation null mean/std;
- permutation p-value;
- retained matched-pair count.

### `dense_fit_diagnostics`

Identity/pair counts and eigenvalues of fitted first-order/degenerate covariance
components.

### `candidates`

Every predeclared hard/dense partition candidate:

- family and `K`;
- state-to-cell map;
- fit/dev feasibility;
- selection score;
- fitted policy solutions;
- dev verification metrics;
- projection diagnostics;
- moment/eigenvalue summaries.

### `selection`

Frozen CBP teacher candidate selected on dev and any feasibility warning.

### `scalar_baselines`

Common-ranking and context-conditioned ranking policies fitted with the same
moments/constraints.

### `continuous_baselines`

Shared continuous bottleneck under:

- evidence-aware objective;
- mean-risk objective.

Includes fit feasibility and held-out dev metrics.

### `factorial_design`

The decisive arms:

```text
hard_cbp__mean_risk
hard_cbp__evidence
dense_state__mean_risk
dense_state__evidence
continuous__mean_risk
continuous__evidence
```

Each arm reports context-level fit solution/feasibility, dev feasibility and
worst-context dev yield.

### `deployed_cbp`

Actual inductive-router system:

- router train/validation history;
- actual-router refitted moments;
- actual policy solution;
- policy-transfer diagnostics;
- dev metrics;
- independent calibration certificate;
- final test metrics.

### `test_tables`

Frozen descriptive test results for all predeclared candidates, scalar
baselines and factorial arms. The selected system is still determined before
these results are read.

### `paired_test_comparisons`

Paired A-B comparisons between deployed CBP and predeclared controls using a
Gaussian multiplier bootstrap over identity-level influence functions. It
reports point differences, confidence intervals and the probability of a
positive difference. This is asymptotic comparative inference, not the
finite-sample deployment certificate.

### `gates`

Gate A-D status with interpretation notes. Gate B uses the deployed router and
a matched-`K` mean-only control. Gate C is computed against feasible
dense/shared-continuous/full-image-neural evidence-aware controls.

## Locked artifact

### `artifact/router.pt`

Router architecture, state dict and metadata.

### `artifact/policy.json`

Per context:

- matcher threshold and target FMR;
- coverage band and identity budget;
- certificate status;
- policy table `beta`;
- decoded `mu_g`, `mu_I`, evidence response;
- optional dual-derived quality score.

### `artifact/state_space.npz`

Teacher construction centroids, teacher state-to-cell mapping and state masses.
This supports audit; deployed inference uses the learned router.
