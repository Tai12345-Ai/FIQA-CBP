# Implementation Assumptions and Failure Conditions

This file records implementation choices that are not automatically implied by
the CBP theory. They must be disclosed in experiments and reconsidered when the
protocol changes.

## A1. Frozen face-recognition matcher

**Assumption.** The main experiment receives embeddings from one fixed FR
checkpoint, fixed detector/alignment/crop pipeline and cosine matcher.

**Why.** It isolates representation/policy effects from additional FR training.

**Failure condition.** Mixing checkpoints, preprocessing or normalization
changes the score distribution and invalidates fitted thresholds/moments.

## A2. One-sided probe filtering

**Assumption.** The gate depends only on the probe image; the reference/gallery
protocol is fixed.

**Why.** This is the convex core formulation implemented by the policy solver.

**Failure condition.** Two-sided filtering requires different coverage,
covariance and optimization objects. Do not reuse the one-sided certificate.

## A3. Identity as the independent unit

**Assumption.** Identities are iid draws from the declared target population;
captures and pairs within/between sampled identities are dependent observations.

**Why.** The pair-risk estimator is an order-2 identity-cluster U-statistic.

**Failure condition.** Related identities, duplicated subjects, leakage across
splits or population shift can invalidate uncertainty interpretation.

## A4. Finite observable state approximation

**Assumption.** Raw normalized embeddings are quantized into a finite state
space on `D_repr`; CBP operators are exact on this finite state space.

**Why.** It makes operator construction, matched controls and exact synthetic
checks computationally tractable.

**Failure condition.** Small or unstable state spaces may erase important raw
image distinctions. State count and assignment stability must be ablated.

## A5. State centroids and hard matcher

**Assumption.** The finite-state construction approximates state-to-state hard
matcher behavior by applying the frozen cosine threshold to state centroids.

**Why.** This defines a reviewable finite-state operator for teacher
construction.

**Failure condition.** A centroid may not represent a multimodal state and may
misestimate actual pair behavior. Deployed moments are therefore refit directly
from identity clusters and actual router labels.

## A6. Identity-balanced empirical population

**Assumption.** Each identity contributes equal total mass; captures within an
identity divide that mass.

**Why.** Web datasets often have strongly unequal capture counts. Image-uniform
weighting would let prolific identities define the population.

**Failure condition.** If deployment intentionally uses capture-weighted rather
than identity-weighted sampling, the estimand and all moments must be changed
explicitly.

## A7. Plug-in Hoeffding decomposition for fitting

**Assumption.** `identity_kernel.py` estimates first-order and degenerate
covariance components from complete identity pairs (or declared uniform pair
subsampling) and constructs `B` for a planned identity budget.

**Why.** This aligns policy fitting with identity-dependent uncertainty.

**Failure condition.** Plug-in `B` is not itself a finite-sample confidence
bound. Small fit identity counts, rare interactions or aggressive pair
subsampling may yield unstable estimates. Independent certification remains
mandatory.

## A8. Incomplete pair fitting

**Assumption.** Primary experiments use all identity pairs. Optional
`max_fit_identity_pairs` uniformly samples pairs for computational reasons.

**Why.** Complete pairs avoid an additional Monte Carlo error layer.

**Failure condition.** Sampled pairs that leave identities without partners are
rejected. Even with partner coverage, pair-sampling uncertainty is not currently
included in the policy robustification; report this as an approximation.

## A9. Policy solver

**Assumption.** CVXPY solves the convex SOCP. A SciPy SLSQP fallback is provided
for accessibility.

**Why.** CVXPY exposes solver status and dual multipliers needed for the analytic
quality score.

**Failure condition.** A fallback solution without duals cannot support the
exact dual-based quality readout; the code returns `NaN` rather than fabricating
one. Always report solver status, residuals and tolerance.

## A10. Fractional action

**Assumption.** `beta_k in [0,1]` represents genuine randomized retention with
independent randomization, or its Rao-Blackwellized expected gate in evaluation.

**Why.** Fractional policies preserve convexity and the stated quadratic
objective.

**Failure condition.** Hard-rounding changes moments, coverage and certificate.
A deterministic production gate must be rounded/fitted before calibration and
certified again.

## A11. Router training

**Assumption.** A shared MLP learns hard teacher cell labels from a single image
embedding. Router CE is a surrogate only.

**Why.** Deployment cannot use pairwise teacher construction at query time.

**Failure condition.** High cell accuracy does not guarantee low policy error.
Use policy disagreement diagnostics and refit all deployed moments from actual
router outputs.

## A12. Candidate selection

**Assumption.** A finite predeclared family is constructed on `repr`, fitted on
`fit`, selected on `dev`, locked before `cal` and evaluated once on `test`.

**Why.** This separates adaptive model selection from independent
certification.

**Failure condition.** Adding or retuning candidates after calibration consumes
the calibration evidence and invalidates the old bound.

## A13. Threshold calibration

**Assumption.** Frozen matcher thresholds are empirical impostor quantiles on
the construction split. Pair count is recorded but not treated as independent
sample size.

**Why.** The threshold defines each operating context before policy fitting.

**Failure condition.** Very-low-FMR quantiles may be unstable when identities or
score support are insufficient. Independent certificate validity does not make
a poorly estimated threshold operationally useful.

## A14. Conservative certificate

**Assumption.** The current certificate uses bounded identity means and a
conservative order-2 U-statistic Hoeffding radius with context-level error-budget
allocation.

**Why.** It provides a transparent valid baseline and explicit nonvacuity test.

**Failure condition.** Bounds may be too conservative at low FMR. A sharper
bound can replace it only with a separately checked proof and simulation
coverage audit.

## A15. E1 permutation diagnostic

**Assumption.** Mean-matched states are selected using nearest neighbors in
standardized `(g, r)` space, and evidence vectors are permuted across states.

**Why.** It provides a falsifiable held-out diagnostic for residual evidence
structure.

**Failure condition.** State dependence, nuisance factors or adaptive choice of
matching tolerance can make the nominal p-value optimistic. Report effect size,
controls and multiplicity; do not call it causal proof.

## A16. Strong continuous baseline

**Assumption.** A shared learned bottleneck with one context head is trained
under matched mean/evidence objectives with the same state information.

**Why.** It tests whether hard partitions are needed beyond the evidence-aware
criterion.

**Failure condition.** Penalty training may fail to reach feasibility. Such a
run is marked infeasible, not treated as a weak comparator. Match optimization
budget and tune it fairly.

## A17. Domain transfer

**Assumption.** A locked artifact may be evaluated descriptively on another
domain, but its original certificate does not transfer.

**Why.** Image-only inference still depends on the learned population, matcher
and reference protocol.

**Failure condition.** New domain, matcher, gallery or acquisition process
requires target-domain moment refitting and independent calibration.

## A18. Raw image datasets and privacy

The core runner only needs embeddings and identity metadata. Do not expose raw
biometric datasets to tools or services unless dataset terms, consent, security
and institutional requirements permit it. Record dataset access and licenses.
