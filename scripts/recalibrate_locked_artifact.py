from __future__ import annotations

import argparse
import numpy as np

from cbp.certification.certificate import certify_one_sided
from cbp.fr.embeddings import EmbeddingTable
from cbp.inference import CBPInferenceEngine
from cbp.metrics.identity import evaluate_one_sided
from cbp.policy.solver import solve_policy
from cbp.statistics.identity_kernel import estimate_identity_cluster_moments
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-domain refit and independent recertification")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--embeddings", required=True, help="NPZ with identity-disjoint fit/cal/test split_labels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = CBPInferenceEngine(args.artifact, device=args.device)
    table = EmbeddingTable.load(args.embeddings).normalized()
    if table.split_labels is None:
        raise ValueError("target NPZ must contain split_labels with fit/cal/test")
    labels = np.asarray(table.split_labels).astype(str)
    required = {"fit", "cal", "test"}
    if not required.issubset(set(labels)):
        raise ValueError(f"target split_labels must include {sorted(required)}")

    cells = engine.predict_cells(table.embeddings)
    n_cells = engine.router.n_cells
    report = {"contexts": {}, "warning": "Policy refit and certificate are target-domain specific."}
    adjusted_delta = args.delta / max(1, len(engine.policy["contexts"]))
    for context, original in engine.policy["contexts"].items():
        idx_fit = np.flatnonzero(labels == "fit")
        idx_cal = np.flatnonzero(labels == "cal")
        idx_test = np.flatnonzero(labels == "test")
        threshold = float(original["threshold"])
        target_fmr = float(original["target_fmr"])
        coverage_low = float(original["coverage_low"])
        coverage_high = float(original["coverage_high"])
        identity_budget = len(np.unique(table.identity_ids[idx_cal]))
        moments, diagnostics = estimate_identity_cluster_moments(
            table.embeddings[idx_fit],
            table.identity_ids[idx_fit],
            cells[idx_fit],
            threshold=threshold,
            alpha=target_fmr,
            identity_budget=identity_budget,
            n_cells=n_cells,
        )
        solution = solve_policy(
            moments,
            rho=args.rho,
            coverage_low=coverage_low,
            coverage_high=coverage_high,
        )
        gate_cal = solution.beta[cells[idx_cal]]
        gate_test = solution.beta[cells[idx_test]]
        certificate = certify_one_sided(
            table.embeddings[idx_cal],
            table.identity_ids[idx_cal],
            gate_cal,
            threshold=threshold,
            target_fmr=target_fmr,
            coverage_target_low=coverage_low,
            coverage_target_high=coverage_high,
            delta=adjusted_delta,
        )
        test_metrics = evaluate_one_sided(
            table.embeddings[idx_test],
            table.identity_ids[idx_test],
            gate_test,
            threshold,
        )
        report["contexts"][context] = {
            "fit_solution": solution,
            "fit_identities": diagnostics.n_identities,
            "fit_pairs": diagnostics.n_pairs,
            "certificate": certificate,
            "test_metrics": test_metrics,
        }
    save_json(report, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
