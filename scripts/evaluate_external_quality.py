from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cbp.baselines.external_scores import fit_external_score_policy, load_external_scores
from cbp.certification.certificate import certify_one_sided
from cbp.config import ExperimentConfig
from cbp.fr.embeddings import EmbeddingTable
from cbp.metrics.identity import evaluate_one_sided
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate external FIQA scores under the same split/certification protocol"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="completed canonical run with frozen thresholds/splits")
    parser.add_argument("--scores", required=True, help="CSV: sample_id, score")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--name", required=True, help="e.g. SER-FIQ, CR-FIQA, PFE-uncertainty")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    table = EmbeddingTable.load(cfg.embedding_npz).normalized()
    scores = load_external_scores(args.scores, table.sample_ids, args.score_column)
    run_dir = Path(args.run_dir)
    split_frame = pd.read_csv(run_dir / "split_manifest.csv")
    with open(run_dir / "report.json", "r", encoding="utf-8") as f:
        canonical = json.load(f)

    split_by_sample = dict(
        zip(split_frame["sample_id"].astype(str), split_frame["split"].astype(str))
    )
    split_labels = np.asarray([split_by_sample.get(str(x), "excluded") for x in table.sample_ids])
    indices = {name: np.flatnonzero(split_labels == name) for name in ("fit", "dev", "cal", "test")}
    if any(len(value) == 0 for value in indices.values()):
        raise RuntimeError("fit/dev/cal/test could not be aligned from split_manifest.csv")

    fit = table.subset(indices["fit"])
    dev = table.subset(indices["dev"])
    cal = table.subset(indices["cal"])
    test = table.subset(indices["test"])
    fit_scores = scores[indices["fit"]]
    dev_scores = scores[indices["dev"]]
    cal_scores = scores[indices["cal"]]
    test_scores = scores[indices["test"]]

    adjusted_delta = cfg.certification_delta / max(1, len(cfg.contexts))
    report = {
        "method": args.name,
        "higher_is_better": not args.lower_is_better,
        "score_file": str(Path(args.scores).resolve()),
        "contexts": {},
    }
    for context in cfg.contexts:
        threshold = float(canonical["thresholds"][context.name]["threshold"])
        identity_budget = int(
            context.identity_budget or len(np.unique(cal.identity_ids))
        )
        policy = fit_external_score_policy(
            fit,
            fit_scores,
            dev,
            dev_scores,
            matcher_threshold=threshold,
            target_fmr=context.target_fmr,
            identity_budget=identity_budget,
            rho=context.rho,
            coverage_low=context.coverage_low,
            coverage_high=context.coverage_high,
            reserve=context.reserve,
            higher_is_better=not args.lower_is_better,
        )
        if not policy.feasible:
            report["contexts"][context.name] = {
                "status": "NO_DEV_FEASIBLE_THRESHOLD"
            }
            continue
        gate_cal = policy.gate(cal_scores)
        gate_test = policy.gate(test_scores)
        certificate = certify_one_sided(
            cal.embeddings,
            cal.identity_ids,
            gate_cal,
            threshold=threshold,
            target_fmr=context.target_fmr,
            coverage_target_low=context.coverage_low,
            coverage_target_high=context.coverage_high,
            delta=adjusted_delta,
        )
        test_metrics = evaluate_one_sided(
            test.embeddings,
            test.identity_ids,
            gate_test,
            threshold,
        )
        report["contexts"][context.name] = {
            "status": "OK",
            "score_threshold": policy.threshold,
            "fit_statistics": policy.fit_statistics,
            "dev_metrics": policy.dev_metrics,
            "certificate": certificate,
            "test_metrics": test_metrics,
        }
    save_json(report, args.output)
    print(json.dumps(report, default=lambda x: getattr(x, "__dict__", str(x)), indent=2))


if __name__ == "__main__":
    main()
