from __future__ import annotations

import argparse
from pathlib import Path

from cbp.fr.embeddings import EmbeddingTable
from cbp.inference import CBPInferenceEngine
from cbp.metrics.identity import evaluate_one_sided
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a locked CBP artifact on another domain")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = CBPInferenceEngine(args.artifact, device=args.device)
    table = EmbeddingTable.load(args.embeddings).normalized()
    report = {
        "artifact": str(Path(args.artifact).resolve()),
        "embeddings": str(Path(args.embeddings).resolve()),
        "contexts": {},
        "warning": (
            "This is locked cross-domain evaluation. The original certificate does not transfer. "
            "Recalibration on independent target-domain identities is required for a target-domain certificate."
        ),
    }
    for context, item in engine.policy["contexts"].items():
        gate = engine.gate_probabilities(table.embeddings, context)
        metrics = evaluate_one_sided(
            table.embeddings,
            table.identity_ids,
            gate,
            threshold=float(item["threshold"]),
        )
        report["contexts"][context] = metrics
    save_json(report, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
