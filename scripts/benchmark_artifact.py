from __future__ import annotations

import argparse
import json
import time
import numpy as np

from cbp.fr.embeddings import EmbeddingTable
from cbp.inference import CBPInferenceEngine
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark locked CBP router/table inference")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    engine = CBPInferenceEngine(args.artifact, device=args.device)
    table = EmbeddingTable.load(args.embeddings).normalized()
    z = table.embeddings[: min(args.batch_size, len(table.embeddings))]
    if len(z) == 0:
        raise ValueError("empty embedding table")
    context = next(iter(engine.policy["contexts"]))

    for _ in range(args.warmup):
        engine.gate_probabilities(z, context)
    durations = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        engine.gate_probabilities(z, context)
        durations.append(time.perf_counter() - start)
    durations = np.asarray(durations)
    report = {
        "scope": "router_plus_policy_lookup_only",
        "excludes": ["face_detection", "alignment", "FR_backbone_embedding"],
        "device": args.device,
        "context": context,
        "batch_size": len(z),
        "repetitions": args.repetitions,
        "latency_ms_mean": float(1000 * durations.mean()),
        "latency_ms_median": float(1000 * np.median(durations)),
        "latency_ms_p95": float(1000 * np.quantile(durations, 0.95)),
        "throughput_samples_per_second": float(len(z) / durations.mean()),
    }
    save_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
