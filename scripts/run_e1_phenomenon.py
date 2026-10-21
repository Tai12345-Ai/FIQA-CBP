"""Inspect the held-out E1 phenomenon diagnostics from a completed run.

The canonical experiment computes E1 before router training and stores it in
report.json. This script intentionally does not recompute E1 with a different
split or hidden hyperparameters after seeing results.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/full_arcface/report.json")
    args = parser.parse_args()
    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)
    payload = {
        "phenomenon_E1": report.get("phenomenon_E1", {}),
        "gate_A": report.get("gates", {}).get("A_phenomenon"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
