from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen-backbone CBP reports")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="outputs/run_comparison.csv")
    args = parser.parse_args()
    rows = []
    for report_path in args.reports:
        report = load(report_path)
        run = Path(report_path).parent.name
        selected = report["selection"]["selected"]
        for context, item in report["deployed_cbp"]["contexts"].items():
            metrics = item["test_metrics"]
            cert = item["certificate"]
            rows.append({
                "run": run,
                "context": context,
                "selected_partition": selected,
                "certificate": cert["status"],
                "coverage": metrics["coverage"],
                "genuine_yield": metrics["genuine_yield"],
                "retained_tar": metrics["retained_tar"],
                "fmr": metrics["fmr"],
                "fmr_upper": cert["fmr_upper"],
                "genuine_yield_lower": cert["genuine_yield_low"],
            })
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
