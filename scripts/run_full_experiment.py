from __future__ import annotations

import argparse
from cbp.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_experiment(args.config)
    print(f"selected={report['selection']['selected']}")
    print(f"gates={report['gates']}")


if __name__ == "__main__":
    main()
