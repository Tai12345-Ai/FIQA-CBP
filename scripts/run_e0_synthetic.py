from __future__ import annotations

import argparse
from cbp.synthetic.counterexample import counterexample_diagnostics, make_cycle_counterexample
from cbp.synthetic.simulator import simulate_u_variance
from cbp.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--identities", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", default="outputs/e0/report.json")
    args = parser.parse_args()

    example = make_cycle_counterexample(args.states)
    analytic = example.operators.variance(example.action, args.identities)
    simulation = simulate_u_variance(
        example.p,
        example.action,
        example.operators.H,
        args.identities,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    report = {
        "counterexample": counterexample_diagnostics(example, args.identities),
        "analytic_variance": analytic,
        "simulation": simulation,
        "relative_error": abs(simulation["variance"] - analytic) / max(analytic, 1e-15),
    }
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
