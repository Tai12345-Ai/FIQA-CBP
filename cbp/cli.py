from __future__ import annotations

import argparse
from pathlib import Path

from cbp.config import ExperimentConfig
from cbp.experiment import run_experiment
from cbp.synthetic.counterexample import counterexample_diagnostics, make_cycle_counterexample
from cbp.synthetic.faces import make_synthetic_face_embeddings
from cbp.synthetic.simulator import simulate_u_variance
from cbp.utils.io import save_json


def command_run(args) -> None:
    report = run_experiment(args.config)
    print(f"completed: {Path(args.config).resolve()}")
    print(f"selected: {report['selection']['selected']}")
    print(f"report: {Path(report['config']['output_dir']) / 'report.json'}")


def command_make_synthetic(args) -> None:
    data = make_synthetic_face_embeddings(
        n_identities=args.identities,
        captures_per_identity=args.captures,
        embedding_dim=args.dim,
        seed=args.seed,
    )
    data.table.save(args.output)
    print(f"saved {len(data.table.embeddings)} embeddings to {args.output}")


def command_e0(args) -> None:
    example = make_cycle_counterexample(args.states)
    analytic = example.operators.variance(example.action, args.identities)
    simulation = simulate_u_variance(
        example.p,
        example.action,
        example.operators.H,
        identity_budget=args.identities,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    result = {
        "counterexample": counterexample_diagnostics(example, args.identities),
        "analytic_variance": analytic,
        "simulation": simulation,
        "relative_error": abs(simulation["variance"] - analytic) / max(analytic, 1e-15),
    }
    if args.output:
        save_json(result, args.output)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cbp", description="Certified Behavioral Partitions")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the frozen-backbone full experiment")
    run.add_argument("--config", required=True)
    run.set_defaults(func=command_run)

    syn = sub.add_parser("make-synthetic", help="write a synthetic embedding NPZ")
    syn.add_argument("--output", required=True)
    syn.add_argument("--identities", type=int, default=300)
    syn.add_argument("--captures", type=int, default=5)
    syn.add_argument("--dim", type=int, default=64)
    syn.add_argument("--seed", type=int, default=20261021)
    syn.set_defaults(func=command_make_synthetic)

    e0 = sub.add_parser("e0", help="run analytic/synthetic operator checks")
    e0.add_argument("--states", type=int, default=16)
    e0.add_argument("--identities", type=int, default=50)
    e0.add_argument("--repetitions", type=int, default=20000)
    e0.add_argument("--seed", type=int, default=20261021)
    e0.add_argument("--output")
    e0.set_defaults(func=command_e0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
