from __future__ import annotations

import argparse
from cbp.synthetic.faces import make_synthetic_face_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_embeddings.npz")
    parser.add_argument("--identities", type=int, default=300)
    parser.add_argument("--captures", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    data = make_synthetic_face_embeddings(
        n_identities=args.identities,
        captures_per_identity=args.captures,
        embedding_dim=args.dim,
        seed=args.seed,
    )
    data.table.save(args.output)
    print(f"saved {len(data.table.embeddings)} samples to {args.output}")


if __name__ == "__main__":
    main()
