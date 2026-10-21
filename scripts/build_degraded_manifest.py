from __future__ import annotations

import argparse

from cbp.data.degradation import build_degraded_manifest
from cbp.data.manifest import FaceManifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic degraded-image manifest for the CFP Run-2 mechanism pilot"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition", required=True, choices=["lowres_32", "blur_3", "jpeg_20"])
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    clean = FaceManifest.from_csv(args.manifest)
    degraded = build_degraded_manifest(
        clean,
        output_root=args.output_root,
        condition=args.condition,
    )
    degraded.to_csv(args.output_manifest)
    print(
        f"condition={args.condition}; rows={len(degraded.frame)}; "
        f"manifest={args.output_manifest}"
    )


if __name__ == "__main__":
    main()
