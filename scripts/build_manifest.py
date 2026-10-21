from __future__ import annotations

import argparse
from cbp.data.manifest import FaceManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root/identity_id/image.jpg layout")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = FaceManifest.from_identity_folders(args.root)
    manifest.to_csv(args.output)
    print(f"wrote {len(manifest.frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
