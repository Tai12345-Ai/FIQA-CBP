from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import numpy as np
from tqdm import tqdm

from cbp.data.manifest import FaceManifest
from cbp.fr.embeddings import EmbeddingTable, InsightFaceExtractor
from cbp.fr.cosine import l2_normalize
from cbp.utils.io import save_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen InsightFace embeddings")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--provider", action="append", default=None)
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--failure-log", default=None)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    manifest = FaceManifest.from_csv(args.manifest).frame
    extractor = InsightFaceExtractor(args.model, providers=args.provider, ctx_id=args.ctx_id)
    embeddings = []
    keep_indices = []
    failures = []
    for row_index, row in enumerate(tqdm(manifest.itertuples(index=False), total=len(manifest))):
        try:
            value = extractor.extract([str(row.image_path)])[0]
            embeddings.append(value)
            keep_indices.append(row_index)
        except Exception as exc:
            failures.append((str(row.image_path), repr(exc)))

    if not embeddings:
        raise RuntimeError("no embedding was extracted")
    kept = manifest.iloc[keep_indices].reset_index(drop=True)
    split_labels = kept["split"].astype(str).to_numpy() if "split" in kept.columns else None
    table = EmbeddingTable(
        embeddings=l2_normalize(np.stack(embeddings)).astype(np.float32),
        identity_ids=kept["identity_id"].astype(str).to_numpy(),
        sample_ids=kept["sample_id"].astype(str).to_numpy(),
        image_paths=kept["image_path"].astype(str).to_numpy(),
        split_labels=split_labels,
    )
    table.save(args.output)

    failure_path = Path(args.failure_log or (str(args.output) + ".failures.tsv"))
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failure_path, "w", encoding="utf-8") as f:
        f.write("image_path\terror\n")
        for path, error in failures:
            f.write(f"{path}\t{error}\n")

    model_dir = Path.home() / ".insightface" / "models" / args.model
    model_hashes = {}
    if model_dir.exists():
        for path in sorted(model_dir.rglob("*.onnx")):
            model_hashes[str(path)] = sha256_file(path)
    metadata_path = Path(args.metadata or (str(args.output) + ".metadata.json"))
    package_versions = {}
    for package in ("insightface", "onnxruntime", "numpy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    save_json(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "output": str(Path(args.output).resolve()),
            "model_name": args.model,
            "providers": args.provider,
            "ctx_id": args.ctx_id,
            "package_versions": package_versions,
            "model_file_hashes": model_hashes,
            "samples_succeeded": len(table.embeddings),
            "samples_failed": len(failures),
            "embedding_dim": int(table.embeddings.shape[1]),
            "normalization": "L2",
            "detector_selection": "highest det_score face",
            "detector_input": "RGB loaded then converted to BGR",
        },
        metadata_path,
    )
    print(
        f"saved {len(table.embeddings)} embeddings to {args.output}; "
        f"failures={len(failures)}; metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
