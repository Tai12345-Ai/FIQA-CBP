from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
from pathlib import Path
import numpy as np
from tqdm import tqdm

from cbp.data.manifest import FaceManifest
from cbp.fr.embeddings import (
    EmbeddingTable,
    FaceDetectionError,
    ImageDecodeError,
    InsightFaceExtractor,
)
from cbp.fr.cosine import l2_normalize
from cbp.utils.io import save_json, sha256_file


def _infer_view_label(row) -> str:
    """Infer CFP frontal/profile labels without changing the manifest schema."""
    for field in ("view", "pose", "subset"):
        value = getattr(row, field, None)
        if value is not None:
            label = str(value).strip().lower()
            if label in {"frontal", "profile"}:
                return label

    sample_id = str(getattr(row, "sample_id", ""))
    image_path = str(getattr(row, "image_path", ""))
    text = f"{sample_id} {image_path}".replace("\\", "/").lower()
    if ":frontal/" in text or "/frontal/" in text:
        return "frontal"
    if ":profile/" in text or "/profile/" in text:
        return "profile"
    return "other"


def _finalize_view_audit(view_counts: dict[str, dict[str, int]]) -> dict[str, dict]:
    audit = {}
    for view, counts in sorted(view_counts.items()):
        values = dict(counts)
        total = int(values["total"])
        detection_attempted = total - int(values["decode_failed"])
        detection_succeeded = detection_attempted - int(values["detection_failed"])
        values["detection_attempted"] = detection_attempted
        values["detection_succeeded"] = detection_succeeded
        values["detection_rate"] = (
            detection_succeeded / detection_attempted if detection_attempted else None
        )
        values["embedding_success_fraction"] = (
            int(values["embedding_succeeded"]) / total if total else None
        )
        audit[view] = values
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen InsightFace embeddings")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--provider", action="append", default=None)
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--det-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument(
        "--min-success-fraction",
        type=float,
        default=0.0,
        help="hard-stop if fewer than this fraction of manifest samples succeed",
    )
    parser.add_argument(
        "--expected-identities",
        type=int,
        default=None,
        help="hard-stop unless the successful cache contains exactly this many identities",
    )
    parser.add_argument("--failure-log", default=None)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    manifest = FaceManifest.from_csv(args.manifest).frame
    if not 0.0 <= args.min_success_fraction <= 1.0:
        raise ValueError("--min-success-fraction must be in [0, 1]")

    extractor = InsightFaceExtractor(
        args.model,
        providers=args.provider,
        ctx_id=args.ctx_id,
        det_size=tuple(args.det_size),
        det_thresh=args.det_thresh,
    )
    embeddings = []
    keep_indices = []
    failures = []
    view_counts = defaultdict(
        lambda: {
            "total": 0,
            "embedding_succeeded": 0,
            "detection_failed": 0,
            "decode_failed": 0,
            "other_failed": 0,
        }
    )
    for row_index, row in enumerate(
        tqdm(manifest.itertuples(index=False), total=len(manifest))
    ):
        view = _infer_view_label(row)
        stats = view_counts[view]
        stats["total"] += 1
        try:
            value = extractor.extract([str(row.image_path)])[0]
            embeddings.append(value)
            keep_indices.append(row_index)
            stats["embedding_succeeded"] += 1
        except FaceDetectionError as exc:
            stats["detection_failed"] += 1
            failures.append(
                {
                    "sample_id": str(row.sample_id),
                    "identity_id": str(row.identity_id),
                    "view": view,
                    "failure_type": "no_face_detected",
                    "image_path": str(row.image_path),
                    "error": repr(exc),
                }
            )
        except ImageDecodeError as exc:
            stats["decode_failed"] += 1
            failures.append(
                {
                    "sample_id": str(row.sample_id),
                    "identity_id": str(row.identity_id),
                    "view": view,
                    "failure_type": "image_decode_error",
                    "image_path": str(row.image_path),
                    "error": repr(exc),
                }
            )
        except Exception as exc:
            stats["other_failed"] += 1
            failures.append(
                {
                    "sample_id": str(row.sample_id),
                    "identity_id": str(row.identity_id),
                    "view": view,
                    "failure_type": type(exc).__name__,
                    "image_path": str(row.image_path),
                    "error": repr(exc),
                }
            )

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

    samples_expected = int(len(manifest))
    samples_succeeded = int(len(table.embeddings))
    samples_failed = int(len(failures))
    success_fraction = samples_succeeded / max(1, samples_expected)
    view_audit = _finalize_view_audit(view_counts)
    manifest_identities = int(manifest["identity_id"].astype(str).nunique())
    identities_succeeded = int(len(np.unique(table.identity_ids)))

    validation_errors = []
    if success_fraction < args.min_success_fraction:
        validation_errors.append(
            f"success fraction {success_fraction:.3%} < required "
            f"{args.min_success_fraction:.3%}"
        )
    if (
        args.expected_identities is not None
        and identities_succeeded != args.expected_identities
    ):
        validation_errors.append(
            f"successful identity count {identities_succeeded} != expected "
            f"{args.expected_identities}"
        )

    failure_path = Path(args.failure_log or (str(args.output) + ".failures.tsv"))
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failure_path, "w", encoding="utf-8") as f:
        f.write(
            "sample_id\tidentity_id\tview\tfailure_type\timage_path\terror\n"
        )
        for failure in failures:
            fields = (
                failure["sample_id"],
                failure["identity_id"],
                failure["view"],
                failure["failure_type"],
                failure["image_path"],
                failure["error"],
            )
            f.write("\t".join(str(value).replace("\t", " ") for value in fields) + "\n")

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
            "providers_requested": args.provider,
            "ctx_id": args.ctx_id,
            "pipeline": "native_insightface_faceanalysis",
            "recognition_input_size": extractor.recognition_input_size,
            "det_size": args.det_size,
            "det_thresh": args.det_thresh,
            "image_decode": "OpenCV IMREAD_COLOR, original-resolution BGR",
            "pre_detector_resize": None,
            "pre_detector_normalization": None,
            "alignment": "InsightFace FaceAnalysis internal landmark-based alignment",
            "package_versions": package_versions,
            "model_file_hashes": model_hashes,
            "samples_expected": samples_expected,
            "samples_succeeded": samples_succeeded,
            "samples_failed": samples_failed,
            "success_fraction": success_fraction,
            "manifest_identities": manifest_identities,
            "identities_succeeded": identities_succeeded,
            "min_success_fraction_required": args.min_success_fraction,
            "expected_identities_required": args.expected_identities,
            "validation_status": "FAIL" if validation_errors else "PASS",
            "validation_errors": validation_errors,
            "view_audit": view_audit,
            "embedding_dim": int(table.embeddings.shape[1]),
            "normalization": "L2",
            "detector_selection": "highest det_score face",
        },
        metadata_path,
    )

    if validation_errors:
        raise RuntimeError(
            "embedding cache validation failed; output was not written: "
            + "; ".join(validation_errors)
        )

    table.save(args.output)
    print(
        f"saved {samples_succeeded}/{samples_expected} embeddings "
        f"({success_fraction:.2%}) to {args.output}; "
        f"identities={identities_succeeded}; failures={samples_failed}; "
        f"metadata={metadata_path}"
    )
    for view, stats in view_audit.items():
        detection_rate = stats["detection_rate"]
        detection_text = "n/a" if detection_rate is None else f"{detection_rate:.2%}"
        embedding_fraction = stats["embedding_success_fraction"]
        embedding_text = (
            "n/a" if embedding_fraction is None else f"{embedding_fraction:.2%}"
        )
        print(
            f"audit[{view}]: total={stats['total']}; "
            f"detection_rate={detection_text}; "
            f"embedding_success={embedding_text}; "
            f"no_face={stats['detection_failed']}; "
            f"decode_failed={stats['decode_failed']}; "
            f"other_failed={stats['other_failed']}"
        )


if __name__ == "__main__":
    main()
