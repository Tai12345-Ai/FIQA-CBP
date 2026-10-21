from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np

from .cosine import l2_normalize


@dataclass
class EmbeddingTable:
    embeddings: np.ndarray
    identity_ids: np.ndarray
    sample_ids: np.ndarray
    image_paths: np.ndarray | None = None
    split_labels: np.ndarray | None = None

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.identity_ids = np.asarray(self.identity_ids).astype(str)
        self.sample_ids = np.asarray(self.sample_ids).astype(str)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [N, D]")
        if len(self.embeddings) != len(self.identity_ids) or len(self.embeddings) != len(self.sample_ids):
            raise ValueError("embedding table arrays must have equal length")
        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("embeddings contain NaN or infinity")
        if len(np.unique(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if self.image_paths is not None:
            self.image_paths = np.asarray(self.image_paths).astype(str)
            if len(self.image_paths) != len(self.embeddings):
                raise ValueError("image_paths must align with embeddings")
        if self.split_labels is not None:
            self.split_labels = np.asarray(self.split_labels).astype(str)
            if len(self.split_labels) != len(self.embeddings):
                raise ValueError("split_labels must align with embeddings")

    def normalized(self) -> "EmbeddingTable":
        return EmbeddingTable(
            l2_normalize(self.embeddings).astype(np.float32),
            self.identity_ids.copy(),
            self.sample_ids.copy(),
            None if self.image_paths is None else self.image_paths.copy(),
            None if self.split_labels is None else self.split_labels.copy(),
        )

    def subset(self, indices: np.ndarray) -> "EmbeddingTable":
        idx = np.asarray(indices)
        return EmbeddingTable(
            self.embeddings[idx],
            self.identity_ids[idx],
            self.sample_ids[idx],
            None if self.image_paths is None else self.image_paths[idx],
            None if self.split_labels is None else self.split_labels[idx],
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "embeddings": self.embeddings,
            "identity_ids": self.identity_ids,
            "sample_ids": self.sample_ids,
        }
        if self.image_paths is not None:
            payload["image_paths"] = self.image_paths
        if self.split_labels is not None:
            payload["split_labels"] = self.split_labels
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path: str | Path) -> "EmbeddingTable":
        x = np.load(path, allow_pickle=False)
        return EmbeddingTable(
            x["embeddings"],
            x["identity_ids"],
            x["sample_ids"],
            x["image_paths"] if "image_paths" in x else None,
            x["split_labels"] if "split_labels" in x else None,
        )


class EmbeddingExtractor(Protocol):
    def extract(self, image_paths: list[str]) -> np.ndarray:
        ...


class InsightFaceExtractor:
    """Optional frozen InsightFace/ArcFace extractor.

    This class intentionally does not download a model by itself beyond whatever
    InsightFace's configured model pack does. The research protocol should record
    the pack name, provider list, and model hashes separately.
    """

    def __init__(self, model_name: str = "buffalo_l", providers: list[str] | None = None, ctx_id: int = 0):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("install optional dependency: pip install insightface onnxruntime-gpu") from e
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def extract(self, image_paths: list[str]) -> np.ndarray:
        from PIL import Image
        out = []
        for path in image_paths:
            image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
            faces = self.app.get(image)
            if not faces:
                raise RuntimeError(f"no face detected: {path}")
            face = max(faces, key=lambda f: float(f.det_score))
            out.append(np.asarray(face.embedding, dtype=np.float32))
        return l2_normalize(np.stack(out)).astype(np.float32)
