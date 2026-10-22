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


class ImageDecodeError(RuntimeError):
    """Raised when an image cannot be decoded before InsightFace inference."""


class FaceDetectionError(RuntimeError):
    """Raised when native InsightFace FaceAnalysis detects no face."""


class InsightFaceExtractor:
    """Frozen native InsightFace detector -> alignment -> recognition pipeline.

    Images are decoded with OpenCV in BGR at their original resolution and are
    passed directly to FaceAnalysis.get. No project-side resize, normalization,
    RGB conversion, or custom alignment occurs before detection. InsightFace
    therefore owns detection, landmark-based face alignment, and recognition
    preprocessing end-to-end.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "install optional dependency: pip install insightface onnxruntime-gpu"
            ) from e

        self.model_name = str(model_name)
        self.ctx_id = int(ctx_id)
        self.det_size = tuple(int(v) for v in det_size)
        self.det_thresh = float(det_thresh)

        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        self.recognizer = self.app.models.get("recognition")
        if self.recognizer is None:
            raise RuntimeError(
                f"InsightFace model pack {self.model_name!r} has no recognition model"
            )

        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=self.det_size,
            det_thresh=self.det_thresh,
        )

        input_size = getattr(self.recognizer, "input_size", None)
        if input_size is None or len(input_size) != 2:
            raise RuntimeError("could not determine InsightFace recognition input size")
        self.recognition_input_size = tuple(int(v) for v in input_size)

    @staticmethod
    def _read_bgr(path: str) -> np.ndarray:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python is required for InsightFace extraction") from e

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError(f"could not decode image: {path}")
        return image

    def extract(self, image_paths: list[str]) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        out = []
        for path in image_paths:
            image = self._read_bgr(path)
            faces = self.app.get(image)
            if not faces:
                raise FaceDetectionError(f"no face detected: {path}")
            face = max(faces, key=lambda f: float(f.det_score))
            embedding = np.asarray(face.embedding, dtype=np.float32)
            if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
                raise RuntimeError(f"invalid InsightFace embedding: {path}")
            out.append(embedding)
        return l2_normalize(np.stack(out)).astype(np.float32)
