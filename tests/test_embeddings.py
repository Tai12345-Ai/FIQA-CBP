import numpy as np
import pytest

from cbp.fr.embeddings import FaceDetectionError, InsightFaceExtractor
from scripts.extract_embeddings import _finalize_view_audit, _infer_view_label


class _FakeFace:
    def __init__(self, det_score, embedding):
        self.det_score = det_score
        self.embedding = np.asarray(embedding, dtype=np.float32)


class _FakeApp:
    def __init__(self, faces):
        self.faces = faces
        self.images = []

    def get(self, image):
        self.images.append(np.asarray(image).copy())
        return list(self.faces)


def _extractor_with_faces(faces, padding_ratio=0.0):
    extractor = InsightFaceExtractor.__new__(InsightFaceExtractor)
    extractor.app = _FakeApp(faces)
    extractor.padding_ratio = float(padding_ratio)
    extractor._read_bgr = lambda path: np.zeros((240, 320, 3), dtype=np.uint8)
    return extractor


def test_native_pipeline_passes_original_resolution_to_faceanalysis():
    extractor = _extractor_with_faces([_FakeFace(0.9, [3.0, 4.0])])

    features = extractor.extract(["face.jpg"])

    assert extractor.app.images[0].shape == (240, 320, 3)
    assert features.shape == (1, 2)
    assert np.allclose(features[0], [0.6, 0.8])


def test_native_pipeline_applies_replicate_padding_before_detection():
    extractor = _extractor_with_faces(
        [_FakeFace(0.9, [3.0, 4.0])],
        padding_ratio=0.25,
    )

    extractor.extract(["face.jpg"])

    # 25% is added independently on every side:
    # H = 240 + 2*60 = 360, W = 320 + 2*80 = 480.
    assert extractor.app.images[0].shape == (360, 480, 3)


def test_native_pipeline_selects_highest_detection_score():
    extractor = _extractor_with_faces(
        [
            _FakeFace(0.2, [1.0, 0.0]),
            _FakeFace(0.9, [0.0, 2.0]),
        ]
    )

    features = extractor.extract(["face.jpg"])

    assert np.allclose(features[0], [0.0, 1.0])


def test_native_pipeline_raises_explicit_detection_error():
    extractor = _extractor_with_faces([])

    with pytest.raises(FaceDetectionError, match="no face detected"):
        extractor.extract(["profile.jpg"])


class _Row:
    def __init__(self, sample_id, image_path, view=None):
        self.sample_id = sample_id
        self.image_path = image_path
        self.view = view


def test_cfp_view_inference_from_sample_id_and_path():
    assert _infer_view_label(_Row("001:frontal/01.jpg", "x.jpg")) == "frontal"
    assert _infer_view_label(_Row("001:profile/01.jpg", "x.jpg")) == "profile"
    assert _infer_view_label(_Row("001:01.jpg", r"C:\cfp\001\profile\01.jpg")) == "profile"


def test_view_audit_reports_detection_and_embedding_rates():
    counts = {
        "frontal": {
            "total": 10,
            "embedding_succeeded": 8,
            "detection_failed": 1,
            "decode_failed": 1,
            "other_failed": 0,
        }
    }

    audit = _finalize_view_audit(counts)["frontal"]

    assert audit["detection_attempted"] == 9
    assert audit["detection_succeeded"] == 8
    assert audit["detection_rate"] == pytest.approx(8 / 9)
    assert audit["embedding_success_fraction"] == pytest.approx(0.8)
