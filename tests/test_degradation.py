from pathlib import Path

import numpy as np
from PIL import Image

from cbp.data.degradation import apply_degradation, build_degraded_manifest
from cbp.data.manifest import FaceManifest


def test_apply_degradation_preserves_canvas_size():
    image = Image.fromarray(
        np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (48, 1, 3))
    )

    for condition in ("lowres_32", "blur_3", "jpeg_20"):
        degraded = apply_degradation(image, condition)
        assert degraded.size == image.size
        assert degraded.mode == "RGB"


def test_build_degraded_manifest_preserves_identity_and_sample_ids(tmp_path: Path):
    source = tmp_path / "source" / "001" / "frontal"
    source.mkdir(parents=True)
    image_path = source / "01.jpg"
    Image.new("RGB", (64, 64), (120, 100, 80)).save(image_path)

    clean = FaceManifest.from_identity_folders(tmp_path / "source")
    degraded = build_degraded_manifest(
        clean,
        output_root=tmp_path / "degraded",
        condition="lowres_32",
    )

    assert degraded.frame["identity_id"].tolist() == clean.frame["identity_id"].tolist()
    assert degraded.frame["sample_id"].tolist() == clean.frame["sample_id"].tolist()
    assert Path(degraded.frame.iloc[0]["image_path"]).exists()
