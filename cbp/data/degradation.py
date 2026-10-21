from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from .manifest import FaceManifest


RUN2_CONDITIONS = ("lowres_32", "blur_3")


def _resampling_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC


def apply_degradation(image: Image.Image, condition: str) -> Image.Image:
    """Apply one deterministic acquisition stress used by the CFP Run-2 pilot.

    The presets are intentionally fixed so the first mechanism diagnostic is
    not tuned after inspecting Gate-A outcomes.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    if condition == "lowres_32":
        width, height = image.size
        short = min(width, height)
        if short <= 0:
            raise ValueError("invalid image size")
        scale = 32.0 / float(short)
        down_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resample = _resampling_bicubic()
        return image.resize(down_size, resample=resample).resize(
            (width, height),
            resample=resample,
        )

    if condition == "blur_3":
        return image.filter(ImageFilter.GaussianBlur(radius=3.0))

    if condition == "jpeg_20":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=20, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    raise ValueError(
        f"unknown degradation condition: {condition}; "
        "expected one of {'lowres_32', 'blur_3', 'jpeg_20'}"
    )


def build_degraded_manifest(
    manifest: FaceManifest,
    output_root: str | Path,
    condition: str,
) -> FaceManifest:
    """Materialize a deterministic degraded CFP copy while preserving sample IDs."""
    output_root = Path(output_root)
    rows = []

    for row in manifest.frame.itertuples(index=False):
        source = Path(str(row.image_path))
        identity_id = str(row.identity_id)
        sample_id = str(row.sample_id)

        # CFP sample IDs are generated as '<identity>:<relative/path>'.
        if ":" in sample_id:
            _, relative = sample_id.split(":", 1)
            relative_path = Path(relative)
        else:
            relative_path = Path(source.name)

        if condition == "jpeg_20":
            relative_path = relative_path.with_suffix(".jpg")
        else:
            relative_path = relative_path.with_suffix(".png")

        destination = output_root / identity_id / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source) as image:
            degraded = apply_degradation(image, condition)
            if condition == "jpeg_20":
                degraded.save(destination, format="JPEG", quality=20, optimize=False)
            else:
                degraded.save(destination, format="PNG", optimize=False)

        rows.append(
            {
                "image_path": str(destination),
                "identity_id": identity_id,
                "sample_id": sample_id,
                "source_image_path": str(source),
                "degradation": condition,
            }
        )

    return FaceManifest(pd.DataFrame(rows))
