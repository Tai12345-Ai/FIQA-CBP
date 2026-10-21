from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {"image_path", "identity_id"}


@dataclass
class FaceManifest:
    frame: pd.DataFrame

    @staticmethod
    def from_csv(path: str | Path) -> "FaceManifest":
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        if "sample_id" not in df.columns:
            df["sample_id"] = [f"s{i:08d}" for i in range(len(df))]
        df["identity_id"] = df["identity_id"].astype(str)
        df["sample_id"] = df["sample_id"].astype(str)
        if df["sample_id"].duplicated().any():
            duplicates = df.loc[df["sample_id"].duplicated(keep=False), "sample_id"].unique()
            preview = duplicates[:5].tolist()
            raise ValueError(f"manifest sample_id values must be unique; duplicates include: {preview}")
        return FaceManifest(df.reset_index(drop=True))

    @staticmethod
    def from_identity_folders(root: str | Path, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> "FaceManifest":
        root = Path(root)
        rows = []
        for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(identity_dir.rglob("*")):
                if path.suffix.lower() in extensions:
                    relative_path = path.relative_to(identity_dir).as_posix()
                    rows.append({
                        "image_path": str(path),
                        "identity_id": identity_dir.name,
                        "sample_id": f"{identity_dir.name}:{relative_path}",
                    })
        if not rows:
            raise ValueError(f"no images found under {root}")
        frame = pd.DataFrame(rows)
        if frame["sample_id"].duplicated().any():
            raise ValueError("generated sample_id values are not unique")
        return FaceManifest(frame)

    def to_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)

    def subset(self, indices: np.ndarray | list[int]) -> "FaceManifest":
        return FaceManifest(self.frame.iloc[np.asarray(indices)].reset_index(drop=True))
