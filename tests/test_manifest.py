from pathlib import Path

from cbp.data.manifest import FaceManifest


def test_nested_identity_folders_produce_unique_sample_ids(tmp_path: Path):
    identity = tmp_path / "001"
    frontal = identity / "frontal"
    profile = identity / "profile"
    frontal.mkdir(parents=True)
    profile.mkdir(parents=True)

    (frontal / "01.jpg").write_bytes(b"front")
    (profile / "01.jpg").write_bytes(b"profile")

    manifest = FaceManifest.from_identity_folders(tmp_path)

    assert len(manifest.frame) == 2
    assert manifest.frame["sample_id"].is_unique
    assert set(manifest.frame["sample_id"]) == {
        "001:frontal/01.jpg",
        "001:profile/01.jpg",
    }
