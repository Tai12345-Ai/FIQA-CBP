import os
from pathlib import Path
import pytest
import yaml

from cbp.experiment import run_experiment
from cbp.synthetic.faces import make_synthetic_face_embeddings


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("CBP_RUN_INTEGRATION") != "1", reason="set CBP_RUN_INTEGRATION=1")
def test_full_synthetic_smoke(tmp_path: Path):
    data = make_synthetic_face_embeddings(80, 3, 16, seed=13)
    embedding_path = tmp_path / "embeddings.npz"
    data.table.save(embedding_path)
    output_dir = tmp_path / "run"
    config = {
        "seed": 13,
        "embedding_npz": str(embedding_path),
        "output_dir": str(output_dir),
        "n_states": 12,
        "k_candidates": [3, 4],
        "spectral_dim": 3,
        "router_epochs": 2,
        "router_batch_size": 32,
        "router_hidden": [16],
        "router_patience": 2,
        "phenomenon_permutations": 10,
        "run_continuous_baseline": False,
        "run_image_neural_baseline": False,
        "certification_delta": 0.2,
        "contexts": [
            {
                "name": "smoke",
                "target_fmr": 0.2,
                "coverage_low": 0.1,
                "coverage_high": 1.0,
                "rho": 0.2,
                "identity_budget": 10,
                "max_impostor_pairs_for_threshold": 10000,
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    report = run_experiment(config_path)
    assert (output_dir / "report.json").exists()
    assert (output_dir / "artifact" / "router.pt").exists()
    assert report["selection"]["selected"].startswith("cbp_evidence")
