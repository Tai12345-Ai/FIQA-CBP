from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cbp.fr.cosine import l2_normalize
from cbp.fr.embeddings import EmbeddingTable


@dataclass
class SyntheticFaceEmbeddings:
    table: EmbeddingTable
    latent_quality: np.ndarray
    degradation_type: np.ndarray


def make_synthetic_face_embeddings(
    n_identities: int = 300,
    captures_per_identity: int = 5,
    embedding_dim: int = 64,
    seed: int = 0,
) -> SyntheticFaceEmbeddings:
    """Create a controllable face-like embedding dataset for smoke experiments.

    Quality controls identity-preserving noise. A second degradation variable
    introduces an anisotropic shift shared by unrelated identities, which can
    create mean-matched but interaction-different states. This is a mechanism
    test, not a substitute for real face data.
    """
    if n_identities < 10 or captures_per_identity < 2:
        raise ValueError("need >=10 identities and >=2 captures per identity")
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.normal(size=(n_identities, embedding_dim)))
    nuisance_a = l2_normalize(rng.normal(size=(embedding_dim,)))
    nuisance_b = l2_normalize(rng.normal(size=(embedding_dim,)))

    embeddings = []
    identity_ids = []
    sample_ids = []
    quality = []
    degradation = []
    for identity in range(n_identities):
        for capture in range(captures_per_identity):
            q = float(rng.beta(2.0, 2.0))
            kind = int(rng.integers(0, 3))
            noise_scale = 0.04 + 0.32 * (1.0 - q)
            vector = centers[identity] + noise_scale * rng.normal(size=embedding_dim)
            if kind == 1:
                vector += (0.02 + 0.24 * (1.0 - q)) * nuisance_a
            elif kind == 2:
                vector -= (0.02 + 0.24 * (1.0 - q)) * nuisance_b
            embeddings.append(vector)
            identity_ids.append(f"id_{identity:05d}")
            sample_ids.append(f"id_{identity:05d}_c{capture:02d}")
            quality.append(q)
            degradation.append(kind)

    table = EmbeddingTable(
        embeddings=l2_normalize(np.asarray(embeddings)).astype(np.float32),
        identity_ids=np.asarray(identity_ids),
        sample_ids=np.asarray(sample_ids),
    )
    return SyntheticFaceEmbeddings(
        table=table,
        latent_quality=np.asarray(quality, dtype=np.float32),
        degradation_type=np.asarray(degradation, dtype=np.int64),
    )
