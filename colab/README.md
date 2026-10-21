# Colab experiments

## Primary notebook

`CBP_CFP_Core_Feasibility_Pilot.ipynb`

This is the current notebook for the first real-face CBP feasibility pilot.

It is specialized for the raw CFP archive downloaded from Kaggle and expects the archive on Google Drive, for example:

```text
MyDrive/
├── CBP/
│   └── CBP_full_extract/
└── datasets/
    └── CFP/
        └── cfp-dataset.zip
```

The notebook mounts Drive, copies the project to local Colab storage, extracts the CFP ZIP to `/content/CFP`, auto-detects `Data/Images`, builds the identity manifest, extracts frozen InsightFace embeddings, and runs the focused CBP pilot.

### Scientific scope

Primary questions:

1. Gate A / E1: held-out mean-matched evidence/influence phenomenon.
2. Gate B / E2: decision relevance against matched mean-only compression.
3. E3: inductive router fidelity on unseen identities.

This pilot is a mechanism study using CFP identities and captures. It is not the official CFP-FP pair-protocol benchmark score.

### Frozen FR model

Default:
- InsightFace `buffalo_l`;
- frozen recognition backbone;
- no FR fine-tuning in the first feasibility pilot.

### Drive cache

The notebook writes the embedding cache and run outputs back to Drive, so later Colab sessions can reuse the extracted embeddings without processing all CFP images again.
