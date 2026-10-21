# Colab experiments

## Primary notebook

`CBP_CFP_Core_Feasibility_Pilot.ipynb`

This is the current notebook for the first real-face CBP feasibility pilot.

## Recommended storage/workflow

Use each service for what it is best at:

- **GitHub**: source code, configs, tests, docs, notebooks.
- **Google Drive**: raw datasets, embedding caches, experiment outputs.
- **Colab local disk**: cloned source and extracted image files for fast runtime I/O.

Recommended Drive layout:

```text
MyDrive/
├── datasets/
│   └── CFP/
│       └── cfp-dataset.zip
└── CBP/
    ├── cache/
    │   └── cfp_buffalo_l_embeddings.npz
    └── runs/
```

The notebook clones the public repository

```text
https://github.com/Tai12345-Ai/FIQA-CBP.git
```

to `/content/FIQA-CBP`, mounts Drive, extracts the CFP ZIP to `/content/CFP`,
builds the identity manifest, extracts frozen InsightFace embeddings, and runs
the focused CBP pilot.

No GitHub token or Colab Secret is required while the repository is public.

## Reproducibility

During development, `GITHUB_REF = "main"` is convenient.

For paper-quality or archival runs, create a release tag and set
`GITHUB_REF` to that immutable tag so the exact source version is recoverable.
The notebook also prints the resolved commit SHA at runtime.

## Scientific scope

Primary questions:

1. Gate A / E1: held-out mean-matched evidence/influence phenomenon.
2. Gate B / E2: decision relevance against matched mean-only compression.
3. E3: inductive router fidelity on unseen identities.

This pilot is a mechanism study using CFP identities and captures. It is not
the official CFP-FP pair-protocol benchmark score.

## Frozen FR model

Default:

- InsightFace `buffalo_l`;
- frozen recognition backbone;
- no FR fine-tuning in the first feasibility pilot.

## Persistent artifacts

Embedding caches and experiment results are written back to Drive so a Colab
runtime reset does not force another full embedding extraction.
