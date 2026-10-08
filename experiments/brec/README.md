# BREC expressivity benchmark

How many of the BREC graph pairs can a small model distinguish when its input is
a canonicalized Laplacian positional encoding?  For every pair a freshly
initialised model is trained with a cosine embedding loss to separate the two
graphs under 32 random relabellings, and the pair counts as distinguished when
the Hotelling T^2 test of BREC passes its reliability check.  The categories are
Basic (60 pairs), Regular (100), Extension (100) and CFI (100), 360 pairs in all;
`parts=[...]` selects other categories, including `4-Vertex_Condition` and
`Distance_Regular` (20 pairs each).

The model is the GINE backbone of [`../wl_transformers`](../wl_transformers) whose
only positional information is a DeepSets embedding of the first 8 Laplacian
eigenvectors, `rho(sum_i phi(v_i, lambda_i))`.  The architecture is the same for
all methods (embedding size 32, 2 GINE layers, 12,656 trainable parameters, 12,664
with the learnable eigenvalue perturbation of `CanonLPE`/`QRRefinedLPE`).

## Setup

From the repository root:

```
conda env create -f experiments/transformer_ogb/env_transformer.yml   # same environment as the Transformer experiments
conda activate prism-transformer
cd experiments/brec
```

The environment contains the `brec` package; the BREC graphs are downloaded into
`./data` on first use and converted once (several minutes).

## Commands, one per row

| row | command | canonicalizer |
|---|---|---|
| PRiSM | `python expressivity/brec_benchmark.py +experiment=brec_prism` | `PRiSM.from_preset("brec")` |
| PRiSM, QR for every multiplicity | `python expressivity/brec_benchmark.py +experiment=brec_prism_qr` | `PRiSM.from_preset("brec_qr_refined")` |
| MAP | `python expressivity/brec_benchmark.py +experiment=brec_map` | `MAP.from_profile("brec")` |
| OAP | `python expressivity/brec_benchmark.py +experiment=brec_oap` | `OAP.from_profile("brec")` |

All four use `experiment/brec_matched.yaml` for the architecture.  The reported
quantity is the number of distinguished pairs per category and in total, printed
at the end:

```
Per part: {'Basic': ..., 'Regular': ..., 'Extension': ..., 'CFI': ...}
Correct in <total> / 360
```

A full run takes hours because every pair trains its own model;
the pairs of a category are independent, so `'parts=[Regular]'` (quoted for the shell) and
`max_pairs_per_part=<n>` can be used to split or shorten a run.

Other settings are plain Hydra overrides: `backbone=transformer embed_dim=... num_heads=...`
for the graph transformer instead of GINE, `pe_type=<name>` for any type of
`wl_transformers.pe_types` (for example `pe_type=LPE` for random-sign LapPE),
`num_vecs`, `embed_dim`, `num_layers`, `epochs`.

## Smoke test

```
python expressivity/brec_benchmark.py +experiment=brec_prism smoke=true
```

evaluates the first two Basic pairs for two epochs (after the one-time dataset
processing).

## Tests

```
python -m pytest tests
```

checks the four experiment configs (architecture, parameter count, canonicalizer) without
needing the dataset.

## Fixed protocol constants

Learning rate 1e-4, weight decay 1e-4, Adam with `ReduceLROnPlateau`, at most 20
epochs per pair (early stop when the training loss drops below 0.2), batch size
16, 32 relabellings per pair, T^2 threshold 72.34.  No seed is set beyond the
`torch_geometric.seed_everything(2022)` call of the `brec` package; `seed=<n>`
adds one.

## Origin

The evaluation loop follows the BREC benchmark, <https://github.com/GraphPKU/BREC>
(MIT License, Copyright (c) 2023 Yanbo Wang), and the BREC example of
<https://github.com/luis-mueller/wl-transformers> (MIT License, Copyright (c) 2024
Luis Müller).  See the repository's `THIRD_PARTY_NOTICES`.
