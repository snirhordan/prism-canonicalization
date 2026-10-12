# Transformer backbone on OGB molecules and Alchemy

A graph transformer with a canonicalized Laplacian positional encoding on
**MolHIV**, **MolClinTox**, **MolBBBP** (OGB) and **Alchemy** (10K subset, 12
regression targets).  Eigenvectors are canonicalized by PRiSM, MAP or OAP from the
`prism` package, or randomly sign flipped (LPE).  The shared model code is in
[`../wl_transformers`](../wl_transformers).

## Setup

From the repository root:

```
conda env create -p ~/envs/prism-transformer -f experiments/transformer_ogb/env_transformer.yml
conda activate ~/envs/prism-transformer
cd experiments/transformer_ogb
```

All commands below are run from `experiments/transformer_ogb`.  Datasets are
downloaded on first use into `./data` (`root=<dir>` changes the location).  OGB
datasets come from `ogb.graphproppred`, Alchemy from the `alchemy_full` entry of
`torch_geometric.datasets.TUDataset`; no data is shipped.  The Alchemy subset is
given by `fine-tuning/{train,val,test}_al_10.index` (10000 / 1000 / 1000 indices
into `alchemy_full`).  Hydra writes run directories to `./outputs`.  A GPU is
required for the full runs, which take several hours each.

## Commands, one per row

An experiment is a named config, `+experiment=<name>`; the seed is the only other
argument.  The reported value of a run is the test score at the best validation
epoch, printed on the last line (`Final | best_val_score=... test_score=...`).

### OGB (`fine-tuning/ogbg.py`, ROC-AUC)

| row | command | seeds |
|---|---|---|
| MolHIV, PRiSM | `python fine-tuning/ogbg.py +experiment=molhiv_prism seed=$s` | 0 1 2 |
| MolHIV, LPE | `python fine-tuning/ogbg.py +experiment=molhiv_lpe seed=$s` | 0 1 2 |
| MolHIV, MAP | `python fine-tuning/ogbg.py +experiment=molhiv_map seed=$s` | 0 1 3 |
| MolHIV, OAP | `python fine-tuning/ogbg.py +experiment=molhiv_oap seed=$s` | 0 1 2 |
| MolClinTox, PRiSM | `python fine-tuning/ogbg.py +experiment=molclintox_prism seed=$s` | 0 1 2 |
| MolClinTox, PRiSM, refined rank (alternative) | `python fine-tuning/ogbg.py +experiment=molclintox_prism_fixed seed=$s` | 0 1 2 3 4 5 |
| MolClinTox, LPE | `python fine-tuning/ogbg.py +experiment=molclintox_lpe seed=$s` | 0 1 2 3 4 5 |
| MolClinTox, MAP | `python fine-tuning/ogbg.py +experiment=molclintox_map seed=$s` | 0 1 2 |
| MolClinTox, OAP | `python fine-tuning/ogbg.py +experiment=molclintox_oap seed=$s` | 0 1 2 3 4 5 |
| MolBBBP, PRiSM | `python fine-tuning/ogbg.py +experiment=molbbbp_prism seed=$s` | 0 1 2 |
| MolBBBP, LPE | `python fine-tuning/ogbg.py +experiment=molbbbp_lpe seed=$s` | 0 1 2 |
| MolBBBP, MAP | `python fine-tuning/ogbg.py +experiment=molbbbp_map seed=$s` | 0 1 2 |
| MolBBBP, OAP | `python fine-tuning/ogbg.py +experiment=molbbbp_oap seed=$s` | 0 1 2 |

### Alchemy (`fine-tuning/alchemy.py`, test MAE)

| row | command | seeds |
|---|---|---|
| Alchemy, PRiSM | `python fine-tuning/alchemy.py +experiment=alchemy_prism seed=$s` | 0 1 2 |
| Alchemy, LPE | `python fine-tuning/alchemy.py +experiment=alchemy_lpe seed=$s` | 0 1 3 |
| Alchemy, MAP | `python fine-tuning/alchemy.py +experiment=alchemy_map seed=$s` | 0 1 3 |
| Alchemy, OAP | `python fine-tuning/alchemy.py +experiment=alchemy_oap seed=$s` | 0 1 3 |

For a seed list use a shell loop, for example

```
for s in 0 1 2; do python fine-tuning/ogbg.py +experiment=molhiv_prism seed=$s; done
```

### What the experiment configs set

| config | `pe_type` | `refine_ordering` | rank information |
|---|---|---|---|
| `molhiv_prism`, `alchemy_prism` | `CanonLPE_min_rope` | true | RoPE on the canonical vertex rank |
| `molclintox_prism` | `CanonLPE_min_sinid` | true | sinusoidal embedding of the rank; preset `transformer_refined_prefix`: the rank of the norm-signature order, as in the original runs |
| `molclintox_prism_fixed` | `CanonLPE_min_sinid` | true | sinusoidal embedding of the rank; preset `transformer_refined`: the rank of the order refined by the canonical columns |
| `molbbbp_prism` | `CanonLPE_min` | true | none (sequential refinement only) |
| `*_lpe` | `LPE` | false | none |
| `*_map` | `MAP_min` | false | none |
| `*_oap` | `OAP_min` | false | none |

Hyperparameters that are common to all rows are in `fine-tuning/configs/{ogbg,alchemy}.yaml`
and the shared `wl_transformers/conf/model/graph_transformer.yaml`:

| | OGB | Alchemy |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 | same |
| learning rate | 1e-4 | 1e-3 (default); 1e-4 for `alchemy_prism` and `alchemy_lpe` |
| schedule | linear warm-up (2.5% of the epochs) then cosine decay to 0 | same |
| epochs, batch size | 100, 32 | 2000, 64 |
| gradient clipping | 5.0 | 5.0 |
| architecture | 12 layers, 16 heads, hidden size 768, dropout 0.1, stochastic depth up to 0.1 | same |
| positional encoding | 96 normalized-Laplacian eigenvectors (+ 3 buffer eigenpairs for canonicalization) | same |
| precision | float16 autocast, gradient scaling | same |

Any value can be overridden on the command line, for instance `lr=1e-4` for the
MAP or OAP Alchemy rows.

## Smoke test and shorter runs

```
python fine-tuning/ogbg.py +experiment=molclintox_prism smoke=true
python fine-tuning/alchemy.py +experiment=alchemy_prism smoke=true
```

`smoke=true` restricts every split to its first `smoke_graphs` graphs (64 for OGB,
128 for Alchemy) and trains for one epoch; it runs on CPU in minutes, after the
dataset download (Alchemy downloads the full 200K-graph dataset).  `num_epochs=<n>`
shortens a full run without changing the data.

## Tests

```
python -m pytest tests
```

checks that every experiment config composes and selects a supported canonicalizer, and which
canonical rank the two MolClinTox PRiSM configs use.
The library tests are in `../wl_transformers/tests`.

## Origin

Adapted from <https://github.com/luis-mueller/wl-transformers> (MIT License,
Copyright (c) 2024 Luis Müller); see `../wl_transformers/LICENSE` and the
repository's `THIRD_PARTY_NOTICES`.
