# wl_transformers

Graph transformer and GINE backbones with canonicalized Laplacian positional
encodings (PE), shared by the Transformer (OGB, Alchemy) and BREC experiments.
It is an installable package; the canonicalization itself is never implemented
here, it is always delegated to the [`prism`](../../prism) package.

```
conda env create -p ~/envs/prism-transformer -f experiments/transformer_ogb/env_transformer.yml
conda activate ~/envs/prism-transformer
```

installs `prism` and `wl_transformers` in editable mode (`pip install -e .[torch]`
and `pip install -e experiments/wl_transformers`).  Weights & Biases logging is
optional (`pip install wandb`, then `wandb_project=<name>`); it is off by default.

## What is in the package

| module | content |
|---|---|
| `wl_transformers.pe_types` | the `pe_type` registry and `make_canonicalizer` |
| `wl_transformers.data` | `Transform`: lowest Laplacian eigenpairs, canonicalization, token index |
| `wl_transformers.models` | eigenvector embedding (DeepSets), pair tokenizer, `Transformer`, `GNN` |
| `wl_transformers.training` | AdamW with decay groups, cosine schedule with warm-up, optional W&B logger |
| `wl_transformers/conf/model/graph_transformer.yaml` | architecture hyperparameters (Hydra config group `model`) |

### Canonicalization through `prism`

`Transform` computes the lowest `num_vecs + 3` eigenpairs of the normalized
Laplacian (a dense `eigh` when at least `n - 1` eigenpairs are requested, i.e. for graphs
with at most 100 nodes, `scipy` Lanczos otherwise), stores them in float32 with NaN padding, canonicalizes them and drops
the 3 buffer eigenpairs again.  The buffer lets an eigenspace that straddles the
`num_vecs` cutoff be resolved whole.

| `pe_type` | canonicalizer | rank information | eigenvalue perturbation | rho |
|---|---|---|---|---|
| `LPE` | none (random sign flips in training) | none | yes | full |
| `CanonLPE_min_rope` | `PRiSM.from_preset("transformer_refined")` | canonical rank, RoPE | no | min |
| `CanonLPE_min_sinid` | `PRiSM.from_preset("transformer_refined")` | canonical rank, sinusoidal | no | min |
| `CanonLPE_min` | `PRiSM.from_preset("transformer_refined")` | none | no | min |
| `CanonLPE`, `CanonLPE_noeps` | `PRiSM.from_preset("brec")` | none | yes / no | full |
| `QRRefinedLPE` | `PRiSM.from_preset("brec_qr_refined")` | none | yes | full |
| `IndRefineNauty` | `PRiSM.from_preset("brec_indrefine")` (needs `pynauty`) | none | no | full |
| `MAP`, `MAP_min` | `prism.baselines.MAP.from_profile("brec")` | none | no | full / min |
| `OAP`, `OAP_min` | `prism.baselines.OAP.from_profile("brec")` | none | no | full / min |

With `refine_ordering=false` the PRiSM types use the preset `brec` (hybrid
extension without ordering refinement); with `refine_ordering=true` they use
`transformer_refined`.  The config key `prism_preset` overrides the choice.
`prism.adapters.transformer.canonicalize_eigenpairs` returns the canonical
eigenvectors together with the canonical vertex order and rank; the rank feeds
the RoPE or sinusoidal embedding.  MAP and OAP return eigenvectors only.

### Hydra

The entry points live in `experiments/transformer_ogb` and `experiments/brec`.
They find the shared model config through
`hydra.searchpath: [pkg://wl_transformers.conf]`, so they can be started from any
directory.

## Numerical notes

* **Partial eigendecomposition.**  For graphs with more than 100 nodes (more than
  97 for `LPE`, which has no buffer) the eigenpairs come from `scipy.sparse.linalg.eigsh` (shift-invert Lanczos) whose
  start vector is random.  Eigenvectors then differ between calls at the 1e-7
  level, and inside degenerate eigenspaces the basis is arbitrary before it is
  canonicalized.  Below that size the pipeline is deterministic.
  The OGB entry point applies `Transform` on the fly, so a graph is
  re-transformed each epoch.
* **Mixed precision.**  Training uses float16 autocast on CUDA, as in all
  reported runs.  On CPU float16 autocast does not exist and bfloat16 is used;
  this is meant for smoke tests only.
* **Data loader workers.**  `num_workers=0` is the default; a positive value
  changes the random-number stream of the shuffling.

## Tests

```
python -m pytest experiments/wl_transformers/tests experiments/transformer_ogb/tests experiments/brec/tests
```

run on CPU in a few seconds inside the environment above.  They cover the `pe_type`
registry, `Transform` (shapes, NaN padding, permutation equivariance of the
canonical PE, buffer handling), forward passes of both backbones, the parameter
counts of the matched BREC architecture, and that every shipped experiment config
composes.

## Origin and license

The model code, the pair tokenizer and the eigenvector embedding are adapted from
*Aligning Transformers with Weisfeiler-Leman* (Müller and Morris, ICML 2024),
<https://github.com/luis-mueller/wl-transformers>, MIT License, Copyright (c) 2024
Luis Müller (`LICENSE` in this directory).  The optimizer and schedule follow
nanoGPT (MIT, Andrej Karpathy).  See `THIRD_PARTY_NOTICES` at the repository root.
Removed from the upstream code: pre-training, checkpoints, SPE, the order-3
tokenizer and Performer attention, none of which the experiments use.
