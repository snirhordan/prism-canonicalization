# prism-canonicalization

Reference implementation of **PRiSM** (Partition, Refine, Solve, Match), the
eigenvector canonicalization of

> Snir Hordan, Nadav Dym, Tim Seppelt. *Weisfeiler Leman is Incomplete On Graphs
> With A Simple Spectrum.* NeurIPS 2026.

All experiments of the paper use **one class**, `prism.PRiSM`. A run selects its
variant through a named preset (`prism/presets.py`), a dictionary of constructor
arguments. The numpy core has no dependency beyond numpy and scipy; PyTorch is
needed only for the MAP and OAP baselines and for the DGL adapter.

```
pip install -e .            # numpy, scipy
pip install -e ".[torch]"   # + torch, for prism.baselines.MAP / OAP
```

## Quick start

```python
import numpy as np
from prism import PRiSM

A = np.array(...)                         # dense (or scipy sparse) adjacency
canon = PRiSM.from_preset("zinc_pna")
pe = canon.pos_enc(A, k=8)                # (n, 8) float32, GNN positional encoding

lam, V = canon.eigendecompose(A)          # eigenpairs of the preset's operator
Vc = canon.canonicalize(V, lam)           # canonical eigenvectors
info = canon.canonicalize(V, lam, return_info=True)   # + order, rank, norm_sig
```

## The `PRiSM` class

| group | argument | values |
|---|---|---|
| spectrum | `operator` | `norm_adj` (D^-1/2 A D^-1/2), `norm_lap` (I - D^-1/2 A D^-1/2), `comb_lap` (D - A) |
| | `drop_trivial`, `k` | drop the trivial eigenpair; number of columns kept |
| | `scope` | `full` canonicalizes all eigenpairs then keeps `k`; `window` truncates to `k + window_buffer` first |
| | `keep` | `last` (largest eigenvalues, `norm_adj`) or `first` (smallest, Laplacians) |
| canonicalization | `sign_rule` | `first_pivot` (as run in all table cells; FastSign when the signature is injective) or `gf2` (Refine + Solve of Algorithm 1) |
| | `mult_mode` | `hybrid` (Householder/Givens for m<=3, QR for m>=4) or `qr` (QR for every m, unique-signature pivots) |
| | `refine_ordering` | sequential refinement of the vertex order (required by `qr` and `gf2`) |
| numerics | `tol_profile` | `adaptive` (relative snapping, eps floor) or `legacy_round` (fixed decimals) |
| | `quantize_float32` | pass eigenpairs through float32 first (transformer / BREC storage) |
| post-processing | `weight`, `weight_param` | `eigval`, `inv_sqrt_lap` (eps), `heat` (t) |

Methods: `canonicalize`, `eigendecompose`, `spectrum`, `pos_enc`, `from_preset`,
`get_config`, and the stand-alone procedures `canonicalize_simple` (Algorithm 1)
and `fastsign` (Algorithm 2). `prism.gf2` exposes the four stages.

Baselines (`prism.baselines`): `MAP`, `OAP` (adapted from the reference code of
Ma et al.; see the file headers), `lap_pe` and `random_sign_flip`
(LapPE / SignNet input). Model-side SignNet stays with the experiments.

## Presets

Each preset reproduces the code path of its harness; `tests/test_golden.py`
checks this against outputs of the original implementations.

| preset | variant in the harness | paper cells |
|---|---|---|
| `zinc_gatedgcn` | `spectral_m3` | `tab:molecular-results`, ZINC / GatedGCN |
| `zinc_pna` | `spectral_qr_refined_eigval` | `tab:molecular-results`, ZINC / PNA |
| `moltox21_gatedgcn` | `spectral_m3_inv_sqrt_lap` | `tab:molecular-results`, MolTox21 / GatedGCN |
| `moltox21_pna` | `spectral_m3_eigval` | `tab:molecular-results`, MolTox21 / PNA |
| `molpcba_gatedgcn` | `spectral_qr_refined_eigval` | `tab:molecular-results`, MolPCBA / GatedGCN |
| `molpcba_pna` | `spectral_qr_refined_eigval` | `tab:molecular-results`, MolPCBA / PNA |
| `transformer_refined` | `CanonLPE_*` with `refine_ordering` | `tab:transformer-results`: MolHIV, MolClinTox, MolBBBP, Alchemy |
| `brec`, `brec_qr_refined` | `CanonLPE_*`, `QRRefinedLPE` | `tab:brec-comparison` |
| `qr_refined_full` | `spectral_qr_refined` on all eigenpairs | `tab:invariance` |
| `rebuttal_comb_lap` | `spectral_m3(refine_ordering=True)` on D - A | `tab:noise-sensitivity`, `tab:runtime-scaling` |
| `prism_gf2` | Algorithm 1 sign rule | not used in a table cell |
| `zinc_gatedgcn_heat_t2`, `molpcba_gatedgcn_plain` | alternative provenance of two cells | see below |

Rank embeddings (RoPE, sinusoidal) of the transformer cells are model options and
not part of PRiSM; `prism.adapters.transformer.canonicalize_eigenpairs` returns the
canonical order and rank that they consume.

## Notes on the correspondence between paper and code

* All table cells run the *hybrid* extension. For m = 1 it flips every column so
  that its first significant entry in the canonical vertex order is positive,
  which is FastSign with an ordering heuristic; Refine and Solve (Algorithm 1)
  were not part of the table runs. Both are available (`sign_rule="gf2"`).
* The transformer and BREC harness canonicalizes float32 eigenpairs with
  fixed-decimal tolerances (`legacy_round`); the molecular harness uses float64
  and relative tolerances (`adaptive`). Both are exposed so that either can be
  reproduced exactly.
* Cells with more than one candidate configuration (ZINC / GatedGCN, MolPCBA /
  GatedGCN) are listed in the preset table above; the extra presets carry the
  alternative.

## Tests

```
python -m pytest
```

* `test_golden.py` compares every preset with `tests/fixtures/golden.npz`.
* `test_invariance.py`: sign-flip, permutation and O(m) invariance.
* `test_fastsign.py`, `test_gf2.py`: Algorithms 1 and 2 and the worked example of the paper.
* `test_api.py`, `test_repo_hygiene.py`: interface and repository checks.

The golden file is generated from the original research code with
`LIVE_SRC=<checkouts> python dev/make_golden.py` (see the script header).

## License

MIT. MAP and OAP are adapted from https://github.com/PKU-ML/canonicalization
(MIT, Copyright (c) 2024 PKU-ML).
