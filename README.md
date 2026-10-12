# prism-canonicalization

Reference implementation and experiment code for

> Snir Hordan, Nadav Dym, Tim Seppelt. *Weisfeiler Leman is Incomplete On Graphs With A Simple Spectrum.*
> Advances in Neural Information Processing Systems (NeurIPS), 2026.

The paper has two results. The negative one: for every k there are pairs of graphs with a **simple
spectrum** (all adjacency eigenvalues distinct) that k-WL, and therefore every message-passing GNN of
that power, cannot tell apart. The positive one: simple-spectrum graphs are separated by a *complete*
canonicalization of their eigenvectors. **PRiSM** (Partition, Refine, Solve, Match) is such a
canonicalization. It partitions the vertices by the absolute values of the eigenvector entries, refines
that partition into a vertex order, solves a linear system over GF(2) for the sign of every eigenvector,
and extends the procedure to repeated eigenvalues by a per-eigenspace basis choice. The canonical
eigenvectors, and the vertex order they induce, are used as positional encodings of GNNs and graph
Transformers.

This repository contains PRiSM as a small numpy package (`prism`), the MAP and OAP baselines on the same
interface, and the code of every experiment table of the paper.

## Contents

* [Installation](#installation)
* [Quick start](#quick-start)
* [Repository map](#repository-map)
* [Reproducing the paper](#reproducing-the-paper)
* [Presets and paper cells](#presets)
* [Reproducibility notes](#reproducibility-notes)
* [Tests](#tests), [Citation](#citation), [License](#license)

## Installation

The core package needs only numpy and scipy:

```
git clone https://github.com/snirhordan/prism-canonicalization.git
cd prism-canonicalization
pip install -e .            # numpy, scipy
pip install -e ".[torch]"   # + torch, for prism.baselines.MAP / OAP
pip install -e ".[dgl]"     # + torch and dgl, for the DGL adapter
pip install -e ".[nauty]"   # + pynauty, for ordering="indrefine_nauty" (preset brec_indrefine)
```

`pynauty` (pinned to 2.8.6, the version the tests were run with) is licensed under the GPL-3.0-or-later and is
installed separately; it is imported only when that ordering is requested, and a missing install raises an
`ImportError` that names the extra.

Each experiment has its own pinned environment file. They do not share a Python environment because the
backbones need different versions of torch (2.1 for the Transformers, 2.2 for the DGL GNNs):

| experiment | environment file | stack |
|---|---|---|
| `experiments/molecular_gnn` | `env_molecular.yml` | Python 3.10, torch 2.2.0+cu118, dgl 2.1.0+cu118, ogb 1.3.6, numpy 1.26.4 |
| `experiments/transformer_ogb`, `experiments/brec` | `experiments/transformer_ogb/env_transformer.yml` | Python 3.10, torch 2.1.0 (CUDA 12.1), torch_geometric 2.4.0, ogb 1.3.6, hydra 1.3.2, `brec` 1.0.0, pynauty 2.8.6, numpy 1.26.4 |
| `experiments/appendix` | `env_appendix.yml` | Python 3.10, CPU torch 2.2.0, torch_geometric 2.4.0, ogb 1.3.6, numpy 1.26.4 |

Create them with `conda env create -p <prefix> -f <file>` and set `PYTHONNOUSERSITE=1` while creating and
using them, so that packages in `~/.local` cannot shadow missing pins. Every file lists numpy explicitly
(torch 2.x of these versions needs numpy 1.x). The per-experiment READMEs give the exact commands.

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

The same call for the baselines (needs torch):

```python
from prism.baselines import MAP, OAP, lap_pe, random_sign_flip
pe_map = MAP.from_profile("gnn").pos_enc(A, k=8)
```

## Repository map

```
prism/                      the package
  prism.py                  class PRiSM, the single entry point
  presets.py                named configurations, one per experimental setting
  core.py, gf2.py           canonicalization procedures; the four stages Partition, Refine, Solve, Match
  indrefine.py              the nauty / projection-matrix vertex ordering (ordering="indrefine_nauty")
  operators.py              normalized adjacency, normalized and combinatorial Laplacian
  baselines/                MAP, OAP (adapted from PKU-ML), lap_pe, random_sign_flip
  adapters/                 dgl.py (DGL graphs), transformer.py (canonical order and rank)
experiments/
  molecular_gnn/            GatedGCN and PNA on ZINC, MolTox21, MolPCBA        (tab:molecular-results)
  transformer_ogb/          graph Transformer on MolHIV, MolClinTox, MolBBBP, Alchemy (tab:transformer-results)
  brec/                     BREC expressivity benchmark                          (tab:brec-comparison)
  wl_transformers/          shared model code of the two experiments above (installable library)
  appendix/                 the five appendix tables, CPU only
results/paper_tables/       the numbers printed in the paper, one CSV per table
tests/                      unit tests and golden tests of the package
dev/make_golden.py          regenerates tests/fixtures/golden.npz from the original research code
THIRD_PARTY_NOTICES        licences of the adapted upstream code
CITATION.cff
```

All experiments canonicalize through `prism.PRiSM.from_preset(...)` or `prism.baselines`; the experiment
directories contain no other canonicalization code. SignNet, the sign-invariant network that the molecular GNN
experiment compares against, is a model (not a canonicalization) and lives in that experiment.

## Reproducing the paper

**Hardware.** Canonicalization (PRiSM, MAP, OAP) runs on CPU, in float64. The paper trained every model on a
single NVIDIA A40 (48 GB) per run, from a pool of eight; no run used more than one GPU. All pipelines were
checked end to end on an A40 (one-epoch smoke runs, below). Run each experiment from its own directory; data,
positional-encoding caches and outputs are created below it (`./data`, `./cache`, `./out`, `./outputs`) and are
ignored by git. No datasets, checkpoints or logs are distributed with the repository.

**Cost.** Per-seed GPU time of the paper's runs, from the training logs: GatedGCN on ZINC 1.0 to 4.3 h, PNA on
ZINC 4.8 to 18.0 h, GatedGCN and PNA on MolTox21 0.4 to 1.5 h, MolPCBA 3.8 to 44.6 h (PNA at the upper end),
Transformers 3 to 9 h, BREC several hours for the 360 pairs. The smoke checks below take seconds to a minute.
The molecular and Transformer harnesses write no checkpoints, so a run that is interrupted starts over; the
molecular harness runs one seed per process with `--seed S` (a run without `--seed` trains the four seeds one
after the other).

**Numbers.** The values quoted below are those printed in the paper (also in `results/paper_tables/`). GPU
training is not bit-deterministic across hardware and library builds. The counts of the CPU experiments of the
appendix depend on the eigenvector basis that `numpy.linalg.eigh` returns for repeated eigenvalues, which
varies with the BLAS/LAPACK build and the processor (see `experiments/appendix/README.md`).

### Main text

#### tab:brec-comparison: BREC expressivity (360 pairs)

* **Experiment:** `experiments/brec`, a GINE backbone whose only positional information is a DeepSets embedding of
  the first 8 Laplacian eigenvectors; embedding size 32, 2 layers, about 12K parameters, identical for all rows.
* **Environment:** the Transformer environment (`experiments/transformer_ogb/env_transformer.yml`); the `brec`
  package downloads and converts the dataset on first use.
* **Presets:** PRiSM `brec_indrefine` (individualization-refinement ordering through pynauty, then spectral
  canonicalization; see the notes below), MAP and OAP with profile `brec`. Two further PRiSM variants are shipped
  for comparison: `brec` (norm-signature ordering) and `brec_qr_refined` (QR for every multiplicity).
* **Seeds:** none beyond the fixed `seed_everything(2022)` of the `brec` package; each pair trains its own model.
* **Smoke check:** about 10 s for two pairs on an A40.

```
cd experiments/brec
python expressivity/brec_benchmark.py +experiment=brec_prism           # PRiSM, IndRefineNauty ordering
python expressivity/brec_benchmark.py +experiment=brec_prism_canonlpe  # PRiSM, norm-signature ordering (preset brec)
python expressivity/brec_benchmark.py +experiment=brec_prism_qr        # PRiSM, QR for every multiplicity
python expressivity/brec_benchmark.py +experiment=brec_map         # MAP
python expressivity/brec_benchmark.py +experiment=brec_oap         # OAP
python expressivity/brec_benchmark.py +experiment=brec_prism smoke=true
```

Paper (pairs distinguished):

| method | Basic /60 | Regular /100 | Ext. /100 | CFI /100 | Total /360 |
|---|---:|---:|---:|---:|---:|
| PRiSM | 60 | 50 | 99 | 3 | 212 |
| MAP | 60 | 42 | 94 | 3 | 199 |
| OAP | 60 | 42 | 94 | 3 | 199 |

#### tab:molecular-results: GatedGCN and PNA on ZINC, MolTox21, MolPCBA

* **Experiment:** `experiments/molecular_gnn`, one config per backbone, dataset and method
  (`configs/<Backbone>_<DATASET>_<method>.json`; 24 table cells).
* **Environment:** `experiments/molecular_gnn/env_molecular.yml`; `python scripts/download_data.py` fetches ZINC and
  the OGB datasets into `./data` (about 2.4 GB, mostly MolPCBA).
* **Presets** (`k` = number of eigenvector columns): ZINC GatedGCN `zinc_gatedgcn` (k=8), ZINC PNA `zinc_pna` (k=8),
  MolTox21 GatedGCN `moltox21_gatedgcn` (k=3), MolTox21 PNA `moltox21_pna` (k=16), MolPCBA GatedGCN
  `molpcba_gatedgcn` (k=3), MolPCBA PNA `molpcba_pna` (k=16). MAP and OAP: profile `gnn`.
* **Seeds:** 41, 95, 12, 35; mean and standard deviation (ddof 0) of the test metric over the four seeds.
  ZINC reports the test MAE of the last epoch; MolTox21 and MolPCBA the test metric at the epoch of best validation metric.
* **Smoke check:** 12 to 25 s (71 s for PNA on MolPCBA) per config on an A40, one epoch on 512/128/128 graphs.
* **Cost notes:** build the positional-encoding cache once on CPU before launching seeds
  (`--build_pe_only --pe_workers 4`; about two hours serially for MolPCBA with k=16); PNA on MolPCBA needs one
  48 GB GPU per process. `PNA_MOLPCBA_SignNet` does not fit in memory (OOM in the table).

```
cd experiments/molecular_gnn
python scripts/download_data.py
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json            # one cell, all four seeds
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json --seed 41   # one seed
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json --smoke

# the whole table
for bb in GatedGCN PNA; do for m in PRiSM SignNet MAP OAP; do
  python main_ZINC_graph_regression.py --config configs/${bb}_ZINC_${m}.json; done; done
for ds in MOLTOX21 MOLPCBA; do for bb in GatedGCN PNA; do for m in PRiSM SignNet MAP OAP; do
  python main_OGBMOL_graph_classification.py --config configs/${bb}_${ds}_${m}.json; done; done; done
```

Paper (mean +- std over four seeds; ZINC: test MAE, lower is better; MolTox21: ROC-AUC; MolPCBA: average precision):

| method | ZINC GatedGCN | ZINC PNA | MolTox21 GatedGCN | MolTox21 PNA | MolPCBA GatedGCN | MolPCBA PNA |
|---|---|---|---|---|---|---|
| PRiSM | **0.110 +- 0.002** | **0.103 +- 0.003** | **0.783 +- 0.006** | 0.762 +- 0.004 | **0.272 +- 0.003** | **0.278 +- 0.001** |
| SignNet | 0.121 +- 0.002 | 0.105 +- 0.007 | 0.769 +- 0.007 | 0.744 +- 0.002 | 0.260 +- 0.002 | OOM |
| MAP | 0.123 +- 0.003 | 0.104 +- 0.003 | 0.777 +- 0.004 | **0.763 +- 0.004** | 0.268 +- 0.002 | 0.274 +- 0.003 |
| OAP | 0.126 +- 0.004 | 0.106 +- 0.002 | **0.783 +- 0.006** | 0.761 +- 0.008 | 0.270 +- 0.002 | 0.272 +- 0.002 |

Two PRiSM cells have an extra config with its own preset for an alternative formulation of the same cell
(`GatedGCN_ZINC_PRiSM_heat_t2`, preset `zinc_gatedgcn_heat_t2`; `GatedGCN_MOLPCBA_PRiSM_plain`, preset
`molpcba_gatedgcn_plain`). Plain LapPE configs are included for completeness.

#### tab:transformer-results: Transformer backbone on OGB and Alchemy

* **Experiment:** `experiments/transformer_ogb`, the graph Transformer of Kim et al. as adapted in
  `experiments/wl_transformers`, with the canonicalized Laplacian positional encoding. PRiSM additionally supplies the
  canonical vertex order, fed to the model as a rank embedding (RoPE on MolHIV and Alchemy, sinusoidal on
  MolClinTox, none on MolBBBP).
* **Environment:** `experiments/transformer_ogb/env_transformer.yml`. Datasets download on first use into `./data`
  (Alchemy downloads the full 200K-graph `alchemy_full` and uses the 10K subset of `fine-tuning/*_al_10.index`).
* **Preset:** PRiSM `transformer_refined` (through `wl_transformers.pe_types`); MAP and OAP profile `brec`.
* **Seeds:** per row below (shell loop `for s in ...`); the reported value is the test score at the best
  validation epoch.
* **Smoke check:** 13 to 20 s on an A40 (64 graphs per split, a few steps).

```
cd experiments/transformer_ogb
for s in 0 1 2; do python fine-tuning/ogbg.py +experiment=molhiv_prism seed=$s; done
python fine-tuning/alchemy.py +experiment=alchemy_prism seed=0
python fine-tuning/ogbg.py +experiment=molclintox_prism smoke=true
```

All 16 rows are named configs: `{molhiv,molclintox,molbbbp}_{prism,lpe,map,oap}` with `fine-tuning/ogbg.py` and
`alchemy_{prism,lpe,map,oap}` with `fine-tuning/alchemy.py`. The seed list of each row is in
`experiments/transformer_ogb/README.md`; the paper reports three seeds per cell, Alchemy seeds {0, 1, 3}.

Paper (ROC-AUC, higher is better, for the three OGB datasets; test MAE, lower is better, for Alchemy):

| method | MolHIV | MolClinTox | MolBBBP | Alchemy |
|---|---|---|---|---|
| PRiSM | **0.7430 +- 0.0233** | **0.8576 +- 0.0038** | **0.7088 +- 0.0293** | **0.11956 +- 0.00130** |
| LPE | 0.6891 +- 0.0068 | 0.7905 +- 0.0444 | 0.6798 +- 0.0179 | 0.12655 +- 0.00056 |
| MAP | 0.7221 +- 0.0190 | 0.7900 +- 0.0440 | 0.7029 +- 0.0019 | 0.13058 +- 0.00038 |
| OAP | 0.6878 +- 0.0197 | 0.7820 +- 0.0530 | 0.6825 +- 0.0323 | 0.13216 +- 0.00488 |

### Appendix

All five appendix scripts run on CPU, in the environment `experiments/appendix/env_appendix.yml`, from
`experiments/appendix`. Datasets are downloaded into `--data-dir` (default `./data`). `--help` lists options for
smaller sample sizes. Run times are wall-clock on one core of a Xeon server.

#### tab:rbound-stats: row bound r = n - L of the GF(2) system

* **Command:** `python rbound_stats.py --data-dir ./data --out results/rbound_stats.csv`
* **Method:** `prism.gf2.partition_stage` (stage Partition, tolerance 1e-6) on the full eigendecomposition of the
  normalized Laplacian, validation splits of ZINC-12K and the eleven OGB-Mol datasets.
* **Seeds:** none (deterministic). **Time:** under a minute to about five minutes once the datasets are processed.

Paper: number of validation graphs, mean nodes, share with r=0, share with exactly one non-singleton block, median r, share with r <= 4, max r.

| dataset | val graphs | mean n | r=0 | 1 block | median r | r<=4 | max r |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZINC-12K | 1,000 | 23.1 | 22.8% | 10.9% | 2.0 | 88.0% | 11 |
| MolHIV | 4,113 | 27.8 | 21.2% | 6.4% | 3.0 | 63.9% | 53 |
| MolTox21 | 783 | 26.8 | 17.5% | 8.6% | 3.0 | 67.4% | 32 |
| MolClinTox | 148 | 32.8 | 27.0% | 9.5% | 2.0 | 70.3% | 37 |
| MolBBBP | 204 | 33.2 | 14.2% | 17.2% | 2.0 | 81.4% | 32 |
| MolBACE | 151 | 37.2 | 3.3% | 10.6% | 3.0 | 81.5% | 6 |
| MolESOL | 113 | 17.7 | 38.1% | 14.2% | 1.0 | 77.0% | 15 |
| MolFreeSolv | 64 | 12.0 | 45.3% | 4.7% | 1.5 | 78.1% | 12 |
| MolLipo | 420 | 27.3 | 14.8% | 7.9% | 3.0 | 74.3% | 25 |
| MolMUV | 9,309 | 25.3 | 12.8% | 3.3% | 3.0 | 73.3% | 17 |
| MolToxCast | 858 | 26.2 | 20.5% | 10.7% | 2.5 | 69.4% | 37 |
| MolPCBA | 43,793 | 27.1 | 13.9% | 4.7% | 3.0 | 74.3% | 71 |

`experiments/appendix/results/rbound_stats.csv` is the output of the port on one machine.

#### tab:multiplicity-stats: eigenvalue multiplicities

* **Command:** `python multiplicity_stats.py --data-dir ./data --out results/multiplicity_stats.csv`
* **Method:** eigenvalues of D^-1/2 A D^-1/2 (`numpy.linalg.eigvalsh`), classes of absolute tolerance 1e-6.
* **Seed:** `random.seed(42)`, 500 graphs per dataset. **Time:** seconds.

| dataset | sampled | mean n | #m=1 | #m=2 | #m=3 | #m>=4 | % simple spectrum |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZINC | 500 | 23.0 | 20.86 | 0.59 | 0.20 | 0.09 | 38.0% |
| MolHIV | 500 | 25.7 | 21.54 | 0.88 | 0.30 | 0.28 | 32.8% |
| MolTox21 | 500 | 17.9 | 14.96 | 0.72 | 0.21 | 0.17 | 38.4% |
| MolClinTox | 500 | 26.6 | 23.34 | 0.59 | 0.25 | 0.24 | 28.6% |
| MolBBBP | 500 | 23.5 | 20.93 | 0.49 | 0.26 | 0.16 | 34.4% |

#### tab:invariance: permutation-equivariance failures

* **Command:** `python invariance_test.py --data-dir ./data --out results/invariance.json`
* **Preset:** PRiSM `qr_refined_full`; MAP and OAP profile `gnn`.
* **Protocol:** 200 training graphs per dataset times 5 random permutations (1,000 trials per cell), full
  eigendecomposition; a trial fails when the canonical encoding of the permuted graph is not a row-permutation
  of the original (row-multiset equality, atol 1e-6). **Time:** a few minutes.

| dataset | PRiSM | MAP | OAP |
|---|---:|---:|---:|
| ZINC | 0 | 692 | 684 |
| MolTox21 | 10 | 772 | 753 |

#### tab:noise-sensitivity: invariance under symmetric matrix noise

* **Command:** `python noise_sensitivity.py --data-dir ./data --out results/noise_sensitivity.json`
* **Preset:** PRiSM `rebuttal_comb_lap` (hybrid with refinement on D - A); MAP and OAP profile `brec`.
* **Protocol:** first 200 ZINC-12K validation graphs, 5 permutations each, failures out of 1,000 per seed, mean +- std
  over seeds 42, 43, 44. **Time:** about 20 minutes.

| sigma | PRiSM | MAP | OAP |
|---|---:|---:|---:|
| 0 | 0.0 +- 0.0 | 753.0 +- 7.0 | 748.7 +- 7.6 |
| 1e-12 | 2.7 +- 3.1 | 944.7 +- 0.6 | 743.0 +- 6.9 |
| 1e-8 | 0.3 +- 0.6 | 447.3 +- 44.0 | 19.7 +- 5.5 |
| 1e-6 | 0.0 +- 0.0 | 8.7 +- 4.6 | 0.0 +- 0.0 |
| 1e-4 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| 1e-2 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |

#### tab:runtime-scaling: canonicalization time

* **Command:** `python runtime_scaling.py --out results/runtime_scaling.json`
* **Preset:** PRiSM `rebuttal_comb_lap`; MAP and OAP profile `brec`. The script pins OpenMP, MKL, OpenBLAS and
  torch to one thread.
* **Protocol:** Erdos-Renyi graphs, 50 per size n in {16, ..., 512}, seed 42, one untimed warm-up call, then the
  timed call; median milliseconds per graph. **Time:** about an hour (MAP and OAP at n = 256, 512 dominate).
  Absolute times depend on the processor.

| n | PRiSM (ms) | MAP (ms) | OAP (ms) | vs MAP | vs OAP |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.04 | 16.06 | 16.46 | 15.4x | 15.8x |
| 32 | 3.54 | 59.12 | 60.05 | 16.7x | 17.0x |
| 64 | 16.16 | 258.23 | 258.88 | 16.0x | 16.0x |
| 128 | 91.86 | 3064.77 | 3064.46 | 33.4x | 33.4x |
| 256 | 757.73 | 12441.65 | 12027.64 | 16.4x | 15.9x |
| 512 | 4155.32 | 12961.90 | 13065.93 | 3.1x | 3.1x |

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
| ordering | `ordering` | `spectral` (norm-value signature, all table cells but the BREC row) or `indrefine_nauty` (eigenspace projection matrices and a nauty canonical labelling, then pivoted QR; needs `pynauty` and the adjacency matrix; `sign_rule`, `mult_mode` and `refine_ordering` must keep their defaults) |
| | `ir_order_decimals`, `ir_proj_decimals` | `indrefine_nauty` only: rounding of the vertex colours and ordering keys (8) and of the projection matrices (10) |
| numerics | `tol_profile` | `adaptive` (relative snapping, eps floor) or `legacy_round` (fixed decimals) |
| | `quantize_float32` | pass eigenpairs through float32 first (transformer / BREC storage) |
| post-processing | `weight`, `weight_param` | `eigval`, `inv_sqrt_lap` (eps), `heat` (t) |

Methods: `canonicalize` (`adjacency=` is required by `ordering="indrefine_nauty"`, which `pos_enc` and
`spectrum` supply themselves), `eigendecompose`, `spectrum`, `pos_enc`, `from_preset`, `get_config` (the
`ordering` arguments appear only when the ordering is not `spectral`, so the configuration, and the
positional-encoding cache key of the molecular harness, of every other instance is unchanged), and the
stand-alone procedures `canonicalize_simple` (Algorithm 1)
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
| `brec_indrefine` | `IndRefineNauty` | `tab:brec-comparison`, PRiSM row (see the reproducibility notes) |
| `brec`, `brec_qr_refined` | `CanonLPE_*`, `QRRefinedLPE` | `tab:brec-comparison` variants, not the printed row |
| `qr_refined_full` | `spectral_qr_refined` on all eigenpairs | `tab:invariance` |
| `rebuttal_comb_lap` | `spectral_m3(refine_ordering=True)` on D - A | `tab:noise-sensitivity`, `tab:runtime-scaling` |
| (no preset) | `prism.gf2.partition_stage` | `tab:rbound-stats` |
| `prism_gf2` | Algorithm 1 sign rule | not used in a table cell |
| `zinc_gatedgcn_heat_t2`, `molpcba_gatedgcn_plain` | alternative provenance of two cells | see below |

Rank embeddings (RoPE, sinusoidal) of the transformer cells are model options and
not part of PRiSM; `prism.adapters.transformer.canonicalize_eigenpairs` returns the
canonical order and rank that they consume.

## Notes on the correspondence between paper and code

* All table cells use a `first_pivot` sign rule; in the BREC PRiSM row the vertex order that picks the pivot is
  the `indrefine_nauty` ordering instead of the norm-signature ordering (see the BREC notes). For a simple eigenvalue (m = 1) it flips every
  column so that its first significant entry in the canonical vertex order is positive, which is FastSign
  with an ordering heuristic; the GF(2) system of Algorithm 1 (Refine and Solve) was not part of the
  table runs. It is available as `sign_rule="gf2"` (preset `prism_gf2`). Repeated eigenvalues are
  handled by the `hybrid` or the `qr` extension, whichever the preset of the cell names.
* The transformer and BREC harness canonicalizes float32 eigenpairs with
  fixed-decimal tolerances (`legacy_round`); the molecular harness uses float64
  and relative tolerances (`adaptive`). Both are exposed so that either can be
  reproduced exactly. The `indrefine_nauty` ordering has its own rounding
  (`ir_order_decimals`, `ir_proj_decimals`) and groups eigenvalues by a relative
  tolerance of 1e-10 (`prism.indrefine.detect_multiplicities`), as the research code does.
* Cells with more than one candidate configuration (ZINC / GatedGCN, MolPCBA /
  GatedGCN) are listed in the preset table above; the extra presets carry the
  alternative.

## Reproducibility notes

What is and is not known to reproduce the paper tables, cell by cell. The numbers in `results/paper_tables/`
are the paper's, not outputs of this repository.

**Sign rule (all tables).**
* Every table number was produced with a first-pivot sign rule (FastSign with an ordering heuristic; the presets
  use `sign_rule="first_pivot"`). Repeated eigenvalues are handled by the `hybrid` or the `qr` extension, as each
  preset names. The BREC PRiSM row is the one exception to the norm-signature ordering: its launch used the
  individualization-refinement ordering (see BREC below), which has its own pivoted QR for every multiplicity.
* The full Algorithm 1 (Partition, Refine, Solve with the GF(2) system, Match) is provided as preset `prism_gf2`
  and tested against the worked example of the paper (`tests/test_gf2.py`). It did not produce any table number.

**Molecular GNNs (`tab:molecular-results`).**
* Two cells have two candidate configurations, both shipped. ZINC / GatedGCN: `zinc_gatedgcn` (`spectral_m3`,
  0.1096 +- 0.0015 over four logged seeds) and `zinc_gatedgcn_heat_t2` (heat weighting t = 2, input dropout 0.2,
  0.1096 +- 0.0019). Both round to the paper's 0.110; the README recommends `zinc_gatedgcn`, the configuration
  whose per-seed logs are documented, and the paper's std (0.002) is closer to the heat run.
  MolPCBA / GatedGCN: `molpcba_gatedgcn` (eigenvalue weighting, 0.2724 +- 0.0021 over four seeds) and
  `molpcba_gatedgcn_plain` (0.2698 +- 0.0022). The README recommends `molpcba_gatedgcn` because its mean rounds
  to the paper's 0.272 and the plain one does not; that the eigenvalue weighting was used for this cell is
  inferred, not recorded.
* The MolPCBA SignNet, MAP and OAP GatedGCN numbers (0.260, 0.268, 0.270) are taken from Ma et al. (Table 6),
  because the local baseline runs were stopped before convergence. The PNA MAP and OAP entries
  (0.274, 0.272) differ from theirs (0.281, 0.279); their source is not established. PNA / SignNet is "OOM" as
  reported by Ma et al. The shipped configs of these baseline cells may not reproduce the printed numbers.
* PNA / ZINC / SignNet: the logged runs give 0.0932 +- 0.0043 (paper 0.105 +- 0.007).
* The OGB loader (inherited from the harness) drops graphs with at most 5 nodes from every split, including test
  (MolTox21: 6015 / 782 / 782 graphs instead of the scaffold split's 6264 / 783 / 784).

**Transformer (`tab:transformer-results`).**
* The shipped configs encode the hyperparameters that were run and are authoritative where they differ from the
  appendix hyperparameter table: 768 hidden / 16 heads / 12 layers, 96 eigenvectors, batch size 32 (OGB) and
  64 (Alchemy), weight decay 0.1, no early stopping (test score at the best validation epoch), graph-token readout.
* MolClinTox PRiSM 0.8576 +- 0.0038 was obtained before a later fix to how the canonical rank is computed for
  `refine_ordering=True`. Runs of the fixed code give 0.841 +- 0.050 over six seeds (0.8375 on seeds 0 to 2).
* Alchemy: the split index files in `fine-tuning/` are read by every Alchemy config and are identical to those of
  the original code. The 10,000 training indices
  contain 9,789 distinct graphs (validation 997 of 1000, test 999 of 1000), and the splits overlap: 59 graphs in
  train and test, 45 in train and val, 6 in val and test. The Alchemy LPE value (0.12655 +- 0.00056) could not be
  regenerated from the available logs.
* Seeds are as run: Alchemy PRiSM 0, 1, 2; Alchemy MAP and OAP 0, 1, 3; MolClinTox LPE and OAP have six seeds
  in the logs. Alchemy MAP and OAP ran at lr 1e-3, PRiSM at 1e-4; the configs say which.

**BREC (`tab:brec-comparison`).**
* The raw per-category results behind 212 / 199 / 199 were not recovered. The only matched ~12K-parameter launches
  found (three runs of the original code whose result logs are empty) used: GNN backbone, `num_vecs` 8, embed dim
  32, 2 layers, 2 heads, mean pooling (12,656 parameters), 20 epochs, lr 1e-4. The MAP and OAP configs shipped here
  use those settings.
* The launch matching the PRiSM row used `pe_type=IndRefineNauty`, and the table's source file describes the row as
  individualization-refinement followed by QR. That variant is now part of this repository: `brec_prism` runs it
  (`PRiSM.from_preset("brec_indrefine")`, `ordering="indrefine_nauty"`, with `pynauty`). It orders the vertices of each
  eigenspace by the diagonal and the sorted absolute row of the eigenspace projection matrix, with a nauty canonical
  labelling of the graph (coloured by the projection-matrix diagonals) as the last tie-break, fixes the sign of a
  simple eigenvector by its first entry in that order, and resolves a repeated eigenvalue by QR on the first linearly
  independent pivot rows. It is golden-tested against the research code (`tests/test_golden.py`). That file was
  edited after the matched launches had started, and the revision ported is its latest one, so it may differ from the
  code that was running.
* The link between those launches and the printed numbers is inferred, and `brec_prism` has not been run to the
  full protocol here, so 212 is not claimed as reproduced. The earlier reconstruction (norm-signature ordering,
  preset `brec`) is kept as `brec_prism_canonlpe`, and `brec_prism_qr` is unchanged; neither is the variant of the
  launch.
* The output does not depend on the eigenvector basis (up to the rounding of the ordering keys). The nauty canonical
  form is unique only up to automorphisms: on a graph with non-trivial automorphisms, different labellings of the
  same graph can lead to different representatives, and the output is then not permutation-equivariant. The tests
  check equivariance on graphs without automorphisms.

**Determinism.**
* `eigsh` with a random start vector is used for graphs with n > 100 (n > 97 for LPE) in the transformer harness
  and, for BREC with 8 eigenvectors, for every graph with n > 12, so those runs are not bit-reproducible.
  On BREC graphs this reaches the canonical eigenvectors: transforming the same graph twice sometimes gives
  eigenvectors that differ by more than 1e-3, for every method (51 sampled graphs, one pair of transforms each:
  10 differ for `IndRefineNauty`, 7 for the norm-signature ordering of `brec`, 18 for MAP; none of the Basic or
  Extension graphs sampled differed). The live research code of the IndRefine ordering shows the same variation;
  given one shared eigendecomposition, it and this repository agree exactly on 255 sampled BREC graphs.
* Repeated eigenvalues make the `eigh` basis depend on the CPU and LAPACK build. Appendix counts that keep
  that basis (MAP, OAP, r-bound) move by a few graphs between machines. Reproduced vs paper, `tab:invariance`
  failures (PRiSM / MAP / OAP): ZINC 0 / 695 / 694 vs 0 / 692 / 684 and MolTox21 9 / 752 / 740 vs
  10 / 772 / 753 on one node; 0 / 677 / 678 and 8 / 770 / 756 on another. In `tab:noise-sensitivity`, PRiSM and
  OAP match on the first node; MAP at sigma 1e-8 and 1e-6 gives 441.3 and 6.0 vs 447.3 and 8.7.
* `tab:multiplicity-stats` reproduces exactly (to the printed precision); `tab:rbound-stats` reproduces exactly on
  one node and differs in 6 of 12 rows on another (percentages by at most 1.6 points; MolToxCast median r 3.0 vs
  2.5; MolPCBA max r 77 vs 71).
* `tab:runtime-scaling` is not reproduced with threads pinned to one core: PRiSM is within 1.1 to 1.6 times the
  paper, MAP and OAP are faster, giving speed-ups of 12, 14, 13, 9, 4.9, 2.5 for n = 16 to 512 (paper 15, 17,
  16, 33, 16, 3.1).

**Tests.** The golden tests for MAP and OAP depend on near-tied tie-breaks and require torch 2.1 or 2.2.

## Tests

```
python -m pytest                       # package: golden, invariance, FastSign, GF(2), API, repository hygiene
```

* `test_golden.py` compares every preset with `tests/fixtures/golden.npz`, generated from the original research
  code with `LIVE_SRC=<checkouts> python dev/make_golden.py` (see the script header); the `brec_indrefine` keys can
  be added to an existing fixture alone with `--only-indrefine`.
* `test_indrefine.py`: basis, permutation and eigenspace-rotation properties of `ordering="indrefine_nauty"`.
* `test_invariance.py`: sign-flip, permutation and O(m) invariance.
* `test_fastsign.py`, `test_gf2.py`: Algorithms 1 and 2 and the worked example of the paper.
* `test_api.py`, `test_repo_hygiene.py`: interface and repository checks.

Each experiment directory has its own `tests/` (configs, shapes, parameter counts, command-line entry points on
synthetic data); run them inside the experiment's environment with `python -m pytest tests`. The package tests
need only numpy and scipy; the tests that import torch or pynauty are skipped without them. The MAP and OAP golden tests round
projection norms to 14 decimals, so they are only reproducible with the torch versions of the paper (2.1 or 2.2;
the `torch` extra pins `torch>=2.1,<2.3`). With a newer torch some of them fail on near-tied tie-breaks.

## Citation

```bibtex
@inproceedings{hordan2026weisfeiler,
  title     = {Weisfeiler Leman is Incomplete On Graphs With A Simple Spectrum},
  author    = {Hordan, Snir and Dym, Nadav and Seppelt, Tim},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

`CITATION.cff` carries the same information for GitHub.

## License

MIT, see `LICENSE`. MAP and OAP are adapted from https://github.com/PKU-ML/canonicalization (MIT, Copyright (c)
2024 PKU-ML). The experiments adapt wl-transformers, BREC, benchmarking-gnns, gnn-lspe, SignNet-BasisNet and the
PNA reference implementation, all MIT; their copyright notices are in `THIRD_PARTY_NOTICES`. Datasets (ZINC, OGB,
Alchemy, BREC) are downloaded from their original hosts and keep their own licences.
