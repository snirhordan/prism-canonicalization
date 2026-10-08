# Appendix tables

Five scripts, one per appendix table of the paper. They run on CPU. Canonicalization
goes through `prism` (`prism.PRiSM.from_preset(...)`, `prism.baselines.MAP/OAP`); the
scripts only load graphs, run the protocol and write the table.

| table | script | what it measures | canonicalizers |
|---|---|---|---|
| `tab:rbound-stats` | `rbound_stats.py` | bound `r = n - L` on the rows of the Z_2 system, 12 datasets | Partition stage `prism.gf2.partition_stage` |
| `tab:multiplicity-stats` | `multiplicity_stats.py` | eigenvalue multiplicities of 500 sampled graphs, 5 datasets | none (eigenvalues only) |
| `tab:invariance` | `invariance_test.py` | permutation-equivariance failures, ZINC and MolTox21 | preset `qr_refined_full`; `MAP`/`OAP` profile `gnn` |
| `tab:noise-sensitivity` | `noise_sensitivity.py` | the same under symmetric matrix noise | preset `rebuttal_comb_lap`; `MAP`/`OAP` profile `brec` |
| `tab:runtime-scaling` | `runtime_scaling.py` | median canonicalization time, Erdos-Renyi graphs | preset `rebuttal_comb_lap`; `MAP`/`OAP` profile `brec` |

## Setup

```
cd experiments/appendix
PYTHONNOUSERSITE=1 conda env create -p ./env_appendix -f env_appendix.yml
conda activate ./env_appendix
export PYTHONNOUSERSITE=1        # keep packages of ~/.local out of the environment
```

The environment holds numpy 1.26.4 (torch 2.2 needs numpy 1.x), scipy, networkx,
CPU torch 2.2.0, PyTorch Geometric 2.4.0, OGB 1.3.6 and the repository's `prism`
package. Run the commands below from this directory.

Datasets are downloaded on first use into `--data-dir` (default `./data`, not
tracked): ZINC (PyG, from the benchmarking-gnns release) into `data/ZINC` and the
OGB-Mol datasets into `data/ogb`. All twelve OGB-Mol datasets together take about
3 GB after processing; the first processing of `ogbg-molpcba` needs a few GB of
memory. If the process dies right after a dataset was processed, run the command
again: the processed files are reused.

Every script writes its result to `--out` (default `results/<name>.<ext>`) and
prints the table. `--help` lists the options; smaller sample sizes
(`--max-graphs`, `--n-graphs`, `--reps`, ...) are available for quick checks.

## Commands and expected values

The expected values are those printed in the paper.

### `tab:rbound-stats`

```
python rbound_stats.py --data-dir ./data --out results/rbound_stats.csv
```

A few minutes on one core once the datasets are processed. Full eigendecomposition of the normalized Laplacian
`I - D^-1/2 A D^-1/2` for the validation split of every dataset, Partition with
tolerance `1e-6`.

| dataset | `|val|` | mean n | r = 0 | 1 block | median r | r <= 4 | max r |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZINC-12K (`ZINC-12K`) | 1,000 | 23.1 | 22.8% | 10.9% | 2.0 | 88.0% | 11 |
| MolHIV (`ogbg-molhiv`) | 4,113 | 27.8 | 21.2% | 6.4% | 3.0 | 63.9% | 53 |
| MolTox21 (`ogbg-moltox21`) | 783 | 26.8 | 17.5% | 8.6% | 3.0 | 67.4% | 32 |
| MolClinTox (`ogbg-molclintox`) | 148 | 32.8 | 27.0% | 9.5% | 2.0 | 70.3% | 37 |
| MolBBBP (`ogbg-molbbbp`) | 204 | 33.2 | 14.2% | 17.2% | 2.0 | 81.4% | 32 |
| MolBACE (`ogbg-molbace`) | 151 | 37.2 | 3.3% | 10.6% | 3.0 | 81.5% | 6 |
| MolESOL (`ogbg-molesol`) | 113 | 17.7 | 38.1% | 14.2% | 1.0 | 77.0% | 15 |
| MolFreeSolv (`ogbg-molfreesolv`) | 64 | 12.0 | 45.3% | 4.7% | 1.5 | 78.1% | 12 |
| MolLipo (`ogbg-mollipo`) | 420 | 27.3 | 14.8% | 7.9% | 3.0 | 74.3% | 25 |
| MolMUV (`ogbg-molmuv`) | 9,309 | 25.3 | 12.8% | 3.3% | 3.0 | 73.3% | 17 |
| MolToxCast (`ogbg-moltoxcast`) | 858 | 26.2 | 20.5% | 10.7% | 2.5 | 69.4% | 37 |
| MolPCBA (`ogbg-molpcba`) | 43,793 | 27.1 | 13.9% | 4.7% | 3.0 | 74.3% | 71 |

### `tab:multiplicity-stats`

```
python multiplicity_stats.py --data-dir ./data --out results/multiplicity_stats.csv
```

Seconds to a minute. 500 graphs per dataset drawn with `random.seed(42)` (all
three splits of ZINC-12K, all graphs of the OGB datasets); eigenvalues of
`D^-1/2 A D^-1/2` by `numpy.linalg.eigvalsh`, classes of absolute tolerance `1e-6`.

| dataset | sampled | mean n | mean #m=1 | #m=2 | #m=3 | #m>=4 | % simple |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZINC | 500 | 23.0 | 20.86 | 0.59 | 0.20 | 0.09 | 38.0% |
| MolHIV | 500 | 25.7 | 21.54 | 0.88 | 0.30 | 0.28 | 32.8% |
| MolTox21 | 500 | 17.9 | 14.96 | 0.72 | 0.21 | 0.17 | 38.4% |
| MolClinTox | 500 | 26.6 | 23.34 | 0.59 | 0.25 | 0.24 | 28.6% |
| MolBBBP | 500 | 23.5 | 20.93 | 0.49 | 0.26 | 0.16 | 34.4% |

### `tab:invariance`

```
python invariance_test.py --data-dir ./data --out results/invariance.json
```

A few minutes on one core (MAP and OAP dominate). 200 graphs per dataset times 5 random
permutations, 1,000 trials per cell. Eigendecomposition of `D^-1/2 A D^-1/2` without
the trivial eigenpair (all `n - 1` eigenvectors; the eigenvectors are those of
the normalized Laplacian). The ZINC graphs are the first 200 molecules of the full
ZINC training file (`data/ZINC/raw/train.pickle`, about 220k molecules; loading it
takes a minute and a few GB of memory) with more than 5 atoms; the MolTox21 graphs are
the first 200 of the `ogbg-moltox21` training split with more than 5 nodes.

| dataset | PRiSM | MAP | OAP |
|---|---:|---:|---:|
| ZINC | 0 | 692 | 684 |
| MolTox21 | 10 | 772 | 753 |

### `tab:noise-sensitivity`

```
python noise_sensitivity.py --data-dir ./data --out results/noise_sensitivity.json
```

About 20 minutes on one core (three seeds). The first 200 ZINC validation graphs; the
failure count is out of 1,000 trials per seed, reported as mean +- standard deviation
(ddof 1) over seeds 42, 43, 44. The operator is the combinatorial Laplacian `D - A`.

| sigma | PRiSM | MAP | OAP |
|---|---:|---:|---:|
| 0 | 0.0 +- 0.0 | 753.0 +- 7.0 | 748.7 +- 7.6 |
| 1e-12 | 2.7 +- 3.1 | 944.7 +- 0.6 | 743.0 +- 6.9 |
| 1e-8 | 0.3 +- 0.6 | 447.3 +- 44.0 | 19.7 +- 5.5 |
| 1e-6 | 0.0 +- 0.0 | 8.7 +- 4.6 | 0.0 +- 0.0 |
| 1e-4 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| 1e-2 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |

### `tab:runtime-scaling`

```
python runtime_scaling.py --out results/runtime_scaling.json
```

About an hour on one core, dominated by MAP and OAP at `n = 256, 512`. The script pins the
OpenMP, MKL, OpenBLAS and PyTorch thread counts to one. Connected Erdos-Renyi graphs
`G(n, p)` with `p = max(log(n)/n, 0.1)`, 50 graphs per size, seed 42; one untimed
warm-up call per method and graph, then one timed call (canonicalization plus the
lexicographic sort of the rows; the eigendecomposition is not timed). Medians in
milliseconds:

| n | PRiSM | MAP | OAP | vs MAP | vs OAP |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.04 | 16.06 | 16.46 | 15.4x | 15.8x |
| 32 | 3.54 | 59.12 | 60.05 | 16.7x | 17.0x |
| 64 | 16.16 | 258.23 | 258.88 | 16.0x | 16.0x |
| 128 | 91.86 | 3064.77 | 3064.46 | 33.4x | 33.4x |
| 256 | 757.73 | 12441.65 | 12027.64 | 16.4x | 15.9x |
| 512 | 4155.32 | 12961.90 | 13065.93 | 3.1x | 3.1x |

Absolute times depend on the processor.

## Reproducibility of the counts

The statistics of `rbound_stats.py`, `invariance_test.py` and `noise_sensitivity.py`
depend on the eigenvector basis that `numpy.linalg.eigh` returns for repeated
eigenvalues. That basis is a property of the LAPACK/BLAS build and of the processor
kernels it selects. On a different machine a few graphs change class in
`rbound_stats.py`, and the failure counts of MAP and OAP in `invariance_test.py` and
`noise_sensitivity.py` move by up to a few percent of the trials; the PRiSM columns
stay at zero or within a few failures. `multiplicity_stats.py` uses eigenvalues only
and does not depend on it.

## Tests

```
python -m pytest tests
```

The tests use small synthetic graphs and need no download.
