# Molecular property prediction with GNN backbones

GatedGCN and PNA on ZINC 12K (MAE), OGB MolTox21 (ROC-AUC) and OGB MolPCBA (average precision)
with the positional encodings compared in the paper: **PRiSM**, **SignNet**, **MAP**, **OAP**
(and plain **LapPE** with random sign flips). Every canonicalization goes through the `prism`
package of this repository: PRiSM through `prism.PRiSM.from_preset(...)`, MAP and OAP through
`prism.baselines`. There is no other canonicalization code in this directory.

## Setup

From this directory:

```bash
PYTHONNOUSERSITE=1 conda env create -p ~/envs/prism-molecular -f env_molecular.yml    # any prefix works
conda activate ~/envs/prism-molecular
export PYTHONNOUSERSITE=1                # keep packages of ~/.local out of the environment
python scripts/download_data.py          # ZINC.pkl -> ./data/molecules, OGB -> ./data/ogb (about 2.4 GB, mostly MolPCBA)
python -m pytest tests                   # unit tests, no GPU or dataset needed
```

`env_molecular.yml` pins the versions of the environment the harness was last run in: Python 3.10, torch 2.2.0+cu118,
dgl 2.1.0+cu118 (from <https://data.dgl.ai/wheels/cu118/repo.html>), torchdata 0.7.1, ogb 1.3.6,
numpy 1.26.4, scipy 1.15.3, and installs `prism` from the repository root (`pip install -e ../..`).
Run all commands from this directory: datasets (`./data`), positional-encoding caches (`./cache`) and
outputs (`./out`) are relative to it and ignored by git. They can be moved with `--data_root`,
`--cache_dir`, `--out_dir` (or `PRISM_DATA_ROOT`, `PRISM_CACHE_DIR`).

Check an installation, on GPU or CPU:

```bash
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json --smoke
```

`--smoke` trains one epoch on 512 / 128 / 128 graphs, with seed 41, without touching the cache, and
writes to `out/.../smoke/`. `--epochs N` (alias `--max_epochs N`) caps the number of epochs of a real run.

## Table cells

There is one config per backbone, dataset and method, named `<Backbone>_<DATASET>_<method>.json`.
A run without `--seed` trains the four seeds **41, 95, 12, 35** one after the other and prints the mean
and standard deviation (`numpy.std`, ddof = 0) of the test metric; `--seed S` trains one seed.

| dataset | backbone | PRiSM preset (`prism.presets`) | k |
|---|---|---|---|
| ZINC | GatedGCN | `zinc_gatedgcn` | 8 |
| ZINC | PNA | `zinc_pna` | 8 |
| MolTox21 | GatedGCN | `moltox21_gatedgcn` | 3 |
| MolTox21 | PNA | `moltox21_pna` | 16 |
| MolPCBA | GatedGCN | `molpcba_gatedgcn` | 3 |
| MolPCBA | PNA | `molpcba_pna` | 16 |

Two PRiSM cells also have an alternative configuration with its own preset, shipped as an extra config:
`GatedGCN_ZINC_PRiSM_heat_t2.json` (preset `zinc_gatedgcn_heat_t2`, heat weighting with t = 2) and
`GatedGCN_MOLPCBA_PRiSM_plain.json` (preset `molpcba_gatedgcn_plain`, without eigenvalue weighting).

`LapPE` configs exist for every cell except PNA / ZINC; they are not a row of the paper table.
`configs/PNA_MOLPCBA_SignNet.json` is reported as OOM in the paper table; a 10-epoch run of this config completed on a 48 GB A40 (about 590 s per epoch), so it is not known to run out of memory on that card.

One command per cell of the table. ZINC uses `main_ZINC_graph_regression.py`, MolTox21 and MolPCBA use
`main_OGBMOL_graph_classification.py`; add `--seed S` to train one seed only (see "Cost and good practice"):

```bash
# ZINC / GatedGCN
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_SignNet.json
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_MAP.json
python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_OAP.json
# ZINC / PNA
python main_ZINC_graph_regression.py --config configs/PNA_ZINC_PRiSM.json
python main_ZINC_graph_regression.py --config configs/PNA_ZINC_SignNet.json
python main_ZINC_graph_regression.py --config configs/PNA_ZINC_MAP.json
python main_ZINC_graph_regression.py --config configs/PNA_ZINC_OAP.json
# MOLTOX21 / GatedGCN
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLTOX21_PRiSM.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLTOX21_SignNet.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLTOX21_MAP.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLTOX21_OAP.json
# MOLTOX21 / PNA
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLTOX21_PRiSM.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLTOX21_SignNet.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLTOX21_MAP.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLTOX21_OAP.json
# MOLPCBA / GatedGCN
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLPCBA_PRiSM.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLPCBA_SignNet.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLPCBA_MAP.json
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLPCBA_OAP.json
# MOLPCBA / PNA
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_PRiSM.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_SignNet.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_MAP.json
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_OAP.json
```

Every config in `configs/` can be run this way: the 24 cells, 5 LapPE configs and the 2 alternative PRiSM configs.

## What a run does

1. The dataset is loaded (ZINC from the benchmarking-gnns pickle, OGB with the scaffold split; graphs with
   at most 5 nodes are dropped from every OGB split, as in the harness of Ma et al.).
2. The positional encoding of every graph is computed once with `prism` (float64 numerics, stored as
   float32, `k` columns, zero padded for small graphs) and cached in `./cache/<dataset>_<method>/k=<k>_<hash>.pkl`.
   The hash covers the full method configuration and `prism.__version__`, so a cache is never reused for a
   different preset or k. Writing is atomic.
3. A model is trained per seed with Adam and `ReduceLROnPlateau` on the validation loss, until the learning rate
   drops below `min_lr`, `epochs` is reached or `max_time` hours have passed (early stopping is off unless
   `--early_stopping_patience` is given). ZINC reports the test MAE of the last epoch; the OGB tasks report the
   test metric at the epoch of best validation metric.

How the positional encoding enters the model (`net_params.pe_init`):

| `pe_init` | positional encoding | in the model |
|---|---|---|
| `prism` | `PRiSM.from_preset(net_params.prism_preset)` | masked GIN network, concatenated with the node embedding, projected |
| `map`, `oap` | `MAP.from_profile('gnn')`, `OAP.from_profile('gnn')` | as `prism` |
| `signnet` | eigenvectors of the normalized Laplacian, no canonicalization | masked GIN network `enc(x) + enc(-x)`, then as `prism` |
| `lap_pe` | as `signnet` | random sign per column and batch during training, added to the node embedding |

Any hyperparameter can be overridden on the command line, for example `--pe_init map --pos_enc_dim 8`,
`--hidden_dim 64`, `--batch_size 64`; `python main_ZINC_graph_regression.py --help` lists them.

## Cost and good practice

* Run one seed per process (`--seed S`) and one process per GPU. PNA on MolPCBA allocates 30 to 45 GB during
  its first batches, so two such processes do not fit on one 48 GB card, and a seed takes tens of hours.
* Build the positional-encoding cache once, on CPU, before launching the seeds:
  `python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_PRiSM.json --build_pe_only --pe_workers 4`.
  Serially this takes roughly two hours for MolPCBA with k = 16; `--pe_workers` splits it over processes and does
  not change the result.
* `CUDA_VISIBLE_DEVICES` set by the scheduler is respected; `--gpu_id` only sets it when it is unset.
* Per-run files are written to `out_dir`: `configs/` (parameters of the run), `results/result_*.txt` (final
  metrics, the lines `TEST MAE`, `TEST AUC` or `TEST AVG PRECISION`) and `results/final_*.json` (all seeds).
  No checkpoints are saved.

## Layout

```
main_ZINC_graph_regression.py            entry point, ZINC
main_OGBMOL_graph_classification.py      entry point, MolTox21 and MolPCBA
configs/                                 one json per cell and method
graphdata/                               datasets and the prism-based positional-encoding pipeline
nets/, layers/                           GatedGCN, PNA, sign-invariant network
train/                                   one epoch of training / evaluation
utils/                                   command line, device and seeding
scripts/download_data.py                 data download
tests/                                   unit tests
```

## Acknowledgments

The training harness follows the code of Ma, Wang, Lim, Jegelka and Wang, *A Canonicalization Perspective
on Invariant and Equivariant Learning* (NeurIPS 2024, <https://github.com/PKU-ML/canonicalization>), which builds on
SignNet (Lim et al., <https://github.com/cptq/SignNet-BasisNet>), gnn-lspe (Dwivedi et al.,
<https://github.com/vijaydwivedi75/gnn-lspe>) and benchmarking-gnns (Dwivedi et al.,
<https://github.com/graphdeeplearning/benchmarking-gnns>). The datasets are ZINC 12K (Irwin et al.; the
benchmarking-gnns subset) and the OGB molecular datasets (Hu et al.). Licences: `THIRD_PARTY_NOTICES.md`.
