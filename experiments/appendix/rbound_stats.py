#!/usr/bin/env python3
"""Row-count bound r = n - L of the Z_2 system solved by PRiSM (table ``tab:rbound-stats``).

For every graph of the validation split, take the full eigendecomposition
(k = n) of the normalized Laplacian I - D^-1/2 A D^-1/2 and run the Partition
stage of PRiSM (``prism.gf2.partition_stage``) on the matrix of eigenvectors.
Partition groups the vertices by the absolute-value signature |U_v| rounded to
``--decimals`` places (tolerance 1e-6 by default).  With L classes the system
solved by Solve has at most r = n - L rows.

Per dataset the table reports the number of validation graphs, the mean number
of nodes, the fraction with r = 0 (FastSign applies), the fraction with exactly
one non-singleton class ("1 block"), the median r, the fraction with r <= 4 and
the maximum r.

    python rbound_stats.py --data-dir ./data --out results/rbound_stats.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from prism import gf2, operators  # noqa: E402

# Dataset ids (ZINC-12K, then the OGB-Mol names); the names of the paper are in README.md.
DATASETS = [
    "ZINC-12K", "ogbg-molhiv", "ogbg-moltox21", "ogbg-molclintox", "ogbg-molbbbp",
    "ogbg-molbace", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-molmuv",
    "ogbg-moltoxcast", "ogbg-molpcba",
]
FIELDS = ["dataset", "val_graphs", "mean_nodes", "frac_r0", "frac_one_block",
          "median_r", "frac_r_le_4", "max_r"]


def partition_stats(A: np.ndarray, decimals: int):
    """``(n, L, number of non-singleton classes)`` of the Partition of a graph."""
    n = A.shape[0]
    _, U = operators.eigendecompose(A, "norm_lap")
    classes = gf2.partition_stage(U, decimals)
    sizes = np.bincount(classes)
    return n, int(sizes.size), int(np.sum(sizes > 1))


def dataset_row(name: str, adjacencies, decimals: int) -> dict:
    """Aggregate statistics of one dataset (an iterable of adjacency matrices)."""
    n_nodes, r_bound, n_blocks = [], [], []
    skipped = 0
    for A in adjacencies:
        if A.shape[0] == 0:
            skipped += 1
            continue
        n, n_classes, nonsingleton = partition_stats(A, decimals)
        n_nodes.append(n)
        r_bound.append(n - n_classes)
        n_blocks.append(nonsingleton)
    if skipped:
        print(f"  {name}: skipped {skipped} empty graphs", file=sys.stderr)
    r_bound, n_blocks = np.array(r_bound), np.array(n_blocks)
    return {
        "dataset": name,
        "val_graphs": int(r_bound.size),
        "mean_nodes": round(float(np.mean(n_nodes)), 2),
        "frac_r0": round(float(np.mean(r_bound == 0)), 4),
        "frac_one_block": round(float(np.mean(n_blocks == 1)), 4),
        "median_r": round(float(np.median(r_bound)), 2),
        "frac_r_le_4": round(float(np.mean(r_bound <= 4)), 4),
        "max_r": int(r_bound.max()),
    }


def validation_adjacencies(name: str, data_dir: str, max_graphs=None):
    """Adjacency matrices of the validation split of a dataset."""
    if name == "ZINC-12K":
        ds = common.load_zinc(data_dir, "val")
        idx = range(len(ds))
    else:
        ds = common.load_ogb(name, data_dir)
        idx = common.ogb_split(ds, "valid")
    if max_graphs is not None:
        idx = list(idx)[:max_graphs]
    return common.pyg_adjacencies(ds, idx)


def format_row(row: dict) -> str:
    return (f"{row['dataset']:<18}{row['val_graphs']:>8}{common.half_up(row['mean_nodes']):>8}"
            f"{common.percent(row['frac_r0']):>8}%{common.percent(row['frac_one_block']):>8}%"
            f"{row['median_r']:>8.1f}{common.percent(row['frac_r_le_4']):>8}%{row['max_r']:>6}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./data", help="dataset root (default ./data)")
    p.add_argument("--out", default="results/rbound_stats.csv", help="output .csv or .json")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--decimals", type=int, default=6,
                   help="Partition rounds |U| to this many decimals (tolerance 10^-decimals)")
    p.add_argument("--max-graphs", type=int, default=None,
                   help="only the first N validation graphs of every dataset (smoke test)")
    args = p.parse_args(argv)

    header = (f"{'dataset':<18}{'|val|':>8}{'n':>8}{'r=0':>9}{'1 block':>9}"
              f"{'med r':>8}{'r<=4':>9}{'max r':>6}")
    print(header)
    rows = []
    for name in args.datasets:
        row = dataset_row(name, validation_adjacencies(name, args.data_dir, args.max_graphs),
                          args.decimals)
        rows.append(row)
        print(format_row(row), flush=True)
    common.write_rows(args.out, rows, FIELDS)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
