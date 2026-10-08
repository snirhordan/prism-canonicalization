#!/usr/bin/env python3
"""Eigenvalue multiplicity statistics (table ``tab:multiplicity-stats``).

For each of five molecular datasets, sample 500 graphs (``random.seed(42)``,
all splits of ZINC-12K; all graphs of the OGB datasets), compute the
eigenvalues of the normalized adjacency D^-1/2 A D^-1/2 in float64 with
``numpy.linalg.eigvalsh`` and group them into multiplicity classes: sorted
eigenvalues within absolute distance 1e-6 of the first member of a class are
equal.  Per graph the number of *distinct* eigenvalues of multiplicity 1, 2, 3
and >= 4 is counted; the table lists their means over the sample and the
fraction of graphs in which all eigenvalues are distinct.

    python multiplicity_stats.py --data-dir ./data --out results/multiplicity_stats.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from prism import operators  # noqa: E402

DATASETS = ["ZINC-12K", "ogbg-molhiv", "ogbg-moltox21", "ogbg-molclintox", "ogbg-molbbbp"]
FIELDS = ["dataset", "sampled_graphs", "mean_nodes", "mean_m1", "mean_m2", "mean_m3",
          "mean_m_ge4", "frac_simple_spectrum"]


def count_multiplicities(eigenvalues: np.ndarray, tol: float = 1e-6):
    """Numbers of distinct eigenvalues of multiplicity 1, 2, 3 and >= 4."""
    eig = np.sort(eigenvalues)
    counts = [0, 0, 0, 0]
    i = 0
    while i < len(eig):
        j = i + 1
        while j < len(eig) and eig[j] - eig[i] < tol:
            j += 1
        counts[min(j - i, 4) - 1] += 1
        i = j
    return tuple(counts)


def dataset_row(name: str, graphs, sample_size: int, seed: int, tol: float) -> dict:
    """Statistics over a random sample of ``graphs`` (indexable, yields adjacency on demand)."""
    indices = list(range(len(graphs)))
    random.seed(seed)
    if len(indices) > sample_size:
        indices = random.sample(indices, sample_size)
    rows, skipped = [], 0
    for i in indices:
        A = graphs[i]
        if A.shape[0] == 0:
            skipped += 1
            continue
        lam = np.linalg.eigvalsh(operators.normalized_adjacency(A))
        rows.append((A.shape[0],) + count_multiplicities(lam, tol))
    if skipped:
        print(f"  {name}: skipped {skipped} empty graphs", file=sys.stderr)
    r = np.array(rows, dtype=np.float64)
    simple = (r[:, 2] == 0) & (r[:, 3] == 0) & (r[:, 4] == 0)
    return {
        "dataset": name,
        "sampled_graphs": int(r.shape[0]),
        "mean_nodes": round(float(r[:, 0].mean()), 2),
        "mean_m1": round(float(r[:, 1].mean()), 3),
        "mean_m2": round(float(r[:, 2].mean()), 3),
        "mean_m3": round(float(r[:, 3].mean()), 3),
        "mean_m_ge4": round(float(r[:, 4].mean()), 3),
        "frac_simple_spectrum": round(float(simple.mean()), 4),
    }


class _Graphs:
    """Indexable view of a PyG dataset that returns adjacency matrices."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        return common.dense_adjacency(data.edge_index.numpy(), int(data.num_nodes))


def load_graphs(name: str, data_dir: str) -> _Graphs:
    if name == "ZINC-12K":
        return _Graphs(common.ZincAll(data_dir))
    return _Graphs(common.load_ogb(name, data_dir))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./data", help="dataset root (default ./data)")
    p.add_argument("--out", default="results/multiplicity_stats.csv", help="output .csv or .json")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-6, help="absolute eigenvalue tolerance")
    args = p.parse_args(argv)

    print(f"{'dataset':<18}{'sampled':>8}{'n':>8}{'#m=1':>8}{'#m=2':>8}{'#m=3':>8}"
          f"{'#m>=4':>8}{'simple':>9}")
    rows = []
    for name in args.datasets:
        row = dataset_row(name, load_graphs(name, args.data_dir), args.sample_size,
                          args.seed, args.tol)
        rows.append(row)
        print(f"{row['dataset']:<18}{row['sampled_graphs']:>8}{common.half_up(row['mean_nodes']):>8}"
              f"{common.half_up(row['mean_m1'], 2):>8}{common.half_up(row['mean_m2'], 2):>8}{common.half_up(row['mean_m3'], 2):>8}"
              f"{common.half_up(row['mean_m_ge4'], 2):>8}{common.percent(row['frac_simple_spectrum']):>8}%",
              flush=True)
    common.write_rows(args.out, rows, FIELDS)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
