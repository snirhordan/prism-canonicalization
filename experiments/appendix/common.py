"""Helpers shared by the appendix scripts: dataset access and result files.

Data layout below ``--data-dir`` (created on first use, downloads happen once)::

    <data-dir>/ZINC/   PyG ``ZINC`` (12k subset and the raw pickles of the full set)
    <data-dir>/ogb/    OGB-Mol datasets (``PygGraphPropPredDataset``)

Graphs are returned as dense symmetric 0/1 float64 adjacency matrices.  All
canonicalization in the scripts goes through ``prism`` (``prism.PRiSM`` and
``prism.baselines``); this module only loads graphs and writes tables.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Iterator, List, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# graphs
# ----------------------------------------------------------------------------

def dense_adjacency(edge_index, num_nodes: int) -> np.ndarray:
    """Symmetric 0/1 float64 adjacency matrix of an edge list (2 x E array-like)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    ei = np.asarray(edge_index)
    if ei.size:
        A[ei[0], ei[1]] = 1.0
        A[ei[1], ei[0]] = 1.0
    return A


def pyg_adjacencies(dataset, indices: Iterable[int] = None) -> Iterator[np.ndarray]:
    """Adjacency matrices of the graphs of a PyG dataset (all, or the given indices)."""
    if indices is None:
        indices = range(len(dataset))
    for i in indices:
        data = dataset[int(i)]
        yield dense_adjacency(data.edge_index.numpy(), int(data.num_nodes))


# ----------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------

def load_zinc(data_dir: str, split: str):
    """ZINC-12K (PyG ``ZINC(subset=True)``), split in ``train``, ``val``, ``test``."""
    from torch_geometric.datasets import ZINC

    return ZINC(root=os.path.join(data_dir, "ZINC"), subset=True, split=split)


class ZincAll:
    """Train, validation and test split of ZINC-12K as one indexable dataset."""

    def __init__(self, data_dir: str):
        self._parts = [load_zinc(data_dir, s) for s in ("train", "val", "test")]
        self._ends = np.cumsum([len(p) for p in self._parts])

    def __len__(self) -> int:
        return int(self._ends[-1])

    def __getitem__(self, i: int):
        part = int(np.searchsorted(self._ends, i, side="right"))
        start = 0 if part == 0 else int(self._ends[part - 1])
        return self._parts[part][i - start]


def load_ogb(name: str, data_dir: str):
    """An OGB-Mol dataset as PyG ``PygGraphPropPredDataset`` (downloaded on first use)."""
    from ogb.graphproppred import PygGraphPropPredDataset

    return PygGraphPropPredDataset(name=name, root=os.path.join(data_dir, "ogb"))


def ogb_split(dataset, split: str) -> List[int]:
    """Graph indices of an OGB split (``train``, ``valid`` or ``test``)."""
    return dataset.get_idx_split()[split].tolist()


def zinc_full_train_adjacencies(data_dir: str, n_graphs: int, min_nodes: int = 6) -> List[np.ndarray]:
    """Adjacencies of the first ``n_graphs`` molecules of the *full* ZINC training file.

    ``n_graphs`` counts the molecules read, molecules with fewer than
    ``min_nodes`` atoms are then dropped (so slightly fewer matrices come back).
    The full file (about 220k molecules, the order of the ZINC release) is the
    ``train.pickle`` that PyG downloads next to the 12k subset; it is read here
    directly, and the 12k subset is not used.
    """
    dataset = load_zinc(data_dir, "train")  # triggers the download and extraction of the raw files
    path = os.path.join(dataset.raw_dir, "train.pickle")
    with open(path, "rb") as f:
        molecules = pickle.load(f)
    out = []
    for mol in molecules[:n_graphs]:
        if mol["num_atom"] < min_nodes:
            continue
        out.append((mol["bond_type"] != 0).numpy().astype(np.float64))
    return out


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------

def write_rows(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write a table as CSV (``.csv``) or JSON (``.json``); parent folders are created."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(list(rows), f, indent=2)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def half_up(value: float, digits: int = 1) -> str:
    """``value`` rounded half up to ``digits`` decimals, as text (``11.95 -> '12.0'``)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP))


def percent(fraction: float, digits: int = 1) -> str:
    """``fraction`` in percent, rounded half up (``0.2115 -> '21.2'``)."""
    return half_up(Decimal(repr(float(fraction))) * 100, digits)


def row_multiset_equal(a: np.ndarray, b: np.ndarray, decimals: int, atol: float) -> bool:
    """Are the rows of ``a`` and ``b`` equal as multisets (after rounding to ``decimals``)?"""
    if a.shape != b.shape:
        return False
    ar, br = np.round(a, decimals), np.round(b, decimals)
    ar = ar[np.lexsort(ar[:, ::-1].T)]
    br = br[np.lexsort(br[:, ::-1].T)]
    return bool(np.allclose(ar, br, atol=atol))
