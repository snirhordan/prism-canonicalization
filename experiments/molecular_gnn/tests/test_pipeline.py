"""The two entry points run end to end (config, positional encoding, training, result files) on synthetic data."""

import glob
import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")
pytest.importorskip("ogb")

from graphdata.molecules import MoleculeDataset  # noqa: E402
from graphdata.ogb_mol import OGBMOLDataset  # noqa: E402
from helpers import random_molecule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Split:
    def __init__(self, graphs, labels):
        self.graph_lists = graphs
        self.graph_labels = labels
        self.n_samples = len(graphs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


def make_split(count, task, rng, n_tasks=12):
    graphs = [random_molecule(rng, task=task) for _ in range(count)]
    if task == "zinc":
        labels = [torch.tensor(float(rng.normal())) for _ in range(count)]
    else:
        labels = []
        for _ in range(count):
            y = rng.integers(0, 2, size=n_tasks).astype(np.float32)
            y[rng.random(n_tasks) < 0.2] = np.nan  # unlabeled entries
            labels.append(torch.from_numpy(y))
    return Split(graphs, labels)


class FakeZinc:
    name = "ZINC"
    num_atom_type, num_bond_type = 28, 4

    def __init__(self):
        rng = np.random.default_rng(0)
        self.train, self.val, self.test = (make_split(n, "zinc", rng) for n in (40, 12, 12))

    def subset(self, *sizes):
        pass

    def collate(self, samples):
        return MoleculeDataset.collate(self, samples)


class FakeOgb:
    name = "ogbg-moltox21"

    def __init__(self):
        from ogb.graphproppred import Evaluator

        rng = np.random.default_rng(0)
        self.train, self.val, self.test = (make_split(n, "ogb", rng) for n in (40, 12, 12))
        self.dataset = type("D", (), {"num_tasks": 12})()
        self.evaluator = Evaluator(name=self.name)

    def subset(self, *sizes):
        pass

    def collate(self, samples):
        return OGBMOLDataset.collate(self, samples)


def run_main(module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["main"] + argv)
    module.main()


ZINC_CASES = {
    "GatedGCN_ZINC_PRiSM": ["--hidden_dim", "20", "--phi_out_dim", "20"],
    "PNA_ZINC_PRiSM": ["--hidden_dim", "20", "--phi_out_dim", "20"],
    "GatedGCN_ZINC_LapPE": ["--hidden_dim", "20"],
}
OGB_CASES = {
    "GatedGCN_MOLTOX21_PRiSM": ["--hidden_dim", "24", "--out_dim", "24", "--phi_out_dim", "24"],
    "PNA_MOLTOX21_PRiSM": ["--hidden_dim", "24", "--out_dim", "24", "--phi_out_dim", "24"],
    "GatedGCN_MOLTOX21_LapPE": ["--hidden_dim", "24", "--out_dim", "24"],
}


@pytest.mark.parametrize("config", sorted(ZINC_CASES))
def test_zinc_entry_point(config, tmp_path, monkeypatch):
    import main_ZINC_graph_regression as main

    monkeypatch.setattr(main, "LoadData", lambda name, root: FakeZinc())
    out = str(tmp_path / "out")
    run_main(main, monkeypatch, ["--config", os.path.join(ROOT, "configs", config + ".json"), "--epochs", "1",
                                 "--seed", "41", "--num_workers", "0", "--out_dir", out,
                                 "--cache_dir", str(tmp_path / "cache")] + ZINC_CASES[config])
    results = glob.glob(os.path.join(out, "results", "result_*.txt"))
    assert len(results) == 1 and "TEST MAE" in open(results[0]).read()
    final = json.load(open(glob.glob(os.path.join(out, "results", "final_*.json"))[0]))
    assert final["seeds"] == [41] and np.isfinite(final["mean"])


@pytest.mark.parametrize("config", sorted(OGB_CASES))
def test_ogb_entry_point(config, tmp_path, monkeypatch):
    import main_OGBMOL_graph_classification as main

    monkeypatch.setattr(main, "LoadData", lambda name, root: FakeOgb())
    out = str(tmp_path / "out")
    run_main(main, monkeypatch, ["--config", os.path.join(ROOT, "configs", config + ".json"), "--epochs", "2",
                                 "--seed", "95", "--num_workers", "0", "--out_dir", out,
                                 "--cache_dir", str(tmp_path / "cache")] + OGB_CASES[config])
    results = glob.glob(os.path.join(out, "results", "result_*.txt"))
    assert len(results) == 1 and "TEST AUC" in open(results[0]).read()
    final = json.load(open(glob.glob(os.path.join(out, "results", "final_*.json"))[0]))
    assert final["seeds"] == [95] and np.isfinite(final["mean"])


def test_build_pe_only_writes_cache(tmp_path, monkeypatch):
    import main_ZINC_graph_regression as main

    monkeypatch.setattr(main, "LoadData", lambda name, root: FakeZinc())
    cache = tmp_path / "cache"
    run_main(main, monkeypatch, ["--config", os.path.join(ROOT, "configs", "PNA_ZINC_PRiSM.json"), "--build_pe_only",
                                 "--cache_dir", str(cache), "--out_dir", str(tmp_path / "out"), "--hidden_dim", "20",
                                 "--phi_out_dim", "20"])
    assert len(glob.glob(str(cache / "ZINC_prism_zinc_pna" / "k=8_*.pkl"))) == 1
    assert not glob.glob(str(tmp_path / "out" / "results" / "result_*"))
