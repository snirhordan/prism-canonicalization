"""Every config builds its network and trains one step on a synthetic batch (CPU)."""

import glob
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")
pytest.importorskip("ogb")

from graphdata.molecules import MoleculeDataset  # noqa: E402
from graphdata.ogb_mol import OGBMOLDataset  # noqa: E402
from graphdata.positional_encoding import add_pos_enc  # noqa: E402
from helpers import fake_dataset  # noqa: E402
from train.pe_input import get_batch_pos_enc  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
FILES = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.splitext(os.path.basename(p))[0])
def test_forward_backward(path, tmp_path):
    cfg = json.load(open(path))
    net_params = cfg["net_params"]
    zinc = cfg["dataset"] == "ZINC"
    ds = fake_dataset("zinc" if zinc else "ogb", n_graphs=6)
    add_pos_enc(ds, net_params["pe_init"], net_params.get("prism_preset"), net_params["pos_enc_dim"],
                cache_dir=str(tmp_path), use_cache=False)
    samples = [(g, torch.tensor(0.5) if zinc else torch.zeros(128 if "PCBA" in cfg["dataset"] else 12))
               for g in ds.train.graph_lists]
    collate = MoleculeDataset.collate if zinc else OGBMOLDataset.collate
    batch, labels, snorm_n = collate(None, samples)
    net_params["device"] = torch.device("cpu")
    if zinc:
        from nets.ZINC_graph_regression.load_net import gnn_model
        net_params["num_atom_type"], net_params["num_bond_type"] = 28, 4
        n_out = 1
    else:
        from nets.OGBMOL_graph_classification.load_net import gnn_model
        n_out = 128 if "PCBA" in cfg["dataset"] else 12
        net_params["n_classes"] = n_out
    if cfg["model"] == "PNA":
        D = torch.cat([g.in_degrees().float() for g in ds.train.graph_lists])
        net_params["avg_d"] = dict(lin=torch.mean(D), exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))
    model = gnn_model(cfg["model"], net_params)
    model.train()
    pe = get_batch_pos_enc(model, batch, torch.device("cpu"), training=True)
    assert pe.shape == (batch.num_nodes(), net_params["pos_enc_dim"])
    out = model(batch, batch.ndata["feat"], pe, batch.edata["feat"], snorm_n)
    assert out.shape == (len(samples), n_out)
    loss = model.loss(out, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    model.eval()
    pe = get_batch_pos_enc(model, batch, torch.device("cpu"), training=False)
    assert torch.isfinite(model(batch, batch.ndata["feat"], pe, batch.edata["feat"], snorm_n)).all()
