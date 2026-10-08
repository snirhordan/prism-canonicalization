"""The configs: one per (backbone, dataset, method), consistent with the PRiSM presets."""

import glob
import json
import os
import re

import pytest

import prism

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
FILES = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))
NAME = re.compile(r"^(GatedGCN|PNA)_(ZINC|MOLTOX21|MOLPCBA)_(PRiSM|SignNet|MAP|OAP|LapPE|PRiSM_heat_t2|PRiSM_plain)$")
DATASET = {"ZINC": "ZINC", "MOLTOX21": "OGBG-MOLTOX21", "MOLPCBA": "OGBG-MOLPCBA"}
KNOWN_NET_KEYS = {
    "L", "hidden_dim", "out_dim", "residual", "edge_feat", "readout", "in_feat_dropout", "dropout", "dropout_2",
    "graph_norm", "batch_norm", "aggregators", "scalers", "towers", "divide_input_first", "divide_input_last",
    "edge_dim", "pretrans_layers", "posttrans_layers", "pe_init", "prism_preset", "pos_enc_dim", "sign_inv_net",
    "sign_inv_layers", "phi_out_dim", "sign_inv_activation",
}
PE_OF_METHOD = {"PRiSM": "prism", "PRiSM_heat_t2": "prism", "PRiSM_plain": "prism", "SignNet": "signnet",
                "MAP": "map", "OAP": "oap", "LapPE": "lap_pe"}


def load(path):
    with open(path) as f:
        return json.load(f)


def test_every_cell_has_every_method():
    names = {os.path.splitext(os.path.basename(f))[0] for f in FILES}
    for backbone in ("GatedGCN", "PNA"):
        for dataset in DATASET:
            for method in ("PRiSM", "SignNet", "MAP", "OAP"):
                assert "%s_%s_%s" % (backbone, dataset, method) in names


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.basename(p))
def test_config_is_consistent(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = NAME.match(name)
    assert m, "config name must be <Backbone>_<DATASET>_<method>"
    backbone, dataset, method = m.groups()
    cfg = load(path)
    assert cfg["model"] == backbone
    assert cfg["dataset"] == DATASET[dataset]
    assert cfg["out_dir"] == "out/%s/" % name
    net = cfg["net_params"]
    assert net["pe_init"] == PE_OF_METHOD[method]
    assert set(net) <= KNOWN_NET_KEYS
    if net["pe_init"] == "prism":
        assert net["prism_preset"] in prism.PRESETS
        assert net["pos_enc_dim"] == prism.PRESETS[net["prism_preset"]]["k"]
    else:
        assert "prism_preset" not in net
    if net["pe_init"] == "lap_pe":
        assert "sign_inv_net" not in net
    else:
        assert net["sign_inv_net"] == "masked_gin"
        assert net["phi_out_dim"] > 0 and net["sign_inv_layers"] >= 2
    for key in ("epochs", "batch_size", "init_lr", "lr_reduce_factor", "lr_schedule_patience", "min_lr",
                "weight_decay", "max_time"):
        assert key in cfg["params"]


def test_no_absolute_paths():
    for path in FILES:
        text = open(path).read()
        assert "/home/" not in text and "/Users/" not in text


def test_prism_cells_use_the_paper_presets():
    expected = {
        "GatedGCN_ZINC_PRiSM": "zinc_gatedgcn", "PNA_ZINC_PRiSM": "zinc_pna",
        "GatedGCN_MOLTOX21_PRiSM": "moltox21_gatedgcn", "PNA_MOLTOX21_PRiSM": "moltox21_pna",
        "GatedGCN_MOLPCBA_PRiSM": "molpcba_gatedgcn", "PNA_MOLPCBA_PRiSM": "molpcba_pna",
        "GatedGCN_ZINC_PRiSM_heat_t2": "zinc_gatedgcn_heat_t2", "GatedGCN_MOLPCBA_PRiSM_plain": "molpcba_gatedgcn_plain",
    }
    for name, preset in expected.items():
        assert load(os.path.join(CONFIG_DIR, name + ".json"))["net_params"]["prism_preset"] == preset
