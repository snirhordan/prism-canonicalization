"""Command line overrides."""

import json
import os

from utils.cli import SMOKE_SIZES, SEEDS, build_parser, load_config

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "PNA_ZINC_PRiSM.json")


def parse(task, *argv):
    args = build_parser(task).parse_args(["--config", CONFIG] + list(argv))
    return args, load_config(args, task)


def test_defaults_run_four_seeds():
    _, (config, params, net, seeds) = parse("zinc")
    assert seeds == list(SEEDS) == [41, 95, 12, 35]
    assert config["out_dir"] == "out/PNA_ZINC_PRiSM/"
    assert net["pe_init"] == "prism" and net["prism_preset"] == "zinc_pna"


def test_overrides():
    _, (config, params, net, seeds) = parse("zinc", "--seed", "7", "--epochs", "3", "--hidden_dim", "20", "--residual",
                                            "False", "--in_feat_dropout", "0.3", "--max_epochs", "5")
    assert seeds == [7] and params["epochs"] == 5 and net["hidden_dim"] == 20
    assert net["residual"] is False and net["in_feat_dropout"] == 0.3


def test_smoke():
    args, (config, params, net, seeds) = parse("zinc", "--smoke")
    assert args.smoke and params["epochs"] == 1 and seeds == [41]
    assert config["out_dir"].rstrip(os.sep).endswith("smoke")
    assert len(SMOKE_SIZES) == 3


def test_other_pe_drops_preset():
    _, (config, params, net, seeds) = parse("zinc", "--pe_init", "map")
    assert net["pe_init"] == "map" and "prism_preset" not in net
