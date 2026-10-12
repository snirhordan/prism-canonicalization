import importlib.util
import os

import pytest
from hydra import compose, initialize_config_dir

from wl_transformers import count_parameters, make_canonicalizer

EXPRESSIVITY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "expressivity"))

spec = importlib.util.spec_from_file_location(
    "brec_benchmark", os.path.join(EXPRESSIVITY, "brec_benchmark.py"))
brec_benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brec_benchmark)

# experiment -> (pe_type, PRiSM preset or baseline class name, parameters)
ROWS = {
    "brec_prism": ("IndRefineNauty", "brec_indrefine", 12656),
    "brec_prism_canonlpe": ("CanonLPE_noeps", "brec", 12656),
    "brec_prism_qr": ("QRRefinedLPE", "brec_qr_refined", 12664),
    "brec_map": ("MAP", "MAP", 12656),
    "brec_oap": ("OAP", "OAP", 12656),
}


def test_every_experiment_file_is_covered():
    files = {f[:-5] for f in os.listdir(os.path.join(EXPRESSIVITY, "experiment"))
             if f.endswith(".yaml") and f != "brec_matched.yaml"}
    assert files == set(ROWS)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_matched_architecture(name):
    pe_type, canon, n_params = ROWS[name]
    with initialize_config_dir(config_dir=EXPRESSIVITY, version_base=None):
        cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
    assert cfg.pe_type == pe_type and cfg.backbone == "gnn"
    assert cfg.wandb_project is None
    assert list(cfg.parts) == ["Basic", "Regular", "Extension", "CFI"]
    model = brec_benchmark.get_model(cfg, "cpu")
    assert count_parameters(model) == n_params
    c = make_canonicalizer(cfg.pe_type, cfg.refine_ordering, cfg.prism_preset)
    if canon in ("MAP", "OAP"):
        assert type(c).__name__ == canon
    else:
        from prism import PRiSM

        assert c.get_config() == PRiSM.from_preset(canon).get_config()


def test_brec_prism_matches_the_indrefine_launch():
    """Hyperparameters of the launch behind the PRiSM row (backbone gnn, num_vecs 8,
    embed_dim 32, 2 layers, 2 heads, mean pooling, 20 epochs, lr 1e-4)."""
    with initialize_config_dir(config_dir=EXPRESSIVITY, version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=brec_prism"])
    assert (cfg.backbone, cfg.pe_type, cfg.num_vecs, cfg.embed_dim, cfg.num_layers,
            cfg.num_heads, cfg.pooling) == ("gnn", "IndRefineNauty", 8, 32, 2, 2, "mean")
    assert cfg.epochs == 20 and cfg.batch_size == 16 and cfg.output_dim == 16
    assert brec_benchmark.LEARNING_RATE == 1e-4
    c = make_canonicalizer(cfg.pe_type, cfg.refine_ordering, cfg.prism_preset)
    assert c.ordering == "indrefine_nauty" and c.needs_adjacency


def test_part_sizes():
    sizes = {k: hi - lo for k, (lo, hi) in brec_benchmark.PART_DICT.items()}
    assert sizes == {"Basic": 60, "Regular": 100, "Extension": 100, "CFI": 100,
                     "4-Vertex_Condition": 20, "Distance_Regular": 20}
