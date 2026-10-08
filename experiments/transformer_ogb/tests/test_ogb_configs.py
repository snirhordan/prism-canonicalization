import os

import pytest
from hydra import compose, initialize_config_dir

from wl_transformers import PE_TYPES, make_canonicalizer

FT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fine-tuning"))
CONFIGS = os.path.join(FT, "configs")

# experiment name -> (base config, dataset_name, pe_type, refine_ordering, lr)
ROWS = {
    "molhiv_prism": ("ogbg", "ogbg-molhiv", "CanonLPE_min_rope", True, 1e-4),
    "molhiv_lpe": ("ogbg", "ogbg-molhiv", "LPE", False, 1e-4),
    "molhiv_map": ("ogbg", "ogbg-molhiv", "MAP_min", False, 1e-4),
    "molhiv_oap": ("ogbg", "ogbg-molhiv", "OAP_min", False, 1e-4),
    "molclintox_prism": ("ogbg", "ogbg-molclintox", "CanonLPE_min_sinid", True, 1e-4),
    "molclintox_lpe": ("ogbg", "ogbg-molclintox", "LPE", False, 1e-4),
    "molclintox_map": ("ogbg", "ogbg-molclintox", "MAP_min", False, 1e-4),
    "molclintox_oap": ("ogbg", "ogbg-molclintox", "OAP_min", False, 1e-4),
    "molbbbp_prism": ("ogbg", "ogbg-molbbbp", "CanonLPE_min", True, 1e-4),
    "molbbbp_lpe": ("ogbg", "ogbg-molbbbp", "LPE", False, 1e-4),
    "molbbbp_map": ("ogbg", "ogbg-molbbbp", "MAP_min", False, 1e-4),
    "molbbbp_oap": ("ogbg", "ogbg-molbbbp", "OAP_min", False, 1e-4),
    "alchemy_prism": ("alchemy", None, "CanonLPE_min_rope", True, 1e-4),
    "alchemy_lpe": ("alchemy", None, "LPE", False, 1e-4),
    "alchemy_map": ("alchemy", None, "MAP_min", False, 1e-3),
    "alchemy_oap": ("alchemy", None, "OAP_min", False, 1e-3),
}


def test_every_experiment_file_is_covered():
    files = {f[:-5] for f in os.listdir(os.path.join(CONFIGS, "experiment")) if f.endswith(".yaml")}
    assert files == set(ROWS)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_experiment_composes(name):
    base, dataset, pe_type, refine, lr = ROWS[name]
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name=base, overrides=[f"+experiment={name}"])
    assert cfg.pe_type == pe_type and cfg.pe_type in PE_TYPES
    assert bool(cfg.refine_ordering) is refine
    assert float(cfg.lr) == lr
    if dataset is not None:
        assert cfg.dataset_name == dataset
    assert cfg.wandb_project is None
    make_canonicalizer(cfg.pe_type, cfg.refine_ordering, cfg.prism_preset)


def test_alchemy_split_files():
    sizes = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(FT, f"{split}_al_10.index")) as fh:
            idx = [int(i) for i in fh.read().strip().split(",")]
        sizes[split] = len(idx)
        assert 0 <= min(idx) and max(idx) < 202579  # indices into alchemy_full
    assert sizes == {"train": 10000, "val": 1000, "test": 1000}
