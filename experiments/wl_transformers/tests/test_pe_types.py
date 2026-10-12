import pytest

from wl_transformers import PE_TYPES, make_canonicalizer, prism_preset_for, resolve_pe_type


def test_registry_contains_every_reported_type():
    for name in ("LPE", "CanonLPE_min_rope", "CanonLPE_min_sinid", "CanonLPE_min",
                 "MAP_min", "OAP_min", "CanonLPE_noeps", "MAP", "OAP", "QRRefinedLPE",
                 "IndRefineNauty"):
        assert name in PE_TYPES


def test_unsupported_type_raises():
    with pytest.raises(ValueError, match="unsupported pe_type"):
        resolve_pe_type("SPE")


@pytest.mark.parametrize("pe_type,refine,preset", [
    ("CanonLPE_min_rope", True, "transformer_refined"),
    ("CanonLPE_min_sinid", True, "transformer_refined"),
    ("CanonLPE_min", True, "transformer_refined"),
    ("CanonLPE_noeps", False, "brec"),
    ("QRRefinedLPE", False, "brec_qr_refined"),
    ("QRRefinedLPE", True, "brec_qr_refined"),
    ("IndRefineNauty", False, "brec_indrefine"),
    ("IndRefineNauty", True, "brec_indrefine"),
])
def test_prism_preset_selection(pe_type, refine, preset):
    pe = resolve_pe_type(pe_type)
    assert prism_preset_for(pe, refine) == preset
    assert prism_preset_for(pe, refine, override="brec") == "brec"


def test_canonicalizers_come_from_prism():
    from prism import PRESETS, PRiSM
    from prism.baselines import MAP, OAP

    assert make_canonicalizer("LPE") is None
    c = make_canonicalizer("CanonLPE_min_rope", refine_ordering=True)
    assert isinstance(c, PRiSM)
    expected = PRiSM.from_preset("transformer_refined").get_config()
    assert c.get_config() == expected
    assert c.get_config()["refine_ordering"] is True
    assert make_canonicalizer("CanonLPE_min", refine_ordering=False).get_config()["refine_ordering"] is False
    assert isinstance(make_canonicalizer("MAP_min"), MAP)
    assert isinstance(make_canonicalizer("OAP_min"), OAP)
    assert "transformer_refined" in PRESETS


def test_prefix_preset_is_requested_not_derived():
    from prism import PRiSM

    pe = resolve_pe_type("CanonLPE_min_sinid")
    for refine in (False, True):
        assert prism_preset_for(pe, refine) != "transformer_refined_prefix"
    c = make_canonicalizer("CanonLPE_min_sinid", True, "transformer_refined_prefix")
    assert isinstance(c, PRiSM)
    assert c.get_config() == PRiSM.from_preset("transformer_refined_prefix").get_config()
    assert c.get_config()["rank_source"] == "pre_refine"
    assert (pe.eps, pe.minimal_rho, pe.rank) == (False, True, "sinusoidal")


def test_baseline_profiles_are_the_transformer_ones():
    assert make_canonicalizer("MAP_min").use_eig_val is False
    assert make_canonicalizer("OAP_min").use_eig_val is False
    assert make_canonicalizer("OAP_min").eig_decimals == 8


def test_indrefine_nauty_comes_from_prism():
    pytest.importorskip("pynauty")
    from prism import PRiSM

    c = make_canonicalizer("IndRefineNauty")
    assert isinstance(c, PRiSM) and c.needs_adjacency
    assert c.get_config() == PRiSM.from_preset("brec_indrefine").get_config()
    pe = resolve_pe_type("IndRefineNauty")
    assert (pe.eps, pe.minimal_rho, pe.rank) == (False, False, None)
