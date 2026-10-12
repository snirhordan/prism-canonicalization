"""Positional-encoding types of the transformer / BREC harness.

A ``pe_type`` string selects (i) which canonicalizer produces the Laplacian
eigenvectors, (ii) the architecture of the eigenvector embedding and (iii) how
the canonical vertex rank enters the model.  Canonicalization itself is always
delegated to the ``prism`` package:

* PRiSM variants use :class:`prism.PRiSM` through a named preset
  (``transformer_refined``, ``brec``, ``brec_qr_refined``, ``brec_indrefine``);
* MAP and OAP use :class:`prism.baselines.MAP` / :class:`prism.baselines.OAP`
  with the ``brec`` profile.

====================  ==========  =====  =====  ===========  =====================
``pe_type``           canonical.  eps    rho    rank         notes
====================  ==========  =====  =====  ===========  =====================
LPE                   none        yes    full   none         random sign flips
CanonLPE              PRiSM       yes    full   none
CanonLPE_noeps        PRiSM       no     full   none
CanonLPE_min          PRiSM       no     min    none
CanonLPE_min_sinid    PRiSM       no     min    sinusoidal   canonical vertex rank
CanonLPE_min_rope     PRiSM       no     min    RoPE         canonical vertex rank
QRRefinedLPE          PRiSM (QR)  yes    full   none         ``brec_qr_refined``
IndRefineNauty        PRiSM (IR)  no     full   none         ``brec_indrefine``, needs pynauty
MAP, MAP_min          MAP         no     full/min  none
OAP, OAP_min          OAP         no     full/min  none
====================  ==========  =====  =====  ===========  =====================

``eps`` is the learnable eigenvalue perturbation of the position-aware
embedding, ``rho`` the size of the set-embedding readout ("min" is a single
linear layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PEType:
    name: str
    canon: Optional[str]        # None, "prism", "prism_qr", "prism_indrefine", "map", "oap"
    eps: bool                   # learnable eigenvalue perturbation
    minimal_rho: bool           # single-layer rho
    rank: Optional[str] = None  # None, "sinusoidal", "rope"


PE_TYPES = {p.name: p for p in (
    PEType("LPE", None, eps=True, minimal_rho=False),
    PEType("CanonLPE", "prism", eps=True, minimal_rho=False),
    PEType("CanonLPE_noeps", "prism", eps=False, minimal_rho=False),
    PEType("CanonLPE_min", "prism", eps=False, minimal_rho=True),
    PEType("CanonLPE_min_sinid", "prism", eps=False, minimal_rho=True, rank="sinusoidal"),
    PEType("CanonLPE_min_rope", "prism", eps=False, minimal_rho=True, rank="rope"),
    PEType("QRRefinedLPE", "prism_qr", eps=True, minimal_rho=False),
    PEType("IndRefineNauty", "prism_indrefine", eps=False, minimal_rho=False),
    PEType("MAP", "map", eps=False, minimal_rho=False),
    PEType("MAP_min", "map", eps=False, minimal_rho=True),
    PEType("OAP", "oap", eps=False, minimal_rho=False),
    PEType("OAP_min", "oap", eps=False, minimal_rho=True),
)}


def resolve_pe_type(name: str) -> PEType:
    try:
        return PE_TYPES[name]
    except KeyError:
        raise ValueError(
            f"unsupported pe_type {name!r}; available: {sorted(PE_TYPES)}") from None


def prism_preset_for(pe: PEType, refine_ordering: bool, override: Optional[str] = None) -> str:
    """Name of the :mod:`prism` preset used by a PRiSM ``pe_type``.

    ``override`` (config key ``prism_preset``) takes precedence.  Otherwise the
    transformer harness uses ``transformer_refined`` when sequential ordering
    refinement is on and ``brec`` when it is off; ``QRRefinedLPE`` always uses
    ``brec_qr_refined`` and ``IndRefineNauty`` ``brec_indrefine``.
    """
    if override:
        return override
    if pe.canon == "prism_qr":
        return "brec_qr_refined"
    if pe.canon == "prism_indrefine":
        return "brec_indrefine"
    return "transformer_refined" if refine_ordering else "brec"


def make_canonicalizer(pe_type: str, refine_ordering: bool = False,
                       prism_preset: Optional[str] = None):
    """Return the canonicalizer object of a ``pe_type`` (``None`` for LPE)."""
    pe = resolve_pe_type(pe_type)
    if pe.canon is None:
        return None
    if pe.canon in ("prism", "prism_qr", "prism_indrefine"):
        from prism import PRiSM

        return PRiSM.from_preset(prism_preset_for(pe, refine_ordering, prism_preset))
    if pe.canon == "map":
        from prism.baselines import MAP

        return MAP.from_profile("brec")
    if pe.canon == "oap":
        from prism.baselines import OAP

        return OAP.from_profile("brec")
    raise ValueError(pe.canon)  # pragma: no cover


def describe_canonicalizer(canonicalizer) -> Optional[dict]:
    """JSON-friendly description of a canonicalizer, for run logs."""
    if canonicalizer is None:
        return None
    if hasattr(canonicalizer, "get_config"):
        return {"class": type(canonicalizer).__name__, **canonicalizer.get_config()}
    return {"class": type(canonicalizer).__name__, **{
        k: v for k, v in vars(canonicalizer).items() if not k.startswith("_")}}
