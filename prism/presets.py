"""Named configurations of :class:`prism.PRiSM`, one per experimental setting.

``PRESETS[name]`` is the keyword dictionary passed to the constructor;
``PRiSM.from_preset(name)`` builds the instance.  Every preset reproduces the
code path its harness executed (see ``README.md`` for the table of paper
cells).  The tables list the harness name of each variant for reference.

Molecular GNN harness (``pe_init`` of the GNN configs)
    ``spectral_m3``                    zinc_gatedgcn
    ``spectral_m3_heat_t2``            zinc_gatedgcn_heat_t2
    ``spectral_qr_refined_eigval``     zinc_pna, molpcba_gatedgcn, molpcba_pna
    ``spectral_qr_refined``            molpcba_gatedgcn_plain
    ``spectral_m3_inv_sqrt_lap``       moltox21_gatedgcn
    ``spectral_m3_eigval``             moltox21_pna

Transformer / BREC harness (``pe_type`` with ``refine_ordering``)
    ``CanonLPE_*`` + refine            transformer_refined
    ``CanonLPE_min_sinid`` + refine    transformer_refined_prefix (rank_source="pre_refine")
    ``CanonLPE_*``                     brec
    ``QRRefinedLPE``                   brec_qr_refined
    ``IndRefineNauty``                 brec_indrefine (ordering="indrefine_nauty")

Appendix experiments
    permutation-noise and runtime scaling      rebuttal_comb_lap
    ``tab:invariance`` (QR-refined, full)      qr_refined_full
"""

from __future__ import annotations

# Molecular GNN harness: D^-1/2 A D^-1/2, trivial eigenpair dropped, the last
# (largest-eigenvalue) k columns kept, adaptive tolerances.
_GNN = dict(operator="norm_adj", drop_trivial=True, keep="last",
            tol_profile="adaptive")

# Transformer / BREC harness: I - D^-1/2 A D^-1/2, nothing dropped, the first
# (smallest-eigenvalue) k columns kept after canonicalizing k + 3, eigenpairs
# stored in float32, fixed-decimal tolerances.
_TRANSFORMER = dict(operator="norm_lap", drop_trivial=False, keep="first",
                    k=96, scope="window", window_buffer=3,
                    tol_profile="legacy_round", quantize_float32=True)

PRESETS = {
    # ---- molecular GNN (Table 1) ------------------------------------------
    "zinc_gatedgcn": dict(_GNN, k=8, scope="full", mult_mode="hybrid",
                          refine_ordering=False),
    "zinc_gatedgcn_heat_t2": dict(_GNN, k=8, scope="full", mult_mode="hybrid",
                                  refine_ordering=False, weight="heat",
                                  weight_param=2.0),
    "zinc_pna": dict(_GNN, k=8, scope="full", mult_mode="qr",
                     refine_ordering=True, weight="eigval"),
    "moltox21_gatedgcn": dict(_GNN, k=3, scope="window", mult_mode="hybrid",
                              refine_ordering=False, weight="inv_sqrt_lap",
                              weight_param=1e-6),
    "moltox21_pna": dict(_GNN, k=16, scope="window", mult_mode="hybrid",
                         refine_ordering=False, weight="eigval"),
    "molpcba_gatedgcn": dict(_GNN, k=3, scope="full", mult_mode="qr",
                             refine_ordering=True, weight="eigval"),
    "molpcba_gatedgcn_plain": dict(_GNN, k=3, scope="full", mult_mode="qr",
                                   refine_ordering=True),
    "molpcba_pna": dict(_GNN, k=16, scope="full", mult_mode="qr",
                        refine_ordering=True, weight="eigval"),
    # ---- transformer and BREC ---------------------------------------------
    "transformer_refined": dict(_TRANSFORMER, mult_mode="hybrid",
                                refine_ordering=True),
    # The MolClinTox cell of tab:transformer-results (README, Reproducibility notes):
    # same eigenvectors as transformer_refined, but the canonical rank fed to the rank
    # embedding is the norm-signature order, as in the research code before a later fix.
    "transformer_refined_prefix": dict(_TRANSFORMER, mult_mode="hybrid",
                                       refine_ordering=True, rank_source="pre_refine"),
    "brec": dict(_TRANSFORMER, mult_mode="hybrid", refine_ordering=False),
    "brec_qr_refined": dict(_TRANSFORMER, mult_mode="qr", refine_ordering=True),
    # Variant of the PRiSM row of tab:brec-comparison (README, Reproducibility notes):
    # vertex order from the projection matrices and a nauty canonical labelling,
    # pivoted QR per eigenspace (needs pynauty and the adjacency matrix; see
    # prism.indrefine).
    "brec_indrefine": dict(_TRANSFORMER, ordering="indrefine_nauty"),
    # ---- appendix ---------------------------------------------------------
    "qr_refined_full": dict(_GNN, k=None, scope="full", mult_mode="qr",
                            refine_ordering=True),
    "rebuttal_comb_lap": dict(operator="comb_lap", drop_trivial=False,
                              keep="first", k=None, scope="full",
                              mult_mode="hybrid", refine_ordering=True,
                              tol_profile="legacy_round"),
    # ---- Algorithm 1 sign rule (Partition, Refine, Solve) ------------------
    "prism_gf2": dict(operator="comb_lap", drop_trivial=False, keep="first",
                      k=None, scope="full", mult_mode="hybrid",
                      refine_ordering=True, sign_rule="gf2",
                      tol_profile="adaptive"),
}
