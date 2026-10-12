"""The PRiSM class: one canonicalizer, configured per experiment.

Every experiment of the paper uses this class; runs differ only in the
constructor arguments, collected as named presets in :mod:`prism.presets`::

    from prism import PRiSM
    canon = PRiSM.from_preset("zinc_pna")
    pe = canon.pos_enc(adjacency, k=8)          # (n, 8) float32

The class has four layers, each controlled by its own group of arguments.

Spectrum
    ``operator``, ``drop_trivial``, ``k``, ``scope``, ``window_buffer``,
    ``keep`` choose the matrix, drop its trivial eigenpair and select which
    eigenpairs are canonicalized and kept.
Canonicalization
    ``sign_rule``, ``mult_mode``, ``refine_ordering``, ``pivot_threshold``
    choose how each eigenspace is resolved.
Ordering
    ``ordering`` selects how the vertex order that picks the pivots is built:
    from the norm-value signature (``spectral``, the default, used by all
    table cells but one) or from the projection matrices and a nauty
    canonical labelling (``indrefine_nauty``, the BREC row; needs the
    optional ``pynauty`` and the adjacency matrix).
Numerics
    ``tol_profile`` and ``quantize_float32`` choose the tolerance profile and
    whether eigenpairs pass through float32 first (as in the transformer
    harness).
Post-processing
    ``weight`` and ``weight_param`` optionally scale the kept columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import core, gf2, indrefine, operators

_SIGN_RULES = ("first_pivot", "gf2")
_MULT_MODES = ("hybrid", "qr")
_SCOPES = ("full", "window")
_KEEP = ("last", "first")
_WEIGHTS = (None, "eigval", "inv_sqrt_lap", "heat")
_ORDERINGS = ("spectral", "indrefine_nauty")


@dataclass
class CanonInfo:
    """Result of :meth:`PRiSM.canonicalize` with ``return_info=True``.

    V           canonical eigenvectors (n, k), float64, input column order.
    order       canonical vertex order (``order[i]`` = vertex at position i).
    rank        inverse permutation (``rank[order[i]] = i``).
    norm_sig    norm-value signature of the input eigenvectors (n, K).
    multiplicities  eigenspace multiplicities used.
    """

    V: np.ndarray
    order: np.ndarray
    rank: np.ndarray
    norm_sig: np.ndarray
    multiplicities: list


class PRiSM:
    """Eigenvector canonicalization: Partition, Refine, Solve, Match, and its
    hybrid extension to eigenvalue multiplicities greater than one.

    Parameters
    ----------
    operator : {"norm_adj", "norm_lap", "comb_lap"}
        Matrix whose eigenvectors are canonicalized when a graph is given.
    drop_trivial : bool
        Drop the trivial eigenpair (largest of ``norm_adj``, smallest of the
        Laplacians) before anything else.
    k : int or None
        Number of columns kept by :meth:`pos_enc` / :meth:`spectrum`
        (None keeps all).
    scope : {"full", "window"}
        ``full`` canonicalizes every eigenpair and then keeps ``k`` of them.
        ``window`` first truncates to ``k + window_buffer`` eigenpairs and
        canonicalizes only those; an eigenspace cut by the window is then
        resolved on its truncated part.
    window_buffer : int
        Extra eigenpairs canonicalized beyond ``k`` and dropped afterwards, so
        an eigenspace straddling the cutoff is resolved whole (``window``).
    keep : {"last", "first"}
        Which end of the ascending spectrum is kept: the ``k`` largest
        (``last``, ``norm_adj``) or the ``k`` smallest (``first``,
        Laplacians) eigenvalues.
    sign_rule : {"first_pivot", "gf2"}
        How the m=1 eigenspaces are resolved.  ``first_pivot`` flips each
        column so that its first significant entry in the vertex order is
        positive (with an injective signature this is FastSign).  ``gf2``
        resolves all m=1 columns jointly by Refine + Solve (Algorithm 1),
        using FastSign when the signature is injective.  Requires
        ``mult_mode="hybrid"`` and ``refine_ordering=True``.
    mult_mode : {"hybrid", "qr"}
        ``hybrid`` uses Householder/Givens rules for m<=3 and QR for m>=4.
        ``qr`` uses QR for every multiplicity, with unique-signature pivots;
        it requires ``refine_ordering=True``.
    refine_ordering : bool
        Process eigenspaces by increasing multiplicity and refine the vertex
        order from already canonical columns after each one.
    tol_profile : {"adaptive", "legacy_round"}
        ``adaptive``: relative tie snapping and an eps-scale floor on the
        eigenvalue grouping tolerance.  ``legacy_round``: fixed-decimal
        rounding and no floor (transformer / BREC harness).
    pivot_threshold : float
        Minimum magnitude for an entry (or residual norm) to act as a pivot.
    quantize_float32 : bool
        Round eigenvalues and eigenvectors through float32 before
        canonicalizing (what the transformer harness stores).
    weight : {None, "eigval", "inv_sqrt_lap", "heat"}
        Optional column scaling by a function of the adjacency eigenvalue
        (``norm_adj`` only).
    weight_param : float or None
        ``eps`` of ``inv_sqrt_lap`` (default 1e-6) or ``t`` of ``heat``
        (default 1.0).
    gf2_decimals : int
        Rounding precision of the ``gf2`` sign rule.
    ordering : {"spectral", "indrefine_nauty"}
        ``spectral`` orders the vertices by the norm-value signature (and, with
        ``refine_ordering``, by the already canonical columns).
        ``indrefine_nauty`` orders them, per eigenspace, by the diagonal and the
        sorted absolute row of the eigenspace projection matrix with a nauty
        canonical labelling of the graph as final tie-break
        (:mod:`prism.indrefine`); all eigenspaces are then resolved by pivoted
        QR.  It needs ``pynauty`` and the adjacency matrix (``adjacency=`` of
        :meth:`canonicalize`; :meth:`pos_enc` and :meth:`spectrum` pass theirs),
        and ``sign_rule``, ``mult_mode`` and ``refine_ordering`` must keep their
        defaults because they do not apply to it.
    ir_order_decimals : int
        ``indrefine_nauty``: decimals to which the projection-matrix diagonals
        (vertex colours of nauty) and the ordering keys are rounded.
    ir_proj_decimals : int
        ``indrefine_nauty``: decimals to which the projection matrices are rounded.
    """

    def __init__(
        self,
        *,
        operator: str = "norm_adj",
        drop_trivial: bool = True,
        k: Optional[int] = None,
        scope: str = "full",
        window_buffer: int = 0,
        keep: str = "last",
        sign_rule: str = "first_pivot",
        mult_mode: str = "hybrid",
        refine_ordering: bool = False,
        tol_profile: str = "adaptive",
        pivot_threshold: float = 1e-8,
        quantize_float32: bool = False,
        weight: Optional[str] = None,
        weight_param: Optional[float] = None,
        gf2_decimals: int = gf2.DECIMALS,
        ordering: str = "spectral",
        ir_order_decimals: int = 8,
        ir_proj_decimals: int = 10,
    ):
        if operator not in operators.OPERATORS:
            raise ValueError(f"operator must be one of {operators.OPERATORS}")
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}")
        if keep not in _KEEP:
            raise ValueError(f"keep must be one of {_KEEP}")
        if sign_rule not in _SIGN_RULES:
            raise ValueError(f"sign_rule must be one of {_SIGN_RULES}")
        if mult_mode not in _MULT_MODES:
            raise ValueError(f"mult_mode must be one of {_MULT_MODES}")
        if weight not in _WEIGHTS:
            raise ValueError(f"weight must be one of {_WEIGHTS}")
        if mult_mode == "qr" and not refine_ordering:
            raise ValueError("mult_mode='qr' requires refine_ordering=True")
        if sign_rule == "gf2" and (mult_mode != "hybrid" or not refine_ordering):
            raise ValueError("sign_rule='gf2' requires mult_mode='hybrid' and "
                             "refine_ordering=True")
        if weight is not None and operator != "norm_adj":
            raise ValueError("column weights are defined for operator='norm_adj'")
        if window_buffer < 0 or (window_buffer and scope != "window"):
            raise ValueError("window_buffer must be >= 0 and needs scope='window'")
        if k is not None and k < 0:
            raise ValueError("k must be non-negative")
        if ordering not in _ORDERINGS:
            raise ValueError(f"ordering must be one of {_ORDERINGS}")
        if ordering == "indrefine_nauty":
            if sign_rule != "first_pivot" or mult_mode != "hybrid" or refine_ordering:
                raise ValueError("ordering='indrefine_nauty' resolves every eigenspace by "
                                 "pivoted QR in its own ordering; sign_rule, mult_mode and "
                                 "refine_ordering do not apply and must keep their defaults "
                                 "('first_pivot', 'hybrid', False)")
            indrefine.require_pynauty()

        self.operator = operator
        self.drop_trivial = drop_trivial
        self.k = k
        self.scope = scope
        self.window_buffer = window_buffer
        self.keep = keep
        self.sign_rule = sign_rule
        self.mult_mode = mult_mode
        self.refine_ordering = refine_ordering
        self.tol_profile = tol_profile
        self.pivot_threshold = pivot_threshold
        self.quantize_float32 = quantize_float32
        self.weight = weight
        self.weight_param = weight_param
        self.gf2_decimals = gf2_decimals
        self.ordering = ordering
        self.ir_order_decimals = ir_order_decimals
        self.ir_proj_decimals = ir_proj_decimals
        self.tol = core.make_tolerances(tol_profile, mult_mode)

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "PRiSM":
        """Build the instance of a named preset (see :mod:`prism.presets`)."""
        from .presets import PRESETS

        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
        kwargs = dict(PRESETS[name])
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def needs_adjacency(self) -> bool:
        """Whether :meth:`canonicalize` needs the adjacency matrix (``indrefine_nauty``)."""
        return self.ordering != "spectral"

    def get_config(self) -> dict:
        """The constructor arguments of this instance.

        The arguments of the ``ordering`` layer are listed only for a non-default
        ordering, so the configuration (and with it the cache key of the molecular
        harness) of every ``spectral`` instance is the same as before they existed.
        """
        cfg = dict(
            operator=self.operator, drop_trivial=self.drop_trivial, k=self.k,
            scope=self.scope, window_buffer=self.window_buffer, keep=self.keep,
            sign_rule=self.sign_rule, mult_mode=self.mult_mode,
            refine_ordering=self.refine_ordering, tol_profile=self.tol_profile,
            pivot_threshold=self.pivot_threshold,
            quantize_float32=self.quantize_float32, weight=self.weight,
            weight_param=self.weight_param, gf2_decimals=self.gf2_decimals,
        )
        if self.ordering != "spectral":
            cfg.update(ordering=self.ordering, ir_order_decimals=self.ir_order_decimals,
                       ir_proj_decimals=self.ir_proj_decimals)
        return cfg

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"PRiSM({args})"

    # -------------------------------------------------------- canonicalization

    def canonicalize(self, V, lam, return_info: bool = False, adjacency=None):
        """Canonicalize eigenvectors given with their eigenvalues.

        Parameters
        ----------
        V : (n, c) array, columns are eigenvectors in eigenvalue order.
        lam : (c,) array, eigenvalues in non-decreasing order (repeated
            eigenvalues appear once per column).
        return_info : bool
            Return a :class:`CanonInfo` (vertex order, rank, norm signature)
            instead of only the canonical eigenvectors.
        adjacency : (n, n) array, optional
            Adjacency matrix of the graph (only its support is used).  Required
            by ``ordering="indrefine_nauty"`` (see :attr:`needs_adjacency`),
            ignored otherwise.

        Returns
        -------
        (n, c) float64 ndarray, or :class:`CanonInfo`.
        """
        lam = np.asarray(lam, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if self.quantize_float32:
            lam = lam.astype(np.float32).astype(np.float64)
            V = V.astype(np.float32).astype(np.float64)

        if self.ordering == "indrefine_nauty":
            return self._canonicalize_indrefine(V, lam, adjacency, return_info)

        tol = self.tol
        mults = core.detect_multiplicities(lam, tol.mult_rtol, tol.mult_atol)
        thr = self.pivot_threshold

        if self.mult_mode == "qr":
            Vc, order = core.canonicalize_qr_refined(V, mults, thr, tol)
        elif self.refine_ordering:
            solver = None
            if self.sign_rule == "gf2":
                solver = self._gf2_flip_mask
            Vc, order = core.canonicalize_hybrid_refined(V, mults, thr, tol, solver)
        else:
            Vc, order = core.canonicalize_hybrid(V, mults, thr, tol)

        if not return_info:
            return Vc
        n = V.shape[0]
        rank = np.zeros(n, dtype=np.int64)
        rank[order] = np.arange(n)
        return CanonInfo(
            V=Vc, order=order, rank=rank,
            norm_sig=core.compute_norm_signature(V, mults),
            multiplicities=mults,
        )

    def _canonicalize_indrefine(self, V, lam, adjacency, return_info):
        if adjacency is None:
            raise ValueError("ordering='indrefine_nauty' needs the adjacency matrix: "
                             "pass adjacency=A to canonicalize (pos_enc and spectrum do)")
        Vc, order, mults = indrefine.canonicalize(
            V, lam, adjacency, threshold=self.pivot_threshold,
            order_decimals=self.ir_order_decimals, proj_decimals=self.ir_proj_decimals)
        if not return_info:
            return Vc
        n = V.shape[0]
        rank = np.zeros(n, dtype=np.int64)
        rank[order] = np.arange(n)
        return CanonInfo(V=Vc, order=order, rank=rank,
                         norm_sig=core.compute_norm_signature(V, mults),
                         multiplicities=mults)

    def _gf2_flip_mask(self, U1: np.ndarray) -> np.ndarray:
        return gf2.sign_vector(U1, self.gf2_decimals, fast_path=True) == 1

    # ------------------------------------------------------ Algorithm 1 and 2

    def canonicalize_simple(self, U):
        """Base PRiSM (Algorithm 1) on a simple-spectrum eigenvector matrix.

        Returns a :class:`prism.gf2.PrismResult` with the canonical matrix
        (rows sorted), the sign vector, the partition and the row order.
        """
        return gf2.prism_canonicalize(np.asarray(U, dtype=np.float64), self.gf2_decimals)

    def fastsign(self, U):
        """FastSign (Algorithm 2): ``(canonical matrix, sign vector)``.

        Valid when the absolute-value signature of ``U`` is injective;
        raises ``ValueError`` otherwise.
        """
        U = np.asarray(U, dtype=np.float64)
        s = gf2.fastsign(U, self.gf2_decimals)
        U_signed = U * np.where(s == 1, -1.0, 1.0)[None, :]
        canonical, _ = gf2.match_stage(U_signed, self.gf2_decimals)
        return canonical, s

    # ------------------------------------------------------ graph-level entry

    def eigendecompose(self, A):
        """Eigendecomposition of ``operator`` for adjacency ``A``.

        Returns ``(lam, V)`` after dropping the trivial eigenpair if
        ``drop_trivial`` is set.  Eigenvalues ascend.
        """
        lam, V = operators.eigendecompose(A, self.operator)
        if self.drop_trivial and lam.shape[0] > 0:
            if self.operator == "norm_adj":
                lam, V = lam[:-1], V[:, :-1]
            else:
                lam, V = lam[1:], V[:, 1:]
        return lam, V

    def spectrum(self, A, k: Optional[int] = None):
        """Canonical eigenpairs kept by the configuration.

        Applies ``scope``, ``window_buffer`` and ``keep`` to the spectrum of
        ``A`` and canonicalizes the selected eigenpairs.  Returns
        ``(lam_kept, V_kept)`` with at most ``k`` columns (all if ``k`` is
        None), in ascending operator-eigenvalue order.
        """
        k = self.k if k is None else k
        lam, V = self.eigendecompose(A)
        ncols = lam.shape[0]
        dim = ncols if k is None else min(ncols, k)
        adj = operators.as_dense_adjacency(A) if self.needs_adjacency else None

        if self.scope == "full":
            Vc = self.canonicalize(V, lam, adjacency=adj)
            lo, hi = self._kept_slice(ncols, dim)
            return lam[lo:hi], Vc[:, lo:hi]

        win = min(ncols, dim + self.window_buffer)
        if self.keep == "last":
            wlo, whi = ncols - win, ncols
        else:
            wlo, whi = 0, win
        Vc = self.canonicalize(V[:, wlo:whi], lam[wlo:whi], adjacency=adj)
        lam_w = lam[wlo:whi]
        lo, hi = self._kept_slice(win, dim)
        return lam_w[lo:hi], Vc[:, lo:hi]

    def _kept_slice(self, ncols: int, dim: int):
        if self.keep == "last":
            return ncols - dim, ncols
        return 0, dim

    def pos_enc(self, A, k: Optional[int] = None) -> np.ndarray:
        """Positional encoding of a graph: canonical columns, weighted, padded.

        Returns an ``(n, k)`` float32 array: the kept canonical eigenvectors
        (scaled by ``weight`` if set) followed by zero columns when the graph
        has fewer than ``k`` non-trivial eigenpairs.
        """
        k = self.k if k is None else k
        if k is None:
            raise ValueError("pos_enc needs k (constructor argument or call argument)")
        lam_k, V_k = self.spectrum(A, k)
        if self.weight is not None:
            w = operators.column_weights(self.weight, lam_k, self.weight_param)
            V_k = V_k * w[np.newaxis, :]
        n = V_k.shape[0]
        out = np.zeros((n, k), dtype=np.float32)
        dim = V_k.shape[1]
        out[:, :dim] = V_k.astype(np.float32)
        return out
