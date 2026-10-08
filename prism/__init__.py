"""PRiSM: Partition, Refine, Solve, Match eigenvector canonicalization.

Reference implementation accompanying "Weisfeiler Leman is Incomplete On
Graphs With A Simple Spectrum" (Hordan, Dym, Seppelt).  A single class,
:class:`PRiSM`, covers every variant used in the experiments; the
configurations are named presets (:data:`PRESETS`).
"""

from .gf2 import fastsign, prism_canonicalize
from .presets import PRESETS
from .prism import CanonInfo, PRiSM

__all__ = ["PRiSM", "CanonInfo", "PRESETS", "prism_canonicalize", "fastsign"]
__version__ = "0.1.0"
