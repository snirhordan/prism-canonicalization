"""Canonicalization baselines compared against PRiSM in the paper.

``MAP`` and ``OAP`` need PyTorch; ``lap_pe`` and ``random_sign_flip`` need only
numpy.  Model-side SignNet stays with the experiments.
"""

from .lappe import lap_pe, random_sign_flip


def __getattr__(name):
    if name == "MAP":
        from .map import MAP
        return MAP
    if name == "OAP":
        from .oap import OAP
        return OAP
    raise AttributeError(name)


__all__ = ["MAP", "OAP", "lap_pe", "random_sign_flip"]
