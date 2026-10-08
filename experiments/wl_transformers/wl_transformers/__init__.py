"""Graph transformer / GINE harness with canonicalized Laplacian positional encodings.

Canonicalization is delegated to the ``prism`` package.  See ``pe_types`` for the
supported ``pe_type`` strings.
"""

from .data import Transform, compute_laplacian_eigen, transform_dataset
from .models import GNN, Transformer, create_gnn, create_transformer
from .pe_types import (PE_TYPES, describe_canonicalizer, make_canonicalizer,
                       prism_preset_for, resolve_pe_type)
from .training import (CosineWithWarmupLR, RunLogger, autocast_dtype, configure_optimizers,
                       count_parameters, ensure_data_dir, log_config)

__all__ = [
    "Transform", "compute_laplacian_eigen", "transform_dataset",
    "GNN", "Transformer", "create_gnn", "create_transformer",
    "PE_TYPES", "describe_canonicalizer", "make_canonicalizer", "prism_preset_for",
    "resolve_pe_type",
    "CosineWithWarmupLR", "RunLogger", "autocast_dtype", "configure_optimizers",
    "count_parameters",
    "ensure_data_dir", "log_config",
]
