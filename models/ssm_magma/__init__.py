"""
SSM-Magma — SpatioSemantic-Magnetic Mamba.

A standalone forecasting model that composes three spectral experts on three
different graph views of METR-LA:

  1. PhysicalSymExpert    — symmetric Laplacian of the physical sensor graph
  2. PhysicalMagExpert    — magnetic Laplacian (directed) of the same graph
  3. SemanticExpert       — learned sensor-embedding kNN graph (TESTAM+-inspired)

Each expert operates in its own spectral basis and applies a bi-axis Mamba
scan over (time, mode). A small horizon-cluster router mixes the experts.

Unlike DiSR-Mamba, this is a *standalone* forecaster, not a residual on top
of a frozen trunk. The gradient comes from masked MAE on the full target,
which is well-conditioned for the trained experts to learn structured signal
(the residual-on-noise failure mode the DiSR campaign documented does not
apply).
"""
from .semantic_graph import SemanticGraph
from .ssm_magma import SSMMagma, build_ssm_magma

__all__ = ["SemanticGraph", "SSMMagma", "build_ssm_magma"]
