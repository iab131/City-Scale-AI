"""
DiSR-Mamba: Directed Spectral Residual Mamba.

A residual branch trained on top of a frozen STAEformer trunk for METR-LA
traffic forecasting. The branch predicts a structured correction Delta_Y so
that the final forecast is:

    Y_hat = Y_STAEformer + alpha * Delta_Y

The branch is built from independently switchable experts:

  - Node-space temporal Mamba residual.
  - Symmetric Laplacian spectral bi-axis Mamba residual.
  - Magnetic Laplacian directed spectral bi-axis Mamba residual.
  - Optional learned-basis spectral residual.

A small horizon/cluster router blends the experts. A congestion-aware loss
upweights low-speed and high-volatility regions.

Public entry points:
  - `DiSRMamba` (models.disr.disr_mamba): full residual model.
  - `build_disr_from_config` (models.disr.disr_mamba): config-driven builder.
  - `STAEFrozenWrapper` (models.disr.staeformer_wrapper): frozen-trunk wrapper.
"""

from .losses import (
    masked_mae,
    masked_rmse,
    masked_mape,
    per_horizon_metrics,
    congestion_mask,
    disr_composite_loss,
)
from .spectral_basis import build_symmetric_basis, project, unproject
from .magnetic_laplacian import (
    estimate_lagged_direction,
    build_magnetic_laplacian,
    eigendecompose_hermitian,
    magnetic_basis_from_adjacency,
)
from .biaxis_mamba import BiAxisMambaBlock, TemporalMambaResidual
from .residual_router import HorizonClusterRouter, build_sensor_clusters
from .disr_mamba import DiSRMamba, build_disr_from_config
from .staeformer_wrapper import STAEFrozenWrapper

__all__ = [
    "masked_mae",
    "masked_rmse",
    "masked_mape",
    "per_horizon_metrics",
    "congestion_mask",
    "disr_composite_loss",
    "build_symmetric_basis",
    "project",
    "unproject",
    "estimate_lagged_direction",
    "build_magnetic_laplacian",
    "eigendecompose_hermitian",
    "magnetic_basis_from_adjacency",
    "BiAxisMambaBlock",
    "TemporalMambaResidual",
    "HorizonClusterRouter",
    "build_sensor_clusters",
    "DiSRMamba",
    "build_disr_from_config",
    "STAEFrozenWrapper",
]
