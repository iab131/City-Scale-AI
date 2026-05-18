"""STAE-Spectral-Magma: STAEformer encoder + three-view spectral Mamba augmentation."""
from .stae_spectral_magma import (
    STAESpectralMagma,
    SpectralMagmaAugmentation,
    build_stae_spectral_magma,
)

__all__ = [
    "STAESpectralMagma",
    "SpectralMagmaAugmentation",
    "build_stae_spectral_magma",
]
