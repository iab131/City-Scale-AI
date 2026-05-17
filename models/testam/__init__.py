"""
TESTAM (Time-Enhanced Spatio-Temporal Attention Model) port for City-Scale-AI.

Reference: Lee & Ko, "Testam: A Time-Enhanced Spatio-Temporal Attention Model with
Mixture of Experts", ICLR 2024 (arXiv:2403.02600).

Three experts:
  - TemporalExpert: temporal-only, no graph (Identity)
  - STExpert: adaptive graph + GCN + temporal attention (Adaptive)
  - AttentionExpert: pure attention, no graph (Dynamic)

A MemoryGate softmax-mixes the experts. Training: end-to-end masked MAE on raw mph;
optional auxiliary gate loss to encourage expert specialization.

Adapted to our pipeline:
  forward(x_norm, tod, dow) -> y_pred_norm of shape [B, T_out, N]
"""
from .testam import TESTAM, build_testam

__all__ = ["TESTAM", "build_testam"]
