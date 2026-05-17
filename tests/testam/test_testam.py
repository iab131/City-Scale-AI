"""Shape and AMP smoke tests for TESTAM port."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.testam import TESTAM


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_testam_forward_shape_cpu():
    B, T, N = 2, 12, 17
    m = TESTAM(N=N, in_steps=T, out_steps=T, d_model=16, num_layers=1,
                ffn_dim=32, num_heads=2, dropout=0.1, memory_size=4,
                d_emb=8, d_gate=16)
    x = torch.randn(B, T, N)
    tod = torch.rand(B, T)
    dow = torch.randint(0, 7, (B, T))
    y = m(x, tod, dow)
    assert y.shape == (B, T, N), y.shape


def test_testam_return_experts():
    B, T, N = 2, 12, 17
    m = TESTAM(N=N, d_model=16, num_layers=1, ffn_dim=32, num_heads=2,
                memory_size=4, d_emb=8, d_gate=16)
    x = torch.randn(B, T, N)
    tod = torch.rand(B, T)
    dow = torch.randint(0, 7, (B, T))
    y, experts, gate = m(x, tod, dow, return_experts=True)
    assert y.shape == (B, T, N)
    assert experts.shape == (B, T, N, 3)
    assert gate.shape == (B, T, N, 3)
    # softmax: gate sums to 1 across experts
    np.testing.assert_allclose(gate.sum(dim=-1).detach().numpy(),
                                np.ones((B, T, N)), atol=1e-5)


def test_testam_paper_default_param_count():
    """Sanity-check the model is in a reasonable size range for METR-LA."""
    m = TESTAM(N=207)
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 5_000_000, n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_testam_bf16_amp_step():
    B, T, N = 4, 12, 207
    m = TESTAM(N=N, d_model=32, num_layers=3, ffn_dim=64, num_heads=4,
                dropout=0.1).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(B, T, N, device="cuda")
    tod = torch.rand(B, T, device="cuda")
    dow = torch.randint(0, 7, (B, T), device="cuda")
    y_true = torch.randn(B, T, N, device="cuda") * 5 + 50
    msk = torch.ones_like(y_true)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_pred_norm = m(x, tod, dow)
        y_pred = y_pred_norm * 5 + 50
        loss = (((y_pred - y_true).abs() * msk).mean()
                / msk.mean().clamp(min=1e-6))
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in m.parameters())
    assert finite, "non-finite gradients under bf16"
    opt.step()
