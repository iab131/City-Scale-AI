"""Shape + AMP smoke tests for SSM-Magma."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.disr.spectral_basis import build_symmetric_basis
from models.disr.magnetic_laplacian import (
    build_magnetic_laplacian, eigendecompose_hermitian,
)
from models.ssm_magma import SSMMagma


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bases_and_clusters(n=30, k=12, seed=0):
    rng = np.random.default_rng(seed)
    A_dir = (rng.random((n, n)) < 0.10).astype(np.float32)
    np.fill_diagonal(A_dir, 0.0)
    A_sym = 0.5 * (A_dir + A_dir.T)
    _, U_sym = build_symmetric_basis(A_sym, k=k, side="low")
    L_q = build_magnetic_laplacian(torch.from_numpy(A_dir), q=0.10)
    _, U_mag = eigendecompose_hermitian(L_q, k=k, side="low")
    cluster_id = torch.from_numpy(rng.integers(0, 4, size=(n,))).long()
    return torch.from_numpy(U_sym).float(), U_mag, cluster_id


def test_ssm_magma_forward_shape():
    n = 30
    dev = _dev()
    U_sym, U_mag, cid = _bases_and_clusters(n=n, k=12)
    m = SSMMagma(
        n_nodes=n, in_steps=12, out_steps=12,
        U_sym=U_sym, U_mag=U_mag, cluster_id=cid.to(dev),
        d_model=16, n_layers=1, k_modes=12,
        d_sem=8, k_neighbors=4, n_clusters=4,
    ).to(dev)
    B = 2
    x = torch.randn(B, 12, n, device=dev)
    tod = torch.rand(B, 12, device=dev)
    dow = torch.randint(0, 7, (B, 12), device=dev)
    x_recent_raw = torch.rand(B, 12, n, device=dev) * 60
    out = m(x, tod, dow, x_recent_raw=x_recent_raw)
    assert out.shape == (B, 12, n), out.shape


def test_ssm_magma_return_experts_shapes():
    n = 30
    dev = _dev()
    U_sym, U_mag, cid = _bases_and_clusters(n=n, k=12)
    m = SSMMagma(
        n_nodes=n, U_sym=U_sym, U_mag=U_mag, cluster_id=cid.to(dev),
        d_model=16, n_layers=1, k_modes=12, d_sem=8, k_neighbors=4,
        n_clusters=4,
    ).to(dev)
    B = 2
    x = torch.randn(B, 12, n, device=dev)
    tod = torch.rand(B, 12, device=dev)
    dow = torch.randint(0, 7, (B, 12), device=dev)
    x_recent_raw = torch.rand(B, 12, n, device=dev) * 60
    res = m(x, tod, dow, x_recent_raw=x_recent_raw, return_experts=True)
    assert res["y_pred_norm"].shape == (B, 12, n)
    assert len(res["per_expert"]) == 3
    assert res["gate"].shape == (B, 12, n, 3)


def test_ssm_magma_paper_default_param_count():
    """SSM-Magma should be in 0.5-3M range for METR-LA defaults."""
    # Use small bases since we just want to check the param count is sane.
    U_sym, U_mag, cid = _bases_and_clusters(n=207, k=48)
    m = SSMMagma(
        n_nodes=207, U_sym=U_sym, U_mag=U_mag, cluster_id=cid,
    )
    n = sum(p.numel() for p in m.parameters())
    assert 200_000 < n < 5_000_000, n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ssm_magma_bf16_amp_step():
    """One forward+backward step under bf16 AMP must produce finite gradients."""
    U_sym, U_mag, cid = _bases_and_clusters(n=207, k=48)
    m = SSMMagma(
        n_nodes=207, U_sym=U_sym, U_mag=U_mag, cluster_id=cid,
        d_model=32, n_layers=1, k_modes=48,
    ).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    B = 4
    x = torch.randn(B, 12, 207, device="cuda")
    tod = torch.rand(B, 12, device="cuda")
    dow = torch.randint(0, 7, (B, 12), device="cuda")
    x_recent_raw = (torch.rand(B, 12, 207, device="cuda") * 60)
    y_true = torch.randn(B, 12, 207, device="cuda") * 5 + 50
    msk = torch.ones_like(y_true)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_pred_norm = m(x, tod, dow, x_recent_raw=x_recent_raw)
        y_pred = y_pred_norm * 5 + 50
        loss = (((y_pred - y_true).abs() * msk).mean()
                / msk.mean().clamp(min=1e-6))
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in m.parameters())
    assert finite, "non-finite gradients under bf16"
    opt.step()
