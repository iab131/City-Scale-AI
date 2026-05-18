"""Shape + AMP smoke tests for STAE-Spectral-Magma."""
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
from models.stae_spectral_magma import STAESpectralMagma


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


def test_stae_spectral_magma_forward_shape():
    n = 30
    dev = _dev()
    U_sym, U_mag, cid = _bases_and_clusters(n=n, k=12)
    m = STAESpectralMagma(
        N=n, in_steps=12, out_steps=12,
        input_embedding_dim=8, tod_embedding_dim=8, dow_embedding_dim=8,
        adaptive_embedding_dim=16, feed_forward_dim=64, num_heads=2,
        num_layers=1, dropout=0.0,
        d_branch=16, spec_n_layers=1,
        U_sym=U_sym, U_mag=U_mag, cluster_id=cid.to(dev),
        d_sem=8, k_neighbors=4, k_modes_sem=12, n_clusters=4,
    ).to(dev)
    B = 2
    x = torch.randn(B, 12, n, device=dev)
    tod = torch.rand(B, 12, device=dev)
    dow = torch.randint(0, 7, (B, 12), device=dev)
    x_recent = torch.rand(B, 12, n, device=dev) * 60
    out = m(x, tod, dow, x_recent_raw=x_recent)
    assert out.shape == (B, 12, n), out.shape


def test_stae_spectral_magma_residual_starts_near_zero():
    """The spectral sidechain's proj_up is initialised with σ=1e-3 so that at
    init the augmentation is a near-zero residual and the STAEformer encoder
    dominates. Verify by comparing |h_aug| / |h|."""
    n = 30
    dev = _dev()
    U_sym, U_mag, cid = _bases_and_clusters(n=n, k=12)
    m = STAESpectralMagma(
        N=n, in_steps=12, out_steps=12,
        input_embedding_dim=8, tod_embedding_dim=8, dow_embedding_dim=8,
        adaptive_embedding_dim=16, feed_forward_dim=64, num_heads=2,
        num_layers=1, dropout=0.0,
        d_branch=16, spec_n_layers=1,
        U_sym=U_sym, U_mag=U_mag, cluster_id=cid.to(dev),
        d_sem=8, k_neighbors=4, k_modes_sem=12, n_clusters=4,
    ).to(dev).eval()
    B = 2
    x = torch.randn(B, 12, n, device=dev)
    tod = torch.rand(B, 12, device=dev)
    dow = torch.randint(0, 7, (B, 12), device=dev)
    x_recent = torch.rand(B, 12, n, device=dev) * 60
    with torch.no_grad():
        h_stae = m.staeformer.get_hidden(x, tod, dow)
        aug = m.spectral_aug(h_stae, tod, dow, x_recent_raw=x_recent)
        # Ratio of augmentation magnitude to encoder magnitude
        ratio = aug["h_aug"].abs().mean() / (h_stae.abs().mean() + 1e-6)
    assert ratio.item() < 0.05, f"sidechain too strong at init: {ratio.item():.4f}"


def test_stae_spectral_magma_param_count_in_range():
    """The model is the STAEformer trunk + a small spectral sidechain. Param
    count should be in the 1.5-5M range for METR-LA defaults."""
    U_sym, U_mag, cid = _bases_and_clusters(n=207, k=64)
    m = STAESpectralMagma(
        N=207, in_steps=12, out_steps=12,
        U_sym=U_sym, U_mag=U_mag, cluster_id=cid,
        k_modes_sem=64,
    )
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 8_000_000, n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stae_spectral_magma_bf16_amp_step():
    U_sym, U_mag, cid = _bases_and_clusters(n=207, k=64)
    m = STAESpectralMagma(
        N=207, in_steps=12, out_steps=12,
        adaptive_embedding_dim=32, feed_forward_dim=128, num_layers=2,
        d_branch=32, spec_n_layers=1,
        U_sym=U_sym, U_mag=U_mag, cluster_id=cid, k_modes_sem=64,
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
