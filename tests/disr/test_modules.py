"""Shape + behaviour tests for DiSR-Mamba modules."""
import math
import os
import sys
import numpy as np
import torch
import pytest

# Allow CPU-only fallback to the GRU stand-in if mamba_ssm not installed.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.disr.losses import (
    masked_mae, masked_rmse, masked_mape,
    congestion_mask, disr_composite_loss,
    per_horizon_metrics, per_speed_regime_mae,
)
from models.disr.biaxis_mamba import (
    BiAxisMambaBlock, TemporalMambaResidual, _HAS_MAMBA,
)
from models.disr.residual_router import HorizonClusterRouter
from models.disr.disr_mamba import DiSRMamba, build_disr_from_config


# ---------------- losses ----------------
def test_masked_mae_ignores_missing():
    pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    true = torch.tensor([[[10.0, 99.0], [1.0, 1.0]]])
    mask = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    # Only valid entries are at t=1 with diff=0 -> MAE = 0
    v = masked_mae(pred, true, mask)
    assert float(v) < 1e-6


def test_masked_metrics_match_per_horizon_table():
    torch.manual_seed(0)
    P = torch.randn(4, 12, 5) * 5 + 50
    Y = P + torch.randn_like(P) * 2
    M = torch.ones_like(P)
    m = per_horizon_metrics(P, Y, M)
    # Sanity: avg_mae > 0, all horizons present
    assert m["avg_mae"] > 0
    for k in ("mae_15", "mae_30", "mae_60"):
        assert k in m


def test_per_speed_regime_partitions_correctly():
    Y = torch.tensor([[[5.0, 25.0, 45.0, 70.0]]])
    P = torch.zeros_like(Y)
    M = torch.ones_like(Y)
    table = per_speed_regime_mae(P, Y, M)
    # Each bucket has one value; means equal those values.
    assert math.isclose(table["mae_lt20"],  5.0, rel_tol=1e-5)
    assert math.isclose(table["mae_20_40"], 25.0, rel_tol=1e-5)
    assert math.isclose(table["mae_40_60"], 45.0, rel_tol=1e-5)
    assert math.isclose(table["mae_ge60"],  70.0, rel_tol=1e-5)


def test_congestion_mask_speed_only():
    y = torch.tensor([[[5.0, 15.0, 25.0, 70.0]]])
    m = congestion_mask(y, speed_thr=20.0)
    assert torch.equal(m, torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]))


def test_congestion_mask_with_volatility():
    y = torch.tensor([[[5.0, 30.0, 50.0, 50.0]]])           # [B,T,N]
    x_recent = torch.tensor([[[40.0]]])
    # diff from 40 mph: 35, 10, 10, 10 -> volatility threshold 5 => all flagged
    m = congestion_mask(y, x_recent=x_recent, speed_thr=20.0, delta_thr=5.0)
    # speed mask: [1,0,0,0]; volatility: all > 5 -> [1,1,1,1]
    assert torch.equal(m, torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]))


def test_disr_composite_loss_pieces():
    torch.manual_seed(0)
    B, T, N = 2, 12, 5
    Y = torch.randn(B, T, N) * 5 + 50
    Y_base = Y + torch.randn(B, T, N) * 2
    Delta = (Y - Y_base) + torch.randn_like(Y) * 0.1
    Y_pred = Y_base + Delta
    Mk = torch.ones_like(Y)
    L, parts = disr_composite_loss(Y_pred, Y, Y_base, Delta, Mk,
                                   x_recent=None,
                                   lambda_res=0.2, lambda_cong=0.1)
    assert "L_main" in parts and "L_res" in parts and "L_cong" in parts
    assert float(L) > 0


# ---------------- modules ----------------
def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("mode_axis", [True, False])
def test_biaxis_mamba_block_shape(mode_axis):
    B, T, K, D = 2, 12, 8, 16
    dev = _dev()
    x = torch.randn(B, T, K, D, device=dev)
    blk = BiAxisMambaBlock(d_model=D, n_layers=1, mode_axis=mode_axis).to(dev)
    y = blk(x.float())
    assert y.shape == (B, T, K, D)


def test_temporal_mamba_residual_shape():
    B, T, N = 2, 12, 17
    dev = _dev()
    x = torch.randn(B, T, N, device=dev)
    tod = torch.rand(B, T, device=dev)
    dow = torch.randint(0, 7, (B, T), device=dev)
    m = TemporalMambaResidual(n_nodes=N, in_steps=T, out_steps=12,
                              d_model=16).to(dev)
    y = m(x.float(), tod.float(), dow)
    assert y.shape == (B, 12, N)


def test_router_outputs_shapes_and_softmax():
    B, T_in, T_out, N, K, E = 3, 12, 12, 20, 4, 3
    cluster_id = torch.randint(0, K, (N,))
    r = HorizonClusterRouter(n_experts=E, n_nodes=N, n_clusters=K,
                              in_steps=T_in, out_steps=T_out, d_router=16,
                              cluster_id=cluster_id)
    tod = torch.rand(B, T_in)
    dow = torch.randint(0, 7, (B, T_in))
    x_raw = torch.rand(B, T_in, N) * 80
    x_norm = torch.randn(B, T_in, N)
    gate, alpha, aux = r(tod, dow, x_raw, x_norm)
    assert gate.shape == (B, T_out, N, E)
    assert alpha.shape == (B, T_out, N)
    # softmax sums to 1 across experts
    np.testing.assert_allclose(gate.sum(dim=-1).detach().numpy(),
                                np.ones((B, T_out, N)), atol=1e-5)
    # alpha bounded
    assert float(alpha.max()) <= r.alpha_max + 1e-6
    assert float(alpha.min()) >= 0.0
    assert "entropy" in aux


def test_disr_mamba_temporal_only_shape():
    B, T_in, T_out, N = 2, 12, 12, 30
    dev = _dev()
    m = DiSRMamba(
        n_nodes=N, in_steps=T_in, out_steps=T_out,
        use_temporal_residual=True,
        use_symmetric_spectral=False,
        use_magnetic_spectral=False,
        d_model=16, n_layers=1,
    ).to(dev)
    x = torch.randn(B, T_in, N, device=dev)
    tod = torch.rand(B, T_in, device=dev)
    dow = torch.randint(0, 7, (B, T_in), device=dev)
    out = m(x, tod, dow)
    assert out["delta_y_norm"].shape == (B, T_out, N)
    assert len(out["per_expert"]) == 1


def test_disr_mamba_all_experts_shape():
    from models.disr.spectral_basis import build_symmetric_basis
    from models.disr.magnetic_laplacian import (
        build_magnetic_laplacian, eigendecompose_hermitian,
    )
    rng = np.random.default_rng(0)
    N, K = 30, 12
    A_dir = (rng.random((N, N)) < 0.06).astype(np.float32)
    np.fill_diagonal(A_dir, 0.0)
    A_sym = 0.5 * (A_dir + A_dir.T)
    _, U_sym = build_symmetric_basis(A_sym, k=K, side="low")
    L_q = build_magnetic_laplacian(torch.from_numpy(A_dir), q=0.10)
    _, U_mag = eigendecompose_hermitian(L_q, k=K, side="low")
    cluster_id = torch.randint(0, 4, (N,))
    dev = _dev()
    m = DiSRMamba(
        n_nodes=N, in_steps=12, out_steps=12,
        use_temporal_residual=True,
        use_symmetric_spectral=True,
        use_magnetic_spectral=True,
        U_sym=torch.from_numpy(U_sym), U_mag=U_mag,
        d_model=16, n_layers=1,
        use_horizon_cluster_router=True, n_clusters=4,
        cluster_id=cluster_id,
    ).to(dev)
    B = 2
    x = torch.randn(B, 12, N, device=dev)
    tod = torch.rand(B, 12, device=dev)
    dow = torch.randint(0, 7, (B, 12), device=dev)
    out = m(x, tod, dow, x_recent_raw=torch.rand(B, 12, N, device=dev) * 60.0)
    assert out["delta_y_norm"].shape == (B, 12, N)
    assert out["expert_weights"].shape == (B, 12, N, 3)
    # alpha respects max
    assert float(out["alpha"].max()) <= m.router.alpha_max + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_amp_step_runs():
    """One forward+backward step in bf16 AMP must produce finite gradients."""
    from models.disr.losses import masked_mae
    B, T, N = 2, 12, 16
    m = TemporalMambaResidual(n_nodes=N, in_steps=T, out_steps=T, d_model=16,
                              n_layers=1).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(B, T, N, device="cuda")
    tod = torch.rand(B, T, device="cuda")
    dow = torch.randint(0, 7, (B, T), device="cuda")
    y_true = torch.randn(B, T, N, device="cuda") * 5 + 50
    msk = torch.ones_like(y_true)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_pred = m(x, tod, dow)
        # treat output as normalised; supervise with masked MAE against a
        # rescaled target.
        loss = masked_mae(y_pred * 5 + 50, y_true, msk)
    loss.backward()
    g_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in m.parameters())
    assert g_finite, "non-finite gradients under bf16 AMP"
    opt.step()
