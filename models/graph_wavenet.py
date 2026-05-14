"""
Graph WaveNet (IJCAI 2019) — Wu et al.
Reference: github.com/nnzhan/Graph-WaveNet

This is a faithful re-implementation that uses our data pipeline directly
(masked normalization, masked MAE in raw mph).

The model combines:
  - Diffusion convolution on a precomputed transition matrix (from adjacency)
  - Adaptive adjacency matrix learned end-to-end (node embeddings E1, E2)
  - Stacked dilated TCN with gated activation (WaveNet)
  - Skip connections to multi-horizon output head

For METR-LA, the reference paper reports 60-min MAE ~3.51. We use it
primarily as a *diverse* ensemble member (TCN + adaptive graph is a
completely different inductive bias from STAEformer's adaptive embedding +
full attention), so the diversity dimension matters more than the absolute
strength.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Diffusion convolution + adaptive adjacency."""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.3, support_len: int = 3, order: int = 2):
        super().__init__()
        c_in_total = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in_total, c_out)
        self.dropout = dropout
        self.order = order

    @staticmethod
    def nconv(x, A):
        # x: [B, C, N, T], A: [N, N]
        return torch.einsum("bcnt,nm->bcmt", x, A)

    def forward(self, x, supports):
        """
        x: [B, C, N, T]
        supports: list of [N, N] adjacency variants
        """
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)                                          # [B, C*(order*K+1), N, T]
        h = h.permute(0, 2, 3, 1)                                          # [B, N, T, C']
        h = self.mlp(h)
        h = h.permute(0, 3, 1, 2)                                          # [B, C_out, N, T]
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet with adaptive adjacency.

    Inputs:
      x_norm: [B, T_in=12, N=207]   masked-normalized speeds
      tod:    [B, T_in]              time-of-day in [0,1) (used as extra channel)
      dow:    [B, T_in]              day-of-week int (used as extra channel)
    Returns:
      y_hat_norm: [B, T_out=12, N]
    """

    def __init__(
        self,
        N: int,
        adj_mx: torch.Tensor,        # symmetric, [N, N]
        in_steps: int = 12,
        out_steps: int = 12,
        in_dim: int = 3,             # speed + tod + dow as channels
        out_dim: int = 1,            # speed prediction
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        dropout: float = 0.3,
        adaptive_adj: bool = True,
        node_emb_dim: int = 10,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.dropout = dropout

        # Build diffusion supports (forward + backward transition)
        # adj_mx is already symmetric in our case (we symmetrize at preprocess).
        # Compute D^-1 @ A as transition, and (D^-1 @ A^T) for the reverse direction.
        # Since symmetric, both are equal — but we keep two slots for compatibility.
        rowsum = adj_mx.sum(dim=1).clamp(min=1e-6)
        T_fwd = adj_mx / rowsum.unsqueeze(1)
        T_bwd = adj_mx / adj_mx.sum(dim=0).clamp(min=1e-6).unsqueeze(0)
        self.register_buffer("T_fwd", T_fwd)
        self.register_buffer("T_bwd", T_bwd)

        self.adaptive_adj = adaptive_adj
        if adaptive_adj:
            self.E1 = nn.Parameter(torch.randn(N, node_emb_dim) * 0.01)
            self.E2 = nn.Parameter(torch.randn(N, node_emb_dim) * 0.01)
            support_len = 3
        else:
            support_len = 2

        # Start conv
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        # Stacked TCN blocks
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gcn = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gcn.append(GCN(dilation_channels, residual_channels,
                                    dropout=dropout, support_len=support_len))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        # End network: skip -> end_conv1 -> end_conv2 -> output
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, out_steps * out_dim, kernel_size=(1, 1))

    def get_adaptive_adj(self):
        """Returns the dynamically learned adjacency."""
        # softmax(relu(E1 @ E2.T)) — the original GWNet recipe
        a = torch.mm(self.E1, self.E2.t())
        a = F.relu(a)
        a = F.softmax(a, dim=1)
        return a

    def forward(self, x_norm, tod, dow, y_tod=None, y_dow=None):
        """
        x_norm: [B, T_in, N]
        tod:    [B, T_in]
        dow:    [B, T_in]
        returns: [B, T_out, N]
        """
        B, T_in, N = x_norm.shape
        # Stack channels: [B, in_dim, N, T_in]
        x = torch.stack([
            x_norm.transpose(1, 2),                                     # [B, N, T_in]
            tod.unsqueeze(1).expand(B, N, T_in),                        # [B, N, T_in]
            dow.float().unsqueeze(1).expand(B, N, T_in),                # [B, N, T_in]
        ], dim=1)                                                        # [B, 3, N, T_in]
        # GWNet expects channels first, sequence last: [B, C, N, T]
        # x already in that shape

        # Pad to receptive field
        if T_in < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T_in, 0, 0, 0))

        x = self.start_conv(x)                                           # [B, residual_channels, N, T_eff]
        skip = 0

        supports = [self.T_fwd, self.T_bwd]
        if self.adaptive_adj:
            supports = supports + [self.get_adaptive_adj()]

        for i in range(len(self.filter_convs)):
            residual = x
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filt * gate

            # skip connection
            s = self.skip_convs[i](x)
            s = s[:, :, :, -1:]                                          # take last time step
            skip = s + (skip if isinstance(skip, torch.Tensor) else 0)

            x = self.gcn[i](x, supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)                                            # [B, T_out, N, 1]
        # Reshape from [B, T_out*out_dim, N, 1] to [B, T_out, N]
        out = x[:, :, :, 0]                                              # [B, T_out, N]
        return out
