import torch
import torch.nn as nn
import torch.nn.functional as F

class FallbackMambaBlock(nn.Module):
    """
    A lightweight fallback sequence-mixing block inspired by Mamba.
    Since mamba_ssm requires custom CUDA kernels that are hard to install on some environments,
    this fallback uses a causal Conv1d + an RNN (to simulate the selective state space scan) + gating.
    It has a similar macroscopic structure (Expansion, Conv, Sequence Mixing, Gating, Projection).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Simulate selective SSM with an RNN (GRU) to capture sequence mixing
        self.ssm_fallback = nn.GRU(self.d_inner, self.d_inner, batch_first=True)
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        x_proj = self.in_proj(x)
        x_mix, z = x_proj.chunk(2, dim=-1) # [B, L, d_inner]
        
        # 1D Causal Conv
        x_mix = x_mix.transpose(1, 2) # [B, d_inner, L]
        x_mix = self.conv1d(x_mix)[:, :, :L] # truncate padding to make it causal
        x_mix = x_mix.transpose(1, 2) # [B, L, d_inner]
        
        x_mix = F.silu(x_mix)
        
        # SSM fallback (sequence mixing)
        x_mix, _ = self.ssm_fallback(x_mix)
        
        # Gating
        out = x_mix * F.silu(z)
        
        out = self.out_proj(out)
        return out

class SpectralMamba(nn.Module):
    def __init__(self, k: int, hidden_dim: int = 128, pred_len: int = 12, num_layers: int = 2):
        super().__init__()
        self.k = k
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(k, hidden_dim)
        self.layers = nn.ModuleList([
            FallbackMambaBlock(d_model=hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Head
        self.head = nn.Linear(hidden_dim, pred_len * k)
        
    def forward(self, x):
        """
        x: [B, L, k]
        returns: [B, pred_len, k]
        """
        x = self.embedding(x)
        
        for layer in self.layers:
            x = x + layer(x)
            
        x = self.norm(x)
        
        # Take the last time step for prediction
        x_last = x[:, -1, :] # [B, hidden_dim]
        out = self.head(x_last) # [B, pred_len * k]
        out = out.view(-1, self.pred_len, self.k)
        return out
