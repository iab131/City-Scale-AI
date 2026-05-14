import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class SpectralMambaReal(nn.Module):
    def __init__(self, k: int, pred_len: int = 12, d_model: int = 64, num_layers: int = 2, 
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.pred_len = pred_len
        self.d_model = d_model
        
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed. Please install it to use this model.")
            
        self.embedding = nn.Linear(k, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ),
                'norm': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * k)
        
    def forward(self, x):
        """
        x: [B, L, k]
        returns: [B, pred_len, k]
        """
        # Embed spectral features into d_model
        x = self.embedding(x) # [B, L, d_model]
        
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['mixer'](x)
            x = layer['dropout'](x)
            x = residual + x
            
        x = self.norm_f(x)
        
        # We use the final sequence step to predict the future sequence
        x_last = x[:, -1, :] # [B, d_model]
        out = self.head(x_last) # [B, pred_len * k]
        out = out.view(-1, self.pred_len, self.k) # [B, pred_len, k]
        
        return out
