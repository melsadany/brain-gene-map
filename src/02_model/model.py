"""Hierarchical multi-output INR for brain gene expression"""

import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """SIREN layer with sine activation"""
    def __init__(self, dim_in, dim_out, w0=1.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.w0 = w0
        self.is_first = is_first
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                           1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.w0,
                                           np.sqrt(6 / self.linear.in_features) / self.w0)
    
    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SIRENEncoder(nn.Module):
    """SIREN-based spatial encoder"""
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0_first, w0_hidden):
        super().__init__()
        
        layers = []
        layers.append(SineLayer(dim_in, dim_hidden, w0=w0_first, is_first=True))
        
        for _ in range(num_layers - 2):
            layers.append(SineLayer(dim_hidden, dim_hidden, w0=w0_hidden))
        
        # Last layer: linear (no sine)
        layers.append(nn.Linear(dim_hidden, dim_out))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class HierarchicalGeneINR(nn.Module):
    """
    Multi-output INR with hierarchical structure-specific encoders
    and structure-specific gene decoders
    """
    def __init__(self, config, num_genes):
        super().__init__()
        
        self.config = config
        self.num_genes = num_genes
        
        # Three structure-specific spatial encoders
        self.encoder_ctx = SIRENEncoder(
            dim_in=3,
            dim_hidden=config.hidden_dim,
            dim_out=config.spatial_dim,
            num_layers=config.num_layers,
            w0_first=config.w0_first,
            w0_hidden=config.w0_hidden
        )
        
        self.encoder_crb = SIRENEncoder(
            dim_in=3,
            dim_hidden=config.hidden_dim,
            dim_out=config.spatial_dim,
            num_layers=config.num_layers,
            w0_first=config.w0_first,
            w0_hidden=config.w0_hidden
        )
        
        self.encoder_sctx = SIRENEncoder(
            dim_in=3,
            dim_hidden=config.hidden_dim,
            dim_out=config.spatial_dim,
            num_layers=config.num_layers,
            w0_first=config.w0_first,
            w0_hidden=config.w0_hidden
        )
        
        # Three structure-specific gene decoders (with intermediate layers)
        self.gene_decoder_ctx = nn.Sequential(
            nn.Linear(config.spatial_dim, config.gene_latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gene_latent_dim * 2, config.gene_latent_dim),
            nn.ReLU(),
            nn.Linear(config.gene_latent_dim, num_genes)
        )
        
        self.gene_decoder_crb = nn.Sequential(
            nn.Linear(config.spatial_dim, config.gene_latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gene_latent_dim * 2, config.gene_latent_dim),
            nn.ReLU(),
            nn.Linear(config.gene_latent_dim, num_genes)
        )
        
        self.gene_decoder_sctx = nn.Sequential(
            nn.Linear(config.spatial_dim, config.gene_latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gene_latent_dim * 2, config.gene_latent_dim),
            nn.ReLU(),
            nn.Linear(config.gene_latent_dim, num_genes)
        )
        
        print(f"Model initialized:")
        print(f"  Spatial encoders: 3 × {config.num_layers} layers × {config.hidden_dim} units")
        print(f"  Gene decoders: 3 × ({config.spatial_dim} → {config.gene_latent_dim * 2} → {config.gene_latent_dim} → {num_genes})")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, inputs):
        """
        Args:
            inputs: (batch, 4) - [x_norm, y_norm, z_norm, structure_idx]
        
        Returns:
            expression: (batch, num_genes)
        """
        coords = inputs[:, :3]  # (batch, 3)
        structure_idx = inputs[:, 3].long()  # (batch,)
        
        # Encode spatial features for each structure
        spatial_ctx = self.encoder_ctx(coords)      # (batch, spatial_dim)
        spatial_crb = self.encoder_crb(coords)      # (batch, spatial_dim)
        spatial_sctx = self.encoder_sctx(coords)    # (batch, spatial_dim)
        
        # Decode with matching structure-specific decoder
        expr_ctx = self.gene_decoder_ctx(spatial_ctx)   # (batch, num_genes)
        expr_crb = self.gene_decoder_crb(spatial_crb)   # (batch, num_genes)
        expr_sctx = self.gene_decoder_sctx(spatial_sctx)  # (batch, num_genes)
        
        # Stack and select based on structure index
        expr_all = torch.stack([expr_ctx, expr_crb, expr_sctx], dim=1)  # (batch, 3, num_genes)
        idx = structure_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_genes)  # (batch, 1, num_genes)
        expression = torch.gather(expr_all, 1, idx).squeeze(1)  # (batch, num_genes)
        
        return expression
