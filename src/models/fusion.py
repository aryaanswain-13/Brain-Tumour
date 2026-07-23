import torch
import torch.nn as nn
from configs.config import cfg

class ConfidenceGatedFusion(nn.Module):
    def __init__(self, encoder_dim=cfg.ENCODER_DIM, morph_dim=cfg.MORPH_FEATURES, fused_dim=cfg.FUSED_DIM, mode=cfg.GATE_MODE):
        super().__init__()
        self.mode = mode
        
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, fused_dim),
        )
        
        self.morph_gate = nn.Linear(morph_dim, fused_dim)

    def forward(self, encoder_features, morph_features, confidence_score):
        enc   = self.encoder_proj(encoder_features)
        morph = self.morph_proj(morph_features)
        conf  = confidence_score.view(-1, 1)
        
        if self.mode == 'A':
            return enc
            
        elif self.mode == 'B':
            G = 0.5
            return G * enc + (1.0 - G) * morph
            
        elif self.mode == 'C':
            G = conf
            return G * enc + (1.0 - G) * morph
            
        elif self.mode == 'D':
            morph_gate = torch.sigmoid(self.morph_gate(morph_features))
            G = conf * morph_gate
            return G * enc + (1.0 - G) * morph
            
        else:
            raise ValueError(f"Unknown GATE_MODE selection: {self.mode}")
