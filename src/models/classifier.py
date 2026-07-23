import torch
import torch.nn as nn
from src.models.fusion import ConfidenceGatedFusion
from configs.config import cfg

class BrainTumorClassifier(nn.Module):
    def __init__(self, encoder_dim=cfg.ENCODER_DIM,
                 morph_dim=cfg.MORPH_FEATURES,
                 fused_dim=cfg.FUSED_DIM, 
                 dropout=0.4,
                 gate_mode=cfg.GATE_MODE):
        super().__init__()
        
        self.fusion = ConfidenceGatedFusion(
            encoder_dim=encoder_dim, 
            morph_dim=morph_dim, 
            fused_dim=fused_dim, 
            mode=gate_mode
        )
        
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.head_idh   = nn.Linear(256, 1)
        self.head_grade = nn.Linear(256, 1)
        self.head_mgmt  = nn.Linear(256, 1)

    def forward(self, encoder_features, morph_features, confidence_score):
        fused  = self.fusion(encoder_features, morph_features, confidence_score)
        shared = self.shared(fused)
        
        return {
            'idh':   self.head_idh(shared),
            'grade': self.head_grade(shared),
            'mgmt':  self.head_mgmt(shared),
        }
