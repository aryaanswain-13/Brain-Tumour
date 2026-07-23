import torch
import torch.nn as nn
from configs.config import cfg

IDH_W   = torch.tensor([2.5])
GRADE_W = torch.tensor([92.0 / 395.0])
MGMT_W  = torch.tensor([0.395])

TASK_W = {'idh': 1.0, 'grade': 1.0, 'mgmt': 0.8}

class MultiTaskLoss(nn.Module):
    def __init__(self, device=cfg.DEVICE):
        super().__init__()
        self.idh_crit   = nn.BCEWithLogitsLoss(pos_weight=IDH_W.to(device))
        self.grade_crit = nn.BCEWithLogitsLoss(pos_weight=GRADE_W.to(device))
        self.mgmt_crit  = nn.BCEWithLogitsLoss(pos_weight=MGMT_W.to(device))

    def forward(self, preds, labels):
        loss = 0.0
        
        idh_valid = ~torch.isnan(labels['idh'])
        if idh_valid.any():
            loss += TASK_W['idh'] * self.idh_crit(
                preds['idh'][idh_valid], labels['idh'][idh_valid].float()
            )
            
        grade_valid = ~torch.isnan(labels['grade'])
        if grade_valid.any():
            loss += TASK_W['grade'] * self.grade_crit(
                preds['grade'][grade_valid], labels['grade'][grade_valid].float()
            )
            
        mgmt_valid = ~torch.isnan(labels['mgmt'])
        if mgmt_valid.any():
            loss += TASK_W['mgmt'] * self.mgmt_crit(
                preds['mgmt'][mgmt_valid], labels['mgmt'][mgmt_valid].float()
            )
            
        return loss

def brats_label_to_channels(label):
    et = (label == 3).float()
    tc = ((label == 1) | (label == 3)).float()
    wt = (label >= 1).float()
    return torch.cat([et, tc, wt], dim=1)
