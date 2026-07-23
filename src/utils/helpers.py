import random
import numpy as np
import torch
from scipy import ndimage
from configs.config import cfg

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_morphological_features(mean_pred):
    et = (mean_pred[0] > cfg.SEG_THRESHOLD).numpy().astype(np.float32)
    tc = (mean_pred[1] > cfg.SEG_THRESHOLD).numpy().astype(np.float32)
    wt = (mean_pred[2] > cfg.SEG_THRESHOLD).numpy().astype(np.float32)
    
    edema = np.clip(wt - tc, 0, 1)
    
    tumor_vol = float(wt.sum())
    core_vol  = float(tc.sum())
    edema_vol = float(edema.sum())
    enh_vol   = float(et.sum())
    
    edema_core_ratio = edema_vol / (core_vol + 1e-6)
    
    eroded  = ndimage.binary_erosion(wt.astype(bool))
    surface = wt.astype(bool) & ~eroded
    sa      = float(surface.sum())
    
    compactness = tumor_vol / (sa**1.5 + 1e-6) if sa > 0 else 0.0
    sphericity  = ((np.pi**(1/3) * (6 * tumor_vol)**(2/3)) / (sa + 1e-6)
                   if (sa > 0 and tumor_vol > 0) else 0.0)
                   
    return {
        'tumor_volume': tumor_vol,
        'core_volume': core_vol,
        'edema_volume': edema_vol,
        'enhancing_volume': enh_vol,
        'edema_core_ratio': edema_core_ratio,
        'surface_area': sa,
        'compactness': compactness,
        'sphericity': sphericity,
    }

def morphology_to_tensor(d):
    t = torch.tensor(list(d.values()), dtype=torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t[:4] = torch.log1p(t[:4])
    t[5] = torch.log1p(t[5])
    t[4] = torch.clamp(t[4], 0, 10)
    t[6] = torch.clamp(t[6], 0, 10)
    t[7] = torch.clamp(t[7], 0, 2)
    
    return t
