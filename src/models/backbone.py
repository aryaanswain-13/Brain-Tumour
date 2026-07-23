import os
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from configs.config import cfg

def build_swinunetr(weights_path=None):
    model = SwinUNETR(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        feature_size=cfg.FEATURE_SIZE,
        use_checkpoint=True,
    ).to(cfg.DEVICE)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading SwinUNETR checkpoint: {weights_path}")
        ckpt = torch.load(weights_path, map_location=cfg.DEVICE, weights_only=False)
        
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
            
        cleaned = {}
        for k, v in state.items():
            new_k = k
            for prefix in ['module.', 'model.', 'backbone.']:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            cleaned[new_k] = v
            
        model_dict = model.state_dict()
        matched     = {k: v for k, v in cleaned.items() if k in model_dict and model_dict[k].shape == v.shape}
        unmatched   = {k: v for k, v in cleaned.items() if k not in model_dict}
        wrong_shape = {k: v for k, v in cleaned.items() if k in model_dict and model_dict[k].shape != v.shape}
        
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        
        epoch_str = f"Epoch: {ckpt['epoch']}" if 'epoch' in ckpt else "Epoch: N/A"
        dice_str  = f"Dice: {ckpt['best_acc']:.4f}" if 'best_acc' in ckpt else (f"Dice: {ckpt['val_dice']:.4f}" if 'val_dice' in ckpt else "Dice: N/A")
        print(f"{epoch_str} | {dice_str}")
        
    if cfg.FREEZE_BACKBONE:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        print("SwinUNETR Backbone FROZEN (eval mode)")
    else:
        print("SwinUNETR Backbone TRAINABLE")
        
    return model

def extract_encoder_features(model, image_tensor):
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(cfg.DEVICE)
        hidden_states = model.swinViT(x)
        bottleneck = hidden_states[4]
        features = bottleneck.mean(dim=[2, 3, 4])
    return features.squeeze(0).cpu()

def tta_predict(model, image_tensor, n_runs=cfg.TTA_RUNS):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_runs):
            aug = image_tensor.clone()
            flipped_axes = []
            
            for axis in [1, 2, 3]:
                if torch.rand(1).item() > 0.5:
                    aug = torch.flip(aug, dims=[axis])
                    flipped_axes.append(axis)
                    
            aug = aug + (torch.rand(1).item() - 0.5) * 0.2
            logits = model(aug.unsqueeze(0).to(cfg.DEVICE))
            probs = torch.sigmoid(logits).squeeze(0).cpu()
            
            for axis in reversed(flipped_axes):
                probs = torch.flip(probs, dims=[axis])
            preds.append(probs)
            
    stack = torch.stack(preds, dim=0)
    mean_pred = stack.mean(dim=0)
    variance = stack.var(dim=0).mean(dim=0, keepdim=True)
    vmin, vmax = variance.min(), variance.max()
    norm_var = (variance - vmin) / (vmax - vmin + 1e-8)
    
    return mean_pred, 1.0 - norm_var, norm_var
