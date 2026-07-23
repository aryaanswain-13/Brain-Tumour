import os
import gc
import torch
from tqdm import tqdm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from configs.config import cfg
from src.training.losses import brats_label_to_channels

def set_finetune_layers(model):
    for p in model.parameters():
        p.requires_grad = False
        
    for p in model.swinViT.layers3.parameters():
        p.requires_grad = True
        
    for p in model.swinViT.layers4.parameters():
        p.requires_grad = True
        
    for name, p in model.named_parameters():
        if any(k in name for k in [
            'encoder10', 'encoder1', 'encoder2',
            'encoder3', 'encoder4',
            'decoder5', 'decoder4', 'decoder3',
            'decoder2', 'decoder1', 'out'
        ]):
            p.requires_grad = True
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Fine-tuning layers unfrozen. Trainable parameters: {trainable/1e6:.2f}M | Frozen: {frozen/1e6:.2f}M")

def train_segmentation(model, train_loader, val_loader, 
                       epochs=cfg.FINETUNE_EPOCHS, 
                       lr=cfg.FINETUNE_LR, 
                       ckpt_path=os.path.join(cfg.CKPT_DIR, 'swinunetr_ucsf_finetuned.pt'),
                       latest_ckpt_path=os.path.join(cfg.CKPT_DIR, 'swinunetr_ucsf_latest.pt'),
                       patience=25,
                       resume=True,
                       device=cfg.DEVICE):
    set_finetune_layers(model)
    model.train()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    best_val_dice = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_dice': []}
    
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming fine-tuning from best checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        best_val_dice = ckpt['val_dice']
        history = ckpt.get('history', {'train_loss': [], 'val_dice': []})
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (Best Val Dice so far: {best_val_dice:.4f})")
        
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    dice_metric = DiceMetric(
        include_background=False,
        reduction=MetricReduction.MEAN,
        get_not_nans=True
    )
    
    print(f"Starting segmentation fine-tuning from epoch {start_epoch} to {epochs}...")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (image, label) in enumerate(progress_bar):
            image = image.to(device)
            label = label.to(device)
            targets = brats_label_to_channels(label)
            
            logits = model(image)
            loss = dice_loss(logits, targets) / cfg.ACCUMULATION_STEPS
            loss.backward()
            
            if ((i + 1) % cfg.ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for image, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                image = image.to(device)
                label = label.to(device)
                targets = brats_label_to_channels(label)
                
                logits = model(image)
                preds = (torch.sigmoid(logits) > cfg.SEG_THRESHOLD).float()
                dice_metric(y_pred=preds, y=targets)
                
        val_dice, _ = dice_metric.aggregate()
        mean_val_dice = val_dice.mean().item()
        
        history['train_loss'].append(avg_train_loss)
        history['val_dice'].append(mean_val_dice)
        
        print(f"Epoch {epoch+1:3d} | Train loss: {avg_train_loss:.4f} | Val Dice: {mean_val_dice:.4f}")
        
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_dice': best_val_dice,
                'best_val_dice': best_val_dice,
                'no_improve': no_improve,
                'history': history,
            }, ckpt_path)
            print(f"  ✓ Best model saved (val Dice: {best_val_dice:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement: {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_dice': mean_val_dice,
            'best_val_dice': best_val_dice,
            'no_improve': no_improve,
            'history': history,
        }, latest_ckpt_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
    return history
