import os
import torch
from tqdm import tqdm
from configs.config import cfg
from src.training.losses import MultiTaskLoss

def train_classifier(clf_model, train_loader, val_loader,
                     epochs=cfg.CLF_EPOCHS, patience=20,
                     best_ckpt_path=None, latest_ckpt_path=None,
                     device=cfg.DEVICE):
    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = MultiTaskLoss(device=device)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    no_improve = 0
    best_state = None
    
    for path in [best_ckpt_path, latest_ckpt_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
                
    for epoch in range(epochs):
        clf_model.train()
        ep_loss, ep_corr, ep_tot = 0.0, 0, 0
        
        for batch in train_loader:
            enc  = batch['encoder'].to(device)
            mor  = batch['morph'].to(device)
            conf = batch['conf_score'].to(device)
            lbl  = {k: batch[k].to(device) for k in ['idh', 'grade', 'mgmt']}
            
            optimizer.zero_grad()
            preds = clf_model(enc, mor, conf)
            loss  = loss_fn(preds, lbl)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            
            v = ~torch.isnan(lbl['idh'])
            if v.any():
                p = torch.sigmoid(preds['idh'][v]) > 0.5
                t = lbl['idh'][v] > 0.5
                ep_corr += (p == t).sum().item()
                ep_tot  += v.sum().item()
                
        scheduler.step()
        tr_loss = ep_loss / len(train_loader)
        tr_acc  = ep_corr / ep_tot if ep_tot > 0 else 0.0
        
        clf_model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                enc  = batch['encoder'].to(device)
                mor  = batch['morph'].to(device)
                conf = batch['conf_score'].to(device)
                lbl  = {k: batch[k].to(device) for k in ['idh', 'grade', 'mgmt']}
                
                preds = clf_model(enc, mor, conf)
                loss  = loss_fn(preds, lbl)
                val_loss += loss.item()
                
                v = ~torch.isnan(lbl['idh'])
                if v.any():
                    p = torch.sigmoid(preds['idh'][v]) > 0.5
                    t = lbl['idh'][v] > 0.5
                    val_corr += (p == t).sum().item()
                    val_tot  += v.sum().item()
                    
        va_loss = val_loss / len(val_loader)
        va_acc  = val_corr / val_tot if val_tot > 0 else 0.0
        
        for k, val in [('train_loss', tr_loss), ('val_loss', va_loss),
                       ('train_acc', tr_acc), ('val_acc', va_acc)]:
            history[k].append(val)
            
        print(f"Ep {epoch+1:2d}/{epochs} | Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | Val loss: {va_loss:.4f} acc: {va_acc:.3f}")
        
        if latest_ckpt_path:
            torch.save({
                'epoch': epoch,
                'model_state': clf_model.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss
            }, latest_ckpt_path)
            
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in clf_model.state_dict().items()}
            if best_ckpt_path:
                torch.save({
                    'epoch': epoch,
                    'model_state': clf_model.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history
                }, best_ckpt_path)
            print(f"  ✓ Best saved (val loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    if best_state:
        clf_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
    return clf_model, history
