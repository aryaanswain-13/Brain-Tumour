import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from configs.config import cfg
from src.data.dataset import build_ucsf_dicts, stratified_split, CachedDataset
from src.models.classifier import BrainTumorClassifier
from src.training.train_clf import train_classifier
from src.evaluation.metrics import (
    compute_classification_metrics, compute_dice, compute_hd95,
    compute_iou, compute_precision_seg, compute_recall_seg, keep_largest_component
)

def evaluate_classifier(clf_model, test_loader, device=cfg.DEVICE):
    clf_model.eval()
    res = {t: {'probs': [], 'labels': []} for t in ['idh', 'grade', 'mgmt']}
    
    with torch.no_grad():
        for batch in test_loader:
            enc  = batch['encoder'].to(device)
            mor  = batch['morph'].to(device)
            conf = batch['conf_score'].to(device)
            
            preds = clf_model(enc, mor, conf)
            for t in ['idh', 'grade', 'mgmt']:
                res[t]['probs'].extend(torch.sigmoid(preds[t]).squeeze(-1).cpu().tolist())
                res[t]['labels'].extend(batch[t].squeeze(-1).cpu().tolist())
                
    metrics = {}
    names = {'idh': 'IDH Status', 'grade': 'WHO CNS Grade', 'mgmt': 'MGMT Promoter'}
    for t in ['idh', 'grade', 'mgmt']:
        probs  = np.array(res[t]['probs'])
        labels = np.array(res[t]['labels'])
        task_metrics = compute_classification_metrics(labels, probs)
        if task_metrics:
            task_metrics['name'] = names[t]
            task_metrics['n'] = len(task_metrics['labels'])
            metrics[t] = task_metrics
            
    return metrics

def evaluate_segmentation(full_cache_dir, csv_path, ucsf_dir):
    all_dicts = build_ucsf_dicts(csv_path, ucsf_dir)
    cached = {f.replace('.pt', '') for f in os.listdir(full_cache_dir) if f.endswith('.pt')}
    all_dicts = [d for d in all_dicts if d['patient_id'] in cached]
    
    _, _, test_d = stratified_split(all_dicts)
    res = {r: {'dice': [], 'hd95': [], 'iou': [], 'prec': [], 'rec': []} for r in ['ET', 'TC', 'WT']}
    
    for item in tqdm(test_d, desc="Segmentation Evaluation"):
        path = os.path.join(full_cache_dir, f"{item['patient_id']}.pt")
        if not os.path.exists(path):
            continue
        try:
            d = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
            
        if 'mean_pred' not in d:
            continue
            
        mp  = d['mean_pred'].numpy()
        lab = d['label'].squeeze().numpy()
        
        for name, pred, true in [
            ('ET', mp[0] > cfg.SEG_THRESHOLD, lab == 3),
            ('TC', mp[1] > cfg.SEG_THRESHOLD, (lab == 1) | (lab == 3)),
            ('WT', mp[2] > cfg.SEG_THRESHOLD, lab >= 1)
        ]:
            pc = keep_largest_component(pred)
            res[name]['dice'].append(compute_dice(pc, true))
            res[name]['hd95'].append(compute_hd95(pc, true))
            res[name]['iou'].append(compute_iou(pc, true))
            res[name]['prec'].append(compute_precision_seg(pc, true))
            res[name]['rec'].append(compute_recall_seg(pc, true))
            
    seg_summary = {}
    for r in ['WT', 'TC', 'ET']:
        seg_summary[r] = {}
        for m in ['dice', 'hd95', 'iou', 'prec', 'rec']:
            vals = [x for x in res[r][m] if not np.isnan(x)]
            seg_summary[r][m] = np.mean(vals) if vals else float('nan')
            seg_summary[r][f'{m}_std'] = np.std(vals) if vals else 0.0
        seg_summary[r]['n'] = len([x for x in res[r]['dice'] if not np.isnan(x)])
        
    return seg_summary

def run_ablation(cache_dir, csv_path, ucsf_dir, epochs=cfg.CLF_EPOCHS, device=cfg.DEVICE):
    all_dicts = build_ucsf_dicts(csv_path, ucsf_dir)
    cached_pids = {f.replace('.pt', '') for f in os.listdir(cache_dir) if f.endswith('.pt')}
    all_dicts = [d for d in all_dicts if d['patient_id'] in cached_pids]
    
    train_d, val_d, test_d = stratified_split(all_dicts)
    
    def get_paths(dicts):
        return [os.path.join(cache_dir, f"{d['patient_id']}.pt") for d in dicts]
        
    train_loader = DataLoader(CachedDataset(get_paths(train_d)), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CachedDataset(get_paths(val_d)), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CachedDataset(get_paths(test_d)), batch_size=1, shuffle=False)
    
    results = {}
    for mode in ['A', 'B', 'C', 'D']:
        print(f"\n=======================================================\n  ABLATION VARIANT {mode}\n=======================================================")
        model = BrainTumorClassifier(gate_mode=mode).to(device)
        
        best_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_best_ablation_{mode}.pth')
        latest_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_latest_ablation_{mode}.pth')
        
        model, hist = train_classifier(
            model, train_loader, val_loader, 
            epochs=epochs, best_ckpt_path=best_ckpt, 
            latest_ckpt_path=latest_ckpt, device=device
        )
        
        test_metrics = evaluate_classifier(model, test_loader, device=device)
        results[mode] = {
            'history': hist,
            'test_metrics': test_metrics
        }
    return results

def run_five_fold_cv(cache_dir, csv_path, ucsf_dir, seed=42, epochs=cfg.CLF_EPOCHS, device=cfg.DEVICE):
    import random
    
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
            
    all_dicts = build_ucsf_dicts(csv_path, ucsf_dir)
    cached_pids = {f.replace('.pt', '') for f in os.listdir(cache_dir) if f.endswith('.pt')}
    all_dicts = [d for d in all_dicts if d['patient_id'] in cached_pids]
    
    print(f"Total patients loaded for 5-Fold CV: {len(all_dicts)}")
    
    labels = np.array([0.0 if d['idh'] == 0.0 else 1.0 if d['idh'] == 1.0 else -1.0 for d in all_dicts])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(all_dicts, labels)):
        print(f"\n=======================================================\n  FOLD {fold+1}/5\n=======================================================")
        set_seed(seed + fold)
        
        train_val = [all_dicts[i] for i in train_val_idx]
        test_d    = [all_dicts[i] for i in test_idx]
        
        n_val = max(1, int(len(train_val) * 0.15))
        val_d   = train_val[:n_val]
        train_d = train_val[n_val:]
        
        print(f"Split: Train={len(train_d)} | Val={len(val_d)} | Test={len(test_d)}")
        
        def get_paths(dicts):
            return [os.path.join(cache_dir, f"{d['patient_id']}.pt") for d in dicts]
            
        train_loader = DataLoader(CachedDataset(get_paths(train_d)), batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(CachedDataset(get_paths(val_d)), batch_size=cfg.BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(CachedDataset(get_paths(test_d)), batch_size=1, shuffle=False)
        
        clf = BrainTumorClassifier().to(device)
        
        best_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_best_cv_fold{fold+1}.pth')
        latest_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_latest_cv_fold{fold+1}.pth')
        
        clf, hist = train_classifier(
            clf, train_loader, val_loader, 
            epochs=epochs, best_ckpt_path=best_ckpt, 
            latest_ckpt_path=latest_ckpt, device=device
        )
        
        metrics = evaluate_classifier(clf, test_loader, device=device)
        fold_results.append(metrics)
        
    return fold_results
