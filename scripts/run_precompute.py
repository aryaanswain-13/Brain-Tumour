import os
import gc
import sys
import argparse
import shutil
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.config import cfg
from src.data.dataset import build_ucsf_dicts
from src.data.transforms import get_ucsf_transforms
from src.models.backbone import build_swinunetr, extract_encoder_features, tta_predict
from src.utils.helpers import extract_morphological_features, morphology_to_tensor, set_seed

def main():
    parser = argparse.ArgumentParser(description="Precompute and cache 3D SwinUNETR features.")
    parser.add_argument("--csv", type=str, default=cfg.UCSF_CSV, help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default=cfg.UCSF_DIR, help="Path to UCSF directory")
    parser.add_argument("--ckpt", type=str, default=os.path.join(cfg.CKPT_DIR, 'swinunetr_ucsf_finetuned.pt'), 
                        help="Path to fine-tuned SwinUNETR model checkpoint")
    parser.add_argument("--cache_dir", type=str, default=cfg.CACHE_DIR, help="Directory to save full cache")
    parser.add_argument("--slim_dir", type=str, default=cfg.SLIM_CACHE, help="Directory to save slim cache")
    parser.add_argument("--tta_runs", type=int, default=cfg.TTA_RUNS, help="Number of TTA runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clear_existing", action="store_true", help="Clear existing cache directory")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    cfg.UCSF_CSV = args.csv
    cfg.UCSF_DIR = args.data_dir
    cfg.TTA_RUNS = args.tta_runs
    
    all_dicts = build_ucsf_dicts(args.csv, args.data_dir)
    
    if not os.path.exists(args.ckpt):
        print(f"WARNING: Fine-tuned checkpoint not found at {args.ckpt}. Loading BraTS weights.")
        model = build_swinunetr(weights_path=cfg.WEIGHTS)
    else:
        model = build_swinunetr(weights_path=args.ckpt)
        
    model.eval()
    
    if args.clear_existing:
        for d in [args.cache_dir, args.slim_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
                
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.slim_dir, exist_ok=True)
    
    transforms = get_ucsf_transforms('val')
    
    computed, skipped, failed = 0, 0, 0
    print("\nStarting full precomputation cache build...")
    
    for item in tqdm(all_dicts, desc="Caching 3D features"):
        pid = item['patient_id']
        path = os.path.join(args.cache_dir, f"{pid}.pt")
        
        if os.path.exists(path):
            skipped += 1
            continue
            
        try:
            sample = transforms({
                't1':    item['t1'], 
                't1c':   item['t1c'],
                't2':    item['t2'], 
                'flair': item['flair'],
                'label': item['label'],
            })
            
            image = sample['image']
            enc_feat = extract_encoder_features(model, image)
            mean_pred, conf_map, _ = tta_predict(model, image, args.tta_runs)
            conf_score = torch.tensor([conf_map.mean().item()])
            morph_feats = extract_morphological_features(mean_pred)
            morph_tensor = morphology_to_tensor(morph_feats)
            
            torch.save({
                'patient_id': pid,
                'encoder':    enc_feat,
                'morph':      morph_tensor,
                'conf_score': conf_score,
                'mean_pred':  mean_pred,
                'label':      sample['label'],
                'idh':        torch.tensor([item['idh']], dtype=torch.float32),
                'grade':      torch.tensor([item['grade']], dtype=torch.float32),
                'mgmt':       torch.tensor([item['mgmt']], dtype=torch.float32),
            }, path)
            
            computed += 1
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\nFailed to process patient {pid}: {e}")
            failed += 1
            
    print("\nRebuilding lightweight memory-slim cache...")
    slim_computed, slim_failed = 0, 0
    full_files = [f for f in os.listdir(args.cache_dir) if f.endswith('.pt')]
    
    for fname in tqdm(full_files, desc="Building slim cache"):
        try:
            full_path = os.path.join(args.cache_dir, fname)
            slim_path = os.path.join(args.slim_dir, fname)
            d = torch.load(full_path, map_location='cpu', weights_only=False)
            
            torch.save({
                'patient_id': d['patient_id'],
                'encoder':    d['encoder'],
                'morph':      d['morph'],
                'conf_score': d['conf_score'],
                'idh':        d['idh'],
                'grade':      d['grade'],
                'mgmt':       d['mgmt'],
            }, slim_path)
            
            slim_computed += 1
        except Exception as e:
            print(f"\nFailed to create slim cache for {fname}: {e}")
            slim_failed += 1
            
    print("Done precomputing feature cache!")

if __name__ == "__main__":
    main()
