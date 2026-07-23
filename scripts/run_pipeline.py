import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from src.data.dataset import build_ucsf_dicts, stratified_split, CachedDataset
from src.models.classifier import BrainTumorClassifier
from src.training.train_clf import train_classifier
from src.evaluation.evaluator import (
    evaluate_classifier, evaluate_segmentation, run_ablation, run_five_fold_cv
)
from src.evaluation.metrics import run_mcnemar_test
from src.utils.helpers import set_seed

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end classifier training and evaluation pipeline.")
    parser.add_argument("--csv", type=str, default=cfg.UCSF_CSV, help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default=cfg.UCSF_DIR, help="Path to UCSF dataset directory")
    parser.add_argument("--slim_dir", type=str, default=cfg.SLIM_CACHE, help="Path to slim cache folder")
    parser.add_argument("--full_dir", type=str, default=cfg.CACHE_DIR, help="Path to full cache folder")
    parser.add_argument("--epochs", type=int, default=cfg.CLF_EPOCHS, help="Classifier training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds list")
    parser.add_argument("--skip_cv", action="store_true", help="Skip 5-Fold Cross-Validation run")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip ablation study run")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save tables and plot figures")
    
    args = parser.parse_args()
    seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    cfg.UCSF_CSV = args.csv
    cfg.UCSF_DIR = args.data_dir
    cfg.SLIM_CACHE = args.slim_dir
    cfg.CACHE_DIR = args.full_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.slim_dir) or len([f for f in os.listdir(args.slim_dir) if f.endswith('.pt')]) == 0:
        print(f"ERROR: Slim cache not found at {args.slim_dir}. Please run precompute cache scripts first.")
        sys.exit(1)
        
    all_dicts = build_ucsf_dicts(args.csv, args.data_dir)
    cached_pids = {f.replace('.pt', '') for f in os.listdir(args.slim_dir) if f.endswith('.pt')}
    filtered_dicts = [d for d in all_dicts if d['patient_id'] in cached_pids]
    
    train_d, val_d, test_d = stratified_split(filtered_dicts)
    print(f"Stratified Split:  Train={len(train_d)} | Val={len(val_d)} | Test={len(test_d)}")
    
    def get_paths(dicts):
        return [os.path.join(args.slim_dir, f"{d['patient_id']}.pt") for d in dicts]
        
    train_loader = DataLoader(CachedDataset(get_paths(train_d)), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CachedDataset(get_paths(val_d)), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(CachedDataset(get_paths(test_d)), batch_size=1, shuffle=False)
    
    print("\n" + "="*50 + "\n  1. Running Multi-Seed Classifier Evaluation\n" + "="*50)
    seed_results = []
    
    for seed in seeds_list:
        print(f"\nTraining Seed {seed}...")
        set_seed(seed)
        classifier = BrainTumorClassifier(gate_mode=cfg.GATE_MODE).to(cfg.DEVICE)
        
        best_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_best_seed{seed}.pth')
        latest_ckpt = os.path.join(cfg.CKPT_DIR, f'clf_latest_seed{seed}.pth')
        
        classifier, _ = train_classifier(
            classifier, train_loader, val_loader,
            epochs=args.epochs, patience=args.patience,
            best_ckpt_path=best_ckpt, latest_ckpt_path=latest_ckpt,
            device=cfg.DEVICE
        )
        
        metrics = evaluate_classifier(classifier, test_loader, device=cfg.DEVICE)
        row = {'seed': seed}
        for task in ['idh', 'grade', 'mgmt']:
            if task in metrics:
                row[f'{task}_auc']  = metrics[task]['auc']
                row[f'{task}_acc']  = metrics[task]['accuracy']
                row[f'{task}_f1']   = metrics[task]['f1']
                row[f'{task}_sens'] = metrics[task]['sensitivity']
                row[f'{task}_spec'] = metrics[task]['specificity']
        seed_results.append(row)
        
    results_df = pd.DataFrame(seed_results)
    results_path = os.path.join(args.output_dir, "classification_seed_results.csv")
    results_df.to_csv(results_path, index=False)
    print(results_df.round(3))
    
    print("\n" + "="*50 + f"\n  AGGREGATE CLASSIFICATION SUMMARY\n" + "="*50)
    for task in ['idh', 'grade', 'mgmt']:
        print(f"\n{task.upper()} Task Metrics:")
        for metric in ['auc', 'acc', 'f1', 'sens', 'spec']:
            col_name = f'{task}_{metric}'
            if col_name in results_df.columns:
                m = results_df[col_name].mean()
                s = results_df[col_name].std()
                print(f"  {metric:<10}: {m:.3f} \u00B1 {s:.3f}")
                
    if not args.skip_cv:
        print("\n" + "="*50 + "\n  2. Running 5-Fold Stratified Cross-Validation\n" + "="*50)
        cv_results = run_five_fold_cv(
            args.slim_dir, args.csv, args.data_dir, 
            seed=42, epochs=args.epochs, device=cfg.DEVICE
        )
        
        cv_rows = []
        for fold, fold_metrics in enumerate(cv_results):
            row = {'fold': fold + 1}
            for task in ['idh', 'grade', 'mgmt']:
                if task in fold_metrics:
                    row[f'{task}_auc']  = fold_metrics[task]['auc']
                    row[f'{task}_acc']  = fold_metrics[task]['accuracy']
                    row[f'{task}_f1']   = fold_metrics[task]['f1']
                    row[f'{task}_sens'] = fold_metrics[task]['sensitivity']
                    row[f'{task}_spec'] = fold_metrics[task]['specificity']
            cv_rows.append(row)
            
        cv_df = pd.DataFrame(cv_rows)
        cv_csv_path = os.path.join(args.output_dir, "classification_cv_results.csv")
        cv_df.to_csv(cv_csv_path, index=False)
        print(cv_df.round(3))
        
    if not args.skip_ablation:
        print("\n" + "="*50 + "\n  3. Running Ablation Study (Fusion Gate Variants)\n" + "="*50)
        ablation_metrics = run_ablation(
            args.slim_dir, args.csv, args.data_dir,
            epochs=args.epochs, device=cfg.DEVICE
        )
        
        ablation_rows = []
        for mode, res in ablation_metrics.items():
            row = {'variant': mode}
            for task in ['idh', 'grade', 'mgmt']:
                t_met = res['test_metrics'].get(task, {})
                row[f'{task}_auc'] = t_met.get('auc', float('nan'))
                row[f'{task}_acc'] = t_met.get('accuracy', float('nan'))
                row[f'{task}_f1']  = t_met.get('f1', float('nan'))
            ablation_rows.append(row)
            
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_csv_path = os.path.join(args.output_dir, "ablation_study_results.csv")
        ablation_df.to_csv(ablation_csv_path, index=False)
        print(ablation_df.round(3))
        
    print("\n" + "="*50 + "\n  4. McNemar's Significance Test (vs. Majority Baseline)\n" + "="*50)
    set_seed(seeds_list[0])
    rep_clf = BrainTumorClassifier(gate_mode=cfg.GATE_MODE).to(cfg.DEVICE)
    rep_clf.load_state_dict(torch.load(os.path.join(cfg.CKPT_DIR, f'clf_best_seed{seeds_list[0]}.pth'), map_location=cfg.DEVICE)['model_state'])
    test_metrics_rep = evaluate_classifier(rep_clf, test_loader, device=cfg.DEVICE)
    majority_class_mapping = {'idh': 0.0, 'grade': 1.0, 'mgmt': 1.0}
    
    for task in ['idh', 'grade', 'mgmt']:
        if task in test_metrics_rep:
            t_m = test_metrics_rep[task]
            res = run_mcnemar_test(t_m['labels'], t_m['preds'], majority_class_mapping[task])
            print(f"\n{task.upper()} Task McNemar Test:")
            if res['applicable']:
                print(f"  Majority class baseline accuracy: {res['baseline_accuracy']:.3f}")
                print(f"  Model accuracy:                  {res['model_accuracy']:.3f}")
                print(f"  McNemar p-value:                 {res['p_value']:.4f}")
                
    if os.path.exists(args.full_dir) and len([f for f in os.listdir(args.full_dir) if f.endswith('.pt')]) > 0:
        print("\n" + "="*50 + "\n  5. Running 3D Segmentation Evaluation\n" + "="*50)
        seg_summary = evaluate_segmentation(args.full_dir, args.csv, args.data_dir)
        for region in ['WT', 'TC', 'ET']:
            m = seg_summary[region]
            print(f"\nRegion {region}:")
            print(f"  Dice coefficient : {m['dice']:.3f} \u00B1 {m['dice_std']:.3f}")
            print(f"  HD95 distance    : {m['hd95']:.1f} \u00B1 {m['hd95_std']:.1f} mm")
            
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        latest_rep = torch.load(os.path.join(cfg.CKPT_DIR, f'clf_latest_seed{seeds_list[0]}.pth'), map_location='cpu')
        hist = latest_rep['history']
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(hist['train_loss']) + 1)
        ax[0].plot(epochs_range, hist['train_loss'], 'b-o', ms=3, label='Train')
        ax[0].plot(epochs_range, hist['val_loss'], 'r-o', ms=3, label='Val')
        ax[0].set_title('Multi-Task Loss')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        
        ax[1].plot(epochs_range, hist['train_acc'], 'b-o', ms=3, label='Train')
        ax[1].plot(epochs_range, hist['val_acc'], 'r-o', ms=3, label='Val')
        ax[1].set_title('Accuracy (IDH task)')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        
        plt.suptitle(f'Classifier Training Curves (Seed {seeds_list[0]})', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "classifier_training_curves.png"))
        plt.close()
    except Exception:
        pass
        
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
