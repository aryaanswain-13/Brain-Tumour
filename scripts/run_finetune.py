import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from configs.config import cfg
from src.data.dataset import build_ucsf_dicts, stratified_split, SegDataset
from src.data.transforms import get_ucsf_transforms
from src.models.backbone import build_swinunetr
from src.training.finetune import train_segmentation
from src.utils.helpers import set_seed

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SwinUNETR segmentation model on UCSF-PDGM.")
    parser.add_argument("--csv", type=str, default=cfg.UCSF_CSV, help="Path to UCSF metadata CSV")
    parser.add_argument("--data_dir", type=str, default=cfg.UCSF_DIR, help="Path to NIfTI image dataset directory")
    parser.add_argument("--epochs", type=int, default=cfg.FINETUNE_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=cfg.FINETUNE_LR, help="Learning rate")
    parser.add_argument("--ckpt", type=str, default=os.path.join(cfg.CKPT_DIR, 'swinunetr_ucsf_finetuned.pt'), help="Output path for best checkpoint")
    parser.add_argument("--latest_ckpt", type=str, default=os.path.join(cfg.CKPT_DIR, 'swinunetr_ucsf_latest.pt'), help="Output path for latest checkpoint")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume training")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    cfg.UCSF_CSV = args.csv
    cfg.UCSF_DIR = args.data_dir
    cfg.FINETUNE_EPOCHS = args.epochs
    cfg.FINETUNE_LR = args.lr
    
    all_dicts = build_ucsf_dicts(args.csv, args.data_dir)
    train_d, val_d, _ = stratified_split(all_dicts)
    
    train_ds = SegDataset(train_d, get_ucsf_transforms('train'))
    val_ds   = SegDataset(val_d,   get_ucsf_transforms('val'))
    
    num_workers = 4 if cfg.DEVICE == 'cuda' else 0
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    model = build_swinunetr(weights_path=cfg.WEIGHTS)
    
    train_segmentation(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_path=args.ckpt,
        latest_ckpt_path=args.latest_ckpt,
        patience=args.patience,
        resume=not args.no_resume,
        device=cfg.DEVICE
    )

if __name__ == "__main__":
    main()
