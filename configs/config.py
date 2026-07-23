import os
import torch

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    UCSF_DIR = os.path.join(BASE_DIR, "data", "UCSF-PDGM-v5")
    UCSF_CSV = os.path.join(BASE_DIR, "data", "UCSF-PDGM-metadata_v5.csv")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    WEIGHTS  = os.path.join(CKPT_DIR, "swinunetr_brats21_fold1.pt")
    
    CACHE_DIR = os.path.join(CKPT_DIR, "feature_cache_3d_ucsf_finetuned")
    SLIM_CACHE = os.path.join(BASE_DIR, "feature_cache_slim_ucsf")

    # --- Segmentation Model Parameters ---
    SPATIAL_SIZE  = (128, 128, 128)
    VOXEL_SPACING = (1.0, 1.0, 1.0)
    IN_CHANNELS  = 4
    OUT_CHANNELS = 3
    FEATURE_SIZE = 48
    ENCODER_DIM  = 768
    TTA_RUNS = 5
    SEG_THRESHOLD  = 0.3
    FREEZE_BACKBONE = True

    # --- Fusion and Classification Parameters ---
    MORPH_FEATURES = 8
    FUSED_DIM      = 512
    GATE_MODE = 'D'
    
    # --- Training Hyper-parameters ---
    BATCH_SIZE = 8
    LR         = 3e-4
    CLF_EPOCHS = 60
    
    # --- Fine-tuning Hyper-parameters ---
    FINETUNE_EPOCHS = 100
    FINETUNE_LR     = 1e-4
    ACCUMULATION_STEPS = 4

    # --- Device Configuration ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()
