import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from configs.config import cfg

def csv_id_to_file_id(csv_id):
    prefix, num = csv_id.rsplit('-', 1)
    return f"{prefix}-{num.zfill(4)}"

def get_patient_paths(csv_id, ucsf_dir):
    file_id = csv_id_to_file_id(csv_id)
    folder_path = os.path.join(ucsf_dir, f"{file_id}_nifti")
    return {
        'folder': folder_path,
        't1':    os.path.join(folder_path, f'{file_id}_T1_bias.nii.gz'),
        't1c':   os.path.join(folder_path, f'{file_id}_T1c_bias.nii.gz'),
        't2':    os.path.join(folder_path, f'{file_id}_T2_bias.nii.gz'),
        'flair': os.path.join(folder_path, f'{file_id}_FLAIR_bias.nii.gz'),
        'label': os.path.join(folder_path, f'{file_id}_tumor_segmentation.nii.gz'),
    }

def verify_patient_files(csv_id, ucsf_dir):
    paths = get_patient_paths(csv_id, ucsf_dir)
    missing = [k for k, v in paths.items() if k != 'folder' and not os.path.exists(v)]
    return (len(missing) == 0), missing

def build_ucsf_dicts(csv_path, ucsf_dir):
    df = pd.read_csv(csv_path)
    data_dicts, missing_count = [], 0
    for _, row in df.iterrows():
        csv_id = row['ID']
        paths = get_patient_paths(csv_id, ucsf_dir)
        if not all(os.path.exists(paths[k]) for k in ['t1', 't1c', 't2', 'flair', 'label']):
            missing_count += 1
            continue
            
        idh_raw   = str(row['IDH']).strip().lower()
        grade_raw = str(row['WHO CNS Grade']).strip()
        mgmt_raw  = str(row['MGMT status']).strip().lower()
        
        idh = (0.0 if idh_raw == 'wildtype'
               else float('nan') if idh_raw in ['', 'nan', 'unknown']
               else 1.0)
               
        grade = (1.0 if grade_raw == '4'
                 else 0.0 if grade_raw in ['2', '3']
                 else float('nan'))
                 
        mgmt = (1.0 if mgmt_raw == 'positive'
                else 0.0 if mgmt_raw == 'negative'
                else float('nan'))
                
        data_dicts.append({
            'patient_id': csv_id,
            't1': paths['t1'], 
            't1c': paths['t1c'],
            't2': paths['t2'], 
            'flair': paths['flair'],
            'label': paths['label'],
            'idh': idh, 
            'grade': grade, 
            'mgmt': mgmt,
        })
    return data_dicts

def stratified_split(all_dicts, val_split=0.15, test_split=0.15, seed=42):
    import random
    rng = random.Random(seed)
    
    def key(d):
        idh = 'x' if np.isnan(d['idh']) else str(int(d['idh']))
        grd = 'x' if np.isnan(d['grade']) else str(int(d['grade']))
        return (idh, grd)
        
    groups = {}
    for d in all_dicts:
        groups.setdefault(key(d), []).append(d)
        
    train_d, val_d, test_d = [], [], []
    for k, group in groups.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)
        
        test_d  += shuffled[:n_test]
        val_d   += shuffled[n_test:n_test + n_val]
        train_d += shuffled[n_test + n_val:]
        
    return train_d, val_d, test_d

class SegDataset(Dataset):
    def __init__(self, data_dicts, transforms):
        self.data = data_dicts
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        out = self.transforms({
            't1':    item['t1'],
            't1c':   item['t1c'],
            't2':    item['t2'],
            'flair': item['flair'],
            'label': item['label'],
        })
        return out['image'], out['label']

class CachedDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return {
            'encoder':    d['encoder'],
            'morph':      d['morph'],
            'conf_score': d['conf_score'].view(1),
            'idh':        d['idh'], 
            'grade':      d['grade'], 
            'mgmt':       d['mgmt'],
            'patient_id': d['patient_id'],
        }
