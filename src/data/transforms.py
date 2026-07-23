import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd,
    Orientationd, Spacingd, NormalizeIntensityd, CropForegroundd,
    ResizeWithPadOrCropd, RandFlipd, RandRotate90d,
    RandShiftIntensityd, ToTensord, MapLabelValued
)
from configs.config import cfg

def get_ucsf_transforms(mode='train'):
    base = [
        LoadImaged(keys=['t1', 't1c', 't2', 'flair', 'label']),
        EnsureChannelFirstd(keys=['t1', 't1c', 't2', 'flair', 'label']),
        ConcatItemsd(keys=['t1', 't1c', 't2', 'flair'], name='image'),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        Spacingd(keys=['image', 'label'], pixdim=cfg.VOXEL_SPACING,
                 mode=('bilinear', 'nearest')),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=cfg.SPATIAL_SIZE),
        MapLabelValued(keys=['label'], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
        ToTensord(keys=['image', 'label'])
    ]
    
    aug = [
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
    ]
    
    if mode == 'train':
        return Compose(base[:-1] + aug + [base[-1]])
    else:
        return Compose(base)
