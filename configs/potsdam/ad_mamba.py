import os

import torch
from torch.utils.data import DataLoader

from admamba.datasets.potsdam_dataset import *
from admamba.losses import *
from admamba.models.ad_mamba import ADMamba
from tools.utils import Lookahead, process_model_params

# Training hyperparameters
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "ad_mamba-swin-1024-crop-ms-e45"
weights_path = os.path.join(
    os.environ.get("ADMAMBA_WEIGHTS_ROOT", "model_weights"),
    "potsdam",
    weights_name,
)
test_weights_name = weights_name
log_name = f"potsdam/{weights_name}"
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None

#  define the network
net = ADMamba(num_classes=num_classes, img_size=1024)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

# define the dataloader
import albumentations as albu
import numpy as np

def get_training_transform():
    train_transform = [
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=1024, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    # Resize to 1024x1024 for validation
    img, mask = np.array(img), np.array(mask)
    aug = albu.Compose([albu.Resize(1024, 1024), get_val_transform()])(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


dataset_root = os.environ.get("ADMAMBA_DATA_POTSDAM", "data/potsdam")
train_dataset = PotsdamDataset(
    data_root=f"{dataset_root}/train", mode="train", mosaic_ratio=0.25, transform=train_aug
)
val_dataset = PotsdamDataset(
    data_root=f"{dataset_root}/test", mode="val", transform=val_aug
)
test_dataset = PotsdamDataset(
    data_root=f"{dataset_root}/test", mode="test", transform=val_aug
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
# Poly learning rate scheduler with warmup using PyTorch built-in schedulers
from torch.optim.lr_scheduler import SequentialLR, LinearLR, PolynomialLR

# Warmup phase: Linear increase from 0.1 to 1.0 over 5 epochs
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)

# Poly decay phase: Polynomial decay over remaining 40 epochs with power=0.9
poly_scheduler = PolynomialLR(optimizer, total_iters=max_epoch-5, power=0.9)

# Combine warmup and poly schedulers
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, poly_scheduler], milestones=[5])

