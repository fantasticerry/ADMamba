import os

import torch
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torch.utils.data import DataLoader

from admamba.datasets.vaihingen_dataset import *
from admamba.losses import *
from admamba.models.ad_mamba import ADMamba
from tools.utils import Lookahead, process_model_params

# Training hyperparameters
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 1
val_batch_size = 1
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "ad_mamba-fdg-order08-rgb-dsm-e45"
weights_path = os.path.join(
    os.environ.get("ADMAMBA_WEIGHTS_ROOT", "model_weights"),
    "vaihingen",
    weights_name,
)
test_weights_name = weights_name
log_name = f"vaihingen/{weights_name}"
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None

# Network: high-order fractional FDG with RGB + DSM input
net = ADMamba(
    num_classes=num_classes,
    use_elevation_gate=False,
    use_geo_msaa=False,
    use_fractional_gate=True,
    fractional_alpha=0.8,
    fractional_memory_length=16,
)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

dataset_root = os.environ.get("ADMAMBA_DATA_VAIHINGEN", "data/vaihingen")
train_dataset = VaihingenDataset(
    data_root=f"{dataset_root}/train",
    mode="train",
    mosaic_ratio=0.25,
    transform=train_aug,
    use_dsm=True,
)
val_dataset = VaihingenDataset(
    data_root=f"{dataset_root}/test", transform=val_aug, use_dsm=True
)
test_dataset = VaihingenDataset(
    data_root=f"{dataset_root}/test", transform=val_aug, use_dsm=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

# Linear warmup for 5 epochs, then polynomial decay (power=0.9) for the rest
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
poly_scheduler = PolynomialLR(optimizer, total_iters=max_epoch - 5, power=0.9)
lr_scheduler = SequentialLR(
    optimizer, schedulers=[warmup_scheduler, poly_scheduler], milestones=[5]
)
