# 近两年 IEEE TGRS/GRSL 遥感语义分割方法对比

> 数据来源：2024-2025年发表在 IEEE TGRS / GRSL 及相关顶刊上的论文，在 ISPRS Vaihingen 和 Potsdam 数据集上的性能对比。

---

## 一、单模态方法（仅使用 RGB/NIRRG 遥感图像）

### 1.1 TGRS/GRSL 2024-2025 论文

| 模型全称 | 期刊 | 年份 | Backbone | Vaihingen OA(%) | Vaihingen mF1(%) | Vaihingen mIoU(%) | Potsdam OA(%) | Potsdam mF1(%) | Potsdam mIoU(%) |
|---|---|---|---|---|---|---|---|---|---|
| RS³Mamba | IEEE GRSL | 2024 | ResNet18 + VMamba-T | 91.64 | 90.34 | 82.78 | 90.49 | 91.69 | 85.01 |
| UNetMamba | IEEE GRSL | 2024 | ResT-Lite | 92.46 | 90.91 | 83.43 | — | — | — |
| SAM_RS | IEEE TGRS | 2024 | SAM (ViT-B/L/H) | — | — | — | — | — | — |
| ConvLSR-Net | IEEE TGRS | 2024 | — | — | — | — | — | — | — |

### 1.2 其他高影响力单模态对比方法

| 模型全称 | 期刊 | 年份 | Backbone | Vaihingen OA(%) | Vaihingen mF1(%) | Vaihingen mIoU(%) | Potsdam OA(%) | Potsdam mF1(%) | Potsdam mIoU(%) |
|---|---|---|---|---|---|---|---|---|---|
| CM-UNet | arXiv | 2024 | ResNet-18 | 93.81 | 92.01 | 85.48 | 91.86 | 93.05 | 87.21 |
| Samba | Heliyon | 2024 | Samba (SSM) | — | 84.23 | 73.56 | — | 90.15 | 82.29 |
| UNetFormer | ISPRS JPRS | 2022 | ResNet-18 | 91.17 | 89.48 | 81.97 | 90.65 | 91.71 | 85.05 |
| DC-Swin | IEEE GRSL | 2022 | Swin-T | 92.28 | 90.66 | 83.07 | — | — | — |
| MANet | IEEE TGRS | 2022 | ResNet-50 | 92.26 | 90.65 | 83.04 | — | — | — |
| BANet | Remote Sensing | 2021 | ResT-Lite | 91.93 | 90.30 | 82.43 | — | — | — |
| ABCNet | ISPRS JPRS | 2021 | ResNet-18 | 91.94 | 90.36 | 82.49 | 91.30 | 92.70 | 86.50 |

> **注：** Samba 从零训练（不使用 ImageNet 预训练权重），与其他方法不完全可比。

---

## 二、多模态方法（RGB/NIRRG + DSM/nDSM 高度图融合）

### 2.1 TGRS/GRSL 2024-2025 论文

| 模型全称 | 期刊 | 年份 | Backbone | Vaihingen OA(%) | Vaihingen mF1(%) | Vaihingen mIoU(%) | Potsdam OA(%) | Potsdam mF1(%) | Potsdam mIoU(%) |
|---|---|---|---|---|---|---|---|---|---|
| MFNet (MMAdapter, ViT-H) | IEEE TGRS | 2025 | SAM ViT-H | **92.97** | **91.71** | **85.03** | **91.71** | **92.70** | **86.69** |
| MFNet (MMLoRA, ViT-H) | IEEE TGRS | 2025 | SAM ViT-H | 92.73 | 91.50 | 84.66 | 91.43 | 92.49 | 86.34 |
| MFNet (MMAdapter, ViT-B) | IEEE TGRS | 2025 | SAM ViT-B | 92.62 | 90.60 | 83.24 | 90.89 | 91.79 | 85.14 |
| FTransUNet | IEEE TGRS | 2024 | R50-ViT-B | 92.40 | 91.21 | 84.23 | 91.34 | 92.41 | 86.20 |
| ASMFNet | IEEE JSTARS | 2024 | — | — | — | — | — | — | — |

### 2.2 其他高影响力多模态对比方法

| 模型全称 | 期刊 | 年份 | Backbone | Vaihingen OA(%) | Vaihingen mF1(%) | Vaihingen mIoU(%) | Potsdam OA(%) | Potsdam mF1(%) | Potsdam mIoU(%) |
|---|---|---|---|---|---|---|---|---|---|
| MultiSenseSeg | — | 2024 | Segformer-B2 | 92.73 | 91.42 | 84.53 | 91.30 | 92.35 | 86.10 |
| FTransDeepLab | — | 2024 | ResNet-101 | 92.61 | 91.00 | 83.87 | 90.97 | 92.08 | 85.62 |
| CMGFNet | — | — | ResNet-34 | 91.72 | 90.00 | 82.26 | 90.21 | 91.40 | 84.53 |
| MFTransNet | — | — | ResNet-34 | 91.22 | 89.62 | 81.61 | 89.96 | 91.11 | 84.04 |
| CMFNet | IEEE JSTARS | 2022 | VGG-16 | 91.40 | 89.48 | 81.44 | 91.16 | 92.10 | 85.63 |
| FuseNet | — | 2016 | VGG-16 | 90.51 | 87.71 | 78.71 | 90.58 | 91.60 | 84.86 |
| vFuseNet | — | 2019 | VGG-16 | 90.49 | 87.89 | 78.92 | 90.22 | 91.26 | 84.26 |
| ESANet | — | — | ResNet-34 | 90.61 | 88.18 | 79.42 | 89.74 | 91.22 | 84.15 |
| SA-GATE | — | — | ResNet-101 | 91.10 | 89.81 | 81.27 | 87.91 | 90.26 | 82.53 |

---

## 三、模型复杂度参考（来自论文报告）

| 模型 | FLOPs (G) | Parameters (M) | Memory (MB) | 测试输入大小 | 来源论文 |
|---|---|---|---|---|---|
| ABCNet | 7.81 | 13.67 | 1008 | 256×256 | RS³Mamba |
| UNetFormer | 5.87 | 11.69 | 1010 | 256×256 | RS³Mamba |
| RS³Mamba | 31.65 | 43.32 | 2332 | 256×256 | RS³Mamba |
| TransUNet | 64.55 | 105.32 | 3122 | 256×256 | RS³Mamba |
| CMTFNet | 17.14 | 30.07 | 1872 | 256×256 | RS³Mamba |
| UNetMamba | 100.52 | 14.76 | 225.71 | 1024×1024 | UNetMamba |
| BANet | 85.43 | 12.73 | 194.61 | 1024×1024 | UNetMamba |
| MANet | 216.82 | 35.86 | 547.87 | 1024×1024 | UNetMamba |
| DC-Swin | 190.04 | 45.63 | 694.62 | 1024×1024 | UNetMamba |
| Samba + UperNet | 232 | 51.9 | — | 512×512 | Samba |
| Segformer (MiT) | 8 | 3.7 | — | 512×512 | Samba |

> **注意：** 不同论文使用的输入尺寸不同，FLOPs 数值不可直接跨论文对比。

---

## 四、数据来源与引用

| 简称 | 论文全称 | DOI / arXiv |
|---|---|---|
| RS³Mamba | RS³Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation | IEEE GRSL, 2024, Art no. 6011405 |
| UNetMamba | UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images | IEEE GRSL, 2024 |
| SAM_RS | SAM-Assisted Remote Sensing Imagery Semantic Segmentation With Object and Boundary Constraints | IEEE TGRS, vol. 62, 2024 |
| ConvLSR-Net | ConvLSR-Net | IEEE TGRS, 2024 |
| FTransUNet | A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation | IEEE TGRS, vol. 62, 2024 |
| MFNet | A Unified Framework with Multimodal Fine-tuning for Remote Sensing Semantic Segmentation | IEEE TGRS, vol. 63, 2025 |
| ASMFNet | Adjacent-Scale Multimodal Fusion Networks for Semantic Segmentation of Remote Sensing Data | IEEE JSTARS, 2024 |
| CM-UNet | CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation | arXiv:2405.10530, 2024 |
| Samba | Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model | Heliyon, 2024 |
| UNetFormer | UNetFormer: A UNet-like Transformer for Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery | ISPRS JPRS, vol. 190, 2022 |
