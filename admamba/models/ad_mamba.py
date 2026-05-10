"""ADMamba: anisotropic-direction Mamba for remote sensing semantic segmentation.

This module exposes the public model classes :class:`ADMamba` (Swin backbone) and
:class:`EfficientADMamba` (lightweight ResNet backbone) along with the building
blocks that implement the four AD-Mamba innovations: 8-direction diagonal
scanning, sparse top-k MoE direction routing, the fractional-order difference
gate, and the elevation-guided multi-scale attention block (GeoMSAA).
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "ADMamba requires the `mamba_ssm` package. Install it with\n"
        "    pip install causal-conv1d>=1.4.0\n"
        "    pip install mamba-ssm\n"
        "(see README for prerequisites)."
    ) from exc


# 8-direction scan / merge utilities

def diagonal_gather(tensor):
    """取出矩阵所有斜向的元素并拼接"""
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def antidiagonal_gather(tensor):
    """取出矩阵所有反斜向的元素并拼接"""
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    """把斜向元素拼接起来的一维向量还原为最初的矩阵形式"""
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    """把反斜向元素拼接起来的一维向量还原为最初的矩阵形式"""
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

class CrossScan(torch.autograd.Function):
    """八项扫描：横向、竖向、斜向、反斜向及其反向"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)  # 横向扫描：从左到右
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 竖向扫描：从上到下
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 横向和竖向反向
    
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)  # 斜向扫描
        xs[:, 5] = antidiagonal_gather(x)  # 反斜向扫描
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])  # 斜向和反斜向反向

        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,C,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,C,H,W))

        y_res = y_rb + y_da
        return y_res


class CrossMerge(torch.autograd.Function):
    """八项扫描结果合并，支持门控权重"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor, gating_weights: torch.Tensor = None):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ctx.gating_weights = gating_weights
        ys = ys.view(B, K, D, -1)
        
        # 如果提供了门控权重，先进行加权
        if gating_weights is not None:
            ys = ys * gating_weights.view(B, K, 1, 1)
        
        # 合并横向和横向反向
        y_horizontal = ys[:, 0] + ys[:, 2].flip(dims=[-1])
        # 合并竖向和竖向反向，并转回横向形式
        y_vertical = ys[:, 1] + ys[:, 3].flip(dims=[-1])
        y_vertical = y_vertical.view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_horizontal + y_vertical
        y_rb = y_rb.view(B, -1, H, W)

        # 合并斜向和反斜向
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,D,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,D,H,W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        gating_weights = ctx.gating_weights
        B, C, L = x.shape
        xs = x.new_empty((B, 8, C, L))
        
        # 横向扫描
        xs[:, 0] = x
        # 竖向扫描
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        # 横向反向
        xs[:, 2] = torch.flip(xs[:, 0], dims=[-1])
        # 竖向反向
        xs[:, 3] = torch.flip(xs[:, 1], dims=[-1])
        
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B,C,H,W))
        xs[:, 5] = antidiagonal_gather(x.view(B,C,H,W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        
        # 如果提供了门控权重，在反向传播时也要考虑
        if gating_weights is not None:
            xs = xs * gating_weights.view(B, 8, 1, 1)
        
        return xs.view(B, 8, C, H, W), None

#####################################################
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


# Channel + spatial attention (used by MSAA / GeoMSAA)
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# wth修改：内联自 mamba_sys.py —— 空间注意力
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# wth修改：内联自 mamba_sys.py —— 多尺度卷积融合（3x3/5x5/7x7）+ 注意力
class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        x_out = self.up(x_fused_s + x_fused_c)
        return x_out


# wth修改：内联自 mamba_sys.py —— 多源多尺度注意力融合模块MSAA
class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        # x2 从低到高（语义信息），x4 从高到低（边缘细节）
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused

#####################################################
# 几何尺度路由模块 (Geo-Scale MSAA)
class GeoMSAA(nn.Module):
    """
    根据地物高度自动分配感受野的 MSAA 模块 (Object-Centric Scale Routing)。
    当 h_map 为 None 时，退化为原始 MSAA 行为（盲目融合）。
    """
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(GeoMSAA, self).__init__()
        mid_dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, mid_dim, kernel_size=1)
        # 多尺度专家：3x3, 5x5, 7x7
        self.branch3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(mid_dim, mid_dim, kernel_size=5, padding=2)
        self.branch7 = nn.Conv2d(mid_dim, mid_dim, kernel_size=7, padding=3)
        # 海拔路由网络 (Elevation Router)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Softmax(dim=1)
        )
        # 通道注意力和空间注意力（保留原始 MSAA 的注意力能力）
        self.channel_attention = ChannelAttentionModule(mid_dim)
        self.spatial_attention = SpatialAttentionModule()
        self.up = nn.Conv2d(mid_dim, out_channels, kernel_size=1)

    def forward(self, x1, x2, x4, h_map=None):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)

        # 通道注意力
        x_fused_c = x_fused * self.channel_attention(x_fused)

        # 多尺度卷积
        f3 = self.branch3(x_fused)
        f5 = self.branch5(x_fused)
        f7 = self.branch7(x_fused)

        if h_map is not None:
            # 海拔路由：根据高度分布自动分配感受野权重
            h_aligned = F.interpolate(h_map, size=x1.shape[-2:], mode='bilinear', align_corners=False)
            weights = self.router(h_aligned)  # [B, 3, 1, 1]
            x_fused_s = f3 * weights[:, 0:1] + f5 * weights[:, 1:2] + f7 * weights[:, 2:3]
        else:
            # 无高度图时退化为原始等权融合
            x_fused_s = f3 + f5 + f7

        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        x_out = self.up(x_fused_s + x_fused_c)
        return x_out

#####################################################
# 残差连接式有限差门控模块
class FiniteDifferenceGate(nn.Module):
    """
    一个参数无关的、基于有限差分的门控模块（残差连接式）。
    它计算序列中的"新颖度"并用它来门控输入，但保留一部分原始信息以避免完全关闭冗余区域。
    
    数学原理：
    - 有限差分是导数的离散版本，衡量函数的变化率
    - 当连续令牌相似时，差分接近0（冗余区域）
    - 当令牌发生剧变时，差分很大（关键边缘）
    - 通过残差连接式门控，让Mamba在冗余区域保留部分信息，在关键边缘"唤醒"
    """
    def __init__(self, dim, use_layer_norm=True, gate_mode='tanh', residual_ratio=0.2):
        """
        Args:
            dim: 特征维度
            use_layer_norm: 是否使用LayerNorm稳定分数
            gate_mode: 门控模式，'tanh' 或 'sigmoid'
            residual_ratio: 残差比例，即使gate=0时也保留的信息比例（默认0.2，即20%）
        """
        super().__init__()
        self.dim = dim
        self.use_layer_norm = use_layer_norm
        self.gate_mode = gate_mode
        self.residual_ratio = residual_ratio  # 残差比例，确保冗余区域不会完全关闭
        
        if use_layer_norm:
            # 使用LayerNorm来稳定"导数"的大小
            self.norm = nn.LayerNorm(dim)
        
        # 可学习的缩放因子，用于调整门控的敏感度
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状为 [Batch, Seq_Len, Dim]
        
        Returns:
            gated_x: 门控后的序列，形状为 [Batch, Seq_Len, Dim]
        """
        B, L, D = x.shape
        
        # 1. 计算差分 (x_t - x_{t-1})
        # 在序列开头填充第一个元素，使长度保持不变
        x_prev = F.pad(x, (0, 0, 1, 0))[:, :-1, :]  # [B, L, D]
        
        # 2. 计算"导数"（有限差分）
        delta = x - x_prev  # [B, L, D]
        
        # 3. 稳定化处理：对delta向量本身进行LayerNorm（如果启用）
        if self.use_layer_norm:
            # 对差分向量进行LayerNorm，保留多维信息
            delta_normalized = self.norm(delta)  # [B, L, D]
            # 计算归一化后差分向量的L2范数
            norm_delta = torch.norm(delta_normalized, p=2, dim=-1, keepdim=True)  # [B, L, 1]
        else:
            # 直接计算差分向量的L2范数
            norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)  # [B, L, 1]
        
        # 4. 应用可学习的缩放因子
        stable_norm_delta = norm_delta * self.scale
        
        # 5. 转换为门控值
        if self.gate_mode == 'tanh':
            # Tanh模式：delta=0时gate=0，delta大时gate接近1
            gate = torch.tanh(stable_norm_delta.abs())
        elif self.gate_mode == 'sigmoid':
            # Sigmoid模式：delta=0时gate=0.5，delta大时gate接近1
            gate = torch.sigmoid(stable_norm_delta)
        else:
            # 默认使用ReLU+归一化，确保gate在[0,1]范围内
            gate = F.relu(stable_norm_delta)
            gate = gate / (gate.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # 6. 关键修改：残差连接式门控
        # gate 在 [0, 1] 范围内，但我们确保最小值为 residual_ratio
        # 这样即使冗余区域（delta≈0）也会保留 residual_ratio 比例的信息
        gate = self.residual_ratio + (1 - self.residual_ratio) * gate
        
        # 7. 应用门控
        # 冗余令牌 (delta≈0) -> gate≈residual_ratio，保留部分信息
        # 剧变令牌 (delta大) -> gate≈1，Mamba会"唤醒"
        gated_x = x * gate  # [B, L, D]
        
        return gated_x
#####################################################
# 分数阶微积分门控模块 (Fractional Calculus Gate)
class FractionalDifferenceGate(nn.Module):
    """
    基于分数阶微积分的门控模块，使用 Grünwald-Letnikov 离散化。
    
    数学原理：
    - 传统有限差分只计算一阶导数 (x_t - x_{t-1})，只能感知"瞬时变化"
    - 分数阶导数 D^α (0 < α < 1) 具有"幂律记忆"，能感知长程依赖
    - Grünwald-Letnikov 定义: D^α f(t) ≈ Σ_{k=0}^{t} w_k * f(t-k)
    - 权重 w_k = (-1)^k * C(α, k)，随 k 增大呈幂律衰减
    
    优势：
    - α → 0: 记忆长，适合捕获大尺度地物（如建筑群）
    - α → 1: 退化为一阶差分，适合捕获边缘
    - 可学习的 α 让模型自适应选择最优记忆长度
    
    论文支撑：
    - FOLOC (2025): 分数阶最优控制
    - HOPE (2024): Hankel 算子参数化增强 SSM 长程记忆
    """
    def __init__(self, dim, alpha_init=0.5, memory_length=16, residual_ratio=0.2,
                 use_dsm=True, learnable_alpha=True):
        """
        Args:
            dim: 特征维度
            alpha_init: 分数阶初始值，范围 (0, 1)，越小记忆越长
            memory_length: 记忆窗口长度，即考虑过去多少个 token
            residual_ratio: 残差比例，避免完全关闭冗余区域
            use_dsm: 是否使用 DSM 数据增强门控
            learnable_alpha: 是否让分数阶 α 可学习
        """
        super().__init__()
        self.dim = dim
        self.memory_length = memory_length
        self.residual_ratio = residual_ratio
        self.use_dsm = use_dsm
        self.learnable_alpha = learnable_alpha
        
        # 分数阶参数：使用 sigmoid 约束到 (0, 1) 范围
        # 存储为 logit 形式以便无约束优化
        if learnable_alpha:
            alpha_logit = math.log(alpha_init / (1 - alpha_init + 1e-8))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit))
        else:
            self.register_buffer('alpha_logit', torch.tensor(math.log(alpha_init / (1 - alpha_init + 1e-8))))
        
        # DSM 分支的分数阶（通常使用更小的 α 以获得更长记忆）
        if use_dsm:
            alpha_h_init = 0.3
            alpha_h_logit = math.log(alpha_h_init / (1 - alpha_h_init + 1e-8))
            if learnable_alpha:
                self.alpha_h_logit = nn.Parameter(torch.tensor(alpha_h_logit))
            else:
                self.register_buffer('alpha_h_logit', torch.tensor(alpha_h_logit))
            
            # DSM 精炼网络
            self.h_refiner = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=1)
            )
            
            # RGB-DSM 融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # 可学习的门控缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        
        # LayerNorm 用于稳定分数阶差分
        self.norm = nn.LayerNorm(dim)
        
        # 预计算最大记忆长度的 GL 权重（会在 forward 时根据实际 α 重新计算）
        self._precompute_gl_weights_cache()
    
    def _precompute_gl_weights_cache(self):
        """预计算一些常用的组合数，加速 GL 权重计算"""
        L = self.memory_length
        indices = torch.arange(L, dtype=torch.float32)
        self.register_buffer('gl_indices', indices)
    
    @property
    def alpha(self):
        """获取当前的分数阶 α，范围 (0, 1)"""
        return torch.sigmoid(self.alpha_logit)
    
    @property
    def alpha_h(self):
        """获取 DSM 分支的分数阶"""
        if self.use_dsm:
            return torch.sigmoid(self.alpha_h_logit)
        return None
    
    def compute_gl_weights(self, alpha, length):
        """
        计算 Grünwald-Letnikov 权重
        
        公式: w_k = (-1)^k * C(α, k) = Π_{j=0}^{k-1} (j - α) / (k!)
        
        递推公式: w_0 = 1, w_k = w_{k-1} * (k - 1 - α) / k
        
        Args:
            alpha: 分数阶，范围 (0, 1)
            length: 权重序列长度
        
        Returns:
            weights: [length] GL 权重
        """
        # 使用列表收集权重，避免就地操作导致的梯度问题
        weights_list = []
        w_prev = torch.ones(1, device=alpha.device, dtype=alpha.dtype)
        weights_list.append(w_prev)
        
        for k in range(1, length):
            w_curr = w_prev * (k - 1 - alpha) / k
            weights_list.append(w_curr)
            w_prev = w_curr
        
        # 拼接成张量
        weights = torch.cat(weights_list, dim=0)
        
        return weights
    
    def apply_fractional_difference(self, x, alpha):
        """
        应用分数阶差分算子
        
        D^α x_t = Σ_{k=0}^{min(t, L-1)} w_k * x_{t-k}
        
        使用 1D 卷积高效实现
        
        Args:
            x: [B, L, D] 输入序列
            alpha: 分数阶
        
        Returns:
            frac_diff: [B, L, D] 分数阶差分结果
        """
        B, L, D = x.shape
        
        # 计算 GL 权重
        mem_len = min(L, self.memory_length)
        gl_weights = self.compute_gl_weights(alpha, mem_len)  # [mem_len]
        
        # 翻转权重用于卷积（因为卷积是 w_0*x_t + w_1*x_{t-1} + ...）
        gl_weights_flipped = gl_weights.flip(0)  # [mem_len]
        
        # 构建卷积核: [out_channels, in_channels/groups, kernel_size]
        # 使用分组卷积，每个通道独立
        kernel = gl_weights_flipped.view(1, 1, -1).expand(D, 1, -1)  # [D, 1, mem_len]
        
        # 填充序列以保持因果性（只看过去）
        x_padded = F.pad(x.transpose(1, 2), (mem_len - 1, 0))  # [B, D, L + mem_len - 1]
        
        # 应用分组卷积
        frac_diff = F.conv1d(x_padded, kernel, groups=D)  # [B, D, L]
        frac_diff = frac_diff.transpose(1, 2)  # [B, L, D]
        
        return frac_diff
    
    def forward(self, x, h_map=None, dir_idx=0, H=0, W=0):
        """
        Args:
            x: [B, L, D] 输入序列
            h_map: [B, 1, H, W] 高度图（可选）
            dir_idx: 当前扫描方向索引 (0-7)
            H, W: 特征图的空间尺寸
        
        Returns:
            gated_x: [B, L, D] 门控后的序列
        """
        B, L, D = x.shape
        
        # 1. 计算 RGB 特征的分数阶差分
        frac_diff_rgb = self.apply_fractional_difference(x, self.alpha)  # [B, L, D]
        
        # 2. 稳定化：对分数阶差分进行 LayerNorm
        frac_diff_rgb_norm = self.norm(frac_diff_rgb)  # [B, L, D]
        
        # 3. 计算差分的 L2 范数作为"新颖度"分数
        norm_rgb = torch.norm(frac_diff_rgb_norm, p=2, dim=-1, keepdim=True)  # [B, L, 1]
        
        # 4. 如果启用 DSM 且提供了高度图
        if self.use_dsm and h_map is not None and H > 0 and W > 0:
            # 对齐高度图到当前特征图尺寸
            h_aligned = F.interpolate(h_map, size=(H, W), mode='bilinear', align_corners=False)
            
            # 精炼高度图
            h_refined = self.h_refiner(h_aligned)
            
            # 对高度图执行 8 向扫描，取当前方向
            h_scanned = CrossScan.apply(h_refined)  # [B, 8, 1, H*W]
            h_seq = h_scanned[:, dir_idx].transpose(1, 2)  # [B, H*W, 1]
            
            # 对高度序列计算分数阶差分
            frac_diff_h = self.apply_fractional_difference(h_seq, self.alpha_h)  # [B, L, 1]
            
            # 高度的分数阶差分范数
            norm_h = frac_diff_h.abs()  # [B, L, 1]
            
            # 融合 RGB 和 DSM 的分数阶门控分数
            # 使用可学习权重平衡两者贡献
            fusion_w = torch.sigmoid(self.fusion_weight)
            gate_score = fusion_w * norm_rgb + (1 - fusion_w) * norm_h * (norm_rgb.mean() + 1e-8)
        else:
            gate_score = norm_rgb
        
        # 5. 应用可学习缩放
        gate_score = gate_score * self.scale
        
        # 6. 转换为门控值 (tanh 确保在 [0, 1])
        gate = torch.tanh(gate_score)
        
        # 7. 残差连接式门控：确保最小保留 residual_ratio 的信息
        gate = self.residual_ratio + (1 - self.residual_ratio) * gate
        
        # 8. 应用门控
        gated_x = x * gate
        
        return gated_x
    
    def get_alpha_info(self):
        """返回当前的分数阶信息，用于日志记录"""
        info = {'alpha_rgb': self.alpha.item()}
        if self.use_dsm and hasattr(self, 'alpha_h_logit'):
            info['alpha_dsm'] = self.alpha_h.item()
        return info


#####################################################
# 海拔感知各向异性门控模块 (Geo-Anisotropic FDG)
class ElevationGuidedGate(nn.Module):
    """
    针对遥感设计的海拔感知门控 (Elevation Geometry Prior)
    利用高度图的梯度(nDSM Gradient)来引导 Mamba 的状态更新。
    当 h_map 为 None 时，退化为原始 FiniteDifferenceGate 行为（纯 RGB 差分）。
    """
    def __init__(self, dim, residual_ratio=0.2):
        super().__init__()
        self.residual_ratio = residual_ratio
        # 几何分支：感知高度变化
        self.h_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # 可学习的高度敏感系数

    def forward(self, x, h_map=None, dir_idx=0, H=0, W=0):
        """
        Args:
            x: [B, L, D] (RGB特征序列)
            h_map: [B, 1, H, W] (高度/nDSM图)，为 None 时退化为纯 RGB 差分门控
            dir_idx: 当前扫描方向索引 (0-7)
            H, W: 特征图的空间尺寸
        Returns:
            gated_x: [B, L, D]
        """
        # RGB 差分 (纹理)
        dx = x - F.pad(x, (0, 0, 1, 0))[:, :-1, :]
        norm_dx = torch.norm(dx, p=2, dim=-1, keepdim=True)  # [B, L, 1]

        if h_map is not None and H > 0 and W > 0:
            # 对齐高度图到当前特征图尺寸
            h_aligned = F.interpolate(h_map, size=(H, W), mode='bilinear', align_corners=False)
            # 精炼高度图
            h_refined = self.h_refiner(h_aligned)
            # 对高度图执行 8 向扫描，取当前方向
            h_scanned = CrossScan.apply(h_refined)  # [B, 8, 1, H*W]
            h_seq = h_scanned[:, dir_idx].transpose(1, 2)  # [B, H*W, 1]
            # Height 差分 (几何跳变)
            dh = (h_seq - F.pad(h_seq, (0, 0, 1, 0))[:, :-1, :]).abs()
            # 几何引导门控: G = tanh(|dx| + alpha * |dh|)
            gate_score = norm_dx + self.alpha * dh * (norm_dx.mean() + 1e-8)
        else:
            # 无高度图时退化为纯 RGB 差分门控
            gate_score = norm_dx

        gate = torch.tanh(gate_score)
        gate = self.residual_ratio + (1 - self.residual_ratio) * gate

        return x * gate


class HardTopKRouting(torch.autograd.Function):
    """
    硬 Top-K 路由 - Straight-Through Estimator 实现
    
    参考 MoCE-IR (CVPR 2025) 和 pytorch-mixtures 的实现
    - 前向：硬选择，只执行被选中的专家
    - 反向：STE，梯度直接传递给选中的专家
    """
    
    @staticmethod
    def forward(ctx, scores: torch.Tensor, topk: int):
        """
        Args:
            scores: [B, 8] 路由分数 (softmax 后)
            topk: 选择数量
        Returns:
            selected_experts: [B, topk] 选中的专家索引
            topk_weights: [B, topk] 选中专家的权重
        """
        topk_weights, selected_experts = torch.topk(scores, topk, dim=-1)
        # 重新归一化 top-k 权重
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        ctx.save_for_backward(selected_experts, topk_weights)
        ctx.topk = topk
        return selected_experts, topk_weights
    
    @staticmethod
    def backward(ctx, grad_selected, grad_weights):
        """
        STE 反向传播：梯度直接传递
        """
        selected_experts, topk_weights = ctx.saved_tensors
        # 重建完整的梯度（只对选中的位置有梯度）
        grad_scores = torch.zeros_like(grad_weights).repeat(1, ctx.topk)
        return None, None


class SparseMoELayer(nn.Module):
    """
    稀疏 MoE 层 - 只执行被选中的专家
    
    参考 pytorch-mixtures 的 TopkMoE 实现
    支持在 Mamba 处理前应用门控模块
    """
    
    def __init__(self, num_experts=8, topk=4, 
                 use_fractional_gate=False, frac_gate=None,
                 use_elevation_gate=False, elev_gate=None,
                 fd_gate=None, residual_ratio=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        # 门控模块引用
        self.frac_gate = frac_gate
        self.elev_gate = elev_gate
        self.fd_gate = fd_gate
        self.use_fractional_gate = use_fractional_gate
        self.use_elevation_gate = use_elevation_gate
        self.residual_ratio = residual_ratio
    
    def apply_gate(self, x: torch.Tensor, expert_idx: int, 
                   h_map=None, H=None, W=None) -> torch.Tensor:
        """
        应用门控模块到输入序列
        
        Args:
            x: [N, L, C] 输入序列
            expert_idx: 专家索引 (0-7)
            h_map: 高度图
            H, W: 特征图尺寸
        Returns:
            gated_x: [N, L, C] 门控后的序列
        """
        if self.use_fractional_gate and self.frac_gate is not None:
            x = self.frac_gate(x, h_map=h_map, dir_idx=expert_idx, H=H, W=W)
        elif self.use_elevation_gate and self.elev_gate is not None:
            x = self.elev_gate(x, h_map=h_map, dir_idx=expert_idx, H=H, W=W)
        elif self.fd_gate is not None:
            x = self.fd_gate(x)
        return x
    
    def forward(self, xs: torch.Tensor, experts: nn.ModuleList, 
                scores: torch.Tensor, h_map=None, H=None, W=None) -> torch.Tensor:
        """
        Args:
            xs: [B, 8, C, L] 8个方向的扫描特征
            experts: nn.ModuleList of 8 Mamba modules
            scores: [B, 8] 路由分数 (softmax 后)
            h_map: 高度图 (可选)
            H, W: 特征图尺寸
        Returns:
            output: [B, C, L] 合并后的输出
        """
        B, K, C, L = xs.shape
        device = xs.device
        
        # Top-K 选择
        selected_experts, topk_weights = HardTopKRouting.apply(scores, self.topk)
        
        # 初始化输出
        output = torch.zeros(B, C, L, device=device, dtype=xs.dtype)
        
        # 批量执行被选中的专家
        for expert_idx in range(self.num_experts):
            # 找出哪些样本选中了第 expert_idx 个专家
            mask = (selected_experts == expert_idx)  # [B, topk]
            
            if mask.any():
                # 获取选中了该专家的样本索引
                sample_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]  # [N]
                
                # 获取这些样本在该方向上的特征
                x_selected = xs[sample_indices, expert_idx]  # [N, C, L]
                
                # 转换为 Mamba 输入格式: [N, L, C]
                x_selected = x_selected.transpose(1, 2)
                
                # 应用门控模块
                x_gated = self.apply_gate(x_selected, expert_idx, h_map, H, W)
                
                # 执行 Mamba
                with torch.amp.autocast('cuda', enabled=False):
                    y_selected = experts[expert_idx](x_gated)  # [N, L, C]
                    y_selected = y_selected.to(xs.dtype)
                
                # 转换回 [N, C, L]
                y_selected = y_selected.transpose(1, 2)
                
                # 计算每个选中样本的权重并累加到输出
                for i, sample_idx in enumerate(sample_indices):
                    weight_positions = mask[sample_idx].nonzero(as_tuple=True)[0]
                    if len(weight_positions) > 0:
                        weight = topk_weights[sample_idx, weight_positions[0]]
                        output[sample_idx] += y_selected[i] * weight
        
        return output


#####################################################

class MambaLayer(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16, 
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01,
                 use_elevation_gate=False, use_fractional_gate=False,
                 fractional_alpha=0.5, fractional_memory_length=16):
        """
        Args:
            in_chs: 输入通道数
            dim: 特征维度
            d_state: Mamba 状态维度
            d_conv: Mamba 1D 卷积宽度
            expand: Mamba 扩展因子
            last_feat_size: 最后特征图尺寸
            enable_moe: 是否启用 MoE 门控
            moe_top_k: MoE top-k 选择
            moe_loss_weight: 负载均衡损失权重
            use_elevation_gate: 是否使用海拔感知门控
            use_fractional_gate: 是否使用分数阶微积分门控（新增）
            fractional_alpha: 分数阶初始值，范围 (0, 1)（新增）
            fractional_memory_length: 分数阶记忆窗口长度（新增）
        """
        super().__init__()
        self.use_elevation_gate = use_elevation_gate
        self.use_fractional_gate = use_fractional_gate
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
                    ConvBNReLU(in_chs, dim, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1)
                    ))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(in_chs, dim, kernel_size=1)#通道卷积
                    ))
        #####################################################
        #YTY修改1：增加Mamba模块，对8向扫描的结果进行处理
        # 八项扫描，每个方向使用独立的Mamba
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim*self.pool_len+in_chs,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width 1d卷积
                expand=expand # Block expansion factor 扩展
            ) for _ in range(8)  # 8个方向：横向、竖向、斜向、反斜向及其反向
        ])
        
        # 残差连接式有限差门控模块：在Mamba处理之前对序列进行门控
        self.fd_gate = FiniteDifferenceGate(
            dim=dim*self.pool_len+in_chs,
            use_layer_norm=True,
            gate_mode='tanh',
            residual_ratio=0.2  # 保留20%的信息，避免完全关闭冗余区域
        )
        
        # 分数阶微积分门控模块（消融开关：use_fractional_gate）
        # 优先级：fractional_gate > elevation_gate > fd_gate
        if self.use_fractional_gate:
            self.frac_gate = FractionalDifferenceGate(
                dim=dim*self.pool_len+in_chs,
                alpha_init=fractional_alpha,
                memory_length=fractional_memory_length,
                residual_ratio=0.2,
                use_dsm=True,  # 启用 DSM 融合
                learnable_alpha=True  # 让分数阶可学习
            )
        
        # 海拔感知门控模块（消融开关：use_elevation_gate）
        if self.use_elevation_gate and not self.use_fractional_gate:
            self.elev_gate = ElevationGuidedGate(
                dim=dim*self.pool_len+in_chs,
                residual_ratio=0.2
            )
        
        # 门控机制参数
        self.enable_moe = enable_moe
        self.moe_top_k = min(moe_top_k, 8)  # 最多选择8个方向
        self.moe_loss_weight = moe_loss_weight
        self.k_group = 8
        
        if self.enable_moe:
            # 门控网络：将池化特征映射到8个方向的权重
            self.gate = nn.Linear(dim*self.pool_len+in_chs, 8, bias=False)
            # 为路由机制添加可学习的噪声参数
            self.gate_noise_scale = nn.Parameter(torch.ones(1) * 0.01)  # 可学习的噪声尺度参数
            self.gate_noise_decay = 0.9  # 噪声衰减率，用于训练稳定性
            # 稀疏 MoE 层 - 只执行被选中的专家
            self.sparse_moe = SparseMoELayer(num_experts=8, topk=moe_top_k)
        else:
            self.gate = None
            self.sparse_moe = None
        #####################################################
    def forward(self, x, h_map=None): # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = []
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        ppm_out.append(res)  # 原始特征放在最后
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        
        #####################################################
        #YTY修改1：增加门控机制和8向扫描
        # 计算门控权重
        scores = None
        if self.enable_moe and self.gate is not None:
            # 使用全局平均池化获取特征表示
            pooled = F.adaptive_avg_pool2d(x, output_size=1).view(B, chs)  # (B, C)
            logits = self.gate(pooled)  # (B, 8)
            
            # 在路由机制的logits中添加可学习的噪声，提高训练稳定性和探索性
            if self.training:
                # 生成可学习的噪声，尺度由gate_noise_scale参数控制
                noise = torch.randn_like(logits) * self.gate_noise_scale
                # 应用噪声衰减，随着训练进行逐渐减少噪声
                training_steps = getattr(self, 'training_steps', 0)
                noise = noise * (self.gate_noise_decay ** training_steps)
                logits = logits + noise
            
            scores = torch.softmax(logits, dim=-1)  # (B, 8)
            # Top-K 选择由 SparseMoELayer 内部完成
        
        # 使用CrossScan进行八项扫描
        xs = CrossScan.apply(x)  # (B, 8, C, H*W)
        
        # 硬 Top-K 路由：只对选中的方向执行 Mamba
        if self.enable_moe and self.gate is not None:
            # 使用稀疏 MoE 层 - 只执行被选中的专家
            # xs: [B, 8, C, H*W] -> [B, 8, C, L]
            xs_flat = xs.view(B, 8, chs, H * W)
            
            # 使用 SparseMoELayer 执行稀疏路由
            y = self.sparse_moe(xs_flat, self.mambas, scores)  # [B, C, L]
            
            # 重塑为 [B, C, H, W]
            y = y.view(B, chs, H, W)
            
        else:
            # 不使用 MoE 时执行所有方向
            ys = []
            for i in range(8):
                x_i = xs[:, i].transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
                if self.use_fractional_gate and hasattr(self, 'frac_gate'):
                    x_i = self.frac_gate(x_i, h_map=h_map, dir_idx=i, H=H, W=W)
                elif self.use_elevation_gate and hasattr(self, 'elev_gate'):
                    x_i = self.elev_gate(x_i, h_map=h_map, dir_idx=i, H=H, W=W)
                else:
                    x_i = self.fd_gate(x_i)
                y_i = self.mambas[i](x_i)
                y_i = y_i.transpose(1, 2)
                ys.append(y_i)
            
            ys = torch.stack(ys, dim=1)
            ys = ys.view(B, 8, chs, H, W)
            y = CrossMerge.apply(ys, None)
            y = y.view(B, chs, H, W)
        
        # 负载均衡损失 (MoCE-IR 风格)
        load_balance_loss = self.compute_load_balancing_loss(scores)
        self.load_balance_loss = load_balance_loss
        
        return y
        #####################################################

    def compute_load_balancing_loss(self, scores):
        """
        MoCE-IR 风格的负载均衡损失
        
        参考 MoCE-IR (CVPR 2025) 的辅助损失设计：
        - Importance Loss: 专家被选中的频率
        - Load Loss: 专家的负载权重
        - CV (Coefficient of Variation) 鼓励均匀分布
        
        Args:
            scores: 门控分数 (B, 8) softmax 后
            
        Returns:
            load_balance_loss: 负载均衡损失
        """
        if scores is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        B = scores.shape[0]
        
        # 1. 计算重要性 (Importance) - 每个专家的路由权重和
        importance = scores.sum(dim=0)  # [8]
        
        # 2. 计算负载 (Load) - 每个样本中被激活的专家数量相关
        # 这里简化处理，使用专家权重的 L2 范数
        load = (scores ** 2).sum(dim=0)  # [8]
        
        # 3. Coefficient of Variation (CV) = (std / mean)^2
        # 方差越小，专家使用越均匀
        def cv(x):
            mean = x.mean()
            std = x.std()
            return (std / (mean + 1e-8)).pow(2)
        
        # 4. MoCE-IR 风格组合损失
        importance_loss = cv(importance)
        load_loss = cv(load)
        
        # 总辅助损失
        aux_loss = 0.5 * importance_loss + 0.5 * load_loss
        
        # 可选：添加路由器 z-loss (防止路由器 logits 过大)
        # 这有助于训练稳定性
        # z_loss 使用 scores 而非 logits，因为此方法接收的是 softmax 后的分数
        z_loss = torch.logsumexp(scores, dim=-1).mean()
        
        total_loss = aux_loss + 0.01 * z_loss
        
        return total_loss

    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        if hasattr(self, 'gate_noise_scale'):
            self.training_steps = step

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16,
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01,
                 use_elevation_gate=False, use_fractional_gate=False,
                 fractional_alpha=0.5, fractional_memory_length=16):
        super(Block, self).__init__()
        self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size,
                               enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight,
                               use_elevation_gate=use_elevation_gate, use_fractional_gate=use_fractional_gate,
                               fractional_alpha=fractional_alpha, fractional_memory_length=fractional_memory_length)
        self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x, h_map=None):
        x = self.mamba(x, h_map=h_map)
        x = self.conv_ffn(x)

        return x
    
    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        self.mamba.update_training_step(step)


class Decoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=128, num_classes=6, last_feat_size=16,
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01,
                 use_elevation_gate=False, use_geo_msaa=False,
                 use_fractional_gate=False, fractional_alpha=0.5, fractional_memory_length=16):
        super().__init__()
        self.use_geo_msaa = use_geo_msaa
        self.use_fractional_gate = use_fractional_gate
        self.b3 = Block(in_chs=encoder_channels[-1], dim=decoder_channels, last_feat_size=last_feat_size,
                       enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight,
                       use_elevation_gate=use_elevation_gate, use_fractional_gate=use_fractional_gate,
                       fractional_alpha=fractional_alpha, fractional_memory_length=fractional_memory_length)
        self.up_conv = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),#最近临近值法
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        #wth 使用 MSAA 融合 x3 (深层上采样后) 与 x0 (浅层) 特征
        #wth 三路输入按 [x3, pre_x0, pre_x0] 组合，输入通道数为 decoder_channels*3，输出与原先保持一致
        if self.use_geo_msaa:
            self.msaa = GeoMSAA(in_channels=decoder_channels * 3, out_channels=decoder_channels)
        else:
            self.msaa = MSAA(in_channels=decoder_channels * 3, out_channels=decoder_channels)
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  Conv(decoder_channels // 2, num_classes, kernel_size=1))
        self.apply(self._init_weights) #对每个模块进行初始化

    def forward(self, x0, x3, h_map=None):
        x3 = self.b3(x3, h_map=h_map)
        x3 = self.up_conv(x3)
        #wth 使用 MSAA 融合 x3 (深层上采样后) 与 x0 (浅层) 特征
        #wth 三路输入按 [x3, pre_x0, pre_x0] 组合，输入通道数为 decoder_channels*3，输出与原先保持一致
        pre_x0 = self.pre_conv(x0)
        if self.use_geo_msaa:
            x = self.msaa(x3, pre_x0, pre_x0, h_map=h_map)
        else:
            x = self.msaa(x3, pre_x0, pre_x0)
        x = self.head(x)
        return x
    
    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        self.b3.update_training_step(step)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class EfficientADMamba(nn.Module):
    def __init__(self,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 decoder_channels=128,
                 last_feat_size=16,  # last_feat_size=input_img_size // 32
                 enable_moe=True,
                 moe_top_k=4,
                 moe_loss_weight=0.01,
                 use_elevation_gate=False,
                 use_geo_msaa=False,
                 use_fractional_gate=False,
                 fractional_alpha=0.5,
                 fractional_memory_length=16
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size,
                              enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight,
                              use_elevation_gate=use_elevation_gate, use_geo_msaa=use_geo_msaa,
                              use_fractional_gate=use_fractional_gate, fractional_alpha=fractional_alpha,
                              fractional_memory_length=fractional_memory_length)

    def forward(self, x, h_map=None):
        x0, x3 = self.backbone(x)
        x = self.decoder(x0, x3, h_map=h_map)
        
        #wth修改1：负载均衡 - 收集负载均衡损失
        total_load_balance_loss = 0.0
        if hasattr(self.decoder.b3, 'load_balance_loss'):
            total_load_balance_loss += self.decoder.b3.load_balance_loss
        
        #wth修改1：负载均衡 - 将负载均衡损失存储为属性，供训练时使用
        self.total_load_balance_loss = total_load_balance_loss

        return x
    
    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        self.decoder.update_training_step(step)
    
    def get_load_balance_loss(self):
        """
        #wth修改1：负载均衡
        获取负载均衡损失，供训练时使用
        
        Returns:
            load_balance_loss: 负载均衡损失张量
        """
        if hasattr(self, 'total_load_balance_loss'):
            return self.total_load_balance_loss
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_fractional_alpha_info(self):
        """
        获取分数阶门控的 alpha 信息，用于日志记录
        
        Returns:
            dict: 包含 alpha_rgb 和 alpha_dsm 的字典，如果未启用分数阶门控则返回空字典
        """
        if hasattr(self.decoder.b3.mamba, 'frac_gate'):
            return self.decoder.b3.mamba.frac_gate.get_alpha_info()
        return {}


class ADMamba(nn.Module):
    def __init__(self,
                 backbone_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                 pretrained=True,
                 num_classes=6,
                 decoder_channels=128,
                 last_feat_size=32,
                 img_size=1024,
                 enable_moe=True,
                 moe_top_k=4,
                 moe_loss_weight=0.01,
                 use_elevation_gate=False,
                 use_geo_msaa=False,
                 use_fractional_gate=False,
                 fractional_alpha=0.5,
                 fractional_memory_length=16
                 ):
        """
        Args:
            backbone_name: 主干网络名称
            pretrained: 是否使用预训练权重
            num_classes: 分类数量
            decoder_channels: 解码器通道数
            last_feat_size: 最后特征图尺寸
            img_size: 输入图像尺寸
            enable_moe: 是否启用 MoE 门控
            moe_top_k: MoE top-k 选择
            moe_loss_weight: 负载均衡损失权重
            use_elevation_gate: 是否使用海拔感知门控
            use_geo_msaa: 是否使用几何感知 MSAA
            use_fractional_gate: 是否使用分数阶微积分门控
            fractional_alpha: 分数阶初始值 (0, 1)，越小记忆越长
            fractional_memory_length: 分数阶记忆窗口长度
        """
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, img_size=img_size,
                                          out_indices=(-4, -1), pretrained=pretrained)#移除分类头

        encoder_channels = self.backbone.feature_info.channels() #[96, 96, 192, 384, 768]
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size,
                              enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight,
                              use_elevation_gate=use_elevation_gate, use_geo_msaa=use_geo_msaa,
                              use_fractional_gate=use_fractional_gate, fractional_alpha=fractional_alpha,
                              fractional_memory_length=fractional_memory_length)

    def forward(self, x, h_map=None):
        x0, x3 = self.backbone(x)
        x0 = x0.permute(0, 3, 1, 2)# (B, H, W, C) -> (B, C, H, W)
        x3 = x3.permute(0, 3, 1, 2)# (B, H, W, C) -> (B, C, H, W)
        x = self.decoder(x0, x3, h_map=h_map)
        
        #wth修改1：负载均衡 - 收集负载均衡损失
        total_load_balance_loss = 0.0
        if hasattr(self.decoder.b3, 'load_balance_loss'):
            total_load_balance_loss += self.decoder.b3.load_balance_loss
        
        #wth修改1：负载均衡 - 将负载均衡损失存储为属性，供训练时使用
        self.total_load_balance_loss = total_load_balance_loss

        return x
    
    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        self.decoder.update_training_step(step)
    
    def get_load_balance_loss(self):
        """
        #wth修改1：负载均衡
        获取负载均衡损失，供训练时使用
        
        Returns:
            load_balance_loss: 负载均衡损失张量
        """
        if hasattr(self, 'total_load_balance_loss'):
            return self.total_load_balance_loss
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_fractional_alpha_info(self):
        """
        获取分数阶门控的 alpha 信息，用于日志记录
        
        Returns:
            dict: 包含 alpha_rgb 和 alpha_dsm 的字典，如果未启用分数阶门控则返回空字典
        """
        if hasattr(self.decoder.b3.mamba, 'frac_gate'):
            return self.decoder.b3.mamba.frac_gate.get_alpha_info()
        return {}
