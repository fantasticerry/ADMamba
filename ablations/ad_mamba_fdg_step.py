import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba_ssm import Mamba

#####################################################
#YTY修改1：增加Scan和Merge
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


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


# wth修改：内联自 mamba_sys.py —— 通道与空间注意力 + 多尺度融合，用于 MSAA
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

class MambaLayer(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16, 
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01):
        super().__init__()
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
        else:
            self.gate = None
        #####################################################
    def forward(self, x): # B, C, H, W
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
        gating_weights = None
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
            
            # Top-K选择
            k = self.moe_top_k
            if k < self.k_group:
                topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
                # 仅对选中的方向保留其 softmax 权重，未选为 0
                gating_weights = torch.zeros_like(scores)
                gating_weights.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            else:
                gating_weights = scores
        
        # 使用CrossScan进行八项扫描
        xs = CrossScan.apply(x)  # (B, 8, C, H*W)
        
        # 对每个方向分别应用Mamba
        ys = []
        for i in range(8):
            # 重新排列为Mamba期望的格式: (B, L, C)
            x_i = xs[:, i].transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
            # 应用残差连接式有限差门控：在Mamba处理之前对序列进行门控
            # 冗余区域保留20%信息，关键边缘完全保留
            x_i = self.fd_gate(x_i)  # (B, H*W, C)
            y_i = self.mambas[i](x_i)  # (B, H*W, C)
            y_i = y_i.transpose(1, 2)  # (B, C, H*W)
            ys.append(y_i)
        
        # 将结果重新组合为 (B, 8, C, H*W)
        ys = torch.stack(ys, dim=1)  # (B, 8, C, H*W)
        ys = ys.view(B, 8, chs, H, W)  # (B, 8, C, H, W)
        
        # 使用CrossMerge合并结果，传入门控权重
        y = CrossMerge.apply(ys, gating_weights)  # (B, C, H*W)
        y = y.view(B, chs, H, W)  # (B, C, H, W)
        
        #wth修改1：负载均衡 - 计算负载均衡损失
        load_balance_loss = self.compute_load_balancing_loss(gating_weights)
        
        #wth修改1：负载均衡 - 将负载均衡损失存储为属性，供训练时使用
        self.load_balance_loss = load_balance_loss
        
        return y
        #####################################################

    def compute_load_balancing_loss(self, gating_weights):
        """
        #wth修改1：负载均衡
        计算负载均衡损失，确保8个专家被均匀使用
        
        Args:
            gating_weights: 门控权重 (B, 8)
            
        Returns:
            load_balance_loss: 负载均衡损失
        """
        if gating_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 计算每个专家的平均利用率 (8,)
        expert_usage = gating_weights.mean(dim=0)
        
        # 计算负载均衡损失：专家利用率的方差
        # 方差越小表示专家使用越均匀
        load_balance_loss = torch.var(expert_usage)
        
        # 可选：添加熵正则化，鼓励更均匀的分布
        # 计算门控权重的熵
        epsilon = 1e-8
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + epsilon), dim=-1)
        entropy_loss = -entropy.mean()  # 负熵，鼓励高熵（均匀分布）
        
        # 组合负载均衡损失和熵损失
        total_load_balance_loss = load_balance_loss + 0.1 * entropy_loss
        
        return total_load_balance_loss

    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        if hasattr(self, 'gate_noise_scale'):
            self.training_steps = step

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence
    
    @torch.no_grad()
    def analyze_directional_cosine(self, x, direction_labels=None):
        """
        计算不同扫描方向组合的Mamba输出余弦相似度。
        
        Args:
            x: 输入特征 (B, C, H, W)
            direction_labels: 可选字典，键为标签，值为方向索引列表。
                               默认为 {"2_direction": [0,1], "4_direction": [0,1,2,3], "8_direction": list(range(8))}
        
        Returns:
            dict: {label: (k,k) 余弦相似度矩阵 (CPU tensor)}
        """
        res = x
        B, C, H, W = res.shape
        ppm_out = []
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        ppm_out.append(res)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        
        gating_weights = None
        if self.enable_moe and self.gate is not None:
            pooled = F.adaptive_avg_pool2d(x, output_size=1).view(B, chs)
            logits = self.gate(pooled)
            if self.training:
                noise = torch.randn_like(logits) * self.gate_noise_scale
                training_steps = getattr(self, 'training_steps', 0)
                noise = noise * (self.gate_noise_decay ** training_steps)
                logits = logits + noise
            gating_weights = torch.softmax(logits, dim=-1)
        
        xs = CrossScan.apply(x)
        dir_outputs = []
        for i in range(8):
            direction_seq = xs[:, i].transpose(1, 2)
            direction_seq = self.fd_gate(direction_seq)
            y_i = self.mambas[i](direction_seq)
            dir_outputs.append(y_i.transpose(1, 2).reshape(B, -1))
        
        if direction_labels is None:
            direction_labels = {
                "2_direction": [0, 1],
                "4_direction": [0, 1, 2, 3],
                "8_direction": list(range(8))
            }
        
        def build_cosine_matrix(indices):
            size = len(indices)
            matrix = x.new_zeros(size, size)
            for i in range(size):
                for j in range(i, size):
                    vec_i = dir_outputs[indices[i]]
                    vec_j = dir_outputs[indices[j]]
                    cos_vals = F.cosine_similarity(vec_i, vec_j, dim=-1).mean()
                    matrix[i, j] = cos_vals
                    if i != j:
                        matrix[j, i] = cos_vals
            return matrix.cpu()
        
        cosine_results = {}
        for label, idxs in direction_labels.items():
            cosine_results[label] = build_cosine_matrix(idxs)
        
        if gating_weights is None:
            gate_scores = torch.ones(self.k_group, device=x.device) / self.k_group
        else:
            gate_scores = gating_weights.mean(dim=0)
        topk = min(4, self.k_group)
        top_indices = torch.topk(gate_scores, k=topk).indices.tolist()
        cosine_results["8_select_4"] = build_cosine_matrix(top_indices)
        
        return cosine_results


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
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01):
        super(Block, self).__init__()
        self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size,
                               enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight)
        self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)

        return x
    
    def update_training_step(self, step: int):
        """更新训练步数，用于控制噪声衰减"""
        self.mamba.update_training_step(step)


class Decoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=128, num_classes=6, last_feat_size=16,
                 enable_moe=True, moe_top_k=4, moe_loss_weight=0.01):
        super().__init__()
        self.b3 = Block(in_chs=encoder_channels[-1], dim=decoder_channels, last_feat_size=last_feat_size,
                       enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight)
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
        self.msaa = MSAA(in_channels=decoder_channels * 3, out_channels=decoder_channels)
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  Conv(decoder_channels // 2, num_classes, kernel_size=1))
        self.apply(self._init_weights) #对每个模块进行初始化

    def forward(self, x0, x3):
        x3 = self.b3(x3)
        x3 = self.up_conv(x3)
        #wth 使用 MSAA 融合 x3 (深层上采样后) 与 x0 (浅层) 特征
        #wth 三路输入按 [x3, pre_x0, pre_x0] 组合，输入通道数为 decoder_channels*3，输出与原先保持一致
        pre_x0 = self.pre_conv(x0)
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
                 moe_loss_weight=0.01
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size,
                              enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x = self.decoder(x0, x3)
        
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
                 moe_loss_weight=0.01
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, img_size=img_size,
                                          out_indices=(-4, -1), pretrained=pretrained)#移除分类头

        encoder_channels = self.backbone.feature_info.channels() #[96, 96, 192, 384, 768]
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size,
                              enable_moe=enable_moe, moe_top_k=moe_top_k, moe_loss_weight=moe_loss_weight)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x0 = x0.permute(0, 3, 1, 2)# (B, H, W, C) -> (B, C, H, W)
        x3 = x3.permute(0, 3, 1, 2)# (B, H, W, C) -> (B, C, H, W)
        x = self.decoder(x0, x3)
        
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

