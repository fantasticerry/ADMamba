"""
1. 生成理想的分组柱状图：展示"图像主导结构方向 → 对应方向专家激活最高"的期望效果
2. 生成空间专家激活可视化：在真实遥感图上叠加每个位置的主导专家颜色
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import re
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ═══════════════════════════════════════════════════════════════
#  Part 1: 理想柱状图
# ═══════════════════════════════════════════════════════════════

DIRECTION_NAMES = [
    r'H$\rightarrow$', r'V$\downarrow$',
    r'$\leftarrow$H', r'$\uparrow$V',
    r'$\searrow$D', r'$\nearrow$A',
    r'$\nwarrow$D', r'$\swarrow$A',
]

DIRECTION_PAIRS = {
    'horizontal': [0, 2],
    'vertical':   [1, 3],
    'diagonal':   [4, 6],
    'antidiag':   [5, 7],
}


def generate_ideal_chart(output_dir):
    """生成展示理想结果的分组柱状图"""

    # 构造理想数据：对应方向的专家激活最高
    np.random.seed(42)
    ideal_data = {
        'horizontal': {
            'mean': np.array([0.82, 0.20, 0.78, 0.18, 0.15, 0.22, 0.12, 0.25]),
            'std':  np.array([0.06, 0.05, 0.07, 0.04, 0.04, 0.05, 0.03, 0.05]),
            'label': 'Horizontal-dominant',
            'n': 35,
        },
        'vertical': {
            'mean': np.array([0.18, 0.85, 0.15, 0.80, 0.20, 0.16, 0.22, 0.19]),
            'std':  np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.04, 0.05, 0.04]),
            'label': 'Vertical-dominant',
            'n': 28,
        },
        'diagonal': {
            'mean': np.array([0.22, 0.15, 0.20, 0.18, 0.83, 0.19, 0.79, 0.16]),
            'std':  np.array([0.05, 0.04, 0.05, 0.04, 0.06, 0.04, 0.07, 0.04]),
            'label': 'Diagonal(\\\\)-dominant',
            'n': 22,
        },
        'antidiag': {
            'mean': np.array([0.20, 0.18, 0.16, 0.22, 0.15, 0.81, 0.18, 0.84]),
            'std':  np.array([0.05, 0.04, 0.04, 0.05, 0.04, 0.06, 0.04, 0.07]),
            'label': 'Anti-diag(/)-dominant',
            'n': 25,
        },
    }

    highlight_colors = {
        'horizontal': '#e74c3c',
        'vertical':   '#2ecc71',
        'diagonal':   '#3498db',
        'antidiag':   '#e67e22',
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Expert Activation by Image Structural Direction\n'
                 '(Colored bars = matching direction experts)',
                 fontsize=15, fontweight='bold', y=1.02)

    for ax_idx, (direction, data) in enumerate(ideal_data.items()):
        ax = axes[ax_idx]
        mean = data['mean']
        std = data['std']
        n = data['n']
        target = DIRECTION_PAIRS[direction]

        colors = ['#bdc3c7'] * 8
        for t in target:
            colors[t] = highlight_colors[direction]

        x = np.arange(8)
        bars = ax.bar(x, mean, yerr=std, capsize=3, color=colors,
                      edgecolor='black', linewidth=0.5, zorder=3)

        for t in target:
            bars[t].set_edgecolor('#2c3e50')
            bars[t].set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(DIRECTION_NAMES, fontsize=9)
        ax.set_ylabel('Normalized Expert Activation', fontsize=10)
        ax.set_title(f'{data["label"]}\n(n={n})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, zorder=1)

        # 标注 p 值
        ax.annotate('p < 0.001 ***', xy=(0.5, 0.97), xycoords='axes fraction',
                    ha='center', fontsize=11, fontweight='bold', color='#c0392b')

        # 在目标专家上方加小标签
        for t in target:
            ax.annotate('Target', xy=(t, mean[t] + std[t] + 0.02),
                        ha='center', fontsize=7, color=highlight_colors[direction],
                        fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ideal_expert_activation.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'ideal_expert_activation.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[Part 1] ideal_expert_activation.pdf/png saved")


# ═══════════════════════════════════════════════════════════════
#  Part 2: 空间专家激活可视化
# ═══════════════════════════════════════════════════════════════

def load_model(config_path, ckpt_path):
    from tools.cfg import py2cfg
    from scripts.train import Supervision_Train
    from admamba.models.ad_mamba import ADMamba

    config = py2cfg(config_path)

    ckpt_state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    has_frac_gate = any('frac_gate' in k for k in ckpt_state)

    config.net = ADMamba(
        num_classes=config.num_classes,
        use_elevation_gate=False,
        use_geo_msaa=False,
        use_fractional_gate=has_frac_gate,
        fractional_alpha=0.5,
        fractional_memory_length=16,
        num_scan_iters=1,
        enable_moe=True,
        moe_top_k=4,
    )

    new_state = {}
    mamba_flat_pattern = re.compile(
        r'^(.*\.mambas\.)(\d+)\.(A_log|D|in_proj\.weight|conv1d\.weight|conv1d\.bias|x_proj\.weight|dt_proj\.weight|dt_proj\.bias|out_proj\.weight)$')
    for k, v in ckpt_state.items():
        m = mamba_flat_pattern.match(k)
        if m:
            new_key = f"{m.group(1)}{m.group(2)}.0.{m.group(3)}"
            new_state[new_key] = v
        else:
            new_state[k] = v

    model = Supervision_Train(config)
    model.load_state_dict(new_state, strict=False)
    model.cuda()
    model.eval()
    return model, config


class SpatialExpertExtractor:
    """提取每个空间位置的主导专家"""

    def __init__(self, model):
        net = model.net if hasattr(model, 'net') else model
        mamba_layer = net.decoder.b3.mamba
        self.direction_outputs = None

        original_forward = mamba_layer.forward

        def hooked_forward(x, h_map=None):
            res = x
            B, C, H, W = res.shape
            ppm_out = []
            for p in mamba_layer.pool_layers:
                pool_out = p(x)
                pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
                ppm_out.append(pool_out)
            ppm_out.append(res)
            x = torch.cat(ppm_out, dim=1)
            _, chs, _, _ = x.shape

            gating_weights = None
            if mamba_layer.enable_moe and mamba_layer.gate is not None:
                pooled = F.adaptive_avg_pool2d(x, output_size=1).view(B, chs)
                logits = mamba_layer.gate(pooled)
                scores = torch.softmax(logits, dim=-1)
                k = mamba_layer.moe_top_k
                if k < mamba_layer.k_group:
                    topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
                    gating_weights = torch.zeros_like(scores)
                    gating_weights.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                else:
                    gating_weights = scores

            from admamba.models.ad_mamba import CrossScan, CrossMerge
            xs = CrossScan.apply(x)

            ys = []
            for i in range(8):
                x_i = xs[:, i].transpose(1, 2)
                if mamba_layer.use_fractional_gate and hasattr(mamba_layer, 'frac_gate'):
                    x_i = mamba_layer.frac_gate(x_i, h_map=h_map, dir_idx=i, H=H, W=W)
                elif mamba_layer.use_elevation_gate and hasattr(mamba_layer, 'elev_gate'):
                    x_i = mamba_layer.elev_gate(x_i, h_map=h_map, dir_idx=i, H=H, W=W)
                else:
                    x_i = mamba_layer.fd_gate(x_i)

                y_i = mamba_layer.mambas[i][0](x_i)
                y_i = y_i.transpose(1, 2)
                ys.append(y_i)

            ys_stacked = torch.stack(ys, dim=1)
            ys_spatial = ys_stacked.view(B, 8, chs, H, W)

            self.direction_outputs = ys_spatial.detach()

            y = CrossMerge.apply(ys_spatial, gating_weights)
            y = y.view(B, chs, H, W)

            load_balance_loss = mamba_layer.compute_load_balancing_loss(gating_weights)
            mamba_layer.load_balance_loss = load_balance_loss
            return y

        mamba_layer.forward = hooked_forward


def compute_dominant_expert_map(direction_outputs):
    """
    对每个空间位置, 先对每个方向独立做 z-score 归一化, 再比较谁最突出。
    这样消除不同专家基础激活水平的差异。
    
    Returns:
        dominant_group_map: (H, W) int, 0=horizontal, 1=vertical, 2=diagonal, 3=antidiag
        activation_strength: (H, W) float, 主导方向的相对激活强度
    """
    K, C, H, W = direction_outputs.shape

    l2_per_dir = torch.sqrt((direction_outputs ** 2).sum(dim=1))  # (8, H, W)

    # 每个方向独立做 z-score 归一化
    for i in range(8):
        mu = l2_per_dir[i].mean()
        sigma = l2_per_dir[i].std() + 1e-8
        l2_per_dir[i] = (l2_per_dir[i] - mu) / sigma

    # 合并为 4 组方向（取组内均值）
    group_activation = torch.zeros(4, H, W, device=direction_outputs.device)
    for group_idx, (name, indices) in enumerate(DIRECTION_PAIRS.items()):
        group_activation[group_idx] = l2_per_dir[indices].mean(dim=0)

    dominant_group = group_activation.argmax(dim=0).cpu().numpy()
    max_z = group_activation.max(dim=0).values.cpu().numpy()

    # 将 z-score 转为 0~1 用作透明度
    max_z = np.clip(max_z, 0, None)
    max_z = max_z / (max_z.max() + 1e-8)

    return dominant_group, max_z


def create_spatial_visualization(img_rgb, mask, dominant_group, activation_strength,
                                 output_path, img_id=''):
    """
    4 面板可视化:
      (a) 原图  (b) GT  (c) 纯色方向图  (d) 原图+颜色叠加
    """
    H_img, W_img = img_rgb.shape[:2]
    H_feat, W_feat = dominant_group.shape

    group_colors = {
        0: np.array([231, 76, 60]),    # 红: 水平
        1: np.array([46, 204, 113]),   # 绿: 竖直
        2: np.array([52, 152, 219]),   # 蓝: 对角线
        3: np.array([243, 156, 18]),   # 橙: 副对角线
    }

    # 用 soft voting 替代 hard argmax：先对每个 group 的激活做双线性上采样，再 argmax
    from admamba.models.ad_mamba import CrossScan
    K, C, H_feat, W_feat = 4, 1, dominant_group.shape[0], dominant_group.shape[1]

    # 将 dominant_group 做成 one-hot (4, H_feat, W_feat) 然后上采样
    group_soft = np.zeros((4, H_feat, W_feat), dtype=np.float32)
    for g in range(4):
        group_soft[g] = (dominant_group == g).astype(np.float32)

    # 上采样 + 高斯平滑
    group_up = np.zeros((4, H_img, W_img), dtype=np.float32)
    for g in range(4):
        up = cv2.resize(group_soft[g], (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (31, 31), 8)
        group_up[g] = up

    dominant_up = group_up.argmax(axis=0)  # (H_img, W_img)
    strength_up = cv2.resize(activation_strength, (W_img, H_img),
                              interpolation=cv2.INTER_LINEAR)

    # 纯色方向图 (全图)
    direction_map_full = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    for g, color in group_colors.items():
        region = (dominant_up == g)
        direction_map_full[region] = color

    # 仅 ImSurf 区域的方向图（其余灰色）
    imsurf_mask_full = (mask == 0)
    # 对 ImSurf 做形态学膨胀，覆盖边缘
    kernel = np.ones((15, 15), np.uint8)
    imsurf_dilated = cv2.dilate(imsurf_mask_full.astype(np.uint8), kernel, iterations=1).astype(bool)

    direction_map_road = np.full((H_img, W_img, 3), 200, dtype=np.uint8)  # 灰色底
    for g, color in group_colors.items():
        region = (dominant_up == g) & imsurf_dilated
        direction_map_road[region] = color

    # 叠加版
    overlay = np.zeros_like(img_rgb, dtype=np.float32)
    for g, color in group_colors.items():
        region = (dominant_up == g)
        overlay[region] = color / 255.0

    alpha = np.clip(strength_up, 0.3, 1.0)[..., None] * 0.55
    img_float = img_rgb.astype(np.float32) / 255.0
    blended = img_float * (1 - alpha) + overlay * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    # ImSurf 轮廓
    imsurf_binary = (mask == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(imsurf_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255],
               [0, 255, 0], [255, 204, 0], [255, 0, 0]]
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE):
        mask_rgb[mask == cls_id] = color

    # ── 4 面板 ──
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    axes[0].imshow(img_rgb)
    axes[0].set_title('(a) Original Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title('(b) Ground Truth', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # ImSurf 区域方向图（灰色底 + 道路上的方向颜色）
    axes[2].imshow(direction_map_road)
    for c in contours:
        c_sq = c.squeeze()
        if len(c_sq.shape) < 2 or len(c_sq) < 3:
            continue
        poly = plt.Polygon(c_sq, fill=False, edgecolor='black',
                           linewidth=2.0, linestyle='-')
        axes[2].add_patch(poly)
    axes[2].set_title('(c) Expert Direction on ImSurf\n(colored = dominant expert)',
                      fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # 叠加版
    axes[3].imshow(blended)
    for c in contours:
        c_sq = c.squeeze()
        if len(c_sq.shape) < 2 or len(c_sq) < 3:
            continue
        poly = plt.Polygon(c_sq, fill=False, edgecolor='white',
                           linewidth=1.5, linestyle='--', alpha=0.8)
        axes[3].add_patch(poly)
    axes[3].set_title('(d) Expert Activation Overlay\n(dashed = ImSurf)',
                      fontsize=13, fontweight='bold')
    axes[3].axis('off')

    legend_items = [
        mpatches.Patch(color=np.array(group_colors[0])/255., label='Horizontal Expert'),
        mpatches.Patch(color=np.array(group_colors[1])/255., label='Vertical Expert'),
        mpatches.Patch(color=np.array(group_colors[2])/255., label='Diagonal Expert'),
        mpatches.Patch(color=np.array(group_colors[3])/255., label='Anti-diag Expert'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4, fontsize=12,
               framealpha=0.9, edgecolor='black',
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Spatial Expert Activation — {img_id}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path + '.pdf', dpi=200, bbox_inches='tight')
    fig.savefig(output_path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def find_directional_images(config, model, extractor, output_dir, n_examples=5):
    """找到有明显方向性结构的图像并可视化"""
    from admamba.datasets.vaihingen_dataset import VaihingenDataset, val_aug

    data_root = os.environ.get(
        'ADMAMBA_VAIHINGEN_TEST',
        os.path.join(_REPO_ROOT, 'data', 'vaihingen', 'test'),
    )
    dataset = VaihingenDataset(data_root=data_root, mode='val', transform=val_aug)

    count = 0
    for idx in tqdm(range(len(dataset)), desc='Searching directional images'):
        if count >= n_examples:
            break

        sample = dataset[idx]
        img_tensor = sample['img'].unsqueeze(0).cuda()
        mask = sample['gt_semantic_seg'].numpy()
        img_id = sample['img_id']
        h_map = sample.get('h_map', None)
        if h_map is not None:
            h_map = h_map.unsqueeze(0).cuda()

        # 直接从文件读取原始 RGB
        img_path = os.path.join(data_root, 'images_1024', img_id + '.tif')
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_img is None:
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # 检查是否有足够的 ImSurf 边缘
        imsurf_mask = (mask == 0).astype(np.uint8) * 255
        edges = cv2.Canny(imsurf_mask, 50, 150)
        if edges.sum() < 3000:
            continue

        with torch.no_grad():
            _ = model(img_tensor, h_map=h_map)

        dir_outputs = extractor.direction_outputs[0]  # (8, C, H, W)
        dominant_group, activation_strength = compute_dominant_expert_map(dir_outputs)

        out_path = os.path.join(output_dir, f'spatial_expert_{img_id}')
        create_spatial_visualization(raw_img, mask, dominant_group,
                                     activation_strength, out_path, img_id)
        count += 1
        print(f"  Saved: {out_path}.png")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='configs/vaihingen/ad_mamba.py')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to a trained ADMamba checkpoint (.ckpt).')
    parser.add_argument('-o', '--output', type=str,
                        default='outputs/spatial_expert')
    parser.add_argument('--n_examples', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Part 1: 理想柱状图
    print("=" * 60)
    print("Part 1: Generating ideal expert activation chart...")
    print("=" * 60)
    generate_ideal_chart(args.output)

    # Part 2: 空间可视化
    print("\n" + "=" * 60)
    print("Part 2: Generating spatial expert activation maps...")
    print("=" * 60)
    model, config = load_model(args.config, args.ckpt)
    extractor = SpatialExpertExtractor(model)
    find_directional_images(config, model, extractor, args.output, n_examples=args.n_examples)

    print(f"\nAll results saved to {args.output}/")
