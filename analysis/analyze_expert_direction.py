"""
方向感知专家激活分析 (Direction-Aware Expert Activation Analysis)

证明 MoE-Mamba 的门控网络能够根据图像中线性特征（如道路）的主导方向，
自适应地为对应方向的扫描专家分配更高的权重。

实验流程：
1. 从标签掩码中提取 ImSurf（不透水面 / 道路）区域
2. 对 ImSurf 区域做梯度方向直方图分析，得到每张图的"主导边缘方向"
3. 用 hook 提取 MoE 门控网络输出的 8 方向专家权重
4. 按主导方向分组统计专家权重，验证方向一致性
5. 生成可视化图表
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from tqdm import tqdm
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.cfg import py2cfg
from scripts.train import Supervision_Train

DIRECTION_NAMES = [
    'H→',    # 0: 横向(左→右)
    'V↓',    # 1: 竖向(上→下)
    '←H',    # 2: 横向反向(右→左)
    '↑V',    # 3: 竖向反向(下→上)
    '↘D',    # 4: 主对角线
    '↗A',    # 5: 副对角线
    '↖D',    # 6: 主对角线反向
    '↙A',    # 7: 副对角线反向
]

DIRECTION_PAIRS = {
    'horizontal': [0, 2],
    'vertical':   [1, 3],
    'diagonal':   [4, 6],
    'antidiag':   [5, 7],
}

DIRECTION_LABELS_CN = {
    'horizontal': '水平方向',
    'vertical':   '竖直方向',
    'diagonal':   '主对角线方向',
    'antidiag':   '副对角线方向',
}

# ────────────────────── 方向检测 ──────────────────────

def compute_dominant_direction(mask, imsurf_class=0, num_bins=180):
    """
    对 ImSurf 区域的边缘做梯度方向直方图，返回主导方向类别。

    Returns:
        direction: str, one of 'horizontal', 'vertical', 'diagonal', 'antidiag'
        orientation_hist: np.ndarray, shape (num_bins,), 归一化方向直方图
        dominant_angle: float, 主导角度 (度)
        confidence: float, 主导方向与其他方向的比值
    """
    binary = (mask == imsurf_class).astype(np.uint8) * 255
    if binary.sum() < 500:
        return None, None, None, 0.0

    edges = cv2.Canny(binary, 50, 150)
    if edges.sum() == 0:
        return None, None, None, 0.0

    grad_x = cv2.Sobel(binary.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(binary.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)  # [-pi, pi]
    angle_deg = np.degrees(angle) % 180  # 映射到 [0, 180)

    edge_mask = edges > 0
    if edge_mask.sum() < 50:
        return None, None, None, 0.0

    mag_at_edges = magnitude[edge_mask]
    ang_at_edges = angle_deg[edge_mask]

    hist, bin_edges = np.histogram(ang_at_edges, bins=num_bins, range=(0, 180),
                                   weights=mag_at_edges)
    hist = hist / (hist.sum() + 1e-8)

    # 将直方图分为4个方向区间
    # 水平边缘 → 竖直走向的结构 (边缘角度~0° 或~180° → 结构走向~90°)
    # 竖直边缘 → 水平走向的结构 (边缘角度~90° → 结构走向~0°)
    # 注意：梯度方向 ⊥ 边缘方向 ⊥ 结构走向
    #
    # 边缘梯度角度 vs 结构走向:
    #   梯度 0°/180° (水平梯度) → 边缘竖直 → 结构竖直
    #   梯度 90° (竖直梯度) → 边缘水平 → 结构水平
    #   梯度 45° → 边缘对角线 → 结构对角线
    #   梯度 135° → 边缘反对角线 → 结构反对角线
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 按结构走向分组（梯度方向 = 结构走向 ± 90° 的关系）
    # 竖直结构：梯度方向在 [0,22.5) ∪ [157.5,180)（水平梯度 → 竖直边缘）
    # 水平结构：梯度方向在 [67.5, 112.5)（竖直梯度 → 水平边缘）
    # 对角线结构：梯度方向在 [112.5, 157.5)（对角梯度 → 对角边缘）
    # 反对角线结构：梯度方向在 [22.5, 67.5)

    dir_energy = {}
    for i, c in enumerate(bin_centers):
        if c < 22.5 or c >= 157.5:
            dir_energy.setdefault('vertical', 0.0)
            dir_energy['vertical'] += hist[i]
        elif 22.5 <= c < 67.5:
            dir_energy.setdefault('antidiag', 0.0)
            dir_energy['antidiag'] += hist[i]
        elif 67.5 <= c < 112.5:
            dir_energy.setdefault('horizontal', 0.0)
            dir_energy['horizontal'] += hist[i]
        elif 112.5 <= c < 157.5:
            dir_energy.setdefault('diagonal', 0.0)
            dir_energy['diagonal'] += hist[i]

    for d in ['horizontal', 'vertical', 'diagonal', 'antidiag']:
        dir_energy.setdefault(d, 0.0)

    dominant = max(dir_energy, key=dir_energy.get)
    total = sum(dir_energy.values()) + 1e-8
    confidence = dir_energy[dominant] / total

    dominant_angle = {'vertical': 90, 'horizontal': 0, 'diagonal': 45, 'antidiag': 135}[dominant]

    return dominant, hist, dominant_angle, confidence


# ────────────────────── 专家权重提取 ──────────────────────

class ExpertWeightExtractor:
    """用 hook 提取 MoE 门控权重"""

    def __init__(self, model):
        self.model = model
        self.gating_weights = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        net = self.model.net if hasattr(self.model, 'net') else self.model
        mamba_layer = net.decoder.b3.mamba

        def hook_fn(module, input, output):
            if hasattr(module, 'enable_moe') and module.enable_moe:
                # 重新计算门控权重
                x = input[0]
                B, C, H, W = x.shape
                ppm_out = []
                for p in module.pool_layers:
                    pool_out = p(x)
                    pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
                    ppm_out.append(pool_out)
                ppm_out.append(x)
                x_cat = torch.cat(ppm_out, dim=1)
                _, chs, _, _ = x_cat.shape

                pooled = F.adaptive_avg_pool2d(x_cat, output_size=1).view(B, chs)
                logits = module.gate(pooled)
                scores = torch.softmax(logits, dim=-1)

                k = module.moe_top_k
                if k < module.k_group:
                    topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
                    gating_weights = torch.zeros_like(scores)
                    gating_weights.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                else:
                    gating_weights = scores

                self.gating_weights = gating_weights.detach().cpu().numpy()

        self._hook = mamba_layer.register_forward_hook(hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


# ────────────────────── 主实验 ──────────────────────

def run_experiment(config_path, ckpt_path=None, output_dir='outputs/expert_direction',
                   confidence_threshold=0.35, max_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    config = py2cfg(config_path)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.weights_path,
                                 config.test_weights_name + '.ckpt')

    print(f"Loading model from: {ckpt_path}")

    # 探测 checkpoint 的 num_scan_iters 和 frac_gate 设置
    ckpt_state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    has_frac_gate = any('frac_gate' in k for k in ckpt_state)
    has_iter_scale = any('iter_scale' in k for k in ckpt_state)
    # 如果没有 iter_scale，则 checkpoint 是 num_scan_iters=1
    ckpt_num_scan_iters = 1
    if has_iter_scale:
        for k, v in ckpt_state.items():
            if 'iter_scale' in k:
                ckpt_num_scan_iters = v.numel()
                break

    print(f"  Checkpoint: frac_gate={has_frac_gate}, num_scan_iters={ckpt_num_scan_iters}")

    # 重建模型，始终使用 num_scan_iters=1 匹配 checkpoint
    from admamba.models.ad_mamba import ADMamba
    config.net = ADMamba(
        num_classes=config.num_classes,
        use_elevation_gate=False,
        use_geo_msaa=False,
        use_fractional_gate=has_frac_gate,
        fractional_alpha=0.5,
        fractional_memory_length=16,
        num_scan_iters=ckpt_num_scan_iters,
        enable_moe=True,
        moe_top_k=4,
    )

    # checkpoint 中 mambas 可能是 flat list (mambas.i.param)
    # 而模型是 nested ModuleList (mambas.i.0.param)
    # 需要做 key 映射
    import re
    new_state = {}
    remap_count = 0
    mamba_flat_pattern = re.compile(
        r'^(.*\.mambas\.)(\d+)\.(A_log|D|in_proj\.weight|conv1d\.weight|conv1d\.bias|x_proj\.weight|dt_proj\.weight|dt_proj\.bias|out_proj\.weight)$')

    for k, v in ckpt_state.items():
        m = mamba_flat_pattern.match(k)
        if m:
            new_key = f"{m.group(1)}{m.group(2)}.0.{m.group(3)}"
            new_state[new_key] = v
            remap_count += 1
        else:
            new_state[k] = v

    print(f"  Remapped {remap_count} mamba keys (flat -> nested)")

    # 用映射后的 state_dict 手动加载
    model = Supervision_Train(config)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_trivial_missing = [k for k in missing
                           if 'num_batches_tracked' not in k and 'iter_scale' not in k]
    if non_trivial_missing:
        print(f"  WARNING: {len(non_trivial_missing)} missing keys (first 5):")
        for k in non_trivial_missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5):")
        for k in unexpected[:5]:
            print(f"    {k}")
    if not non_trivial_missing and not unexpected:
        print("  Model loaded perfectly - all keys matched!")
    model.cuda()
    model.eval()

    extractor = ExpertWeightExtractor(model)

    test_dataset = config.test_dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, pin_memory=True, shuffle=False
    )

    results = []
    direction_groups = defaultdict(list)

    print("Running inference and extracting expert weights...")
    for idx, batch in enumerate(tqdm(test_loader)):
        if max_samples and idx >= max_samples:
            break

        img = batch['img'].cuda()
        mask = batch['gt_semantic_seg'].numpy()[0]  # (H, W)
        img_id = batch['img_id'][0]
        h_map = batch.get('h_map', None)
        if h_map is not None:
            h_map = h_map.cuda()

        # 1. 检测主导方向
        dominant_dir, hist, dom_angle, confidence = compute_dominant_direction(mask)

        if dominant_dir is None or confidence < confidence_threshold:
            continue

        # 2. 前向推理 → hook 提取专家权重
        with torch.no_grad():
            _ = model(img, h_map=h_map)

        gating_weights = extractor.gating_weights[0]  # (8,)

        record = {
            'img_id': img_id,
            'dominant_direction': dominant_dir,
            'confidence': confidence,
            'dominant_angle': dom_angle,
            'gating_weights': gating_weights.copy(),
            'direction_hist': hist,
        }
        results.append(record)
        direction_groups[dominant_dir].append(record)

    extractor.remove()

    print(f"\n=== 统计结果 ===")
    print(f"总样本数: {len(results)}")
    for d, recs in direction_groups.items():
        print(f"  {DIRECTION_LABELS_CN[d]}: {len(recs)} 张图像")

    # ── 计算分组统计 ──
    group_stats = {}
    for direction, records in direction_groups.items():
        weights = np.stack([r['gating_weights'] for r in records])  # (N, 8)
        mean_w = weights.mean(axis=0)
        std_w = weights.std(axis=0)

        expected_experts = DIRECTION_PAIRS[direction]
        expected_weight = mean_w[expected_experts].sum()
        other_experts = [i for i in range(8) if i not in expected_experts]
        other_weight = mean_w[other_experts].mean() * 2

        group_stats[direction] = {
            'mean_weights': mean_w,
            'std_weights': std_w,
            'n_samples': len(records),
            'expected_experts': expected_experts,
            'expected_weight_sum': expected_weight,
            'other_weight_sum': other_weight,
        }

        print(f"\n--- {DIRECTION_LABELS_CN[direction]} ({len(records)} 样本) ---")
        print(f"  平均专家权重: {mean_w}")
        print(f"  对应方向专家 {expected_experts} 的权重和: {expected_weight:.4f}")
        print(f"  其他方向专家的平均权重和(×2): {other_weight:.4f}")
        print(f"  比率: {expected_weight / (other_weight + 1e-8):.2f}x")

    # ── 统计检验 ──
    print("\n=== 统计检验 ===")
    for direction, records in direction_groups.items():
        if len(records) < 5:
            print(f"  {DIRECTION_LABELS_CN[direction]}: 样本太少，跳过检验")
            continue

        weights = np.stack([r['gating_weights'] for r in records])
        expected = DIRECTION_PAIRS[direction]
        expected_w = weights[:, expected].sum(axis=1)
        other_indices = [i for i in range(8) if i not in expected]
        other_w = weights[:, other_indices].sum(axis=1) / 3.0

        t_stat, p_value = stats.ttest_rel(expected_w, other_w)
        print(f"  {DIRECTION_LABELS_CN[direction]}: t={t_stat:.3f}, p={p_value:.6f} "
              f"{'*** (p<0.001)' if p_value < 0.001 else '** (p<0.01)' if p_value < 0.01 else '* (p<0.05)' if p_value < 0.05 else '(不显著)'}")

    # ── 生成可视化 ──
    generate_visualizations(group_stats, direction_groups, results, output_dir)

    # ── 保存原始数据 ──
    save_raw_data(results, group_stats, output_dir)

    print(f"\n结果已保存到 {output_dir}/")
    return results, group_stats


# ────────────────────── 可视化 ──────────────────────

def generate_visualizations(group_stats, direction_groups, results, output_dir):
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    # ━━━ Figure 1: 分组柱状图 ━━━
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Weights by Dominant Image Direction', fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    highlight_colors = {
        'horizontal': ['#e74c3c' if i in [0, 2] else '#95a5a6' for i in range(8)],
        'vertical':   ['#2ecc71' if i in [1, 3] else '#95a5a6' for i in range(8)],
        'diagonal':   ['#3498db' if i in [4, 6] else '#95a5a6' for i in range(8)],
        'antidiag':   ['#e67e22' if i in [5, 7] else '#95a5a6' for i in range(8)],
    }

    for ax_idx, (direction, label_cn) in enumerate(DIRECTION_LABELS_CN.items()):
        ax = axes[ax_idx // 2, ax_idx % 2]

        if direction not in group_stats:
            ax.set_title(f'{label_cn}\n(No samples)')
            ax.set_visible(False)
            continue

        stats_d = group_stats[direction]
        mean_w = stats_d['mean_weights']
        std_w = stats_d['std_weights']
        n = stats_d['n_samples']
        expected = stats_d['expected_experts']

        bar_colors = highlight_colors[direction]
        x = np.arange(8)
        bars = ax.bar(x, mean_w, yerr=std_w, capsize=3, color=bar_colors,
                      edgecolor='black', linewidth=0.5)

        for i in expected:
            bars[i].set_edgecolor('#2c3e50')
            bars[i].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(DIRECTION_NAMES, fontsize=9)
        ax.set_ylabel('Expert Weight')
        ax.set_title(f'{label_cn} (n={n})', fontsize=13, fontweight='bold')

        expected_sum = mean_w[expected].sum()
        ax.annotate(f'Target experts sum: {expected_sum:.3f}',
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'expert_weights_by_direction.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'expert_weights_by_direction.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [1/4] expert_weights_by_direction.pdf/png")

    # ━━━ Figure 2: 雷达图（蛛网图） ━━━
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
    fig.suptitle('Expert Weight Radar by Direction Group', fontsize=16, fontweight='bold', y=1.02)

    radar_colors = {
        'horizontal': '#e74c3c',
        'vertical':   '#2ecc71',
        'diagonal':   '#3498db',
        'antidiag':   '#e67e22',
    }

    for ax_idx, (direction, label_cn) in enumerate(DIRECTION_LABELS_CN.items()):
        ax = axes[ax_idx // 2, ax_idx % 2]

        if direction not in group_stats:
            ax.set_visible(False)
            continue

        mean_w = group_stats[direction]['mean_weights']
        n = group_stats[direction]['n_samples']

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False).tolist()
        angles += angles[:1]
        values = mean_w.tolist() + [mean_w[0]]

        ax.plot(angles, values, 'o-', color=radar_colors[direction], linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=radar_colors[direction])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIRECTION_NAMES, fontsize=9)
        ax.set_title(f'{label_cn} (n={n})', fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'expert_weights_radar.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'expert_weights_radar.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [2/4] expert_weights_radar.pdf/png")

    # ━━━ Figure 3: 对应方向 vs 非对应方向的权重对比 ━━━
    fig, ax = plt.subplots(figsize=(10, 6))

    directions_with_data = [d for d in DIRECTION_LABELS_CN if d in group_stats]
    x = np.arange(len(directions_with_data))
    width = 0.35

    target_weights = []
    other_weights = []
    labels = []

    for d in directions_with_data:
        s = group_stats[d]
        target_weights.append(s['expected_weight_sum'])
        other_weights.append(s['other_weight_sum'])
        labels.append(DIRECTION_LABELS_CN[d])

    bars1 = ax.bar(x - width/2, target_weights, width, label='Target Direction Experts',
                   color='#27ae60', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, other_weights, width, label='Other Direction Experts (avg×2)',
                   color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dominant Direction in Image', fontsize=12)
    ax.set_ylabel('Sum of Expert Weights', fontsize=12)
    ax.set_title('Target vs Non-target Expert Weights', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)

    for b1, b2 in zip(bars1, bars2):
        ratio = b1.get_height() / (b2.get_height() + 1e-8)
        ax.annotate(f'{ratio:.2f}x',
                    xy=(b1.get_x() + width, max(b1.get_height(), b2.get_height()) + 0.01),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'target_vs_other_weights.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'target_vs_other_weights.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [3/4] target_vs_other_weights.pdf/png")

    # ━━━ Figure 4: 热力图 ━━━
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = []
    row_labels = []
    for d in ['horizontal', 'vertical', 'diagonal', 'antidiag']:
        if d in group_stats:
            matrix.append(group_stats[d]['mean_weights'])
            row_labels.append(DIRECTION_LABELS_CN[d])

    if matrix:
        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(DIRECTION_NAMES, fontsize=10)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=11)

        for i in range(len(row_labels)):
            for j in range(8):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                               fontsize=9, color='black' if matrix[i, j] < matrix.max() * 0.7 else 'white')

        fig.colorbar(im, ax=ax, shrink=0.8, label='Expert Weight')
        ax.set_title('Expert Weight Heatmap (rows=image direction, cols=expert direction)',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'expert_weights_heatmap.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'expert_weights_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [4/4] expert_weights_heatmap.pdf/png")


def save_raw_data(results, group_stats, output_dir):
    import json

    data = []
    for r in results:
        data.append({
            'img_id': r['img_id'],
            'dominant_direction': r['dominant_direction'],
            'confidence': float(r['confidence']),
            'dominant_angle': float(r['dominant_angle']),
            'gating_weights': r['gating_weights'].tolist(),
        })

    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump(data, f, indent=2)

    summary = {}
    for d, s in group_stats.items():
        summary[d] = {
            'n_samples': s['n_samples'],
            'mean_weights': s['mean_weights'].tolist(),
            'std_weights': s['std_weights'].tolist(),
            'expected_experts': s['expected_experts'],
            'expected_weight_sum': float(s['expected_weight_sum']),
            'other_weight_sum': float(s['other_weight_sum']),
            'ratio': float(s['expected_weight_sum'] / (s['other_weight_sum'] + 1e-8)),
        }

    with open(os.path.join(output_dir, 'group_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Raw data saved to {output_dir}/raw_results.json")
    print(f"  Summary saved to {output_dir}/group_summary.json")


# ────────────────────── 入口 ──────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direction-Aware Expert Activation Analysis')
    parser.add_argument('-c', '--config', type=str,
                        default='configs/vaihingen/ad_mamba.py',
                        help='Path to config file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint (default: auto from config)')
    parser.add_argument('-o', '--output', type=str,
                        default='outputs/expert_direction',
                        help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.35,
                        help='Minimum confidence threshold for direction classification')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process')
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        max_samples=args.max_samples,
    )
