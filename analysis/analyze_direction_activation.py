"""
方向感知特征激活分析 (Direction-Aware Feature Activation Analysis)

核心思路：分析每个方向 Mamba 专家的输出特征在道路区域的激活强度，
证明与道路走向匹配的扫描方向能够产生更强的特征响应。

分析方法：
  1. Hook 截取 MambaLayer 的 8 个方向 Mamba 输出 (B,8,C,H,W)
  2. 根据 GT 掩码提取 ImSurf（道路/不透水面）区域
  3. 分析 ImSurf 边缘的梯度方向 → 判定道路主导走向
  4. 对每个方向的 Mamba 输出，计算在道路区域的平均特征激活强度
  5. 验证：道路走向与扫描方向匹配时，特征激活更强
"""

import os
import sys
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.cfg import py2cfg
from scripts.train import Supervision_Train

DIRECTION_NAMES_EN = [
    'H(L->R)', 'V(T->B)', 'H(R->L)', 'V(B->T)',
    'D(\\)', 'A(/)', 'D(\\)rev', 'A(/)rev',
]

DIRECTION_PAIRS = {
    'horizontal': [0, 2],
    'vertical':   [1, 3],
    'diagonal':   [4, 6],
    'antidiag':   [5, 7],
}

DIRECTION_LABELS_EN = {
    'horizontal': 'Horizontal',
    'vertical':   'Vertical',
    'diagonal':   'Diagonal(\\)',
    'antidiag':   'Anti-diag(/)',
}


def compute_dominant_direction(mask, imsurf_class=0, num_bins=180):
    """对 ImSurf 边缘做梯度方向直方图，返回主导结构走向"""
    binary = (mask == imsurf_class).astype(np.uint8) * 255
    if binary.sum() < 500:
        return None, 0.0

    edges = cv2.Canny(binary, 50, 150)
    if edges.sum() == 0:
        return None, 0.0

    grad_x = cv2.Sobel(binary.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(binary.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle_deg = np.degrees(np.arctan2(grad_y, grad_x)) % 180

    edge_mask = edges > 0
    if edge_mask.sum() < 50:
        return None, 0.0

    mag_at_edges = magnitude[edge_mask]
    ang_at_edges = angle_deg[edge_mask]

    hist, bin_edges = np.histogram(ang_at_edges, bins=num_bins, range=(0, 180),
                                   weights=mag_at_edges)
    hist = hist / (hist.sum() + 1e-8)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    dir_energy = {'horizontal': 0.0, 'vertical': 0.0, 'diagonal': 0.0, 'antidiag': 0.0}
    for i, c in enumerate(bin_centers):
        if c < 22.5 or c >= 157.5:
            dir_energy['vertical'] += hist[i]
        elif 22.5 <= c < 67.5:
            dir_energy['antidiag'] += hist[i]
        elif 67.5 <= c < 112.5:
            dir_energy['horizontal'] += hist[i]
        elif 112.5 <= c < 157.5:
            dir_energy['diagonal'] += hist[i]

    dominant = max(dir_energy, key=dir_energy.get)
    total = sum(dir_energy.values()) + 1e-8
    confidence = dir_energy[dominant] / total

    return dominant, confidence


class PerDirectionActivationExtractor:
    """截取每个方向 Mamba 的输出特征 (B,8,C,H,W)"""

    def __init__(self, model):
        self.model = model
        self.direction_outputs = None
        self.gating_weights = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        net = self.model.net if hasattr(self.model, 'net') else self.model
        mamba_layer = net.decoder.b3.mamba

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

            # 门控权重
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

                if mamba_layer.num_scan_iters > 1:
                    y_i_accum = torch.zeros_like(x_i)
                    x_iter = x_i
                    for iter_idx in range(mamba_layer.num_scan_iters):
                        y_iter = mamba_layer.mambas[i][iter_idx](x_iter)
                        y_i_accum = y_i_accum + mamba_layer.iter_scale[iter_idx] * y_iter
                        x_iter = x_i + y_iter
                    y_i = y_i_accum
                else:
                    y_i = mamba_layer.mambas[i][0](x_i)

                y_i = y_i.transpose(1, 2)
                ys.append(y_i)

            ys_stacked = torch.stack(ys, dim=1)
            ys_spatial = ys_stacked.view(B, 8, chs, H, W)

            # 保存截取数据
            self.direction_outputs = ys_spatial.detach().cpu()
            self.gating_weights = gating_weights.detach().cpu().numpy() if gating_weights is not None else None

            y = CrossMerge.apply(ys_spatial, gating_weights)
            y = y.view(B, chs, H, W)

            load_balance_loss = mamba_layer.compute_load_balancing_loss(gating_weights)
            mamba_layer.load_balance_loss = load_balance_loss

            return y

        mamba_layer.forward = hooked_forward

    def remove(self):
        pass


def compute_activation_metrics(direction_outputs, mask, imsurf_class=0, feat_h=None, feat_w=None):
    """
    计算每个方向在道路区域的特征激活强度
    
    Args:
        direction_outputs: (8, C, H_feat, W_feat)
        mask: (H_img, W_img) GT 掩码
        imsurf_class: ImSurf 类别 id
    
    Returns:
        metrics dict: 每个方向的激活强度指标
    """
    _, C, H_feat, W_feat = direction_outputs.shape
    H_img, W_img = mask.shape

    road_mask = (mask == imsurf_class)
    road_mask_resized = cv2.resize(road_mask.astype(np.uint8),
                                    (W_feat, H_feat),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

    if road_mask_resized.sum() < 10:
        return None

    non_road_mask = ~road_mask_resized
    if non_road_mask.sum() < 10:
        return None

    metrics = {}
    for d in range(8):
        feat = direction_outputs[d]  # (C, H, W)
        feat_np = feat.numpy()

        # 道路区域的平均特征强度 (channel-wise L2 norm)
        road_feat = feat_np[:, road_mask_resized]  # (C, N_road)
        non_road_feat = feat_np[:, non_road_mask]  # (C, N_non_road)

        road_l2 = np.sqrt((road_feat ** 2).sum(axis=0)).mean()
        non_road_l2 = np.sqrt((non_road_feat ** 2).sum(axis=0)).mean()

        road_mean_abs = np.abs(road_feat).mean()
        non_road_mean_abs = np.abs(non_road_feat).mean()

        # 道路区域 vs 非道路区域的激活比率
        activation_ratio = road_l2 / (non_road_l2 + 1e-8)

        # 道路区域的特征方差（信息丰富度）
        road_variance = road_feat.var(axis=1).mean()

        metrics[d] = {
            'road_l2': float(road_l2),
            'non_road_l2': float(non_road_l2),
            'activation_ratio': float(activation_ratio),
            'road_mean_abs': float(road_mean_abs),
            'non_road_mean_abs': float(non_road_mean_abs),
            'road_variance': float(road_variance),
        }

    return metrics


def run_experiment(config_path, ckpt_path=None, output_dir='outputs/direction_activation',
                   confidence_threshold=0.35, max_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    config = py2cfg(config_path)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.weights_path,
                                 config.test_weights_name + '.ckpt')

    print(f"Loading model from: {ckpt_path}")

    ckpt_state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    has_frac_gate = any('frac_gate' in k for k in ckpt_state)
    has_iter_scale = any('iter_scale' in k for k in ckpt_state)
    ckpt_num_scan_iters = 1
    if has_iter_scale:
        for k, v in ckpt_state.items():
            if 'iter_scale' in k:
                ckpt_num_scan_iters = v.numel()
                break

    print(f"  Checkpoint: frac_gate={has_frac_gate}, num_scan_iters={ckpt_num_scan_iters}")

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

    model = Supervision_Train(config)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_trivial = [k for k in missing if 'num_batches_tracked' not in k and 'iter_scale' not in k]
    if non_trivial:
        print(f"  WARNING: {len(non_trivial)} missing keys")
    else:
        print(f"  Model loaded OK (remapped {remap_count} keys)")
    model.cuda()
    model.eval()

    extractor = PerDirectionActivationExtractor(model)

    test_dataset = config.test_dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, pin_memory=True, shuffle=False
    )

    results = []
    direction_groups = defaultdict(list)

    print("Running inference and extracting per-direction activations...")
    for idx, batch in enumerate(tqdm(test_loader)):
        if max_samples and idx >= max_samples:
            break

        img = batch['img'].cuda()
        mask = batch['gt_semantic_seg'].numpy()[0]
        img_id = batch['img_id'][0]
        h_map = batch.get('h_map', None)
        if h_map is not None:
            h_map = h_map.cuda()

        dominant_dir, confidence = compute_dominant_direction(mask)
        if dominant_dir is None or confidence < confidence_threshold:
            continue

        with torch.no_grad():
            _ = model(img, h_map=h_map)

        dir_outputs = extractor.direction_outputs[0]  # (8, C, H, W)
        gating_w = extractor.gating_weights[0] if extractor.gating_weights is not None else None

        metrics = compute_activation_metrics(dir_outputs, mask)
        if metrics is None:
            continue

        record = {
            'img_id': img_id,
            'dominant_direction': dominant_dir,
            'confidence': confidence,
            'activation_metrics': metrics,
            'gating_weights': gating_w.tolist() if gating_w is not None else None,
        }
        results.append(record)
        direction_groups[dominant_dir].append(record)

    print(f"\nTotal valid samples: {len(results)}")
    for d, recs in direction_groups.items():
        print(f"  {DIRECTION_LABELS_EN[d]}: {len(recs)} images")

    # ── 统计分析 ──
    analyze_and_visualize(results, direction_groups, output_dir)

    return results


def analyze_and_visualize(results, direction_groups, output_dir):
    """核心分析：使用道路选择性指标（activation_ratio = road/non-road）"""

    group_activation = {}
    for direction, records in direction_groups.items():
        n = len(records)
        if n < 2:
            continue

        all_ratios = np.zeros((n, 8))
        all_road_l2 = np.zeros((n, 8))

        for i, r in enumerate(records):
            for d in range(8):
                all_ratios[i, d] = r['activation_metrics'][d]['activation_ratio']
                all_road_l2[i, d] = r['activation_metrics'][d]['road_l2']

        # 用 activation_ratio 做归一化：每个样本内 8 方向做 min-max 归一化
        all_ratios_norm = np.zeros_like(all_ratios)
        for i in range(n):
            mn, mx = all_ratios[i].min(), all_ratios[i].max()
            if mx > mn:
                all_ratios_norm[i] = (all_ratios[i] - mn) / (mx - mn)

        # 同时保留原始 road_l2 的归一化
        all_road_l2_norm = np.zeros_like(all_road_l2)
        for i in range(n):
            mn, mx = all_road_l2[i].min(), all_road_l2[i].max()
            if mx > mn:
                all_road_l2_norm[i] = (all_road_l2[i] - mn) / (mx - mn)

        # 跳过有 0 激活专家的样本（已坍塌的专家）
        active_mask = (all_road_l2.min(axis=1) > 0) | True  # 保留所有

        group_activation[direction] = {
            'mean_ratio_norm': all_ratios_norm.mean(axis=0),
            'std_ratio_norm': all_ratios_norm.std(axis=0),
            'mean_ratio_raw': all_ratios.mean(axis=0),
            'std_ratio_raw': all_ratios.std(axis=0),
            'mean_road_l2': all_road_l2_norm.mean(axis=0),
            'std_road_l2': all_road_l2_norm.std(axis=0),
            'n_samples': n,
            'raw_ratios_norm': all_ratios_norm,
            'raw_road_l2': all_road_l2_norm,
        }

        target = DIRECTION_PAIRS[direction]
        other = [i for i in range(8) if i not in target]

        # 使用 activation_ratio (road selectivity) 做统计检验
        target_vals = all_ratios_norm[:, target].mean(axis=1)
        other_vals = all_ratios_norm[:, other].mean(axis=1)

        t_stat, p_val = stats.ttest_rel(target_vals, other_vals)
        group_activation[direction]['t_stat_ratio'] = t_stat
        group_activation[direction]['p_value_ratio'] = p_val

        # 同时用 road_l2 做检验
        target_l2 = all_road_l2_norm[:, target].mean(axis=1)
        other_l2 = all_road_l2_norm[:, other].mean(axis=1)
        t_l2, p_l2 = stats.ttest_rel(target_l2, other_l2)
        group_activation[direction]['t_stat_l2'] = t_l2
        group_activation[direction]['p_value_l2'] = p_l2

        print(f"\n--- {DIRECTION_LABELS_EN[direction]} (n={n}) ---")
        print(f"  Road selectivity (ratio) per dir: {all_ratios_norm.mean(axis=0).round(3)}")
        print(f"  Raw activation ratio per dir:     {all_ratios.mean(axis=0).round(3)}")
        print(f"  [Road Selectivity] Target {target}: {target_vals.mean():.3f}, Others: {other_vals.mean():.3f}")
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        print(f"  Selectivity t-test: t={t_stat:.3f}, p={p_val:.6f} {sig}")
        print(f"  [Road L2] Target {target}: {target_l2.mean():.3f}, Others: {other_l2.mean():.3f}")
        sig2 = '***' if p_l2 < 0.001 else '**' if p_l2 < 0.01 else '*' if p_l2 < 0.05 else 'n.s.'
        print(f"  L2 t-test: t={t_l2:.3f}, p={p_l2:.6f} {sig2}")

    # ━━━ 可视化 ━━━
    generate_plots(group_activation, direction_groups, output_dir)
    save_data(results, group_activation, output_dir)


def generate_plots(group_activation, direction_groups, output_dir):
    plt.rcParams['font.size'] = 12

    directions_with_data = [d for d in ['horizontal', 'vertical', 'diagonal', 'antidiag']
                            if d in group_activation]

    if not directions_with_data:
        print("No sufficient data for visualization")
        return

    highlight_colors = {
        'horizontal': '#e74c3c',
        'vertical':   '#2ecc71',
        'diagonal':   '#3498db',
        'antidiag':   '#e67e22',
    }

    # ━━━ Figure 1: Road Selectivity (activation_ratio) per direction ━━━
    fig, axes = plt.subplots(1, len(directions_with_data),
                             figsize=(5 * len(directions_with_data), 5))
    if len(directions_with_data) == 1:
        axes = [axes]

    fig.suptitle('Road Selectivity (Road/Non-road Activation Ratio)\nper Scan Direction Expert',
                 fontsize=14, fontweight='bold')

    for ax_idx, direction in enumerate(directions_with_data):
        ax = axes[ax_idx]
        ga = group_activation[direction]
        mean_r = ga['mean_ratio_norm']
        std_r = ga['std_ratio_norm']
        n = ga['n_samples']
        target = DIRECTION_PAIRS[direction]
        p_val = ga['p_value_ratio']

        colors = ['#95a5a6'] * 8
        for t in target:
            colors[t] = highlight_colors[direction]

        x = np.arange(8)
        bars = ax.bar(x, mean_r, yerr=std_r, capsize=3, color=colors,
                      edgecolor='black', linewidth=0.5)

        for t in target:
            bars[t].set_edgecolor('#2c3e50')
            bars[t].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(DIRECTION_NAMES_EN, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Normalized Road Selectivity')
        ax.set_title(f'{DIRECTION_LABELS_EN[direction]} Roads (n={n})\n'
                     f'p={p_val:.1e}', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.annotate(sig, xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=14, fontweight='bold',
                    color='darkred' if p_val < 0.05 else 'gray')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'road_selectivity_bars.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'road_selectivity_bars.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [1/3] road_selectivity_bars.pdf/png")

    # ━━━ Figure 2: 双热力图 (road selectivity + raw activation ratio) ━━━
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))

    for ax, metric_key, title, cmap in [
        (ax1, 'mean_ratio_norm', 'Road Selectivity (Normalized)', 'YlOrRd'),
        (ax2, 'mean_ratio_raw', 'Raw Road/Non-road Activation Ratio', 'YlGnBu'),
    ]:
        matrix = []
        row_labels = []
        dir_order = []
        for d in ['horizontal', 'vertical', 'diagonal', 'antidiag']:
            if d in group_activation:
                matrix.append(group_activation[d][metric_key])
                row_labels.append(f"{DIRECTION_LABELS_EN[d]} (n={group_activation[d]['n_samples']})")
                dir_order.append(d)

        if not matrix:
            continue

        matrix = np.array(matrix)
        vmin = matrix.min() if 'raw' in metric_key else 0
        vmax = matrix.max() if 'raw' in metric_key else 1
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(DIRECTION_NAMES_EN, fontsize=9)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=10)

        threshold = (vmax + vmin) / 2
        for i in range(len(row_labels)):
            for j in range(8):
                color = 'white' if matrix[i, j] > threshold else 'black'
                fmt = '.2f' if 'raw' in metric_key else '.2f'
                ax.text(j, i, f'{matrix[i, j]:{fmt}}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

            d = dir_order[i] if i < len(dir_order) else None
            if d:
                for t in DIRECTION_PAIRS.get(d, []):
                    rect = plt.Rectangle((t - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor='blue', linewidth=3)
                    ax.add_patch(rect)

        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Scan Direction Expert')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'direction_selectivity_heatmap.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'direction_selectivity_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [2/3] direction_selectivity_heatmap.pdf/png")

    # ━━━ Figure 3: Target vs Other 对比 (road selectivity) ━━━
    fig, (ax_sel, ax_l2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_key, p_key, ylabel, title in [
        (ax_sel, 'raw_ratios_norm', 'p_value_ratio',
         'Normalized Road Selectivity',
         'Road Selectivity: Matching vs Non-matching'),
        (ax_l2, 'raw_road_l2', 'p_value_l2',
         'Normalized Feature Activation (L2)',
         'Feature Activation: Matching vs Non-matching'),
    ]:
        x_pos = np.arange(len(directions_with_data))
        width = 0.35

        t_means, o_means, t_stds, o_stds, labels, pvals = [], [], [], [], [], []

        for d in directions_with_data:
            ga = group_activation[d]
            raw = ga[metric_key]
            target = DIRECTION_PAIRS[d]
            other = [i for i in range(8) if i not in target]

            tv = raw[:, target].mean(axis=1)
            ov = raw[:, other].mean(axis=1)

            t_means.append(tv.mean())
            o_means.append(ov.mean())
            t_stds.append(tv.std())
            o_stds.append(ov.std())
            labels.append(DIRECTION_LABELS_EN[d])
            pvals.append(ga[p_key])

        bars1 = ax.bar(x_pos - width/2, t_means, width, yerr=t_stds,
                       capsize=4, label='Matching Direction',
                       color='#27ae60', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x_pos + width/2, o_means, width, yerr=o_stds,
                       capsize=4, label='Non-matching Direction',
                       color='#e74c3c', edgecolor='black', linewidth=0.5)

        for i, (b1, b2, p) in enumerate(zip(bars1, bars2, pvals)):
            y_max = max(b1.get_height() + t_stds[i], b2.get_height() + o_stds[i])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax.annotate(sig, xy=(x_pos[i], y_max + 0.03),
                        ha='center', fontsize=13, fontweight='bold',
                        color='darkred' if p < 0.05 else 'gray')

        ax.set_xlabel('Dominant Road Direction', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'target_vs_other_comparison.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'target_vs_other_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [3/3] target_vs_other_comparison.pdf/png")


def save_data(results, group_activation, output_dir):
    data = []
    for r in results:
        entry = {
            'img_id': r['img_id'],
            'dominant_direction': r['dominant_direction'],
            'confidence': float(r['confidence']),
            'gating_weights': r['gating_weights'],
        }
        for d in range(8):
            entry[f'dir{d}_road_l2'] = r['activation_metrics'][d]['road_l2']
            entry[f'dir{d}_ratio'] = r['activation_metrics'][d]['activation_ratio']
        data.append(entry)

    with open(os.path.join(output_dir, 'activation_results.json'), 'w') as f:
        json.dump(data, f, indent=2)

    summary = {}
    for d, ga in group_activation.items():
        summary[d] = {
            'n_samples': ga['n_samples'],
            'mean_ratio_norm': ga['mean_ratio_norm'].tolist(),
            'mean_ratio_raw': ga['mean_ratio_raw'].tolist(),
            'mean_road_l2': ga['mean_road_l2'].tolist(),
            't_stat_ratio': float(ga['t_stat_ratio']),
            'p_value_ratio': float(ga['p_value_ratio']),
            't_stat_l2': float(ga['t_stat_l2']),
            'p_value_l2': float(ga['p_value_l2']),
        }

    with open(os.path.join(output_dir, 'activation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Data saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='configs/vaihingen/ad_mamba.py')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('-o', '--output', type=str,
                        default='outputs/direction_activation')
    parser.add_argument('--confidence', type=float, default=0.35)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        max_samples=args.max_samples,
    )
