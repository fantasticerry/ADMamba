"""
v3: 覆盖 ImSurf + Building，原图降低对比度，专家区域用高饱和色凸显
"""

import os, sys, cv2, numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.environ.get(
    'ADMAMBA_FIG_OUTPUT', os.path.join(_REPO_ROOT, 'outputs', 'final_figures')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_ROOT = os.environ.get(
    'ADMAMBA_VAIHINGEN_TEST', os.path.join(_REPO_ROOT, 'data', 'vaihingen', 'test')
)

# 高饱和强对比色 (RGB)
DIR_COLORS = {
    'horizontal': np.array([255, 50, 50]),     # 亮红
    'vertical':   np.array([0, 220, 80]),      # 亮绿
    'diagonal':   np.array([30, 120, 255]),     # 亮蓝
    'antidiag':   np.array([255, 170, 0]),      # 亮橙
}
DIR_LABELS = {
    'horizontal': 'Horizontal Expert',
    'vertical':   'Vertical Expert',
    'diagonal':   'Diagonal(\\) Expert',
    'antidiag':   'Anti-diag(/) Expert',
}


def compute_local_direction_map(mask, target_classes=(0, 1), patch_radius=40):
    """对 ImSurf+Building 的边界做局部梯度方向分析"""
    H, W = mask.shape
    binary = np.zeros((H, W), dtype=np.float32)
    for cls in target_classes:
        binary += (mask == cls).astype(np.float32)
    binary = np.clip(binary, 0, 1)

    grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle_deg = np.degrees(np.arctan2(grad_y, grad_x)) % 180

    dir_energy = np.zeros((4, H, W), dtype=np.float64)
    dir_energy[1] = magnitude * ((angle_deg < 22.5) | (angle_deg >= 157.5))   # vertical
    dir_energy[3] = magnitude * ((angle_deg >= 22.5) & (angle_deg < 67.5))    # antidiag
    dir_energy[0] = magnitude * ((angle_deg >= 67.5) & (angle_deg < 112.5))   # horizontal
    dir_energy[2] = magnitude * ((angle_deg >= 112.5) & (angle_deg < 157.5))  # diagonal

    for i in range(4):
        dir_energy[i] = gaussian_filter(dir_energy[i], sigma=patch_radius)

    direction_map = dir_energy.argmax(axis=0)
    return direction_map, binary


def find_star_positions(region_mask, direction_map, n_stars=5):
    """在不同方向区域找代表性位置"""
    stars = []
    for dir_id in range(4):
        dir_region = (direction_map == dir_id) & (region_mask > 0)
        if dir_region.sum() < 300:
            continue
        labeled, n_features = scipy_label(dir_region)
        if n_features == 0:
            continue

        best_area = 0
        best_center = None
        for comp_id in range(1, min(n_features + 1, 50)):
            area = (labeled == comp_id).sum()
            if area > best_area:
                best_area = area
                ys, xs = np.where(labeled == comp_id)
                best_center = (int(xs.mean()), int(ys.mean()))

        if best_center and best_area > 800:
            dir_name = ['horizontal', 'vertical', 'diagonal', 'antidiag'][dir_id]
            stars.append({'pos': best_center, 'direction': dir_name, 'area': best_area})

    stars.sort(key=lambda x: x['area'], reverse=True)
    return stars[:n_stars]


def create_illustration(img_id):
    img_path = os.path.join(DATA_ROOT, 'images_1024', img_id + '.tif')
    mask_path = os.path.join(DATA_ROOT, 'masks_1024', img_id + '.png')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    H, W = img.shape[:2]

    # 仅 ImSurf(0) 区域
    target_mask = (mask == 0).astype(np.uint8)

    direction_map, binary_region = compute_local_direction_map(mask, target_classes=(0,))

    dir_names = ['horizontal', 'vertical', 'diagonal', 'antidiag']

    # ── 原图降低对比度 ──
    img_dark = img.astype(np.float64)
    img_dark = img_dark * 0.40 + 35
    img_dark = np.clip(img_dark, 0, 255)

    # ── 构建柔和光晕叠加层 ──
    # 对每个方向生成平滑的"光谱"热力图，而不是硬边界
    glow_layer = np.zeros((H, W, 3), dtype=np.float64)
    alpha_layer = np.zeros((H, W), dtype=np.float64)

    for dir_id in range(4):
        color = DIR_COLORS[dir_names[dir_id]].astype(np.float64) / 255.0
        region = ((direction_map == dir_id) & (target_mask > 0)).astype(np.float64)

        # 多级高斯模糊产生光晕扩散效果
        glow = gaussian_filter(region, sigma=25) * 0.5
        glow += gaussian_filter(region, sigma=12) * 0.3
        glow += gaussian_filter(region, sigma=5) * 0.2
        glow = np.clip(glow, 0, 1)

        for ch in range(3):
            glow_layer[:, :, ch] += glow * color[ch]
        alpha_layer += glow

    # 归一化防止多方向重叠过亮
    alpha_layer = np.clip(alpha_layer, 0, 1)
    glow_norm = glow_layer / (alpha_layer[..., None] + 1e-8) * alpha_layer[..., None]

    # 混合：暗化原图 + 光晕叠加
    blend_strength = 0.65
    overlay = img_dark / 255.0
    overlay = overlay * (1 - alpha_layer[..., None] * blend_strength) + \
              glow_norm * blend_strength
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # 找星号
    stars = find_star_positions(target_mask, direction_map, n_stars=5)

    arrow_dirs = {
        'horizontal': (1, 0),
        'vertical':   (0, 1),
        'diagonal':   (0.707, 0.707),
        'antidiag':   (0.707, -0.707),
    }

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(img)
    axes[0].axis('off')

    axes[1].imshow(overlay)

    shown_dirs = set()
    for star in stars:
        cx, cy = star['pos']
        d = star['direction']
        color_rgb = DIR_COLORS[d] / 255.0
        shown_dirs.add(d)

        # 大星号
        axes[1].plot(cx, cy, marker='*', markersize=26,
                    color='white', markeredgecolor='black',
                    markeredgewidth=1.5, zorder=10)

        # 方向箭头（加粗白色+黑色边框效果）
        dx, dy = arrow_dirs[d]
        arrow_len = 65
        # 黑色底层箭头（做边框）
        axes[1].annotate('',
            xy=(cx + dx * arrow_len, cy + dy * arrow_len),
            xytext=(cx - dx * arrow_len, cy - dy * arrow_len),
            arrowprops=dict(arrowstyle='->', color='black', lw=5, mutation_scale=20),
            zorder=8)
        # 白色顶层箭头
        axes[1].annotate('',
            xy=(cx + dx * arrow_len, cy + dy * arrow_len),
            xytext=(cx - dx * arrow_len, cy - dy * arrow_len),
            arrowprops=dict(arrowstyle='->', color='white', lw=3, mutation_scale=18),
            zorder=9)

        # 标签（高对比色底）
        label_text = d.capitalize() if d != 'antidiag' else 'Anti-diag'
        lx = cx + dx * arrow_len * 1.6
        ly = cy + dy * arrow_len * 1.6
        lx = np.clip(lx, 80, W - 80)
        ly = np.clip(ly, 40, H - 40)
        axes[1].text(lx, ly, label_text,
                    fontsize=28, fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.35',
                             facecolor=tuple(color_rgb), alpha=0.95,
                             edgecolor='white', linewidth=2),
                    zorder=11)

    # 图例
    legend_items = [
        mpatches.Patch(color=DIR_COLORS[d]/255., label=DIR_LABELS[d])
        for d in ['horizontal', 'vertical', 'diagonal', 'antidiag']
        if d in shown_dirs
    ]
    if legend_items:
        leg = axes[1].legend(handles=legend_items, loc='upper left', fontsize=22,
                      framealpha=0.95, edgecolor='black', fancybox=True,
                      shadow=True)
        leg.get_frame().set_linewidth(2)

    axes[1].axis('off')

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUTPUT_DIR, f'expert_v3_{img_id}')
    fig.savefig(path + '.pdf', dpi=200, bbox_inches='tight')
    fig.savefig(path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png (dirs: {shown_dirs})")


if __name__ == '__main__':
    candidates = [
        'top_mosaic_09cm_area4_0_3',      # Building 55%, 有大建筑
        'top_mosaic_09cm_area6_0_5',       # Building 54%
        'top_mosaic_09cm_area33_0_3',      # 平衡: 36%+40%
        'top_mosaic_09cm_area29_0_3',      # 45%+29%
        'top_mosaic_09cm_area29_0_5',      # 54%+19%
        'top_mosaic_09cm_area4_0_5',       # Building 49%
        'top_mosaic_09cm_area27_0_7',      # 平衡
        'top_mosaic_09cm_area27_0_3',      # Building 30%
        'top_mosaic_09cm_area38_0_6',      # Building 32%
        'top_mosaic_09cm_area33_0_5',      # 十字路口
        'top_mosaic_09cm_area22_0_5',      # 原来效果好的
        'top_mosaic_09cm_area35_0_4',
        'top_mosaic_09cm_area8_0_5',
        'top_mosaic_09cm_area38_0_5',
    ]

    for img_id in candidates:
        create_illustration(img_id)
    print(f"\nAll saved to {OUTPUT_DIR}/")
