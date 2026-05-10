"""
分析不同扫描方向的余弦相似度
使用Vaihingen数据集的图片进行测试
针对 AD-Mamba 模型的多方向扫描
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import sys

# Make repo root importable when running this script directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import itertools
from admamba.models.ad_mamba import (
    ADMamba,
    diagonal_gather,
    antidiagonal_gather,
    diagonal_scatter,
    antidiagonal_scatter,
)

def load_images(image_dir, num_images=5, img_size=1024):
    """加载指定数量的图片"""
    import random
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    # 随机打乱并选择前 num_images 张
    random.shuffle(image_files)
    image_files = image_files[:num_images]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
        print(f"已加载: {img_file}")
    
    return torch.stack(images)

def create_scan_sequences(x):
    """
    创建8个扫描方向的序列
    Args:
        x: 输入特征 (B, C, H, W)
    Returns:
        8个方向的扫描序列，每个形状为 (B, L, C)
    """
    B, C, H, W = x.shape
    
    # 横向：从左到右
    seq_horizontal = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
    
    # 竖向：从上到下
    seq_vertical = x.transpose(2, 3).flatten(2).transpose(1, 2)  # (B, H*W, C)
    
    # 横向反向：从右到左
    seq_horizontal_rev = seq_horizontal.flip(dims=[1])
    
    # 竖向反向：从下到上
    seq_vertical_rev = seq_vertical.flip(dims=[1])
    
    # 斜向扫描
    seq_diagonal = diagonal_gather(x).transpose(1, 2)  # (B, H*W, C)
    
    # 反斜向扫描
    seq_antidiagonal = antidiagonal_gather(x).transpose(1, 2)  # (B, H*W, C)
    
    # 斜向反向
    seq_diagonal_rev = seq_diagonal.flip(dims=[1])
    
    # 反斜向反向
    seq_antidiagonal_rev = seq_antidiagonal.flip(dims=[1])
    
    return [seq_horizontal, seq_vertical, seq_horizontal_rev, seq_vertical_rev,
            seq_diagonal, seq_antidiagonal, seq_diagonal_rev, seq_antidiagonal_rev]

def restore_to_2d(seq, H, W, direction):
    """
    将扫描序列还原到2D空间
    Args:
        seq: 扫描序列 (B, L, C)
        H, W: 原始高宽
        direction: 扫描方向索引
    Returns:
        还原后的2D特征 (B, C, H, W)
    """
    B, L, C = seq.shape
    
    if direction == 0:  # 横向
        return seq.transpose(1, 2).view(B, C, H, W)
    elif direction == 1:  # 竖向
        return seq.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
    elif direction == 2:  # 横向反向
        return seq.flip(dims=[1]).transpose(1, 2).view(B, C, H, W)
    elif direction == 3:  # 竖向反向
        return seq.flip(dims=[1]).transpose(1, 2).view(B, C, W, H).transpose(2, 3)
    elif direction == 4:  # 斜向
        return diagonal_scatter(seq.transpose(1, 2), (B, C, H, W))
    elif direction == 5:  # 反斜向
        return antidiagonal_scatter(seq.transpose(1, 2), (B, C, H, W))
    elif direction == 6:  # 斜向反向
        return diagonal_scatter(seq.flip(dims=[1]).transpose(1, 2), (B, C, H, W))
    elif direction == 7:  # 反斜向反向
        return antidiagonal_scatter(seq.flip(dims=[1]).transpose(1, 2), (B, C, H, W))

def analyze_directional_cosine(mamba_layer, x, measure_stage='pre_mamba'):
    """
    计算不同扫描方向组合的Mamba输出余弦相似度。
    对原始模型，使用同一个Mamba处理4个不同方向的扫描序列。
    
    Args:
        mamba_layer: MambaLayer实例
        x: 输入特征 (B, C, H, W)
    
    Returns:
        dict: {label: (k,k) 余弦相似度矩阵 (CPU tensor)}
    """
    res = x
    B, C, H, W = res.shape
    
    # PPM处理（新模型先放pool再放res）
    ppm_out = []
    for p in mamba_layer.pool_layers:
        pool_out = p(x)
        pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
        ppm_out.append(pool_out)
    ppm_out.append(res)
    x = torch.cat(ppm_out, dim=1)
    _, chs, _, _ = x.shape
    
    # 创建8个方向的扫描序列
    scan_seqs = create_scan_sequences(x)
    
    # 计算预/后 Mamba 的方向特征
    dir_vectors = []
    for i, seq in enumerate(scan_seqs):
        seq_gated = mamba_layer.fd_gate(seq)
        if measure_stage == 'pre_mamba':
            vec = seq_gated
        else:
            y_i = mamba_layer.mambas[i](seq_gated)
            vec = y_i
        dir_vectors.append(vec)  # (B, L, C)
    
    direction_labels = {
        "2_direction": [0, 1],
        "4_direction": [0, 1, 2, 3],
        "8_direction": [0, 1, 2, 3, 4, 5, 6, 7],
    }
    
    def build_cosine_matrix(indices):
        size = len(indices)
        matrix = x.new_zeros(size, size)
        for i in range(size):
            for j in range(i, size):
                vec_i = dir_vectors[indices[i]]  # (B, H*W, C)
                vec_j = dir_vectors[indices[j]]  # (B, H*W, C)
                # 在特征维度C上计算每个位置的余弦相似度，然后平均
                cos_vals = F.cosine_similarity(vec_i, vec_j, dim=-1)  # (B, H*W)
                matrix[i, j] = cos_vals.mean()
                if i != j:
                    matrix[j, i] = cos_vals.mean()
        return matrix.cpu()
    
    cosine_results = {}
    for label, idxs in direction_labels.items():
        cosine_results[label] = build_cosine_matrix(idxs)
    
    # 8选4：在所有组合中选择平均相似度最小的4个方向
    full_matrix = cosine_results["8_direction"]
    best_matrix = None
    best_score = None
    best_combo = None
    for combo in itertools.combinations(range(8), 4):
        combo = list(combo)
        sub_matrix = full_matrix[combo][:, combo]
        mask = ~torch.eye(4, dtype=torch.bool)
        avg_sim = sub_matrix[mask].mean().item()
        if best_score is None or avg_sim < best_score:
            best_score = avg_sim
            best_matrix = sub_matrix.clone()
            best_combo = combo
    cosine_results["8_select_4"] = best_matrix
    cosine_results["_8_select_4_combo"] = best_combo
    
    return cosine_results

def analyze_cosine_similarity(model, images, device='cuda', measure_stage='pre_mamba'):
    """分析余弦相似度"""
    model = model.to(device)
    model.eval()
    images = images.to(device)
    
    all_results = []
    
    with torch.no_grad():
        for i in range(images.shape[0]):
            x = images[i:i+1]
            # 通过backbone获取特征
            x0, x3 = model.backbone(x)
            x3 = x3.permute(0, 3, 1, 2)  # Swin输出 (B, H, W, C) -> (B, C, H, W)
            
            # 调用余弦相似度分析
            result = analyze_directional_cosine(model.decoder.b3.mamba, x3, measure_stage=measure_stage)
            all_results.append(result)
            print(f"已处理图片 {i+1}/{images.shape[0]}")
    
    # 计算平均结果
    avg_results = {}
    tensor_keys = [k for k in all_results[0].keys() if isinstance(all_results[0][k], torch.Tensor)]
    for key in tensor_keys:
        matrices = [r[key] for r in all_results]
        avg_results[key] = torch.stack(matrices).mean(0)
    
    # 记录组合信息
    if "_8_select_4_combo" in all_results[0]:
        combos = [set(r["_8_select_4_combo"]) for r in all_results]
        combo_counts = {}
        for combo in combos:
            combo_tuple = tuple(sorted(combo))
            combo_counts[combo_tuple] = combo_counts.get(combo_tuple, 0) + 1
        best_combo = max(combo_counts.items(), key=lambda x: x[1])[0]
        avg_results["_8_select_4_combo"] = best_combo
    
    return avg_results

def print_results(results):
    """打印结果"""
    print("\n" + "="*60)
    print("余弦相似度分析结果（AD-Mamba 8 向 - 8 个独立 Mamba）")
    print("="*60)
    
    selected_combo = results.get("_8_select_4_combo", None)
    
    for label, matrix in results.items():
        if isinstance(label, str) and label.startswith("_"):
            continue
        if not torch.is_tensor(matrix):
            continue
        print(f"\n【{label}】")
        print("-" * 40)
        
        # 获取方向数量
        n = matrix.shape[0]
        
        # 打印矩阵
        if label == "2_direction":
            names = ["横向", "竖向"]
        elif label == "4_direction":
            names = ["横向", "竖向", "横向反向", "竖向反向"]
        else:
            names = [f"方向{i}" for i in range(n)]
        
        # 打印表头
        header = "        " + "  ".join([f"{name[:4]:>6}" for name in names])
        print(header)
        
        # 打印每行
        for i in range(n):
            row = f"{names[i][:6]:>6}  "
            row += "  ".join([f"{matrix[i,j].item():>6.3f}" for j in range(n)])
            print(row)
        
        # 计算平均相似度（排除对角线）
        mask = ~torch.eye(n, dtype=torch.bool)
        avg_sim = matrix[mask].mean().item()
        print(f"\n平均相似度（非对角线）: {avg_sim:.4f}")
    
    if selected_combo is not None:
        combo_str = ", ".join([f"方向{i}" for i in selected_combo])
        print(f"\n8_select_4 选择的方向组合: [{combo_str}]")

def main():
    # 配置
    image_dir = os.environ.get(
        "ADMAMBA_VAIHINGEN_IMAGES",
        os.path.join(_REPO_ROOT, "data", "vaihingen", "train_images"),
    )
    num_images = 50  # 使用5张图片
    img_size = 512  # 与训练时一致
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"图片目录: {image_dir}")
    print(f"图片数量: {num_images}")
    
    # 加载图片
    print("\n正在加载图片...")
    images = load_images(image_dir, num_images, img_size)
    print(f"图片形状: {images.shape}")
    
    # 初始化模型
    measure_stage = 'pre_mamba'  # 'pre_mamba' or 'post_mamba'
    print(f"\n测量阶段: {measure_stage}")
    print("\n正在初始化模型...")
    model = ADMamba(
        backbone_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        pretrained=False,
        num_classes=6,
        decoder_channels=128,
        last_feat_size=16,
        img_size=img_size,
        enable_moe=True,
        moe_top_k=4
    )
    
    # 加载训练好的权重
    ckpt_path = os.environ.get(
        "ADMAMBA_CKPT",
        os.path.join(_REPO_ROOT, "model_weights", "vaihingen", "ad_mamba.ckpt"),
    )
    print(f"正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # PyTorch Lightning checkpoint格式
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    print("模型初始化完成")
    
    # 分析余弦相似度
    print("\n正在分析余弦相似度...")
    results = analyze_cosine_similarity(model, images, device, measure_stage=measure_stage)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    main()
