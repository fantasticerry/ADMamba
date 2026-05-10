"""
模型性能测试脚本
测试不同模型的FPS、GFLOPS和参数大小
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import *

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")
    print("Will use simplified FLOPs calculation.")

# 尝试导入fvcore作为备选
try:
    from fvcore.nn import FlopCountMode, flop_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_size(size_bytes):
    """格式化参数大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def calculate_flops_thop(model, input_size=(1, 3, 512, 512)):
    """使用thop计算FLOPs"""
    # 运行时重新检查thop是否可用
    try:
        from thop import profile
    except ImportError:
        return None, None
    
    model.eval()
    dummy_input = torch.randn(input_size).cuda()
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_gflops = flops / 1e9
        return flops_gflops, params
    except Exception as e:
        print(f"   Error calculating FLOPs with thop: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_flops_simple(model, input_size=(1, 3, 512, 512)):
    """简化的FLOPs计算（仅估算）"""
    # 这是一个非常简化的估算，实际应该使用thop或fvcore
    # 这里只提供一个占位符
    return None


def detect_model_input_size(model):
    """自动检测模型期望的输入尺寸"""
    img_size = None
    
    # 方法1: 从backbone的patch_embed获取（timm模型，最可靠的方法）
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'patch_embed'):
        patch_embed = model.backbone.patch_embed
        if hasattr(patch_embed, 'img_size'):
            img_size_attr = patch_embed.img_size
            if isinstance(img_size_attr, (list, tuple)) and len(img_size_attr) > 0:
                img_size = int(img_size_attr[0])
            elif isinstance(img_size_attr, (int, float)):
                img_size = int(img_size_attr)
            if img_size is not None:
                return img_size
    
    # 方法2: 从backbone的img_size属性获取（timm模型）
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'img_size'):
        img_size_attr = model.backbone.img_size
        if isinstance(img_size_attr, (list, tuple)) and len(img_size_attr) > 0:
            img_size = int(img_size_attr[0])
        elif isinstance(img_size_attr, (int, float)):
            img_size = int(img_size_attr)
        if img_size is not None:
            return img_size
    
    # 方法3: 递归查找所有子模块中的img_size
    if hasattr(model, 'backbone'):
        for name, module in model.backbone.named_modules():
            if hasattr(module, 'img_size'):
                img_size_attr = module.img_size
                if isinstance(img_size_attr, (list, tuple)) and len(img_size_attr) > 0:
                    img_size = int(img_size_attr[0])
                elif isinstance(img_size_attr, (int, float)):
                    img_size = int(img_size_attr)
                if img_size is not None:
                    return img_size
    
    return img_size


def measure_fps(model, input_size=(1, 3, 512, 512), num_iterations=100, warmup=10):
    """测量模型推理速度（FPS）"""
    model.eval()
    model.cuda()
    
    # 创建虚拟输入
    dummy_input = torch.randn(input_size).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 同步CUDA
    torch.cuda.synchronize()
    
    # 测试推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    return fps, avg_time * 1000  # 返回FPS和平均时间(ms)


def measure_fps_with_dataloader(model, dataloader, num_batches=50):
    """使用真实数据测量FPS"""
    model.eval()
    model.cuda()
    
    times = []
    batch_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            
            img = batch['img'].cuda()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(img)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            batch_count += 1
    
    if len(times) == 0:
        return None, None
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return fps, avg_time * 1000  # 返回FPS和平均时间(ms)


def benchmark_model(config_path, input_size=(1, 3, 512, 512), use_real_data=False):
    """对模型进行性能测试"""
    print(f"\n{'='*60}")
    print(f"Benchmarking model from: {config_path}")
    print(f"{'='*60}\n")
    
    # 加载配置
    config = py2cfg(config_path)
    
    # 创建模型（不加载权重，只测试结构）
    try:
        # 优先使用config中的net
        if hasattr(config, 'net') and config.net is not None:
            model = config.net
        else:
            # 尝试从checkpoint加载（但只用于获取模型结构，不加载权重）
            if hasattr(config, 'test_weights_name') and hasattr(config, 'weights_path'):
                checkpoint_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
                if os.path.exists(checkpoint_path):
                    try:
                        # 尝试加载checkpoint获取模型结构
                        pl_model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config)
                        model = pl_model.net  # 获取实际的网络
                    except:
                        # 如果加载失败，使用config中的net
                        model = config.net
                else:
                    # 如果没有checkpoint，直接创建模型
                    model = config.net
            else:
                # 如果config中没有net，尝试创建
                raise ValueError("Config does not have 'net' attribute")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    model.eval()
    model.cuda()
    
    # 自动检测模型期望的输入尺寸
    detected_img_size = detect_model_input_size(model)
    if detected_img_size is not None:
        # 使用检测到的尺寸，但保持batch和channel不变
        actual_input_size = (input_size[0], input_size[1], detected_img_size, detected_img_size)
        if actual_input_size != input_size:
            print(f"   Detected model input size: {detected_img_size}x{detected_img_size}")
            print(f"   Using input size: {actual_input_size} (overriding user input: {input_size})\n")
            input_size = actual_input_size
    else:
        print(f"   Using user-specified input size: {input_size}\n")
    
    # 1. 计算参数数量
    print("1. Calculating Parameters...")
    total_params, trainable_params = count_parameters(model)
    param_size_mb = total_params * 4 / (1024 ** 2)  # 假设float32，4字节/参数
    
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {format_size(total_params * 4)}")
    print(f"   Model Size (MB): {param_size_mb:.2f} MB\n")
    
    # 2. 计算FLOPs
    print("2. Calculating FLOPs...")
    flops_gflops = None
    # 运行时重新检查thop是否可用
    try:
        import thop
        from thop import profile
        thop_available = True
        print(f"   thop version: {getattr(thop, '__version__', 'unknown')}")
    except ImportError as e:
        thop_available = False
        print(f"   thop import failed: {e}")
    
    if thop_available:
        print("   Using thop to calculate FLOPs...")
        flops_gflops, params_thop = calculate_flops_thop(model, input_size)
        if flops_gflops is not None:
            print(f"   FLOPs: {flops_gflops:.2f} GFLOPs")
            print(f"   Parameters (thop): {params_thop:,}\n")
        else:
            print("   Failed to calculate FLOPs with thop\n")
    else:
        print("   thop not installed, skipping FLOPs calculation")
        print("   Install with: pip install thop\n")
    
    # 3. 测量FPS
    print("3. Measuring FPS...")
    if use_real_data and hasattr(config, 'test_dataset'):
        print("   Using real dataset...")
        test_loader = DataLoader(
            config.test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        fps, avg_time_ms = measure_fps_with_dataloader(model, test_loader, num_batches=50)
        if fps is not None:
            print(f"   FPS: {fps:.2f}")
            print(f"   Average Inference Time: {avg_time_ms:.2f} ms\n")
        else:
            print("   Failed to measure FPS with real data\n")
    else:
        print(f"   Using dummy input with size: {input_size}...")
        fps, avg_time_ms = measure_fps(model, input_size=input_size, num_iterations=100, warmup=10)
        print(f"   FPS: {fps:.2f}")
        print(f"   Average Inference Time: {avg_time_ms:.2f} ms\n")
    
    # 汇总结果
    results = {
        'config_path': str(config_path),
        'total_params': total_params,
        'params_m': total_params / 1e6,
        'model_size_mb': param_size_mb,
        'flops_gflops': flops_gflops,
        'fps': fps,
        'avg_time_ms': avg_time_ms
    }
    
    print(f"{'='*60}")
    print("Summary:")
    print(f"  Parameters: {total_params/1e6:.2f}M")
    print(f"  Model Size: {param_size_mb:.2f} MB")
    if flops_gflops is not None:
        print(f"  FLOPs: {flops_gflops:.2f} GFLOPs")
    if fps is not None:
        print(f"  FPS: {fps:.2f}")
        print(f"  Inference Time: {avg_time_ms:.2f} ms")
    print(f"{'='*60}\n")
    
    return results


def benchmark_multiple_models(config_dir, input_size=(1, 3, 512, 512), use_real_data=False):
    """测试多个模型"""
    config_dir = Path(config_dir)
    config_files = list(config_dir.glob("*.py"))
    
    if len(config_files) == 0:
        print(f"No config files found in {config_dir}")
        return
    
    all_results = []
    
    for config_file in config_files:
        try:
            results = benchmark_model(config_file, input_size=input_size, use_real_data=use_real_data)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking {config_file}: {e}\n")
            continue
    
    # 打印对比表格
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(f"{'Model':<30} {'Params(M)':<12} {'Size(MB)':<12} {'GFLOPs':<12} {'FPS':<12} {'Time(ms)':<12}")
        print("-"*80)
        
        for r in all_results:
            model_name = Path(r['config_path']).stem
            params = f"{r['params_m']:.2f}"
            size = f"{r['model_size_mb']:.2f}"
            flops = f"{r['flops_gflops']:.2f}" if r['flops_gflops'] is not None else "N/A"
            fps = f"{r['fps']:.2f}" if r['fps'] is not None else "N/A"
            time_ms = f"{r['avg_time_ms']:.2f}" if r['avg_time_ms'] is not None else "N/A"
            
            print(f"{model_name:<30} {params:<12} {size:<12} {flops:<12} {fps:<12} {time_ms:<12}")
        
        print("="*80 + "\n")


def get_args():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to config file")
    parser.add_argument("-d", "--config_dir", type=Path, help="Path to config directory (benchmark all models)")
    parser.add_argument("--input_size", type=int, nargs=4, default=[1, 3, 512, 512],
                        help="Input size as [batch, channels, height, width]")
    parser.add_argument("--use_real_data", action='store_true',
                        help="Use real dataset for FPS measurement")
    return parser.parse_args()


def main():
    args = get_args()
    
    input_size = tuple(args.input_size)
    
    if args.config_dir:
        # 测试目录中的所有模型
        benchmark_multiple_models(args.config_dir, input_size=input_size, use_real_data=args.use_real_data)
    elif args.config_path:
        # 测试单个模型
        benchmark_model(args.config_path, input_size=input_size, use_real_data=args.use_real_data)
    else:
        print("Please provide either -c/--config_path or -d/--config_dir")
        print("Example:")
        print("  python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py")
        print("  python scripts/benchmark_model.py -d configs/vaihingen/")
        print("  python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py --input_size 1 3 1024 1024")


if __name__ == "__main__":
    main()

