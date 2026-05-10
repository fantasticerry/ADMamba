"""
消融实验 Benchmark 脚本
测试不同扫描策略和门控模块的 FLOPs、FPS 和参数量
统一使用 Swin-B backbone, 512x512 输入
包含自定义 Mamba FLOPs 计算器，精确统计选择性扫描算子
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gc
import importlib.util
import time

import mamba_ssm
import torch

INPUT_SIZE = (1, 3, 512, 512)
IMG_SIZE = 512
LAST_FEAT_SIZE = IMG_SIZE // 32  # = 16
NUM_CLASSES = 6
WARMUP = 10
ITERS = 100


def count_mamba(module, input, output):
    """
    Custom FLOPs counter for mamba_ssm.Mamba.
    Accounts for all internal ops including the selective scan recurrence
    that fused CUDA kernels hide from standard profilers.

    Selective scan per-step cost:
      exp(delta*A)       -> 2EN   (mul + exp)
      delta * B_t        -> EN
      A_bar*h + B_bar*x  -> 3EN   (2 mul + 1 add)
      C_t * h + sum      -> 2EN   (mul + reduce)
      D * x              -> 2E
      Total              -> L * (8EN + 2E)
    """
    x = input[0]
    B, L, D = x.shape
    E = module.d_inner
    N = module.d_state
    K = module.d_conv
    R = module.dt_rank

    flops = 0
    flops += B * L * D * 2 * E            # in_proj: Linear(D -> 2E)
    flops += B * L * E * K                 # conv1d: depthwise, K taps
    flops += B * L * E * 4                 # SiLU(x): x * sigmoid(x)
    flops += B * L * E * (R + 2 * N)      # x_proj: Linear(E -> R+2N)
    flops += B * L * R * E                 # dt_proj: Linear(R -> E)
    flops += B * L * (8 * E * N + 2 * E)  # selective scan recurrence
    flops += B * L * E * 5                 # gate: SiLU(z) + z*y
    flops += B * L * E * D                 # out_proj: Linear(E -> D)

    module.total_ops += torch.DoubleTensor([flops])


MAMBA_CUSTOM_OPS = {mamba_ssm.Mamba: count_mamba}


def load_module_from_file(filepath, module_name, patch_mamba2=False):
    if patch_mamba2:
        if not hasattr(mamba_ssm, 'Mamba2'):
            mamba_ssm.Mamba2 = mamba_ssm.Mamba
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark_model(model, name):
    model.eval().cuda()
    total_params = sum(p.numel() for p in model.parameters())

    flops_val = 0.0
    try:
        from thop import profile
        dummy = torch.randn(*INPUT_SIZE).cuda()
        flops_val, _ = profile(model, inputs=(dummy,), verbose=False,
                               custom_ops=MAMBA_CUSTOM_OPS)
    except Exception as e:
        print(f"    [WARN] FLOPs error: {e}")

    dummy = torch.randn(*INPUT_SIZE).cuda()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(ITERS):
            model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = ITERS / elapsed
    time_ms = elapsed / ITERS * 1000

    result = {
        'name': name,
        'params_m': total_params / 1e6,
        'gflops': flops_val / 1e9,
        'fps': fps,
        'time_ms': time_ms,
    }
    print(f"  [{name}]  Params={result['params_m']:.2f}M  GFLOPs={result['gflops']:.2f}  "
          f"FPS={result['fps']:.2f}  Time={result['time_ms']:.2f}ms")

    del model, dummy
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    ablations_dir = os.path.join(_REPO_ROOT, 'ablations')
    main_model_path = os.path.join(_REPO_ROOT, 'admamba', 'models', 'ad_mamba.py')
    results = []

    # ===== 1. Dynamic Scan / 1-row Scan (ablations/ad_mamba_baseline.py) =====
    print("\n[1/6] Loading ad_mamba_baseline.py  (Dynamic Scan / 1-row Scan)")
    mod_orig = load_module_from_file(os.path.join(ablations_dir, 'ad_mamba_baseline.py'), 'pm_orig')
    model = mod_orig.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True
    )
    r = benchmark_model(model, 'Dynamic Scan / 1-row Scan')
    results.append({**r, 'name': 'Dynamic Scan'})
    results.append({**r, 'name': '1-row Scan'})
    del mod_orig

    # ===== 2. 4-row Scan (ablations/ad_mamba_4dir.py, Mamba2) =====
    print("\n[2/6] Loading ad_mamba_4dir.py  (4-row Scan)")
    mod_4dir = load_module_from_file(os.path.join(ablations_dir, 'ad_mamba_4dir.py'), 'pm_4dir', patch_mamba2=True)
    model = mod_4dir.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True
    )
    results.append(benchmark_model(model, '4-row Scan'))
    del mod_4dir

    # ===== 3. 8-row Scan (ablations/ad_mamba_8dir.py, Mamba2) =====
    print("\n[3/6] Loading ad_mamba_8dir.py  (8-row Scan)")
    mod_8dir = load_module_from_file(os.path.join(ablations_dir, 'ad_mamba_8dir.py'), 'pm_8dir', patch_mamba2=True)
    model = mod_8dir.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True
    )
    results.append(benchmark_model(model, '8-row Scan'))
    del mod_8dir

    # ===== 4. 8-select-4 / + 1st-order FDG (RGB) (ablations/ad_mamba_fdg_step.py) =====
    print("\n[4/6] Loading ad_mamba_fdg_step.py  (8-select-4 / +FDG RGB)")
    mod_5 = load_module_from_file(os.path.join(ablations_dir, 'ad_mamba_fdg_step.py'), 'pm_5')
    model = mod_5.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True,
        enable_moe=True, moe_top_k=4
    )
    r = benchmark_model(model, '8-select-4 / +1st-order FDG(RGB)')
    results.append({**r, 'name': '8-select-4'})
    results.append({**r, 'name': '+ 1st-order FDG (RGB)'})
    del mod_5

    # ===== 5. + Elevation-Guided Gate (RGB+DSM) (current admamba/models/ad_mamba.py, ElevationGuidedGate) =====
    print("\n[5/6] Loading ad_mamba.py  (+Elevation-Guided Gate RGB+DSM)")
    mod_cur = load_module_from_file(main_model_path, 'pm_cur_elev')
    model = mod_cur.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True,
        enable_moe=True, moe_top_k=4,
        use_elevation_gate=True,
        use_fractional_gate=False,
        use_geo_msaa=False,
    )
    results.append(benchmark_model(model, '+ Elevation-Guided Gate (RGB+DSM)'))
    del mod_cur

    # ===== 6. + Fractional Calculus Gate / FCG (RGB+DSM) (current admamba/models/ad_mamba.py, FractionalCalculusGate) =====
    print("\n[6/6] Loading ad_mamba.py  (+FCG RGB+DSM)")
    mod_cur2 = load_module_from_file(main_model_path, 'pm_cur_frac')
    model = mod_cur2.ADMamba(
        num_classes=NUM_CLASSES, last_feat_size=LAST_FEAT_SIZE, img_size=IMG_SIZE, pretrained=True,
        enable_moe=True, moe_top_k=4,
        use_elevation_gate=False,
        use_fractional_gate=True,
        fractional_alpha=0.5,
        fractional_memory_length=16,
        use_geo_msaa=False,
    )
    results.append(benchmark_model(model, '+ Fractional Calculus Gate (RGB+DSM)'))
    del mod_cur2

    # ===== Print summary =====
    print("\n" + "=" * 85)
    print("ABLATION BENCHMARK SUMMARY  (Swin-B backbone, 512x512 input)")
    print("=" * 85)
    print(f"  {'Methods':<42} {'Params(M)':<12} {'GFLOPs':<12} {'FPS':<10} {'Time(ms)':<10}")
    print("  " + "-" * 82)
    for r in results:
        print(f"  {r['name']:<42} {r['params_m']:<12.2f} {r['gflops']:<12.2f} {r['fps']:<10.2f} {r['time_ms']:<10.2f}")
    print("=" * 85)

    # ===== LaTeX output =====
    print("\n% --- LaTeX table columns: Methods & Params(M) & GFLOPs & FPS ---")
    print("% Paste into your ablation table:")
    for r in results:
        n = r['name'].replace('_', r'\_').replace('+', r'+')
        print(f"        \\quad {n} & {r['params_m']:.2f} & {r['gflops']:.2f} & {r['fps']:.1f} \\\\")


if __name__ == "__main__":
    main()
