---
name: admamba env reproduce
overview: 在 /root/autodl-tmp 下搭建可复现 AD-Mamba 的 conda 环境，做"前向 + 1~2 步真实训练"烟囱测试，全部跑通后用本机已验证的命令覆写 README.md 的 Installation 一节。
todos:
  - id: create_env
    content: 在 /root/autodl-tmp/envs/admamba 创建 python=3.10 conda env 并装 pip/wheel/ninja 基础工具
    status: completed
  - id: install_torch
    content: 装 torch==2.3.1+cu121 / torchvision==0.18.1+cu121，验证 torch.cuda.is_available()
    status: completed
  - id: install_pkg
    content: 在仓库根跑 pip install -e . 装项目本体 + Python 依赖
    status: completed
  - id: install_cuda_kernels
    content: 设 CUDA_HOME=/usr/local/cuda-12.1 + MAX_JOBS=4，--no-build-isolation 安装 causal-conv1d==1.4.0 与 mamba-ssm==2.2.2，并 import 验证
    status: completed
  - id: smoke_forward
    content: 跑 scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py 验证前向 + Mamba 内核
    status: completed
  - id: smoke_train
    content: 以 max_steps=2 跑一次 Vaihingen 真数据训练烟囱，看到 SMOKE_OK
    status: completed
  - id: update_readme
    content: 全部跑通后，用 StrReplace 覆写 README.md Installation 一节为本机已验证步骤（不动其他 md）
    status: completed
isProject: false
---

## 目标
- 搭建一个干净、可复现的 AD-Mamba 训练/推理环境（含 `mamba-ssm` CUDA 内核）。
- 用 Vaihingen 已切好的 1024 patch 做一次前向 + 1~2 个训练 step 的烟囱测试。
- 跑通后用经过本机验证的命令更新 [README.md](/root/autodl-tmp/ADMamba/README.md) 的 Installation 一节（按用户规则不新建 md）。

## 已知前提（已侦察确认）
- GPU：RTX 4090 24GB；系统装有 CUDA Toolkit 12.1 + gcc 11.4 + miniconda 24.4。
- 数据盘 `/root/autodl-tmp` 剩 46G，系统盘 30G — env 装数据盘。
- Vaihingen 已切 1024 patch（train 747 / test 113，含 dsm），Potsdam 缺失（不影响烟囱测试）。
- 项目要求 `python>=3.10`，依赖 `pytorch-lightning==2.3.0`、`timm==0.9.16`，CUDA 内核 `causal-conv1d>=1.4.0`、`mamba-ssm>=1.2.0`（源码编译）。

## 关键技术决策
- **PyTorch 版本**：`torch==2.3.1+cu121, torchvision==0.18.1+cu121`。
  - 理由：与 `pytorch-lightning==2.3.0` 同代 + 与系统 `nvcc 12.1` 一致 + `mamba-ssm 1.2.x` 在 PT 2.3/cu121 组合下编译成功率高，比 README 的 cu118 示例更贴合本机。
- **Conda env 路径**：`/root/autodl-tmp/envs/admamba`（数据盘），通过 `--prefix` 指定，避免占系统盘。
- **pip 缓存**：`PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache`、`TMPDIR=/root/autodl-tmp/tmp`，把 mamba-ssm 编译期临时文件挪到数据盘。
- **mamba-ssm 编译加速**：`MAMBA_FORCE_BUILD=TRUE`、`MAX_JOBS=4`（控制内存峰值，4090 24G + 系统内存常见够用），并设 `CUDA_HOME=/usr/local/cuda-12.1`。
- **缺失资源占位**：避免触碰已存在 Vaihingen 数据；Potsdam 烟囱不跑。
- 烟囱测试不动 `configs/`，临时通过 CLI/脚本控制 step 数（见下方步骤）。

## 步骤拆解

### 1. 建 env + 装基础工具
```bash
conda create -y --prefix /root/autodl-tmp/envs/admamba python=3.10
conda activate /root/autodl-tmp/envs/admamba
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
export TMPDIR=/root/autodl-tmp/tmp && mkdir -p $TMPDIR
pip install -U pip wheel setuptools packaging ninja
```

### 2. 装 PyTorch（cu121）
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### 3. 安装项目本体（editable）
```bash
cd /root/autodl-tmp/ADMamba
pip install -e .
```
- 注意：[requirements.txt](/root/autodl-tmp/ADMamba/requirements.txt) 把 `causal-conv1d`、`mamba-ssm` 也写进去了，但 [pyproject.toml](/root/autodl-tmp/ADMamba/pyproject.toml) 把它们隔离到 `[cuda]` extra；走 `pip install -e .` 不会立刻拉这两个，避免 PT 还没就绪时编译失败。

### 4. 装 CUDA 内核（关键步骤，最容易踩坑）
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4
pip install --no-build-isolation causal-conv1d==1.4.0
pip install --no-build-isolation mamba-ssm==2.2.2
```
- 用 `--no-build-isolation` 让构建复用已安装的 torch，否则 pip 会另外拉一个不匹配的 torch 进 isolated env。
- 选钉版本：`causal-conv1d 1.4.0` + `mamba-ssm 2.2.2` 与 `torch 2.3.1+cu121` 是已知可编译组合（mamba-ssm 1.2.x 在新版 setup 上常出问题）。
- 验证：`python -c "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba ok')"`

### 5. 前向烟囱：[scripts/benchmark_model.py](/root/autodl-tmp/ADMamba/scripts/benchmark_model.py)
```bash
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py --input_size 1 3 512 512
```
- 期望输出 `Parameters / GFLOPs / FPS`。如果这一步过，证明：env、PyTorch、mamba 内核、`admamba.models.ad_mamba.ADMamba` 全部对。

### 6. 真实数据 1~2 步训练烟囱
- [scripts/train.py](/root/autodl-tmp/ADMamba/scripts/train.py) 用 `pl.Trainer(max_epochs=config.max_epoch)` 控制，没有 `--max-steps` CLI。为了不修改任何 config/脚本，写一个一次性 inline runner：

```bash
ADMAMBA_DATA_VAIHINGEN=/root/autodl-tmp/ADMamba/data/vaihingen \
ADMAMBA_WEIGHTS_ROOT=/root/autodl-tmp/ADMamba/_smoke_weights \
python - <<'PY'
import os, sys, pytorch_lightning as pl
sys.path.insert(0, "/root/autodl-tmp/ADMamba")
from tools.cfg import py2cfg
from scripts.train import Supervision_Train
cfg = py2cfg("configs/vaihingen/ad_mamba.py")
model = Supervision_Train(cfg)
trainer = pl.Trainer(max_steps=2, limit_val_batches=1, num_sanity_val_steps=0,
                     accelerator="gpu", devices=1, precision="16-mixed",
                     logger=False, enable_checkpointing=False)
trainer.fit(model, cfg.train_loader, cfg.val_loader)
print("SMOKE_OK")
PY
```
- `max_steps=2` + 关 ckpt + 关 logger + 跳 sanity_val，避免污染 `lightning_logs/` 和 `model_weights/`。
- 看到 `SMOKE_OK` 即视为"跑通"。

### 7. 跑通后更新 [README.md](/root/autodl-tmp/ADMamba/README.md) 的 Installation 一节
- 用 `StrReplace` 覆写第 64–86 行 Installation 区块（不动其他章节，符合用户"不要改我没要求改的代码"规则）。
- 新内容：
  - 标注本机已验证组合（CUDA Toolkit 12.1 / RTX 4090 / Ubuntu 22.04 / Python 3.10 / torch 2.3.1+cu121 / mamba-ssm 2.2.2）。
  - 写出步骤 1–4 的精确命令（含 `--prefix` 装到数据盘的可选写法、`--no-build-isolation`、`MAX_JOBS`、`CUDA_HOME` 这些坑点）。
  - 烟囱测试两条命令（步骤 5、6）作为 "Verify install" 子节。
- 不改其他 md，不新建 md。

## 失败回退
- 如果步骤 4 编译 `mamba-ssm` 失败（最常见原因：内存不足或版本对不上），回退方案：
  1. `pip cache purge` 后重试；
  2. 降到 `mamba-ssm==1.2.0.post1 + causal-conv1d==1.2.0.post2`（README 的旧组合）；
  3. 实在不行，用 `MAMBA_FORCE_BUILD=TRUE` 强制本地编译并 `MAX_JOBS=2`。
- 失败时不更新 README，把详细错误回报给您再决定方向。