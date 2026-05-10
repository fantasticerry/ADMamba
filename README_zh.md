# AD-Mamba

**English:** [README.md](README.md)

AD-Mamba（Anti-Dilution Mamba）面向遥感语义分割，在 [PyramidMamba](https://arxiv.org/abs/2406.10828) 基础上针对航拍影像做了三项扩展：

- **八方向对角扫描。** 自定义 `CrossScan` / `CrossMerge` 自动微分类对，为 Mamba 状态空间模型提供水平、垂直、两条对角线及其反向序列。
- **稀疏 top-k MoE 方向路由。** 可学习门控按样本选择信息量最大的扫描方向，并带有受 MoCE-IR 启发的负载均衡辅助损失。
- **分数阶差分门（FDG）。** 用 Grünwald–Letnikov 分数阶导数门（可选 DSM/nDSM 融合）替代原一阶差分门，沿各扫描方向建模长程依赖。


验证数据集：**ISPRS Vaihingen**、**ISPRS Potsdam**。

<p align="center">
  <img src="assets/vai.png" alt="Vaihingen qualitative result" width="48%"/>
  <img src="assets/pot.png" alt="Potsdam qualitative result" width="48%"/>
</p>

实现改编自 [`WangLibo1995/GeoSeg`](https://github.com/WangLibo1995/GeoSeg)，沿用其 `pytorch_lightning` + `timm` 训练脚手架。

## 仓库结构

```text
ADMamba/
├── admamba/                # 可安装的 Python 包
│   ├── datasets/           # Vaihingen / Potsdam 数据加载
│   ├── losses/             # 交叉熵、Dice、Lovasz 等
│   └── models/             # ad_mamba.py 与参考网络
├── ablations/              # benchmark_ablation.py 使用的独立快照
│   ├── ad_mamba_baseline.py    # 原始单行扫描
│   ├── ad_mamba_4dir.py        # 四方向扫描
│   ├── ad_mamba_8dir.py        # 八方向扫描
│   └── ad_mamba_fdg_step.py    # 八选四 + 一阶 FDG（仅 RGB）
├── configs/                # tools/cfg.py 读取的 py 配置
│   ├── vaihingen/
│   └── potsdam/
├── tools/                  # 配置 / 指标 / 数据预处理
├── scripts/                # 训练、测试、推理、基准
│   ├── train.py
│   ├── test_{vaihingen,potsdam}.py
│   ├── inference_huge_image.py
│   ├── benchmark_model.py
│   └── benchmark_ablation.py
├── analysis/               # 论文图表与事后分析
├── docs/                   # 长文说明（benchmark、扫描分析）
├── assets/                 # 论文配图
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## 安装

> 本环境已在下列组合上验证：Ubuntu 22.04 / RTX 4090 / CUDA Toolkit 12.1 / gcc 11.4 /
> Python 3.10 / `torch==2.3.1+cu121` / `causal-conv1d==1.4.0` /
> `mamba-ssm==2.2.2` / `transformers==4.43.4`。

### 1. 创建 conda 环境

完整环境约 **6.4 GB**（PyTorch、mamba-ssm wheel、其余依赖）。在 AutoDL 类「系统盘小、数据盘大」的机器上可选：

- **系统盘**（conda 默认路径，如 `/root/miniconda3/envs/`）：保存自定义镜像时会打进镜像，换实例可复用；除非系统盘极其紧张，否则推荐。
- **数据盘**（`--prefix /root/autodl-tmp/envs/admamba`）：不占系统盘，但**不会**随镜像打包（数据盘内容留在宿主机）。

无论哪种，建议把 pip 缓存与编译临时目录放在数据盘，避免镜像膨胀：

```bash
conda create -y -n admamba python=3.10
conda activate admamba

export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
export TMPDIR=/root/autodl-tmp/tmp && mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
pip install -U pip wheel "setuptools<81" packaging ninja
```

必须 **`setuptools<81`**：`pytorch_lightning==2.3.0` 仍依赖已弃用的 `pkg_resources`。

### 2. 安装 PyTorch（cu121）

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出: 2.3.1+cu121 True
```

### 3. 以可编辑模式安装本项目

```bash
cd <仓库根目录>
pip install -e .
pip install "transformers==4.43.4"    # mamba-ssm 2.2.2 需要 4.4.x API
```

国内 PyPI 慢时可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 4. 安装 CUDA 扩展（`causal-conv1d`、`mamba-ssm`）

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4

# causal-conv1d 可从 sdist 正常编译（4090 机器上约 9 分钟）：
pip install --no-build-isolation causal-conv1d==1.4.0
```

`mamba-ssm==2.2.2` 的 PyPI **sdist 不完整**（缺少 `csrc/selective_scan/*.cpp`），源码编译会失败。请从官方发布页下载与 **torch / CUDA / Python** 匹配的预编译 wheel：
[state-spaces/mamba v2.2.2 Releases](https://github.com/state-spaces/mamba/releases/tag/v2.2.2)。

下列文件名对应上述验证组合（约 308 MB；标 cu122 的 wheel 可与 cu121 运行时共用）：

```bash
mkdir -p /root/autodl-tmp/wheels && cd /root/autodl-tmp/wheels
WHEEL=mamba_ssm-2.2.2+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 直连 GitHub 往往很慢或被拦；可用镜像加速：
curl -fL --retry 3 -o "$WHEEL" \
  "https://ghfast.top/https://github.com/state-spaces/mamba/releases/download/v2.2.2/$WHEEL"
pip install --no-build-isolation "$WHEEL"
```

### 5. 验证安装

```bash
python -c "
import torch, causal_conv1d, mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('mamba_ssm', mamba_ssm.__version__)
"
# 期望: torch 2.3.1+cu121 cuda True / mamba_ssm 2.2.2
```

**前向冒烟**（无需训练数据，约 1 分钟）：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 无法访问 huggingface.co 时需要
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py
# 期望末尾：约 229.63M 参数 / ~400 GFLOPs / 4090 上 FPS ~19
```

**两步真实数据训练冒烟**（使用 Vaihingen 1024 patch，约 15 秒，不落盘 checkpoint/logger）：

```bash
export ADMAMBA_DATA_VAIHINGEN=$(pwd)/data/vaihingen
export ADMAMBA_WEIGHTS_ROOT=$(pwd)/_smoke_weights
python - <<'PY'
import sys, pytorch_lightning as pl
sys.path.insert(0, ".")
from tools.cfg import py2cfg
from scripts.train import Supervision_Train
cfg = py2cfg("configs/vaihingen/ad_mamba.py")
trainer = pl.Trainer(max_steps=2, limit_val_batches=1, num_sanity_val_steps=0,
                     accelerator="gpu", devices=1, logger=False,
                     enable_checkpointing=False)
trainer.fit(Supervision_Train(cfg), cfg.train_loader, cfg.val_loader)
print("SMOKE_OK")
PY
```

出现 **`SMOKE_OK`** 表示环境、Mamba 内核、数据管线、损失与反传均已连通。

安装完成后可直接：

```python
from admamba.models import ADMamba

model = ADMamba(num_classes=6, use_fractional_gate=True, fractional_alpha=0.8)
```

## 环境与网络注意事项

在租用 GPU、上游镜像慢或被墙时整理的经验。

### Conda 激活

部分终端未加载 conda 钩子，`conda activate admamba` 会报错：`Run 'conda init' before 'conda activate'`。

任选其一：执行一次 `conda init bash` 后重开终端；或在当前 shell 显式加载：

```bash
source /root/miniconda3/etc/profile.d/conda.sh   # Miniconda 路径按机器修改
conda activate admamba
```

等价写法：

```bash
source /root/miniconda3/bin/activate admamba
```

### pip / PyTorch 大包下载

`torch`、重度编译包可能出现 `BrokenPipeError`、`ConnectTimeout` 或中途卡住，可加 **`--timeout 300 --retries 5`** 重试。纯 Python 依赖安装可加清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 编译 `causal-conv1d`

日志长时间停在 `Building wheel for causal-conv1d`、`nvcc`/ninja 在后台编译属于**正常现象**。

### 勿把 `transformers` 升到 5.x

装好 `mamba-ssm` 后若被拉到 `transformers` 5.x，会因 API / torch 版本检查报错；请固定 **`transformers==4.43.4`**（见上文 §3）。

### 优先使用官方 `mamba-ssm` wheel

裸执行 `pip install mamba-ssm==2.2.2` 会先探测 GitHub release（外网差时常「半天没输出」），且 PyPI sdist 无法本地重编 CUDA。**务必按 §4** 用镜像下载对应 `.whl` 再本地 `pip install`。

### Hugging Face Hub / timm 预训练权重

`ADMamba` 默认 `pretrained=True`，主干为 Swin；`timm` 会从 **huggingface.co** 拉取约 **365 MB** 的 `model.safetensors`。访问超时请在运行 **`train.py`、`test_*.py`、`benchmark_model.py`、`inference_huge_image.py`、`benchmark_ablation.py`** 及一切会构建网络的脚本**之前**设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120    # 可选
python scripts/train.py -c configs/vaihingen/ad_mamba.py
```

首次成功下载会出现 tqdm 进度条（如 `model.safetensors: … 365M/365M`）。权重缓存在 **`~/.cache/huggingface/hub/`**，之后同一模型通常静默读缓存。若要强制重新下载：

```bash
rm -rf ~/.cache/huggingface/hub
```

调试 Hub 请求：`HF_HUB_VERBOSITY=debug`。

### 混合精度训练器

若在 Lightning 里使用 `precision="16-mixed"`，可能在 `mamba_ssm` 内触发 dtype 不一致。**`scripts/train.py` 使用默认 FP32 路径**；除非自行修补内核栈，否则不要随意对整个管线开 AMP。

### Lightning 的 `pkg_resources` 警告

在 `setuptools<81` 下 Lightning 可能打印 `pkg_resources` 弃用警告；在 Lightning 移除该依赖前可视为**噪声**，一般可忽略。

## 数据准备

目录约定与原版 GeoSeg 一致。下载官方数据后放入 `data/`：

```text
data/
├── vaihingen/
│   ├── train_images/  ├── train_masks/  ├── train_dsm/
│   └── test_images/   ├── test_masks/   ├── test_masks_eroded/  └── test_dsm/
└── potsdam/  （与 vaihingen 结构对应）
```

使用 `tools/` 下脚本将原图裁成 1024×1024 patch：

```bash
python tools/vaihingen_patch_split.py \
  --img-dir  "data/vaihingen/train_images" \
  --mask-dir "data/vaihingen/train_masks" \
  --output-img-dir  "data/vaihingen/train/images_1024" \
  --output-mask-dir "data/vaihingen/train/masks_1024" \
  --mode train --split-size 1024 --stride 512

python tools/vaihingen_patch_split.py \
  --img-dir  "data/vaihingen/test_images" \
  --mask-dir "data/vaihingen/test_masks_eroded" \
  --output-img-dir  "data/vaihingen/test/images_1024" \
  --output-mask-dir "data/vaihingen/test/masks_1024" \
  --mode val --split-size 1024 --stride 1024 --eroded

python tools/vaihingen_dsm_split.py            # 切分 nDSM patch
python tools/potsdam_patch_split.py  --help
```

配置通过环境变量读取数据根路径，无需改配置文件：

| 变量                             | 默认值           |
| -------------------------------- | ---------------- |
| `ADMAMBA_DATA_VAIHINGEN`         | `data/vaihingen` |
| `ADMAMBA_DATA_POTSDAM`           | `data/potsdam`   |
| `ADMAMBA_WEIGHTS_ROOT`           | `model_weights`  |

## 训练

若无法直连 Hugging Face，请先导出镜像端点（详见上文 **环境与网络注意事项**）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
# Vaihingen：默认分数阶 FDG（alpha=0.8）+ RGB+DSM
python scripts/train.py -c configs/vaihingen/ad_mamba.py

# Potsdam
python scripts/train.py -c configs/potsdam/ad_mamba.py
```

检查点保存在 `<ADMAMBA_WEIGHTS_ROOT>/<数据集>/<weights_name>/`，CSV 日志在 `lightning_logs/`。

## 测试

`-t` 选择测试时增强：`None`、`lr`（翻转）、`d4`（多尺度+翻转）。`--rgb` 输出彩色预测图。

加载 timm 预训练主干与训练阶段同源；网络不通时请设置 `HF_ENDPOINT`（见 **环境与网络注意事项**）。

```bash
python scripts/test_vaihingen.py \
  -c configs/vaihingen/ad_mamba.py \
  -o output/vaihingen/ad_mamba -t d4 --rgb

python scripts/test_potsdam.py \
  -c configs/potsdam/ad_mamba.py \
  -o output/potsdam/ad_mamba -t lr --rgb
```

超大图推理示例：

```bash
python scripts/inference_huge_image.py \
  -i data/vaihingen/test_images \
  -c configs/vaihingen/ad_mamba.py \
  -o output/vaihingen/ad_mamba_huge \
  -t lr -ph 512 -pw 512 -b 2 -d pv
```

## Top-k 消融（Vaihingen）

MoE 方向路由三种 top-k 变体共用同一训练配方；按 `k` 选配置：

```bash
python scripts/train.py -c configs/vaihingen/ad_mamba_topk1.py
python scripts/train.py -c configs/vaihingen/ad_mamba_topk2.py
python scripts/train.py -c configs/vaihingen/ad_mamba_topk3.py

python scripts/test_vaihingen.py \
  -c configs/vaihingen/ad_mamba_topk2.py \
  -o output/vaihingen/ad_mamba_topk2 -t d4 --rgb
```

## 扫描 / 门控消融基准

`scripts/benchmark_ablation.py` 在六种设计点（1 行、4 行、8 行、八选四、一阶 FDG、分数阶 FDG）上统计 FLOPs / FPS / 参数量；同时加载 `ablations/` 下四个快照与正式实现 `admamba/models/ad_mamba.py`：

```bash
python scripts/benchmark_ablation.py
```

对 canonical 模型按配置批量跑 benchmark：

```bash
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py
python scripts/benchmark_model.py -d configs/vaihingen/  # 批量
```

## 分析脚本（论文图）

```bash
python analysis/analyze_cosine_similarity.py
python analysis/analyze_direction_activation.py -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/analyze_expert_direction.py    -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/plot_ideal_and_spatial.py      -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/plot_final_figures.py
```

分析脚本需设置 `ADMAMBA_VAIHINGEN_TEST` / `ADMAMBA_VAIHINGEN_IMAGES`，或保证数据位于仓库根下相对路径 `data/vaihingen/...`。

## 已知问题

- `enable_moe=True` 时，`SparseMoELayer.apply_gate` 会静默绕过 `FractionalDifferenceGate` 与 `ElevationGuidedGate` —— 门控模块挂在 `MambaLayer` 上但未接入稀疏路径。`configs/` 中带分数阶 FDG 的实验走的是 MoE 关闭时的稠密路径；纯 MoE 变体共享扫描方式但不经过分数阶门控。修复接线在路线图内。
- `HardTopKRouting.backward` 丢弃 `grad_scores` 并返回 `None`，路由 logits 无梯度 —— 符合 STE 常见写法，但门控无法仅靠路由决策本身微调。
- `MambaLayer.update_training_step(step)` 为门控内噪声衰减预留，`scripts/train.py` 中的 Trainer **尚未调用**。

欢迎提交 PR 修复上述问题。

## 致谢

本项目建立在下列工作之上：

- [GeoSeg / PyramidMamba](https://github.com/WangLibo1995/GeoSeg)
- [Mamba](https://github.com/state-spaces/mamba) 与 `mamba-ssm` CUDA 内核
- [pytorch-lightning](https://www.pytorchlightning.ai/)、[timm](https://github.com/rwightman/pytorch-image-models)、
  [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)、
  [ttach](https://github.com/qubvel/ttach)、[catalyst](https://github.com/catalyst-team/catalyst)

