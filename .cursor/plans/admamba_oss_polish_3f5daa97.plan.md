---
name: admamba oss polish
overview: 把仓库范围收敛到 Vaihingen + Potsdam 两个数据集，统一 ablation 类名为 ADMamba，修复依赖冲突与文档过期问题，补齐 pyproject.toml / CITATION.cff 等开源元数据，使其达到可发布的开源仓库标准。
todos:
  - id: drop_uavid_loveda
    content: 删除 uavid/loveda 相关 configs / datasets / scripts / tools 文件，同步从 inference_huge_image.py 移除 uavid 分支
    status: completed
  - id: rename_ablation_classes
    content: ablations/ 下 4 个文件重命名内部 PyramidMamba→ADMamba，同步更新 scripts/benchmark_ablation.py 调用
    status: completed
  - id: fix_brand_strings
    content: 修正 analysis/analyze_cosine_similarity.py 注释/打印中的 PyramidMamba 品牌，重命名 docs/MAMBA_SCAN_IMPROVEMENTS.md 为 AD_MAMBA_SCAN_IMPROVEMENTS.md 并替换标题/萾款
    status: completed
  - id: fix_requirements
    content: 从 requirements.txt 删除冲突的 lightning==2.0.0
    status: completed
  - id: lazy_package_init
    content: 重写 admamba/__init__.py 为 lazy，并在 admamba/datasets/__init__.py 显式暴露 VaihingenDataset/PotsdamDataset
    status: completed
  - id: align_reference_configs
    content: 对齐 configs/{vaihingen,potsdam}/{dcswin,ftunetformer,unetformer}.py 的 weights_path / dataset_root 与 ad_mamba.py 风格（env 驱动）
    status: completed
  - id: add_pyproject_citation
    content: 新增 pyproject.toml (支持 pip install -e .) 与 CITATION.cff
    status: completed
  - id: rewrite_readme
    content: README.md 刷新：删除 uavid/loveda 节、嵌入 assets 图、补 pip install -e . 说明、Citation 占位位 AD-Mamba bibtex
    status: completed
  - id: fix_benchmark_doc
    content: 重写 docs/BENCHMARK_USAGE.md 中的 路径/脚本名为新结构
    status: completed
  - id: verify
    content: 最终验收：compileall 、无 mamba_ssm 下 import admamba 、rg 验证无残留
    status: completed
isProject: false
---

## 1. 收敛数据集范围至 Vaihingen + Potsdam

### 删除

- 目录：[configs/loveda/](configs/loveda/)、[configs/uavid/](configs/uavid/)
- 数据集模块：[admamba/datasets/loveda_dataset.py](admamba/datasets/loveda_dataset.py)、[admamba/datasets/uavid_dataset.py](admamba/datasets/uavid_dataset.py)
- 脚本：[scripts/test_loveda.py](scripts/test_loveda.py)、[scripts/test_uavid.py](scripts/test_uavid.py)、[scripts/inference_uavid.py](scripts/inference_uavid.py)
- 预处理：[tools/loveda_mask_convert.py](tools/loveda_mask_convert.py)、[tools/uavid_patch_split.py](tools/uavid_patch_split.py)

### 修改

- [scripts/inference_huge_image.py](scripts/inference_huge_image.py)：删 `uavid2rgb`、把 `--dataset` choices 收敛为 `["pv"]`、移除 `args.dataset == 'uavid'` 分支。
- [README.md](README.md)：删除 UAVid / LoveDA 相关章节、命令、environment-variable 行 `ADMAMBA_DATA_UAVID`、`Repository layout` 中的 `uavid_dataset` / `loveda_dataset` 等描述。

## 2. ablation 与品牌统一：PyramidMamba → ADMamba

- [ablations/ad_mamba_baseline.py](ablations/ad_mamba_baseline.py)、[ablations/ad_mamba_4dir.py](ablations/ad_mamba_4dir.py)、[ablations/ad_mamba_8dir.py](ablations/ad_mamba_8dir.py)、[ablations/ad_mamba_fdg_step.py](ablations/ad_mamba_fdg_step.py)：把内部 `class PyramidMamba` → `ADMamba`，`class EfficientPyramidMamba` → `EfficientADMamba`。
- [scripts/benchmark_ablation.py](scripts/benchmark_ablation.py)：把 `mod_orig.PyramidMamba(...)`、`mod_4dir.PyramidMamba(...)`、`mod_8dir.PyramidMamba(...)`、`mod_5.PyramidMamba(...)` 全部改为 `.ADMamba(...)`。
- [analysis/analyze_cosine_similarity.py](analysis/analyze_cosine_similarity.py) 行 4、242：把字符串 "PyramidMamba" 改成 "AD-Mamba"。
- [docs/MAMBA_SCAN_IMPROVEMENTS.md](docs/MAMBA_SCAN_IMPROVEMENTS.md)：重命名为 `docs/AD_MAMBA_SCAN_IMPROVEMENTS.md`，同时把第 1 行标题 "PyramidMamba 扫描机制改进方案：基于前沿数学理论" → "AD-Mamba 扫描机制设计：基于前沿数学理论"，把第 1190 行落款 "作者：PyramidMamba 研究团队" → "作者：AD-Mamba"。

## 3. 修复依赖与启动期硬依赖

- [requirements.txt](requirements.txt)：删除 `lightning==2.0.0` 这一行（与 `pytorch-lightning==2.3.0` 冲突；代码只用后者）。
- [admamba/__init__.py](admamba/__init__.py)：改为只暴露版本号 + `__all__`，**不**急加载 `models`，让 `import admamba` 不再硬依赖 `mamba_ssm`。具体：

```python
"""AD-Mamba: Anisotropic-Direction Mamba for remote sensing semantic segmentation."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

  使用者按需 `from admamba.models import ADMamba`、`from admamba.datasets import VaihingenDataset` 即可。
- [admamba/datasets/__init__.py](admamba/datasets/__init__.py)：改为显式 re-export

```python
from .vaihingen_dataset import VaihingenDataset
from .potsdam_dataset import PotsdamDataset

__all__ = ["VaihingenDataset", "PotsdamDataset"]
```

## 4. 把对照模型 config 的风格对齐 ad_mamba.py

[configs/vaihingen/](configs/vaihingen/) 与 [configs/potsdam/](configs/potsdam/) 下的 `dcswin.py`、`ftunetformer.py`、`unetformer.py`（共 6 个文件）与 ad_mamba.py 风格不一致：缺 `import os` / `import torch`、`weights_path` 与 `dataset_root` 仍硬编码。统一改为：

```python
import os
import torch
...
weights_path = os.path.join(
    os.environ.get("ADMAMBA_WEIGHTS_ROOT", "model_weights"),
    "vaihingen", weights_name,
)
dataset_root = os.environ.get("ADMAMBA_DATA_VAIHINGEN", "data/vaihingen")
train_dataset = VaihingenDataset(data_root=f"{dataset_root}/train", ...)
```

Potsdam 下同理替换 `ADMAMBA_DATA_POTSDAM`。

## 5. 补齐开源仓库元数据

### 新增 [pyproject.toml](pyproject.toml)（PEP 621 + setuptools）

声明 `name="admamba"`、`version="0.1.0"`、`requires-python=">=3.10"`、`dependencies` 从 `requirements.txt` 同步（mamba-ssm/causal-conv1d 放 `[project.optional-dependencies] cuda` 以便 CPU 用户能至少装核），`packages=["admamba", "admamba.datasets", "admamba.losses", "admamba.models", "tools"]`。这样支持 `pip install -e .`。

### 新增 [CITATION.cff](CITATION.cff)

最小 GitHub 兼容 schema：作者、title="AD-Mamba"、license=GPL-3.0、url、preferred-citation 指向论文（暂留占位）。

### [README.md](README.md) 完整刷新

- 顶部加 `assets/vai.png` / `assets/pot.png` 缩略图引用（否则两张图躺着没用）。
- Repository layout 中删 uavid/loveda 行；把 `from admamba.models import ADMamba` 示例保留；新增 `pip install -e .` 说明。
- Data preparation 表只留 Vaihingen / Potsdam；environment variable 表删 `ADMAMBA_DATA_UAVID`。
- Training / Testing 章节只保留 Vaihingen + Potsdam 命令。
- `## Top-k ablation` / `## Scan / gate ablation benchmark` 两节保留（仅依赖 Vaihingen）。
- Citation 区块：在 PyramidMamba bibtex 之上预留 AD-Mamba 占位条目（标 TODO）。

## 6. 修复 docs/BENCHMARK_USAGE.md 全部命令

逐条替换：
- `python benchmark_model.py` → `python scripts/benchmark_model.py`
- `config/vaihingen/PyramidMamba.py` → `configs/vaihingen/ad_mamba.py`
- 输出示例中 `Benchmarking model from: config/vaihingen/PyramidMamba.py` → `... configs/vaihingen/ad_mamba.py`

## 7. 验收

- `python -m compileall admamba ablations configs scripts tools analysis` 通过。
- `python -c "import admamba; from admamba.datasets import VaihingenDataset"` 在**未装** mamba_ssm 的环境也能成功。
- `rg "PyramidMamba|/root/autodl|from geoseg|from GeoSeg|train_supervision"` 只在 README 致谢章节、CITATION 区块、docs/AD_MAMBA_SCAN_IMPROVEMENTS.md 学术对比段落出现（合理引用）。
- `rg -i "uavid|loveda"` 无残留。
- 顶级目录列表保持 `admamba/ ablations/ analysis/ assets/ configs/ docs/ scripts/ tools/` + `LICENSE README.md requirements.txt pyproject.toml CITATION.cff .gitignore`。

## 不在本次范围内

- 修复 README "Known issues" 列出的 3 项功能性 bug（MoE 路径上 FDG/GeoMSAA 被旁路、HardTopKRouting 反传不传梯度、scheduled noise decay 未挂钩）—— 这些是后续 research roadmap，不属于"开源仓库标准"清理。
- 添加 GitHub Actions CI（可后续单独提）。
- 翻译 docs/AD_MAMBA_SCAN_IMPROVEMENTS.md（中文长文，保留中文写作风格）。
