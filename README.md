# AD-Mamba

AD-Mamba (Anisotropic-Direction Mamba) is a remote-sensing semantic
segmentation framework that extends [PyramidMamba](https://arxiv.org/abs/2406.10828)
with four ideas tailored to overhead imagery:

- **8-direction diagonal scanning.** A custom `CrossScan` / `CrossMerge`
  autograd pair feeds the Mamba state-space model with horizontal, vertical,
  diagonal, and anti-diagonal sequences (and their reverses).
- **Sparse top-k MoE direction routing.** A learnable gate selects the most
  informative scan directions per sample, with a load-balancing auxiliary
  loss inspired by MoCE-IR.
- **Fractional-order difference gate (FDG).** A Grünwald-Letnikov fractional
  derivative gate (with optional DSM/nDSM fusion) replaces the original
  first-order difference gate to capture long-range dependencies along each
  scan direction.
- **Elevation-guided multi-scale attention (GeoMSAA).** When height data is
  available, a small router blends 3×3 / 5×5 / 7×7 fusion branches in the
  decoder according to the local elevation profile.

Validated datasets: **ISPRS Vaihingen** and **ISPRS Potsdam**.

<p align="center">
  <img src="assets/vai.png" alt="Vaihingen qualitative result" width="48%"/>
  <img src="assets/pot.png" alt="Potsdam qualitative result" width="48%"/>
</p>

The implementation is adapted from
[`WangLibo1995/GeoSeg`](https://github.com/WangLibo1995/GeoSeg) and inherits
its `pytorch_lightning` + `timm` training scaffold.

## Repository layout

```text
ADMamba/
├── admamba/                # Importable Python package
│   ├── datasets/           # Vaihingen / Potsdam loaders
│   ├── losses/             # Cross-entropy, Dice, Lovasz, ...
│   └── models/             # ad_mamba.py + reference networks
├── ablations/              # Standalone snapshots used by benchmark_ablation.py
│   ├── ad_mamba_baseline.py    # Original 1-row scan
│   ├── ad_mamba_4dir.py        # 4-direction scan
│   ├── ad_mamba_8dir.py        # 8-direction scan
│   └── ad_mamba_fdg_step.py    # 8-select-4 + 1st-order FDG (RGB only)
├── configs/                # py-config files consumed by tools/cfg.py
│   ├── vaihingen/
│   └── potsdam/
├── tools/                  # cfg / metric / data preprocessing utilities
├── scripts/                # Training, testing, inference, benchmarking
│   ├── train.py
│   ├── test_{vaihingen,potsdam}.py
│   ├── inference_huge_image.py
│   ├── benchmark_model.py
│   └── benchmark_ablation.py
├── analysis/               # Paper figures & post-hoc analyses
├── docs/                   # Long-form notes (benchmark, scan analysis)
├── assets/                 # Static figures used in the paper
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
conda create -n admamba python=3.10
conda activate admamba

# Install PyTorch matching your CUDA toolkit. Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the package and its Python dependencies in editable mode:
pip install -e .

# Mamba SSM kernels (CUDA-only). Install AFTER PyTorch is in place:
pip install -e ".[cuda]"
```

If you prefer the legacy flow without `pyproject.toml`:

```bash
pip install -r requirements.txt
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=1.2.0
```

After installation the package is importable directly:

```python
from admamba.models import ADMamba

model = ADMamba(num_classes=6, use_fractional_gate=True, fractional_alpha=0.8)
```

## Data preparation

The model expects the same folder structure as the original GeoSeg project.
Download the official datasets and place them under `data/`:

```text
data/
├── vaihingen/
│   ├── train_images/  ├── train_masks/  ├── train_dsm/
│   └── test_images/   ├── test_masks/   ├── test_masks_eroded/  └── test_dsm/
└── potsdam/  (mirrors vaihingen/)
```

Use the helpers under `tools/` to crop the originals into 1024×1024 patches:

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

python tools/vaihingen_dsm_split.py            # split nDSM patches
python tools/potsdam_patch_split.py  --help
```

Configs read the data root from the environment so non-default paths do not
require editing the config files:

| Variable                         | Default                       |
| -------------------------------- | ----------------------------- |
| `ADMAMBA_DATA_VAIHINGEN`         | `data/vaihingen`              |
| `ADMAMBA_DATA_POTSDAM`           | `data/potsdam`                |
| `ADMAMBA_WEIGHTS_ROOT`           | `model_weights`               |

## Training

```bash
# Vaihingen with the default fractional-order FDG (alpha = 0.8) + RGB+DSM
python scripts/train.py -c configs/vaihingen/ad_mamba.py

# Potsdam
python scripts/train.py -c configs/potsdam/ad_mamba.py
```

Checkpoints land under `<ADMAMBA_WEIGHTS_ROOT>/<dataset>/<weights_name>/`.
CSV logs land under `lightning_logs/`.

## Testing

`-t` selects test-time augmentation: `None`, `lr` (flip), or `d4`
(multi-scale + flip). `--rgb` writes color-coded predictions.

```bash
python scripts/test_vaihingen.py \
  -c configs/vaihingen/ad_mamba.py \
  -o output/vaihingen/ad_mamba -t d4 --rgb

python scripts/test_potsdam.py \
  -c configs/potsdam/ad_mamba.py \
  -o output/potsdam/ad_mamba -t lr --rgb
```

For inference on a custom huge image:

```bash
python scripts/inference_huge_image.py \
  -i data/vaihingen/test_images \
  -c configs/vaihingen/ad_mamba.py \
  -o output/vaihingen/ad_mamba_huge \
  -t lr -ph 512 -pw 512 -b 2 -d pv
```

## Top-k ablation (Vaihingen)

The three top-k variants of the MoE direction router share the same training
recipe; pick a config by `k`:

```bash
python scripts/train.py -c configs/vaihingen/ad_mamba_topk1.py
python scripts/train.py -c configs/vaihingen/ad_mamba_topk2.py
python scripts/train.py -c configs/vaihingen/ad_mamba_topk3.py

python scripts/test_vaihingen.py \
  -c configs/vaihingen/ad_mamba_topk2.py \
  -o output/vaihingen/ad_mamba_topk2 -t d4 --rgb
```

## Scan / gate ablation benchmark

`scripts/benchmark_ablation.py` measures FLOPs / FPS / parameters across the
six AD-Mamba design points (1-row, 4-row, 8-row, 8-select-4, 1st-order FDG,
fractional FDG). It loads the four standalone module snapshots stored in
`ablations/` together with the canonical `admamba/models/ad_mamba.py`:

```bash
python scripts/benchmark_ablation.py
```

Per-config benchmarking on the canonical model:

```bash
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py
python scripts/benchmark_model.py -d configs/vaihingen/  # batch
```

## Analyses (paper figures)

```bash
python analysis/analyze_cosine_similarity.py
python analysis/analyze_direction_activation.py -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/analyze_expert_direction.py    -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/plot_ideal_and_spatial.py      -c configs/vaihingen/ad_mamba.py --ckpt <path/to.ckpt>
python analysis/plot_final_figures.py
```

The analysis scripts expect either `ADMAMBA_VAIHINGEN_TEST` /
`ADMAMBA_VAIHINGEN_IMAGES` to be set, or the data to live under
`data/vaihingen/...` relative to the repository root.

## Known issues

- When `enable_moe=True`, the `FractionalDifferenceGate` and
  `ElevationGuidedGate` modules are silently bypassed inside
  `SparseMoELayer.apply_gate` — the gating modules are attributes of
  `MambaLayer` and are not wired through to the sparse path. The
  configurations in `configs/` therefore route their fractional-FDG
  experiments through the dense path that `MambaLayer.forward` takes when
  the relevant flag is set; the MoE-only variants share the same scan but
  do not receive fractional gating. Fixing the wiring is on the roadmap.
- `HardTopKRouting.backward` discards the computed `grad_scores` and
  returns `None`, so the routing logits receive no gradient. This is the
  textbook STE behaviour but means the gate cannot be tuned by the routing
  decision alone.
- `MambaLayer.update_training_step(step)` is exposed for scheduled noise
  decay in the gate, but the trainer in `scripts/train.py` does not yet
  call it.

Pull requests addressing any of the above are welcome.

## Acknowledgements

This project would not exist without the work it is built on:

- [GeoSeg / PyramidMamba](https://github.com/WangLibo1995/GeoSeg)
- [Mamba](https://github.com/state-spaces/mamba) and the `mamba-ssm` kernels
- [pytorch-lightning](https://www.pytorchlightning.ai/), [timm](https://github.com/rwightman/pytorch-image-models),
  [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt),
  [ttach](https://github.com/qubvel/ttach), and [catalyst](https://github.com/catalyst-team/catalyst)

## Citation

If you find AD-Mamba useful, please cite this repository together with the
upstream PyramidMamba and Mamba papers:

```bibtex
@software{admamba2025,
  title  = {AD-Mamba: Anisotropic-Direction Mamba for Remote Sensing Semantic Segmentation},
  author = {AD-Mamba contributors},
  year   = {2025},
  url    = {https://github.com/your-org/ADMamba},
  note   = {TODO: replace with paper bibtex once published}
}

@article{wang2024pyramidmamba,
  title={PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation of Remote Sensing Imagery},
  author={Wang, Libo and others},
  journal={arXiv preprint arXiv:2406.10828},
  year={2024}
}
```
