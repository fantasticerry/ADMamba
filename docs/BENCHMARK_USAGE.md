# 模型性能测试工具使用说明

## 功能
`benchmark_model.py` 脚本用于测试不同模型的性能指标：
- **参数数量 (Parameters)**: 模型的总参数数量和可训练参数数量
- **模型大小 (Model Size)**: 模型文件大小（MB）
- **GFLOPS**: 模型的计算量（需要安装thop库）
- **FPS**: 模型推理速度（每秒处理的帧数）
- **推理时间 (Inference Time)**: 单次推理的平均时间（毫秒）

## 安装依赖

### 必需依赖
项目已有的依赖（PyTorch等）

### 可选依赖（用于计算GFLOPS）
```bash
pip install thop
```

## 使用方法

### 1. 测试单个模型

```bash
# 使用默认输入尺寸 (1, 3, 512, 512)
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py

# 指定输入尺寸
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py --input_size 1 3 1024 1024

# 使用真实数据集测试FPS（更准确）
python scripts/benchmark_model.py -c configs/vaihingen/ad_mamba.py --use_real_data
```

### 2. 批量测试多个模型

```bash
# 测试 configs 目录下的所有模型
python scripts/benchmark_model.py -d configs/vaihingen/

# 指定输入尺寸
python scripts/benchmark_model.py -d configs/vaihingen/ --input_size 1 3 512 512

# 使用真实数据集
python scripts/benchmark_model.py -d configs/vaihingen/ --use_real_data
```

## 参数说明

- `-c, --config_path`: 单个配置文件路径
- `-d, --config_dir`: 配置文件目录（批量测试）
- `--input_size`: 输入尺寸，格式为 `[batch, channels, height, width]`，默认 `1 3 512 512`
  - **注意**: 脚本会自动检测模型期望的输入尺寸。如果模型有固定的输入尺寸要求（如timm模型的img_size），脚本会自动使用该尺寸，覆盖用户指定的尺寸
- `--use_real_data`: 使用真实数据集测试FPS（更准确，但需要数据集存在）

## 输出说明

脚本会输出以下信息：

1. **参数统计**
   - Total Parameters: 总参数数量
   - Trainable Parameters: 可训练参数数量
   - Model Size: 模型大小（MB）

2. **计算量 (FLOPs)**
   - GFLOPs: 十亿次浮点运算（需要thop库）

3. **推理速度**
   - FPS: 每秒处理的帧数
   - Average Inference Time: 平均推理时间（毫秒）

4. **对比表格**（批量测试时）
   - 所有模型的性能对比表格

## 示例输出

```
============================================================
Benchmarking model from: configs/vaihingen/ad_mamba.py
============================================================

1. Calculating Parameters...
   Total Parameters: 12,345,678 (12.35M)
   Trainable Parameters: 12,345,678
   Model Size: 47.09 MB
   Model Size (MB): 47.09 MB

2. Calculating FLOPs...
   FLOPs: 15.23 GFLOPs
   Parameters (thop): 12,345,678

3. Measuring FPS...
   Using dummy input with size: (1, 3, 512, 512)...
   FPS: 45.67
   Average Inference Time: 21.90 ms

============================================================
Summary:
  Parameters: 12.35M
  Model Size: 47.09 MB
  FLOPs: 15.23 GFLOPs
  FPS: 45.67
  Inference Time: 21.90 ms
============================================================
```

## 注意事项

1. **GPU要求**: 脚本需要在有CUDA的GPU上运行
2. **内存**: 确保GPU有足够内存加载模型
3. **thop库**: 如果不安装thop，将无法计算GFLOPS，但其他指标仍可正常测试
4. **输入尺寸**: 
   - 脚本会自动检测模型期望的输入尺寸（从backbone的img_size或patch_embed获取）
   - 如果检测到模型有固定输入尺寸要求，会自动使用该尺寸
   - 不同的输入尺寸会影响FPS和FLOPs，请根据实际使用场景设置
5. **真实数据**: 使用 `--use_real_data` 选项可以获得更准确的FPS，但需要确保数据集路径正确

## 常见问题

**Q: 为什么FPS测试结果不稳定？**
A: FPS会受到GPU状态、内存占用等因素影响。建议多次运行取平均值，或使用 `--use_real_data` 选项。

**Q: 如何比较不同模型的性能？**
A: 使用 `-d` 选项批量测试，脚本会自动生成对比表格。

**Q: 为什么GFLOPS显示为N/A？**
A: 需要安装thop库：`pip install thop`

