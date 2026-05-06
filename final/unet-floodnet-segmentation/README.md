# 基于 U-Net 的 FloodNet 灾害场景图像语义分割研究

本项目是一个完整可运行的课程期末大作业项目，使用 PyTorch 在 FloodNet-Supervised_v1.0 数据集上训练标准 U-Net，完成 10 类灾害场景语义分割，并生成学习率、输入尺寸、数据增强策略和损失函数的调参可视化结果。

代码可先在 Windows 本地检查，正式训练建议上传到云服务器运行，目标 GPU 为 NVIDIA RTX 5090。

## 数据集说明

请将数据集放在项目根目录的 `data/` 下，目录应为：

```text
data/
├── FloodNet-Supervised_v1.0/
│   ├── train/
│   │   ├── train-org-img/
│   │   └── train-label-img/
│   ├── val/
│   │   ├── val-org-img/
│   │   └── val-label-img/
│   ├── test/
│   │   ├── test-org-img/
│   │   └── test-label-img/
│   └── DATASET-VERSION...
└── ColorMasks-FloodNetv1.0/
    ├── ColorMasks-TrainSet/
    ├── ColorMasks-ValSet/
    ├── ColorMasks-TestSet/
    └── ColorPalette-...
```

训练时只使用 `FloodNet-Supervised_v1.0` 中的 label mask。`ColorMasks-FloodNetv1.0` 只作为数据样例图里的官方彩色 mask 参考。

类别编号固定为：

| id | class |
|---:|---|
| 0 | background |
| 1 | building-flooded |
| 2 | building-non-flooded |
| 3 | road-flooded |
| 4 | road-non-flooded |
| 5 | water |
| 6 | tree |
| 7 | vehicle |
| 8 | pool |
| 9 | grass |

## 目录结构

```text
unet-floodnet-segmentation/
├── README.md
├── report_outline.md
├── requirements.txt
├── config.yaml
├── inspect_dataset.py
├── train_and_tune.py
├── visualize_results.py
├── data/
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   ├── summaries/
│   └── predictions/
└── report_assets/
    ├── dataset_examples/
    ├── curves/
    ├── bars/
    ├── qualitative/
    └── failure_cases/
```

核心 Python 文件只有 3 个：`inspect_dataset.py`、`train_and_tune.py`、`visualize_results.py`。

## 运行顺序

第一步：安装依赖

```bash
pip install -r requirements.txt
```

第二步：检查数据集

```bash
python inspect_dataset.py --config config.yaml
```

第三步：快速检查训练流程

```bash
python train_and_tune.py --config config.yaml --mode quick
```

第四步：批量调参

```bash
python train_and_tune.py --config config.yaml --mode tune
```

第五步：最终训练

```bash
python train_and_tune.py --config config.yaml --mode final
```

第六步：测试集评估

```bash
python train_and_tune.py \
  --config config.yaml \
  --mode test \
  --checkpoint outputs/checkpoints/Exp_FINAL_UNET/best_model.pth
```

第七步：生成报告图片

```bash
python visualize_results.py --config config.yaml
```

## Windows 本地检查

Windows 上建议先运行：

```powershell
python inspect_dataset.py --config config.yaml
python train_and_tune.py --config config.yaml --mode quick
```

如果本地没有 CUDA，训练脚本会自动退回 CPU。CPU 只适合验证代码流程，不适合完整调参和最终训练。

## AutoDL 正式训练

上传项目到 AutoDL 后，确认数据仍在项目根目录的 `data/` 下，然后按“运行顺序”依次执行。RTX 5090 默认可使用 AMP 混合精度，`config.yaml` 中默认：

- `batch_size_256: 16`
- `batch_size_512: 8`
- `num_workers: 4`
- `use_amp: true`

如果显存不足，优先降低 `batch_size_512`，或者把实验输入尺寸改为 `256`。

## 输出文件说明

每个实验会生成独立目录：

```text
outputs/logs/<experiment_name>/history.csv
outputs/checkpoints/<experiment_name>/best_model.pth
outputs/checkpoints/<experiment_name>/last_model.pth
outputs/summaries/<experiment_name>/config.json
outputs/summaries/<experiment_name>/summary.json
outputs/predictions/<experiment_name>/
```

调参完成后会生成：

```text
outputs/summaries/best_tuning_config.json
```

报告图片会保存到：

```text
report_assets/dataset_examples/
report_assets/curves/
report_assets/bars/
report_assets/qualitative/
report_assets/failure_cases/
```

## 常见问题

如果训练太慢，降低 `tuning.subset_ratio` 或先只运行 `quick` 模式。

如果 mask unique values 不在 0 到 9，先运行：

```bash
python inspect_dataset.py --config config.yaml
```

如果 `ColorMasks-FloodNetv1.0` 不存在，不影响训练，只会影响数据样例图中的官方彩色 mask 列。

