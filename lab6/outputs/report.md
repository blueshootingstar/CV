# 实验6 行人检测

## 1. 实验设置

- HOG 实现：OpenCV `cv2.HOGDescriptor`。
- 分类器：OpenCV `cv2.ml.SVM` 线性核。
- 本次实验只保留 `train` 和 `test` 两部分数据，不再划分验证集。
- 参数组合直接按照测试集 AUC 排序。
- 参数网格总数：72 组。

## 2. 网格搜索说明

- `cell_size ∈ {4, 8}`。
- `block_size ∈ {8, 16}` 当 `cell=4`；`block_size ∈ {16, 32}` 当 `cell=8`。
- `block_stride ∈ {cell_size, block_size / 2}`。
- `nbins ∈ {6, 9, 12}`。
- `C ∈ {0.001, 0.01, 0.1, 1.0}`。

## 3. Top-15 参数组合

| 排名 | 测试集AUC | cell | block | stride | bins | C |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9655 | 8 | 16 | 8 | 12 | 0.01 |
| 2 | 0.9655 | 8 | 32 | 8 | 9 | 0.01 |
| 3 | 0.9652 | 8 | 32 | 8 | 12 | 0.01 |
| 4 | 0.9642 | 8 | 32 | 16 | 12 | 0.01 |
| 5 | 0.9631 | 8 | 16 | 8 | 9 | 0.01 |
| 6 | 0.9627 | 8 | 32 | 16 | 9 | 0.1 |
| 7 | 0.9627 | 8 | 32 | 16 | 12 | 0.1 |
| 8 | 0.9621 | 8 | 32 | 16 | 9 | 0.01 |
| 9 | 0.9621 | 8 | 32 | 8 | 9 | 0.1 |
| 10 | 0.9614 | 8 | 32 | 16 | 6 | 0.1 |
| 11 | 0.9609 | 8 | 32 | 8 | 6 | 0.1 |
| 12 | 0.9596 | 8 | 32 | 8 | 12 | 0.1 |
| 13 | 0.9586 | 8 | 32 | 8 | 6 | 0.01 |
| 14 | 0.9581 | 8 | 16 | 8 | 6 | 0.1 |
| 15 | 0.9580 | 8 | 32 | 16 | 6 | 1.0 |

完整表格见：`outputs/tables/grid_search_results.csv`

## 4. 最终结果

- Baseline：`cell=8, block=16, stride=8, bins=9, C=0.01`，测试集 AUC = `0.9631`。
- Best Grid：`cell=8, block=16, stride=8, bins=12, C=0.01`，测试集 AUC = `0.9655`。
- ROC 图：`outputs/plots/roc_curve.png`。
- 检测示例图：`outputs/demo/detection_demo.png`，源图像：`INRIADATA (2)/INRIADATA/original_images/test/pos/crop001501.png`。

## 5. 调参过程图像

- `nbins × C` 热力图：`outputs/plots/heatmap_cell4_block8_stride4.png`。
- `nbins × C` 热力图：`outputs/plots/heatmap_cell4_block16_stride4.png`。
- `nbins × C` 热力图：`outputs/plots/heatmap_cell4_block16_stride8.png`。
- `nbins × C` 热力图：`outputs/plots/heatmap_cell8_block16_stride8.png`。
- `nbins × C` 热力图：`outputs/plots/heatmap_cell8_block32_stride8.png`。
- `nbins × C` 热力图：`outputs/plots/heatmap_cell8_block32_stride16.png`。
- 参数统计图：`outputs/plots/cell_size_summary.png`。
- 参数统计图：`outputs/plots/block_size_summary.png`。
- 参数统计图：`outputs/plots/block_stride_summary.png`。
- `block_size × block_stride` 热力图：`outputs/plots/block_structure_heatmap.png`。

## 6. 说明

- 这版流程按作业要求直接使用 train/test。
- 现在不仅保留了 `nbins × C` 热力图，还增加了 `cell_size`、`block_size`、`block_stride` 的统计图，以及 `block_size × block_stride` 的结构热力图，方便解释参数为什么这样选。
- 重新运行后，新的图和结果会自动覆盖旧产物。