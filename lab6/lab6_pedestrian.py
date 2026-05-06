import csv
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import auc, roc_curve


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
cv2.setRNGSeed(RANDOM_SEED)

WINDOW_WIDTH = 64
WINDOW_HEIGHT = 128
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
NEG_SAMPLES_PER_IMAGE = 2
TOP_RESULT_COUNT = 15

BASE_DIR = Path(r"INRIADATA (2)\INRIADATA")
TRAIN_POS_DIR = BASE_DIR / r"normalized_images\train\pos"
TRAIN_NEG_DIR = BASE_DIR / r"normalized_images\train\neg"
TEST_POS_DIR = BASE_DIR / r"original_images\test\pos"
TEST_NEG_DIR = BASE_DIR / r"original_images\test\neg"
TEST_ANN_DIR = BASE_DIR / r"original_images\test\annotations"

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
DEMO_DIR = OUTPUT_DIR / "demo"
TABLE_DIR = OUTPUT_DIR / "tables"


@dataclass(frozen=True)
class HogSvmParams:
    win_stride: int
    block_size: int
    block_stride: int
    cell_size: int
    nbins: int
    c_value: float


def ensure_dirs():
    for directory in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, DEMO_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_gray_image(path):
    return np.asarray(Image.open(str(path)).convert("L"))


def load_bgr_image(path):
    rgb = np.asarray(Image.open(str(path)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_hog_descriptor(params):
    return cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(params.block_size, params.block_size),
        _blockStride=(params.block_stride, params.block_stride),
        _cellSize=(params.cell_size, params.cell_size),
        _nbins=params.nbins,
    )


def center_crop_positive(image):
    height, width = image.shape[:2]
    left = max((width - WINDOW_WIDTH) // 2, 0)
    top = max((height - WINDOW_HEIGHT) // 2, 0)
    return image[top : top + WINDOW_HEIGHT, left : left + WINDOW_WIDTH]


def sample_negative_crops(image, samples_per_image, rng):
    height, width = image.shape[:2]
    if width < WINDOW_WIDTH or height < WINDOW_HEIGHT:
        return []
    crops = []
    max_x = width - WINDOW_WIDTH
    max_y = height - WINDOW_HEIGHT
    for _ in range(samples_per_image):
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        crops.append(image[y : y + WINDOW_HEIGHT, x : x + WINDOW_WIDTH])
    return crops


def compute_hog_descriptor(hog, image):
    if image.shape[1] != WINDOW_WIDTH or image.shape[0] != WINDOW_HEIGHT:
        image = cv2.resize(image, WINDOW_SIZE)
    return hog.compute(image).reshape(-1)


def build_training_windows(train_pos_paths, train_neg_paths):
    pos_windows = []
    neg_windows = []

    for path in train_pos_paths:
        image = load_gray_image(path)
        crop = center_crop_positive(image)
        if crop.shape[:2] == (WINDOW_HEIGHT, WINDOW_WIDTH):
            pos_windows.append(crop)

    for neg_index, path in enumerate(train_neg_paths):
        image = load_gray_image(path)
        rng = random.Random(RANDOM_SEED + neg_index)
        neg_windows.extend(sample_negative_crops(image, NEG_SAMPLES_PER_IMAGE, rng))

    windows = pos_windows + neg_windows
    labels = np.asarray([1] * len(pos_windows) + [0] * len(neg_windows), dtype=np.int32)
    return windows, labels


def parse_annotation_boxes(annotation_path):
    pattern = re.compile(
        r"Bounding box.*:\s*\(([-\d]+),\s*([-\d]+)\)\s*-\s*\(([-\d]+),\s*([-\d]+)\)"
    )
    text = annotation_path.read_text(encoding="latin1")
    boxes = []
    for match in pattern.finditer(text):
        xmin, ymin, xmax, ymax = map(int, match.groups())
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def crop_box_with_aspect(image, box):
    xmin, ymin, xmax, ymax = box
    img_h, img_w = image.shape[:2]

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)

    width = max(1, xmax - xmin + 1)
    height = max(1, ymax - ymin + 1)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    target_ratio = WINDOW_WIDTH / float(WINDOW_HEIGHT)
    current_ratio = width / float(height)
    if current_ratio < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio

    width *= 1.1
    height *= 1.1

    left = max(0, int(round(center_x - width / 2.0)))
    top = max(0, int(round(center_y - height / 2.0)))
    right = min(img_w, int(round(center_x + width / 2.0)))
    bottom = min(img_h, int(round(center_y + height / 2.0)))

    if right - left < 8 or bottom - top < 8:
        return None

    patch = image[top:bottom, left:right]
    return cv2.resize(patch, WINDOW_SIZE)


def build_test_windows():
    windows = []
    labels = []

    for ann_path in sorted(TEST_ANN_DIR.glob("*.txt")):
        image_path = TEST_POS_DIR / (ann_path.stem + ".png")
        image = load_gray_image(image_path)
        for box in parse_annotation_boxes(ann_path):
            crop = crop_box_with_aspect(image, box)
            if crop is not None:
                windows.append(crop)
                labels.append(1)

    for neg_index, image_path in enumerate(sorted(TEST_NEG_DIR.glob("*.png"))):
        image = load_gray_image(image_path)
        rng = random.Random(1000 + RANDOM_SEED + neg_index)
        for crop in sample_negative_crops(image, NEG_SAMPLES_PER_IMAGE, rng):
            windows.append(crop)
            labels.append(0)

    return windows, np.asarray(labels, dtype=np.int32)


def extract_features(windows, params):
    hog = create_hog_descriptor(params)
    return np.asarray([compute_hog_descriptor(hog, window) for window in windows], dtype=np.float32)


def train_svm(x_train, y_train, params):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(params.c_value)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm


def decision_function(svm, x):
    _, raw = svm.predict(x, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    return (-raw.reshape(-1)).astype(np.float32)


def build_param_grid():
    grid = []
    c_values = [0.001, 0.01, 0.1, 1.0]
    bin_values = [6, 9, 12]

    for cell_size, block_options in [(4, [8, 16]), (8, [16, 32])]:
        for block_size in block_options:
            stride_options = sorted(set([cell_size, block_size // 2]))
            for block_stride in stride_options:
                if block_stride >= block_size:
                    continue
                for nbins in bin_values:
                    for c_value in c_values:
                        grid.append(
                            HogSvmParams(
                                win_stride=8,
                                block_size=block_size,
                                block_stride=block_stride,
                                cell_size=cell_size,
                                nbins=nbins,
                                c_value=c_value,
                            )
                        )
    return grid


def evaluate_grid(params_list, train_windows, train_labels, test_windows, test_labels):
    results = []
    for index, params in enumerate(params_list, start=1):
        x_train = extract_features(train_windows, params)
        x_test = extract_features(test_windows, params)
        svm = train_svm(x_train, train_labels, params)
        scores = decision_function(svm, x_test)
        fpr, tpr, _ = roc_curve(test_labels, scores)
        roc_auc = auc(fpr, tpr)
        results.append({"index": index, "params": params, "auc": float(roc_auc)})
        print("[{}/{}] test_auc={:.4f} params={}".format(index, len(params_list), roc_auc, params))
    results.sort(key=lambda item: item["auc"], reverse=True)
    return results


def export_linear_detector(svm):
    support = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    return np.append(-support[0], rho).astype(np.float32)


def evaluate_on_test(params, train_windows, train_labels, test_windows, test_labels, model_name):
    x_train = extract_features(train_windows, params)
    svm = train_svm(x_train, train_labels, params)

    x_test = extract_features(test_windows, params)
    scores = decision_function(svm, x_test)
    fpr, tpr, thresholds = roc_curve(test_labels, scores)
    roc_auc = auc(fpr, tpr)

    detector = export_linear_detector(svm)
    model_path = MODEL_DIR / "{}.pkl".format(model_name)
    with model_path.open("wb") as handle:
        pickle.dump({"params": params, "detector": detector}, handle)

    return {
        "model": svm,
        "detector": detector,
        "params": params,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(roc_auc),
        "model_path": model_path,
        "test_size": int(len(test_labels)),
        "test_positives": int(test_labels.sum()),
        "test_negatives": int((1 - test_labels).sum()),
    }


def save_grid_results(results):
    csv_path = TABLE_DIR / "grid_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "test_auc", "cell_size", "block_size", "block_stride", "nbins", "C"])
        for rank, item in enumerate(results, start=1):
            params = item["params"]
            writer.writerow(
                [
                    rank,
                    "{:.6f}".format(item["auc"]),
                    params.cell_size,
                    params.block_size,
                    params.block_stride,
                    params.nbins,
                    params.c_value,
                ]
            )
    return csv_path


def save_nbins_c_heatmaps(results):
    grouped = defaultdict(dict)
    for item in results:
        params = item["params"]
        key = (params.cell_size, params.block_size, params.block_stride)
        grouped[key][(params.nbins, params.c_value)] = item["auc"]

    heatmap_paths = []
    c_values = [0.001, 0.01, 0.1, 1.0]
    bin_values = [6, 9, 12]

    for key in sorted(grouped):
        cell_size, block_size, block_stride = key
        matrix = np.zeros((len(bin_values), len(c_values)), dtype=np.float32)
        for row, nbins in enumerate(bin_values):
            for col, c_value in enumerate(c_values):
                matrix[row, col] = grouped[key][(nbins, c_value)]

        plt.figure(figsize=(6, 4.5))
        image = plt.imshow(matrix, cmap="viridis", aspect="auto")
        plt.colorbar(image, label="Test AUC")
        plt.xticks(np.arange(len(c_values)), [str(value) for value in c_values])
        plt.yticks(np.arange(len(bin_values)), [str(value) for value in bin_values])
        plt.xlabel("C")
        plt.ylabel("nbins")
        plt.title("cell={} block={} stride={}".format(cell_size, block_size, block_stride))
        for row in range(len(bin_values)):
            for col in range(len(c_values)):
                plt.text(col, row, "{:.4f}".format(matrix[row, col]), ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        heatmap_path = PLOT_DIR / "heatmap_cell{}_block{}_stride{}.png".format(cell_size, block_size, block_stride)
        plt.savefig(str(heatmap_path), dpi=160)
        plt.close()
        heatmap_paths.append(heatmap_path)

    return heatmap_paths


def _collect_param_stats(results, attr_name):
    grouped = defaultdict(list)
    for item in results:
        grouped[getattr(item["params"], attr_name)].append(item["auc"])
    keys = sorted(grouped)
    mean_values = [float(np.mean(grouped[key])) for key in keys]
    best_values = [float(np.max(grouped[key])) for key in keys]
    return keys, mean_values, best_values


def save_parameter_summary_plot(results, attr_name, title, output_name):
    keys, mean_values, best_values = _collect_param_stats(results, attr_name)
    positions = np.arange(len(keys))
    width = 0.36

    plt.figure(figsize=(7, 4.6))
    plt.bar(positions - width / 2, mean_values, width=width, label="Mean AUC", color="#4C78A8")
    plt.bar(positions + width / 2, best_values, width=width, label="Best AUC", color="#F58518")
    plt.xticks(positions, [str(key) for key in keys])
    plt.xlabel(attr_name)
    plt.ylabel("AUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path = PLOT_DIR / output_name
    plt.savefig(str(output_path), dpi=160)
    plt.close()
    return output_path


def save_structure_heatmap(results):
    block_sizes = sorted({item["params"].block_size for item in results})
    block_strides = sorted({item["params"].block_stride for item in results})
    grouped = defaultdict(list)

    for item in results:
        key = (item["params"].block_size, item["params"].block_stride)
        grouped[key].append(item["auc"])

    matrix = np.full((len(block_sizes), len(block_strides)), np.nan, dtype=np.float32)
    for row, block_size in enumerate(block_sizes):
        for col, block_stride in enumerate(block_strides):
            values = grouped.get((block_size, block_stride), [])
            if values:
                matrix[row, col] = float(np.mean(values))

    plt.figure(figsize=(6.2, 4.8))
    image = plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(image, label="Mean Test AUC")
    plt.xticks(np.arange(len(block_strides)), [str(value) for value in block_strides])
    plt.yticks(np.arange(len(block_sizes)), [str(value) for value in block_sizes])
    plt.xlabel("block_stride")
    plt.ylabel("block_size")
    plt.title("Block Size vs Block Stride")
    for row in range(len(block_sizes)):
        for col in range(len(block_strides)):
            if not np.isnan(matrix[row, col]):
                plt.text(col, row, "{:.4f}".format(matrix[row, col]), ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    output_path = PLOT_DIR / "block_structure_heatmap.png"
    plt.savefig(str(output_path), dpi=160)
    plt.close()
    return output_path


def save_roc_plot(baseline_result, tuned_result):
    plt.figure(figsize=(7, 6))
    plt.plot(
        baseline_result["fpr"],
        baseline_result["tpr"],
        label="Baseline AUC={:.4f}".format(baseline_result["auc"]),
    )
    plt.plot(
        tuned_result["fpr"],
        tuned_result["tpr"],
        label="Best Grid AUC={:.4f}".format(tuned_result["auc"]),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("INRIA HOG + OpenCV Linear SVM ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_path = PLOT_DIR / "roc_curve.png"
    plt.savefig(str(plot_path), dpi=160)
    plt.close()
    return plot_path


def save_detection_demo(result):
    image_path = sorted(TEST_POS_DIR.glob("*.png"))[0]
    image = load_bgr_image(image_path)

    hog = create_hog_descriptor(result["params"])
    hog.setSVMDetector(result["detector"])

    rects, weights = hog.detectMultiScale(
        image,
        hitThreshold=0.0,
        winStride=(result["params"].win_stride, result["params"].win_stride),
        padding=(8, 8),
        scale=1.05,
    )

    for (x, y, w, h), score in zip(rects[:8], weights[:8]):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            image,
            "{:.2f}".format(float(score)),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path = DEMO_DIR / "detection_demo.png"
    cv2.imwrite(str(output_path), image)
    return output_path, image_path


def format_param_line(params):
    return "cell={}, block={}, stride={}, bins={}, C={}".format(
        params.cell_size,
        params.block_size,
        params.block_stride,
        params.nbins,
        params.c_value,
    )


def save_summary(
    grid_results,
    baseline_result,
    tuned_result,
    roc_path,
    demo_path,
    demo_source,
    nbins_c_heatmaps,
    summary_plot_paths,
    structure_heatmap_path,
    csv_path,
):
    lines = []
    lines.append("# 实验6 行人检测")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append("- HOG 实现：OpenCV `cv2.HOGDescriptor`。")
    lines.append("- 分类器：OpenCV `cv2.ml.SVM` 线性核。")
    lines.append("- 本次实验只保留 `train` 和 `test` 两部分数据，不再划分验证集。")
    lines.append("- 参数组合直接按照测试集 AUC 排序。")
    lines.append("- 参数网格总数：{} 组。".format(len(grid_results)))
    lines.append("")
    lines.append("## 2. 网格搜索说明")
    lines.append("")
    lines.append("- `cell_size ∈ {4, 8}`。")
    lines.append("- `block_size ∈ {8, 16}` 当 `cell=4`；`block_size ∈ {16, 32}` 当 `cell=8`。")
    lines.append("- `block_stride ∈ {cell_size, block_size / 2}`。")
    lines.append("- `nbins ∈ {6, 9, 12}`。")
    lines.append("- `C ∈ {0.001, 0.01, 0.1, 1.0}`。")
    lines.append("")
    lines.append("## 3. Top-{} 参数组合".format(TOP_RESULT_COUNT))
    lines.append("")
    lines.append("| 排名 | 测试集AUC | cell | block | stride | bins | C |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rank, item in enumerate(grid_results[:TOP_RESULT_COUNT], start=1):
        params = item["params"]
        lines.append(
            "| {} | {:.4f} | {} | {} | {} | {} | {} |".format(
                rank,
                item["auc"],
                params.cell_size,
                params.block_size,
                params.block_stride,
                params.nbins,
                params.c_value,
            )
        )
    lines.append("")
    lines.append("完整表格见：`{}`".format(csv_path.as_posix()))
    lines.append("")
    lines.append("## 4. 最终结果")
    lines.append("")
    lines.append("- Baseline：`{}`，测试集 AUC = `{:.4f}`。".format(format_param_line(baseline_result["params"]), baseline_result["auc"]))
    lines.append("- Best Grid：`{}`，测试集 AUC = `{:.4f}`。".format(format_param_line(tuned_result["params"]), tuned_result["auc"]))
    lines.append("- ROC 图：`{}`。".format(roc_path.as_posix()))
    lines.append("- 检测示例图：`{}`，源图像：`{}`。".format(demo_path.as_posix(), demo_source.as_posix()))
    lines.append("")
    lines.append("## 5. 调参过程图像")
    lines.append("")
    for heatmap_path in nbins_c_heatmaps:
        lines.append("- `nbins × C` 热力图：`{}`。".format(heatmap_path.as_posix()))
    for summary_path in summary_plot_paths:
        lines.append("- 参数统计图：`{}`。".format(summary_path.as_posix()))
    lines.append("- `block_size × block_stride` 热力图：`{}`。".format(structure_heatmap_path.as_posix()))
    lines.append("")
    lines.append("## 6. 说明")
    lines.append("")
    lines.append("- 这版流程按作业要求直接使用 train/test。")
    lines.append("- 现在不仅保留了 `nbins × C` 热力图，还增加了 `cell_size`、`block_size`、`block_stride` 的统计图，以及 `block_size × block_stride` 的结构热力图，方便解释参数为什么这样选。")
    lines.append("- 重新运行后，新的图和结果会自动覆盖旧产物。")

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    ensure_dirs()

    train_pos_paths = sorted(TRAIN_POS_DIR.glob("*.png"))
    train_neg_paths = sorted(TRAIN_NEG_DIR.glob("*.png"))
    train_windows, train_labels = build_training_windows(train_pos_paths, train_neg_paths)
    test_windows, test_labels = build_test_windows()

    baseline_params = HogSvmParams(8, 16, 8, 8, 9, 0.01)
    grid_params = build_param_grid()
    grid_results = evaluate_grid(grid_params, train_windows, train_labels, test_windows, test_labels)
    best_params = grid_results[0]["params"]

    csv_path = save_grid_results(grid_results)
    nbins_c_heatmaps = save_nbins_c_heatmaps(grid_results)
    cell_plot = save_parameter_summary_plot(grid_results, "cell_size", "Cell Size vs AUC", "cell_size_summary.png")
    block_plot = save_parameter_summary_plot(grid_results, "block_size", "Block Size vs AUC", "block_size_summary.png")
    stride_plot = save_parameter_summary_plot(grid_results, "block_stride", "Block Stride vs AUC", "block_stride_summary.png")
    structure_heatmap_path = save_structure_heatmap(grid_results)

    baseline_result = evaluate_on_test(
        baseline_params,
        train_windows,
        train_labels,
        test_windows,
        test_labels,
        "baseline_opencv_hog_svm",
    )
    tuned_result = evaluate_on_test(
        best_params,
        train_windows,
        train_labels,
        test_windows,
        test_labels,
        "best_grid_opencv_hog_svm",
    )

    roc_path = save_roc_plot(baseline_result, tuned_result)
    demo_path, demo_source = save_detection_demo(tuned_result)
    report_path = save_summary(
        grid_results,
        baseline_result,
        tuned_result,
        roc_path,
        demo_path,
        demo_source,
        nbins_c_heatmaps,
        [cell_plot, block_plot, stride_plot],
        structure_heatmap_path,
        csv_path,
    )

    print("grid_count={}".format(len(grid_params)))
    print("best_test_auc_from_grid={:.4f}".format(grid_results[0]["auc"]))
    print("baseline_test_auc={:.4f}".format(baseline_result["auc"]))
    print("best_test_auc={:.4f}".format(tuned_result["auc"]))
    print("report={}".format(report_path.resolve()))
    print("roc_plot={}".format(roc_path.resolve()))
    print("demo={}".format(demo_path.resolve()))


if __name__ == "__main__":
    main()
