from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


CONFIG = {
    "random_seed": 7,
    "window_size": 20,
    "train_positive": 2100,
    "train_negative": 3200,
    "val_positive": 300,
    "val_negative": 500,
    "feature_stride": 2,
    "feature_size_step": 1,
    "max_features": 10000,
    "cascade_stages": 5,
    "weak_learners_per_stage": [10, 16, 24, 32, 40],
    "stage_detection_rate": 0.999,
    "final_min_recall": 0.92,
    "scale_growth": 1.25,
    "max_scale": 8.0,
    "sliding_window_step_ratio": 0.2,
    "nms_iou_threshold": 0.25,
    "cluster_iou_threshold": 0.18,
    "min_cluster_support": 2,
    "max_image_side_for_detection": 360,
    "external_scan_limit": 25,
    "external_showcase_count": 5,
}


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
MODEL_PATH = OUTPUT_DIR / "trained_cascade.json"
REPORT_PATH = OUTPUT_DIR / "report.txt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SCORE_HIST_PATH = OUTPUT_DIR / "score_histogram.png"
ROC_PATH = OUTPUT_DIR / "roc_curve.png"
PR_PATH = OUTPUT_DIR / "pr_curve.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"


def find_patch_dataset_dir(base_dir: Path) -> Path:
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and (path / "faces").is_dir() and (path / "nonfaces").is_dir():
            return path
    raise FileNotFoundError("Could not find the MIT face/nonface dataset directory.")


PATCH_DATASET_DIR = find_patch_dataset_dir(BASE_DIR)
FACES_DIR = PATCH_DATASET_DIR / "faces"
NONFACES_DIR = PATCH_DATASET_DIR / "nonfaces"
CMU_TEST_DIRS = [BASE_DIR / "test", BASE_DIR / "newtest", BASE_DIR / "test-low"]


@dataclass(frozen=True)
class HaarFeature:
    kind: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class WeakClassifier:
    feature_index: int
    threshold: float
    polarity: int
    alpha: float
    error: float


@dataclass
class CascadeStage:
    learners: list[WeakClassifier]
    threshold: float
    train_detection_rate: float
    train_false_positive_rate: float
    val_detection_rate: float
    val_false_positive_rate: float


@dataclass
class TrainedCascade:
    features: list[HaarFeature]
    stages: list[CascadeStage]
    final_threshold: float


def log(message: str) -> None:
    print(message, flush=True)


def cleanup_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ["result_*.png", "cmu_result_*.png", "summary.png", "report.txt", "metrics.json", "trained_cascade.json"]:
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink()


def read_image_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        try:
            image.seek(0)
        except EOFError:
            pass
        return np.array(image.convert("L"))


def read_image_color(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        try:
            image.seek(0)
        except EOFError:
            pass
        rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"failed to encode {path}")
    encoded.tofile(str(path))


def prepare_patch(image: np.ndarray, window_size: int) -> np.ndarray:
    if image.shape[:2] != (window_size, window_size):
        image = cv2.resize(image, (window_size, window_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.equalizeHist(image)
    return image.astype(np.float32)


def load_patch_batch(paths: Iterable[Path], window_size: int) -> np.ndarray:
    return np.stack([prepare_patch(read_image_gray(path), window_size) for path in paths]).astype(np.float32)


def integral_images(images: np.ndarray) -> np.ndarray:
    return np.pad(
        np.cumsum(np.cumsum(images, axis=1), axis=2),
        ((0, 0), (1, 0), (1, 0)),
        mode="constant",
    )


def integral_image(image: np.ndarray) -> np.ndarray:
    return integral_images(image[np.newaxis, ...])[0]


def rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return ii[..., y + h, x + w] - ii[..., y, x + w] - ii[..., y + h, x] + ii[..., y, x]


def evaluate_feature(ii: np.ndarray, feature: HaarFeature) -> np.ndarray:
    x, y, w, h = feature.x, feature.y, feature.w, feature.h
    if feature.kind == "two_x":
        return rect_sum(ii, x, y, w, h) - rect_sum(ii, x + w, y, w, h)
    if feature.kind == "two_y":
        return rect_sum(ii, x, y, w, h) - rect_sum(ii, x, y + h, w, h)
    if feature.kind == "three_x":
        return (
            rect_sum(ii, x, y, w, h)
            - rect_sum(ii, x + w, y, w, h)
            + rect_sum(ii, x + 2 * w, y, w, h)
        )
    if feature.kind == "three_y":
        return (
            rect_sum(ii, x, y, w, h)
            - rect_sum(ii, x, y + h, w, h)
            + rect_sum(ii, x, y + 2 * h, w, h)
        )
    if feature.kind == "four":
        return (
            rect_sum(ii, x, y, w, h)
            - rect_sum(ii, x + w, y, w, h)
            - rect_sum(ii, x, y + h, w, h)
            + rect_sum(ii, x + w, y + h, w, h)
        )
    raise ValueError(f"unknown feature kind: {feature.kind}")


def generate_feature_pool(window_size: int, stride: int, size_step: int) -> list[HaarFeature]:
    templates = {
        "two_x": (2, 1),
        "two_y": (1, 2),
        "three_x": (3, 1),
        "three_y": (1, 3),
        "four": (2, 2),
    }
    features: list[HaarFeature] = []
    for kind, (unit_w, unit_h) in templates.items():
        max_w = window_size // unit_w
        max_h = window_size // unit_h
        for w in range(2, max_w + 1, size_step):
            for h in range(2, max_h + 1, size_step):
                total_w = unit_w * w
                total_h = unit_h * h
                for y in range(0, window_size - total_h + 1, stride):
                    for x in range(0, window_size - total_w + 1, stride):
                        features.append(HaarFeature(kind, x, y, w, h))
    return features


def compute_feature_matrix(integrals_set: np.ndarray, features: list[HaarFeature]) -> np.ndarray:
    matrix = np.zeros((integrals_set.shape[0], len(features)), dtype=np.float32)
    for feature_index, feature in enumerate(features):
        matrix[:, feature_index] = evaluate_feature(integrals_set, feature)
    return matrix


def split_dataset(
    paths: list[Path],
    train_count: int,
    val_count: int,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = paths[:]
    rng.shuffle(shuffled)
    train_end = min(train_count, len(shuffled))
    val_end = min(train_end + val_count, len(shuffled))
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def threshold_from_split(sorted_values: np.ndarray, split_index: int) -> float:
    if split_index <= 0:
        return float(sorted_values[0] - 1e-6)
    if split_index >= len(sorted_values):
        return float(sorted_values[-1] + 1e-6)
    return float((sorted_values[split_index - 1] + sorted_values[split_index]) / 2.0)


def stump_predict(values: np.ndarray, threshold: float, polarity: int) -> np.ndarray:
    if polarity == 1:
        return np.where(values < threshold, 1.0, -1.0)
    return np.where(values >= threshold, 1.0, -1.0)


def find_best_stump(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    used_features: set[int],
) -> WeakClassifier:
    best = WeakClassifier(-1, 0.0, 1, 0.0, float("inf"))
    for feature_index in range(feature_matrix.shape[1]):
        if feature_index in used_features:
            continue

        values = feature_matrix[:, feature_index]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[order]
        sorted_weights = weights[order]

        pos_weights = sorted_weights * (sorted_labels == 1)
        neg_weights = sorted_weights * (sorted_labels == -1)

        cum_pos = np.concatenate(([0.0], np.cumsum(pos_weights)))
        cum_neg = np.concatenate(([0.0], np.cumsum(neg_weights)))
        total_pos = cum_pos[-1]
        total_neg = cum_neg[-1]

        error_left_positive = cum_neg + (total_pos - cum_pos)
        error_right_positive = cum_pos + (total_neg - cum_neg)

        split_left = int(np.argmin(error_left_positive))
        split_right = int(np.argmin(error_right_positive))
        err_left = float(error_left_positive[split_left])
        err_right = float(error_right_positive[split_right])

        if err_left <= err_right:
            error = err_left
            threshold = threshold_from_split(sorted_values, split_left)
            polarity = 1
        else:
            error = err_right
            threshold = threshold_from_split(sorted_values, split_right)
            polarity = -1

        if error < best.error:
            best = WeakClassifier(feature_index, threshold, polarity, 0.0, error)

    return best


def train_adaboost(feature_matrix: np.ndarray, labels: np.ndarray, rounds: int) -> list[WeakClassifier]:
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == -1))
    weights = np.where(labels == 1, 1.0 / (2 * positive_count), 1.0 / (2 * negative_count)).astype(np.float64)

    learners: list[WeakClassifier] = []
    used_features: set[int] = set()

    for round_index in range(rounds):
        learner = find_best_stump(feature_matrix, labels, weights, used_features)
        if learner.feature_index < 0:
            break
        learner.error = max(min(learner.error, 1.0 - 1e-10), 1e-10)
        learner.alpha = 0.5 * math.log((1.0 - learner.error) / learner.error)

        predictions = stump_predict(
            feature_matrix[:, learner.feature_index],
            learner.threshold,
            learner.polarity,
        )
        weights *= np.exp(-learner.alpha * labels * predictions)
        weights /= np.sum(weights)

        learners.append(learner)
        used_features.add(learner.feature_index)
        log(
            f"    round {round_index + 1}/{rounds}: "
            f"feature={learner.feature_index}, error={learner.error:.4f}, alpha={learner.alpha:.4f}"
        )

        if learner.error <= 1e-8:
            break

    return learners


def strong_classifier_scores(feature_matrix: np.ndarray, learners: list[WeakClassifier]) -> np.ndarray:
    scores = np.zeros(feature_matrix.shape[0], dtype=np.float64)
    for learner in learners:
        predictions = stump_predict(
            feature_matrix[:, learner.feature_index],
            learner.threshold,
            learner.polarity,
        )
        scores += learner.alpha * predictions
    return scores


def cascade_margins_from_matrix(feature_matrix: np.ndarray, stages: list[CascadeStage]) -> np.ndarray:
    margins = np.zeros(feature_matrix.shape[0], dtype=np.float64)
    for stage in stages:
        margins += strong_classifier_scores(feature_matrix, stage.learners) - stage.threshold
    return margins


def train_cascade(
    x_pos_train: np.ndarray,
    x_neg_train: np.ndarray,
    x_pos_val: np.ndarray,
    x_neg_val: np.ndarray,
    cfg: dict,
) -> list[CascadeStage]:
    stages: list[CascadeStage] = []
    active_neg_train = np.arange(x_neg_train.shape[0])
    active_neg_val = np.arange(x_neg_val.shape[0])

    for stage_index in range(cfg["cascade_stages"]):
        rounds = cfg["weak_learners_per_stage"][stage_index]
        log(
            f"\n[Stage {stage_index + 1}] positives={x_pos_train.shape[0]}, "
            f"hard_negatives={len(active_neg_train)}, rounds={rounds}"
        )

        stage_train = np.vstack([x_pos_train, x_neg_train[active_neg_train]])
        stage_labels = np.concatenate(
            [np.ones(x_pos_train.shape[0], dtype=np.int32), -np.ones(len(active_neg_train), dtype=np.int32)]
        )

        learners = train_adaboost(stage_train, stage_labels, rounds)
        train_pos_scores = strong_classifier_scores(x_pos_train, learners)
        threshold = float(np.quantile(train_pos_scores, 1.0 - cfg["stage_detection_rate"]))

        train_neg_scores = strong_classifier_scores(x_neg_train[active_neg_train], learners)
        val_pos_scores = strong_classifier_scores(x_pos_val, learners)

        train_det = float(np.mean(train_pos_scores >= threshold))
        train_fp = float(np.mean(train_neg_scores >= threshold)) if len(train_neg_scores) else 0.0
        val_det = float(np.mean(val_pos_scores >= threshold))

        if len(active_neg_val):
            val_neg_scores = strong_classifier_scores(x_neg_val[active_neg_val], learners)
            val_fp = float(np.mean(val_neg_scores >= threshold))
            active_neg_val = active_neg_val[val_neg_scores >= threshold]
        else:
            val_fp = 0.0

        active_neg_train = active_neg_train[train_neg_scores >= threshold]

        stages.append(
            CascadeStage(
                learners=learners,
                threshold=threshold,
                train_detection_rate=train_det,
                train_false_positive_rate=train_fp,
                val_detection_rate=val_det,
                val_false_positive_rate=val_fp,
            )
        )

        log(
            f"  threshold={threshold:.4f}, "
            f"train_det={train_det:.4f}, train_fp={train_fp:.4f}, "
            f"val_det={val_det:.4f}, val_fp={val_fp:.4f}, "
            f"remaining_hard_negatives={len(active_neg_train)}"
        )

        if len(active_neg_train) == 0:
            break

    return stages


def calibrate_final_threshold(pos_margins: np.ndarray, neg_margins: np.ndarray, min_recall: float) -> float:
    candidate_thresholds = sorted(set(np.round(np.concatenate([pos_margins, neg_margins]), 4)))
    selected = float(np.min(pos_margins))
    best_key: tuple[float, float, float] | None = None

    for threshold in candidate_thresholds:
        recall = float(np.mean(pos_margins >= threshold))
        if recall < min_recall:
            continue
        fpr = float(np.mean(neg_margins >= threshold))
        tp = int(np.sum(pos_margins >= threshold))
        fp = int(np.sum(neg_margins >= threshold))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        key = (-fpr, precision, threshold)
        if best_key is None or key > best_key:
            best_key = key
            selected = float(threshold)

    val_fp = float(np.mean(neg_margins >= selected))
    val_recall = float(np.mean(pos_margins >= selected))
    log(
        f"final threshold={selected:.4f}, "
        f"val_recall={val_recall:.4f}, val_fp={val_fp:.4f}"
    )
    return selected


def evaluate_stage_on_patch(ii: np.ndarray, features: list[HaarFeature], stage: CascadeStage) -> tuple[bool, float]:
    score = 0.0
    for learner in stage.learners:
        value = float(evaluate_feature(ii, features[learner.feature_index]))
        if learner.polarity == 1:
            prediction = 1.0 if value < learner.threshold else -1.0
        else:
            prediction = 1.0 if value >= learner.threshold else -1.0
        score += learner.alpha * prediction
    margin = score - stage.threshold
    return margin >= 0.0, margin


def cascade_predict(ii: np.ndarray, features: list[HaarFeature], stages: list[CascadeStage]) -> tuple[bool, float]:
    total_margin = 0.0
    for stage in stages:
        passed, margin = evaluate_stage_on_patch(ii, features, stage)
        total_margin += margin
        if not passed:
            return False, total_margin
    return True, total_margin


def evaluate_patch_metrics(
    x_pos_test: np.ndarray,
    x_neg_test: np.ndarray,
    model: TrainedCascade,
) -> dict[str, float | int]:
    pos_margins = cascade_margins_from_matrix(x_pos_test, model.stages)
    neg_margins = cascade_margins_from_matrix(x_neg_test, model.stages)

    tp = int(np.sum(pos_margins >= model.final_threshold))
    fn = int(np.sum(pos_margins < model.final_threshold))
    fp = int(np.sum(neg_margins >= model.final_threshold))
    tn = int(np.sum(neg_margins < model.final_threshold))

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    true_negative_rate = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_negative_rate": true_negative_rate,
    }


def generate_scales(height: int, width: int, cfg: dict) -> list[float]:
    max_scale = min(cfg["max_scale"], min(height, width) / cfg["window_size"])
    scales: list[float] = []
    scale = 1.0
    while scale <= max_scale + 1e-9:
        scales.append(scale)
        scale *= cfg["scale_growth"]
    return scales


def non_max_suppression(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []

    array = np.array(boxes, dtype=np.float32)
    x1 = array[:, 0]
    y1 = array[:, 1]
    x2 = array[:, 0] + array[:, 2]
    y2 = array[:, 1] + array[:, 3]
    scores = array[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][iou <= iou_threshold]

    return [boxes[index] for index in keep]


def box_iou(box_a: tuple[int, int, int, int, float], box_b: tuple[int, int, int, int, float]) -> float:
    ax1, ay1, aw, ah, _ = box_a
    bx1, by1, bw, bh, _ = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    inter_w = max(0, xx2 - xx1)
    inter_h = max(0, yy2 - yy1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def cluster_supported_detections(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float,
    min_support: int,
) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []

    clusters: list[list[tuple[int, int, int, int, float]]] = []
    sorted_boxes = sorted(boxes, key=lambda item: item[4], reverse=True)

    for box in sorted_boxes:
        assigned = False
        for cluster in clusters:
            if max(box_iou(box, member) for member in cluster) >= iou_threshold:
                cluster.append(box)
                assigned = True
                break
        if not assigned:
            clusters.append([box])

    fused: list[tuple[int, int, int, int, float]] = []
    for cluster in clusters:
        if len(cluster) < min_support:
            continue
        scores = np.array([member[4] for member in cluster], dtype=np.float64)
        weights = np.maximum(scores, 1e-6)
        xs = np.array([member[0] for member in cluster], dtype=np.float64)
        ys = np.array([member[1] for member in cluster], dtype=np.float64)
        ws = np.array([member[2] for member in cluster], dtype=np.float64)
        hs = np.array([member[3] for member in cluster], dtype=np.float64)
        fused_score = float(np.max(scores) + 0.15 * (len(cluster) - 1))
        fused.append(
            (
                int(round(np.average(xs, weights=weights))),
                int(round(np.average(ys, weights=weights))),
                int(round(np.average(ws, weights=weights))),
                int(round(np.average(hs, weights=weights))),
                fused_score,
            )
        )
    return fused


def resize_for_detection(color: np.ndarray, cfg: dict) -> np.ndarray:
    height, width = color.shape[:2]
    max_side = max(height, width)
    if max_side <= cfg["max_image_side_for_detection"]:
        return color
    scale = cfg["max_image_side_for_detection"] / max_side
    return cv2.resize(color, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def detect_faces_in_image(
    gray: np.ndarray,
    model: TrainedCascade,
    cfg: dict,
) -> list[tuple[int, int, int, int, float]]:
    raw_detections: list[tuple[int, int, int, int, float]] = []

    for scale in generate_scales(gray.shape[0], gray.shape[1], cfg):
        win = max(cfg["window_size"], int(round(cfg["window_size"] * scale)))
        if win > gray.shape[0] or win > gray.shape[1]:
            continue
        step = max(4, int(round(win * cfg["sliding_window_step_ratio"])))
        for y in range(0, gray.shape[0] - win + 1, step):
            for x in range(0, gray.shape[1] - win + 1, step):
                patch = gray[y : y + win, x : x + win]
                patch = prepare_patch(patch, cfg["window_size"])
                ii = integral_image(patch)
                passed, score = cascade_predict(ii, model.features, model.stages)
                if passed and score >= model.final_threshold:
                    raw_detections.append((x, y, win, win, float(score)))

    clustered = cluster_supported_detections(
        raw_detections,
        cfg["cluster_iou_threshold"],
        cfg["min_cluster_support"],
    )
    return non_max_suppression(clustered, cfg["nms_iou_threshold"])


def draw_detections(
    image: np.ndarray,
    detections: list[tuple[int, int, int, int, float]],
) -> np.ndarray:
    result = image.copy()
    for index, (x, y, w, h, score) in enumerate(detections, start=1):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            result,
            f"{index}:{score:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        result,
        f"detected={len(detections)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def scan_external_test_images() -> list[Path]:
    extensions = {".gif", ".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
    candidates: list[Path] = []
    for folder in CMU_TEST_DIRS:
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in extensions:
                    candidates.append(path)
    return candidates


def create_summary_image(result_paths: list[Path], output_path: Path) -> None:
    if not result_paths:
        return

    images = []
    for path in result_paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_h = 260
        scale = target_h / image.shape[0]
        resized = cv2.resize(image, (int(image.shape[1] * scale), target_h), interpolation=cv2.INTER_LINEAR)
        images.append(resized)

    if not images:
        return

    rows = []
    per_row = 2
    for index in range(0, len(images), per_row):
        row_images = images[index : index + per_row]
        row = np.hstack(row_images)
        rows.append(row)

    max_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, padding])
        padded_rows.append(row)

    write_image(output_path, np.vstack(padded_rows))


def save_model(model: TrainedCascade, cfg: dict) -> None:
    data = {
        "config": cfg,
        "final_threshold": model.final_threshold,
        "features": [asdict(feature) for feature in model.features],
        "stages": [
            {
                "threshold": stage.threshold,
                "train_detection_rate": stage.train_detection_rate,
                "train_false_positive_rate": stage.train_false_positive_rate,
                "val_detection_rate": stage.val_detection_rate,
                "val_false_positive_rate": stage.val_false_positive_rate,
                "learners": [asdict(learner) for learner in stage.learners],
            }
            for stage in model.stages
        ],
    }
    MODEL_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_report(
    cfg: dict,
    total_faces: int,
    total_nonfaces: int,
    feature_pool_size: int,
    selected_feature_count: int,
    model: TrainedCascade,
    patch_test_counts: tuple[int, int],
    patch_metrics: dict[str, float | int],
    cmu_result_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    lines = [
        "Lab 5 report",
        "==============================",
        "Training dataset: MIT face/nonface patch dataset",
        f"Positive samples (all): {total_faces}",
        f"Negative samples (all): {total_nonfaces}",
        f"Patch size: {cfg['window_size']}x{cfg['window_size']}",
        f"Haar feature pool size: {feature_pool_size}",
        f"Selected Haar features: {selected_feature_count}",
        f"Cascade stages: {len(model.stages)}",
        f"Final threshold: {model.final_threshold:.4f}",
        "",
        "Held-out MIT patch test split:",
        f"  Positive test patches: {patch_test_counts[0]}",
        f"  Negative test patches: {patch_test_counts[1]}",
        f"  Accuracy: {patch_metrics['accuracy']:.4f}",
        f"  Precision: {patch_metrics['precision']:.4f}",
        f"  Recall: {patch_metrics['recall']:.4f}",
        f"  F1: {patch_metrics['f1']:.4f}",
        f"  False positive rate: {patch_metrics['false_positive_rate']:.4f}",
        f"  Confusion matrix: TP={patch_metrics['tp']}, FN={patch_metrics['fn']}, FP={patch_metrics['fp']}, TN={patch_metrics['tn']}",
        "  Interpretation: these metrics are measured on isolated 20x20 face/non-face patches.",
        "  Scene-level sliding-window detection is harder because one image produces many candidate windows,",
        "  so even a low patch-level false positive rate can still create several false alarms on a full image.",
        "",
        "External visualization dataset:",
        "  CMU frontal_images (folders: test, newtest, test-low)",
        "  Note: used only for qualitative face detection visualization, not for patch-level classification metrics.",
        f"  Visualization results saved: {len(cmu_result_paths)}",
        "",
        "Generated charts:",
        f"  - {SCORE_HIST_PATH.name}",
        f"  - {ROC_PATH.name}",
        f"  - {PR_PATH.name}",
        f"  - {CONFUSION_MATRIX_PATH.name}",
    ]

    for path in cmu_result_paths:
        lines.append(f"  - {path.name}")

    lines.extend(
        [
            "",
            "Per-stage summary:",
        ]
    )

    for stage_index, stage in enumerate(model.stages, start=1):
        lines.append(
            f"  Stage {stage_index}: learners={len(stage.learners)}, "
            f"threshold={stage.threshold:.4f}, train_det={stage.train_detection_rate:.4f}, "
            f"train_fp={stage.train_false_positive_rate:.4f}, val_det={stage.val_detection_rate:.4f}, "
            f"val_fp={stage.val_false_positive_rate:.4f}"
        )

    lines.extend(
        [
            "",
            f"Elapsed time (seconds): {elapsed_seconds:.2f}",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_metrics_json(
    patch_metrics: dict[str, float | int],
    test_positive_count: int,
    test_negative_count: int,
    cmu_result_paths: list[Path],
) -> None:
    METRICS_PATH.write_text(
        json.dumps(
            {
                "patch_test_positive_count": test_positive_count,
                "patch_test_negative_count": test_negative_count,
                "patch_metrics": patch_metrics,
                "chart_files": [
                    SCORE_HIST_PATH.name,
                    ROC_PATH.name,
                    PR_PATH.name,
                    CONFUSION_MATRIX_PATH.name,
                ],
                "cmu_result_images": [path.name for path in cmu_result_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def generate_analysis_charts(
    pos_margins: np.ndarray,
    neg_margins: np.ndarray,
    model: TrainedCascade,
    patch_metrics: dict[str, float | int],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    labels = np.concatenate([np.ones(len(pos_margins)), np.zeros(len(neg_margins))])
    scores = np.concatenate([pos_margins, neg_margins])

    plt.figure(figsize=(8, 5))
    plt.hist(pos_margins, bins=40, alpha=0.65, label="face patches", color="#2ca02c")
    plt.hist(neg_margins, bins=40, alpha=0.65, label="non-face patches", color="#d62728")
    plt.axvline(model.final_threshold, color="#1f77b4", linestyle="--", linewidth=2, label="final threshold")
    plt.xlabel("Cascade margin score")
    plt.ylabel("Count")
    plt.title("Held-out MIT Patch Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCORE_HIST_PATH, dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Held-out MIT Patch Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="#ff7f0e", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Held-out MIT Patch Test Set")
    plt.tight_layout()
    plt.savefig(PR_PATH, dpi=160)
    plt.close()

    confusion = np.array(
        [
            [patch_metrics["tp"], patch_metrics["fn"]],
            [patch_metrics["fp"], patch_metrics["tn"]],
        ],
        dtype=np.int32,
    )
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.xticks([0, 1], ["Pred face", "Pred non-face"])
    plt.yticks([0, 1], ["True face", "True non-face"])
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            plt.text(col, row, str(confusion[row, col]), ha="center", va="center", color="black", fontsize=12)
    plt.title("Confusion Matrix on Held-out MIT Patch Test Set")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def select_cmu_showcase_images(model: TrainedCascade, cfg: dict) -> list[Path]:
    candidates = scan_external_test_images()
    selected_with_faces: list[tuple[Path, np.ndarray, list[tuple[int, int, int, int, float]]]] = []
    selected_without_faces: list[tuple[Path, np.ndarray, list[tuple[int, int, int, int, float]]]] = []

    for path in candidates[: cfg["external_scan_limit"]]:
        color = resize_for_detection(read_image_color(path), cfg)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = detect_faces_in_image(gray, model, cfg)
        record = (path, color, detections)
        if detections:
            selected_with_faces.append(record)
        elif len(selected_without_faces) < cfg["external_showcase_count"]:
            selected_without_faces.append(record)

    selected_with_faces.sort(
        key=lambda item: (
            len(item[2]),
            -max((det[4] for det in item[2]), default=0.0),
            item[0].name,
        )
    )

    chosen = selected_with_faces[: cfg["external_showcase_count"]]
    if len(chosen) < cfg["external_showcase_count"]:
        chosen.extend(selected_without_faces[: cfg["external_showcase_count"] - len(chosen)])

    result_paths: list[Path] = []
    for index, (source_path, color, detections) in enumerate(chosen, start=1):
        rendered = draw_detections(color, detections)
        save_path = OUTPUT_DIR / f"cmu_result_{index:02d}_{source_path.stem}.png"
        write_image(save_path, rendered)
        result_paths.append(save_path)
        log(f"  {source_path.name}: detections={len(detections)} -> {save_path.name}")

    return result_paths


def main() -> None:
    start_time = time.perf_counter()
    cleanup_outputs()

    rng = random.Random(CONFIG["random_seed"])
    face_paths = sorted(FACES_DIR.glob("*.bmp"))
    nonface_paths = sorted(NONFACES_DIR.glob("*.bmp"))

    log("=" * 60)
    log("Self-trained Haar + AdaBoost Face Detector")
    log("=" * 60)
    log(f"training patch dataset: {PATCH_DATASET_DIR}")
    log(f"faces={len(face_paths)}, nonfaces={len(nonface_paths)}")

    train_faces, val_faces, test_faces = split_dataset(
        face_paths, CONFIG["train_positive"], CONFIG["val_positive"], rng
    )
    train_nonfaces, val_nonfaces, test_nonfaces = split_dataset(
        nonface_paths, CONFIG["train_negative"], CONFIG["val_negative"], rng
    )

    log("\n[1/6] Loading train/validation/test patch splits ...")
    pos_train_images = load_patch_batch(train_faces, CONFIG["window_size"])
    neg_train_images = load_patch_batch(train_nonfaces, CONFIG["window_size"])
    pos_val_images = load_patch_batch(val_faces, CONFIG["window_size"])
    neg_val_images = load_patch_batch(val_nonfaces, CONFIG["window_size"])
    pos_test_images = load_patch_batch(test_faces, CONFIG["window_size"])
    neg_test_images = load_patch_batch(test_nonfaces, CONFIG["window_size"])

    log("[2/6] Building Haar feature pool ...")
    full_feature_pool = generate_feature_pool(
        CONFIG["window_size"],
        CONFIG["feature_stride"],
        CONFIG["feature_size_step"],
    )
    selected_features = (
        rng.sample(full_feature_pool, CONFIG["max_features"])
        if len(full_feature_pool) > CONFIG["max_features"]
        else full_feature_pool
    )
    log(f"feature pool size={len(full_feature_pool)}, selected={len(selected_features)}")

    log("[3/6] Computing integral images and feature matrices ...")
    x_pos_train = compute_feature_matrix(integral_images(pos_train_images), selected_features)
    x_neg_train = compute_feature_matrix(integral_images(neg_train_images), selected_features)
    x_pos_val = compute_feature_matrix(integral_images(pos_val_images), selected_features)
    x_neg_val = compute_feature_matrix(integral_images(neg_val_images), selected_features)
    x_pos_test = compute_feature_matrix(integral_images(pos_test_images), selected_features)
    x_neg_test = compute_feature_matrix(integral_images(neg_test_images), selected_features)

    log("[4/6] Training cascade ...")
    stages = train_cascade(x_pos_train, x_neg_train, x_pos_val, x_neg_val, CONFIG)
    val_pos_margins = cascade_margins_from_matrix(x_pos_val, stages)
    val_neg_margins = cascade_margins_from_matrix(x_neg_val, stages)
    final_threshold = calibrate_final_threshold(
        val_pos_margins,
        val_neg_margins,
        CONFIG["final_min_recall"],
    )
    model = TrainedCascade(selected_features, stages, final_threshold)
    save_model(model, CONFIG)
    log(f"model saved to: {MODEL_PATH.name}")

    log("[5/6] Evaluating held-out MIT patch test split ...")
    patch_metrics = evaluate_patch_metrics(x_pos_test, x_neg_test, model)
    log(
        "  patch metrics: "
        f"acc={patch_metrics['accuracy']:.4f}, "
        f"precision={patch_metrics['precision']:.4f}, "
        f"recall={patch_metrics['recall']:.4f}, "
        f"f1={patch_metrics['f1']:.4f}, "
        f"fpr={patch_metrics['false_positive_rate']:.4f}"
    )
    pos_test_margins = cascade_margins_from_matrix(x_pos_test, model.stages)
    neg_test_margins = cascade_margins_from_matrix(x_neg_test, model.stages)
    generate_analysis_charts(pos_test_margins, neg_test_margins, model, patch_metrics)

    log("[6/6] Running external visualization on CMU frontal_images ...")
    cmu_result_paths = select_cmu_showcase_images(model, CONFIG)
    create_summary_image(cmu_result_paths, OUTPUT_DIR / "summary.png")

    elapsed_seconds = time.perf_counter() - start_time
    write_report(
        CONFIG,
        len(face_paths),
        len(nonface_paths),
        len(full_feature_pool),
        len(selected_features),
        model,
        (len(test_faces), len(test_nonfaces)),
        patch_metrics,
        cmu_result_paths,
        elapsed_seconds,
    )
    save_metrics_json(patch_metrics, len(test_faces), len(test_nonfaces), cmu_result_paths)

    log("\nFinished.")
    log(f"summary image: {OUTPUT_DIR / 'summary.png'}")
    log(f"report file: {REPORT_PATH}")
    log(f"elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
