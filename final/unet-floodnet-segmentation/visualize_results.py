import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps

from train_and_tune import (
    CLASS_NAMES,
    MEAN,
    PALETTE,
    STD,
    FloodNetDataset,
    UNet,
    build_samples,
    colorize_mask,
    load_config,
    resolve_device,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_report_dirs():
    for path in [
        "report_assets/dataset_examples",
        "report_assets/curves",
        "report_assets/bars",
        "report_assets/qualitative",
        "report_assets/failure_cases",
        "outputs/predictions/Exp_FINAL_UNET",
    ]:
        ensure_dir(path)


def read_history(experiment_name):
    path = Path("outputs/logs") / experiment_name / "history.csv"
    if not path.exists():
        print(f"Skip missing history: {path}")
        return None
    return pd.read_csv(path)


def read_summary(experiment_name):
    path = Path("outputs/summaries") / experiment_name / "summary.json"
    if not path.exists():
        print(f"Skip missing summary: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_best_tuning_config():
    path = Path("outputs/summaries/best_tuning_config.json")
    if not path.exists():
        print(f"Skip missing best tuning config: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_line_plot(series, output_path, title, ylabel):
    if not series:
        print(f"Skip {output_path}: no data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df, column, style in series:
        ax.plot(df["epoch"], df[column], style, label=label, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_bar_plot(labels, values, output_path, title, ylabel, color="#4C78A8"):
    if not labels:
        print(f"Skip {output_path}: no data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_learning_rate_curves():
    lr_exps = [
        ("lr=0.01", "Exp_LR_0_01"),
        ("lr=0.001", "Exp_LR_0_001"),
        ("lr=0.0001", "Exp_LR_0_0001"),
    ]
    histories = [(label, read_history(exp)) for label, exp in lr_exps]
    histories = [(label, df) for label, df in histories if df is not None]

    save_line_plot(
        [(label, df, "val_miou", "-") for label, df in histories],
        "report_assets/curves/lr_val_miou_curve.png",
        "Learning Rate Tuning: val mIoU",
        "val mIoU",
    )
    save_line_plot(
        [(label, df, "val_dice", "-") for label, df in histories],
        "report_assets/curves/lr_val_dice_curve.png",
        "Learning Rate Tuning: val Dice",
        "val Dice",
    )

    loss_series = []
    for label, df in histories:
        loss_series.append((f"{label} train", df, "train_loss", "--"))
        loss_series.append((f"{label} val", df, "val_loss", "-"))
    save_line_plot(
        loss_series,
        "report_assets/curves/lr_loss_curve.png",
        "Learning Rate Tuning: Loss",
        "loss",
    )


def plot_size_bars():
    items = [("256", "Exp_SIZE_256"), ("512", "Exp_SIZE_512")]
    labels, miou, dice, time_values = [], [], [], []
    for label, exp in items:
        summary = read_summary(exp)
        if summary is None:
            continue
        labels.append(label)
        miou.append(float(summary["best_val_miou"]))
        dice.append(float(summary["best_val_dice"]))
        time_values.append(float(summary["total_time_minutes"]))
    save_bar_plot(labels, miou, "report_assets/bars/size_miou_bar.png", "Input Size Tuning: mIoU", "best val mIoU")
    save_bar_plot(labels, dice, "report_assets/bars/size_dice_bar.png", "Input Size Tuning: Dice", "best val Dice", "#59A14F")
    save_bar_plot(labels, time_values, "report_assets/bars/size_time_bar.png", "Input Size Tuning: Time", "minutes", "#F28E2B")


def plot_augmentation_bars():
    items = [("none", "Exp_AUG_NONE"), ("flip", "Exp_AUG_FLIP"), ("strong", "Exp_AUG_STRONG")]
    labels, miou, dice = [], [], []
    for label, exp in items:
        summary = read_summary(exp)
        if summary is None:
            continue
        labels.append(label)
        miou.append(float(summary["best_val_miou"]))
        dice.append(float(summary["best_val_dice"]))
    save_bar_plot(labels, miou, "report_assets/bars/augmentation_miou_bar.png", "Augmentation Tuning: mIoU", "best val mIoU")
    save_bar_plot(labels, dice, "report_assets/bars/augmentation_dice_bar.png", "Augmentation Tuning: Dice", "best val Dice", "#59A14F")


def plot_loss_bars_and_curve():
    items = [("CE", "Exp_LOSS_CE"), ("Dice", "Exp_LOSS_DICE"), ("CE + Dice", "Exp_LOSS_CE_DICE")]
    labels, miou, dice = [], [], []
    val_loss_series = []
    for label, exp in items:
        summary = read_summary(exp)
        history = read_history(exp)
        if summary is not None:
            labels.append(label)
            miou.append(float(summary["best_val_miou"]))
            dice.append(float(summary["best_val_dice"]))
        if history is not None:
            val_loss_series.append((label, history, "val_loss", "-"))
    save_bar_plot(labels, miou, "report_assets/bars/loss_miou_bar.png", "Loss Tuning: mIoU", "best val mIoU")
    save_bar_plot(labels, dice, "report_assets/bars/loss_dice_bar.png", "Loss Tuning: Dice", "best val Dice", "#59A14F")
    save_line_plot(val_loss_series, "report_assets/curves/loss_val_loss_curve.png", "Loss Tuning: val loss", "val loss")


def plot_final_curves():
    history = read_history("Exp_FINAL_UNET")
    if history is None:
        return
    save_line_plot(
        [
            ("train_loss", history, "train_loss", "-"),
            ("val_loss", history, "val_loss", "-"),
        ],
        "report_assets/curves/final_loss_curve.png",
        "Final U-Net Training Loss",
        "loss",
    )
    save_line_plot(
        [
            ("val_pixel_acc", history, "val_pixel_acc", "-"),
            ("val_miou", history, "val_miou", "-"),
            ("val_dice", history, "val_dice", "-"),
        ],
        "report_assets/curves/final_metrics_curve.png",
        "Final U-Net Validation Metrics",
        "score",
    )


def make_augmentation_examples(config):
    samples = build_samples(config["paths"]["train_images"], config["paths"]["train_masks"])
    image = Image.open(samples[0]["image"]).convert("RGB").resize((256, 256))
    none = image.copy()
    flip = ImageOps.mirror(image)
    strong = ImageOps.mirror(image)
    strong = ImageEnhance.Brightness(strong).enhance(1.2)
    strong = ImageEnhance.Contrast(strong).enhance(1.25)
    examples = [("original", image), ("none", none), ("flip", flip), ("strong", strong)]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (title, img) in zip(axes, examples):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    output_path = "report_assets/dataset_examples/augmentation_examples.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def denormalize(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * STD + MEAN) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def load_final_model(config):
    checkpoint_path = Path("outputs/checkpoints/Exp_FINAL_UNET/best_model.pth")
    if not checkpoint_path.exists():
        print(f"Skip qualitative visualization: missing {checkpoint_path}")
        return None, None, None
    device = resolve_device(config)
    state = torch.load(checkpoint_path, map_location=device)
    num_classes = int(state.get("num_classes", config.get("num_classes", 10)))
    base_channels = int(state.get("base_channels", config.get("model", {}).get("base_channels", 32)))
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state, device


@torch.no_grad()
def predict_one(model, image_tensor, device):
    logits = model(image_tensor.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def overlay_image(image, mask_color, alpha=0.45):
    return np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * mask_color.astype(np.float32), 0, 255).astype(np.uint8)


def confusion_from_pred(pred, target, num_classes):
    valid = (target >= 0) & (target < num_classes)
    indices = target[valid].reshape(-1) * num_classes + pred[valid].reshape(-1)
    conf = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf.astype(np.int64)


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64)
    diag = np.diag(confusion)
    gt = confusion.sum(axis=1)
    pred = confusion.sum(axis=0)
    iou_den = gt + pred - diag
    dice_den = gt + pred
    ious = np.divide(diag, iou_den, out=np.full_like(diag, np.nan), where=iou_den > 0)
    dices = np.divide(2.0 * diag, dice_den, out=np.full_like(diag, np.nan), where=dice_den > 0)
    return {
        "miou": float(np.nanmean(ious)) if np.any(iou_den > 0) else 0.0,
        "dice": float(np.nanmean(dices)) if np.any(dice_den > 0) else 0.0,
    }


@torch.no_grad()
def make_qualitative_results(config):
    model, state, device = load_final_model(config)
    if model is None:
        return
    image_size = int(state.get("image_size", config["quick_check"]["image_size"]))
    split = "test"
    dataset = FloodNetDataset(config, split, image_size=image_size, augmentation="none", subset_ratio=1.0, seed=config.get("seed", 42))
    sample_count = min(6, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int).tolist()

    rows = []
    for idx in indices:
        image_tensor, mask_tensor = dataset[idx]
        pred = predict_one(model, image_tensor, device)
        image = denormalize(image_tensor)
        gt_color = colorize_mask(mask_tensor.numpy())
        pred_color = colorize_mask(pred)
        overlay = overlay_image(image, pred_color)
        rows.append((idx, image, gt_color, pred_color, overlay))

        sample_id = dataset.samples[idx]["id"]
        Image.fromarray(pred_color).save(Path("outputs/predictions/Exp_FINAL_UNET") / f"{sample_id}_pred_color.png")

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, len(rows) * 3.2))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["Image", "Ground Truth", "Prediction", "Overlay"]
    for row_idx, (_, image, gt_color, pred_color, overlay) in enumerate(rows):
        for col_idx, (title, arr) in enumerate(zip(titles, [image, gt_color, pred_color, overlay])):
            axes[row_idx, col_idx].imshow(arr)
            axes[row_idx, col_idx].set_title(title)
            axes[row_idx, col_idx].axis("off")
    plt.tight_layout()
    output_path = "report_assets/qualitative/final_qualitative_results.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")

    fig, axes = plt.subplots(len(rows), 2, figsize=(8, len(rows) * 3.2))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, (_, image, _, _, overlay) in enumerate(rows):
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title("Image")
        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title("Image + Prediction")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].axis("off")
    plt.tight_layout()
    output_path = "report_assets/qualitative/final_overlay_results.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


@torch.no_grad()
def make_failure_cases(config, max_candidates=60):
    model, state, device = load_final_model(config)
    if model is None:
        return
    image_size = int(state.get("image_size", config["quick_check"]["image_size"]))
    num_classes = int(state.get("num_classes", config.get("num_classes", 10)))
    dataset = FloodNetDataset(config, "test", image_size=image_size, augmentation="none", subset_ratio=1.0, sample_limit=max_candidates, seed=config.get("seed", 42))

    scored = []
    for idx in range(len(dataset)):
        image_tensor, mask_tensor = dataset[idx]
        pred = predict_one(model, image_tensor, device)
        conf = confusion_from_pred(pred, mask_tensor.numpy(), num_classes)
        metric = metrics_from_confusion(conf)
        scored.append((metric["miou"], metric["dice"], idx, image_tensor, mask_tensor, pred))
    scored.sort(key=lambda x: (x[0], x[1]))
    selected = scored[: min(5, len(scored))]
    if not selected:
        print("Skip failure cases: no samples")
        return

    fig, axes = plt.subplots(len(selected), 4, figsize=(14, len(selected) * 3.2))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, (miou, dice, _, image_tensor, mask_tensor, pred) in enumerate(selected):
        image = denormalize(image_tensor)
        gt_color = colorize_mask(mask_tensor.numpy())
        pred_color = colorize_mask(pred)
        diff = np.zeros_like(image)
        diff[pred != mask_tensor.numpy()] = np.array([255, 40, 40], dtype=np.uint8)
        panels = [image, gt_color, pred_color, diff]
        titles = ["Image", "Ground Truth", "Prediction", f"Difference mIoU={miou:.3f} Dice={dice:.3f}"]
        for col_idx, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_idx, col_idx].imshow(panel)
            axes[row_idx, col_idx].set_title(title)
            axes[row_idx, col_idx].axis("off")
    plt.tight_layout()
    output_path = "report_assets/failure_cases/failure_cases.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def make_palette_legend():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(CLASS_NAMES))
    ax.barh(y, np.ones(len(CLASS_NAMES)), color=PALETTE / 255.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)])
    ax.set_xticks([])
    ax.set_title("FloodNet color palette")
    ax.invert_yaxis()
    plt.tight_layout()
    output_path = "report_assets/dataset_examples/palette_legend.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from FloodNet U-Net experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    create_report_dirs()
    read_best_tuning_config()

    plot_learning_rate_curves()
    plot_size_bars()
    plot_augmentation_bars()
    make_augmentation_examples(config)
    plot_loss_bars_and_curve()
    plot_final_curves()
    make_qualitative_results(config)
    make_failure_cases(config)
    make_palette_legend()


if __name__ == "__main__":
    main()
