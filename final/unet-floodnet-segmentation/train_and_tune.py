import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASS_NAMES = [
    "background",
    "building-flooded",
    "building-non-flooded",
    "road-flooded",
    "road-non-flooded",
    "water",
    "tree",
    "vehicle",
    "pool",
    "grass",
]

PALETTE = np.array(
    [
        (0, 0, 0),
        (255, 0, 0),
        (180, 120, 120),
        (160, 150, 20),
        (140, 140, 140),
        (61, 230, 250),
        (0, 82, 255),
        (255, 0, 245),
        (255, 235, 0),
        (4, 250, 7),
    ],
    dtype=np.uint8,
)

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_output_dirs():
    for path in [
        "outputs/logs",
        "outputs/checkpoints",
        "outputs/summaries",
        "outputs/predictions",
        "report_assets/dataset_examples",
        "report_assets/curves",
        "report_assets/bars",
        "report_assets/qualitative",
        "report_assets/failure_cases",
    ]:
        ensure_dir(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config):
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def list_image_files(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def mask_to_sample_id(mask_path):
    stem = Path(mask_path).stem
    if stem.endswith("_lab"):
        return stem[:-4]
    return stem.replace("_lab", "")


def build_image_index(image_dir):
    image_index = {}
    for path in list_image_files(image_dir):
        key = path.stem.lower()
        current = image_index.get(key)
        if current is None:
            image_index[key] = path
            continue
        if IMAGE_EXTS.index(path.suffix.lower()) < IMAGE_EXTS.index(current.suffix.lower()):
            image_index[key] = path
    return image_index


def build_samples(image_dir, mask_dir):
    image_index = build_image_index(image_dir)
    masks = list_image_files(mask_dir)
    samples = []
    missing = []
    for mask_path in masks:
        sample_id = mask_to_sample_id(mask_path)
        image_path = image_index.get(sample_id.lower())
        if image_path is None:
            missing.append(str(mask_path))
        else:
            samples.append({"id": sample_id, "image": str(image_path), "mask": str(mask_path)})
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} masks cannot find matching original images. Preview:\n{preview}")
    if not samples:
        raise RuntimeError(f"No matched samples found in image_dir={image_dir}, mask_dir={mask_dir}")
    return samples


def rgb_to_index(rgb):
    rgb = rgb[..., :3].astype(np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = class_id
    if np.any(out == 255):
        unknown = np.unique(rgb[out == 255].reshape(-1, 3), axis=0)[:10].tolist()
        raise ValueError(f"RGB mask contains colors outside FloodNet palette: {unknown}")
    return out


def read_mask_as_index(mask_path, num_classes):
    image = Image.open(mask_path)
    arr = np.array(image)
    if arr.ndim == 2:
        mask = arr.astype(np.int64)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        mask = rgb_to_index(arr).astype(np.int64)
    else:
        raise ValueError(f"Unsupported mask shape {arr.shape} in {mask_path}")
    min_value = int(mask.min())
    max_value = int(mask.max())
    if min_value < 0 or max_value >= num_classes:
        raise ValueError(f"Mask {mask_path} has invalid values. min={min_value}, max={max_value}, expected 0..{num_classes - 1}")
    return mask.astype(np.uint8)


def colorize_mask(mask):
    mask = np.asarray(mask)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    color[valid] = PALETTE[mask[valid]]
    return color


def denormalize_image_tensor(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * STD + MEAN) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


class FloodNetDataset(Dataset):
    def __init__(
        self,
        config,
        split,
        image_size=256,
        augmentation="none",
        subset_ratio=1.0,
        sample_limit=None,
        seed=42,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        if augmentation not in ["none", "flip", "strong"]:
            raise ValueError(f"Invalid augmentation: {augmentation}")

        self.config = config
        self.split = split
        self.image_size = int(image_size)
        self.augmentation = augmentation if split == "train" else "none"
        self.num_classes = int(config.get("num_classes", 10))
        self.seed = seed

        image_dir = config["paths"][f"{split}_images"]
        mask_dir = config["paths"][f"{split}_masks"]
        samples = build_samples(image_dir, mask_dir)

        rng = random.Random(seed)
        if subset_ratio is not None and float(subset_ratio) < 1.0:
            subset_size = max(1, int(len(samples) * float(subset_ratio)))
            samples = sorted(rng.sample(samples, subset_size), key=lambda x: x["id"])
        if sample_limit is not None:
            samples = samples[: int(sample_limit)]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, image, mask):
        if self.augmentation == "none":
            return image, mask

        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if self.augmentation == "strong":
            if random.random() < 0.2:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)
            if random.random() < 0.8:
                brightness = random.uniform(0.75, 1.25)
                contrast = random.uniform(0.75, 1.25)
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
        return image, mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image"]).convert("RGB")
        mask_np = read_mask_as_index(item["mask"], self.num_classes)
        mask = Image.fromarray(mask_np)

        image, mask = self._apply_augmentation(image, mask)
        image = image.resize((self.image_size, self.image_size), BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - MEAN) / STD
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, base_channels=32):
        super().__init__()
        c = int(base_channels)
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def validate_targets(targets, num_classes):
    min_value = int(targets.min().item())
    max_value = int(targets.max().item())
    if min_value < 0 or max_value >= num_classes:
        raise ValueError(f"Invalid target values. min={min_value}, max={max_value}, expected 0..{num_classes - 1}")


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        validate_targets(targets, self.num_classes)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.num_classes = num_classes

    def forward(self, logits, targets):
        validate_targets(targets, self.num_classes)
        return self.ce(logits, targets) + self.dice(logits, targets)


def build_loss(loss_name, num_classes):
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "dice":
        return DiceLoss(num_classes)
    if loss_name == "ce_dice":
        return CEDiceLoss(num_classes)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_batch_confusion(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    targets = targets.detach()
    preds = preds.detach()
    valid = (targets >= 0) & (targets < num_classes)
    indices = targets[valid] * num_classes + preds[valid]
    conf = torch.bincount(indices, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes).cpu().numpy().astype(np.int64)


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64)
    diag = np.diag(confusion)
    gt = confusion.sum(axis=1)
    pred = confusion.sum(axis=0)
    total = confusion.sum()
    pixel_acc = float(diag.sum() / total) if total > 0 else 0.0

    iou_den = gt + pred - diag
    valid_iou = iou_den > 0
    ious = np.divide(diag, iou_den, out=np.full_like(diag, np.nan), where=valid_iou)
    miou = float(np.nanmean(ious)) if np.any(valid_iou) else 0.0

    dice_den = gt + pred
    valid_dice = dice_den > 0
    dices = np.divide(2.0 * diag, dice_den, out=np.full_like(diag, np.nan), where=valid_dice)
    mean_dice = float(np.nanmean(dices)) if np.any(valid_dice) else 0.0
    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "dice": mean_dice,
        "per_class_iou": np.nan_to_num(ious).tolist(),
        "per_class_dice": np.nan_to_num(dices).tolist(),
    }


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, num_classes):
    model.train()
    running_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        validate_targets(masks, num_classes)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{running_loss / max(total_samples, 1):.4f}")
    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, num_classes):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    progress = tqdm(loader, desc="eval", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        validate_targets(masks, num_classes)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        confusion += compute_batch_confusion(logits, masks, num_classes)

    metrics = metrics_from_confusion(confusion)
    return running_loss / max(total_samples, 1), metrics


def save_history(history, path):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_pixel_acc",
        "val_miou",
        "val_dice",
        "learning_rate",
        "epoch_time_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def batch_size_for_image_size(config, image_size):
    key = "batch_size_512" if int(image_size) >= 512 else "batch_size_256"
    return int(config["training"].get(key, 8))


def make_experiment_dirs(experiment_name):
    dirs = {
        "logs": Path("outputs/logs") / experiment_name,
        "checkpoints": Path("outputs/checkpoints") / experiment_name,
        "summaries": Path("outputs/summaries") / experiment_name,
        "predictions": Path("outputs/predictions") / experiment_name,
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def train_experiment(
    config,
    experiment_name,
    image_size,
    learning_rate,
    loss_name,
    augmentation,
    epochs,
    subset_ratio,
    train_limit=None,
    val_limit=None,
):
    create_output_dirs()
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(config)
    use_amp = bool(config["training"].get("use_amp", True)) and device.type == "cuda"
    num_classes = int(config.get("num_classes", 10))
    base_channels = int(config.get("model", {}).get("base_channels", 32))
    batch_size = batch_size_for_image_size(config, image_size)
    num_workers = int(config["training"].get("num_workers", 4))

    print(f"\n[{experiment_name}] device={device}, amp={use_amp}, image_size={image_size}, batch_size={batch_size}")
    train_dataset = FloodNetDataset(
        config,
        "train",
        image_size=image_size,
        augmentation=augmentation,
        subset_ratio=subset_ratio,
        sample_limit=train_limit,
        seed=seed,
    )
    val_dataset = FloodNetDataset(
        config,
        "val",
        image_size=image_size,
        augmentation="none",
        subset_ratio=subset_ratio,
        sample_limit=val_limit,
        seed=seed,
    )
    print(f"[{experiment_name}] train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")

    train_loader = make_loader(train_dataset, batch_size, True, num_workers, device)
    val_loader = make_loader(val_dataset, batch_size, False, num_workers, device)

    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels).to(device)
    criterion = build_loss(loss_name, num_classes)
    optimizer_name = config["training"].get("optimizer", "adam").lower()
    if optimizer_name != "adam":
        raise ValueError(f"Only adam is supported by this project, got: {optimizer_name}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dirs = make_experiment_dirs(experiment_name)
    exp_config = {
        "experiment_name": experiment_name,
        "image_size": int(image_size),
        "learning_rate": float(learning_rate),
        "augmentation": augmentation,
        "loss": loss_name,
        "epochs": int(epochs),
        "subset_ratio": float(subset_ratio),
        "train_limit": train_limit,
        "val_limit": val_limit,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_amp": use_amp,
        "base_channels": base_channels,
    }
    save_json(exp_config, dirs["summaries"] / "config.json")

    history = []
    best_epoch = 0
    best_val_miou = -1.0
    best_metrics = {"pixel_acc": 0.0, "miou": 0.0, "dice": 0.0}
    start_time = time.time()

    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.time()
        print(f"\n[{experiment_name}] Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, num_classes)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp, num_classes)
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_miou": val_metrics["miou"],
            "val_dice": val_metrics["dice"],
            "learning_rate": float(learning_rate),
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        save_history(history, dirs["logs"] / "history.csv")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "experiment_name": experiment_name,
            "num_classes": num_classes,
            "image_size": int(image_size),
            "learning_rate": float(learning_rate),
            "augmentation": augmentation,
            "loss": loss_name,
            "base_channels": base_channels,
            "class_names": CLASS_NAMES,
            "palette": PALETTE.astype(int).tolist(),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, dirs["checkpoints"] / "last_model.pth")
        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(checkpoint, dirs["checkpoints"] / "best_model.pth")

        print(
            f"[{experiment_name}] "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_metrics['pixel_acc']:.4f}, val_miou={val_metrics['miou']:.4f}, "
            f"val_dice={val_metrics['dice']:.4f}, time={epoch_time:.1f}s"
        )

    total_time_minutes = (time.time() - start_time) / 60.0
    final_metrics = history[-1]
    summary = {
        "experiment_name": experiment_name,
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_val_miou),
        "best_val_dice": float(best_metrics["dice"]),
        "best_val_pixel_acc": float(best_metrics["pixel_acc"]),
        "final_val_miou": float(final_metrics["val_miou"]),
        "final_val_dice": float(final_metrics["val_dice"]),
        "final_val_pixel_acc": float(final_metrics["val_pixel_acc"]),
        "total_time_minutes": float(total_time_minutes),
        "image_size": int(image_size),
        "learning_rate": float(learning_rate),
        "augmentation": augmentation,
        "loss": loss_name,
        "epochs": int(epochs),
        "subset_ratio": float(subset_ratio),
    }
    save_json(summary, dirs["summaries"] / "summary.json")
    return summary


def format_lr_name(lr):
    text = f"{float(lr):g}".replace(".", "_")
    return text


def select_best(summaries, metric="best_val_miou"):
    return max(summaries, key=lambda item: float(item.get(metric, -1.0)))


def run_quick(config, args=None):
    quick = config["quick_check"]
    return train_experiment(
        config,
        "Exp_QUICK_CHECK",
        args.image_size if args and args.image_size is not None else quick["image_size"],
        args.lr if args and args.lr is not None else quick["lr"],
        args.loss if args and args.loss is not None else quick["loss"],
        args.augmentation if args and args.augmentation is not None else quick["augmentation"],
        args.epochs if args and args.epochs is not None else quick["epochs"],
        subset_ratio=1.0,
        train_limit=quick["train_limit"],
        val_limit=quick["val_limit"],
    )


def run_single(config, args):
    quick = config["quick_check"]
    tuning = config["tuning"]
    return train_experiment(
        config,
        args.experiment_name or "Exp_SINGLE",
        args.image_size or quick["image_size"],
        args.lr or quick["lr"],
        args.loss or quick["loss"],
        args.augmentation or quick["augmentation"],
        args.epochs or tuning["epochs"],
        args.subset_ratio if args.subset_ratio is not None else tuning["subset_ratio"],
    )


def run_tune(config):
    tuning = config["tuning"]
    epochs = int(tuning["epochs"])
    subset_ratio = float(tuning["subset_ratio"])

    lr_summaries = []
    for lr in config["learning_rates"]:
        exp_name = f"Exp_LR_{format_lr_name(lr)}"
        lr_summaries.append(train_experiment(config, exp_name, 256, lr, "ce", "flip", epochs, subset_ratio))
    best_lr_summary = select_best(lr_summaries)
    best_lr = float(best_lr_summary["learning_rate"])

    size_summaries = []
    for size in config["image_sizes"]:
        exp_name = f"Exp_SIZE_{int(size)}"
        size_summaries.append(train_experiment(config, exp_name, int(size), best_lr, "ce", "flip", epochs, subset_ratio))
    best_size_summary = select_best(size_summaries)
    best_size = int(best_size_summary["image_size"])

    aug_name_map = {"none": "NONE", "flip": "FLIP", "strong": "STRONG"}
    aug_summaries = []
    for aug in config["augmentations"]:
        exp_name = f"Exp_AUG_{aug_name_map[aug]}"
        aug_summaries.append(train_experiment(config, exp_name, best_size, best_lr, "ce", aug, epochs, subset_ratio))
    best_aug_summary = select_best(aug_summaries)
    best_aug = best_aug_summary["augmentation"]

    loss_name_map = {"ce": "CE", "dice": "DICE", "ce_dice": "CE_DICE"}
    loss_summaries = []
    for loss_name in config["losses"]:
        exp_name = f"Exp_LOSS_{loss_name_map[loss_name]}"
        loss_summaries.append(train_experiment(config, exp_name, best_size, best_lr, loss_name, best_aug, epochs, subset_ratio))
    best_loss_summary = select_best(loss_summaries)
    best_loss = best_loss_summary["loss"]

    best_config = {
        "best_lr": best_lr,
        "best_size": best_size,
        "best_aug": best_aug,
        "best_loss": best_loss,
        "selection_metric": config.get("selection_metric", "val_miou"),
        "stages": {
            "learning_rate": lr_summaries,
            "image_size": size_summaries,
            "augmentation": aug_summaries,
            "loss": loss_summaries,
        },
    }
    save_json(best_config, "outputs/summaries/best_tuning_config.json")
    print("\nBest tuning config saved to outputs/summaries/best_tuning_config.json")
    print(json.dumps({k: best_config[k] for k in ["best_lr", "best_size", "best_aug", "best_loss"]}, indent=2))
    return best_config


def load_best_tuning_or_default(config):
    path = Path("outputs/summaries/best_tuning_config.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            best = json.load(f)
        return {
            "lr": best["best_lr"],
            "image_size": best["best_size"],
            "augmentation": best["best_aug"],
            "loss": best["best_loss"],
        }

    print("best_tuning_config.json not found. Use config.yaml defaults for final training.")
    quick = config["quick_check"]
    return {
        "lr": quick["lr"],
        "image_size": quick["image_size"],
        "augmentation": quick["augmentation"],
        "loss": quick["loss"],
    }


def run_final(config, args=None):
    best = load_best_tuning_or_default(config)
    final_cfg = config["final"]
    return train_experiment(
        config,
        "Exp_FINAL_UNET",
        args.image_size if args and args.image_size is not None else best["image_size"],
        args.lr if args and args.lr is not None else best["lr"],
        args.loss if args and args.loss is not None else best["loss"],
        args.augmentation if args and args.augmentation is not None else best["augmentation"],
        args.epochs if args and args.epochs is not None else final_cfg["epochs"],
        args.subset_ratio if args and args.subset_ratio is not None else final_cfg["subset_ratio"],
    )


@torch.no_grad()
def run_test(config, checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required in test mode")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    create_output_dirs()
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)
    state = torch.load(checkpoint_path, map_location=device)
    num_classes = int(state.get("num_classes", config.get("num_classes", 10)))
    image_size = int(state.get("image_size", config["quick_check"]["image_size"]))
    base_channels = int(state.get("base_channels", config.get("model", {}).get("base_channels", 32)))
    loss_name = state.get("loss", config["quick_check"]["loss"])
    use_amp = bool(config["training"].get("use_amp", True)) and device.type == "cuda"
    batch_size = batch_size_for_image_size(config, image_size)
    num_workers = int(config["training"].get("num_workers", 4))

    dataset = FloodNetDataset(config, "test", image_size=image_size, augmentation="none", subset_ratio=1.0, seed=config.get("seed", 42))
    loader = make_loader(dataset, batch_size, False, num_workers, device)
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(state["model_state_dict"])
    criterion = build_loss(loss_name, num_classes)
    test_loss, metrics = evaluate(model, loader, criterion, device, use_amp, num_classes)

    summary = {
        "checkpoint": str(checkpoint_path),
        "test_loss": float(test_loss),
        "test_pixel_acc": float(metrics["pixel_acc"]),
        "test_miou": float(metrics["miou"]),
        "test_dice": float(metrics["dice"]),
        "per_class_iou": metrics["per_class_iou"],
        "per_class_dice": metrics["per_class_dice"],
        "image_size": image_size,
        "loss": loss_name,
    }
    save_json(summary, "outputs/summaries/test_summary.json")
    print("\n========== Test Set Metrics ==========")
    print(f"test_loss: {test_loss:.4f}")
    print(f"pixel_acc: {metrics['pixel_acc']:.4f}")
    print(f"mIoU:      {metrics['miou']:.4f}")
    print(f"Dice:      {metrics['dice']:.4f}")
    print("Saved to outputs/summaries/test_summary.json")
    print("======================================\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train, tune and evaluate U-Net on FloodNet.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", choices=["quick", "single", "tune", "final", "test"], required=True)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--image_size", type=int, default=None, choices=[256, 512])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", default=None, choices=["ce", "dice", "ce_dice"])
    parser.add_argument("--augmentation", default=None, choices=["none", "flip", "strong"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset_ratio", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs()

    if args.mode == "quick":
        run_quick(config, args)
    elif args.mode == "single":
        run_single(config, args)
    elif args.mode == "tune":
        run_tune(config)
    elif args.mode == "final":
        run_final(config, args)
    elif args.mode == "test":
        run_test(config, args.checkpoint)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
