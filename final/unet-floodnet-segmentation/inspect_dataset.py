import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


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


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_image_files(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def mask_to_sample_id(mask_path):
    stem = mask_path.stem
    if stem.endswith("_lab"):
        return stem[:-4]
    return stem.replace("_lab", "")


def build_image_index(image_dir):
    index = {}
    for path in list_image_files(image_dir):
        key = path.stem.lower()
        current = index.get(key)
        if current is None:
            index[key] = path
            continue
        if IMAGE_EXTS.index(path.suffix.lower()) < IMAGE_EXTS.index(current.suffix.lower()):
            index[key] = path
    return index


def match_split(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_files = list_image_files(image_dir)
    mask_files = list_image_files(mask_dir)
    image_index = build_image_index(image_dir)
    mask_ids = {mask_to_sample_id(p).lower() for p in mask_files}

    matched = []
    missing_images = []
    for mask_path in mask_files:
        sample_id = mask_to_sample_id(mask_path)
        image_path = image_index.get(sample_id.lower())
        if image_path is None:
            missing_images.append(str(mask_path))
        else:
            matched.append({"id": sample_id, "image": str(image_path), "mask": str(mask_path)})

    missing_masks = []
    for image_path in image_files:
        if image_path.stem.lower() not in mask_ids:
            missing_masks.append(str(image_path))

    return {
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "matched_count": len(matched),
        "missing_image_count": len(missing_images),
        "missing_mask_count": len(missing_masks),
        "missing_images": missing_images[:30],
        "missing_masks": missing_masks[:30],
        "matched": matched,
    }


def rgb_to_index(rgb):
    rgb = rgb[..., :3].astype(np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = class_id
    unknown = np.unique(rgb[out == 255].reshape(-1, 3), axis=0) if np.any(out == 255) else np.empty((0, 3))
    if unknown.size > 0:
        preview = unknown[:10].tolist()
        raise ValueError(f"RGB mask contains colors not in FloodNet palette: {preview}")
    return out


def read_mask_as_index(mask_path):
    image = Image.open(mask_path)
    arr = np.array(image)
    if arr.ndim == 2:
        mask = arr.astype(np.int64)
        mask_type = "index"
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        mask = rgb_to_index(arr).astype(np.int64)
        mask_type = "rgb_palette"
    else:
        raise ValueError(f"Unsupported mask shape {arr.shape} in {mask_path}")

    unique_values = sorted(np.unique(mask).astype(int).tolist())
    invalid_values = [v for v in unique_values if v < 0 or v >= len(CLASS_NAMES)]
    if invalid_values:
        raise ValueError(f"Mask {mask_path} has invalid class ids: {invalid_values}")
    return mask, mask_type, unique_values, image.mode


def colorize_mask(mask):
    mask = np.asarray(mask)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    color[valid] = PALETTE[mask[valid]]
    return color


def find_official_color_mask(config, split, mask_path):
    color_root = Path(config.get("colormask_dir", ""))
    folder_map = {
        "train": "ColorMasks-TrainSet",
        "val": "ColorMasks-ValSet",
        "test": "ColorMasks-TestSet",
    }
    candidate = color_root / folder_map[split] / Path(mask_path).name
    if candidate.exists():
        return candidate
    return None


def make_dataset_samples(config, stats_by_split, output_path, sample_rows=6):
    candidates = []
    for split in ["train", "val", "test"]:
        for item in stats_by_split[split]["matched"]:
            candidates.append((split, item))
    if not candidates:
        print("No matched samples found; skip dataset sample figure.")
        return

    rng = random.Random(config.get("seed", 42))
    selected = rng.sample(candidates, min(sample_rows, len(candidates)))
    rows = len(selected)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, (split, item) in enumerate(selected):
        image = Image.open(item["image"]).convert("RGB")
        mask, _, _, _ = read_mask_as_index(item["mask"])
        mask_color = colorize_mask(mask)
        official_path = find_official_color_mask(config, split, item["mask"])
        official = Image.open(official_path).convert("RGB") if official_path else None

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{split} image {item['id']}")
        axes[row, 1].imshow(mask_color)
        axes[row, 1].set_title("label mask colorized")
        if official is not None:
            axes[row, 2].imshow(official)
            axes[row, 2].set_title("official color mask")
        else:
            axes[row, 2].imshow(mask_color)
            axes[row, 2].set_title("official color mask missing")
        for col in range(cols):
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_class_distribution(stats_by_split, output_path, sample_count=60, stride=8, seed=42):
    all_masks = []
    for split in ["train", "val", "test"]:
        all_masks.extend([item["mask"] for item in stats_by_split[split]["matched"]])
    if not all_masks:
        print("No matched masks found; skip class distribution figure.")
        return {"sampled_masks": 0, "stride": stride, "counts": [0] * len(CLASS_NAMES)}

    rng = random.Random(seed)
    selected = rng.sample(all_masks, min(sample_count, len(all_masks)))
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for mask_path in selected:
        mask, _, _, _ = read_mask_as_index(mask_path)
        sampled_pixels = mask[::stride, ::stride]
        counts += np.bincount(sampled_pixels.reshape(-1), minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]

    ratios = counts / max(counts.sum(), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(CLASS_NAMES, ratios, color=PALETTE / 255.0)
    ax.set_ylabel("pixel ratio")
    ax.set_title(f"Class distribution estimate ({len(selected)} masks, stride={stride})")
    ax.set_ylim(0, max(0.02, ratios.max() * 1.15))
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"sampled_masks": len(selected), "stride": stride, "counts": counts.astype(int).tolist()}


def inspect_masks(stats_by_split, sample_count, seed):
    all_masks = []
    for split in ["train", "val", "test"]:
        all_masks.extend([(split, item["mask"]) for item in stats_by_split[split]["matched"]])
    rng = random.Random(seed)
    selected = rng.sample(all_masks, min(sample_count, len(all_masks))) if all_masks else []

    mask_type_counter = Counter()
    mode_counter = Counter()
    unique_values = set()
    errors = []
    checked = []
    for split, mask_path in selected:
        try:
            _, mask_type, values, mode = read_mask_as_index(mask_path)
            mask_type_counter[mask_type] += 1
            mode_counter[mode] += 1
            unique_values.update(values)
            checked.append({"split": split, "path": mask_path, "unique_values": values, "mode": mode, "mask_type": mask_type})
        except Exception as exc:
            errors.append({"split": split, "path": mask_path, "error": str(exc)})

    return {
        "checked_count": len(selected),
        "mask_types": dict(mask_type_counter),
        "pil_modes": dict(mode_counter),
        "unique_values_union": sorted(int(v) for v in unique_values),
        "errors": errors,
        "checked_examples": checked[:20],
    }


def check_required_dirs(config):
    required = ["train_images", "train_masks", "val_images", "val_masks", "test_images", "test_masks"]
    missing = []
    for key in required:
        path = Path(config["paths"][key])
        if not path.exists():
            missing.append({"key": key, "path": str(path)})
    return missing


def print_summary(stats_by_split, mask_check, missing_dirs, can_train):
    print("\n========== FloodNet Dataset Inspection ==========")
    if missing_dirs:
        print("Missing directories:")
        for item in missing_dirs:
            print(f"  - {item['key']}: {item['path']}")

    for split in ["train", "val", "test"]:
        stats = stats_by_split[split]
        print(
            f"{split.capitalize()}: "
            f"images={stats['image_count']} | "
            f"masks={stats['mask_count']} | "
            f"matched={stats['matched_count']} | "
            f"missing_images={stats['missing_image_count']} | "
            f"missing_masks={stats['missing_mask_count']}"
        )

    print(f"Mask type counts: {mask_check['mask_types']}")
    print(f"PIL mode counts: {mask_check['pil_modes']}")
    print(f"Unique values union: {mask_check['unique_values_union']}")
    if mask_check["errors"]:
        print("Mask check errors:")
        for err in mask_check["errors"][:10]:
            print(f"  - {err['path']}: {err['error']}")
    print(f"Ready for training: {'YES' if can_train else 'NO'}")
    print("=================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect FloodNet dataset structure and masks.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mask_sample_count", type=int, default=30, help="Number of masks used for format inspection")
    parser.add_argument("--distribution_sample_count", type=int, default=60, help="Number of masks used for class distribution estimate")
    parser.add_argument("--distribution_stride", type=int, default=8, help="Pixel stride used for class distribution estimate")
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    ensure_dir("outputs/summaries")
    ensure_dir("report_assets/dataset_examples")

    missing_dirs = check_required_dirs(config)
    stats_by_split = {}
    for split in ["train", "val", "test"]:
        stats_by_split[split] = match_split(config["paths"][f"{split}_images"], config["paths"][f"{split}_masks"])

    mask_check = inspect_masks(stats_by_split, args.mask_sample_count, config.get("seed", 42))
    distribution = make_class_distribution(
        stats_by_split,
        "report_assets/dataset_examples/class_distribution.png",
        sample_count=args.distribution_sample_count,
        stride=args.distribution_stride,
        seed=config.get("seed", 42),
    )
    make_dataset_samples(config, stats_by_split, "report_assets/dataset_examples/dataset_samples.png")

    can_train = (
        not missing_dirs
        and all(stats_by_split[s]["matched_count"] > 0 for s in ["train", "val", "test"])
        and all(stats_by_split[s]["missing_image_count"] == 0 for s in ["train", "val", "test"])
        and len(mask_check["errors"]) == 0
        and all(v in range(len(CLASS_NAMES)) for v in mask_check["unique_values_union"])
    )

    summary = {
        "config": str(Path(args.config)),
        "class_names": CLASS_NAMES,
        "palette": PALETTE.astype(int).tolist(),
        "missing_dirs": missing_dirs,
        "splits": {
            split: {k: v for k, v in stats.items() if k != "matched"}
            for split, stats in stats_by_split.items()
        },
        "mask_check": mask_check,
        "class_distribution_estimate": distribution,
        "can_start_training": can_train,
    }
    with open("outputs/summaries/dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(stats_by_split, mask_check, missing_dirs, can_train)


if __name__ == "__main__":
    main()
