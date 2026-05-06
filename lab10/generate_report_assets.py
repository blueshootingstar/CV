import json
import math
import typing
from collections import OrderedDict as TypingOrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from torch import nn
from torch.utils.data import DataLoader

if not hasattr(typing, "OrderedDict"):
    typing.OrderedDict = TypingOrderedDict

from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def configure_matplotlib() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_test_loader(data_dir: Path) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    return image * 0.3081 + 0.1307


def load_model(model_path: Path) -> nn.Module:
    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_training_curves(metrics: dict, output_path: Path) -> None:
    history = metrics["history"]
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    test_loss = [item["test_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    test_acc = [item["test_acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, test_loss, marker="s", linewidth=2, label="Test Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, label="Train Acc")
    axes[1].plot(epochs, test_acc, marker="s", linewidth=2, label="Test Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.93, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_network_structure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.axis("off")
    boxes = [
        "输入图像\n1x28x28",
        "卷积层1\nConv2d(1,32,3x3)",
        "ReLU",
        "最大池化\nMaxPool2d(2x2)",
        "卷积层2\nConv2d(32,64,3x3)",
        "ReLU",
        "最大池化\nMaxPool2d(2x2)",
        "展平\nFlatten",
        "全连接层\nLinear(64x7x7,128)",
        "ReLU + Dropout(0.2)",
        "输出层\nLinear(128,10)",
    ]

    top = 0.94
    step = 0.078
    for index, text in enumerate(boxes):
        y = top - index * step
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f3f7ff", ec="#4b6cb7", lw=1.5),
        )
        if index < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.045),
                xytext=(0.5, y - 0.015),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#444444"),
            )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def collect_predictions(model: nn.Module, dataloader: DataLoader) -> tuple:
    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

            if len(sample_images) < 16:
                remaining = 16 - len(sample_images)
                take = min(remaining, images.size(0))
                for i in range(take):
                    sample_images.append(images[i].cpu())
                    sample_preds.append(int(preds[i].cpu()))
                    sample_labels.append(int(labels[i].cpu()))

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        sample_images,
        sample_preds,
        sample_labels,
    )


def plot_sample_predictions(images, preds, labels, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 12.5))
    for idx, ax in enumerate(axes.flat):
        image = denormalize(images[idx]).squeeze(0).numpy()
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"True: {labels[idx]}\nPred: {preds[idx]}",
            fontsize=25,
            pad=8,
        )
        ax.axis("off")
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, output_path: Path) -> None:
    matrix = torch.zeros((10, 10), dtype=torch.int32)
    for true_label, pred_label in zip(labels, preds):
        matrix[int(true_label), int(pred_label)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(10):
        for j in range(10):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="#111111")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(metrics: dict, output_path: Path) -> None:
    history = metrics["history"]
    lines = [
        "实验环境：PyTorch + MNIST",
        "训练设备：CPU",
        "训练轮数：3",
        "",
    ]
    for item in history:
        lines.append(
            "Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            "test_loss={test_loss:.4f}, test_acc={test_acc:.4f}".format(**item)
        )
    lines.append("")
    lines.append("最终测试准确率：{:.2f}%".format(metrics["final_test_accuracy"] * 100))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#666666"),
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    metrics = load_metrics(outputs_dir / "metrics.json")
    model = load_model(outputs_dir / "mnist_cnn.pth")
    test_loader = build_test_loader(base_dir / "data")
    preds, labels, sample_images, sample_preds, sample_labels = collect_predictions(model, test_loader)

    plot_training_curves(metrics, outputs_dir / "training_curves.png")
    plot_network_structure(outputs_dir / "cnn_structure.png")
    plot_sample_predictions(sample_images, sample_preds, sample_labels, outputs_dir / "sample_predictions.png")
    plot_confusion_matrix(preds, labels, outputs_dir / "confusion_matrix.png")
    plot_metrics_summary(metrics, outputs_dir / "metrics_summary.png")


if __name__ == "__main__":
    main()
