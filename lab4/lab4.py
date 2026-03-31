"""
PCA + KNN 人脸识别框架
数据集：ORL人脸数据库（40人，每人10张）

每次运行会输出：
  - 当前参数配置
  - 识别准确率
  - 混淆矩阵图（confusion_matrix.png）
  - PCA降维维度 vs 准确率曲线（pca_dimension_accuracy.png）
  - KNN的K值 vs 准确率曲线（knn_k_accuracy.png）
  - 训练集比例 vs 准确率曲线（train_ratio_accuracy.png）
  - 分类个数 vs 准确率曲线（num_classes_accuracy.png）
  - 距离度量 vs 准确率柱状图（knn_metric_accuracy.png）
"""

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                           调参区域 
# ============================================================
CONFIG = {
    "dataset_path": "ORL",           # 数据集路径（相对路径）
    "num_classes": 40,               # 使用多少个人的数据（最多40）
    "train_ratio": 0.7,              # 训练集比例（0.7 = 70%训练，30%测试）
    "pca_components": 50,            # PCA降维后的维度数
    "knn_k": 3,                      # KNN的K值
    "knn_metric": "euclidean",       # KNN距离度量：euclidean, manhattan, cosine
    "random_seed": 42,               # 随机种子
}
# ============================================================


def load_dataset(path, num_classes):
    """读取ORL数据集，返回图片数组和标签"""
    images = []
    labels = []
    img_shape = None

    all_folders = sorted(
        [d for d in os.listdir(path) if d.startswith('s') and os.path.isdir(os.path.join(path, d))],
        key=lambda x: int(x[1:])
    )
    # 只取前 num_classes 个人
    folders = all_folders[:num_classes]

    for idx, folder_name in enumerate(folders):
        folder = os.path.join(path, folder_name)
        for j in range(1, 11):
            img_path = os.path.join(folder, f"{j}.bmp")
            if not os.path.exists(img_path):
                continue
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img_shape is None:
                img_shape = img.shape
            images.append(img.flatten())
            labels.append(idx + 1)

    return np.array(images), np.array(labels), img_shape


def do_pca(X_train, X_test, n_components):
    """PCA降维"""
    pca = PCA(n_components=n_components, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def do_knn(X_train, y_train, X_test, k, metric):
    """KNN分类"""
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """画混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"混淆矩阵已保存：{save_path}")


def sweep_pca_dimension(X_train, y_train, X_test, y_test, k, metric, cfg,
                        dim_list=None, save_path="pca_dimension_accuracy.png"):
    """扫描不同PCA维度对准确率的影响"""
    if dim_list is None:
        max_dim = min(X_train.shape[0], X_train.shape[1])
        dim_list = list(range(5, min(max_dim, 81), 5))

    accuracies = []
    for dim in dim_list:
        X_tr_pca, X_te_pca, _ = do_pca(X_train, X_test, dim)
        y_pred = do_knn(X_tr_pca, y_train, X_te_pca, k, metric)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"  PCA维度={dim:>3d}, 准确率={acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(dim_list, accuracies, 'b-o', markersize=4)
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Accuracy')
    plt.title(f'PCA Dimension vs Accuracy\nClasses={cfg["num_classes"]}, Ratio={cfg["train_ratio"]}, K={k}, Metric={metric}', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"PCA维度-准确率曲线已保存：{save_path}")
    return dim_list, accuracies


def sweep_knn_k(X_train_pca, y_train, X_test_pca, y_test, metric, cfg,
                k_list=None, save_path="knn_k_accuracy.png"):
    """扫描不同K值对准确率的影响"""
    if k_list is None:
        k_list = list(range(1, 16))

    accuracies = []
    for k in k_list:
        y_pred = do_knn(X_train_pca, y_train, X_test_pca, k, metric)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"  K={k:>2d}, 准确率={acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_list, accuracies, 'r-o', markersize=4)
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title(f'KNN K-value vs Accuracy\nClasses={cfg["num_classes"]}, Ratio={cfg["train_ratio"]}, PCA={cfg["pca_components"]}, Metric={metric}', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"K值-准确率曲线已保存：{save_path}")
    return k_list, accuracies


def sweep_train_ratio(images, labels, pca_dim, k, metric, seed, cfg,
                      ratio_list=None, save_path="train_ratio_accuracy.png"):
    """扫描不同训练集比例对准确率的影响"""
    if ratio_list is None:
        ratio_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    accuracies = []
    for ratio in ratio_list:
        X_tr, X_te, y_tr, y_te = train_test_split(
            images, labels, train_size=ratio, random_state=seed, stratify=labels
        )
        X_tr_pca, X_te_pca, _ = do_pca(X_tr, X_te, min(pca_dim, X_tr.shape[0] - 1))
        y_pred = do_knn(X_tr_pca, y_tr, X_te_pca, k, metric)
        acc = accuracy_score(y_te, y_pred)
        accuracies.append(acc)
        print(f"  训练比例={ratio:.1f}, 训练={len(X_tr)}, 测试={len(X_te)}, 准确率={acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ratio_list, accuracies, 'g-o', markersize=6)
    plt.xlabel('Train Ratio')
    plt.ylabel('Accuracy')
    plt.title(f'Train Ratio vs Accuracy\nClasses={cfg["num_classes"]}, PCA={pca_dim}, K={k}, Metric={metric}', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"训练比例-准确率曲线已保存：{save_path}")
    return ratio_list, accuracies


def sweep_num_classes(dataset_path, pca_dim, k, metric, train_ratio, seed, cfg,
                      class_list=None, save_path="num_classes_accuracy.png"):
    """扫描不同分类个数对准确率的影响"""
    if class_list is None:
        class_list = [5, 10, 15, 20, 25, 30, 35, 40]

    accuracies = []
    for nc in class_list:
        imgs, lbls, _ = load_dataset(dataset_path, nc)
        X_tr, X_te, y_tr, y_te = train_test_split(
            imgs, lbls, train_size=train_ratio, random_state=seed, stratify=lbls
        )
        dim = min(pca_dim, X_tr.shape[0] - 1)
        X_tr_pca, X_te_pca, _ = do_pca(X_tr, X_te, dim)
        y_pred = do_knn(X_tr_pca, y_tr, X_te_pca, k, metric)
        acc = accuracy_score(y_te, y_pred)
        accuracies.append(acc)
        print(f"  分类个数={nc:>2d}, 准确率={acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(class_list, accuracies, 'm-o', markersize=6)
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy')
    plt.title(f'Number of Classes vs Accuracy\nRatio={train_ratio}, PCA={pca_dim}, K={k}, Metric={metric}', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"分类个数-准确率曲线已保存：{save_path}")
    return class_list, accuracies


def sweep_knn_metric(X_train_pca, y_train, X_test_pca, y_test, k, cfg,
                     metric_list=None, save_path="knn_metric_accuracy.png"):
    """扫描不同距离度量对准确率的影响"""
    if metric_list is None:
        metric_list = ["euclidean", "manhattan", "cosine", "chebyshev"]

    accuracies = []
    for metric in metric_list:
        y_pred = do_knn(X_train_pca, y_train, X_test_pca, k, metric)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"  距离度量={metric:<12s}, 准确率={acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.bar(metric_list, accuracies, color=['#4A90D9', '#E8524A', '#50C878', '#FFB347'])
    plt.xlabel('Distance Metric')
    plt.ylabel('Accuracy')
    plt.title(f'Distance Metric vs Accuracy\nClasses={cfg["num_classes"]}, Ratio={cfg["train_ratio"]}, PCA={cfg["pca_components"]}, K={k}', fontsize=10)
    plt.ylim(min(accuracies) - 0.05, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"距离度量-准确率柱状图已保存：{save_path}")
    return metric_list, accuracies


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    cfg = CONFIG

    # ---------- 打印当前参数配置 ----------
    print("=" * 50)
    print("当前参数配置：")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print("=" * 50)

    # ---------- 1. 加载数据集 ----------
    print("\n[1/6] 加载数据集...")
    images, labels, img_shape = load_dataset(cfg["dataset_path"], cfg["num_classes"])
    print(f"  图片总数: {len(images)}")
    print(f"  图片尺寸: {img_shape}")
    print(f"  分类个数: {len(np.unique(labels))}")
    print(f"  每张图片展平后维度: {images.shape[1]}")

    # ---------- 2. 划分训练集和测试集 ----------
    print("\n[2/6] 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        train_size=cfg["train_ratio"],
        random_state=cfg["random_seed"],
        stratify=labels
    )
    print(f"  训练集: {len(X_train)} 张")
    print(f"  测试集: {len(X_test)} 张")

    # ---------- 3. PCA降维 ----------
    print(f"\n[3/6] PCA降维（{images.shape[1]} -> {cfg['pca_components']}）...")
    X_train_pca, X_test_pca, pca = do_pca(X_train, X_test, cfg["pca_components"])

    # ---------- 4. KNN分类 ----------
    print(f"\n[4/6] KNN分类（K={cfg['knn_k']}，距离={cfg['knn_metric']}）...")
    y_pred = do_knn(X_train_pca, y_train, X_test_pca, cfg["knn_k"], cfg["knn_metric"])
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ★ 识别准确率: {acc:.4f} ({acc*100:.2f}%)")

    # ---------- 5. 可视化结果 ----------
    print("\n[5/6] 生成混淆矩阵...")
    plot_confusion_matrix(y_test, y_pred)

    # ---------- 6. 参数扫描 ----------
    print("\n[6/6] 参数扫描...")

    print("\n--- 扫描PCA维度 ---")
    sweep_pca_dimension(X_train, y_train, X_test, y_test,
                        cfg["knn_k"], cfg["knn_metric"], cfg)

    print("\n--- 扫描KNN的K值 ---")
    sweep_knn_k(X_train_pca, y_train, X_test_pca, y_test,
                cfg["knn_metric"], cfg)

    print("\n--- 扫描训练集比例 ---")
    sweep_train_ratio(images, labels, cfg["pca_components"],
                      cfg["knn_k"], cfg["knn_metric"],
                      cfg["random_seed"], cfg)

    print("\n--- 扫描分类个数 ---")
    sweep_num_classes(cfg["dataset_path"], cfg["pca_components"],
                      cfg["knn_k"], cfg["knn_metric"],
                      cfg["train_ratio"], cfg["random_seed"], cfg)

    print("\n--- 扫描距离度量 ---")
    sweep_knn_metric(X_train_pca, y_train, X_test_pca, y_test,
                     cfg["knn_k"], cfg)

    print("全部完成！")

