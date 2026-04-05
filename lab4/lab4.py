import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# 设置中文字体（防止绘图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 实验配置区域
# ============================================================
CONFIG = {
    'data_path': 'ORL',         # 数据集路径
    'num_classes': 40,          # 类别数
    'train_ratio': 0.8,         # 训练集比例
    'pca_components': 50,       # 基准 PCA 降维维度
    'knn_k': 1,                 # 基准 KNN 邻居数
    'knn_metric': 'cosine',     # 基准 KNN 距离度量
    'random_seed': 42           # 随机种子
}

def load_orl_faces(data_path, num_classes=40):
    """加载 ORL 人脸数据集，支持 .bmp 格式"""
    X, y = [], []
    h, w = 112, 92  # ORL 标准尺寸
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据集目录: {data_path}，请检查路径是否正确。")

    print(f"正在从 {data_path} 加载数据 (类别数: {num_classes})...")
    for i in range(1, num_classes + 1):
        person_path = os.path.join(data_path, f's{i}')
        if not os.path.exists(person_path):
            continue
            
        for j in range(1, 11):
            # 兼容性处理：优先读取 .bmp，找不到再试 .pgm
            img_path = os.path.join(person_path, f'{j}.bmp')
            if not os.path.exists(img_path):
                img_path = os.path.join(person_path, f'{j}.pgm')
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img.flatten())
                y.append(i)
    
    return np.array(X), np.array(y), h, w

def visualize_advanced_results(pca, X_train_pca, y_train, X_test, h, w):
    """生成高级可视化图表：特征空间、特征脸、图像重构"""
    print("正在生成高级可视化图表...")
    
    # --- 1. 特征空间可视化 (2D & 3D) ---
    plt.figure(figsize=(15, 6))
    
    # 2D 投影
    plt.subplot(1, 2, 1)
    unique_labels = np.unique(y_train)
    display_classes = min(10, len(unique_labels))
    for i in range(display_classes):
        label = unique_labels[i]
        indices = np.where(y_train == label)
        plt.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], label=f'Class {label}', s=30, alpha=0.7)
    plt.title(f"PCA 2D 投影 (前 {display_classes} 类)")
    plt.xlabel("主成分 1"); plt.ylabel("主成分 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # 3D 投影
    ax = plt.subplot(1, 2, 2, projection='3d')
    for i in range(min(5, len(unique_labels))):
        label = unique_labels[i]
        indices = np.where(y_train == label)
        ax.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], X_train_pca[indices, 2], label=f'Class {label}')
    ax.set_title("PCA 3D 投影 (前 5 类)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    
    plt.tight_layout()
    plt.savefig("pca_feature_space.png", dpi=300)
    plt.close()

    # --- 2. 特征脸 (Eigenfaces) 展示 ---
    n_eigenfaces = 12
    eigenfaces = pca.components_[:n_eigenfaces]
    
    plt.figure(figsize=(12, 8))
    plt.suptitle("前 12 个特征脸 (Eigenfaces)", fontsize=16)
    for i in range(n_eigenfaces):
        plt.subplot(3, 4, i + 1)
        plt.imshow(eigenfaces[i].reshape(h, w), cmap='gray')
        plt.title(f"特征脸 {i+1}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("eigenfaces_display.png", dpi=300)
    plt.close()

    # --- 3. 重建图像对比 ---
    sample_idx = 0 
    original_img = X_test[sample_idx]
    components = pca.transform(original_img.reshape(1, -1))
    reconstructed_img = pca.inverse_transform(components)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img.reshape(h, w), cmap='gray')
    plt.title("原始图像"); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img.reshape(h, w), cmap='gray')
    plt.title(f"PCA 重构图像 (维度={pca.n_components})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("image_reconstruction_comparison.png", dpi=300)
    plt.close()

def run_experiment(X_train, X_test, y_train, y_test, n_components, k, metric, h, w, visualize=False):
    """运行单次实验逻辑"""
    # PCA 降维 (这里确保 n_components 不超过样本数)
    actual_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=actual_components, whiten=True, svd_solver="full")
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # KNN 分类
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train_pca, y_train)
    
    # 预测
    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    if visualize:
        visualize_advanced_results(pca, X_train_pca, y_train, X_test, h, w)
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title(f"混淆矩阵 (准确率: {acc:.4f})")
        plt.xlabel("预测类别"); plt.ylabel("真实类别")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.close()
        
    return acc

def main():
    # 1. 加载数据
    try:
        X, y, h, w = load_orl_faces(CONFIG['data_path'], CONFIG['num_classes'])
        print(f"数据集加载完毕: {len(X)} 张图片, 尺寸 {h}x{w}")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 基准划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=CONFIG['train_ratio'], random_state=CONFIG['random_seed'], stratify=y
    )

    # 3. 基准实验
    print("\n--- 正在运行基准配置实验 ---")
    default_acc = run_experiment(
        X_train, X_test, y_train, y_test, 
        CONFIG['pca_components'], CONFIG['knn_k'], CONFIG['knn_metric'], 
        h, w, visualize=True
    )
    print(f"基准参数准确率: {default_acc:.4f}")

    # 4. 全参数扫描
    print("\n[开始全参数扫描...]")

    # --- PCA 维度扫描 ---
    print("扫描 PCA 维度...")
    pca_dims = range(5, 101, 5)
    pca_accs = [run_experiment(X_train, X_test, y_train, y_test, d, CONFIG['knn_k'], CONFIG['knn_metric'], h, w) for d in pca_dims]
    plt.figure(figsize=(8, 5))
    plt.plot(pca_dims, pca_accs, marker='o', color='tab:blue')
    plt.title("PCA 维度对准确率的影响"); plt.xlabel("维度"); plt.ylabel("准确率"); plt.grid(True)
    plt.savefig("pca_dimension_accuracy.png"); plt.close()

    # --- KNN K值扫描 ---
    print("扫描 KNN K值...")
    k_list = range(1, 16)
    k_accs = [run_experiment(X_train, X_test, y_train, y_test, CONFIG['pca_components'], k, CONFIG['knn_metric'], h, w) for k in k_list]
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, k_accs, marker='o', color='tab:red')
    plt.title("KNN K值对准确率的影响"); plt.xlabel("K值"); plt.ylabel("准确率"); plt.grid(True)
    plt.savefig("knn_k_accuracy.png"); plt.close()

    # --- 训练集比例扫描 ---
    print("扫描 训练集比例...")
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio_accs = []
    for r in ratios:
        tr_X, te_X, tr_y, te_y = train_test_split(X, y, train_size=r, random_state=CONFIG['random_seed'], stratify=y)
        acc = run_experiment(tr_X, te_X, tr_y, te_y, CONFIG['pca_components'], CONFIG['knn_k'], CONFIG['knn_metric'], h, w)
        ratio_accs.append(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(ratios, ratio_accs, marker='o', color='tab:green')
    plt.title("训练集比例对准确率的影响"); plt.xlabel("比例"); plt.ylabel("准确率"); plt.grid(True)
    plt.savefig("train_ratio_accuracy.png"); plt.close()

    # --- 距离度量对比 ---
    print("扫描 距离度量...")
    metrics = ["euclidean", "manhattan", "cosine", "chebyshev"]
    metric_accs = [run_experiment(X_train, X_test, y_train, y_test, CONFIG['pca_components'], CONFIG['knn_k'], m, h, w) for m in metrics]
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, metric_accs, color=['#4A90D9', '#E8524A', '#50C878', '#FFB347'])
    plt.title("距离度量对准确率的影响"); plt.ylabel("准确率"); plt.ylim(min(metric_accs) - 0.05, 1.0)
    plt.savefig("knn_metric_accuracy.png"); plt.close()

    # --- 分类个数扫描 (已修复维度冲突问题) ---
    print("扫描 分类个数...")
    class_list = range(6, 41, 2) 
    class_accs = []
    for nc in class_list:
        sub_X, sub_y, _, _ = load_orl_faces(CONFIG['data_path'], num_classes=nc)
        tr_X, te_X, tr_y, te_y = train_test_split(sub_X, sub_y, train_size=CONFIG['train_ratio'], random_state=CONFIG['random_seed'], stratify=sub_y)
        # 确保 pca_components 不超过当前子集的训练样本数
        safe_dim = min(CONFIG['pca_components'], tr_X.shape[0])
        acc = run_experiment(tr_X, te_X, tr_y, te_y, safe_dim, CONFIG['knn_k'], CONFIG['knn_metric'], h, w)
        class_accs.append(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(class_list, class_accs, marker='o', color='tab:purple')
    plt.title("分类类别数对准确率的影响"); plt.xlabel("类别数"); plt.ylabel("准确率"); plt.grid(True)
    plt.savefig("num_classes_accuracy.png"); plt.close()

    print("\n" + "="*40)
    print("所有实验已完成！共生成 9 张图表。")
    print("="*40)

if __name__ == "__main__":
    main()