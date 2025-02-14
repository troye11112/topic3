# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题


def visualize_clusters(embeddings, labels, method="tsne", sample_rate=1.0):
    """
    可视化聚类结果：
    - 支持 'pca', 'tsne' 或 'umap' 降维（需安装 umap-learn）
    - 当数据量大时，可通过 sample_rate 参数随机抽样部分数据展示
    - 绘制每个簇的质心和标签
    """
    # 如果数据量较大，随机抽样展示部分数据
    if sample_rate < 1.0:
        N = embeddings.shape[0]
        indices = np.random.choice(N, size=int(N * sample_rate), replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # 降维：根据指定方法选择
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == "tsne":
        n_samples = embeddings.shape[0]
        # TSNE 的 perplexity 必须小于样本数量，因此这里动态调整
        perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
        reducer = TSNE(n_components=2, init="pca", random_state=42, perplexity=perplexity)
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP 需要安装 umap-learn，回退使用 PCA")
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
    else:
        print(f"未知降维方法 '{method}'，使用 PCA")
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=10, alpha=0.6)
    plt.colorbar(scatter)
    plt.title("人脸聚类结果")
    plt.xlabel("降维1")
    plt.ylabel("降维2")
    
    # 计算并绘制每个簇的质心与标签
    # unique_labels = np.unique(labels)
    # for label in unique_labels:
    #     cluster_points = reduced[labels == label]
    #     centroid = np.mean(cluster_points, axis=0)
    #     plt.scatter(centroid[0], centroid[1], marker="X", s=200, edgecolors="k", c="red")
    #     plt.text(centroid[0], centroid[1], f" {label}", fontsize=12, color="black")
    
    plt.show()
