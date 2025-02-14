# clustering.py
import numpy as np
import faiss
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
import collections

class IncrementalFaceClustering:
    def __init__(self, embedding_dim=128, eps=0.4, min_samples=3, k=10, use_gpu=False, M=32):
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.min_samples = min_samples
        self.k = k
        self.use_gpu = use_gpu
        self.embeddings = None  # 存放归一化后的向量，形状 (N, embedding_dim)
        self.labels = None

        # 使用内积（faiss.METRIC_INNER_PRODUCT）构建 FAISS 索引，前提：向量需归一化
        self.index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add_embeddings(self, new_embeddings):
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        self.index.add(new_embeddings)
        self.update_clusters()

    def get_neighbors(self, embeddings, k, eps):
        D, I = self.index.search(embeddings, k)
        N = embeddings.shape[0]
        sparse_matrix = sp.lil_matrix((N, N))
        
        # 用于收集调试信息的列表（选填）
        debug_distances = []

        for i in range(N):
            for j in range(1, k):  # 跳过自身
                cosine_sim = D[i, j]
                cosine_dist = max(0, 1 - cosine_sim)  # 保证距离非负
                # 如果需要打印部分数据的距离信息，可以限制只打印前几个样本
                if i < 5:  # 例如，只打印前 5 个样本的邻居信息
                    print(f"样本 {i} 的邻居 {I[i, j]}: cosine_sim = {cosine_sim:.4f}, cosine_dist = {cosine_dist:.4f}")
                    debug_distances.append(cosine_dist)
                if cosine_dist < eps:
                    sparse_matrix[i, I[i, j]] = cosine_dist

        # 输出一些统计信息（可选）
        print(f"非零元素数量: {sparse_matrix.nnz}")
        return sparse_matrix, debug_distances

    def update_clusters(self):
        sparse_matrix, debug_distances = self.get_neighbors(self.embeddings, self.k, self.eps)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        self.labels = dbscan.fit_predict(sparse_matrix)
        self.labels = self.refine_clusters(self.labels)
        self.labels = self.assign_outliers(self.labels)

    def refine_clusters(self, labels):
        cluster_count = collections.Counter(labels[labels != -1])
        if not cluster_count:
            return labels
        dominant_cluster = max(cluster_count, key=cluster_count.get)
        for i, label in enumerate(labels):
            if label == -1:
                labels[i] = dominant_cluster
        return labels

    def assign_outliers(self, labels):
        unclassified = np.where(labels == -1)[0]
        if len(unclassified) == 0:
            return labels
        D, I = self.index.search(self.embeddings[unclassified], 5)
        for idx, sample_idx in enumerate(unclassified):
            for neighbor in I[idx]:
                if labels[neighbor] != -1:
                    labels[sample_idx] = labels[neighbor]
                    break
        return labels

