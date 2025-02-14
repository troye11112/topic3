# main.py
from clustering import IncrementalFaceClustering
from data_loader import load_features_from_folder
from visualization import visualize_clusters

def main():
    folder_path = r"C:\Users\honey\Desktop\bionta_human_face\5face"
    
    # 加载人脸特征数据
    features = load_features_from_folder(folder_path)
    
    if features.size == 0:
        print("没有加载到任何人脸数据，请检查 JSON 文件格式和路径。")
        return
    if features.ndim < 2:
        print(f"加载到 {features.shape[0]} 条人脸数据，但未检测到特征维度。")
        return
    else:
        print(f"加载 {features.shape[0]} 条人脸数据，每条数据 {features.shape[1]} 维。")
    
    clustering = IncrementalFaceClustering(embedding_dim=features.shape[1],
                                             eps=0.5, min_samples=2, k=10, use_gpu=False)
    clustering.add_embeddings(features)
    visualize_clusters(clustering.embeddings, clustering.labels)

if __name__ == "__main__":
    main()
