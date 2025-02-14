# data_loader.py
import os
import json
import base64
import numpy as np
import glob

def load_features_from_json(json_file):
    """
    读取单个 JSON 文件，解析其中的 "feature" 字段（base64 编码的 float32 数组），
    返回二维 numpy 数组，每行代表一个人脸的特征向量。
    支持 JSON 文件顶层为列表或单个对象。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果文件顶层是字典，则将其转为单元素列表
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        records = []

    features = []
    for record in records:
        if "feature" in record:
            b64_str = record["feature"]
            try:
                b = base64.b64decode(b64_str)
                arr = np.frombuffer(b, dtype=np.float32)
                features.append(arr)
            except Exception as e:
                print(f"文件 {json_file} 中解析 feature 时出错: {e}")
    if features:
        features = np.stack(features)
        print(f"文件 {json_file} 加载后特征形状：{features.shape}")
    else:
        # 如果没有解析到任何 feature，返回空数组，形状设为 (0, )
        features = np.empty((0,))
    return features

def load_features_from_folder(folder_path):
    """
    递归遍历指定文件夹下所有 JSON 文件，解析每个 JSON 文件中的人脸特征，
    并将所有特征合并成一个二维 numpy 数组返回。
    """
    # 使用递归搜索，匹配所有子文件夹中的 *.json 文件
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
    all_features = []
    for json_file in json_files:
        features = load_features_from_json(json_file)
        # 确保 features 非空且为二维数组
        if features.size > 0 and features.ndim == 2:
            all_features.append(features)
    if all_features:
        all_features = np.concatenate(all_features, axis=0)
    else:
        # 如果没有加载到数据，返回空数组并指定第二维（比如 128）可以根据实际情况调整
        all_features = np.empty((0, 128))
    return all_features
