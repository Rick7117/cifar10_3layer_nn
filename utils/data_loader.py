import numpy as np
import pickle
import os

def load_cifar10_batch(file_path):
    """加载单个CIFAR-10批次文件"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    
    return features, labels

def load_cifar10(data_dir):
    """加载完整的CIFAR-10数据集"""
    train_data = []
    train_labels = []
    
    # 加载训练数据
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        features, labels = load_cifar10_batch(file_path)
        train_data.append(features)
        train_labels.append(labels)
    
    # 合并训练数据
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    
    # 加载测试数据
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    
    # 归一化像素值到[0,1]
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0
    
    return (train_data, train_labels), (test_data, test_labels)

def preprocess_data(data, labels, num_classes=10):
    """数据预处理：展平图像并one-hot编码标签"""
    # 展平图像 (N, 32, 32, 3) -> (N, 3072)
    flattened_data = data.reshape(data.shape[0], -1)
    
    # One-hot编码标签
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    
    return flattened_data, one_hot_labels