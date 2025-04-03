import numpy as np
import os

import pickle

from tqdm import tqdm


def load_cifar10(batch_names, save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')):
    X_list = []
    y_list = [] 
    
    for batch in batch_names:
        with open(os.path.join(save_dir,  batch), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        X_list.append(data_dict[b'data'])
        y_list.append(data_dict[b'labels'])
    
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y

def preprocess_data(X, y, ratio=0.8):
    # 归一化
    X = X.astype('float32') / 255.0
    
    # 独热编码
    y_onehot = np.eye(10)[y.flatten()]
    
    # 展平数据供全连接网络使用
    X = X.reshape(X.shape[0], -1)
    
    # 划分训练验证集
    indices = np.random.permutation(X.shape[0])
    split = int(ratio * X.shape[0])
    
    # 打印数据集统计信息
    total_samples = X.shape[0]
    print(f'训练集：{split}样本（{split/total_samples*100:.2f}%） \
        验证集：{total_samples-split}样本（{(total_samples-split)/total_samples*100:.2f}%）')
    
    return {
        'train_X': X[indices[:split]],
        'train_y': y_onehot[indices[:split]],
        'val_X': X[indices[split:]],
        'val_y': y_onehot[indices[split:]]
    }

def load_processed_data(batch_names, ratio=0.8):
    """
    加载并预处理CIFAR-10数据
    Args:
        batch_names: 要加载的数据批次列表
        ratio: 训练集划分比例
    Returns:
        预处理后的数据字典，包含训练/验证集的X和y
    """
    X, y = load_cifar10(batch_names)
    return preprocess_data(X, y, ratio)