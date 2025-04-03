import numpy as np
import os
import pickle
import itertools
from train import train_model
from utils.data_loader import load_cifar10, preprocess_data

def generate_search_space():
    """
    生成超参数搜索空间
    :return: 超参数组合列表
    """
    search_space = {
        'hidden_size': [50, 100, 200],
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'reg_lambda': [0.01, 0.1, 1.0],
        'activation': ['relu', 'sigmoid'],
        'batch_size': [32, 64, 128],
        'num_epochs': [50],
        'lr_decay_every': [20],
        'lr_decay_factor': [0.5],
        'print_every': [10]
    }
    
    # 生成所有可能的组合
    keys = search_space.keys()
    values = (search_space[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    return combinations

def hyperparameter_tuning(data_dir, save_dir):
    """
    执行超参数搜索
    :param data_dir: CIFAR-10数据目录
    :param save_dir: 结果保存目录
    :return: 最佳模型配置
    """
    # 加载数据
    (train_data, train_labels), (test_data, test_labels) = load_cifar10(data_dir)
    
    # 预处理数据
    train_data, train_labels = preprocess_data(train_data, train_labels)
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # 划分验证集(10%训练数据)
    val_size = int(0.1 * train_data.shape[0])
    val_data = train_data[:val_size]
    val_labels = train_labels[:val_size]
    train_data = train_data[val_size:]
    train_labels = train_labels[val_size:]
    
    # 生成搜索空间
    search_space = generate_search_space()
    
    # 结果记录
    results = []
    best_val_acc = 0.0
    best_config = None
    
    # 遍历所有超参数组合
    for i, config in enumerate(search_space):
        print(f'\nTuning config {i+1}/{len(search_space)}: {config}')
        
        # 训练模型
        model, history = train_model(train_data, train_labels, val_data, val_labels, config)
        
        # 记录结果
        result = {
            'config': config,
            'val_accuracy': max(history['val_accuracies']),
            'val_loss': min(history['val_losses']),
            'train_loss': min(history['train_losses'])
        }
        results.append(result)
        
        # 更新最佳配置
        if result['val_accuracy'] > best_val_acc:
            best_val_acc = result['val_accuracy']
            best_config = config
            
            # 保存最佳模型
            save_path = os.path.join(save_dir, 'best_model.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 保存当前结果
        results_path = os.path.join(save_dir, 'tuning_results.csv')
        with open(results_path, 'w') as f:
            f.write('config,val_accuracy,val_loss,train_loss\n')
            for r in results:
                f.write(f"{r['config']},{r['val_accuracy']},{r['val_loss']},{r['train_loss']}\n")
    
    return best_config