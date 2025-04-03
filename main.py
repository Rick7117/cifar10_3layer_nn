import os
import numpy as np
from utils.data_loader import load_processed_data
from model import ThreeLayerNN
from train import Trainer
from test import Tester
from param_search import ParamSearcher
import matplotlib.pyplot as plt
from utils.visualization import plot_training_curves, plot_parameter_distribution
import yaml

if __name__ == '__main__':
    # 加载数据
    processed = load_processed_data(['data_batch_1'])
    
    # 参数搜索配置
    # 从配置文件加载参数
    try:        
        with open(os.path.join('configs', 'search_space.yaml')) as f:
            param_grid = yaml.safe_load(f)
    except Exception as e:
        print(f'加载配置文件失败: {e}')
        exit(1)
    print('\n=== 超参数搜索 ===')
    # 执行参数搜索
    searcher = ParamSearcher(
        (processed['train_X'], processed['train_y']),
        (processed['val_X'], processed['val_y'])
    )
    best_params, best_acc = searcher.grid_search(param_grid)
    
    # 数据预处理
    processed = load_processed_data([
        'data_batch_1', 
        'data_batch_2', 
        'data_batch_3', 
        'data_batch_4', 
        'data_batch_5']
        )
    
    # 创建全连接网络
    model = ThreeLayerNN(3072, best_params['hidden_size'], 10)
    trainer = Trainer(
        model=model,
        train_data=(processed['train_X'], processed['train_y']),
        val_data=(processed['val_X'], processed['val_y']),
        lr=best_params['lr'],
        reg_lambda=best_params['reg_lambda'],
        save_path=os.path.join('experiments', 'best_model.npz'), 
        lr_decay=0.95
    )
    
    print('\n=== 模型训练 ===')
    print(f'学习率: {best_params["lr"]}')
    print(f'隐藏层维度: {best_params["hidden_size"]}')
    print(f'正则化系数: {best_params["reg_lambda"]}')
    print(f'训练轮数: {trainer.epochs}')
    print(f'批次大小: {trainer.batch_size}\n')
    
    train_loss, val_acc = trainer.train()
    
    # 可视化训练过程
    plot_training_curves(train_loss, val_acc, os.path.join('figure'))
    
    # 测试最终模型
    print('\n=== 模型测试 ===')
    test_processed = load_processed_data(['test_batch'], ratio=0.0)
    tester = Tester(model, (test_processed['val_X'], test_processed['val_y']), os.path.abspath(os.path.join('experiments', 'best_model.npz')))
    test_acc = tester.test_report()
    
    # 参数可视化
    plot_parameter_distribution(model.params, os.path.join('figure'))