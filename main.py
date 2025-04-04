import os
import argparse
import numpy as np
from utils.data_loader import load_processed_data
from model import ThreeLayerNN
from train import Trainer
from test import Tester
from param_search import ParamSearcher
from utils.visualization import plot_training_curves, plot_parameter_distribution
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10分类器训练管道')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'test', 'visualize', 'param_search'],
                      help='运行模式: train|test|visualize')
    parser.add_argument('--config', type=str, default='configs/best_params.yaml',
                      help='配置文件路径')
    parser.add_argument('--model_path', type=str, 
                      default='experiments/best_model.npz',
                      help='测试/可视化模式下的模型路径')
    return parser.parse_args()

def train_mode(args):
    # 参数搜索配置
    try:
        with open(args.config) as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f'加载配置文件失败: {e}')
        exit(1)

    # # 初始化参数搜索
    # processed = load_processed_data(['data_batch_1'])
    # searcher = ParamSearcher(
    #     (processed['train_X'], processed['train_y']),
    #     (processed['val_X'], processed['val_y'])
    # )
    # best_params, best_acc = searcher.grid_search(param_grid)

    # 全量数据加载
    processed = load_processed_data(['data_batch_1','data_batch_2','data_batch_3',
                                    'data_batch_4','data_batch_5'])

    # 模型训练
    model = ThreeLayerNN(3072, params['hidden_size'], 10, activation=params['activation_func'])
    trainer = Trainer(
        model=model,
        train_data=(processed['train_X'], processed['train_y']),
        val_data=(processed['val_X'], processed['val_y']),
        lr=params['lr'],
        reg_lambda=params['reg_lambda'],
        save_path=os.path.join('experiments', 'best_model.npz'),
        lr_decay=0.95
    )
    
    print('\n=== 模型训练 ===')
    train_loss, val_acc = trainer.train()
    plot_training_curves(train_loss, val_acc, 'figure')

def test_mode(args):
    # 加载配置文件
    try:
        with open(args.config) as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f'加载配置文件失败: {e}')
        exit(1)

    # 加载测试数据
    test_data = load_processed_data(['test_batch'], ratio=0.0)
    
    # 初始化模型
    model = ThreeLayerNN(3072, params['hidden_size'], 10, activation=params.get('activation_func', 'relu'))
    model.load_params(args.model_path)
    tester = Tester(model, (test_data['val_X'], test_data['val_y']), args.model_path)
    
    print('\n=== 模型测试 ===')
    tester.test_report()

def param_search_mode(args):
    # 加载参数搜索配置
    try:
        with open(os.path.join('configs', 'search_space.yaml')) as f:
            param_grid = yaml.safe_load(f)
    except Exception as e:
        print(f'加载配置文件失败: {e}')
        exit(1)

    # 加载少量数据
    processed = load_processed_data(['data_batch_1'])

    print('\n=== 超参数搜索 ===')
    searcher = ParamSearcher(
        (processed['train_X'], processed['train_y']),
        (processed['val_X'], processed['val_y'])
    )
    best_params, best_acc = searcher.grid_search(param_grid)

    # 保存最佳参数
    best_params_path = os.path.join('configs', 'best_params.yaml')
    with open(best_params_path, 'w') as f:
        yaml.safe_dump(best_params, f)
    print(f'最佳参数已保存至：{best_params_path}')
    print(f'验证集准确率：{best_acc:.2%}')

def visualize_mode(args):
    # 加载模型参数
    model = ThreeLayerNN(3072, 512, 10)
    model.load_params(args.model_path)
    
    print('\n=== 参数可视化 ===')
    plot_parameter_distribution(model.params, 'figure')

if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'visualize':
        visualize_mode(args)
    elif args.mode == 'param_search':
        param_search_mode(args)
    else:
        print(f'未知模式: {args.mode}')
        exit(1)
    # # 加载数据
    # processed = load_processed_data(['data_batch_1'])
    
    # # 参数搜索配置
    # # 从配置文件加载参数
    # try:        
    #     with open(os.path.join('configs', 'search_space.yaml')) as f:
    #         param_grid = yaml.safe_load(f)
    # except Exception as e:
    #     print(f'加载配置文件失败: {e}')
    #     exit(1)
    # print('\n=== 超参数搜索 ===')
    # # 执行参数搜索
    # searcher = ParamSearcher(
    #     (processed['train_X'], processed['train_y']),
    #     (processed['val_X'], processed['val_y'])
    # )
    # best_params, best_acc = searcher.grid_search(param_grid)
    
    # # 数据预处理
    # processed = load_processed_data([
    #     'data_batch_1', 
    #     'data_batch_2', 
    #     'data_batch_3', 
    #     'data_batch_4', 
    #     'data_batch_5']
    #     )
    
    # # 创建全连接网络
    # model = ThreeLayerNN(3072, best_params['hidden_size'], 10)
    # trainer = Trainer(
    #     model=model,
    #     train_data=(processed['train_X'], processed['train_y']),
    #     val_data=(processed['val_X'], processed['val_y']),
    #     lr=best_params['lr'],
    #     reg_lambda=best_params['reg_lambda'],
    #     save_path=os.path.join('experiments', 'best_model.npz'), 
    #     lr_decay=0.95
    # )
    
    # print('\n=== 模型训练 ===')
    # print(f'学习率: {best_params["lr"]}')
    # print(f'隐藏层维度: {best_params["hidden_size"]}')
    # print(f'正则化系数: {best_params["reg_lambda"]}')
    # print(f'训练轮数: {trainer.epochs}')
    # print(f'批次大小: {trainer.batch_size}\n')
    
    # train_loss, val_acc = trainer.train()
    
    # # 可视化训练过程
    # plot_training_curves(train_loss, val_acc, os.path.join('figure'))
    
    # # 测试最终模型
    # print('\n=== 模型测试 ===')
    # test_processed = load_processed_data(['test_batch'], ratio=0.0)
    # tester = Tester(model, (test_processed['val_X'], test_processed['val_y']), os.path.abspath(os.path.join('experiments', 'best_model.npz')))
    # test_acc = tester.test_report()