import os
import argparse
import numpy as np
from utils.data_loader import load_processed_data
from model import ThreeLayerNN
from train import Trainer
from test import Tester
from param_search import ParamSearcher
from utils.visualization import plot_training_curves, plot_parameter_distribution, plot_parameter_heatmap
from utils.analyze_results import summarize_results, export_to_latex
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10分类器训练管道')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'test', 'visualize', 'param_search'],
                      help='运行模式: train|test|visualize')
    parser.add_argument('--config', type=str, default='configs/base_params.yaml',
                      help='配置文件路径')
    parser.add_argument('--search_space', type=str, default='configs/search_space.yaml',
                        help='配置文件路径')
    parser.add_argument('--model_path', type=str, 
                      default='experiments/best_model.npz',
                      help='测试/可视化模式下的模型路径')
    parser.add_argument('--save_model_path', type=str, 
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
        save_path=args.save_model_path,
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
        with open(args.search_space) as f:
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

    # 汇总搜索结果
    summarize_results('experiments')

def visualize_mode(args):
    # 加载模型参数
    model = ThreeLayerNN(3072, 512, 10)
    model.load_params(args.model_path)
    
    print('\n=== 参数可视化 ===')
    plot_parameter_distribution(model.params, 'figure')
    plot_parameter_heatmap(model.params, 'figure')
    df = summarize_results('experiments')
    
    # 导出为LaTeX表格
    column_mapping = {
        'learning_rate': '学习率',
        'hidden_size': '隐藏层大小',
        'reg_lambda': '正则化系数',
        'batch_size': '批量大小',
        'activation': '激活函数',
        'best_val_acc': '最佳验证准确率',
        'final_train_loss': '最终训练损失'
    }
    
    latex_path = os.path.join('experiments', 'results_table.tex')
    export_to_latex(df, latex_path, caption="CIFAR-10三层神经网络实验结果", 
                   label="tab:cifar10_results", column_mapping=column_mapping)

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