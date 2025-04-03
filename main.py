import argparse
import os
import pickle
from train import train_model
from utils.data_loader import load_cifar10, preprocess_data

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 3层神经网络训练脚本')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='CIFAR-10数据集目录路径')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                        help='模型保存目录路径')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='隐藏层神经元数量')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--reg_lambda', type=float, default=0.01,
                        help='L2正则化系数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练周期数')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载并预处理数据
    (train_data, train_labels), (test_data, test_labels) = load_cifar10(args.data_dir)
    train_data, train_labels = preprocess_data(train_data, train_labels)
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # 训练配置
    config = {
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'reg_lambda': args.reg_lambda,
        'activation': 'relu',
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr_decay_every': 20,
        'lr_decay_factor': 0.5,
        'print_every': 10
    }
    
    # 训练模型
    model, history = train_model(train_data, train_labels, test_data, test_labels, config)
    
    # 保存模型
    model_path = os.path.join(args.save_dir, 'trained_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f'\n训练完成，模型已保存至：{model_path}')

if __name__ == '__main__':
    main()