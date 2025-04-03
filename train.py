import numpy as np
import os
import pickle
from models.neural_net import ThreeLayerNN
from utils.metrics import accuracy
from utils.visualization import plot_training_curves

def train_model(train_data, train_labels, val_data, val_labels, config):
    """
    训练三层神经网络
    :param train_data: 训练数据
    :param train_labels: 训练标签
    :param val_data: 验证数据
    :param val_labels: 验证标签
    :param config: 训练配置
    :return: 训练好的模型和训练历史
    """
    # 初始化模型
    input_size = train_data.shape[1]
    model = ThreeLayerNN(input_size, config['hidden_size'], 10, config['activation'])
    
    # 训练参数
    learning_rate = config['learning_rate']
    reg_lambda = config['reg_lambda']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model = None
    
    # 训练循环
    for epoch in range(num_epochs):
        # 学习率衰减
        if epoch % config['lr_decay_every'] == 0 and epoch > 0:
            learning_rate *= config['lr_decay_factor']
        
        # 随机打乱数据
        permutation = np.random.permutation(train_data.shape[0])
        train_data_shuffled = train_data[permutation]
        train_labels_shuffled = train_labels[permutation]
        
        # 小批量训练
        for i in range(0, train_data.shape[0], batch_size):
            batch_data = train_data_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            
            # 前向传播
            probs = model.forward(batch_data)
            
            # 计算交叉熵损失
            correct_log_probs = -np.log(probs[range(batch_size), batch_labels])
            data_loss = np.sum(correct_log_probs) / batch_size
            
            # 计算L2正则化损失
            reg_loss = 0.5 * reg_lambda * (
                np.sum(model.params['W1'] ** 2) + 
                np.sum(model.params['W2'] ** 2) + 
                np.sum(model.params['W3'] ** 2)
            )
            
            # 反向传播
            grads = model.backward(batch_labels, reg_lambda)
            
            # 参数更新
            for param in model.params:
                model.params[param] -= learning_rate * grads[f'd{param}']
        
        # 计算训练损失
        train_probs = model.forward(train_data)
        train_correct_log_probs = -np.log(train_probs[range(len(train_labels)), train_labels])
        train_data_loss = np.sum(train_correct_log_probs) / len(train_labels)
        train_reg_loss = 0.5 * reg_lambda * (
            np.sum(model.params['W1'] ** 2) + 
            np.sum(model.params['W2'] ** 2) + 
            np.sum(model.params['W3'] ** 2)
        )
        train_loss = train_data_loss + train_reg_loss
        train_losses.append(train_loss)
        
        # 计算验证集性能
        val_probs = model.forward(val_data)
        val_correct_log_probs = -np.log(val_probs[range(len(val_labels)), val_labels])
        val_data_loss = np.sum(val_correct_log_probs) / len(val_labels)
        val_reg_loss = 0.5 * reg_lambda * (
            np.sum(model.params['W1'] ** 2) + 
            np.sum(model.params['W2'] ** 2) + 
            np.sum(model.params['W3'] ** 2)
        )
        val_loss = val_data_loss + val_reg_loss
        val_losses.append(val_loss)
        
        val_acc = accuracy(val_labels, val_probs)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = {
                'params': {k: v.copy() for k, v in model.params.items()},
                'config': config,
                'epoch': epoch
            }
        
        # 打印训练信息
        if epoch % config['print_every'] == 0:
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    return best_model, {'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accuracies}

def save_model(model, save_path):
    """保存模型到文件"""
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(load_path):
    """从文件加载模型"""
    with open(load_path, 'rb') as f:
        return pickle.load(f)