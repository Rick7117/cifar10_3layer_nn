import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(weights, layer_name):
    """
    可视化神经网络权重
    :param weights: 权重矩阵
    :param layer_name: 层名称
    """
    plt.figure(figsize=(10, 5))
    plt.title(f'Weight Visualization - {layer_name}')
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Output Units')
    plt.ylabel('Input Units')
    plt.show()

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """
    绘制训练过程中的损失和准确率曲线
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param val_accuracies: 验证准确率列表
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()