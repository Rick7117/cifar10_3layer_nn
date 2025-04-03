import numpy as np
import pickle
from models.neural_net import ThreeLayerNN
from utils.metrics import accuracy
from utils.visualization import visualize_weights

def test_model(test_data, test_labels, model_path):
    """
    测试模型性能
    :param test_data: 测试数据
    :param test_labels: 测试标签
    :param model_path: 模型文件路径
    :return: 测试准确率
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # 初始化模型
    input_size = test_data.shape[1]
    model = ThreeLayerNN(input_size, model_dict['config']['hidden_size'], 10, model_dict['config']['activation'])
    model.params = model_dict['params']
    
    # 前向传播
    probs = model.forward(test_data)
    
    # 计算准确率
    test_acc = accuracy(test_labels, probs)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # 可视化权重
    visualize_weights(model.params['W1'], 'First Layer Weights')
    visualize_weights(model.params['W2'], 'Second Layer Weights')
    visualize_weights(model.params['W3'], 'Output Layer Weights')
    
    return test_acc