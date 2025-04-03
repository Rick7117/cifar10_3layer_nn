import numpy as np
from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        三层神经网络初始化
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param activation: 激活函数类型(relu/sigmoid)
        """
        self.activation = activation
        
        # 创建网络层
        self.layers = [
            LinearLayer(input_size, hidden_size),
            ReLULayer() if activation == 'relu' else SigmoidLayer(),
            LinearLayer(hidden_size, hidden_size),
            ReLULayer() if activation == 'relu' else SigmoidLayer(),
            LinearLayer(hidden_size, output_size),
            SoftmaxLayer()
        ]
        
        self.model = Sequential(self.layers)
    
    def forward(self, X):
        """前向传播"""
        return self.model.forward(X)
    
    def backward(self, y, reg_lambda=0.01):
        """
        反向传播
        :param y: 真实标签
        :param reg_lambda: L2正则化系数
        :return: 梯度字典
        """
        # 计算输出层梯度
        probs = self.model.layers[-1].output
        delta = probs.copy()
        delta[range(len(y)), y] -= 1
        delta /= len(y)
        
        # 反向传播
        self.model.backward(delta)
        
        # 添加L2正则化
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.grads['W'] += reg_lambda * layer.params['W']
        
        # 收集所有梯度
        grads = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                grads[f'W{i//2+1}'] = layer.grads['W']
                grads[f'b{i//2+1}'] = layer.grads['b']
        
        return grads