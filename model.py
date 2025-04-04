import numpy as np
import yaml
import os

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("激活函数必须是'relu','sigmoid'或'tanh'")
        
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, hidden_size) * np.sqrt(2./hidden_size),
            'b2': np.zeros(hidden_size),
            'W3': np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size),
            'b3': np.zeros(output_size)
        }
        self.activation = activation
        
        # 定义激活函数方法
        # self.relu = lambda x: np.maximum(0, x)
        # self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # self.tanh = lambda x: np.tanh(x)
        self.cache = {}
        # 记录网络架构参数用于验证
        self.architecture = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size
        }

    def load_params(self, file_path):

        """
        从npz文件加载模型参数
        :param file_path: 参数文件路径
        """
        try:
            # 加载NPZ参数文件
            params = np.load(file_path)
            
            # 验证必须参数是否存在
            required_keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
            for key in required_keys:
                if key not in params:
                    raise ValueError(f'缺失关键参数: {key}')

            # 验证参数维度
            if params['W1'].shape != (self.architecture['input_size'], self.architecture['hidden_size']):
                raise ValueError('W1维度不匹配')
            if params['b1'].shape != (self.architecture['hidden_size'],):
                raise ValueError('b1维度不匹配')
            if params['W2'].shape != (self.architecture['hidden_size'], self.architecture['hidden_size']):
                raise ValueError('W2维度不匹配')
            if params['b2'].shape != (self.architecture['hidden_size'],):
                raise ValueError('b2维度不匹配')
            if params['W3'].shape != (self.architecture['hidden_size'], self.architecture['output_size']):
                raise ValueError('W3维度不匹配')
            if params['b3'].shape != (self.architecture['output_size'],):
                raise ValueError('b3维度不匹配')

            # 更新模型参数
            self.params = {key: params[key] for key in params.files}
            
        except Exception as e:
            print(f'参数加载失败: {e}')
            raise

    # 新增激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1 - np.square(self.tanh(x))

    def relu(self, x):
        """ReLU函数"""
        return np.maximum(0, x)

    def relu_deriv(self, x):
        """ReLU函数的导数"""
        return (x > 0).astype(np.float32)

    # 修改forward传播
    def forward(self, X):
        # Layer 1
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = self.relu(z1) if self.activation == 'relu' else \
             self.sigmoid(z1) if self.activation == 'sigmoid' else \
             self.tanh(z1)
        
        # Layer 2
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        a2 = self.relu(z2) if self.activation == 'relu' else \
             self.sigmoid(z2) if self.activation == 'sigmoid' else \
             self.tanh(z2)
        
        # Output layer
        z3 = a2.dot(self.params['W3']) + self.params['b3']
        # 输出层应用softmax
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        self.cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}
        return a3

    # 修改backward传播
    def backward(self, X, y, reg_lambda):
        m = X.shape[0]
        grads = {}
        
        # 根据激活函数选择导数计算方法
        deriv_func = {
            'relu': self.relu_deriv,
            'sigmoid': self.sigmoid_deriv,
            'tanh': self.tanh_deriv
        }[self.activation]

        # Output layer gradients
        delta3 = self.cache['a3'] - y
        grads['W3'] = self.cache['a2'].T.dot(delta3)/m + reg_lambda*self.params['W3']
        grads['b3'] = np.sum(delta3, axis=0)/m
        
        # Hidden layer 2 gradients
        delta2 = delta3.dot(self.params['W3'].T) * deriv_func(self.cache['z2'])
        grads['W2'] = self.cache['a1'].T.dot(delta2)/m + reg_lambda*self.params['W2']
        grads['b2'] = np.sum(delta2, axis=0)/m
        
        # Hidden layer 1 gradients
        delta1 = delta2.dot(self.params['W2'].T) * deriv_func(self.cache['z1'])
        grads['W1'] = X.T.dot(delta1)/m + reg_lambda*self.params['W1']
        grads['b1'] = np.sum(delta1, axis=0)/m
        
        return grads