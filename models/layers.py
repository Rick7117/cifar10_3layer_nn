import numpy as np

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_grad(x):
    """ReLU梯度计算"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    """Sigmoid梯度计算"""
    s = sigmoid(x)
    return s * (1 - s)

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad):
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis=0)
        return np.dot(grad, self.W.T)

class ReLULayer:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x
        return relu(x)
    
    def backward(self, grad):
        return grad * relu_grad(self.x)

class SigmoidLayer:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x
        return sigmoid(x)
    
    def backward(self, grad):
        return grad * sigmoid_grad(self.x)

class SoftmaxLayer:
    def __init__(self):
        self.y = None
    
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y
    
    def backward(self, grad):
        return self.y - grad