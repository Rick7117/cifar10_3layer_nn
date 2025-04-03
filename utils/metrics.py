import numpy as np

def accuracy(labels, probs):
    """
    计算分类准确率
    :param labels: 真实标签 (N,)
    :param probs: 预测概率 (N, C)
    :return: 准确率
    """
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels)