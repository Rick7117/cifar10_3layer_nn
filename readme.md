# CIFAR-10 三层神经网络分类器

这是一个使用纯NumPy实现的三层神经网络，用于CIFAR-10图像分类任务。该项目不依赖于PyTorch或TensorFlow等深度学习框架，旨在展示神经网络的基本原理和实现。

## 项目结构

```
cifar10_3layer_nn/
├── models/                    # 模型相关代码
│   ├── __init__.py
│   ├── neural_net.py          # 三层神经网络实现
│   └── layers.py              # 激活函数等组件
├── utils/                     # 工具函数
│   ├── data_loader.py         # CIFAR-10数据加载与预处理
│   ├── metrics.py             # 准确率等计算
│   └── visualization.py       # 权重可视化
├── configs/                   # 超参数配置
│   ├── search_space.yaml      # 参数搜索空间
│   └── best_params.yaml       # 最优参数配置
├── experiments/               # 实验记录
│   ├── best_model.pkl         # 最优模型权重
├── main.py                    # 主程序入口
├── train.py                   # 训练流程
├── test.py                    # 测试流程
├── param_search.py   # 参数搜索
└── requirements.txt           # 依赖库
```

## 环境配置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/cifar10_3layer_nn.git
cd cifar10_3layer_nn
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

CIFAR-10数据集需要手动下载，从[官方网站](https://www.cs.toronto.edu/~kriz/cifar.html)获取。

## 训练模型

### 使用默认参数训练

```bash
python main.py --mode train
```

### 使用自定义参数训练

```bash
python main.py --mode train --lr 0.01 --hidden_size 512 --batch_size 32 --epochs 50
```

### 使用配置文件训练

```bash
python main.py --mode train --config configs/best_params.yaml
```

## 测试模型

```bash
python main.py --mode test --model_path experiments/best_model.npz
```

## 超参数调优

```bash
python parameter_tuning.py
```

## 最优参数

根据我们的实验，最优参数配置如下：

- 激活函数: ReLU
- 批量大小: 32
- 隐藏层大小: 512
- 学习率: 0.01
- 正则化系数: 0.001

## 可视化

训练过程中的损失和准确率曲线会自动保存在`figures`目录下。您也可以使用以下命令可视化模型权重：

```bash
python main.py --mode visualize --model_path experiments/best_model.pkl
```

## 性能

在CIFAR-10测试集上，我们的三层神经网络达到了约52.86%的准确率，这对于没有使用卷积层的简单网络来说是一个不错的结果。

## 贡献

欢迎提交问题和拉取请求，共同改进这个项目！

## 许可证

MIT