import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_curves(train_loss, val_acc, save_dir):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def plot_parameter_distribution(params, save_dir):
    # 将参数分为权重(W)和偏置(b)两类
    w_params = {k: v for k, v in params.items() if k.startswith('W')}
    b_params = {k: v for k, v in params.items() if k.startswith('b')}
    
    # 计算每类参数的数量，确定子图列数
    w_count = len(w_params)
    b_count = len(b_params)
    cols = max(w_count, b_count)
    
    # 创建足够大的图形以容纳所有子图
    plt.figure(figsize=(cols*5, 10))
    
    # 第一行绘制权重参数
    for i, (k, v) in enumerate(w_params.items()):
        plt.subplot(2, cols, i+1)
        plt.hist(v.flatten(), bins=50)
        plt.title(f'{k} Distribution')
    
    # 第二行绘制偏置参数
    for i, (k, v) in enumerate(b_params.items()):
        plt.subplot(2, cols, cols+i+1)
        plt.hist(v.flatten(), bins=50)
        plt.title(f'{k} Distribution')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'params_distribution.png'))
    plt.close()


def plot_parameter_heatmap(params, save_dir):
    """Plot heatmaps for weight matrices W and bias vectors b, combining related parameters
    (W1/b1, W2/b2, W3/b3) in the same figure with a 2x1 layout.
    
    Args:
        params (dict): Dictionary containing model parameters, e.g. {'W1': w1_array, 'b1': b1_array, ...}
        save_dir (str): Directory path to save the images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group related parameters
    param_groups = {}
    for name, param in params.items():
        if name.startswith('W'):
            layer_num = name[1:]  # Get the layer number
            if layer_num not in param_groups:
                param_groups[layer_num] = {}
            param_groups[layer_num]['W'] = param
        elif name.startswith('b'):
            layer_num = name[1:]  # Get the layer number
            if layer_num not in param_groups:
                param_groups[layer_num] = {}
            param_groups[layer_num]['b'] = param
    
    # Plot each group (W and b) together
    for layer_num, group in param_groups.items():
        if 'W' in group and 'b' in group:
            # Get parameters
            W = group['W']
            b = group['b']
            
            # Calculate figure size based on weight matrix
            height, width = W.shape
            aspect_ratio = min(max(width / height, 0.25), 4.0)
            figsize = (max(10, min(20, 12 * aspect_ratio)), max(12, min(20, 12)))
            
            # Create figure with 2x1 subplot layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot weight matrix
            im1 = ax1.imshow(W, cmap='viridis')
            ax1.set_title(f'Layer {layer_num} Weight Matrix (shape: {W.shape})')
            ax1.set_xlabel('Output Dimension')
            ax1.set_ylabel('Input Dimension')
            plt.colorbar(im1, ax=ax1)
            
            # Process and plot bias vector
            if len(b.shape) == 1:
                if len(b) > 100:
                    # Reshape long vectors into a matrix
                    cols = int(np.sqrt(len(b)))
                    rows = (len(b) + cols - 1) // cols
                    temp_array = np.zeros(rows * cols)
                    temp_array[:len(b)] = b
                    b_2d = temp_array.reshape(rows, cols)
                else:
                    b_2d = b.reshape(-1, 1)
            else:
                b_2d = b
            
            im2 = ax2.imshow(b_2d, cmap='viridis')
            if len(b.shape) == 1:
                ax2.set_title(f'Layer {layer_num} Bias Vector {b.shape} reshape to ({cols}, {rows})')
            else:
                ax2.set_title(f'Layer {layer_num} Bias Vector {b.shape}')
            ax2.set_xlabel('Dimension')
            ax2.set_ylabel('Sample')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'param_heatmap_layer{layer_num}.png'))
            plt.close()