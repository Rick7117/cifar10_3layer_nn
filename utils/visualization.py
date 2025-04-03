import matplotlib.pyplot as plt
import os

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
    plt.figure(figsize=(15,10))
    for i, (k, v) in enumerate(params.items()):
        plt.subplot(2,3,i+1)
        plt.hist(v.flatten(), bins=50)
        plt.title(f'{k} Distribution')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'params_distribution.png'))
    plt.close()