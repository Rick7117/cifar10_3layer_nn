import numpy as np
import itertools
from train import Trainer
from model import ThreeLayerNN

import os
import yaml

class ParamSearcher:
    def __init__(self, train_data, val_data, base_save_dir=None):
        # if base_save_dir is None:
        #     base_save_dir = os.path.join('experiments')
        self.train_data = train_data
        self.val_data = val_data
        # self.base_save_dir = base_save_dir
        # os.makedirs(base_save_dir, exist_ok=True)

    def grid_search(self, param_grid):
        best_acc = 0
        best_params = {}
        
        # 生成参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        for params in combinations:
            # 创建实验目录
            # exp_name = f"lr{params['lr']}_h{params['hidden_size']}_reg{params['reg_lambda']}_batchSize{params['batch_size']}_activation{params['activation_func']}"
            print(f'\n=== 超参数探索 ===')
            print(f'学习率: {params["lr"]}')
            print(f'隐藏层维度: {params["hidden_size"]}')
            print(f'正则化系数: {params["reg_lambda"]}')
            print(f'批次大小: {params["batch_size"]}')
            print(f'激活函数: {params["activation_func"]}')

            # save_path = os.path.join(self.base_save_dir, exp_name, 'model.npz')
            
            # 初始化模型
            model = ThreeLayerNN(
                    input_size=3072,
                    hidden_size=params['hidden_size'],
                    output_size=10,
                    activation=params['activation_func']
                )
            
            # 训练配置
            trainer = Trainer(
                model=model,
                train_data=self.train_data,
                val_data=self.val_data,
                lr=params['lr'],
                reg_lambda=params['reg_lambda'],
                save_path=None,
                batch_size=params['batch_size']
            )
            
            # 执行训练
            train_loss, val_acc = trainer.train()
            
            # 记录最佳参数
            if max(val_acc) > best_acc:
                best_acc = max(val_acc)
                best_params = params
                
            # 保存训练结果，包括训练损失、验证准确率和超参数
            # np.savez(
            #     os.path.join(self.base_save_dir, exp_name, 'results.npz'),
            #     train_loss=train_loss,
            #     val_acc=val_acc,
            #     params=params
            # )

        # 保存最佳参数到configs目录
        self.save_best_params(best_params)
        
        return best_params, best_acc

    def save_best_params(self, best_params):
        os.makedirs('configs', exist_ok=True)
        with open(os.path.join('configs', 'best_params.yaml'), 'w') as f:
            yaml.dump(best_params, f)

    def load_results(self, exp_dir):
        results = np.load(os.path.join(exp_dir, 'results.npz'), allow_pickle=True)
        return {
            'train_loss': results['train_loss'],
            'val_acc': results['val_acc'],
            'params': results['params'].item()
        }