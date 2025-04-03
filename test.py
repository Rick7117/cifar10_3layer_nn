import os
from train import Trainer

class Tester:
    def __init__(self, model, test_data, model_path):
        self.model = model
        self.test_X, self.test_y = test_data
        self.model_path = model_path
        self.trainer = Trainer(
            model=model,
            train_data=(self.test_X, self.test_y), 
            val_data=(self.test_X, self.test_y),
            lr=0.001,
            reg_lambda=0.001,
            save_path=os.path.join(os.path.dirname(self.model_path), 'test_model.npz')
        )

    def test_report(self):
        # 使用Trainer的评估方法
        acc = self.trainer.evaluate(self.test_X, self.test_y)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

    def gradient_check(self, X_sample, epsilon=1e-7):
        # 实现数值梯度检验
        original_params = {k:v.copy() for k,v in self.model.params.items()}
        grad_analytic = self.model.backward(X_sample, self.test_y, 0)
        
        for param_name in self.model.params:
            param = self.model.params[param_name]
            grad_num = np.zeros_like(param)
            
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    old_val = param[i,j]
                    
                    # 正向扰动
                    param[i,j] = old_val + epsilon
                    loss_plus = self.model.compute_loss(self.model.forward(X_sample), self.test_y)
                    
                    # 负向扰动
                    param[i,j] = old_val - epsilon
                    loss_minus = self.model.compute_loss(self.model.forward(X_sample), self.test_y)
                    
                    # 恢复原值
                    param[i,j] = old_val
                    
                    # 计算数值梯度
                    grad_num[i,j] = (loss_plus - loss_minus) / (2*epsilon)
            
            # 计算相对误差
            diff = np.linalg.norm(grad_num - grad_analytic[param_name])
            relative_error = diff / (np.linalg.norm(grad_num) + np.linalg.norm(grad_analytic[param_name]))
            print(f'Gradient check for {param_name}: relative error {relative_error:.2e}')
        
        self.model.params = original_params