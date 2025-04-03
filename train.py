import os
import numpy as np

class Trainer:
    def __init__(self, model, train_data, val_data, lr, reg_lambda, save_path, lr_decay=0.95, epochs=50, batch_size=64):
        self.model = model
        self.train_X, self.train_y = train_data
        self.val_X, self.val_y = val_data
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path) + '.npz')
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.best_val_acc = 0
        
        # 创建保存目录
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        # 交叉熵损失
        log_probs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_probs) / m
        # L2正则化
        reg_loss = 0.5 * self.reg_lambda * (
            np.sum(np.square(self.model.params['W1'])) +
            np.sum(np.square(self.model.params['W2'])) +
            np.sum(np.square(self.model.params['W3']))
        )
        return loss + reg_loss

    def train_epoch(self):
        indices = np.arange(self.train_X.shape[0])
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            X_batch = self.train_X[batch_idx]
            y_batch = self.train_y[batch_idx]
            
            # 前向传播
            y_pred = self.model.forward(X_batch)
            
            # 反向传播
            grads = self.model.backward(X_batch, y_batch, self.reg_lambda)
            
            # 参数更新
            for param in self.model.params:
                self.model.params[param] -= self.lr * grads[f'{param}']

    def evaluate(self, X, y):
        y_pred = self.model.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy
    
    def train(self):
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.epochs):
            self.train_epoch()
            
            # 计算训练损失
            y_pred = self.model.forward(self.train_X)
            train_loss = self.compute_loss(y_pred, self.train_y)
            train_losses.append(train_loss)
            
            # 验证集评估
            val_acc = self.evaluate(self.val_X, self.val_y)
            val_accuracies.append(val_acc)
            
            # 学习率衰减
            self.lr *= self.lr_decay
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                np.savez(self.save_path, **self.model.params)
            
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f}")
        
        return train_losses, val_accuracies