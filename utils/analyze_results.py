import os
import numpy as np
import pandas as pd

def summarize_results(exp_root):
    results = []
    
    for exp_name in os.listdir(exp_root):
        exp_dir = os.path.join(exp_root, exp_name)
        result_path = os.path.join(exp_dir, 'results.npz')
        
        if not os.path.exists(result_path):
            continue
            
        try:
            data = np.load(result_path, allow_pickle=True)
            params = data['params'].item()
            
            record = {
                'learning_rate': params['lr'],
                'hidden_size': params['hidden_size'],
                'reg_lambda': params['reg_lambda'],
                'batch_size': params['batch_size'],
                'activation': params['activation_func'],
                'best_val_acc': np.max(data['val_acc']),
                'final_train_loss': data['train_loss'][-1],
            }
            results.append(record)
        except Exception as e:
            print(f'Error loading {exp_name}: {str(e)}')
    
    df = pd.DataFrame(results)
    summary_path = os.path.join(exp_root, 'summary.csv')
    df.to_csv(summary_path, index=False)
    print(f'生成汇总表格，共{len(df)}条实验记录')
    print(f'结果已保存至: experiments/summary.csv')
    return df

def export_to_latex(df, output_path, caption="实验结果汇总", label="tab:exp_results", column_mapping=None):
    """
    将DataFrame导出为LaTeX表格格式
    
    参数:
        df: pandas DataFrame, 包含实验结果
        output_path: str, 输出的LaTeX文件路径
        caption: str, 表格标题
        label: str, 表格标签，用于在LaTeX文档中引用
        column_mapping: dict, 列名映射，将DataFrame的列名映射为LaTeX表格中的列名
    """
    # 如果提供了列名映射，则重命名列
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # 设置数值格式
    formatters = {
        'learning_rate': lambda x: f"{x:.4f}",
        'reg_lambda': lambda x: f"{x:.6f}",
        'best_val_acc': lambda x: f"{x:.4f}",
        'final_train_loss': lambda x: f"{x:.4f}"
    }
    
    # 生成LaTeX表格代码
    latex_code = df.to_latex(
        index=False,
        float_format="%.4f",
        formatters=formatters,
        caption=caption,
        label=label,
        escape=False
    )
    
    # 添加一些LaTeX表格的美化设置
    latex_preamble = "\\usepackage{booktabs}\n\\usepackage{multirow}\n"
    
    # 将LaTeX代码写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% 请在LaTeX文档的导言区添加以下包\n")
        f.write("% " + latex_preamble)
        f.write("\n% 在文档中插入以下代码\n")
        f.write(latex_code)
    
    print(f"LaTeX表格已保存至: {output_path}")
    return latex_code

if __name__ == '__main__':
    df = summarize_results('experiments')
    
    # 导出为LaTeX表格
    column_mapping = {
        'learning_rate': '学习率',
        'hidden_size': '隐藏层大小',
        'reg_lambda': '正则化系数',
        'batch_size': '批量大小',
        'activation': '激活函数',
        'best_val_acc': '最佳验证准确率',
        'final_train_loss': '最终训练损失'
    }
    
    latex_path = os.path.join('experiments', 'results_table.tex')
    export_to_latex(df, latex_path, caption="CIFAR-10三层神经网络实验结果", 
                   label="tab:cifar10_results", column_mapping=column_mapping)
