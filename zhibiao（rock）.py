import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, cohen_kappa_score, log_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


rock_names = {
    '花岗岩': 'Granite',
    '片麻岩': 'Gneiss',
    '片岩': 'Schist',
    '白云岩': 'Dolomite',
    '石灰岩': 'Limestone',
    '凝灰岩': 'Tuff',
    '砂岩': 'Sandstone',
    '辉长岩': 'Gabbro',
    '玄武岩': 'Basalt',
}


def compute_metrics(y_true, y_pred_prob):
    # 检查输入是否为二维数组，第一维是样本数量，第二维是类别数量
    if not isinstance(y_pred_prob, np.ndarray) or len(y_pred_prob.shape) != 2:
        raise ValueError("y_pred_prob should be a 2D numpy array.")

    # 确保 y_true 和 y_pred_prob 的样本数量一致
    if len(y_true) != y_pred_prob.shape[0]:
        raise ValueError("The number of samples in y_true and y_pred_prob does not match.")

    # 计算预测类别
    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        # 设置 zero_division 参数
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_prob)
    }

    try:
        
        num_classes = y_pred_prob.shape[1]
        y_true_one_hot = np.eye(num_classes)[y_true]
        metrics['roc_auc'] = roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovr')
        metrics['average_precision'] = average_precision_score(y_true_one_hot, y_pred_prob)
    except ValueError as e:
        print(f"Warning: Unable to compute ROC AUC or Average Precision due to {e}")

    return metrics


def plot_confusion_matrix(cm, classes, save_path):
    
    total_samples_per_class = cm.sum(axis=1, keepdims=True)
    
    total_samples_per_class[total_samples_per_class == 0] = 1
    
    cm_percentage = cm / total_samples_per_class

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Probability)')
    plt.savefig(os.path.join(os.getcwd(), save_path))
    plt.close()


def plot_losses(losses, save_path, class_names):
    if 'total' in losses:
        epochs = range(1, len(losses['total']) + 1)

        # 绘制总的损失函数图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses['total'], label='Total Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Rock Thin Section Classification Loss Function')
        plt.legend()
        total_loss_save_path = os.path.join(os.path.dirname(save_path), "total_loss_" + os.path.basename(save_path))
        plt.savefig(total_loss_save_path)
        plt.close()
    else:
        # 如果没有总损失，跳过绘制总损失图
        total_loss_save_path = None

    # 找到任意一个类别损失的长度作为 epochs
    first_key = next(iter(losses))
    epochs = range(1, len(losses[first_key]) + 1)

    # 反转 rock_names 映射（英文到中文）
    english_to_chinese = {v: k for k, v in rock_names.items()}
    
    # 绘制每个类别分别的损失函数图
    plt.figure(figsize=(10, 6))
    for label, loss in losses.items():
        if label != 'total':
            
            class_english_name = label.split('_')[0]
            if class_english_name in english_to_chinese:
                class_name = english_to_chinese[class_english_name]  # 转为中文名称
                plt.plot(epochs, loss, label=class_name, linestyle='--')
    
    if any(losses.values()):  # 检查是否有有效的损失数据
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Class-wise Rock Thin Section Classification Loss Function')
        plt.legend()
    
    class_wise_loss_save_path = os.path.join(os.path.dirname(save_path), "class_wise_loss_" + os.path.basename(save_path))
    plt.savefig(class_wise_loss_save_path)
    plt.close()


def save_metrics(metrics, epoch, save_path, classes):
    with open(save_path, 'a') as file:
        file.write(f"Epoch {epoch}:\n")

        # 写入混淆矩阵
        if 'confusion_matrix' in metrics:
            cm_str = '\n'.join(['\t'.join([str(cell) for cell in row]) for row in metrics['confusion_matrix']])
            file.write(f"confusion_matrix:\n{cm_str}\n")

        # 写入其他指标
        for key, value in metrics.items():
            if key != 'confusion_matrix' and key not in classes:
                file.write(f"{key}: {value:.4f}\n")  

        # 写入每个类别的损失
        for class_name in classes:
            loss_key = f'{class_name}_loss'
            if loss_key in metrics:
                file.write(f"{loss_key}: {metrics[loss_key]:.4f}\n")

        file.write("\n")
