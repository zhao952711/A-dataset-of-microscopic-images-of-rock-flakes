import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, cohen_kappa_score, log_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 中文矿物名称及其对应的英文名称
mineral_names = {
    '鲕粒': 'Oolite',
    '橄榄石': 'Olivine',
    '黑云母': 'Biotite',
    '红柱石': 'Andalusite',
    '角闪石': 'Hornblende',
    '普通辉石': 'Augite',
    '十字石': 'staurolite',
    '石榴子石': 'Garnet',
    '斜长石': 'Plagioclase',
    '阳起石': 'Actinolite'
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
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_prob)
    }

    try:
        # 对于多分类问题，roc_auc_score 需要 one-hot 编码的目标值
        num_classes = y_pred_prob.shape[1]
        y_true_one_hot = np.eye(num_classes)[y_true]
        metrics['roc_auc'] = roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovr')
        metrics['average_precision'] = average_precision_score(y_true_one_hot, y_pred_prob)
    except ValueError as e:
        print(f"Warning: Unable to compute ROC AUC or Average Precision due to {e}")

    return metrics


def plot_confusion_matrix(cm, classes, save_path):
    # 计算每个真实类别的样本总数
    total_samples_per_class = cm.sum(axis=1, keepdims=True)
    # 避免除零错误
    total_samples_per_class[total_samples_per_class == 0] = 1
    # 计算每个类别的预测概率
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
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses['total'], label='Total Loss', color='red')
    else:
        # 当没有总损失时，找到任意一个类别损失的长度作为 epochs
        first_key = next(iter(losses))
        epochs = range(1, len(losses[first_key]) + 1)
        plt.figure(figsize=(10, 6))

    for i, (label, loss) in enumerate(losses.items()):
        if label == 'total':
            continue
        class_index = int(label.split('_')[1])
        if class_index < len(class_names):
            class_name = class_names[class_index]
            plt.plot(epochs, loss, label=class_name, linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mineral Thin Section Classification Loss Function')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), save_path))
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
            if key != 'confusion_matrix' and not key.startswith('class_'):
                file.write(f"{key}: {value:.4f}\n")  # 格式化浮点数为小数点后四位

        # 写入每个类别的损失
        for i in range(len(classes)):
            loss_key = f'class_{i}_loss'
            if loss_key in metrics:
                file.write(f"{loss_key}: {metrics[loss_key]:.4f}\n")

        file.write("\n")


if __name__ == '__main__':
    # 如果有主程序逻辑，可以在这里添加
    pass    
