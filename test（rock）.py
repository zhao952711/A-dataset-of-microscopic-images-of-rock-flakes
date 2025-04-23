import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models 
from tca_resnet34 import MineralResNet34  
from zhibiao import compute_metrics, plot_confusion_matrix, save_metrics  
import time

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 打印使用的设备信息以确认配置

# 数据预处理（与 train.py 中 'val' 阶段一致）
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 定义数据集路径
data_dir = r'data'

# 中文岩石名称及其对应的英文名称
rock_names = {
    '花岗岩': 'Granite',
    '片麻岩': 'Gneiss',
    '片岩': 'Schist',
    '白云岩': 'Dolomite',
    '石灰岩': 'Limestone',
    '凝灰岩': 'Tuff',
    '砂岩': 'Sandstone',
    '辉长岩': 'Gabbro',
    '玄武岩': 'Basalt'
}


def prepare_test_data():
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # 使用与训练时相同的 batch_size
    return test_loader, test_dataset.classes


def load_model(num_classes, model_path):
    # 创建模型实例，并加载预训练的 ImageNet 权重
    model = MineralResNet34(num_classes=num_classes, weights=models.ResNet34_Weights.IMAGENET1K_V1, freeze_conv=True)
    model = model.to(device)

    # 加载整个模型的状态字典
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def evaluate_model(model, dataloader, classes):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time

    all_probs = np.vstack(all_probs)

    # 将类别名称转换为英文
    classes = [rock_names[name] for name in classes]

    metrics = compute_metrics(np.array(all_labels), all_probs)
    metrics['test_time'] = test_time

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:\n{value}")
            cm_image_path = os.path.join('zhibiao/hunxiao', f'confusion_matrix_test.png')
            os.makedirs(os.path.dirname(cm_image_path), exist_ok=True)
            plot_confusion_matrix(value, classes, cm_image_path)  # 绘制并保存混淆矩阵
        elif key == 'test_time':
            print(f"{key}: {value:.4f} seconds")
        else:
            print(f"{key}: {value}")

    # 保存评估结果
    save_metrics(metrics, 'Test', 'test_parameters.txt', classes)

    print("Model evaluation completed.")


if __name__ == '__main__':
    # 准备测试数据
    test_loader, classes = prepare_test_data()

    # 加载模型
    num_classes = len(classes)
    model_path = os.path.join('best_model.pth') 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight file not found at {model_path}. Please check the path.")

    model = load_model(num_classes, model_path)

    # 评估模型
    evaluate_model(model, test_loader, classes)

    print("Model evaluation completed.")
