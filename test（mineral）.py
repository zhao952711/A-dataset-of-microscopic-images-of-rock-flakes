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
print(f"Using device: {device}")

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


def prepare_test_data():
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    return test_loader, test_dataset.classes


def load_model(num_classes, model_path):
    # 创建 TCA-ResNet34 模型实例，使用预训练权重
    model = MineralResNet34(num_classes=num_classes, weights=models.ResNet34_Weights.IMAGENET1K_V1, freeze_conv=False)
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
    classes = [mineral_names[name] for name in classes]

    metrics = compute_metrics(np.array(all_labels), all_probs)

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:\n{value}")
            cm_image_path = os.path.join('zhibiao/hunxiao', f'confusion_matrix_test.png')
            os.makedirs(os.path.dirname(cm_image_path), exist_ok=True)
            plot_confusion_matrix(value, classes, cm_image_path)
        else:
            print(f"{key}: {value}")

    # 添加测试时间到评估指标中
    metrics['test_time'] = test_time
    print(f"Test Time: {test_time} seconds")

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
    
