import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models  
from tca_resnet34 import MineralResNet34  
from zhibiao import compute_metrics, plot_confusion_matrix, plot_losses, save_metrics 
from sklearn.preprocessing import label_binarize
import time

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
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


def prepare_data():
    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 将类别名称转换为英文
    classes = [rock_names[name] for name in image_datasets['train'].classes]

    return dataloaders, dataset_sizes, classes


def create_model(num_classes):
    # 创建模型实例，使用ResNet34预训练权重
    model = MineralResNet34(
        num_classes=num_classes,
        weights=models.ResNet34_Weights.IMAGENET1K_V1,  # 加载预训练权重
        freeze_conv=True
    )
    model = model.to(device)
    return model


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, classes, num_epochs=50,
                patience=5):  
    losses = {'total': []}
    for i in range(len(classes)):
        losses[classes[i]] = []  

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        all_probs = []

        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'val':
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.append(torch.softmax(outputs, dim=1).cpu())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                losses['total'].append(epoch_loss)

                # 将所有标签二值化以适应每个类别的损失计算
                y_true_one_hot = label_binarize(all_labels, classes=list(range(len(classes))))

                # 确保 all_probs 是一个 2D NumPy 数组
                all_probs_array = torch.cat(all_probs, dim=0).numpy()

                # 计算每个类别的损失
                for i in range(len(classes)):
                    class_labels = torch.tensor(y_true_one_hot[:, i], dtype=torch.float32).to(device)
                    class_loss = nn.BCELoss()(torch.from_numpy(all_probs_array[:, i]).float().to(device),
                                              class_labels).item()
                    losses[classes[i]].append(class_loss)

                metrics = compute_metrics(all_labels, all_probs_array)

                # 添加每个类别的损失到metrics字典中
                for i in range(len(classes)):
                    metrics[f'{classes[i]}_loss'] = losses[classes[i]][-1]

                save_metrics(metrics, epoch, 'tca_resnet34_parameters.txt', classes)

                # 计算混淆矩阵概率
                cm = metrics['confusion_matrix']
                cm_prob = cm / cm.sum(axis=1, keepdims=True)

                # 绘制混淆矩阵并保存到指定目录
                cm_image_path = os.path.join('zhibiao/hunxiao', f'confusion_matrix_epoch_{epoch}.png')
                os.makedirs(os.path.dirname(cm_image_path), exist_ok=True)
                plot_confusion_matrix(cm_prob, classes, cm_image_path)

                # 判断是否是最后一个epoch或者触发了早停
                is_last_epoch = epoch == num_epochs - 1
                should_plot = is_last_epoch or (epochs_no_improve >= patience)

                if should_plot:
                    # 绘制损失图像并保存到指定目录
                    loss_image_path = os.path.join('zhibiao/sunshi', f'loss_epoch_{epoch}.png')
                    os.makedirs(os.path.dirname(loss_image_path), exist_ok=True)
                    plot_losses(losses, loss_image_path, classes)

                # 更新最佳验证损失并检查早停条件
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict().copy()
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'tca_resnet34_best_model.pth')  
                    print("Validation loss improved.")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                        break

                # 使用 get_last_lr() 获取最新学习率
                last_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {last_lr}")
                scheduler.step(epoch_loss)

        if epochs_no_improve >= patience:
            break

        print()

    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算总训练时长

    # 将总训练时长写入结果文件
    with open('tca_resnet34_parameters.txt', 'a') as file:
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")

    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':
    # 准备数据
    dataloaders, dataset_sizes, classes = prepare_data()

    # 创建模型、损失函数和优化器
    model = create_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 开始预训练
    model_pretrained = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, classes,
                                   num_epochs=50, patience=5)

    print("Pretraining completed and classifier parameters saved.")
