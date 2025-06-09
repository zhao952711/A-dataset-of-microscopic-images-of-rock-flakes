import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class CA_ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # 轻量坐标注意力（仅1x1卷积）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(planes, planes // 8, 1),
            nn.ReLU(),
            nn.Conv2d(planes // 8, planes, 1),
            nn.Sigmoid()
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.ca(out)  # 注意力加权

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class MineralResNet34(nn.Module):
    def __init__(self, num_classes=9, weights=models.ResNet34_Weights.IMAGENET1K_V1, freeze_conv=True):
        super(MineralResNet34, self).__init__()

        # 加载预训练的ResNet34模型，并使用weights参数
        self.resnet = models.resnet34(weights=weights)

        # 保存原层全连接层的输入特征数
        self.num_ftrs = self.resnet.fc.in_features

        # 冻结卷积层
        if freeze_conv:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 替换第2、3个残差组的残差块为CA_ResBlock
        downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.resnet.layer2 = nn.Sequential(
            CA_ResBlock(64, 128, stride=2, downsample=downsample2),
            CA_ResBlock(128, 128),
            CA_ResBlock(128, 128),
            CA_ResBlock(128, 128)
        )

        downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.resnet.layer3 = nn.Sequential(
            CA_ResBlock(128, 256, stride=2, downsample=downsample3),
            CA_ResBlock(256, 256),
            CA_ResBlock(256, 256),
            CA_ResBlock(256, 256),
            CA_ResBlock(256, 256),
            CA_ResBlock(256, 256)
        )

        # 移除最后一层全连接层
        self.resnet.fc = nn.Identity()  # 返回特征向量

        # 添加新的分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x
    

