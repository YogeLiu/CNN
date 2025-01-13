import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 第一个卷积层，卷积操作 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层，卷积操作 + BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不一致，使用1x1卷积来匹配通道数
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # 正常的残差块：先卷积，再batchnorm，再ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接：输入和输出相加
        out += self.shortcut(x)

        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()

        # 初始卷积层，输入的通道数为3（RGB图像），输出为64个通道，卷积核大小为3，步幅为1
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义ResNet的四个阶段，每个阶段包含多个残差块
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 最后的全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 最后一个阶段输出的通道数为512

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # 第一层的残差块可能会改变步幅和输出通道
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 后续的残差块保持步幅为1
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 四个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.fc(x)

        return x


def ResNet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    model = ResNet18()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
