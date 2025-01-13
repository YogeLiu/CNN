import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.alexnet import AlexNet
from models.resnet import ResNet18
from utils.data_loader import CatDogDataset
from utils.transforms import get_train_transform, get_test_transform
import config
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train():
    # 创建模型
    # model = AlexNet(num_classes=config.NUM_CLASSES)
    model = ResNet18(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 加载数据
    train_dataset = CatDogDataset(root_dir=config.TRAIN_DIR, transform=get_train_transform())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                # logging.info(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], " f"Step [{i+1}/{len(train_loader)}], " f"Loss: {running_loss/100:.4f}")
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], " f"Step [{i+1}/{len(train_loader)}], " f"Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # 保存模型
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(config.MODEL_SAVE_PATH):
                os.makedirs(config.MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    train()
