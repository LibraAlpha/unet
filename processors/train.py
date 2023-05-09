import logging

import numpy as np
from model.unet import Unet
from utils.carvana import CarvanaDataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, random_split
from torchvision.transforms import transforms
from torch import optim
import torch.nn as nn


def train(
        model,
        device,
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
        weight_decay=1e-8
):
    root = Path('../data/')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CarvanaDataset(root, transform=transform)

    total_sample = len(dataset)
    test_ratio = 0.2  # 测试数据所占比例
    test_size = int(total_sample * test_ratio)
    train_size = total_sample - test_size

    train_data_set, test_data_set = random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    logging.info(f"""
        Start Training with:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {learning_rate}
        Device: {device}        
                 """)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = nn.CrossEntropyLoss()

    # loop epochs
    for epoch in range(epochs):
        model.train()

    return


if __name__ == '__main__':
    print('END')
