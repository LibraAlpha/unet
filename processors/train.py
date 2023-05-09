import numpy as np
from model.unet import Unet
from utils.carvana import CarvanaDataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler,random_split
from torchvision.transforms import transforms

def train(
        model,
        device,
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
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



    return

if __name__ == '__main__':

