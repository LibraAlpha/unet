import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class CarvanaDataset(VisionDataset):
    def __init__(self, root, image_folder='train', mask_folder='train_masks', transform=None, target_transform=None):
        super(CarvanaDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.image_dir = os.path.join(self.root, image_folder)
        self.mask_dir = os.path.join(self.root, mask_folder)
        self.image_filenames = os.listdir(self.image_dir)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename.replace(".jpg", "_mask.gif")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    @staticmethod
    def load_image(path):
        image = Image.open(path).convert('RGB')
        return image

    @staticmethod
    def load_mask(path):
        mask = Image.open(path).convert('L')
        return mask
