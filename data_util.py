"""
PyTorch Dataset
"""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import pandas as pd
import os


class CarDataset(Dataset):
    """
    Car dataset
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            transform (callable, optional): Optional image transform
            flag (string): train or eval
        """
        self.data = pd.read_csv(csv_file)
        self.image = self.data['image']
        self.label = self.data['label']
        self.transform = transform
        self.root = '../../car_data/'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.image.iloc[index]
        label = self.label.iloc[index]
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, int(label)
