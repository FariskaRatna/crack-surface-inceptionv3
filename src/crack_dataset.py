import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import pandas as pd

class CrackDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = {'Positive': 1, 'Negative': 0}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        label_int = self.label_map[label_str]

        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return image, label_tensor