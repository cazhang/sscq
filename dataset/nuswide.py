"""
NUS-WIDE dataset
"""

import os
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, transforms_train, transforms_test, num_workers=16):
    """
    Loading nus-wide dataset.

    Returns
        train_dataloader, database_dataloader, query_dataloader: Data loader.
    """

    # By default, use ALL retrieval database images as the training set
    if num_train != 10500:
        train_dataset = NusWideDatasetTraALL(
            root,
            'database.txt',
            transform=transforms_train,
        )
    # Use 10,500 images as the training set
    else:
        train_dataset = NusWideDatasetTraALL(
            root,
            'train.txt',
            transform=transforms_train,
        )

    database_dataset = NusWideDatasetTraALL(
        root,
        'database.txt',
        transform=transforms_test,
    )

    query_dataset = NusWideDatasetTraALL(
        root,
        'test.txt',
        transform=transforms_test,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        #pin_memory=True,
        num_workers=num_workers,
    )
    database_dataloader = DataLoader(
        database_dataset,
        batch_size=batch_size,
        shuffle=False,
        #pin_memory=True,
        num_workers=num_workers,
    )
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        #pin_memory=True,
        num_workers=num_workers,
    )

    return train_dataloader, database_dataloader, query_dataloader


class NusWideDatasetTraALL(Dataset):
    def __init__(self, root, img_txt, transform=None):
        self.root = root
        self.transform = transform
        img_txt_path = os.path.join(root, img_txt)

        # Read files
        img, label = [], []
        with open(img_txt_path, 'r') as f:
            for i in f:
                i = i.strip()
                img.append(i.split()[0])
                label.append([int(j) for j in i.split()[1:]])
        
        self.data = np.array(img)
        self.targets = np.array(label, dtype=np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]#, index

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.FloatTensor(self.targets)
