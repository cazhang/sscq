"""
FLICKR25K dataset
"""

import os
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, transforms_train, transforms_test, random_seed=None, num_workers=16):
    """
    Loading flickr25k dataset.

    Returns
        train_dataloader, database_dataloader, query_dataloader: Data loader.
    """

    # Initial Flickr25k dataset
    Flickr25k.init(root, num_query, num_train, random_seed)
    query_dataset = Flickr25k(root, 'query', transforms_test)
    train_dataset = Flickr25k(root, 'train', transforms_train)
    retrieval_dataset = Flickr25k(root, 'retrieval', transforms_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        #pin_memory=True,
        num_workers=num_workers,
    )
    database_dataloader = DataLoader(
        retrieval_dataset,
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


class Flickr25k(Dataset):
    """
    Flicker25k dataset.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = Flickr25k.TRAIN_DATA
            self.targets = Flickr25k.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Flickr25k.QUERY_DATA
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Flickr25k.RETRIEVAL_DATA
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]#, index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train, random_seed=None):
        """
        Initialize dataset

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        # Load dataset
        img_txt_path = os.path.join(root, 'img.txt')
        targets_txt_path = os.path.join(root, 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        if random_seed:
            np.random.seed(random_seed)
        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        if num_train == 'ALL':
            train_index = perm_index[num_query:]
        else:
            train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        Flickr25k.QUERY_DATA = data[query_index]
        Flickr25k.QUERY_TARGETS = targets[query_index, :]

        Flickr25k.TRAIN_DATA = data[train_index]
        Flickr25k.TRAIN_TARGETS = targets[train_index, :]

        Flickr25k.RETRIEVAL_DATA = data[retrieval_index]
        Flickr25k.RETRIEVAL_TARGETS = targets[retrieval_index, :]
