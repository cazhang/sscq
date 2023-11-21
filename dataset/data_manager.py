"""
Dataset Manager
"""

import torch
import torchvision
import torchvision.transforms as transforms

from dataset import nuswide, flickr25k


class TwoCropTransform:
    """Create two augmented views of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def load_data(data_dir, dataset, batch_size, random_seed=None, num_workers=8):
    """Data loader"""
    if dataset == "cifar10":
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3, (0.1, 2.0)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ])
         
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=TwoCropTransform(transforms_train))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

        databaseset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms_test)
        databaseloader = torch.utils.data.DataLoader(databaseset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms_test)
        queryloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return trainloader, databaseloader, queryloader, trainset, databaseset, testset

    else:
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3, (0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transforms_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if dataset == 'nuswide':
            num_train = 'ALL'
            trainloader, databaseloader, queryloader = nuswide.load_data(root=data_dir, num_query=2100, num_train=num_train, batch_size=batch_size,
                transforms_train=TwoCropTransform(transforms_train), transforms_test=transforms_test, num_workers=num_workers)
        elif dataset == 'flickr25k':
            num_train = 'ALL'
            trainloader, databaseloader, queryloader = flickr25k.load_data(root=data_dir, num_query=2000, num_train=num_train, batch_size=batch_size,
                transforms_train=TwoCropTransform(transforms_train), transforms_test=transforms_test, random_seed=random_seed, num_workers=num_workers)
        else:
            raise ValueError("Not known dataset. Please check dataloader for the implementation.")

        return trainloader, databaseloader, queryloader

            
