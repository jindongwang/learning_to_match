# coding=utf-8
from torchvision import datasets, transforms
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io


class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class myDataset(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        class_dirs = [os.path.join(root, i) for i in os.listdir(root)]
        imgs = []
        for i in class_dirs:
            imgs += [os.path.join(i, img) for img in os.listdir(i)]
        np.random.shuffle(imgs)
        imgs_mun = len(imgs)
        # target:val = 8 ：2
        if self.train:
            self.imgs = imgs[:int(0.3*imgs_mun)]
        else:
            self.imgs = imgs[int(0.3*imgs_mun):]
        if transform:
            self.transforms = transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            start_center = (256 - 224 - 1) / 2
            self.transforms = transforms.Compose(
                [transforms.Resize([224, 224]),
                 PlaceCrop(224, start_center, start_center),
                 transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = int(img_path.strip().split('/')[10])
        print(img_path, label)
        #data = Image.open(img_path)
        data = io.imread(img_path)
        data = Image.fromarray(data)
        if data.getbands()[0] == 'L':
            data = data.convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


def load_training(root_path, domain, batch_size, kwargs, train_val_split=.5):
    """Load training dataloader

    Args:
        root_path (str): root path for dataset
        domain (str): domain name
        batch_size (int): batch size
        kwargs (dict): other params
        train_val_split (float, optional): train-valid split. '-1' means there's no validation set. Defaults to .5.

    Returns:
        dataloader
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [ResizeImage(256),
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize])
    data = datasets.ImageFolder(root=os.path.join(
        root_path, domain), transform=transform)
    if train_val_split <= 0:
        train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return train_loader
    else:
        train_loader, val_loader = load_train_valid_split(
            data, batch_size, kwargs, val_ratio=train_val_split)
        return train_loader, val_loader


def load_testing(root_path, domain, batch_size, kwargs):
    """Load test dataloader

    Args:
        root_path (str): root path for dataset
        domain (str): domain name
        batch_size (int): batch size
        kwargs (dict): other params

    Returns:
        dataloader
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [ResizeImage(256),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor(),
         normalize])
    dataset = datasets.ImageFolder(root=os.path.join(
        root_path, domain), transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader


def load_train_valid_split(dataset, batch_size, kwargs, val_ratio=0.4):
    validation_split = val_ratio
    shuffle_dataset = True
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, **kwargs, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, **kwargs, drop_last=True)
    return train_loader, validation_loader

# Dataloader class for meta data
class MetaDataset(data.Dataset):
    def __init__(self, meta_info) -> None:
        self.meta_info = meta_info
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [ResizeImage(256),
             transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])

    def __getitem__(self, index: int):
        data = io.imread(self.meta_info[index][0])
        data = Image.fromarray(data)
        if data.getbands()[0] == 'L' or len(data.getbands()) > 3:
            data = data.convert('RGB')
        data = self.transform(data)
        label = self.meta_info[index][1]
        return data, label

    def __len__(self):
        return len(self.meta_info)


def load_metadata(meta_dataset, batch_size=8):
    meta_loader = torch.utils.data.DataLoader(
        meta_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return meta_loader
