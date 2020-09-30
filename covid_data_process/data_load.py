import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image, ImageOps
from skimage import io

TARGET_ONLY = {
    'train': 'train_list_covid19_labeled.txt',
    'test': 'test_list_covid19.txt',
    'valid': 'val_list_covid19.txt'
}

SOURCE_ONLY = {
    'train': 'train_list_pneumonia.txt',
    'valid': 'val_list_pneumonia.txt',
    'test': 'test_list_covid19.txt'
}


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

class COVID19(torch.utils.data.Dataset):
    def __init__(self, folder, filename, train='train'):
        self.folder = folder
        self.filename = filename
        with open(os.path.join(self.folder, 'labels', self.filename), 'r') as fp:
            self.all_imgs = fp.readlines()
        self.train = train
        if train == 'train':
            self.transforms = transforms.Compose(
                    [transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
        else:
            start_center = (256 - 224 - 1) / 2
            self.transforms = transforms.Compose(
                    [transforms.Resize((224, 224)),
                    transforms.ToTensor()])

    def __getitem__(self, idx):
        line = self.all_imgs[idx].strip().split(',')
        img_name, label = line[0], line[1]
        label = int(label)
        lab = 'all_data_pneumonia' if self.filename.__contains__('pneumonia') else 'all_data_covid'
        img_path = os.path.join(self.folder, lab, self.train, img_name)
        data = io.imread(img_path)
        data = Image.fromarray(data)
        data = data.convert('RGB')
        data = self.transforms(data)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.all_imgs)

def load_covid_data(root_path, batch_size, kwargs):
    train_data = COVID19(root_path, SOURCE_ONLY['train'], train='train')
    valid_data = COVID19(root_path, SOURCE_ONLY['valid'], train='val')
    test_data = COVID19(root_path, SOURCE_ONLY['test'], train='test')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders
