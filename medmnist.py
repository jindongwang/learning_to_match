import numpy as np
import torch
import torchvision
import os
import data_loader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io


if __name__ == "__main__":
    root_path = '/home/jindwang/mine/data/chest'
    
    data = np.load(os.path.join(root_path, 'c.npz'))
    for mod in ['train', 'val', 'test']:
        folder = f'{root_path}/c/{mod}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        imgs = data[f"{mod}_images"]
        labels = data[f"{mod}_labels"]
        j = 1
        for i in range(len(imgs)):
            img = Image.fromarray(data['train_images'][i]).convert('RGB')
            label = int(labels[i].sum())
            if label == 0:
                if not os.path.exists(os.path.join(folder, str(label))):
                    os.mkdir(os.path.join(folder, str(label)))
                img.save(os.path.join(folder, str(label), f"{mod}_{label}_{j}.png"), format='png')
            # print(f"{mod}_{label}_{j}.png")
                j += 1
