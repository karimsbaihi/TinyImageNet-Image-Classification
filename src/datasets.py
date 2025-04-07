from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

root = '/home/karim/Documents/3Y/ML/project/'
wnids_path=root+'data/tiny-imagenet-200/wnids.txt'
words_path=root+'data/tiny-imagenet-200/words.txt'
class TinyImageNetTrain(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root = os.path.join(root_dir, 'train')
        self.transform = transform
        self.class_to_idx = class_to_idx

        # 1. Gather (image_path, label) pairs
        self.samples = []
        for wnid in os.listdir(self.root):
            img_dir = os.path.join(self.root, wnid, 'images')
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                label   = self.class_to_idx[wnid]
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 2. Load image and apply transforms
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label



class TinyImageNetVal(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root = os.path.join(root_dir, 'val')
        self.transform = transform
        self.class_to_idx = class_to_idx

        # 1. Read val_annotations.txt
        df = pd.read_csv(
            os.path.join(self.root, 'val_annotations.txt'),
            sep='\t', header=None,
            names=['filename','wnid','x0','y0','x1','y1']
        )
        # 2. Build samples list
        self.samples = [
            (os.path.join(self.root,'images',row.filename),
             class_to_idx[row.wnid])
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
