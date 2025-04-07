import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class TinyImageNetWithBoxes(Dataset):
    def __init__(self, root_dir, transform=None, annotations_file=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotations_file (string): Path to the annotations file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_file = annotations_file
        
        # Load the annotations (bounding box data)
        self.annotations = pd.read_csv(annotations_file, delimiter='\t', header=None)
        self.annotations.columns = ['image', 'class', 'x1', 'y1', 'x2', 'y2']  # Bounding box format
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(os.listdir(os.path.join(root_dir, 'train')))}
        self.image_paths = [os.path.join(root_dir, 'train', row['class'], row['image']) for _, row in self.annotations.iterrows()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")
        
        # Get the bounding box coordinates
        bbox = self.annotations.iloc[idx, 2:].values.astype('float')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, bbox

'''
here is the directory 
data/
└── tiny-imagenet-200/
    ├── wnids.txt
    └── words.txt
    └── train/
        ├── class_1/
              └── images
                 └── image1.jpg
                 └── image2.jpg
                 └── ...
        ├── class_1_boxes.txt
        ├── class_2/
        └── class_3/
        └── ...
    └── test/
        └── images
            └── image1.jpg
            └── image2.jpg
            └── ...
    └── val/
        └── images
            └── image1.jpg
            └── image2.jpg
            └── ...
        └── val_annotations.txt   
     
        '''