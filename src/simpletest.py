from TinyImageNetWithBoxes import TinyImageNetWithBoxes
from datasets import TinyImageNetTrain
from data_utils import parse_class_mappings
from torch.utils.data import DataLoader

# Define the paths to the mapping files
path = '/home/karim/Documents/3Y/ML/project/'
wnids_path=path+'data/tiny-imagenet-200/wnids.txt'
words_path=path+'data/tiny-imagenet-200/words.txt'

# Get class_to_idx mapping
class_to_idx, wnid_to_words = parse_class_mappings(wnids_path, words_path)

# Test class_to_idx mapping
print(f"Total classes: {len(class_to_idx)}")
print(f"Sample class_to_idx: {dict(list(class_to_idx.items())[:5])}")  # Print first 5 classes

# Create datasets
train_dataset = TinyImageNetTrain('data', class_to_idx, train_transform)
val_dataset = TinyImageNetVal('data', class_to_idx, val_transform)

# Test train dataset loading
print(f"Number of samples in train dataset: {len(train_dataset)}")
img, label = train_dataset[0]  # Get the first sample
print(f"First image shape: {img.size()} | Label: {label}")

# Test val dataset loading
print(f"Number of samples in validation dataset: {len(val_dataset)}")
img, label = val_dataset[0]  # Get the first sample
print(f"First image shape: {img.size()} | Label: {label}")

# Test DataLoader iteration
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
data_iter = iter(train_loader)
images, labels = next(data_iter)  # Get the first batch
print(f"Batch images shape: {images.shape} | Labels: {labels}")

# Test model forward pass with a batch of data
model = create_model().cuda()  # Ensure the model is on GPU if you're using CUDA
images, labels = next(data_iter)
images = images.cuda()  # Move images to GPU if CUDA is used
outputs = model(images)
print(f"Model output shape: {outputs.shape}")
