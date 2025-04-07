import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from data_utils import parse_class_mappings
from datasets import TinyImageNetTrain, TinyImageNetVal
from model import create_model
from transforms import val_transform, train_transform

# For reproducibility
random.seed(42)

# Paths
base_path = '/home/karim/Documents/3Y/ML/project/data/tiny-imagenet-200'

# Load class mappings
class_to_idx, wnid_to_words = parse_class_mappings(
    wnids_path=base_path + '/wnids.txt',
    words_path=base_path + '/words.txt'
)

# Datasets
train_ds = TinyImageNetTrain(base_path, class_to_idx, train_transform)
val_ds = TinyImageNetVal(base_path, class_to_idx, val_transform)

# ✅ Inspect dataset structure (First 5 samples)
print("First 5 samples in the training dataset:")
for i in range(5):
    print(f"Sample {i}: {train_ds[i]}")

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = create_model(num_classes=200).to(device)  # 🔥 Use num_classes=200 for TinyImageNet

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(3):
    model.train()
    total_batches = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"\rEpoch {epoch+1} - Progress: {100*(batch_idx+1)/total_batches:.2f}%", end='')

    print()
    
    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Epoch {epoch+1} Completed: Val Acc = {100 * correct / total:.2f}%\n")
