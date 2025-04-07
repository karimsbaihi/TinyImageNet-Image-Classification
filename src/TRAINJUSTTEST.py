import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random
from data_utils   import parse_class_mappings
from datasets     import TinyImageNetTrain, TinyImageNetVal
from model        import create_model
from transforms   import val_transform, train_transform
from collections import defaultdict

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
val_ds   = TinyImageNetVal  (base_path, class_to_idx, val_transform)

# ✅ Inspect dataset structure (First 5 samples)
print("First 5 samples in the training dataset:")
for i in range(5):
    print(f"Sample {i}: {train_ds[i]}")

# Select the first 10 classes (Ensure these are valid class IDs)
selected_classes = list(class_to_idx.values())[:10]  # Pick first 10 class IDs
print(f"Selected classes: {selected_classes}")

def get_subset_by_class(dataset, samples_per_class, allowed_classes):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        print(f"Dataset sample {idx} has label: {label}")  # Debugging labels
        if label in allowed_classes and len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
        if all(len(v) == samples_per_class for v in class_indices.values()):
            break
    indices = [i for indices in class_indices.values() for i in indices]
    print(f"Number of selected indices: {len(indices)}")  # Debugging
    return Subset(dataset, indices)

# Get subsets with the correct class IDs
train_subset = get_subset_by_class(train_ds, samples_per_class=10, allowed_classes=selected_classes)
val_subset   = get_subset_by_class(val_ds, samples_per_class=5, allowed_classes=selected_classes)

# Check if the subsets are empty
if len(train_subset) == 0 or len(val_subset) == 0:
    print("Error: The subsets are empty. Please check the data selection.")
    exit()

# Dataloaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_subset, batch_size=32)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = create_model(num_classes=10).to(device)  # 🔥 set num_classes=10 if your model allows it!

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(5):
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
