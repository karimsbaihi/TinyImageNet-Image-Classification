import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from data_utils import parse_class_mappings
from datasets import TinyImageNetTrain, TinyImageNetVal
from model import create_model
from transforms import val_transform, train_transform

# 🔁 For reproducibility
random.seed(42)

# ⚠️ If running on Colab, make sure to mount your drive and set this path correctly
base_path = './data/tiny-imagenet-200'

# 🔠 Load class mappings
class_to_idx, wnid_to_words = parse_class_mappings(
    wnids_path=base_path + '/wnids.txt',
    words_path=base_path + '/words.txt'
)

# 🧠 Load full datasets
train_ds = TinyImageNetTrain(base_path, class_to_idx, train_transform)
val_ds   = TinyImageNetVal(base_path, class_to_idx, val_transform)

# 🧳 Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)

# ⚡ Use TPU with PyTorch/XLA
import torch_xla
import torch_xla.core.xla_model as xm
device = xm.xla_device()

print(f"Using device: {device}")

# 🧠 Model
model = create_model(num_classes=200).to(device)  # Assuming TinyImageNet has 200 classes

# 🎯 Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🚀 Training loop
for epoch in range(5):
    model.train()
    total_batches = len(train_loader)
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # Needed for TPU
        xm.mark_step()  # Sync TPU

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"Epoch [{epoch+1}/5], Step [{batch_idx+1}/{total_batches}], Loss: {running_loss / (batch_idx+1):.4f}")

    # ✅ Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        xm.mark_step()  # Sync TPU after eval

    acc = 100 * correct / total
    print(f"✅ Epoch {epoch + 1} Completed: Val Accuracy = {acc:.2f}%\n")
