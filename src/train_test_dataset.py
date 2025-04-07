import torch
from torchvision import transforms
from src.utils import TinyImageNetTrain

def test_train_dataset():
    transform = transforms.ToTensor()
    dataset = TinyImageNetTrain(
        root_dir="/home/karim/Documents/3Y/ML/project/data/tiny-imagenet-200",
        class_to_idx={"n01443537": 0, "n01629819": 1},  # Sample 2 classes
        transform=transform
    )
    
    # Verify dataset properties
    assert len(dataset) > 0, "No training samples found!"
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert img.shape == (3, 64, 64), f"Expected (3,64,64), got {img.shape}"
    assert label in [0, 1], f"Invalid label {label}"
    
    print("✅ train_dataset.py: All tests passed!")

if __name__ == "__main__":
    test_train_dataset()