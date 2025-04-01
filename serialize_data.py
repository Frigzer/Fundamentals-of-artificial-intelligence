import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

# === Transformacje ===
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# === Serializacja ===
def serialize_dataset(imagefolder_root, output_path):
    dataset = ImageFolder(imagefolder_root)
    os.makedirs(output_path, exist_ok=True)

    for idx, (img_path, label) in enumerate(dataset.samples):
        img = Image.open(img_path).convert("RGB")
        data = {
            "image": img,
            "label": label
        }
        file_name = f"{Path(img_path).stem}_{idx}.pkl"
        with open(os.path.join(output_path, file_name), "wb") as f:
            pickle.dump(data, f)

    print(f"Zserializowano: {len(dataset.samples)} plików → {output_path}")

# === Dataset ===
class SerializedASLDataset(Dataset):
    def __init__(self, serialized_path, transform=None):
        self.serialized_path = serialized_path
        self.data_files = sorted(os.listdir(serialized_path))
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(os.path.join(self.serialized_path, self.data_files[idx]), "rb") as f:
            data = pickle.load(f)
        img = data["image"]
        label = data["label"]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_serialized_dataloaders(root="serialized", batch_size=32):
    train_dataset = SerializedASLDataset(f"{root}/train", transform=train_transform)
    val_dataset = SerializedASLDataset(f"{root}/val", transform=test_transform)
    test_dataset = SerializedASLDataset(f"{root}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# === Podgląd batcha ===
def show_batch(data_loader, class_names=None, title=""):
    images, labels = next(iter(data_loader))
    grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)

    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"{title} – Przykładowe obrazy")
    plt.axis("off")
    print("Etykiety:", labels[:8].tolist())
    plt.show()


# === MAIN ===
if __name__ == "__main__":
    # Serializacja (uruchamiana tylko raz)
    serialize_dataset("data/train", "serialized/train")
    serialize_dataset("data/val", "serialized/val")
    serialize_dataset("data/test", "serialized/test")

    # Wczytanie danych
    train_loader, val_loader, test_loader = get_serialized_dataloaders()

    # Podgląd danych
    show_batch(train_loader, title="Train")
    show_batch(val_loader, title="Validation")
    show_batch(test_loader, title="Test")
