import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ustawienie urządzenia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Używane urządzenie: {device}')

# Parametry
data_dir = './data'
batch_size = 64
num_classes = 36
epochs = 5
learning_rate = 0.001
image_size = 64  # można dopasować

# Transformacje obrazu
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Załaduj dane
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Prosty model CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (image_size // 8) * (image_size // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

# Inicjalizacja
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trening
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Ewaluacja
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Dokładność na danych testowych: {accuracy:.2f}%")

# Pokaż kilka predykcji
classes = train_dataset.classes

# Pobierz losowe indeksy z całego test setu
indices = random.sample(range(len(test_dataset)), 6)
images = []
labels = []

for idx in indices:
    img, label = test_dataset[idx]
    images.append(img)
    labels.append(label)

# Połącz obrazy w batch
images = torch.stack(images).to(device)
labels = torch.tensor(labels).to(device)
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, preds = torch.max(outputs, 1)

# Wyświetl 6 obrazków
plt.figure(figsize=(12, 6))
for i in range(6):
    img = images[i].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # unnormalize
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"True: {classes[labels[i]]}, Pred: {classes[preds[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
