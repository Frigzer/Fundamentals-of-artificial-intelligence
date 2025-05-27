# run_model.py

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from model import ASLClassifier
from evaluate import get_class_names

# === Konfiguracja ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pth"  # zapisany model
image_path = "your_image.jpg"  # przykładowy obrazek do testu

# === Przygotowanie modelu ===
num_classes = 36
model = ASLClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Transformacja (musi być taka sama jak przy treningu!) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Wczytanie obrazu ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # dodaj batch dimension

# === Predykcja ===
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

class_names = get_class_names()
predicted_label = class_names[predicted.item()]

# === Wyświetlenie ===
plt.imshow(img)
plt.title(f"Predykcja: {predicted_label}")
plt.axis("off")
plt.show()
