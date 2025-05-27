# run_model.py

import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from model import ASLClassifier
from evaluate import get_class_names

# === Konfiguracja ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pth"          # zapisany model
image_folder = "photos"     # folder z nowymi zdjęciami

# === Przygotowanie modelu ===
num_classes = 36
model = ASLClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Transformacja ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Pobranie klas ===
class_names = get_class_names()

# === Wczytanie i analiza wielu zdjęć ===
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]

if not image_files:
    print(f"Brak plików .jpg/.png w folderze: {image_folder}")
else:
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_label = class_names[predicted.item()]

        # === Wyświetlenie ===
        plt.imshow(img)
        plt.title(f"{filename}\nPredykcja: {predicted_label}")
        plt.axis("off")
        plt.show()
