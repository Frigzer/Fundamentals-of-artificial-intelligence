import torch
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision.datasets import ImageFolder


# === Nazwy klas na podstawie folderów (działa jak ImageFolder) ===
def get_class_names(path="data/train"):
    dataset = ImageFolder(path)
    return dataset.classes


# === Wykres funkcji kosztu ===
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Funkcja kosztu (Loss)')
    plt.show()


# === Wykres słupkowy metryk per klasa ===
def plot_classification_metrics(report_dict, class_names):
    metrics = ['precision', 'recall', 'f1-score']
    values = {metric: [] for metric in metrics}

    for cls in class_names:
        for metric in metrics:
            values[metric].append(report_dict[cls][metric])

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, values['precision'], width, label='Precision')
    plt.bar(x, values['recall'], width, label='Recall')
    plt.bar(x + width, values['f1-score'], width, label='F1-Score')

    plt.xlabel('Klasa')
    plt.ylabel('Wartość')
    plt.title('Precision / Recall / F1-score per klasa')
    plt.xticks(ticks=x, labels=class_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Zapis metryk do CSV ===
def save_metrics_to_csv(report_dict, class_names, path="results/classification_metrics.csv"):
    metrics = ['precision', 'recall', 'f1-score']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class'] + metrics)
        for cls in class_names:
            row = [cls] + [report_dict[cls][m] for m in metrics]
            writer.writerow(row)
    print(f"Zapisano metryki do pliku: {path}")


# === Ewaluacja modelu na danych testowych ===
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"\nDokładność (Accuracy): {accuracy * 100:.2f}%\n")

    class_names = get_class_names()
    report = classification_report(all_labels, all_predicted, target_names=class_names, output_dict=True)
    print("Raport klasyfikacji:")
    print(classification_report(all_labels, all_predicted, target_names=class_names))

    # Wykres metryk i zapis CSV
    plot_classification_metrics(report, class_names)
    save_metrics_to_csv(report, class_names)

    # Macierz pomyłek
    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywista')
    plt.title('Macierz Pomyłek')
    plt.tight_layout()
    plt.show()


# === Pokazuje przykładowe obrazki z predykcjami ===
def visualize_predictions(model, test_loader, device="cpu"):
    model.eval()
    class_names = get_class_names()

    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    num_samples = min(len(images), 10)
    fig, axs = plt.subplots(2, 5, figsize=(16, 7))

    for i in range(num_samples):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        image = (image * 0.5) + 0.5  # od-normowanie
        true_label = labels[i].cpu().item()
        predicted_label = predictions[i].cpu().item()

        axs[i // 5, i % 5].imshow(image)
        axs[i // 5, i % 5].set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_label]}')
        axs[i // 5, i % 5].axis('off')

    plt.tight_layout()
    plt.show()
