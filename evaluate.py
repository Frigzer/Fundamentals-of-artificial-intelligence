# evaluate.py

import torch
import csv
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision.datasets import ImageFolder


# === Nazwy klas na podstawie folderów (działa jak ImageFolder) ===
def get_class_names(path="data/train"):
    dataset = ImageFolder(path)  # Tworzy zbior danych obrazów z folderów, gdzie każdy folder to jedna klasa
    return dataset.classes  # Zwraca listę nazw klas (czyli nazw folderów)


# === Wykres funkcji kosztu ===
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))  # Tworzy wykres o zadanym rozmiarze
    plt.plot(train_losses, label='Training Loss')  # Rysuje stratę z treningu
    plt.plot(val_losses, label='Validation Loss')  # Rysuje stratę z walidacji
    plt.xlabel('Epoch')  # Oś X to epoki
    plt.ylabel('Loss')  # Oś Y to wartość funkcji kosztu
    plt.legend()  # Dodaje legendę
    plt.title('Funkcja kosztu (Loss)')  # Tytuł wykresu
    plt.show()  # Wyświetlenie wykresu


# === Wykres słupkowy metryk per klasa ===
def plot_classification_metrics(report_dict, class_names):
    metrics = ['precision', 'recall', 'f1-score']  # Interesujące metryki
    values = {metric: [] for metric in metrics}  # Tworzymy puste listy dla każdej metryki

    for cls in class_names:  # Dla każdej klasy
        for metric in metrics:
            values[metric].append(report_dict[cls][metric])  # Pobieramy wartości z raportu

    x = np.arange(len(class_names))  # Pozycje na osi X
    width = 0.25  # Szerokość słupków

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, values['precision'], width, label='Precision')
    plt.bar(x, values['recall'], width, label='Recall')
    plt.bar(x + width, values['f1-score'], width, label='F1-Score')

    plt.xlabel('Klasa')
    plt.ylabel('Wartość')
    plt.title('Precision / Recall / F1-score per klasa')
    plt.xticks(ticks=x, labels=class_names, rotation=90)  # Etykiety klas na osi X
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Zapis metryk do CSV ===
def save_metrics_to_csv(report_dict, class_names, path="results/classification_metrics.csv"):
    metrics = ['precision', 'recall', 'f1-score']
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Tworzy folder jeśli nie istnieje
    with open(path, mode='w', newline='') as file:  # Otwiera plik CSV do zapisu
        writer = csv.writer(file)
        writer.writerow(['class'] + metrics)  # Nagłówki kolumn
        for cls in class_names:
            row = [cls] + [report_dict[cls][m] for m in metrics]  # Dane dla jednej klasy
            writer.writerow(row)
    print(f"Zapisano metryki do pliku: {path}")


# === Ewaluacja modelu na danych testowych ===
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()  # Przełączenie modelu w tryb ewaluacji
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():  # Bez śledzenia gradientów
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Przeniesienie na urządzenie
            outputs = model(inputs)  # Przewidywania
            _, predicted = torch.max(outputs, 1)  # Klasa z najwyższym prawdopodobieństwem
            total += labels.size(0)  # Zliczanie całkowitej liczby przykładów
            correct += (predicted == labels).sum().item()  # Zliczanie poprawnych
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"\nDokładność (Accuracy): {accuracy * 100:.2f}%\n")

    class_names = get_class_names()
    report = classification_report(all_labels, all_predicted, target_names=class_names, output_dict=True)
    print("Raport klasyfikacji:")
    print(classification_report(all_labels, all_predicted, target_names=class_names))

    plot_classification_metrics(report, class_names)  # Wykresy precision/recall/f1
    save_metrics_to_csv(report, class_names)  # Zapis CSV

    cm = confusion_matrix(all_labels, all_predicted)  # Macierz pomyłek
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywista')
    plt.title('Macierz Pomyłek')
    plt.tight_layout()
    plt.show()


# === Pokazuje przykładowe obrazki z predykcjami ===
def visualize_predictions(model, test_loader, device="cpu", show_only_errors=False, max_samples=10):
    model.eval()
    class_names = get_class_names()

    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    all_samples = list(zip(images, labels, predictions))

    if show_only_errors:
        all_samples = [sample for sample in all_samples if sample[1] != sample[2]]

    random.shuffle(all_samples)
    all_samples = all_samples[:max_samples]
    num_samples = len(all_samples)

    cols = 5
    rows = (num_samples + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

    if rows == 1:
        axs = [axs]

    for i in range(rows * cols):
        ax = axs[i // cols][i % cols] if rows > 1 else axs[i % cols]
        if i < num_samples:
            image, true_label, predicted_label = all_samples[i]
            image = image.numpy().transpose(1, 2, 0)
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            image = std * image + mean

            is_correct = true_label == predicted_label
            color = 'green' if is_correct else 'red'

            ax.imshow(image)
            ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_label]}', color=color,
                         fontsize=10)
        else:
            ax.axis('off')

        ax.axis('off')

    plt.tight_layout(h_pad=3.0)
    plt.show()


# === Pokazuje tylko błędne predykcje z całego zbioru testowego ===
def visualize_mistakes(model, test_loader, device="cpu", max_samples=10):
    import matplotlib.pyplot as plt
    import random
    model.eval()
    class_names = get_class_names()

    mistakes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for img, true, pred in zip(images, labels, predictions):
                if true != pred:
                    mistakes.append((img.cpu(), true.cpu(), pred.cpu()))

    if not mistakes:
        print("Brak błędnych predykcji w zbiorze testowym.")
        return

    random.shuffle(mistakes)
    mistakes = mistakes[:max_samples]
    num_samples = len(mistakes)

    cols = 5
    rows = (num_samples + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

    if rows == 1:
        axs = [axs]

    for i in range(rows * cols):
        ax = axs[i // cols][i % cols] if rows > 1 else axs[i % cols]
        if i < num_samples:
            img, true, pred = mistakes[i]
            image = img.numpy().transpose(1, 2, 0)
            image = (image * 0.5) + 0.5

            ax.imshow(image)
            ax.set_title(f"True: {class_names[true]}\nPred: {class_names[pred]}", color="red", fontsize=10)
        else:
            ax.axis("off")
        ax.axis("off")

    plt.tight_layout(h_pad=3.0)
    plt.show()
