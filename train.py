# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu"):
    start_time = time.time()
    model.to(device)  # Przenieisenie modelu na cpu/gpu
    criterion = nn.CrossEntropyLoss()  # Funkcja kosztu
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optymalizator
    '''
    Adam - szybki i adaptacyjny optymalizator
    '''

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # ustawia model w tryb treningowy
        running_loss = 0.0  # Do śledzenia błędu
        correct = 0  # Do śledznia dokładności
        total = 0  # ---------||------------

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zerowanie gradientu, przed nowym batchem trzeba zawsze wyzerować

            outputs = model(inputs)  # Przepuszacznie danych przez sieć
            loss = criterion(outputs, labels)  # obliczenie straty
            loss.backward()  # Obliczanie gradientów
            optimizer.step()  # Aktualizacja wag

            running_loss += loss.item()  # loss to tensor a item wyciaga wartość, sumuje strate z kazdego batcha
            _, predicted = outputs.max(1)  # zwraca przewidywanie
            total += labels.size(0)  # labels.size(0) to batch size
            correct += predicted.eq(labels).sum().item()  # sprawdza czy dobrze trafiło, sum liczy ile trafień

        train_acc = 100. * correct / total
        train_losses.append(running_loss)

        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining complete, elapsed time: {total_time:.2f} seconds")

    return train_losses, val_losses
