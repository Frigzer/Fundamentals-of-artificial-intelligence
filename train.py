import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu"):
    start_time = time.time()
    model.to(device) # Przenieisenie modelu na cpu/gpu
    criterion = nn.CrossEntropyLoss() # Funkcja kosztu
    optimizer = optim.Adam(model.parameters(), lr=lr) # optymalizator
    '''
    Adam - szybki i adaptacyjny optymalizator
    '''

    for epoch in range(epochs):
        model.train() # ustawia model w tryb treningowy
        running_loss = 0.0 # Do śledzenia błędu
        correct = 0 # Do śledznia dokładności
        total = 0 # Do śledzenia dokładności

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Zerowanie gradientu, przed nowym batchem trzeba zawsze wyzeerować

            outputs = model(inputs) # Przepuszacznie danych przez sieć
            loss = criterion(outputs, labels) # obliczenie straty
            loss.backward() # Obliczanie gradientów
            optimizer.step() # Aktualizacja wag

            running_loss += loss.item() # loss to tensor a item wyciaga wartosc, sumuje strate z kazdego batcha
            _, predicted = outputs.max(1) # zwraca przewidywanie
            total += labels.size(0) # labels.size(0) to batch size
            correct += predicted.eq(labels).sum().item() # sprawdza czy dobrze trafilo, sum liczy ile trafien

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Accuracy: {train_acc:.2f}%")

        evaluate(model, val_loader, device)

    total_time = time.time() - start_time
    print(f"\nTraining complete, elapsed time: {total_time:.2f} seconds")

def evaluate(model, data_loader, device="cpu"):
    model.eval() # tryb testowy
    correct = 0
    total = 0
    with torch.no_grad(): # wyłącza śledzenie gradientów
        for inputs, labels in data_loader: # to samo co w treningu
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
