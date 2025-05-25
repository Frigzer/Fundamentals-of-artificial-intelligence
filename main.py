# main.py

import torch
from model import ASLClassifier
from train import train_model
from serialize_data import get_serialized_dataloaders
from evaluate import plot_training_curves, evaluate_model, visualize_predictions, visualize_mistakes

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}\n")

    # Wczytanie danych
    train_loader, val_loader, test_loader = get_serialized_dataloaders(batch_size=32)
    model = ASLClassifier(num_classes=36)

    # Trening
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=0.001,
        device=device
    )

    # Wykres funkcji kosztu
    plot_training_curves(train_losses, val_losses)

    # Ewaluacja i metryki
    evaluate_model(model, test_loader, device=device)

    # Podgląd przykładowych przewidywań
    visualize_predictions(model, test_loader, device=device)

    # Podgląd błędnych przewidywań
    visualize_mistakes(model, test_loader, device=device)

    # Zapisz wytrenowany model
    torch.save(model.state_dict(), "model.pth")