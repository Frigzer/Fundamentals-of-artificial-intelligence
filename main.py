from model import ASLClassifier
from train import train_model
from serialize_data import get_serialized_dataloaders
import torch

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_serialized_dataloaders(batch_size=32)
    model = ASLClassifier(num_classes=36)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}\n")
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)
