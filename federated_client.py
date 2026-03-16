import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import PneumoniaCNN

# Local training function for a federated client
def train_local_model(global_weights, train_loader, epochs=2):
    """
    Trains a local model starting from the global weights.
    Returns the updated weights and dataset size for FedAvg.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)
    model.load_state_dict(global_weights)   # initialize with global model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"   Local Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Return weights + dataset size for weighted FedAvg
    return model.state_dict(), len(train_loader.dataset)