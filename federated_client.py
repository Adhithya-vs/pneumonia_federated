import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import PneumoniaCNN


def train_local_model(global_weights, train_loader, epochs=2):
    """
    Trains a local model starting from the global weights.

    Args:
        global_weights : state_dict from the global model
        train_loader   : DataLoader for this client's local data
        epochs         : number of local training epochs

    Returns:
        (state_dict, dataset_size) for FedAvg aggregation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unwrap random_split to get base dataset classes
    base_dataset = train_loader.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    num_classes = len(base_dataset.classes) if hasattr(base_dataset, 'classes') else 3

    model = PneumoniaCNN(num_classes=num_classes).to(device)
    model.load_state_dict(global_weights)

    # Weighted loss — same as train.py
    pos_weight = torch.tensor([1.0, 5.0, 5.0]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct      = 0
        total        = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total   += labels.numel()
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"      Local Epoch {epoch + 1}/{epochs} | "
            f"Loss: {running_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )
        scheduler.step()

    return model.state_dict(), len(train_loader.dataset)