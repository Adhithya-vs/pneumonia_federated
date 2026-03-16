import torch
import torch.nn as nn
import random
from models.cnn_model import PneumoniaCNN
from utils import client_loaders, test_loader   # client_loaders = [train_loader_client1, train_loader_client2, ...]
from federated_client import train_local_model  # import the client function

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Using CPU")

# Initialize global model
global_model = PneumoniaCNN().to(device)

def federated_avg(client_weights, client_sizes):
    """Weighted average of client updates"""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = sum(
            client_sizes[i] * client_weights[i][key] for i in range(len(client_weights))
        ) / sum(client_sizes)
    global_model.load_state_dict(global_dict)

# Federated training loop
ROUNDS = 20          # more rounds for better convergence
LOCAL_EPOCHS = 2     # local training epochs per client
CLIENTS_PER_ROUND = 3  # sample subset of clients each round

for round in range(ROUNDS):
    print(f"\n📘 Round {round+1}/{ROUNDS}")
    client_weights = []
    client_sizes = []

    # Random client sampling
    selected_clients = random.sample(range(len(client_loaders)), CLIENTS_PER_ROUND)

    for client_id in selected_clients:
        print(f"   🔹 Training client {client_id+1}")
        weights, size = train_local_model(global_model.state_dict(), client_loaders[client_id], epochs=LOCAL_EPOCHS)
        client_weights.append(weights)
        client_sizes.append(size)

    # Aggregate updates
    federated_avg(client_weights, client_sizes)

    # Evaluate global model
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"🌍 Global Accuracy after Round {round+1}: {acc:.2f}%")

# Save final global model
torch.save(global_model.state_dict(), "models/federated_global.pth")
print("✅ Federated global model saved!")