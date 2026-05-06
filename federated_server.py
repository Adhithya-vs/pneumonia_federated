import torch
import torch.nn as nn
import random
import os
from models.cnn_model import PneumoniaCNN
from utils import client_loaders, test_loader, train_data
from federated_client import train_local_model

# =========================================================
# DEVICE SETUP
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  Using CPU")

# =========================================================
# VALIDATE CLIENT LOADERS
# =========================================================

if len(client_loaders) == 0:
    raise RuntimeError(
        "❌ No client loaders found. "
        "Check your CSV files and image paths in utils.py."
    )

print(f"✅ {len(client_loaders)} federated clients available.")

# =========================================================
# INITIALIZE GLOBAL MODEL
# =========================================================

num_classes  = len(train_data.classes) if hasattr(train_data, 'classes') else 3
global_model = PneumoniaCNN(num_classes=num_classes).to(device)

os.makedirs("models", exist_ok=True)

# =========================================================
# FEDERATED AVERAGING
# =========================================================

def federated_avg(client_weights, client_sizes):
    """Weighted FedAvg aggregation."""
    total_size  = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = sum(
            client_sizes[i] * client_weights[i][key]
            for i in range(len(client_weights))
        ) / total_size

    global_model.load_state_dict(global_dict)

# =========================================================
# TRAINING CONFIG
# =========================================================

ROUNDS            = 30   # increased from 20
LOCAL_EPOCHS      = 3    # increased from 2
CLIENTS_PER_ROUND = min(2, len(client_loaders))

print(
    f"⚙️   Rounds: {ROUNDS} | "
    f"Local Epochs: {LOCAL_EPOCHS} | "
    f"Clients/Round: {CLIENTS_PER_ROUND}"
)

# =========================================================
# FEDERATED TRAINING LOOP
# =========================================================

best_accuracy = 0.0

for round_num in range(ROUNDS):
    print(f"\n📘 Round {round_num + 1}/{ROUNDS}")

    client_weights = []
    client_sizes   = []

    selected_clients = random.sample(
        range(len(client_loaders)),
        CLIENTS_PER_ROUND
    )

    for client_id in selected_clients:
        print(f"   🔹 Training client {client_id + 1}")
        weights, size = train_local_model(
            global_model.state_dict(),
            client_loaders[client_id],
            epochs=LOCAL_EPOCHS
        )
        client_weights.append(weights)
        client_sizes.append(size)

    if not client_weights:
        print("⚠️  No updates received — skipping round.")
        continue

    federated_avg(client_weights, client_sizes)

    # =====================================================
    # EVALUATE GLOBAL MODEL
    # =====================================================

    global_model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs   = global_model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total   += labels.numel()
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"🌍 Global Accuracy after Round {round_num + 1}: {acc:.2f}%")

    # Save round checkpoint
    torch.save(
        global_model.state_dict(),
        f"models/federated_model_round{round_num + 1}.pth"
    )

    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(global_model.state_dict(), "models/federated_best.pth")
        print(f"   💾 New best model saved ({acc:.2f}%)")

# =========================================================
# SAVE FINAL MODEL
# =========================================================

torch.save(global_model.state_dict(), "models/federated_global.pth")
print(f"\n✅ Training complete.")
print(f"   Best Accuracy : {best_accuracy:.2f}%")
print(f"   Final model   : models/federated_global.pth")
print(f"   Best model    : models/federated_best.pth")