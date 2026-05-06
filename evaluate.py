import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from models.cnn_model import PneumoniaCNN
from utils import test_loader, train_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_path):
    """Evaluate a saved model on the test set. Returns (accuracy, avg_loss)."""
    num_classes = len(train_data.classes) if hasattr(train_data, 'classes') else 3
    model = PneumoniaCNN(num_classes=num_classes).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    correct   = 0
    total     = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs   = model(images)
            loss      = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total   += labels.numel()
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, test_loss / len(test_loader)


# =========================================================
# EVALUATE ALL FEDERATED ROUND CHECKPOINTS
# =========================================================

ROUNDS       = 30
accuracies   = []
losses       = []
valid_rounds = []

for round_num in range(1, ROUNDS + 1):
    path = f"models/federated_model_round{round_num}.pth"
    if not os.path.exists(path):
        print(f"⚠️  Round {round_num} checkpoint not found — skipping.")
        continue
    acc, loss = evaluate_model(path)
    accuracies.append(acc)
    losses.append(loss)
    valid_rounds.append(round_num)
    print(f"🔄 Round {round_num:>2}: Accuracy = {acc:.2f}% | Loss = {loss:.4f}")

# =========================================================
# EVALUATE NAMED MODELS
# =========================================================

for label, path in [
    ("Best federated",  "models/federated_best.pth"),
    ("Final federated", "models/federated_global.pth"),
    ("Centralised",     "models/best_model.pth"),
]:
    if os.path.exists(path):
        acc, loss = evaluate_model(path)
        print(f"\n{'=' * 45}")
        print(f"  {label}")
        print(f"  Accuracy : {acc:.2f}%")
        print(f"  Loss     : {loss:.4f}")
        print(f"{'=' * 45}")
    else:
        print(f"\n⚠️  {label} not found at {path}")

# =========================================================
# PLOT
# =========================================================

if valid_rounds:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)", color="steelblue")
    ax1.plot(valid_rounds, accuracies, marker='o', color="steelblue", label="Accuracy (%)")
    ax1.tick_params(axis='y', labelcolor="steelblue")
    ax1.set_ylim(80, 100)

    ax2 = ax1.twinx()
    ax2.set_ylabel("BCE Loss", color="tomato")
    ax2.plot(valid_rounds, losses, marker='s', linestyle='--', color="tomato", label="Loss")
    ax2.tick_params(axis='y', labelcolor="tomato")

    plt.title("Federated Model Performance Across Rounds")
    fig.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/federated_performance.png", dpi=120)
    plt.show()
    print("\n📊 Plot saved to static/federated_performance.png")