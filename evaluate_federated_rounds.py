import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.cnn_model import PneumoniaCNN
from utils import test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path):
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct, total, test_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# Evaluate all rounds
ROUNDS = 5  # adjust if you trained more
accuracies = []
losses = []

for round in range(1, ROUNDS+1):
    model_path = f"models/federated_model_round{round}.pth"
    acc, loss = evaluate_model(model_path)
    accuracies.append(acc)
    losses.append(loss)
    print(f"🔄 Round {round}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}")

# Plot accuracy vs. round
plt.figure(figsize=(8,5))
plt.plot(range(1, ROUNDS+1), accuracies, marker='o', label="Accuracy (%)")
plt.plot(range(1, ROUNDS+1), losses, marker='s', label="Loss")
plt.title("Federated Model Performance Across Rounds")
plt.xlabel("Round")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()