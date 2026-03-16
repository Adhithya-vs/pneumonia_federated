import torch
import torch.nn as nn
from models.cnn_model import PneumoniaCNN
from utils import test_loader

# Load federated model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("models/federated_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# Evaluation loop
correct = 0
total = 0
test_loss = 0.0

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

print(f"✅ Federated Model Test Accuracy: {accuracy:.2f}%")
print(f"📉 Federated Model Test Loss: {avg_loss:.4f}")