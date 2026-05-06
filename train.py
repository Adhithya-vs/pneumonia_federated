import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.cnn_model import PneumoniaCNN
from utils import train_loader, val_loader, train_data

# =========================================================
# DEVICE SETUP
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  Using CPU (no GPU detected)")

os.makedirs("models", exist_ok=True)

# =========================================================
# MODEL, LOSS, OPTIMIZER
# =========================================================

num_classes = len(train_data.classes) if hasattr(train_data, 'classes') else 3
model = PneumoniaCNN(num_classes=num_classes).to(device)

# Weighted loss — upweight covid & tuberculosis (rare in dataset)
pos_weight = torch.tensor([1.0, 5.0, 5.0]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Two-stage optimizer: lower LR for pretrained backbone, higher for new head
backbone_params = list(model.base.parameters())[:-2]
head_params     = list(model.base.fc.parameters())

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': head_params,     'lr': 1e-3}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# =========================================================
# TRAINING LOOP
# =========================================================

num_epochs   = 15
best_val_acc = 0.0

for epoch in range(num_epochs):

    # --- Training ---
    model.train()
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

    train_acc  = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch + 1:>2}/{num_epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    # --- Validation ---
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs   = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total   += labels.numel()
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"           Validation Accuracy : {val_acc:.2f}%")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("           ✅ Best model saved!")

print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.2f}%")