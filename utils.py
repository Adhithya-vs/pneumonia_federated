import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import csv
import os

# =========================================================
# IMAGE TRANSFORMS
# Train: with augmentation | Val/Test: clean
# =========================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Keep a plain transform alias for Flask inference
transform = val_transform

# =========================================================
# CUSTOM DATASET FOR MULTI-LABEL CLASSIFICATION
# =========================================================

class MultiDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.items     = []
        self.classes   = []

        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                self.classes = header[1:]
                for row in reader:
                    self.items.append((
                        row[0],
                        [float(x) for x in row[1:]]
                    ))
        else:
            print(
                f"⚠️  Warning: {csv_file} not found.\n"
                "Please run create_csv.py first."
            )
            self.classes = ["pneumonia", "covid", "tuberculosis"]

    def __len__(self):
        return max(1, len(self.items))

    def __getitem__(self, idx):
        if not self.items:
            img = Image.new('L', (224, 224))
            labels = [0.0] * len(self.classes)
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(labels, dtype=torch.float32)

        img_name, labels = self.items[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("L")
        except Exception:
            print(f"⚠️  Missing image: {img_path}")
            image = Image.new('L', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)

# =========================================================
# LOAD DATASETS
# =========================================================

BASE_PATH = "data/chest_xray"

train_data = MultiDiseaseDataset(
    csv_file=os.path.join(BASE_PATH, "train_labels.csv"),
    img_dir=os.path.join(BASE_PATH, "train_images"),
    transform=train_transform        # augmented
)

val_data = MultiDiseaseDataset(
    csv_file=os.path.join(BASE_PATH, "val_labels.csv"),
    img_dir=os.path.join(BASE_PATH, "val_images"),
    transform=val_transform          # clean
)

test_data = MultiDiseaseDataset(
    csv_file=os.path.join(BASE_PATH, "test_labels.csv"),
    img_dir=os.path.join(BASE_PATH, "test_images"),
    transform=val_transform          # clean
)

# =========================================================
# STANDARD DATALOADERS
# =========================================================

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

print("Train samples     :", len(train_data.items))
print("Validation samples:", len(val_data.items))
print("Test samples      :", len(test_data.items))
print("Classes           :", train_data.classes)

# =========================================================
# FEDERATED LEARNING CLIENT LOADERS
# =========================================================

NUM_CLIENTS = 2

safe_len    = max(NUM_CLIENTS, len(train_data))
client_size = safe_len // NUM_CLIENTS
remaining   = safe_len - (client_size * NUM_CLIENTS)

split_sizes = [client_size] * NUM_CLIENTS
split_sizes[0] += remaining

client_datasets = random_split(train_data, split_sizes)

client_loaders = [
    DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    for ds in client_datasets
]

print(f"Created {NUM_CLIENTS} federated client loaders successfully")

# =========================================================
# PREPROCESS IMAGE FOR FLASK APP INFERENCE
# =========================================================

def preprocess_image(filepath):
    """Load and preprocess a single image for model inference."""
    _transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(filepath).convert("L")
    return _transform(image).unsqueeze(0)

# =========================================================
# GENERATE GRAD-CAM HEATMAP
# =========================================================

def generate_gradcam(model, image_tensor, class_idx, orig_path, save_path):
    """Grad-CAM overlay saved to save_path."""
    model.eval()

    gradients   = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Find last Conv2d layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in model.")

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        outputs = model(image_tensor)
        loss    = outputs[0, class_idx]
        model.zero_grad()
        loss.backward()

        grads = gradients[0].cpu().data.numpy()[0]
        acts  = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        orig_img = cv2.imread(orig_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load image at {orig_path}")
        orig_img = cv2.resize(orig_img, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(save_path, overlay)
        print(f"Grad-CAM saved to: {save_path}")

    finally:
        fwd_handle.remove()
        bwd_handle.remove()