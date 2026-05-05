import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
from PIL import Image
import numpy as np
import cv2
import csv
import os
from torch.utils.data import Dataset, DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Force grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom Dataset for Multi-label Classification
class MultiDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.items = []
        self.classes = []
        
        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                self.classes = header[1:] # e.g. ["Pneumonia", "COVID-19"]
                for row in reader:
                    self.items.append((row[0], [float(x) for x in row[1:]]))
        else:
            print(f"⚠️ Warning: {csv_file} not found. Please create it with headers [filename, Disease1, Disease2, ...]")
            self.classes = ["Pneumonia", "COVID-19", "Tuberculosis"] # Default fallback classes

    def __len__(self):
        return max(1, len(self.items)) # return 1 if empty to avoid DataLoader crashes during init

    def __getitem__(self, idx):
        if not self.items:
            # Fallback returning a blank image and zero labels if dataset isn't configured yet
            img = Image.new('L', (224, 224))
            labels = [0.0] * len(self.classes)
            if self.transform: img = self.transform(img)
            return img, torch.tensor(labels, dtype=torch.float32)

        img_name, labels = self.items[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            image = Image.new('L', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32)

# Load datasets
train_data = MultiDiseaseDataset(csv_file="data/chest_xray/train_labels.csv", img_dir="data/chest_xray/train_images", transform=transform)
val_data   = MultiDiseaseDataset(csv_file="data/chest_xray/val_labels.csv", img_dir="data/chest_xray/val_images", transform=transform)
test_data  = MultiDiseaseDataset(csv_file="data/chest_xray/test_labels.csv", img_dir="data/chest_xray/test_images", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# Quick check
print("Train samples:", len(train_data.items))
print("Validation samples:", len(val_data.items))
print("Test samples:", len(test_data.items))
print("Classes:", train_data.classes)


# -----------------------------
# Extra helpers for Flask app
# -----------------------------

# Preprocess uploaded image for inference
def preprocess_image(filepath):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(filepath).convert("L")  # ensure grayscale
    image = transform(image).unsqueeze(0)      # add batch dimension
    return image

# Generate Grad-CAM heatmap overlay
def generate_gradcam(model, image_tensor, class_idx, orig_path, save_path):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on last conv layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward + backward pass
    outputs = model(image_tensor)
    loss = outputs[0, class_idx]
    model.zero_grad()
    loss.backward()

    # Extract gradients & activations
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Overlay heatmap on original uploaded image
    orig_img = cv2.imread(orig_path)   # <-- FIX: use actual uploaded file path
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image at {orig_path}")
    orig_img = cv2.resize(orig_img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(save_path, overlay)