import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
from PIL import Image
import numpy as np
import cv2

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Force grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
train_data = datasets.ImageFolder(root="data/chest_xray/train", transform=transform)
val_data   = datasets.ImageFolder(root="data/chest_xray/val", transform=transform)
test_data  = datasets.ImageFolder(root="data/chest_xray/test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# Quick check
print("Train samples:", len(train_data))
print("Validation samples:", len(val_data))
print("Test samples:", len(test_data))
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