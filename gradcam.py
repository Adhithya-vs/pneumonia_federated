import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.cnn_model import PneumoniaCNN
from utils import test_loader, train_data

# =========================================================
# LOAD MODEL
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_data.classes) if hasattr(train_data, 'classes') else 3
model = PneumoniaCNN(num_classes=num_classes).to(device)
model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device, weights_only=True)
)
model.eval()

# =========================================================
# PICK ONE TEST IMAGE
# =========================================================

images, labels = next(iter(test_loader))
image = images[0].unsqueeze(0).to(device)
label = labels[0]

with torch.no_grad():
    probs = torch.sigmoid(model(image))[0]

class_idx = torch.argmax(probs).item()

print(f"Classes         : {train_data.classes}")
print(f"True labels     : {label.tolist()}")
print(f"Predicted probs : {[f'{p:.2f}' for p in probs.tolist()]}")
print(f"GradCAM target  : class {class_idx} ({train_data.classes[class_idx]})")

# =========================================================
# GRAD-CAM
# =========================================================

gradients   = []
activations = []

def bwd_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def fwd_hook(module, input, output):
    activations.append(output)

target_layer = model.conv3   # last conv via property
fh = target_layer.register_forward_hook(fwd_hook)
bh = target_layer.register_full_backward_hook(bwd_hook)

output = model(image)
model.zero_grad()
output[0, class_idx].backward()

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

fh.remove()
bh.remove()

# =========================================================
# OVERLAY
# =========================================================

img_np  = image.squeeze().cpu().numpy()
img_np  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
img_rgb = np.stack([img_np, img_np, img_np], axis=2)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
overlay = np.clip(0.5 * img_rgb + 0.5 * heatmap, 0, 1)

# =========================================================
# SHOW + SAVE
# =========================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img_np, cmap='gray')
axes[0].set_title("Original X-Ray")
axes[0].axis('off')

axes[1].imshow(cam, cmap='jet')
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title(f"Overlay — {train_data.classes[class_idx]}")
axes[2].axis('off')

plt.tight_layout()
import os; os.makedirs("static", exist_ok=True)
plt.savefig("static/gradcam_result.png", dpi=120, bbox_inches='tight')
plt.show()
print("✅ Grad-CAM saved to static/gradcam_result.png")