import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.cnn_model import PneumoniaCNN
from utils import test_loader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# Pick one test image
images, labels = next(iter(test_loader))
image = images[0].unsqueeze(0).to(device)
label = labels[0].item()

# Grad-CAM setup (no use_cuda argument here)
target_layer = model.conv3
cam = GradCAM(model=model, target_layers=[target_layer])

# Generate CAM
grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(label)])
grayscale_cam = grayscale_cam[0, :]  # first image

# Overlay CAM on original image
img = image.squeeze().cpu().numpy()
img = np.stack([img, img, img], axis=2)  # convert grayscale to 3-channel
img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1]

visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# Show result
plt.imshow(visualization)
plt.title(f"Grad-CAM (Label: {label})")
plt.axis("off")
plt.show()

# Save result (for Flask later)
output_path = "static/gradcam_result.png"
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ Grad-CAM result saved to {output_path}")