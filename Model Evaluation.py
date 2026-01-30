#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# METHOD 1
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ===========================
# 1. Model Setup
# ===========================
NUM_CLASSES = 11
MODEL_PATH = "/kaggle/input/vehicle-classification/pytorch/default/1/efficientnet_b0_best.pth"  # update path if needed

# 11 classes in correct order
class_names = [
    "articulated truck",
    "background",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "non-motorized vehicle",
    "pedestrian",
    "pickup truck",
    "single unit truck",
    "work van"
]

# Create EfficientNet-B0 architecture
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)

# Load checkpoint (your .pth is a checkpoint)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===========================
# 2. Image Transform
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===========================
# 3. Prediction Function
# ===========================
def predict(img_path):
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # Add batch dimension

    # Run model
    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(1).item()

    # Return class name
    return class_names[pred_idx]

# ===========================
# 4. Example Usage
# ===========================
# Replace with your actual image path
image_path = "/kaggle/input/timggg/2023-EQE350-SUV-AVP-DR.webp"

print("Predicted class:", predict(image_path))



#METHOD 2
from PIL import Image
import matplotlib.pyplot as plt

image_path = "/kaggle/input/timggg/2023-EQE350-SUV-AVP-DR.webp"  # change to your image
predicted_class = predict(image_path)

# Show image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted Class: {predicted_class}")
plt.show()

