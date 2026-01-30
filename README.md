# ICT423-Vehicle-Classification-Project with Transfer Learning for Domain Adaptation

A deep learning model for fine-grained vehicle classification using transfer learning to address domain shift across different geographic locations.

# Overview

This project implements a two-level transfer learning approach for vehicle classification, achieving significant accuracy improvements (from ~30% to 95%) when deploying models across different geographic locations.

# Key Features

- **7-Class Fine-Grained Classification**: car, bus, taxi, bike, pickup, truck, trailer
- **Two-Level Transfer Learning**:
  - Level 1: Pre-trained ImageNet weights
  - Level 2: Location-specific fine-tuning
- **Domain Adaptation**: Robust performance across different geographic deployments
- **Multiple Architecture Support**: ResNet, EfficientNet, ConvNeXt
- **Comprehensive Evaluation**: Training curves, confusion matrices, per-class metrics

# Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 4GB+ VRAM)
- 8GB RAM minimum (16GB recommended)

# Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-classification.git
cd vehicle-classification

# Install dependencies
pip install -r requirements.txt
```

# Usage

See 'Model Evaluation' for code.

# Model Performance

- **| Overall Accuracy | 97.18% |**
- **| Training Images | 415,331 |**
- **| Validation Images | 103,833 |**
- **| Number of Classes | 11 |**
- **| Model Size | ~43 MB |**

# Per-Class Performance

See `classification_report.txt` for detailed metrics including precision, recall, and F1-score for each vehicle class.

# Model Architecture

**Base Model**: EfficientNet-B0
- **Parameters**: ~5.3M
- **Input Size**: 224x224x3
- **Output Classes**: 11 vehicle categories

# Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (with cosine annealing)
- **Batch Size**: 4-16 (memory optimized)
- **Epochs**: 10-30 with early stopping
- **Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter
  - Random crop

# Research Contributions

1. **Empirical Evidence**: Demonstrates up to 50% performance degradation from domain shift
2. **Transfer Learning Framework**: Two-level approach for location adaptation
3. **Fine-Grained Classification**: 7-class system beyond typical 4-class categorization
4. **Architecture Comparison**: Evaluation of YOLO variants and CNNs for deployment
5. **Geographic Adaptation**: Methodology for location-aware vehicle classification

# Results & Visualizations

The model includes comprehensive visualizations:

- **Training History**: Loss and accuracy curves over epochs
- **Confusion Matrix**: Both raw counts and normalized percentages
- **Per-Class Performance**: Precision, recall, F1-score bar charts
- **Sample Predictions**: Visual grid showing correct/incorrect classifications
- **Learning Curves**: Accuracy progression with best checkpoint markers

# Advanced Usage

# Fine-tuning on New Location Data
```python
from torch.utils.data import DataLoader

# Load your location-specific dataset
new_train_loader = DataLoader(new_dataset, batch_size=16)

# Load pre-trained model
model = torch.load('efficientnet_b0_best.pth')

# Fine-tune with lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Train for fewer epochs
for epoch in range(5):
    # Training loop here
    pass
```

# Batch Prediction
```python
import os
from pathlib import Path

def predict_folder(model, folder_path, transform):
    results = []
    for img_file in Path(folder_path).glob('*.jpg'):
        image = Image.open(img_file)
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            results.append({
                'filename': img_file.name,
                'class': class_names[predicted.item()]
            })
    return results
```

# Troubleshooting

# Out of Memory Errors
- Reduce batch size to 4 or 8
- Use smaller model (efficientnet_b0 instead of resnet50)
- Resize images to 128x128 instead of 224x224

# Low Accuracy
- Ensure proper data normalization
- Check class distribution (imbalanced data?)
- Increase training epochs
- Try data augmentation

# Slow Training
- Enable GPU (check with `torch.cuda.is_available()`)
- Increase batch size if memory allows
- Reduce image resolution
- Use mixed precision training

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Acknowledgments

- Dataset: MIO-TCD
- Pre-trained models: ImageNet weights via PyTorch/timm
- Framework: PyTorch, torchvision
- Environment: Kaggle Notebooks

# Contact

- **Author**: Ajayi Enoch
- **Email**: ajayi.enoch213@gmail.com

# Updates

- **v1.0.0** (January 2025): Initial release with EfficientNet-B0
- Future plans: Multi-location dataset, real-time inference, model quantization

---

**Note**: This is a research model. For production deployment, additional validation and testing are recommended.
