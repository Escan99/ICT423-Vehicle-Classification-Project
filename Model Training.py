#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================================
# CELL 1: Install Required Packages
# ============================================================================
# Run this cell first - it will take 1-2 minutes

get_ipython().system('pip install -q torch torchvision timm pandas matplotlib seaborn scikit-learn pillow tqdm')

print("✓ All packages installed successfully!")

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

import os
import tarfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

print("✓ All libraries imported successfully!")

# ============================================================================
# CELL 3: Check GPU and Setup Device
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected - training will be slower")
    print("Tip: Enable GPU in Runtime → Change runtime type (Colab)")
    print("     Or Accelerator → GPU T4 x2 (Kaggle)")

# ============================================================================
# CELL 4: Create Directory Structure
# ============================================================================

from pathlib import Path

# Define directory paths
BASE_DIR = Path("/kaggle/working/")
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create all directories
print("Creating directory structure...")
for dir_path in [BASE_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {dir_path}")

print("\n" + "="*60)
print("Directory structure created successfully!")
print("="*60)

# Verify directories exist
print("\nVerifying directories:")
for dir_path in [BASE_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    exists = "✓ EXISTS" if dir_path.exists() else "✗ MISSING"
    print(f"{exists}: {dir_path}")

print("\nDirectory structure:")
print(f"📁 {BASE_DIR}")
print(f"  ├── 📁 dataset/    (for extracted dataset)")
print(f"  ├── 📁 models/     (for saved model weights)")
print(f"  └── 📁 results/    (for plots and metrics)")

# ==========================================================
# CELL 5: Download Dataset 
# ============================================================================

print("="*60)
print("DOWNLOADING DATASET")
print("="*60)

import gdown

# Your dataset URL
DATASET_URL = "https://drive.google.com/uc?id=1tHvA7_DSChKHqBNWeUhyA1Azh1x0u2uu"
FILE_ID = "1tHvA7_DSChKHqBNWeUhyA1Azh1x0u2uu"

tar_path = BASE_DIR / "dataset.tar"

# METHOD 1: Try gdown with fuzzy download (bypasses some limits)
print("\nMETHOD 1: Trying gdown with --fuzzy flag...")
try:
    fuzzy_url = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"
    cmd = f"gdown --fuzzy '{fuzzy_url}' -O {str(tar_path)}"
    os.system(cmd)
    
    if tar_path.exists() and tar_path.stat().st_size > 1000:
        file_size_mb = tar_path.stat().st_size / (1024*1024)
        print(f"✓ Download complete! File size: {file_size_mb:.2f} MB")
    else:
        raise Exception("Download failed or file too small")
        
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    
# Final verification
if tar_path.exists():
    file_size_mb = tar_path.stat().st_size / (1024*1024)
    print(f"\n{'='*60}")
    print(f"✓ DATASET READY!")
    print(f"{'='*60}")
    print(f"File: {tar_path}")
    print(f"Size: {file_size_mb:.2f} MB")
else:
    print("\n❌ All download methods failed!")
    print("Please manually download and upload to Google Drive")

# ==================================================================
# CELL 6: Extract Dataset
# ============================================================================

print("\n" + "="*60)
print("EXTRACTING DATASET")
print("="*60)

print("\nExtracting dataset (this may take a while for large files)...")


with tarfile.open(tar_path, 'r') as tar:
    members = tar.getmembers()
    print(f"Total files to extract: {len(members)}")
    tar.extractall(path=BASE_DIR)

print(f"✓ Dataset extracted to: {DATASET_DIR}")

# Verify extraction
if DATASET_DIR.exists():
    print(f"\n✓ Extraction successful!")
    print(f"\nDataset contents:")
    get_ipython().system('ls -lh {DATASET_DIR}')
else:
    print("❌ Extraction failed - dataset directory not found")

# ======================================
# CELL 7: Debug - Show actual files vs CSV
# ======================================
import os
import pandas as pd
from pathlib import Path

# Find dataset
DATASET_DIR = Path("/kaggle/working")

# Show CSV content
train_csv = pd.read_csv(DATASET_DIR / 'gt_train.csv', header=None, names=['image_id', 'class'])
print("First 5 CSV entries:")
print(train_csv.head())

# Show actual files
print("\nFirst 10 actual files in train folder:")
files = os.listdir(DATASET_DIR / 'train')[:10]
for f in files:
    print(f"  {f}")

print("\nComparison:")
print(f"CSV image_id example: {train_csv['image_id'].iloc[0]}")
print(f"Actual file example: {files[0]}")
# ======================================
# CELL 8: Load and Prepare Data (FIXED FOR SUBFOLDER STRUCTURE)
# ============================================================================

print("\n" + "="*60)
print("LOADING AND PREPARING DATA")
print("="*60)

# Find dataset location
print("\nSearching for dataset location...")
possible_paths = [
    BASE_DIR / "dataset",
    BASE_DIR,
    Path("/kaggle/working"),
    Path("/content/vehicle_classification"),
]

actual_dataset_dir = None
for path in possible_paths:
    train_path = path / "train"
    csv_path = path / "gt_train.csv"
    if train_path.exists() and csv_path.exists():
        actual_dataset_dir = path
        print(f"✓ Found dataset at: {actual_dataset_dir}")
        break

if actual_dataset_dir is None:
    raise FileNotFoundError("Dataset not found.")

DATASET_DIR = actual_dataset_dir

def load_and_prepare_data(dataset_dir):
    """Load CSV annotations and prepare dataset"""
    
    # Load CSV
    train_csv = pd.read_csv(dataset_dir / 'gt_train.csv', header=None, names=['image_id', 'class'])
    
    print(f"\nDataset structure detected:")
    print(f"CSV has {len(train_csv)} entries")
    print(f"Sample CSV entry: {train_csv['image_id'].iloc[0]} -> {train_csv['class'].iloc[0]}")
    
    # Check train folder structure
    train_folder = dataset_dir / 'train'
    contents = os.listdir(train_folder)
    
    # Detect if images are in class subfolders
    first_item = train_folder / contents[0]
    if first_item.is_dir():
        print(f"\n✓ Images organized in class subfolders")
        print(f"Class folders: {contents}")
        
        # Build a lookup of all images with their full paths
        print("\nScanning for images in subfolders...")
        image_lookup = {}
        
        for class_folder in contents:
            class_path = train_folder / class_folder
            if class_path.is_dir():
                files = os.listdir(class_path)
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Extract image ID from filename (remove extension)
                        img_id = file.rsplit('.', 1)[0]
                        # Remove leading zeros if present
                        img_id_clean = str(int(img_id)) if img_id.isdigit() else img_id
                        image_lookup[img_id_clean] = str(class_path / file)
        
        print(f"✓ Found {len(image_lookup)} images")
        
        # Map CSV image_ids to actual file paths
        train_csv['image_path'] = train_csv['image_id'].astype(str).map(image_lookup)
        
        # Check for missing images
        missing = train_csv['image_path'].isna().sum()
        if missing > 0:
            print(f"⚠️  Warning: {missing} images not found")
            # Remove rows with missing images
            train_csv = train_csv.dropna(subset=['image_path'])
            print(f"Proceeding with {len(train_csv)} images that were found")
    
    else:
        print("✓ Images directly in train folder")
        # Original logic for direct image files
        # Try to detect extension
        extensions = ['.jpg', '.jpeg', '.png', '.JPG']
        found_ext = None
        test_id = str(train_csv['image_id'].iloc[0])
        
        for ext in extensions:
            test_path = train_folder / f"{test_id}{ext}"
            if test_path.exists():
                found_ext = ext
                break
        
        if found_ext:
            train_csv['image_path'] = train_csv['image_id'].astype(str).apply(
                lambda x: str(train_folder / f"{x}{found_ext}")
            )
        else:
            raise FileNotFoundError("Could not determine image file format")
    
    # Handle test set
    test_csv = None
    test_csv_path = dataset_dir / 'gt_test.csv'
    if test_csv_path.exists():
        test_csv = pd.read_csv(test_csv_path, header=None, names=['image_id', 'class'])
        
        test_folder = dataset_dir / 'test'
        if (test_folder / os.listdir(test_folder)[0]).is_dir():
            # Test also has subfolders
            image_lookup_test = {}
            for class_folder in os.listdir(test_folder):
                class_path = test_folder / class_folder
                if class_path.is_dir():
                    files = os.listdir(class_path)
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_id = file.rsplit('.', 1)[0]
                            img_id_clean = str(int(img_id)) if img_id.isdigit() else img_id
                            image_lookup_test[img_id_clean] = str(class_path / file)
            
            test_csv['image_path'] = test_csv['image_id'].astype(str).map(image_lookup_test)
            test_csv = test_csv.dropna(subset=['image_path'])
    
    # Get unique classes
    classes = sorted(train_csv['class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Add numeric labels
    train_csv['label'] = train_csv['class'].map(class_to_idx)
    if test_csv is not None:
        test_csv['label'] = test_csv['class'].map(class_to_idx)
    
    print(f"\nDataset Statistics:")
    print(f"Total training images: {len(train_csv)}")
    if test_csv is not None:
        print(f"Total test images: {len(test_csv)}")
    print(f"Number of classes: {len(classes)}")
    print(f"\nClasses: {classes}")
    print(f"\nClass distribution:")
    print(train_csv['class'].value_counts())
    
    # Verify a few paths
    print(f"\nSample image paths:")
    for path in train_csv['image_path'].head(3):
        exists = "✓" if Path(path).exists() else "✗"
        print(f"{exists} {path}")
    
    return train_csv, test_csv, classes, class_to_idx

# Load data
train_df, test_df, CLASSES, class_to_idx = load_and_prepare_data(DATASET_DIR)

# Create validation split
if test_df is None:
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
    print(f"\n✓ Created validation split: {len(train_df)} train, {len(val_df)} validation")
else:
    val_df = test_df
    print(f"\n✓ Using provided test set for validation")

# ======================================================
# CELL 9: Create Custom Dataset Class
# ============================================================================

class VehicleDataset(Dataset):
    """Custom Dataset for vehicle classification"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

print("✓ VehicleDataset class defined")

# ===================================================
# CELL 10: Setup Data Augmentation and Create DataLoaders (OPTIMIZED)
# ============================================================================

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = VehicleDataset(train_df, transform=train_transform)
val_dataset = VehicleDataset(val_df, transform=val_transform)

# MEMORY OPTIMIZATION: Reduce batch size and workers
BATCH_SIZE = 32  # Use 8 if still having issues
NUM_WORKERS = 0  # Set to 0 to reduce memory usage (was 2)

# Create dataloaders with memory optimizations
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=False,  # Disable pin_memory to save RAM
    persistent_workers=False  # Don't keep workers alive
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=False
)

print("✓ DataLoaders created (MEMORY OPTIMIZED):")
print(f"  Train batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Num workers: {NUM_WORKERS}")
print("\n💡 If still running out of memory, reduce BATCH_SIZE to 8 or 4")

# ================================================================
# CELL 11: Define VehicleClassifier Class
# ============================================================================

class VehicleClassifier:
    """
    Implements two-level transfer learning:
    Level 1: Pre-trained weights from ImageNet
    Level 2: Location-specific fine-tuning
    """
    
    def __init__(self, model_name='resnet50', num_classes=7, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model = self._build_model(pretrained)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def _build_model(self, pretrained):
        """Build model architecture"""
        print(f"\n{'='*60}")
        print(f"LEVEL 1: Loading {self.model_name} with ImageNet pre-trained weights")
        print(f"{'='*60}")
        
        if 'resnet' in self.model_name:
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif self.model_name == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                
        elif 'efficientnet' in self.model_name:
            model = timm.create_model(self.model_name, pretrained=pretrained, num_classes=self.num_classes)
            
        elif 'vit' in self.model_name:
            model = timm.create_model(self.model_name, pretrained=pretrained, num_classes=self.num_classes)
            
        elif 'convnext' in self.model_name:
            model = timm.create_model(self.model_name, pretrained=pretrained, num_classes=self.num_classes)
        
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        model = model.to(self.device)
        print(f"✓ Model loaded successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def fine_tune(self, train_loader, val_loader, epochs=30, lr=0.001, patience=5):
        """Level 2: Location-specific fine-tuning"""
        print(f"\n{'='*60}")
        print(f"LEVEL 2: Fine-tuning on location-specific dataset")
        print(f"{'='*60}\n")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, MODELS_DIR / f'{self.model_name}_best.pth')
                print(f"✓ Best model saved! (Val Acc: {best_acc:.2f}%)")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
        print(f"{'='*60}")
        
        return best_acc
    
    def load_best_model(self):
        """Load the best saved model"""
        checkpoint = torch.load(MODELS_DIR / f'{self.model_name}_best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Best model loaded (Val Acc: {checkpoint['best_acc']:.2f}%)")
    
    def evaluate(self, val_loader, class_names):
        """Comprehensive evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, cm, all_preds, all_labels

print("✓ VehicleClassifier class defined")

# =============================================================
# CELL 12: Define Visualization Functions
# ============================================================================

def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results):
    """Plot comparison of different architectures"""
    df = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model Architecture Comparison', fontsize=16)
    
    axes[0].bar(df.index, df['accuracy'], color='steelblue')
    axes[0].set_title('Accuracy')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(df.index, df['model_size_mb'], color='coral')
    axes[1].set_title('Model Size')
    axes[1].set_ylabel('Size (MB)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

print("✓ Visualization functions defined")

# =========================================================
# ============================================================================
# CELL 13: AGGRESSIVE MEMORY OPTIMIZATION
# ============================================================================

import gc
import torch

# Clear all memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Reduce batch size even more
BATCH_SIZE = 32  # Very small batch size

# Recreate dataloaders with minimal memory
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

print(f"✓ DataLoaders recreated with BATCH_SIZE={BATCH_SIZE}")
# ============================================================================
# CELL 14: TRAIN SINGLE MODEL (Run this for quick testing)
# ============================================================================

print("\n" + "="*60)
print("OPTION 1: SINGLE MODEL TRAINING")
print("="*60 + "\n")

# Initialize classifier
classifier = VehicleClassifier(
    model_name='efficientnet_b0',  # Options: resnet50, efficientnet_b0, resnet101, etc.
    num_classes=len(CLASSES),
    pretrained=True
)

# Train the model
best_acc = classifier.fine_tune(
    train_loader, 
    val_loader, 
    epochs=3,      # Adjust as needed
    lr=0.001,       # Learning rate
    patience=5      # Early stopping patience
)

# ============================================================================
# CELL 15: GET PREDICTIONS FOR VISUALIZATION
# ============================================================================

print("Getting predictions from validation set...")

classifier.model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in val_loader:
        images = images.to(device)
        outputs = classifier.model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

preds = np.array(all_preds)
labels = np.array(all_labels)

print(f"✓ Got predictions for {len(labels)} validation images")
print(f"Validation Accuracy: {(preds == labels).mean() * 100:.2f}%")
# ============================================================================
# CELL 16: COMPREHENSIVE VISUALIZATION CODE
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# 1. TRAINING HISTORY PLOTS
# ============================================================================

def plot_training_history(history, model_name, save_dir):
    """Plot training and validation loss and accuracy"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plot saved")

# ============================================================================
# 2. CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir, normalize=False):
    """Plot confusion matrix with percentages"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    suffix = '_normalized' if normalize else ''
    plt.savefig(f'{save_dir}/{model_name}_confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix plot saved")

# ============================================================================
# 3. PER-CLASS PERFORMANCE
# ============================================================================

def plot_per_class_metrics(y_true, y_pred, class_names, model_name, save_dir):
    """Plot precision, recall, F1-score for each class"""
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    df = df.sort_values('F1-Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    ax.barh(x - width, df['Precision'], width, label='Precision', color='#3498db')
    ax.barh(x, df['Recall'], width, label='Recall', color='#2ecc71')
    ax.barh(x + width, df['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_ylabel('Vehicle Class', fontsize=12, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Class'])
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Per-class performance plot saved")
    
    return df

# ============================================================================
# 4. CLASS DISTRIBUTION
# ============================================================================

def plot_class_distribution(train_df, val_df, class_names, save_dir):
    """Plot class distribution in train and validation sets"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set distribution
    train_counts = train_df['class'].value_counts()
    axes[0].barh(range(len(train_counts)), train_counts.values, color='steelblue')
    axes[0].set_yticks(range(len(train_counts)))
    axes[0].set_yticklabels(train_counts.index)
    axes[0].set_xlabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add counts on bars
    for i, v in enumerate(train_counts.values):
        axes[0].text(v + max(train_counts.values)*0.01, i, f'{v:,}', va='center')
    
    # Validation set distribution
    val_counts = val_df['class'].value_counts()
    axes[1].barh(range(len(val_counts)), val_counts.values, color='coral')
    axes[1].set_yticks(range(len(val_counts)))
    axes[1].set_yticklabels(val_counts.index)
    axes[1].set_xlabel('Number of Images', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Set Class Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(val_counts.values):
        axes[1].text(v + max(val_counts.values)*0.01, i, f'{v:,}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Class distribution plot saved")

# ============================================================================
# 5. SAMPLE PREDICTIONS VISUALIZATION
# ============================================================================

def plot_sample_predictions(model, val_loader, class_names, device, save_dir, num_samples=16):
    """Display sample predictions with confidence scores"""
    
    model.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    images_shown = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            for i in range(len(images)):
                if images_shown >= num_samples:
                    break
                
                # Denormalize image
                img = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                true_label = class_names[labels[i]]
                pred_label = class_names[predictions[i]]
                confidence = confidences[i].item()
                
                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')
                
                color = 'green' if true_label == pred_label else 'red'
                title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}'
                axes[images_shown].set_title(title, fontsize=10, color=color, fontweight='bold')
                
                images_shown += 1
            
            if images_shown >= num_samples:
                break
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Sample predictions plot saved")

# ============================================================================
# 6. LEARNING CURVES
# ============================================================================

def plot_learning_curves(history, save_dir):
    """Plot learning curves with error bands"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with fill between for visual appeal
    ax.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax.fill_between(epochs, history['train_acc'], alpha=0.2, color='blue')
    ax.fill_between(epochs, history['val_acc'], alpha=0.2, color='red')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curves - Accuracy Over Time', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add best validation accuracy marker
    best_epoch = np.argmax(history['val_acc'])
    best_acc = history['val_acc'][best_epoch]
    ax.plot(best_epoch + 1, best_acc, 'r*', markersize=20, label=f'Best Val Acc: {best_acc:.2f}%')
    ax.annotate(f'Best: {best_acc:.2f}%', 
                xy=(best_epoch + 1, best_acc), 
                xytext=(best_epoch + 1, best_acc - 5),
                fontsize=11, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Learning curves plot saved")

# ============================================================================
# 7. COMPREHENSIVE REPORT
# ============================================================================

def generate_comprehensive_report(y_true, y_pred, class_names, model_name, save_dir):
    """Generate and save detailed classification report"""
    
    from sklearn.metrics import classification_report, accuracy_score
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Save to file
    with open(f'{save_dir}/{model_name}_classification_report.txt', 'w') as f:
        f.write(f"VEHICLE CLASSIFICATION REPORT - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(report)
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print('='*80)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print(report)
    print(f"✓ Detailed report saved to {save_dir}/{model_name}_classification_report.txt")

# ============================================================================
# USAGE EXAMPLE - Run this after training
# ============================================================================

# After training completes, run these visualizations:

# 1. Training history
plot_training_history(classifier.history, 'efficientnet_b0', RESULTS_DIR)

# 2. Confusion matrices (both normalized and not)
plot_confusion_matrix(labels, preds, CLASSES, 'efficientnet_b0', RESULTS_DIR, normalize=False)
plot_confusion_matrix(labels, preds, CLASSES, 'efficientnet_b0', RESULTS_DIR, normalize=True)

# 3. Per-class performance
performance_df = plot_per_class_metrics(labels, preds, CLASSES, 'efficientnet_b0', RESULTS_DIR)
print("\nPer-Class Performance Summary:")
print(performance_df)

# 4. Class distribution
plot_class_distribution(train_df, val_df, CLASSES, RESULTS_DIR)

# 5. Sample predictions
plot_sample_predictions(classifier.model, val_loader, CLASSES, device, RESULTS_DIR)

# 6. Learning curves
plot_learning_curves(classifier.history, RESULTS_DIR)

# 7. Comprehensive report
generate_comprehensive_report(labels, preds, CLASSES, 'efficientnet_b0', RESULTS_DIR)

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"All plots saved to: {RESULTS_DIR}")
# ==================================================
# CELL 17: DOWNLOAD ALL OUTPUTS - FINAL COMPREHENSIVE ZIP
# ==================================================
import os
import shutil

print("Creating comprehensive output package...")

# Create a clean output folder
output_folder = '/kaggle/working/FINAL_OUTPUT'
os.makedirs(output_folder, exist_ok=True)

# Copy trained model
print("📦 Copying model...")
shutil.copy('/kaggle/working/models/efficientnet_b0_best.pth', 
            f'{output_folder}/efficientnet_b0_best.pth')

# Copy all visualizations
print("📊 Copying visualizations...")
if os.path.exists('/kaggle/working/results'):
    for file in os.listdir('/kaggle/working/results'):
        if file.endswith(('.png', '.jpg', '.txt', '.csv')):
            shutil.copy(f'/kaggle/working/results/{file}', 
                       f'{output_folder}/{file}')

# Create a summary file
print("📝 Creating summary...")
with open(f'{output_folder}/SUMMARY.txt', 'w') as f:
    f.write("VEHICLE CLASSIFICATION - TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: EfficientNet-B0\n")
    f.write(f"Dataset: {len(train_df)} training, {len(val_df)} validation images\n")
    f.write(f"Classes: {len(CLASSES)}\n")
    f.write(f"Class names: {', '.join(CLASSES)}\n\n")
    f.write("Files included:\n")
    f.write("- efficientnet_b0_best.pth (trained model weights)\n")
    f.write("- All visualization plots (.png files)\n")
    f.write("- Classification report (.txt file)\n")

# Zip everything
print("🗜️ Creating final zip file...")
shutil.make_archive('/kaggle/working/COMPLETE_OUTPUT', 'zip', output_folder)

file_size = os.path.getsize('/kaggle/working/COMPLETE_OUTPUT.zip') / (1024*1024)
print(f"\n✅ SUCCESS!")
print(f"="*60)
print(f"File: COMPLETE_OUTPUT.zip")
print(f"Size: {file_size:.2f} MB")
print(f"Location: /kaggle/working/COMPLETE_OUTPUT.zip")
print(f"="*60)

print("\n📥 TO DOWNLOAD:")
print("1. Click folder icon (📁) on left sidebar")
print("2. Navigate to /kaggle/working/")
print("3. Find COMPLETE_OUTPUT.zip")
print("4. Right-click → Download")

print("\n📦 Package contains:")
print("  ✓ Trained model (.pth file)")
print("  ✓ All plots and visualizations")
print("  ✓ Classification reports")
print("  ✓ Summary file")
# ============================================================================
# CELL 17: SAVE OUTPUTS TO DEVICE
# ============================================================================
FileLink("COMPLETE_OUTPUT.zip")

