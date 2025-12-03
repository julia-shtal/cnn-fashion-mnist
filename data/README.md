# Dataset

This directory contains the Fashion MNIST dataset in preprocessed NPZ format for training and testing the CNN model.

## Files

```
data/
├── README.md           # This file
├── train_data.npz      # Training dataset (~48,000 samples)
├── test_data.npz       # Test dataset (~12,000 samples)
└── all_data.npz        # (Optional) Combined dataset before splitting
```

**Note:** `.npz` files are not tracked in Git. Download them from releases or generate from source images.

---

## Dataset Information

### Fashion MNIST Overview

Fashion MNIST is a dataset of Zalando's article images consisting of:
- **60,000 training images** (originally)
- **10,000 test images** (originally)
- **28×28 grayscale images** (784 pixels)
- **10 classes** representing different fashion categories

### Class Labels

| Label | Description | Examples |
|-------|-------------|----------|
| 0 | T-shirt/top | Simple T-shirts, tank tops |
| 1 | Trouser | Pants, jeans, dress pants |
| 2 | Pullover | Sweaters, hoodies, pullovers |
| 3 | Dress | All types of dresses |
| 4 | Coat | Winter coats, rain coats |
| 5 | Sandal | Open-toed footwear |
| 6 | Shirt | Button-up shirts, blouses |
| 7 | Sneaker | Athletic shoes, casual sneakers |
| 8 | Bag | Handbags, backpacks, purses |
| 9 | Ankle boot | Short boots |

### File Format

**NPZ Format** (Compressed NumPy Archive)
- **Keys**: `'X'` (images), `'T'` (labels)
- **X shape**: `(N, 28, 28, 1)` where N is number of samples
- **T shape**: `(N,)` - integer labels from 0-9
- **Data type**: X = float/uint8, T = int32
- **Pixel range**: Raw pixel values (typically 0-255, depends on source images)

> ⚠️ **Important:** Data stored in NPZ files is **NOT normalized**. Normalization (scaling to [0, 1]) is performed automatically by `train.py` during training. You are responsible for normalization if using the data outside of this project's training pipeline.

---

## Download Instructions

### Option 1: Download from GitHub Releases (Recommended)

**Using wget (Linux/Mac/Git Bash on Windows):**
```bash
# Navigate to data directory
cd data/

# Download training data
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/train_data.npz

# Download test data
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/test_data.npz
```

**Using curl (Alternative):**
```bash
cd data/
curl -L -O https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/train_data.npz
curl -L -O https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/test_data.npz
```

**Using PowerShell (Windows):**
```powershell
cd data

# Download training data
Invoke-WebRequest -Uri "https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/train_data.npz" -OutFile "train_data.npz"

# Download test data
Invoke-WebRequest -Uri "https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/test_data.npz" -OutFile "test_data.npz"
```

**Using Browser:**
1. Go to [Releases](https://github.com/julia-shtal/cnn-fashion-mnist/releases/latest)
2. Download `train_data.npz` and `test_data.npz`
3. Move files to the `data/` folder

---

### Option 2: Generate from Source Images

If you have the original Fashion MNIST images in PNG format:

```bash
# Step 1: Convert images to single NPZ file
python src/convert_data.py path/to/fashion_mnist_images 28 28 1 data/all_data.npz

# Step 2: Split into train and test sets
python src/split_data.py data/all_data.npz --ratio 0.8

# This creates:
# - data/train_data.npz (80% of data)
# - data/test_data.npz (20% of data)
```

**Image Requirements:**
- Format: PNG
- Naming convention: `<label>-<id>.png` (e.g., `0-1.png`, `5-42.png`)
- Organization: All images in a single folder
- Size: Any size (will be resized to 28×28)

---

## Verify Downloaded Data

After downloading, verify the files:

### Quick Check
```bash
# Check if files exist and their sizes
ls -lh data/*.npz

# Expected output:
# train_data.npz  ~20-25 MB
# test_data.npz   ~5-7 MB
```

### Detailed Verification
```python
import numpy as np

# Load training data
train_data = np.load('data/train_data.npz')
X_train, T_train = train_data['X'], train_data['T']

# Load test data
test_data = np.load('data/test_data.npz')
X_test, T_test = test_data['X'], test_data['T']

# Print statistics
print("Training Data:")
print(f"  Images shape: {X_train.shape}")
print(f"  Labels shape: {T_train.shape}")
print(f"  Pixel range: [{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"  Unique labels: {np.unique(T_train)}")
print(f"  Class distribution: {np.bincount(T_train)}")

print("\nTest Data:")
print(f"  Images shape: {X_test.shape}")
print(f"  Labels shape: {T_test.shape}")
print(f"  Pixel range: [{X_test.min():.1f}, {X_test.max():.1f}]")
print(f"  Unique labels: {np.unique(T_test)}")
print(f"  Class distribution: {np.bincount(T_test)}")
```

**Expected Output:**
```
Training Data:
  Images shape: (48000, 28, 28, 1)
  Labels shape: (48000,)
  Pixel range: [0.0, 255.0]  # Raw pixel values (not normalized)
  Unique labels: [0 1 2 3 4 5 6 7 8 9]
  Class distribution: [... distribution ...]

Test Data:
  Images shape: (12000, 28, 28, 1)
  Labels shape: (12000,)
  Pixel range: [0.0, 255.0]  # Raw pixel values (not normalized)
  Unique labels: [0 1 2 3 4 5 6 7 8 9]
  Class distribution: [... distribution ...]
```

---

## Usage Examples

### Load Data in Python

```python
import numpy as np

# Load training data
data = np.load('data/train_data.npz')
X_train = data['X']  # Shape: (N, 28, 28, 1) - Raw pixel values
T_train = data['T']  # Shape: (N,) - Labels

print(f"Loaded {len(X_train)} training samples")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")
```

### Visualize Samples

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('data/train_data.npz')
X, T = data['X'], data['T']

# Plot first 10 samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for idx, ax in enumerate(axes.flat):
    # Note: No normalization needed for visualization
    ax.imshow(X[idx].squeeze(), cmap='gray')
    ax.set_title(f'{class_names[T[idx]]} ({T[idx]})')
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()
```

### Use with PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FashionMNISTDataset(Dataset):
    def __init__(self, npz_path, normalize=True):
        data = np.load(npz_path)
        X = data['X']

        # Normalize if needed (for use outside train.py)
        if normalize:
            X = (X - X.min()) / (X.max() - X.min())

        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.T = torch.LongTensor(data['T'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

# Create dataset and loader
train_dataset = FashionMNISTDataset('data/train_data.npz', normalize=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Dataset size: {len(train_dataset)}")
```

### Using with train.py (Automatic Normalization)

```bash
# Normalization is handled automatically by train.py
python src/train.py data/train_data.npz data/test_data.npz

# The correct_data() function in train.py performs:
# 1. Normalization: X = (X - X.min()) / (X.max() - X.min())
# 2. Class balancing via oversampling
# 3. Class weight calculation
```

---

## Data Preprocessing

### What's Already Done

The NPZ files contain **partially preprocessed data**:
- ✅ Resized to 28×28 pixels
- ✅ Converted to grayscale (1 channel)
- ✅ Stored in (N, H, W, C) format
- ✅ Compressed for efficient storage

### What's NOT Done (User Responsibility)

The following are **NOT** applied to stored NPZ files:
- ❌ **Normalization** - Pixel values are raw (0-255 range typically)
- ❌ **Standardization** - No zero-mean, unit-variance scaling
- ❌ **Class balancing** - Original class distribution preserved
- ❌ **Data augmentation** - No rotations, flips, or transformations

### Automatic Preprocessing in train.py

When you run the training script, `correct_data()` function automatically applies:

```python
# From train.py - correct_data() function
def correct_data(X, T):
    # 1. Normalize to [0, 1] range
    X = (X - X.min()) / (X.max() - X.min())

    # 2. Balance dataset via oversampling
    X, T = balance_dataset(X, T)

    # 3. Calculate class weights for loss function
    unique_labels, counts = np.unique(T, return_counts=True)
    class_weights = torch.FloatTensor(1.0 / counts)
    class_weights = class_weights / class_weights.sum()

    return X, T, class_weights
```

### Manual Normalization (If Using Data Elsewhere)

If you're using the data outside of `train.py`, normalize it yourself:

```python
import numpy as np

# Load raw data
data = np.load('data/train_data.npz')
X = data['X']

# Method 1: Min-Max normalization to [0, 1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# Method 2: Standardization (zero mean, unit variance)
mean = X.mean()
std = X.std()
X_standardized = (X - mean) / std

# Method 3: Per-image normalization
X_per_image = np.array([
    (img - img.min()) / (img.max() - img.min() + 1e-7)
    for img in X
])
```

---

## File Sizes

| File | Approximate Size | Samples | Description |
|------|-----------------|---------|-------------|
| `train_data.npz` | 20-25 MB | ~48,000 | 80% of original data |
| `test_data.npz` | 5-7 MB | ~12,000 | 20% of original data |
| `all_data.npz` | 25-30 MB | ~60,000 | Combined (if generated) |

**Compression:** Files use NumPy's `savez_compressed` for efficient storage.

---

## ⚠️ Troubleshooting

### Issue: "File not found: train_data.npz"

**Solution:** Download the files from releases or generate them using the scripts above.

### Issue: "KeyError: 'X' is not a file in the archive"

**Solution:** You might have files from an older version. Re-download from releases or regenerate:
```bash
python src/split_data.py data/all_data.npz
```

### Issue: "Pixel values are in range [0, 1] but I expected [0, 255]"

**Solution:** This is normal if you're loading data that was already processed by `train.py`. The raw NPZ files should have [0, 255] range. If you need raw data, regenerate from source images.

### Issue: "Model performance is poor when using data directly"

**Solution:** You likely forgot to normalize the data. `train.py` does this automatically, but if using data elsewhere:
```python
X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
```

### Issue: "File size is 0 or corrupted"

**Solution:** 
1. Delete the corrupted file
2. Re-download from releases
3. Verify file integrity:
   ```bash
   # Linux/Mac
   md5sum data/train_data.npz

   # Windows PowerShell
   Get-FileHash data\train_data.npz -Algorithm MD5
   ```

### Issue: "Out of memory when loading data"

**Solution:** Load data in batches or use memory mapping:
```python
# Memory-mapped array (doesn't load entire file)
data = np.load('data/train_data.npz')
X = data['X']  # Only loads when accessed
```

---

## Data Source

Original Fashion MNIST dataset:
- **Paper:** "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"
- **Authors:** Han Xiao, Kashif Rasul, Roland Vollgraf (Zalando Research)
- **Year:** 2017
- **Repository:** [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **License:** MIT License

---

## Data License

The Fashion MNIST dataset is made available under the **MIT License**. See the [original repository](https://github.com/zalandoresearch/fashion-mnist) for details.

---

## Tips

- **Raw data storage** - NPZ files store raw pixel values; normalization is done during training
- **Don't commit NPZ files to Git** - They're already in `.gitignore`
- **Use releases for distribution** - Keeps repository size manageable
- **Verify data integrity** - Always check file sizes and shapes after download
- **Normalize when needed** - If using data outside `train.py`, normalize manually
- **Back up your data** - Keep a local copy of processed datasets
- **Version your data** - If you modify preprocessing, create new releases (v1.1, v2.0, etc.)

---

## Data Processing Pipeline

```
Raw Images (PNG)
    ↓ [convert_data.py]
Resized & Converted (NPZ) - Raw pixel values [0-255]
    ↓ [split_data.py]
Train/Test Split (NPZ) - Still raw pixel values
    ↓ [train.py - correct_data()]
Normalized [0-1] + Balanced + Weighted
    ↓ [train.py - training]
Trained Model
```

**Key Point:** Normalization happens at the **training stage**, not the **data preparation stage**.

---

## Need Help?

- Check the main [README.md](../README.md) for project documentation
- Open an [issue](https://github.com/julia-shtal/cnn-fashion-mnist/issues) if you encounter problems
- See [data preprocessing scripts](../src/) for generation details

---

**Happy Training! 🚀**
