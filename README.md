# Fashion MNIST CNN Classifier with Data Augmentation Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project implementing **LeNet-5 CNN architecture** for Fashion MNIST image classification, with comprehensive experiments analyzing the impact of class balancing and data preprocessing on model performance.

## 🎯 Project Overview

This project explores how **data preprocessing and class balancing** affect CNN performance in image classification tasks. Using the Fashion MNIST dataset (28×28 grayscale images of clothing items), we train and evaluate LeNet-5 models under four different scenarios to understand:

- Impact of class imbalance on model accuracy
- Effectiveness of oversampling techniques
- Generalization capabilities across different data distributions

### Key Features

- ✨ **LeNet-5 Implementation**: Classic CNN architecture optimized for grayscale image classification
- 📊 **Data Balancing**: Automated oversampling to handle class imbalance
- 🔬 **Multi-Scenario Training**: 4 experimental setups comparing corrected vs uncorrected data
- 📈 **Comprehensive Evaluation**: Confusion matrices, error rates, and performance metrics
- 🛠️ **Modular Codebase**: Clean, refactored Python scripts with clear separation of concerns

---

## 🏗️ Project Structure

```
cnn-fashion-mnist/
├── src/
│   ├── train.py              # Main training script (4 scenarios)
│   ├── convert_data.py       # Image → NPZ converter with preprocessing
│   ├── split_data.py         # Train/test dataset splitter
│   └── __init__.py
├── data/
│   ├── README.md             # Dataset download instructions
│   ├── train_data.npz        # Training set (not tracked, see releases)
│   └── test_data.npz         # Test set (not tracked, see releases)
├── models/                   # Saved model checkpoints (gitignored)
├── results/                  # Confusion matrices and plots (gitignored)
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 Model Architecture

**LeNet-5 Convolutional Neural Network**

```
Input (1×28×28)
    ↓
[Conv2D: 1→6, 5×5, padding=2] + ReLU + MaxPool(2×2)
    ↓
[Conv2D: 6→16, 5×5] + ReLU + MaxPool(2×2)
    ↓
Flatten (16×5×5 = 400)
    ↓
[FC: 400→120] + ReLU
    ↓
[FC: 120→84] + ReLU
    ↓
[FC: 84→10] (Output)
```

**Parameters:**
- Total layers: 7 (2 conv + 3 fc)
- Activation: ReLU
- Pooling: Max pooling (2×2)
- Output: 10 classes (Fashion MNIST categories)

---

## 🔬 Experimental Scenarios

The project evaluates 4 training-testing combinations:

| Scenario | Training Data | Testing Data | Purpose |
|----------|--------------|-------------|---------|
| **1** | Uncorrected | Clean (external) | Baseline performance on unseen data |
| **2** | Uncorrected | Uncorrected (split) | Performance with class imbalance |
| **3** | Corrected | Clean (external) | Impact of balancing on generalization |
| **4** | Corrected | Corrected (split) | Best-case scenario with balanced data |

**Data Corrections Applied:**
1. **Normalization**: Pixel values scaled to [0, 1] range
2. **Class Balancing**: Oversampling minority classes to match majority class size
3. **Weighted Loss**: Class weights inversely proportional to frequencies

---

## 📦 Dataset

### Fashion MNIST
- **Size**: ~40,000 training + 10,000 test images
- **Format**: 28×28 grayscale PNG images
- **Classes**: 10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **File Format**: Compressed NumPy arrays (`.npz`)

### Download Instructions

**Option 1: From GitHub Releases (Recommended)**
```bash
# Download preprocessed datasets
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/train_data.npz
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/test_data.npz

# Move to data folder
mv train_data.npz test_data.npz data/
```

**Option 2: Generate from Raw Images**
```bash
# If you have raw Fashion MNIST images
python src/convert_data.py path/to/images 28 28 1 data/all_data.npz
python src/split_data.py data/all_data.npz --ratio 0.8
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/julia-shtal/cnn-fashion-mnist.git
cd cnn-fashion-mnist
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# See data/README.md for detailed instructions
# Quick download:
cd data
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/train_data.npz
wget https://github.com/julia-shtal/cnn-fashion-mnist/releases/download/v1.0/test_data.npz
cd ..
```

---

## 💻 Usage

### Basic Training

Run all 4 experimental scenarios:

```bash
python src/train.py data/train_data.npz data/test_data.npz
```

**Output:**
- Training progress logs (every 2 epochs)
- Test error rates for each scenario
- Confusion matrices (matplotlib plots)
- Summary comparison table

### Advanced Usage

#### 1. Convert Custom Images to NPZ

```bash
python src/convert_data.py <image_folder> <height> <width> <channels> <output_file>

# Example: Convert 28×28 grayscale images
python src/convert_data.py data/raw_images 28 28 1 data/custom_data.npz
```

**Requirements:**
- Images must be PNG format
- Filenames: `<label>-<id>.png` (e.g., `0-1.png`, `5-42.png`)
- Labels extracted from filename prefix

#### 2. Split Dataset

```bash
python src/split_data.py <input_npz> [--train <output>] [--test <output>] [--ratio <float>]

# Examples:
python src/split_data.py data/all_data.npz
python src/split_data.py data/all_data.npz --ratio 0.9  # 90/10 split
python src/split_data.py data/all_data.npz --train data/my_train.npz --test data/my_test.npz
```

---

## 📊 Results

### Expected Performance

| Scenario | Description | Expected Error Rate |
|----------|-------------|---------------------|
| 1 | Baseline (Uncorrected→Clean) | ~8-12% |
| 2 | Imbalanced (Uncorrected→Uncorrected) | ~6-10% |
| 3 | Balanced (Corrected→Clean) | ~7-11% |
| 4 | Best Case (Corrected→Corrected) | ~5-8% |

*Note: Actual results depend on random initialization and data splits*

### Sample Output

```
Training data shape: (48000, 28, 28, 1), Labels shape: (48000,)
Unique labels: [0 1 2 3 4 5 6 7 8 9]

Scenario 1: Training on uncorrected data, testing on clean data
Epoch 2/10 completed
Epoch 4/10 completed
...
Test error: 0.0987

[Confusion Matrix Plot]

Summary of Results:
Scenario 1 (Uncorrected→Clean): 0.0987
Scenario 2 (Uncorrected→Uncorrected): 0.0823
Scenario 3 (Corrected→Clean): 0.0891
Scenario 4 (Corrected→Corrected): 0.0645
```

---

## 🔧 Configuration

### Hyperparameters

Edit in `src/train.py`:

```python
# Training parameters
batch_size = 32           # Batch size for training
num_epochs = 10           # Number of training epochs
learning_rate = 0.001     # Adam optimizer learning rate
test_percentage = 20      # % of data for internal validation

# Model parameters
num_classes = 10          # Number of output classes
```

### Data Preprocessing

Edit in `src/convert_data.py`:

```python
# Default Fashion MNIST configuration
DEFAULT_HEIGHT = 28
DEFAULT_WIDTH = 28
DEFAULT_CHANNELS = 1

# Image resampling method: LANCZOS (high quality)
img.resize((width, height), Image.Resampling.LANCZOS)
```

---

## 🛠️ Development

### Code Quality

The codebase follows clean code principles:
- **Modular functions**: Single responsibility principle
- **Type hints**: Clear function signatures
- **Docstrings**: Comprehensive documentation
- **Constants**: Named constants instead of magic numbers
- **Error handling**: Graceful failure with informative messages

### Running Tests

```bash
# Verify data conversion
python src/convert_data.py test_images/ 28 28 1 test.npz
python -c "import numpy as np; d=np.load('test.npz'); print(d['X'].shape, d['T'].shape)"

# Verify splitting
python src/split_data.py test.npz
ls -lh train_data.npz test_data.npz
```

---

## 📚 Technical Details

### Technologies Used

- **Deep Learning Framework**: PyTorch 2.0+
- **Numerical Computing**: NumPy
- **Image Processing**: Pillow (PIL)
- **Visualization**: Matplotlib
- **Data Loading**: PyTorch DataLoader with custom Dataset class

### Key Implementation Features

1. **Custom Dataset Class**: `ImageDataset` handles tensor conversion and dimension reordering (NHWC → NCHW)
2. **Data Balancing**: Oversampling implementation in `balance_dataset()`
3. **Modular Training**: `run_training_scenario()` eliminates code duplication
4. **Efficient DataLoaders**: Multi-threaded data loading with automatic batching
5. **GPU Acceleration**: Automatic CUDA detection and utilization

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add data augmentation (rotation, flip, zoom)
- [ ] Implement other architectures (ResNet, VGG)
- [ ] Add TensorBoard logging
- [ ] Hyperparameter tuning with Optuna
- [ ] Export models to ONNX format
- [ ] Add unit tests with pytest
- [ ] Implement early stopping

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Julia Shtal**
- GitHub: [@julia-shtal](https://github.com/julia-shtal)
- LinkedIn: [https://www.linkedin.com/in/iuliia-shtal]

---

## 🙏 Acknowledgments

- **Fashion MNIST Dataset**: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- **LeNet Architecture**: Yann LeCun et al. (1998)
- **PyTorch**: Facebook AI Research
- Inspired by classical computer vision research and modern deep learning practices

---

## 📖 References

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition."
2. Xiao, H., Rasul, K., & Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" (Chapter 9: Convolutional Networks)

---

## 📧 Contact

For questions or suggestions, please open an issue or contact via email.

---

**⭐ Star this repo if you find it helpful!**
