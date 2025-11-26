import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


class ImageDataset(Dataset):
    """
    Custom Dataset for handling image data and labels.
    Handles tensor conversion, dimension reordering, and one-hot to index conversion.
    """

    def __init__(self, X, T):
        """
        Initialize dataset with images and labels.

        Args:
            X: numpy array of shape (N, height, width, channels)
            T: numpy array of labels (one-hot encoded or class indices)
        """
        # Convert to PyTorch format: (N, channels, height, width)
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)

        # Convert one-hot labels to class indices if needed
        if len(T.shape) > 1:
            T = np.argmax(T, axis=1)
        self.T = torch.LongTensor(T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


class LeNet(nn.Module):
    """
    LeNet-5 CNN architecture for image classification.
    Architecture: Conv1(1->6) -> ReLU -> MaxPool -> Conv2(6->16) -> ReLU -> MaxPool
                  -> FC1(400->120) -> ReLU -> FC2(120->84) -> ReLU -> FC3(84->classes)
    """

    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        # Convolutional feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1->6 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),            # 6->16 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected classification layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def balance_dataset(X, T):
    """
    Balance dataset using oversampling to match the largest class size.

    Args:
        X: numpy array of images
        T: numpy array of labels

    Returns:
        Balanced X and T arrays
    """
    unique_labels, counts = np.unique(T, return_counts=True)
    max_count = np.max(counts)

    balanced_X = []
    balanced_T = []

    for label in unique_labels:
        indices = np.where(T == label)[0]
        n_samples = len(indices)
        n_copies = int(np.ceil(max_count / n_samples))

        # Replicate samples to match max_count
        label_indices = np.tile(indices, n_copies)[:max_count]
        balanced_X.append(X[label_indices])
        balanced_T.append(T[label_indices])

    return np.concatenate(balanced_X), np.concatenate(balanced_T)


def correct_data(X, T):
    """
    Normalize data and balance classes via oversampling.

    Args:
        X: numpy array of images
        T: numpy array of labels

    Returns:
        Normalized and balanced X, T, and class weights tensor
    """
    # Normalize to [0, 1] range
    X = (X - X.min()) / (X.max() - X.min())

    # Balance dataset
    X, T = balance_dataset(X, T)

    # Calculate class weights
    unique_labels, counts = np.unique(T, return_counts=True)
    class_weights = torch.FloatTensor(1.0 / counts)
    class_weights = class_weights / class_weights.sum()

    return X, T, class_weights


def custom_confusion_matrix(true_labels, pred_labels, num_classes):
    """
    Compute confusion matrix from true and predicted labels.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t][p] += 1
    return conf_matrix


def plot_confusion_matrix(conf_matrix, save_path=None, title='Confusion Matrix'):
    """
    Visualize confusion matrix with matplotlib and save to file.

    Args:
        conf_matrix: Confusion matrix to plot
        save_path: Path to save the figure (if None, only displays)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Confusion matrix saved: {save_path}")

    plt.close()  # Close to avoid memory issues with multiple plots


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Train the neural network model.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        device: CPU or GPU device

    Returns:
        Trained model
    """
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Progress logging every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs} completed')

    return model


def evaluate_model(model, test_loader, device, num_classes):
    """
    Evaluate trained model on test data.

    Args:
        model: Trained neural network
        test_loader: DataLoader for test data
        device: CPU or GPU device
        num_classes: Number of classes

    Returns:
        error_rate: Model error rate
        conf_matrix: Confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(predicted)
            all_labels.extend(labels)

            total += labels.shape[0]

            # Handle one-hot encoded labels
            if len(labels.shape) > 1:
                labels = np.argmax(labels, axis=1)

            correct += (predicted == labels).sum()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    error_rate = 1 - (correct / total)
    conf_matrix = custom_confusion_matrix(all_labels, all_preds, num_classes)

    return error_rate, conf_matrix


def create_dataloaders(X_train, T_train, X_train_corrected, T_train_corrected,
                       X_clean_test, T_clean_test, batch_size, test_percentage):
    """
    Create all required DataLoaders for training and testing scenarios.

    Args:
        X_train, T_train: Original training data and labels
        X_train_corrected, T_train_corrected: Corrected training data and labels
        X_clean_test, T_clean_test: Clean test data and labels
        batch_size: Batch size for DataLoaders
        test_percentage: Percentage of data to use for testing

    Returns:
        Dictionary containing all DataLoaders
    """
    # Create datasets
    uncorrected_dataset = ImageDataset(X_train, T_train)
    corrected_dataset = ImageDataset(X_train_corrected, T_train_corrected)
    clean_test_dataset = ImageDataset(X_clean_test, T_clean_test)

    # Calculate split sizes for uncorrected data
    uncorrected_test_size = int(len(uncorrected_dataset) * test_percentage / 100)
    uncorrected_train_size = len(uncorrected_dataset) - uncorrected_test_size

    # Calculate split sizes for corrected data
    corrected_test_size = int(len(corrected_dataset) * test_percentage / 100)
    corrected_train_size = len(corrected_dataset) - corrected_test_size

    # Perform train-test splits
    uncorrected_train_dataset, uncorrected_test_dataset = random_split(
        uncorrected_dataset, [uncorrected_train_size, uncorrected_test_size]
    )

    corrected_train_dataset, corrected_test_dataset = random_split(
        corrected_dataset, [corrected_train_size, corrected_test_size]
    )

    # Create DataLoaders
    loaders = {
        'uncorrected_train': DataLoader(uncorrected_train_dataset, batch_size=batch_size, shuffle=True),
        'corrected_train': DataLoader(corrected_train_dataset, batch_size=batch_size, shuffle=True),
        'clean_test': DataLoader(clean_test_dataset, batch_size=batch_size),
        'uncorrected_test': DataLoader(uncorrected_test_dataset, batch_size=batch_size),
        'corrected_test': DataLoader(corrected_test_dataset, batch_size=batch_size)
    }

    return loaders


def run_training_scenario(scenario_num, description, model, train_loader, test_loader,
                         criterion, optimizer, num_epochs, device, num_classes,
                         results_dir, timestamp):
    """
    Execute a single training and evaluation scenario.

    Args:
        scenario_num: Scenario number for logging
        description: Description of the scenario
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to use
        num_classes: Number of classes
        results_dir: Directory to save results
        timestamp: Timestamp string for file naming

    Returns:
        error_rate: Test error rate
        conf_matrix: Confusion matrix
    """
    print(f"\nScenario {scenario_num}: {description}")
    model = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    error_rate, conf_matrix = evaluate_model(model, test_loader, device, num_classes)
    print(f"  Test error: {error_rate:.4f}")

    # Create meaningful filename for confusion matrix
    # Format: confusion_matrix_scenario1_uncorrected-to-clean_20251126_143025.png
    scenario_name = description.lower()\
        .replace("training on ", "")\
        .replace("testing on ", "to-")\
        .replace("data, ", "")\
        .replace(" ", "-")\
        .replace(",", "")

    filename = f"confusion_matrix_scenario{scenario_num}_{scenario_name}_{timestamp}.png"
    save_path = os.path.join(results_dir, filename)

    plot_title = f'Confusion Matrix - Scenario {scenario_num}\n{description}'
    plot_confusion_matrix(conf_matrix, save_path=save_path, title=plot_title)

    return error_rate, conf_matrix


def save_summary_results(results_dict, results_dir, timestamp):
    """
    Save experiment summary to a text file.

    Args:
        results_dict: Dictionary containing scenario results
        results_dir: Directory to save results
        timestamp: Timestamp string for file naming
    """
    summary_filename = f"experiment_summary_{timestamp}.txt"
    summary_path = os.path.join(results_dir, summary_filename)

    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CNN Fashion MNIST Experiment Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Summary of Results:\n")
        f.write("-" * 70 + "\n")
        for scenario, error_rate in results_dict.items():
            f.write(f"{scenario}: {error_rate:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Scenario Descriptions:\n")
        f.write("=" * 70 + "\n")
        f.write("Scenario 1: Baseline - Train on uncorrected, test on clean data\n")
        f.write("            (Evaluates model on unseen clean distribution)\n\n")
        f.write("Scenario 2: Imbalanced - Train on uncorrected, test on uncorrected\n")
        f.write("            (Performance with class imbalance)\n\n")
        f.write("Scenario 3: Balanced - Train on corrected, test on clean data\n")
        f.write("            (Impact of balancing on generalization)\n\n")
        f.write("Scenario 4: Best Case - Train on corrected, test on corrected\n")
        f.write("            (Optimal scenario with balanced data)\n")

    print(f"\nExperiment summary saved: {summary_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CNN Training and Testing Script')
    parser.add_argument('train_data', type=str, help='Path to training data NPZ file')
    parser.add_argument('clean_test_data', type=str, help='Path to clean test data NPZ file')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    args = parser.parse_args()

    # Create results directory with timestamp subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Load training data
    train_data = np.load(args.train_data)
    X_train, T_train = train_data['X'], train_data['T']

    # Load clean test data
    clean_test_data = np.load(args.clean_test_data)
    X_clean_test, T_clean_test, class_weights_clean_data = correct_data(
        clean_test_data['X'], clean_test_data['T']
    )

    # Correct training data (normalize and balance)
    X_train_corrected, T_train_corrected, class_weights = correct_data(X_train.copy(), T_train)

    # Configuration parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(T_train))
    batch_size = 32
    num_epochs = 10
    test_percentage = 20
    learning_rate = 0.001

    print(f"Training data shape: {X_train.shape}, Labels shape: {T_train.shape}")
    print(f"Unique labels: {np.unique(T_train)}")
    print(f"Device: {device}")

    # Create all required DataLoaders
    loaders = create_dataloaders(X_train, T_train, X_train_corrected, T_train_corrected,
                                 X_clean_test, T_clean_test, batch_size, test_percentage)

    # Dictionary to store all results
    results = {}

    # Scenario 1: Train on uncorrected, test on clean
    model1 = LeNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    error_rate1, conf_matrix1 = run_training_scenario(
        1, "Training on uncorrected data, testing on clean data",
        model1, loaders['uncorrected_train'], loaders['clean_test'],
        criterion, optimizer1, num_epochs, device, num_classes,
        results_dir, timestamp
    )
    results["Scenario 1 (Uncorrected→Clean)"] = error_rate1

    # Scenario 2: Train on uncorrected, test on uncorrected
    model2 = LeNet(num_classes).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    error_rate2, conf_matrix2 = run_training_scenario(
        2, "Training on uncorrected data, testing on uncorrected data",
        model2, loaders['uncorrected_train'], loaders['uncorrected_test'],
        criterion, optimizer2, num_epochs, device, num_classes,
        results_dir, timestamp
    )
    results["Scenario 2 (Uncorrected→Uncorrected)"] = error_rate2

    # Scenario 3: Train on corrected, test on clean
    model3 = LeNet(num_classes).to(device)
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
    error_rate3, conf_matrix3 = run_training_scenario(
        3, "Training on corrected data, testing on clean data",
        model3, loaders['corrected_train'], loaders['clean_test'],
        criterion_weighted, optimizer3, num_epochs, device, num_classes,
        results_dir, timestamp
    )
    results["Scenario 3 (Corrected→Clean)"] = error_rate3

    # Scenario 4: Train on corrected, test on corrected
    model4 = LeNet(num_classes).to(device)
    optimizer4 = optim.Adam(model4.parameters(), lr=learning_rate)
    error_rate4, conf_matrix4 = run_training_scenario(
        4, "Training on corrected data, testing on corrected test data",
        model4, loaders['corrected_train'], loaders['corrected_test'],
        criterion_weighted, optimizer4, num_epochs, device, num_classes,
        results_dir, timestamp
    )
    results["Scenario 4 (Corrected→Corrected)"] = error_rate4

    # Print summary of all results
    print("\n" + "="*70)
    print("Summary of Results:")
    print("="*70)
    for scenario, error_rate in results.items():
        print(f"{scenario}: {error_rate:.4f}")
    print("="*70)

    # Save summary to file
    save_summary_results(results, results_dir, timestamp)

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
