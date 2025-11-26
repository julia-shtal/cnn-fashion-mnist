import numpy as np

# Configuration
INPUT_NPZ = 'all_data.npz'
TRAIN_OUTPUT = 'train_data.npz'
TEST_OUTPUT = 'test_data.npz'
TRAIN_RATIO = 0.8

def split_dataset(input_file, train_output, test_output, train_ratio=0.8):
    """
    Split a single NPZ file into train and test sets.
    
    Args:
        input_file: Path to input NPZ file
        train_output: Path to save training NPZ
        test_output: Path to save test NPZ
        train_ratio: Fraction for training (0.8 = 80% train, 20% test)
    """
    data = np.load(input_file)
    X, T = data['X'], data['T']
    
    # Shuffle and split
    num_total = len(X)
    idxs = np.arange(num_total)
    np.random.shuffle(idxs)
    split = int(train_ratio * num_total)
    train_idx, test_idx = idxs[:split], idxs[split:]
    
    # Save split files
    np.savez_compressed(train_output, X=X[train_idx], T=T[train_idx])
    np.savez_compressed(test_output, X=X[test_idx], T=T[test_idx])
    
    print(f"Loaded: {input_file}")
    print(f"Train: {len(train_idx)} samples → {train_output}")
    print(f"Test:  {len(test_idx)} samples → {test_output}")

if __name__ == "__main__":
    split_dataset(INPUT_NPZ, TRAIN_OUTPUT, TEST_OUTPUT, TRAIN_RATIO)
