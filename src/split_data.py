import numpy as np
import argparse


def split_dataset(input_file, train_output, test_output, train_ratio=0.8):
    """
    Split a single NPZ file into train and test sets.

    Args:
        input_file: Path to input NPZ file
        train_output: Path to save training NPZ
        test_output: Path to save test NPZ
        train_ratio: Fraction for training (default 0.8 = 80% train, 20% test)
    """
    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    X, T = data['X'], data['T']

    print(f"Total samples: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Labels shape: {T.shape}")

    # Shuffle and split
    num_total = len(X)
    idxs = np.arange(num_total)
    np.random.shuffle(idxs)
    split = int(train_ratio * num_total)
    train_idx, test_idx = idxs[:split], idxs[split:]

    # Save split files
    print(f"\nSplitting with ratio: {train_ratio:.1%} train, {1 - train_ratio:.1%} test")
    np.savez_compressed(train_output, X=X[train_idx], T=T[train_idx])
    np.savez_compressed(test_output, X=X[test_idx], T=T[test_idx])

    print(f"\nResults:")
    print(f"  Training set: {len(train_idx)} samples -> {train_output}")
    print(f"  Test set:     {len(test_idx)} samples -> {test_output}")
    print(f"\nSplit completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split NPZ dataset into train/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split (80/20)
  python split_data.py data/all_data.npz

  # Custom ratio (90/10 split)
  python split_data.py data/all_data.npz --ratio 0.9

  # Custom output filenames
  python split_data.py data/all_data.npz --train data/my_train.npz --test data/my_test.npz

  # Both custom ratio and names
  python split_data.py data/all_data.npz --ratio 0.7 --train custom_train.npz --test custom_test.npz
        """
    )

    parser.add_argument('input_file', type=str,
                        help='Path to input NPZ file to split')
    parser.add_argument('--train', type=str, default='train_data.npz',
                        help='Output training NPZ file (default: train_data.npz)')
    parser.add_argument('--test', type=str, default='test_data.npz',
                        help='Output test NPZ file (default: test_data.npz)')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Training data ratio between 0 and 1 (default: 0.8 = 80%% train, 20%% test)')

    args = parser.parse_args()

    # Validate ratio
    if not 0 < args.ratio < 1:
        parser.error("Ratio must be between 0 and 1")

    # Run the split
    split_dataset(args.input_file, args.train, args.test, args.ratio)
