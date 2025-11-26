import sys
import os
import numpy as np
from PIL import Image
import argparse


# Configuration constants
DEFAULT_HEIGHT = 28
DEFAULT_WIDTH = 28
DEFAULT_CHANNELS = 1
PROGRESS_INTERVAL = 100
IMAGE_EXTENSION = '.png'
LABEL_SEPARATOR = '-'


def parse_command_line_arguments():
    """
    Parse command line arguments for image folder, dimensions, and output file.

    Returns:
        argparse.Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Convert image data to NPZ format')
    parser.add_argument('image_folder', type=str, help='Path to folder with images')
    parser.add_argument('h', type=int, help='Height of output images')
    parser.add_argument('w', type=int, help='Width of output images')
    parser.add_argument('c', type=int, help='Number of channels')
    parser.add_argument('npz_output_file', type=str, help='Output NPZ file path')
    return parser.parse_args()


def extract_label_from_filename(filename):
    """
    Extract label from filename format like '0-1.png'.
    Label is the first part before the hyphen.

    Args:
        filename: Image filename

    Returns:
        Integer label or None if extraction fails
    """
    try:
        label = int(filename.split(LABEL_SEPARATOR)[0])
        return label
    except (ValueError, IndexError):
        print(f"Warning: Could not extract label from filename {filename}")
        return None


def get_image_files(folder_path):
    """
    Retrieve all PNG files from the specified folder.

    Args:
        folder_path: Path to the folder containing images

    Returns:
        Sorted list of PNG filenames
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(IMAGE_EXTENSION)]
    return sorted(image_files)


def resize_and_convert_image(img, height, width, channels):
    """
    Resize image and convert to appropriate format and channels.

    Args:
        img: PIL Image object
        height: Target height
        width: Target width
        channels: Number of channels (1 for grayscale, 3 for RGB)

    Returns:
        numpy array of processed image
    """
    # Convert to grayscale if single channel required
    if channels == 1:
        img = img.convert('L')

    # Resize using Lanczos resampling for quality preservation
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Reshape to include channel dimension if grayscale
    if channels == 1:
        img_array = img_array.reshape(height, width, 1)

    return img_array


def process_single_image(img_path, height, width, channels):
    """
    Load and preprocess a single image file.

    Args:
        img_path: Path to image file
        height: Target height
        width: Target width
        channels: Number of channels

    Returns:
        Processed image array or None if processing fails
    """
    try:
        with Image.open(img_path) as img:
            img_array = resize_and_convert_image(img, height, width, channels)
            return img_array
    except Exception as e:
        print(f"Error processing image {os.path.basename(img_path)}: {str(e)}")
        return None


def load_and_preprocess_images(folder_path, height, width, channels):
    """
    Load all images from folder and preprocess them into arrays.

    Args:
        folder_path: Path to folder with images
        height: Target image height
        width: Target image width
        channels: Number of channels per image

    Returns:
        Tuple of (X, T) arrays where X contains images and T contains labels
    """
    image_files = get_image_files(folder_path)

    # Initialize arrays to store all images and labels
    X = np.zeros((len(image_files), height, width, channels))
    T = np.zeros(len(image_files), dtype=np.int32)

    valid_images = 0

    for idx, image_file in enumerate(image_files):
        # Extract label from filename
        label = extract_label_from_filename(image_file)
        if label is None:
            continue

        # Process image
        img_path = os.path.join(folder_path, image_file)
        img_array = process_single_image(img_path, height, width, channels)

        if img_array is not None:
            X[valid_images] = img_array
            T[valid_images] = label
            valid_images += 1

        # Progress logging
        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")

    # Trim arrays to remove unused space
    X = X[:valid_images]
    T = T[:valid_images]

    return X, T


def validate_arguments(args):
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        True if valid, False otherwise
    """
    # Warn if non-standard dimensions
    if args.h != DEFAULT_HEIGHT or args.w != DEFAULT_WIDTH or args.c != DEFAULT_CHANNELS:
        print(f"Warning: Recommended dimensions are h={DEFAULT_HEIGHT}, w={DEFAULT_WIDTH}, c={DEFAULT_CHANNELS}")

    # Check if folder exists
    if not os.path.exists(args.image_folder):
        print(f"Error: Folder {args.image_folder} does not exist")
        return False

    return True


def print_dataset_statistics(X, T):
    """
    Print statistics about the loaded dataset.

    Args:
        X: Images array
        T: Labels array
    """
    print(f"\nDataset statistics:")
    print(f"Total images processed: {len(T)}")
    print(f"Unique labels: {np.unique(T)}")
    print(f"X shape: {X.shape}")
    print(f"T shape: {T.shape}")


def main():
    # Parse and validate arguments
    args = parse_command_line_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    # Load and preprocess images
    print("Loading and preprocessing images...")
    X, T = load_and_preprocess_images(args.image_folder, args.h, args.w, args.c)

    # Print statistics
    print_dataset_statistics(X, T)

    # Save to NPZ file
    print(f"\nSaving processed data to {args.npz_output_file}")
    np.savez_compressed(args.npz_output_file, X=X, T=T)
    print("Complete!")


if __name__ == "__main__":
    main()
