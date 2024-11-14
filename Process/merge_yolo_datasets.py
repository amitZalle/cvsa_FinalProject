import os
import shutil
from glob import glob
from pathlib import Path
import argparse

def merge_yolo_datasets(dataset_paths, output_dir):
    """
    Merges multiple YOLO datasets into one, preserving the train and validation split,
    and handles duplicate filenames by renaming with a dataset prefix.
    
    Args:
    - dataset_paths: List of paths to YOLO-format datasets with train/validation subdirectories.
    - output_dir: Output directory for the merged dataset.
    """
    # Define subdirectories in the merged output for train and validation
    train_images_dir = Path(output_dir) / 'train' / 'images'
    train_labels_dir = Path(output_dir) / 'train' / 'labels'
    val_images_dir = Path(output_dir) / 'validation' / 'images'
    val_labels_dir = Path(output_dir) / 'validation' / 'labels'
    
    # Create all necessary directories
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Helper function to copy and rename files from source to destination
    def copy_files(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, dataset_prefix):
        for img_path in glob(str(src_img_dir / '*.jpg')) + glob(str(src_img_dir / '*.png')):
            img_filename = f"{dataset_prefix}_{Path(img_path).name}"
            shutil.copy(img_path, dst_img_dir / img_filename)
        
        for lbl_path in glob(str(src_lbl_dir / '*.txt')):
            lbl_filename = f"{dataset_prefix}_{Path(lbl_path).name}"
            shutil.copy(lbl_path, dst_lbl_dir / lbl_filename)

    # Process each dataset path
    for i, dataset_path in enumerate(dataset_paths):
        dataset_prefix = f"dataset{i+1}"  # Unique prefix for each dataset
        
        dataset_train_images = Path(dataset_path) / 'train' / 'images'
        dataset_train_labels = Path(dataset_path) / 'train' / 'labels'
        dataset_val_images = Path(dataset_path) / 'validation' / 'images'
        dataset_val_labels = Path(dataset_path) / 'validation' / 'labels'

        # Copy training files with unique prefixes
        copy_files(dataset_train_images, dataset_train_labels, train_images_dir, train_labels_dir, dataset_prefix)
        
        # Copy validation files with unique prefixes
        copy_files(dataset_val_images, dataset_val_labels, val_images_dir, val_labels_dir, dataset_prefix)

    print(f"Dataset merged successfully into '{output_dir}'")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets with train/validation split into one.")
    parser.add_argument('output_dir', type=str, help="Output directory for the merged dataset")
    parser.add_argument('dataset_paths', type=str, nargs='+', help="Paths to YOLO datasets to merge")
    args = parser.parse_args()

    merge_yolo_datasets(args.dataset_paths, args.output_dir)
