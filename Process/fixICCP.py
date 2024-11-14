from PIL import Image
import os

def fix_iccp_warning(image_path):
    """Fixes the ICCP warning by re-saving the image without the ICC profile."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.save(image_path, format="PNG", icc_profile=None)

def process_images_in_directories(directories):
    """
    Applies the ICCP fix to all images in the specified directories.
    
    Args:
        directories (list): List of directory paths to process images in.
    """
    for directory in directories:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    fix_iccp_warning(image_path)
                    print(f"Processed: {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

# List of directories containing images to process
directories = [
#    "synthetic_data/coco/tweezers/yolo_data/train/images",
#    "synthetic_data/coco/tweezers/yolo_data/validation/images",
#    "synthetic_data/coco/needle_holder/yolo_data/train/images",
#    "synthetic_data/coco/needle_holder/yolo_data/validation/images",
#    "synthetic_data/coco/both/yolo_data/train/images",
#    "synthetic_data/coco/both/yolo_data/validation/images",
    "synthetic_data/needle/train/images",
    "synthetic_data/needle/validation/images",
    "synthetic_data/tweezers/train/images",
    "synthetic_data/tweezers/validation/images",
    "synthetic_data/both/train/images",
    "synthetic_data/both/validation/images",
    "synthetic_data/all/train/images",
    "synthetic_data/all/validation/images",

]

# Run the processing function
process_images_in_directories(directories)
