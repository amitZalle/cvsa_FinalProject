import os
from PIL import Image

def convert_images_to_rgb(directories):
    """Convert all images in the specified directories to RGB format."""
    for directory in directories:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    # Check for common image file extensions
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                        file_path = os.path.join(root, file)
                        try:
                            # Open the image
                            img = Image.open(file_path)
                            
                            # Convert to RGB
                            rgb_img = img.convert('RGB')

                            # Save back to the original path or to a new location
                            rgb_img.save(file_path)
                            #print(f"Converted {file_path} to RGB.")
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
            print(f"Converted {directory} to RGB.")
        else:
            print(f"Directory does not exist: {directory}")

# List of directories to process
directories_to_convert = [
    "synthetic_data/coco/tweezers/yolo_data/train/images",
    "synthetic_data/coco/tweezers/yolo_data/validation/images",
    "synthetic_data/coco/needle_holder/yolo_data/train/images",
    "synthetic_data/coco/needle_holder/yolo_data/validation/images",
    "synthetic_data/coco/both/yolo_data/train/images",
    "synthetic_data/coco/both/yolo_data/validation/images",

]

convert_images_to_rgb(directories_to_convert)
