import os
import json

# List of paths to JSON annotation files
annotation_files = [
    "/home/student/FinalProject/synthetic_data/hdri/needle_holder/coco_data/coco_annotations.json",
    "/home/student/FinalProject/synthetic_data/hdri/tweezers/coco_data/coco_annotations.json",
    "/home/student/FinalProject/synthetic_data/hdri/both/coco_data/coco_annotations.json",
    "/home/student/FinalProject/synthetic_data/coco/needle_holder/coco_data/coco_annotations.json",
    "/home/student/FinalProject/synthetic_data/coco/tweezers/coco_data/coco_annotations.json",
    "/home/student/FinalProject/synthetic_data/coco/both/coco_data/coco_annotations.json",
 
]

# Loop through each annotation file
for annotation_file in annotation_files:
    # Load the JSON data
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Remove 'images/' prefix from file_name in each image entry
    for image in annotations['images']:
        image['file_name'] = image['file_name'].replace("images/", "")

    # Define the path for the new 'annotations' directory
    annotations_dir = os.path.join(os.path.dirname(annotation_file), "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Save the updated annotation file in the new folder
    new_annotation_path = os.path.join(annotations_dir, os.path.basename(annotation_file))
    with open(new_annotation_path, 'w') as f:
        json.dump(annotations, f)

    print(f"Updated file names and saved to: {new_annotation_path}")
