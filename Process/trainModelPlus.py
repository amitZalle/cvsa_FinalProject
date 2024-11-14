from ultralytics import YOLO
import os

# List of YAML files for each COCO dataset
dataset_configs = [
    # "HdriNeedle.yaml",
    # "HdriTweezers.yaml",
    #"HdriBoth.yaml",
    # "CocoNeedle.yaml",
    # "CocoTweezers.yaml",
    # "CocoBoth.yaml",
    #"yamls/Needle.yaml",
    #"yamls/Tweezers.yaml",
    #"yamls/Both.yaml",
    "yamls/All.yaml",
]

# Path to the initial YOLO segmentation model weights
initial_model_path = "yolov8l-seg.pt"  # Always start with the base model
output_base_dir = "segmentation_models"  # Base directory for saved models
os.makedirs(output_base_dir, exist_ok=True)

# Training loop for each dataset
for config in dataset_configs:
    # Use the name of the config file as the folder name for each model (e.g., "HdriNeedle")
    model_name = os.path.splitext(os.path.basename(config))[0] + '_l'

    # Load the model with the initial weights for each training
    model = YOLO(initial_model_path)

    # Train the model and save the results
    model.train(
        data=config,
        epochs=10,
        batch=16,
        task="segment",
        project=output_base_dir,
        name=model_name
    )

print("Training completed. Each model is saved individually in the 'segmentation_models' directory.")
