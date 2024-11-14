from ultralytics import YOLO
import cv2
import os

# List of paths to your trained models
model_paths = [
   # "/home/student/FinalProject/segmentation_models/Needle/weights/best.pt",
   # "/home/student/FinalProject/segmentation_models/Tweezers/weights/best.pt",
   # "/home/student/FinalProject/segmentation_models/Both/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/All_prob/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/pseudo_trained_model11/weights/best.pt",
   "/home/student/FinalProject/segmentation_models/All/weights/best.pt"
]

# Path to the input image and output location
image_path = "pseudo_data/HCframes_1/yolo_data/train/images/frame_1_5353.jpg"  # Replace with your image path
output_dir = "predictions"  # Directory to save the output images
os.makedirs(output_dir, exist_ok=True)

# Process the single image with each model separately
for model_path in model_paths:
    # Load the current YOLOv8 segmentation model
    model = YOLO(model_path)
    
    # Generate a unique output filename based on the model name
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    output_image_path = os.path.join(output_dir, f"output_{model_name}.jpg")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        continue

    # Run YOLO segmentation model on the image
    results = model.predict(image, task="segment")  # Set task to "segment" for segmentation

    # Draw the segmented masks on the image
    annotated_image = results[0].plot()  # Automatically draws masks and annotations on the image

    # Save the processed image
    cv2.imwrite(output_image_path, annotated_image)

    print(f"Processed image with model {model_name}, saved to {output_image_path}")

print("All models have processed the image.")
