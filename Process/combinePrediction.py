import os
import cv2
import torch
from ultralytics import YOLO

# Paths
videos_folder = '/path/to/videos'  # Directory containing the videos
models = [
    YOLO('/path/to/model1.pt'), 
    YOLO('/path/to/model2.pt'), 
    YOLO('/path/to/model3.pt')
]  # List of YOLO models

# Function to normalize bounding boxes for YOLO format
def normalize_bbox(x1, y1, x2, y2, image_width, image_height):
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height

# Loop through each video in the videos folder
video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
for video_file in video_files:
    input_video_path = os.path.join(videos_folder, video_file)
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        combined_predictions = {}  # Dictionary to store the highest-confidence predictions per class

        # Run each model on the current frame
        for model in models:
            results = model(frame)

            # Process results to find the highest confidence boxes for each label
            for result in results:
                boxes = result.boxes.xyxy  # Get bounding boxes in xyxy format
                confidences = result.boxes.conf  # Get confidence scores
                class_ids = result.boxes.cls  # Get class indices
                segmentations = result.masks  # Get segmentation masks if available

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_id = int(class_id)
                    if class_id not in combined_predictions or conf > combined_predictions[class_id]['conf']:
                        # Update the highest confidence box and segmentation for the class
                        combined_predictions[class_id] = {
                            'conf': conf.item(),
                            'box': box.tolist(),
                            'segmentation': segmentations if segmentations else None
                        }

        # Save or process the combined predictions for this frame
        frame_filename = f"combined_frame_{frame_count}.jpg"
        cv2.imwrite(os.path.join('/path/to/save/combined_frames', frame_filename), frame)

        # Save labels to file or print for inspection
        label_filename = f"combined_frame_{frame_count}.txt"
        with open(os.path.join('/path/to/save/combined_labels', label_filename), 'w') as label_file:
            for class_id, data in combined_predictions.items():
                x1, y1, x2, y2 = data['box'][:4]
                image_height, image_width = frame.shape[:2]

                # Normalize the bounding box coordinates
                x_center, y_center, width, height = normalize_bbox(x1, y1, x2, y2, image_width, image_height)
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height} {data['conf']}\n")

        frame_count += 1

    # Release the video capture object
    cap.release()

print("Combined predictions for all videos have been processed and saved.")
