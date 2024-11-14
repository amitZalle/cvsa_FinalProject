import cv2
import os
import shutil
from ultralytics import YOLO
import torch
import numpy as np

# Paths
source_folder = '/datashare/project/vids_tune'
frames_folder = 'pseudo_data/HCframes_1/frames'
labels_folder = 'pseudo_data/HCframes_1/labels'
bbox_folder = 'pseudo_data/HCframes_1/bbox'
dataset_folder = 'pseudo_data/HCframes_1/yolo_data'
train_images_folder = os.path.join(dataset_folder, 'train/images')
train_labels_folder = os.path.join(dataset_folder, 'train/labels')
val_images_folder = os.path.join(dataset_folder, 'val/images')
val_labels_folder = os.path.join(dataset_folder, 'val/labels')

# Function to delete contents of a directory
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Load the trained YOLOv8 model
model = YOLO('segmentation_models/All/weights/best.pt')  # path to the saved model

# Define source folder and output folders for high-confidence frames and labels
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(bbox_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)
#os.makedirs(f'pseudo_data/HCframes_1/segs', exist_ok=True)

# Clear existing data in the relevant directories
clear_directory(frames_folder)
clear_directory(labels_folder)
clear_directory(bbox_folder)
clear_directory(dataset_folder)

# Function to normalize bounding boxes
def normalize_bbox(x1, y1, x2, y2, image_width, image_height):
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height

# normalize masks
def normalize_coords(x, y, image_width, image_height):
    """
    Normalize the given (x, y) coordinates to the range [0, 1] based on image width and height.
    
    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        
    Returns:
        (float, float): The normalized x and y coordinates in the range [0, 1].
    """
    norm_x = x / image_width
    norm_y = y / image_height
    return norm_x, norm_y


# Get a list of all video files in the source folder
# TODO: not end with small
video_files = [
    f for f in os.listdir(source_folder)
    if f.endswith(('.mp4', '.avi', '.mov')) and not f.endswith("small.mp4") and not f.endswith("small.avi") and not f.endswith("small.mov")
]
#video_files = ["4_2_24_B_2_small.mp4"]

# Process each video file
for j, video_file in enumerate(video_files):
    print("starting video: ", j)
    input_video_path = os.path.join(source_folder, video_file)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    frame_count = 0

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model on the current frame
        results = model(frame)

        # Frame dimensions
        frame_height, frame_width = frame.shape[:2]
        frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

        # Threshold for proximity to center (20% of frame dimensions)
        center_threshold_x = frame_width * 0.2
        center_threshold_y = frame_height * 0.2

        # Dictionary to keep track of boxes with confidence > 0.5 for needle_driver and tweezers
        high_conf_boxes = {}

        # Extract bounding boxes, labels, and confidences
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes in xyxy format
            confidences = result.boxes.conf  # Get confidence scores
            class_ids = result.boxes.cls  # Get class indices
            class_names = result.names  # Get class names
         

            if result.masks:
                masks = result.masks.data  # Get segmentation masks

                for box, conf, class_id, mask in zip(boxes, confidences, class_ids, masks):
                    #print(torch.count_nonzero(mask.data))
                    class_id = int(class_id)
                    x1, y1, x2, y2 = map(float, box[:4])  # Get the coordinates of the box
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    print(f"{frame_center_x},{frame_center_y}")
                    print(f"{center_threshold_x},{center_threshold_y}")
                    # Ensure that we only process needle_driver and tweezers with conf > 0.5, and proximity to frame center
                    if ((class_names[class_id] == 'Needle_driver' and conf > 0.6) or (class_names[class_id] == 'Tweezers' and conf>0.25)) and \
                       abs(box_center_x - frame_center_x) <= center_threshold_x and \
                       abs(box_center_y - frame_center_y) <= center_threshold_y:
                        
                        print(f"in: center:{box_center_x}, {box_center_y}. conf:{conf}")
                        if class_id not in high_conf_boxes or conf > high_conf_boxes[class_id]['conf']:
                            high_conf_boxes[class_id] = {'conf': conf, 'box': box, 'size': max(width, height), 'mask': mask}

                    else:
                        print(f"out: center:{box_center_x}, {box_center_y}. conf:{conf}")
        if high_conf_boxes:
            # Save the frame with high confidence detections
            frame_filename = f"frame_{j}_{frame_count}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Save the label file for this frame with normalized coordinates
            label_filename = f"frame_{j}_{frame_count}.txt"
            label_path = os.path.join(labels_folder, label_filename)
            bbox_path = os.path.join(bbox_folder, label_filename)

            with open(label_path, 'w') as label_file:
                for class_id, data in high_conf_boxes.items():
                    mask = data['mask']  # Get the mask

                    # Convert mask to binary and find contours
                    mask_bin = (mask > 0).float().cpu().numpy()  # Ensure it's a binary mask (0 or 1)

                    # Convert to uint8 (8-bit single-channel)
                    mask_bin = (mask_bin * 255).astype(np.uint8)


                    #print(frame,np.unique(mask_bin))  # Check the unique values in the mask
                    #cv2.imwrite(f'pseudo_data/HCframes_1/segs/frame_{j}_{frame_count}_{class_id}.png', mask_bin)

                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
                    #print("contours", contours)
                    for contour in contours:
                        contour = contour.flatten()
                        #print("contour",contour)
                        if len(contour) >= 6:  # Only save contours that have enough points (3 or more)
                        # Normalize the polygon coordinates and write them in YOLO format
                            normalized_contour = []
                            for i in range(0, len(contour), 2):
                                norm_x, norm_y = normalize_coords(contour[i], contour[i + 1], frame_width, frame_height)
                                normalized_contour.append(f"{norm_x} {norm_y}")
                                #normalized_contour.append(f"{contour[i]} {contour[i+1]}")
        
                            # Write class index and normalized polygon coordinates
                            label_file.write(f"{class_id} {' '.join(normalized_contour)}\n")

#            with open(bbox_path , 'w') as label_file:
#                for class_id, data in high_conf_boxes.items():
#                    box = data['box']
#                    x1, y1, x2, y2 = map(float, box[:4])  # Get the coordinates of the box
#                    image_height, image_width = frame.shape[:2]  # Get image dimensions
#
#                    # Normalize the bounding box coordinates
#                    x_center, y_center, width, height = normalize_bbox(x1, y1, x2, y2, image_width, image_height)
#                    label = class_names[class_id]  # Get the class label
#
#                    label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        frame_count += 1

    # Release resources for the current video
    cap.release()

print("Frames with high-confidence detections for 'needle_driver' and 'tweezers' have been saved and normalized.")

# Define paths for dataset organization
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Move high-confidence frames and labels to the dataset folders
for file_name in os.listdir(frames_folder):
    if file_name.endswith('.jpg'):
        # Move frame
        src_frame = os.path.join(frames_folder, file_name)
        dest_frame = os.path.join(train_images_folder, file_name)
        shutil.move(src_frame, dest_frame)
        
        # Move corresponding label
        label_file_name = file_name.replace('.jpg', '.txt')
        src_label = os.path.join(labels_folder, label_file_name)
        dest_label = os.path.join(train_labels_folder, label_file_name)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dest_label)
        else:
            print(f"Label file {label_file_name} not found for {file_name}")

# List all image files in the training images folder
image_files = [f for f in os.listdir(train_images_folder) if f.endswith('.jpg')]

print(f"Dataset prepared successfully with {len(image_files)} frames.")

# Ensure validation directories exist
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Copy validation data from synthetic source
shutil.copytree('synthetic_data/all/validation/images', val_images_folder, dirs_exist_ok=True)
shutil.copytree('synthetic_data/all/validation/labels', val_labels_folder, dirs_exist_ok=True)

# add original dataset to pseudo
shutil.copytree('synthetic_data/all/train/images', 'pseudo_data/HCframes_1/yolo_data/train/images', dirs_exist_ok=True)
shutil.copytree('synthetic_data/all/train/labels', 'pseudo_data/HCframes_1/yolo_data/train/labels', dirs_exist_ok=True)
