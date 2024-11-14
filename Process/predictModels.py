from ultralytics import YOLO
import cv2
import os

# List of paths to your trained models
model_paths = [
    #"/home/student/FinalProject/segmentation_models/Needle/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/Tweezers/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/Both/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/All_prob/weights/best.pt",
    #"/home/student/FinalProject/segmentation_models/All_l2/weights/best.pt"
    
]

# Path to the input video and output location
video_path = "/datashare/project/vids_test/4_2_24_A_1_small.mp4"  # Replace with your video path
#video_path = "/datashare/project/vids_tune/4_2_24_B_2_small.mp4"  # Replace with your video path
output_dir = "predictions"  # Directory to save the output videos
os.makedirs(output_dir, exist_ok=True)

# Duration to process in seconds
duration_seconds = 15

# Process the video with each model separately
for model_path in model_paths:
    # Load the current YOLOv8 segmentation model
    model = YOLO(model_path)
    
    # Generate a unique output video filename based on the model name
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    output_video_path = os.path.join(output_dir, f"output_final_{model_name}.mp4")
    #output_video_path = os.path.join(output_dir, f"output_tune_{model_name}.mp4")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Calculate the maximum frames to process
    max_frames = fps * duration_seconds

    # Define the video writer to save output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO segmentation model on the frame
        results = model.predict(frame, task="segment")  # Set task to "segment" for segmentation

        # Draw the segmented masks on the frame
        annotated_frame = results[0].plot()  # Automatically draws masks and annotations on the frame

        # Write the processed frame to the output video
        out.write(annotated_frame)

        # Increment frame count and print progress
        frame_count += 1
        print(f"Processed frame {frame_count} with model {model_name}...")

    # Release resources for the current model
    cap.release()
    out.release()

    print(f"Video processing completed for model {model_name}. Output saved to: {output_video_path}")

print("All models have processed the first 15 seconds of the video.")
