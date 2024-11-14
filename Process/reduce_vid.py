import cv2

input_path = '/home/student/FinalProject/reduced_test_All.mp4'
output_path = 'reducedd_test_All.mp4'

# Open the video file
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)  # reduce width by half
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)  # reduce height by half
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video codec and output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize frame
    frame = cv2.resize(frame, (width, height))
    out.write(frame)

cap.release()
out.release()
