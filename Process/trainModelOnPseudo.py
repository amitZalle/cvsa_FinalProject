from ultralytics import YOLO

# Load the pretrained YOLO model
#TODO: change the model
pretrained_model_path = 'segmentation_models/All/weights/best.pt'  # Path to the pre-trained model
new_model_save_path = '/home/student/FinalProject/pseudo_trained_model.pt'  # Path to save the new fine-tuned model
output_base_dir = 'segmentation_models'
model_name = 'pseudo_trained_model'

# Load the model
model = YOLO(pretrained_model_path)


# Load the model with the initial weights for each training
model = YOLO(pretrained_model_path)

# Train the model and save the results
model.train(
    data='/home/student/FinalProject/yamls/pseudo.yaml',
    epochs=10,
    batch=32,
    task="segment",
    project=output_base_dir,
    name=model_name
)


# Save the new model separately
model.save(new_model_save_path)

print(f"New fine-tuned model saved to {new_model_save_path}")
