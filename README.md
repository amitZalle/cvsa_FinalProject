conda create -n synth python=3.10

conda activate synth

pip install blenderproc



pip install pycocotools

pip install ultralytics








To use the existing models, you can download the selected weight from "..." and run either "predict.py" for image or "video.py" for video. make sure to adjust the path to the weights and video/image.

For the creation of the data run "synthetic_data_generator.py". for this you must add the data to the files, and adjust the path to it in the python file.




For the understanding on the creation of the model itself you can read below(the files below will be in a seperate folder "Process"):






















We start our process with Synthetic Data Generation. this is done by using the given hdri and coco background images and needle_holder and tweezers objects and camera.json file to render images with blenderproc.

to do that you must run the python file "synthetic_data_generator.py" which will run the script "generate_frame.py" multiple times to create different datasets combinations.






now you must fix the format of the coco_data created so we can use the git below to turn it into yolo format. we will do so with "modifyannotations.py".






Then we will use the coco_to_yolo.py file from the git library "https://github.com/Koldim2001/COCO_to_YOLOv8.git" with some changes - this is important - to turn the datasets from coco format to yolo format with ~15% validation split.

you must run the next lines to install the requierments and then turn the datasets(notice we move to the folder needed and then return:


cd COCO_to_YOLOv8

pip install -r requirements.txt

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/hdri/needle_holder/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/hdri/needle_holder/yolo_data" --autosplit=True --percent_val=15

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/hdri/tweezers/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/hdri/tweezers/yolo_data" --autosplit=True --percent_val=15

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/hdri/both/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/hdri/both/yolo_data" --autosplit=True --percent_val=15

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/coco/needle_holder/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/coco/needle_holder/yolo_data" --autosplit=True --percent_val=15

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/coco/tweezers/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/coco/tweezers/yolo_data" --autosplit=True --percent_val=15

python coco_to_yolo.py --coco_dataset="/home/student/FinalProject/synthetic_data/coco/both/coco_data" --yolo_dataset="/home/student/FinalProject/synthetic_data/coco/both/yolo_data" --autosplit=True --percent_val=15

cd ..





another thing we need is to turn the photos created with coco background to RGB images with "turnRGB.py".






after that we merge the different datasets created to 4 main ones: those with only needles, those with only tweezers, those with only both, and one with all.


python merge_yolo_datasets.py synthetic_data/needle synthetic_data/hdri/needle_holder/yolo_data synthetic_data/coco/needle_holder/yolo_data

python merge_yolo_datasets.py synthetic_data/tweezers synthetic_data/hdri/tweezers/yolo_data synthetic_data/coco/tweezers/yolo_data

python merge_yolo_datasets.py synthetic_data/both synthetic_data/hdri/both/yolo_data synthetic_data/coco/both/yolo_data

python merge_yolo_datasets.py synthetic_data/all synthetic_data/hdri/needle_holder/yolo_data synthetic_data/coco/needle_holder/yolo_data synthetic_data/hdri/tweezers/yolo_data synthetic_data/coco/tweezers/yolo_data synthetic_data/hdri/both/yolo_data synthetic_data/coco/both/yolo_data






now we must fix the ICCP of the coco images with the file "fixICCP.py" to run the training smoothly.






To train you must run the file "trainModels.py". in this file you can choose from which of the 4 datasets you want to train the model, you can choose more than one model to train. all will appear in the "semantation_models" directory.






After the training you can see the result of the models with "predictModels.py". again you can choose which models you trained you want to see, and add more videos to the predictions.






For the Domain adaptation we used pseudo labeling. You can create the pseudo data using the "createHCframes.py".


to train the model on this data you shoud run "trainModelOnPseudo.py".


and finally to see the results you can run "predictPseudo.py" to see the pure results of the model.

