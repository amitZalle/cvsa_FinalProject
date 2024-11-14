import os
import subprocess

os.environ['PATH'] = "/anaconda/envs/synth/bin:" + os.environ['PATH']

# go over the needles objects
for needle_idx in range(15):
    # generate total 250 HDRIs needle object alone(17 each)
    command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                    "--obj1_path", f"/datashare/project/surgical_tools_models/needle_holder/NH{needle_idx + 1}.obj",
                                                    "--obj1_type", "1",
                                                    "--camera_path", "/datashare/project/camera.json",
                                                    "--back_path", "/datashare/project/haven/hdris",
                                                    "--back_type", "hdri",
                                                    "--output_dir", "/home/student/FinalProject/synthetic_data/hdri/needle_holder",
                                                    "--img_num", "17"]),#f"{int(500 / len(obj_files))}"
    subprocess.run(command, shell=True, executable='/bin/bash')

    # generate total 250 COCOs needle object alone(17 each)
    command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                    "--obj1_path", f"/datashare/project/surgical_tools_models/needle_holder/NH{needle_idx + 1}.obj",
                                                    "--obj1_type", "1",
                                                    "--camera_path", "/datashare/project/camera.json",
                                                    "--back_path", "/datashare/project/train2017",
                                                    "--back_type", "coco",
                                                    "--output_dir", "/home/student/FinalProject/synthetic_data/coco/needle_holder",
                                                    "--img_num", "17"]),#f"{int(500 / len(obj_files))}"
    subprocess.run(command, shell=True, executable='/bin/bash')


    for tweezer_idx in range(10):
        if needle_idx==0:
            # generate total 250 HDRIs tweezer object alone(25 each)
            command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                      "--obj1_path", f"/datashare/project/surgical_tools_models/tweezers/T{tweezer_idx + 1}.obj",
                                                      "--obj1_type", "2",
                                                      "--camera_path", "/datashare/project/camera.json",
                                                      "--back_path", "/datashare/project/haven/hdris",
                                                      "--back_type", "hdri",
                                                      "--output_dir", "/home/student/FinalProject/synthetic_data/hdri/tweezers",
                                                      "--img_num", "25"]),#f"{int(500 / len(obj_files))}"
            subprocess.run(command, shell=True, executable='/bin/bash')
 
            # generate total 250 COCOs tweezer object alone(25 each)
            command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                      "--obj1_path", f"/datashare/project/surgical_tools_models/tweezers/T{tweezer_idx + 1}.obj",
                                                      "--obj1_type", "2",
                                                      "--camera_path", "/datashare/project/camera.json",
                                                      "--back_path", "/datashare/project/train2017",
                                                      "--back_type", "coco",
                                                      "--output_dir", "/home/student/FinalProject/synthetic_data/coco/tweezers",
                                                      "--img_num", "25"]),#f"{int(500 / len(obj_files))}"
            subprocess.run(command, shell=True, executable='/bin/bash')


        # generate total 500 HDRIs tweezer and needle objects(3 each)
        command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                    "--obj1_path", f"/datashare/project/surgical_tools_models/needle_holder/NH{needle_idx + 1}.obj",
                                                    "--obj1_type", "1",
                                                    "--obj2_path", f"/datashare/project/surgical_tools_models/tweezers/T{tweezer_idx + 1}.obj",
                                                    "--camera_path", "/datashare/project/camera.json",
                                                    "--back_path", "/datashare/project/haven/hdris",
                                                    "--back_type", "hdri",
                                                    "--output_dir", "/home/student/FinalProject/synthetic_data/hdri/both",
                                                    "--img_num", "3"]),#f"{int(500 / len(obj_files))}"
        subprocess.run(command, shell=True, executable='/bin/bash')

        # generate total 500 COCOs tweezer and needle objects(3 each)
        command = ' '.join(["blenderproc", "run", "/home/student/FinalProject/generate_frame.py",
                                                    "--obj1_path", f"/datashare/project/surgical_tools_models/needle_holder/NH{needle_idx + 1}.obj",
                                                    "--obj1_type", "1",
                                                    "--obj2_path", f"/datashare/project/surgical_tools_models/tweezers/T{tweezer_idx + 1}.obj",
                                                    "--camera_path", "/datashare/project/camera.json",
                                                    "--back_path", "/datashare/project/train2017",
                                                    "--back_type", "coco",
                                                    "--output_dir", "/home/student/FinalProject/synthetic_data/coco/both",
                                                    "--img_num", "3"]),#f"{int(500 / len(obj_files))}"
        subprocess.run(command, shell=True, executable='/bin/bash')



