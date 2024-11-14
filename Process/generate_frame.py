import blenderproc as bproc
import argparse
import numpy as np
import random
import os
import json
from blenderproc.python.camera import CameraUtility
import bpy
import glob
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--obj1_path', default=None,
                    help="Path to the file of the first obj")
# 1 for needle driver, 2 for tweezers
parser.add_argument('--obj1_type', default=None,
                    help="the type of the first obj")
# obj 2 must be of type tweezers
parser.add_argument('--obj2_path', default=None,
                    help="Path to the file of the second obj if any")
parser.add_argument('--camera_path', default = "/datashare/project/camera.json", 
                    help="Path to where the camera.json file is stored")
parser.add_argument('--back_path', default=None, help="Path to where the background file is stored")
parser.add_argument('--back_type', default="hdri", help="type of the background file")
parser.add_argument('--output_dir', default="synthetic_data/outputs",
                    help="Path to where the final files, will be saved")
parser.add_argument('--img_num', default=10, type=int,
                    help="The number of images of the object we should take and render")
args = parser.parse_args()

bproc.init()

#load the objects
obj1 = bproc.loader.load_obj(args.obj1_path)[0]
obj1.set_cp("category_id", args.obj1_type)  
if args.obj2_path:
    obj2 = bproc.loader.load_obj(args.obj2_path)[0]
    obj2.set_cp("category_id", 2)


# define the light
light_types = ["POINT", "SUN", "SPOT"]

light1 = bproc.types.Light()
if args.obj2_path:
    light2 = bproc.types.Light()

# Load the camera settings from the JSON file
with open(args.camera_path, "r") as f:
    camera_settings = json.load(f)

# Extract the settings
fx = camera_settings["fx"]
fy = camera_settings["fy"]
cx = camera_settings["cx"]
cy = camera_settings["cy"]
width = camera_settings["width"]
height = camera_settings["height"]

# Set the intrinsics in BlenderProc
CameraUtility.set_intrinsics_from_K_matrix(
    K=[[fx, 0, cx],
       [0, fy, cy],
       [0,  0,  1]],
    image_width=width,
    image_height=height
)


# get hdris files
if args.back_type == 'hdri':
    back_files = glob.glob(os.path.join(args.back_path, "*", "*.hdr"))
elif args.back_type == 'coco':
    back_files = glob.glob(os.path.join(args.back_path, "*.jpg"))
else:
    print("Error in background")



def set_light(light, location):

    light_type = random.choice(light_types)
    light1.set_type(light_type)

    # Sample light position near the object with randomized distance and elevation
    light_position = bproc.sampler.shell(
        center=location,
        radius_min=1.0,
        radius_max=4.0,
        elevation_min=1.0,
        elevation_max=60.0
    )

    light.set_location(light_position)

    # Randomly adjust light intensity
    light.set_energy(random.uniform(150, 900))

    # Set light color with random RGB values
    light.set_color([
        random.random(),  # Red component
        random.random(),  # Green component
        random.random()   # Blue component
    ])


def set_camera():

#    if args.obj2_path:
#        location = (obj1.get_location()+obj2.get_location())/2
#    else:
    location = obj1.get_location()

    # Randomly set camera position within a spherical area around the object
    cam_position = bproc.sampler.shell(
        center=location,
        radius_min=10,  # Slightly modified radius range
        radius_max=18,
        elevation_min=-85,
        elevation_max=85
    )
   
    # Define a look-at point near the object with a small random variation
    target_position = obj1.get_location() + np.random.random(3) - 0.5  # Random offset in each axis from [-0.5, 0.5]
    
    # Calculate rotation for the camera to face the target, with random in-plane rotation
    orientation_matrix = bproc.camera.rotation_from_forward_vec(
        target_position - cam_position, 
        inplane_rot=np.random.uniform(-0.75, 0.75)
    )
    
    # Assuming the background is loaded
    # TODO change line
    bg_width = 1920  # Replace with actual width of your background
    bg_height = 1080  # Replace with actual height of your background
    bproc.camera.set_resolution(bg_width, bg_height)

    # Build the final transformation matrix for the camera setup
    camera_transform_matrix = bproc.math.build_transformation_mat(cam_position, orientation_matrix)
    
    return camera_transform_matrix

def set_random_location(obj, image_bounds):
    """
    Set a random location within the image bounds for an object.
    """
    # Randomly select x, y, z within image bounds to spread across the scene
    x = random.uniform(image_bounds['x_min'], image_bounds['x_max'])
    y = random.uniform(image_bounds['y_min'], image_bounds['y_max'])
    z = 0
    location = np.array([x, y, z])
    obj.set_location(location)
    return location

def set_objs():
    # Define the bounds for the scene within which the objects can be placed
    image_bounds = {
        'x_min': 0, 'x_max': 1,
        'y_min': 0, 'y_max': 1,
    }

    # Set a random location for the first object
    location_obj1 = set_random_location(obj1, image_bounds)

    if args.obj2_path:
        # Decide whether to allow overlap (10% probability)
        allow_overlap = random.random() < 0.10

        if allow_overlap:
             # Place the second object without distance constraints
            location_obj2 = set_random_location(obj2, image_bounds)
        else:
            # Place the second object with a minimum separation constraint
            min_separation = 1  # Minimum distance to prevent overlap
            max_attempts = 10  # Max retries to find a valid location
            attempts = 0
            while attempts < max_attempts:
                x_left = obj1.get_location()[0]<0.5
                x_down = obj1.get_location()[1]<0.5                
                image_bounds = {
                    'x_min': obj1.get_location()[0]+0.1 if x_left else 0, 'x_max': 1 if x_left else obj1.get_location()[0]-0.1,
                    'y_min': obj1.get_location()[1]+0.1 if x_down else 0, 'y_max': 1 if x_down else obj1.get_location()[1]-0.1,
                }

                location_obj2 = set_random_location(obj2, image_bounds)
                distance = np.linalg.norm(location_obj2 - location_obj1)
        
                # Ensure objects are separated if overlap is not allowed
                if distance >= min_separation:
                   break
                attempts += 1

def set_mist(mist):
    if mist:
        bpy.context.scene.world.mist_settings.use_mist = True
        bpy.context.scene.world.mist_settings.intensity = random.uniform(0.1, 0.5)
        bpy.context.scene.world.mist_settings.start = random.uniform(5, 20)
        bpy.context.scene.world.mist_settings.depth = random.uniform(10, 30)
    else:
        bpy.context.scene.world.mist_settings.use_mist = False


imgs= 0
iter = 0
while iter<500 and imgs<args.img_num:

    set_objs()

    set_light(light1, np.array(obj1.get_location()))
    if args.obj2_path:
        set_light(light2, np.array(obj2.get_location()))

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)
        
    cam2world_matrix = set_camera()

    set_mist(random.random()<0.1)

    if args.back_type == 'hdri':
        random_back = random.choice(back_files)
        bproc.world.set_world_background_hdr_img(random_back)

    
    # Only add camera pose if object is still visible
    if obj1 in bproc.camera.visible_objects(cam2world_matrix) and (not args.obj2_path or obj2 in bproc.camera.visible_objects(cam2world_matrix)):
        bproc.camera.add_camera_pose(cam2world_matrix, frame=imgs)
        imgs += 1
    iter += 1

#####bproc.renderer.set_max_amount_of_samples(100) # to speed up rendering, reduce the number of samples
# keep the background we added if there is one
if args.back_type=='hdri':
    bproc.renderer.set_output_format(enable_transparency=False)
else:
    bproc.renderer.set_output_format(enable_transparency=True)

# add segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# Render the image
data = bproc.renderer.render()

# Write data to coco file
bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                        instance_segmaps=data["instance_segmaps"],
                        instance_attribute_maps=data["instance_attribute_maps"],
                        colors=data["colors"],
                        mask_encoding_format="polygon",
                        append_to_existing_output=True)


def overlay_background(rendered_image_path, background_image_path):
    # Load the background image
    background = Image.open(background_image_path).convert("RGBA")

    # Load the rendered image
    rendered_image = Image.open(rendered_image_path).convert("RGBA")

    # Resize background to match rendered image size
    background = background.resize(rendered_image.size)

    background.paste(rendered_image , (0,0), mask = rendered_image) 

    return background

if args.back_type == 'coco':

    # Get all rendered images
    rendered_images = glob.glob(os.path.join(args.output_dir, "*", "*", "*.png"))  # Change to *.jpg if necessary

    # Process each rendered image
    for rendered_image_path in rendered_images:
       
        # Randomly select a background for each rendered image
        random_background_path = random.choice(back_files)

        # Overlay the rendered image on the JPEG background
        final_image = overlay_background(rendered_image_path, random_background_path)

        # Save the final image in the same location as the rendered image
        final_image.save(rendered_image_path)
