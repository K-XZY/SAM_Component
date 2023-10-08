"""
 _____  ___  ___  ___  _____            _                      
/  ___|/ _ \ |  \/  | |  __ \          | |                     
\ `--./ /_\ \| .  . | | |  \/ __ _ _ __| |__   __ _  __ _  ___ 
 `--. \  _  || |\/| | | | __ / _` | '__| '_ \ / _` |/ _` |/ _ \
/\__/ / | | || |  | | | |_\ \ (_| | |  | |_) | (_| | (_| |  __/
\____/\_| |_/\_|  |_/  \____/\__,_|_|  |_.__/ \__,_|\__, |\___|
                                                     __/ |     
                                                    |___/      
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
import os
from skimage.transform import resize
from transformers import pipeline 
from PIL import Image
import plotly.express as px
import torch
from time import process_time



# Helper functions
def batch_save_image(image_arrayList,folder_name):
    """
    Save a numpy array as a jpg image.

    :param image_array: numpy array of shape (height, width, channels)
    :param folder_name: String, path to save the image including the filename
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(folder_name,'exist')
    # Ensure the array is of type uint8
    i = 0
    for image_array in image_arrayList:
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        # Create an image from the array and save it
        image = Image.fromarray(image_array)
        file_name = folder_name+'/Image'+str(i)+'.jpg'
        image.save(file_name, "JPEG")
        i+=1
    print('saved')

# Image Cropping
def auto_scale_image(image, target_size=None):
    """
    Automatically crop and (optionally) resize an image.

    :param image: numpy array of shape (height, width, channels)
    :param target_size: (optional) Tuple (target_height, target_width) to resize the image.
    :return: cropped and resized image
    """
    # Identify non-zero (non-masked) regions
    non_zeros = np.where(np.any(image != 0, axis=-1))
    y_min, y_max = np.min(non_zeros[0]), np.max(non_zeros[0])
    x_min, x_max = np.min(non_zeros[1]), np.max(non_zeros[1])

    # Crop image to bounding box
    cropped_img = image[y_min:y_max+1, x_min:x_max+1]

    # Resize if target size is provided
    if target_size is not None:
        target_height, target_width = target_size
        cropped_img = resize(cropped_img, 
                             (target_height, target_width),
                             anti_aliasing=True,
                             preserve_range=True).astype(image.dtype)

    return cropped_img


def apply_mask(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    Apply a mask to an image.

    :param image: numpy array of shape (height, width, channels)
    :param mask: boolean numpy array of shape (height, width)
    :return: masked image
    """
    # Ensure the mask has the same shape as the image
    mask_reshaped = mask[:, :, np.newaxis]

    # Apply the mask to the image
    masked_image = np.where(mask_reshaped, image, 0)

    return masked_image


# SAM (Segment Anaything Model) Utilization
def SAM(image_url:str, save_flag = True) -> list:
    raw_image = Image.open(image_url).convert("RGB")
    print("> image loaded")

    # Load the model

    device = 0 if torch.cuda.is_available() else -1
    print("> device:", device)
    generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)
    
    print("> model loaded")

    # Generate the mask
    print("> generating mask ...")
    t0 = process_time()
    outputs = generator(raw_image,points_per_batch = 64)
    t1 = process_time()
    print("> mask generated")
    print("> time elapsed:", (t1 - t0)*10, "seconds")

    masks = outputs["masks"]
    print(f'number of masks: {len(masks)}')
    print(f'shape of masks: {masks[0].shape}')

    # Apply the mask to the image
    segments = [apply_mask(raw_image,mask) for mask in masks]

    scale = (180,180)
    print('scaling segments to',scale)

    t0 = process_time()
    croppedSegments = [auto_scale_image(segment,scale) for segment in segments]
    t1 = process_time()
    print("> segments cropped")
    print(f'> time elapsed: {(t1 - t0)*10} seconds')

    # Save the segments
    if(save_flag):
        folder_name = 'saves'
        print('saving images to',folder_name)

        t0 = process_time()
        batch_save_image(croppedSegments,folder_name)
        t1 = process_time()
        print(f'> time elapsed: {(t1 - t0)*10} seconds')
    else:
        print('> not saving images as save_flag is set to False.')

    return croppedSegments


