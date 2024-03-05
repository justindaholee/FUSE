'''
Cell Image Processing Utilities

This script provides utility functions for extracting cell images from multi-frame
images and masks, processing the cell images, and previewing mask overlays.

Dependencies:

    numpy
    PIL
    tqdm
    matplotlib

Functions:

    read_multiframe_tif(filename: str,
                        channel_selection: List[int]=[1]) -> list[np.ndarray]:
        Reads a multi-frame tif file and returns a list of ndarrays of frames
        for the selected channels.
    _process_image(img_data, size=(28, 28)) -> np.ndarray:
        Rescale, convert to grayscale, pad, and normalize an input image.
    extract_cells(images_path: str, masks_path: str,
                channel: str) -> Dict[str, np.ndarray]:
        Extracts individual cell images from a multi-frame image and mask file, 
        and writes them to a dictionary.
    rearrange_dimensions(image):
        Rearranges the dimensions of an image for downstream processing such
        that they are ordered as (channels, frames, height, width).
    show_overlay(img, masks, parameters, image_name, outlines):
        Displays the original image without overlays on the left and with
        overlays on the right in grayscale.
        

@author: Shani Zuniga
'''
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm.autonotebook import tqdm


def read_multiframe_tif(filename: str,
                        channel_selection: List[int]=[1]) -> list[np.ndarray]:
    """
    Reads a multi-frame tif file and returns a list of ndarrays of frames
    for the selected channels.

    Args:
        filename: The name of the tif file to read.
        channel_selection: A list of binary values (0 or 1), where the length
            is equal to the number of channels, and each value indicates whether
            the corresponding channel should be extracted (1) or not (0).
            (Default  [1], no channels)

    Returns:
        A list of numpy ndarrays containing the frames for the selected channels.
        The frames are in order of channel selected, then frame number.
    """
    img = Image.open(filename)
    n_frames_total = img.n_frames
    n_channels = len(channel_selection)

    selected_frames = []

    for channel, is_selected in enumerate(channel_selection):
        if is_selected:
            channel_frames = []
            for i in range(channel, n_frames_total, n_channels):
                img.seek(i)
                channel_frames.append(np.array(img))
            selected_frames.extend(np.array(channel_frames))
    return selected_frames

def _process_image(img_data, size=(28, 28)) -> np.ndarray:
    """
    Rescale, convert to grayscale, pad, and normalize an input image.

    Args:
        img_data (numpy.array): Input image as a numpy array.
        size (tuple, optional): Desired output size (width, height). Default: (28, 28).

    Returns:
        np.ndarray of processed and rescaled image with pixel values in range [0, 1].
    """
    # Normalize the image to the range [0, 1]
    min_val, max_val = np.min(img_data), np.max(img_data)
    normalized_image = (img_data - min_val) / (max_val - min_val)

    # Convert image to PIL Image format, to grayscale, and pad to the desired size
    img = Image.fromarray((normalized_image * 255).astype(np.uint8))
    img = img.convert('L')
    img = ImageOps.pad(img, size, method=Image.NEAREST)

    # Convert the image back to a numpy array
    img_array = np.array(img) / 255
    
    del min_val, max_val, normalized_image, img
    return img_array

def extract_cells(images_path: str, masks_path: str,
                  channel_selection: list[int]=[1]) -> Dict[str, np.ndarray]:
    """
    Extracts individual cell images from a multi-frame image and mask file, and writes
    them to a dictionary.

    Args:
        images_path (str): The path to the multi-frame image file.
        masks_path (str): The path to the multi-frame mask file.
        channel_selection: A list or tuple of binary values (0 or 1), where
            the length is equal to the number of channels, and each value
            indicates whether the corresponding channel should be extracted (1)
            or not (0). (Default  [1], no channels)

    Returns:
        dict: A dictionary containing processed cell images with keys in the format
        "frame_{frame_idx}_cell_{cell_id}".
    """
    image_frames = read_multiframe_tif(images_path, channel_selection)
    mask_frames = read_multiframe_tif(masks_path)

    cell_dict = {}
    for frame_idx, (image_frame, mask_frame) in enumerate(
        tqdm(zip(image_frames, mask_frames),
             total=len(image_frames),
             desc='Extracting cells',
             unit="frame")):
        cell_ids = set(mask_frame.flatten())
        if 0 in cell_ids:
            cell_ids.remove(0)

        for cell_id in cell_ids:
            cell_coords = (mask_frame == cell_id).nonzero()
            x_min, x_max = min(cell_coords[0]), max(cell_coords[0])
            y_min, y_max = min(cell_coords[1]), max(cell_coords[1])

            cell_mask = (mask_frame == cell_id)
            clipped_image = image_frame * cell_mask
            cell_image = clipped_image[x_min:x_max+1, y_min:y_max+1]
            processed_img = _process_image(cell_image)

            if processed_img is not None:
                cell_dict[f"frame_{frame_idx}_cell_{cell_id}"] = processed_img
    del image_frames, mask_frames
    return cell_dict

def rearrange_dimensions(image):
    """
    Rearranges the dimensions of an image for downstream processing such
    that they are ordered as (channels, frames, height, width).

    Args:
        image (ndarray): The input image.

    Returns:
        numpy.ndarray: The processed image with rearranged dimensions.
    """
    shape = image.shape
    dim_indices = {'frames': -1, 'height': -1, 'width': -1, 'channels': -1}
    sorted_dim_indices = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
    dim_indices['height'], dim_indices['width'] = sorted_dim_indices[:2]
    for i, dim in enumerate(shape):
        if dim in [1, 2, 3, 4] and i not in sorted_dim_indices[:2]:
            if dim_indices['channels'] == -1:
                dim_indices['channels'] = i
        elif i not in sorted_dim_indices[:2]:
            if dim_indices['frames'] == -1:
                dim_indices['frames'] = i
    for dim, idx in dim_indices.items():
        if idx == -1:
            if len(shape) == 2:
                image = image[:, :, np.newaxis]
                dim_indices[dim] = 2 
            elif len(shape) == 3:
                image = image[:, :, :, np.newaxis]
                dim_indices[dim] = 3
    ordered_dim_indices = (
        dim_indices["channels"], dim_indices["frames"], 
        dim_indices["height"], dim_indices["width"]
    )
    rearranged_image = np.transpose(
        image, ordered_dim_indices)
    return rearranged_image

def show_overlay(img, masks, parameters, image_name, outlines, show_output=True):
    """
    Displays the original image without overlays on the left and with overlays on
    the right in grayscale.

    Args:
        img (numpy.ndarray): The image to display.
        masks (numpy.ndarray): A binary mask of the regions of interest.
        parameters (str): Additional parameters to display on the plot.
        image_name (str): Name of image file to display
        outlines (list): list of mask outlines
        no_output (bool): boolean if to display resulting overlay (for testing)

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(img, cmap='gray')
    ax1.set_title(f"Sample Image: {image_name}")
    ax1.axis('off')

    ax2.imshow(img)
    ax2.set_title(parameters)
    ax2.axis('off')

    for i, o in enumerate(outlines):
        y, x = np.nonzero(masks == (i + 1))
        ymed = np.median(y)
        xmed = np.median(x)
        ax2.text(xmed, ymed, '%d' % (i), color='white')
        ax2.plot(o[:, 0], o[:, 1], color='white')

    if show_output:
        plt.show()
    else:
        plt.close()