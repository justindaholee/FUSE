'''
Cell Image Processing Utilities

This script provides utility functions for extracting cell images from multi-frame
images and masks, and processing the cell images.

Dependencies:

    typing
    h5py
    numpy
    PIL

Functions:

read_multiframe_tiff(filename: str) -> np.ndarray: 
    Reads a multi-frame TIFF file and returns an ndarray of its frames.
process_image(img_data, size=(28, 28)) -> np.ndarray:
    Rescale, convert to grayscale, pad, and normalize an input image.
extract_cells_as_hdf5(images_path: str, masks_path: str,
                      output_file: str, channel: str) -> None:
    Extracts individual cell images from a multi-frame image and mask file, and writes
    them to an HDF5 file.
extract_cells_as_dict(images_path: str, masks_path: str,
                      channel: str) -> Dict[str, np.ndarray]:
    Extracts individual cell images from a multi-frame image and mask file, and writes
    them to a dictionary.

@author: Shani Zuniga
'''
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from PIL import Image, ImageOps

def read_multiframe_tiff(filename: str) -> np.ndarray:
    """
    Reads a multi-frame TIFF file and returns an ndarray of its frames.

    Args:
        filename: The name of the TIFF file to read.

    Returns:
        A numpy ndarray containing the frames of the TIFF file.
    """
    img = Image.open(filename)
    frames = []

    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames)

def process_image(img_data, size=(28, 28)) -> np.ndarray:
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

def extract_cells_as_hdf5(images_path: str,
                          masks_path: str, 
                          output_file: str,
                          channel: str) -> None:
    """
    Extracts individual cell images from a multi-frame image and mask file, and writes
    them to an HDF5 file that can be accessed with keys like: "frame_{frame_idx}_
    cell_{cell_id}".

    Args:
        image_path: The path to the multi-frame image file.
        mask_path: The path to the multi-frame mask file.
        output_file: The path to the output HDF5 file.
        channel: The index of the channel to extract (if the image is multichannel).

    Returns:
        None
    """
    image_frames = read_multiframe_tiff(images_path)
    mask_frames = read_multiframe_tiff(masks_path)

    with h5py.File(output_file, 'w') as hf:
        for frame_idx, (image_frame, mask_frame) in enumerate(zip(image_frames, 
                                                                  mask_frames)):
            if image_frame.ndim > 2:
                image_frame = image_frame[..., channel]

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
                processed_img = process_image(cell_image)

                hf.create_dataset(
                    name=f"frame_{frame_idx}_cell_{cell_id}",
                    data=processed_img
                    )

def extract_cells_as_dict(images_path: str,
                          masks_path: str,
                          channel: str) -> Dict[str, np.ndarray]:
    """
    Extracts individual cell images from a multi-frame image and mask file, and writes
    them to a dictionary.

    Args:
        images_path (str): The path to the multi-frame image file.
        masks_path (str): The path to the multi-frame mask file.
        channel (str): The index of channel to extract (if image is multichannel).

    Returns:
        dict: A dictionary containing processed cell images with keys in the format
        "frame_{frame_idx}_cell_{cell_id}".
    """
    image_frames = read_multiframe_tiff(images_path)
    mask_frames = read_multiframe_tiff(masks_path)

    cell_dict = {}
    for frame_idx, (image_frame, mask_frame) in enumerate(zip(image_frames, 
                                                                mask_frames)):
        if image_frame.ndim > 2:
            image_frame = image_frame[..., channel]

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
            processed_img = process_image(cell_image)

            cell_dict[f"frame_{frame_idx}_cell_{cell_id}"] = processed_img
    return cell_dict

# TODO: Test the functions below
# TODO: Add the functions below this point to the file header.
def overlay_masks_labels(frame_data: pd.DataFrame,
                         img: np.ndarray,
                         masks: List[np.ndarray],
                         show_labels: bool = True,
                         col_name_label: str = 'Label',
                         col_name_cell_id: str = 'ROI'
                         ) -> None:
    """
    Overlay masks and labels onto the original image.

    Args:
        frame_data: DataFrame containing cell information for the current frame.
        img: Original image as a numpy array.
        masks: List of numpy arrays representing the masks for each cell.
        show_labels: Whether to display label of each cell (default: True).
        col_name_label: Name of 'label' feature in frame_data.
        col_name_cell_id: Name of 'cell_id' feature in frame_data.
    
    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for index, row in frame_data.iterrows():
        cell_id = row[col_name_cell_id]
        label = row[col_name_label]
        mask = masks[cell_id]

        # Create mask outline by finding contours and adding them to the ax object
        contours = plt.contour(mask, levels=[0.5], colors='r', linewidths=1)
        for path in contours.collections:
            ax.add_artist(path)
        
        # Plot the label on the centroid of the cell
        if show_labels and not pd.isna(label):
            x, y = row['x'], row['y']
            ax.text(x, y, str(label), color='w', fontsize=12, ha='center', va='center')

    plt.axis('off')

def display_frame(frame_idx: int,
                  df: pd.DataFrame,
                  img_data: np.ndarray, 
                  mask_data: np.ndarray) -> None:
    """
    Display a single frame with masks and labels overlayed on the original image.

    Args:
        frame_idx: Index of the frame to display.
        df: DataFrame containing cell information for all frames.
        img_data: Numpy array containing image data for all frames.
        mask_data: Numpy array containing mask data for all frames.
    
    Returns:
        None
    """
    frame_df = df[df['frame'] == frame_idx]
    frame_img = img_data[frame_idx]
    frame_masks = [mask_data[frame_idx, cell_id] for cell_id in frame_df['cell_id']]

    overlay_masks_labels(frame_df, frame_img, frame_masks)
