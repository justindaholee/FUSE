import os
import h5py
import numpy as np
from PIL import Image, ImageOps

def read_multiframe_tiff(filename: str):
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

def process_image(img_data, size=(28, 28)):
    """
    Rescale, convert to grayscale, pad, and normalize an input image.

    Args:
        img_data (numpy.array): Input image as a numpy array.
        size (tuple, optional): Desired output size (width, height). Default is (28, 28).

    Returns:
        numpy.array: Processed and rescaled image with pixel values in range [0, 1].
    """
    # Normalize the image to the range [0, 1]
    min_val, max_val = np.min(img_data), np.max(img_data)
    normalized_image = (img_data - min_val) / (max_val - min_val)

    # Convert the normalized image to PIL Image format, convert to grayscale, and pad to the desired size
    img = Image.fromarray((normalized_image * 255).astype(np.uint8))
    img = img.convert('L')
    img = ImageOps.pad(img, size, method=Image.NEAREST)

    # Convert the image back to a numpy array
    img_array = np.array(img) / 255
    
    del min_val, max_val, normalized_image, img
    return img_array

def extract_cells(images_path: str, masks_path: str, output_file: str, channel: str):
    """
    Extracts individual cell images from a multi-frame image and mask file, and writes them to an HDF5 file.

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
        for frame_idx, (image_frame, mask_frame) in enumerate(zip(image_frames, mask_frames)):
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

                hf.create_dataset(f"frame_{frame_idx}_cell_{cell_id}", data=processed_img)

def calculate_iou(cell1: int, frame1: np.ndarray, cell2: int, frame2: np.ndarray):
    """
    Calculates the Intersection over Union (IoU) of two cells in two frames.

    Args:
        cell1: The ID of the first cell to compare.
        frame1: A numpy ndarray representing the first frame.
        cell2: The ID of the second cell to compare.
        frame2: A numpy ndarray representing the second frame.

    Returns:
        The IoU of the two cells, as a float between 0 and 1.

    Raises:
        ValueError: If either cell ID is zero (which represents the background).
    """
    if cell1 != 0 and cell2 != 0:
        mask1 = frame1 == cell1
        mask2 = frame2 == cell2
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union
    else:
        raise ValueError("Both cells must have non-zero IDs")
    return iou

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    
    Args:
        vec1: np.array, the first vector
        vec2: np.array, the second vector
    Returns:
        float, the cosine similarity between vec1 and vec2
    """
    dot_product = np.dot(vec1, vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    similarity = dot_product / (vec1_norm * vec2_norm)
    return similarity