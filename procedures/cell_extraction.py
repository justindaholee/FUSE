import os
import numpy as np
from PIL import Image
import h5py

def read_multiframe_tiff(filename):
    img = Image.open(filename)
    frames = []

    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames)

def rescale_image(image):
    min_val, max_val = np.min(image), np.max(image)
    rescaled_image = (image - min_val) / (max_val - min_val) * 255
    return rescaled_image.astype(np.uint8)

def extract_cells(image_path, mask_path, output_file, channel):
    image_frames = read_multiframe_tiff(image_path)
    mask_frames = read_multiframe_tiff(mask_path)

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
                rescaled_image = rescale_image(cell_image)

                hf.create_dataset(f"frame_{frame_idx}_cell_{cell_id}", data=rescaled_image)

input_image_path = "/content/drive/Shareddrives/H2O2/??FLOPSEE??/2023_04_17_CellImages/20230417_oROSGrFluorobrite cellovernight_tseries/2023-04-18_EXP_test/segmentations/THE_S10_stack.tif"
input_mask_path = "/content/drive/Shareddrives/H2O2/??FLOPSEE??/2023_04_17_CellImages/20230417_oROSGrFluorobrite cellovernight_tseries/2023-04-18_EXP_test/segmentations/seg_S10_stack.tif"
output_file = "/content/drive/Shareddrives/H2O2/??FLOPSEE??/2023_04_17_CellImages/20230417_oROSGrFluorobrite cellovernight_tseries/2023-04-18_EXP_test/segmentations/cells.h5"

# extract_cells(input_image_path, input_mask_path, output_file, 0)