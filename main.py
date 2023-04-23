import numpy as np
import pandas as pd
from Lineage_Library import Cell, Library
from functions import read_multiframe_tiff
from functions import extract_cells

cells_path = "data\RFP_GFP_MIDDLE5\cells_RFP_GFP_MIDDLE5.hdf5" # (img name e.g., 'frame_0_cell_1')
masks_path = "data\RFP_GFP_MIDDLE5\seg_RFP_GFP_MIDDLE5.tif"
info_path = "data\RFP_GFP_MIDDLE5\EXP_MIDDLE5_1.csv"
channel = 'RFP'

# TODO: Check that data exists in the correct format

# import data
masks = read_multiframe_tiff(masks_path)
info_df = pd.read_csv(info_path)
df = info_df[info_df['Channel'] == channel]

# TODO: train or import model.h5

# Initializes library
lib = Library()
for cell in np.unique(masks[0]):
    if cell != 0: # skip over 0 (background)
        # x, y = df['Centroid'][(df['Frame'] == 0) & (df['ROI'] == cell)] # read string as numbers
        x = cell # TODO: Replace with actual x 
        y = cell # TODO: Replace with actual y

        new_cell = Cell(cell, cell, 0, x, y)
        lib.add_cell(new_cell)
print(lib.to_dataframe) 


prev_mask = masks[0]
id, lin, fr, x, y = 0, 1, 2, 3, 4
for i, mask in enumerate(masks[1:]):
    print(f"{i}: {mask.shape}")

    for recent_cell in lib.all_recent():
        # 1. compute iou score for each potential next cell
        scores = []
        for new_cell in np.unique(mask)[1:]:
            # only proceed if the x and y are within certain distance
            distance = None

            # compute score
            iou_score = None

            scores.append({
                'next_cell_id': new_cell,
                'lineage_id': recent_cell[lin],
                'iou_score': iou_score,
                'distance': distance
            })

        
        