'''
Frame-by-frame

This script contains the frame_by_frame function, which is used to identify cells in
each frame of the image sequence. The function takes a lineage library object, a list
of binary masks, a pandas DataFrame, a dictionary of cell feature vectors, a search
radius, and a connectivity value as input. The function iterates through each frame
of the image sequence and compares each cell in the current frame to each cell in the
previous frame. If the cells are similar, the cell in the current frame is added to
the lineage library. The function returns a lineage library object containing the
identified cells and their lineages.

Dependencies:
    math
    typing
    numpy
    pandas
    tqdm
    scipy
    fuse_toolkit
'''

import math
from typing import List

import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm
from scipy.spatial.distance import cdist

from .lineage_management import Library
from .cell_similarity_metrics import calculate_iou, cosine_similarity


def frame_by_frame(lib: Library, masks: List[np.ndarray], df: pd.DataFrame, 
                   cell_vectors: dict, search_radius: float, connectivity: int,
                   iou_weight=0.6, visual_weight=0.4, must_overlap=True
                   ) -> Library:
    """
    Process each frame of the image sequence and identify cells that are similar to 
    previously identified cells in the lineage library.

    Args:
    - lib: A lineage library object.
    - masks: List of numpy arrays with binary masks for each frame.
    - df: A pandas DataFrame containing information about each cell in each frame.
    - cell_vectors: Dictionary with feature vectors for each cell.
    - search_radius: Float value for max distance between similar cells.
    - connectivity: Integer value for minimum frames per lineage.
    - iou_weight: float value to scale iou score by; default=0.6
    - visual_weight: float value to scale visual score by; default=0.4
    - must_overlap: boolean kconsider cells across frames only if they overlap
    Returns:
    - A lineage library object containing identified cells and their lineages.
    """
    for i, mask in tqdm(enumerate(masks[1:]), total=len(masks)-1,
                    desc="Processing Frames", unit="frame"):
        current_frame = i + 1
        temp_df = df[df['Frame'] == current_frame]

        scores = []
        for recent_cell in lib.all_recent():
            curr_mask = masks[current_frame]
            prev_mask = masks[recent_cell['frame']]

            key = f'frame_{recent_cell["frame"]}_cell_{recent_cell["cell_id"]}'
            if key not in cell_vectors:
                continue
            recent_vec = cell_vectors[key]

            new_cells = np.unique(curr_mask)[1:]
            new_cells = new_cells[new_cells != 0]

            recent_cell_coords = np.array([recent_cell['x'], recent_cell['y']])
            new_cell_coords = (
                temp_df[temp_df['ROI'].isin(new_cells - 1)][['x', 'y']].to_numpy()
                )
            distances = cdist(recent_cell_coords.reshape(1, -1), new_cell_coords)[0]
            new_cells = new_cells[distances < search_radius]

            for new_cell in new_cells:
                if lib.is_recent_cell(current_frame, new_cell) == -1:
                    x_new, y_new = (
                        temp_df[temp_df['ROI'] == (new_cell - 1)].iloc[0][['x', 'y']]
                        )
                    
                    iou_score = calculate_iou(recent_cell['cell_id'], prev_mask, 
                                            new_cell, mask)

                    if (must_overlap and iou_score > 0) or (not must_overlap):
                        key = f'frame_{current_frame}_cell_{new_cell}'
                        if key not in cell_vectors:
                            continue
                        new_vec = cell_vectors[key]
                        visual_score = cosine_similarity(recent_vec, new_vec)

                        distance = math.sqrt(
                            (x_new - recent_cell['x'])**2 +
                            (y_new - recent_cell['y'])**2
                            )

                        scores.append({
                            'next_cell_id': new_cell,
                            'next_cell_x': x_new,
                            'next_cell_y': y_new,
                            'lineage_id': recent_cell['lineage_id'],
                            'iou_score': iou_score,
                            'visual_score': visual_score,
                            'distance': distance
                            })
        lib.identify_cells(current_frame, scores, iou_weight, visual_weight)
    lib.remove_short_lineages(connectivity, len(masks))
    return lib