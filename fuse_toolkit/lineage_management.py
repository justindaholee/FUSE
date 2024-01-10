'''
Cell Lineage Management Classes

This script defines two classes, Cell and Library, that are used to create and manage
a collection of cells and their respective lineages for tracking purposes. The Library
class provides methods for adding cells, accessing recent cells, and converting the
library to a DataFrame, among other operations. The Cell class represents individual
cells and their attributes: cell ID, lineage ID, frame, and centroid coordinates.

Dependencies:
    pandas
    numpy
    collections

Classes:
    Cell:Represents an individual cell with attributes and methods for cell information.
    Library: Manages collection of cells and their respective lineages for ID purposes.

@author: Shani Zuniga
'''
from typing import List, Dict, Union
from collections import deque

import heapq
import pandas as pd
import numpy as np

class Cell:
    def __init__(self, cell_id: int, lineage_id: int, frame: int, x: float, y: float):
        """
        Initializes a new Cell object.

        Args:
            cell_id: The unique ID of the cell.
            lineage_id: The ID of the cell's lineage.
            frame: The frame index in which the cell first appears.
            x: The x-coordinate of the cell's centroid.
            y: The y-coordinate of the cell's centroid.

        Returns:
            None
        """
        self.cell_id = cell_id
        self.lineage_id = lineage_id
        self.frame = frame
        self.x = x
        self.y = y
    
    def __repr__(self):
        return (
            f'Cell {self.cell_id} from Lineage {self.lineage_id}',
            f'at Frame {self.frame} with centroid ({self.x}, {self.y})'
            )

class Library:
    def __init__(self, init_mask: np.ndarray, df: pd.DataFrame):
        """
        Initializes a new Library object and populates it with Cell objects based on an 
        initial mask and DataFrame.

        Args:
            init_mask: A numpy ndarray representing the initial cell mask.
            df: A pandas DataFrame containing cell information.

        Returns:
            None
        """
        self.lineages = {}
        for cell in np.unique(init_mask):
            if cell != 0:
                cell_info = df[
                    (df['Frame']==0) & (df['ROI']==(cell-1))].reset_index()
                x = cell_info.loc[0, 'x']
                y = cell_info.loc[0, 'y']
                new_cell = Cell(cell, cell, 0, x, y)
                self.add_cell(new_cell)
        del cell, cell_info, x, y, new_cell

    def add_cell(self, cell: Cell) -> None:
        """
        Adds a Cell object to the Library object.

        Args:
            cell: A Cell object to be added to the Library.

        Returns:
            None
        """
        if cell.lineage_id not in self.lineages.keys():
            self.lineages[cell.lineage_id] = deque()
        self.lineages[cell.lineage_id].append(cell)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the Library object to a pandas DataFrame.

        Returns:
            A pandas DataFrame containing information about each Cell object
            in the Library.
        """
        data = []
        for id, lineage in self.lineages.items():
            for cell in lineage:
                data.append({
                    'cell_id': cell.cell_id,
                    'lineage_id': id,
                    'Frame': cell.frame,
                    'x': cell.x,
                    'y': cell.y
                    })
        return pd.DataFrame(data)
    
    def all_recent(self) -> List[Dict[str, Union[int, float]]]:
        """
        Returns a list of dictionaries representing the most recent Cell object
        in each lineage.

        Returns:
            A list of dictionaries, where each dictionary represents the most recent
            Cell object in a lineage. Each dictionary has the following keys:
            'cell_id', 'lineage_id', 'frame', 'x', 'y', with corresponding values for
            each attribute of the Cell object.
        """
        recent_cells = []
        for id, lineage in self.lineages.items():
            if len(lineage) > 0:
                cell = lineage[-1]
                recent_cells.append({
                    'cell_id': cell.cell_id,
                    'lineage_id': id,
                    'frame': cell.frame,
                    'x': cell.x,
                    'y': cell.y
                    })
        return recent_cells

    def is_recent_cell(self, frame: int, cell_id: int) -> int:
        """
        Checks if a cell is a recent cell based on the frame number and cell id.

        Args:
            frame: The frame number to check.
            cell_id: The cell id to check.

        Returns:
            The lineage number the cell was found in if it is a recent cell; else, -1.
        """
        for id, lineage in self.lineages.items():
            if len(lineage) > 0:
                recent_cell = lineage[-1]
                if recent_cell.frame == frame and recent_cell.cell_id == cell_id:
                    return id
        return -1
    
    def identify_cells(self, current_frame: int, scores: List[Dict], 
                       iou_weight=0.6, visual_weight=0.4) -> None:
        """
        Find the best matching cell based on IoU and visual scores, and add it to the
        Lineage Library.

        Args:
            current_frame (int): Frame number of potential matching cells.
            scores (list of dict): Potential matching cells with their
                features and scores.
            iou_weight (float): value to scale iou score by; default=0.6
            visual_weight (float): value to scale visual score by; default=0.4
        Returns:
            None
        """
        if not scores:
            return

        visual_scores = np.array([score['visual_score'] for score in scores])
        min_vis_score = np.min(visual_scores)
        max_vis_score = np.max(visual_scores)
        vis_score_range = max_vis_score - min_vis_score
        visual_scores = ((visual_scores - min_vis_score) / vis_score_range
                        if vis_score_range != 0 else np.ones_like(visual_scores))
        normalized_scores = (
            iou_weight * np.array([score['iou_score'] for score in scores]) +
            visual_weight * visual_scores)

        max_heap = [(-score, index) for index, score in enumerate(normalized_scores)]
        heapq.heapify(max_heap)
        
        matched_cells = set()
        matched_lineages = set()
        while max_heap:
            _, match_index = heapq.heappop(max_heap)
            score = scores[match_index]
            if (score['next_cell_id'] in matched_cells or
                score['lineage_id'] in matched_lineages):
                continue
            matched_cell = Cell(
                cell_id = score['next_cell_id'],
                lineage_id = score['lineage_id'],
                frame = current_frame,
                x = score['next_cell_x'],
                y = score['next_cell_y']
            )
            self.add_cell(matched_cell)
            matched_cells.add(score['next_cell_id'])
            matched_lineages.add(score['lineage_id'])
    
    def remove_short_lineages(self, min_percent: float, total_frames: int,
                              current_frame = -1) -> None:
            """
            Remove lineages that have been tracked for fewer frames than 
            min_percent/100*total_frames.

            Args:
                min_percent: float
                    Minimum percentage of total frames that a lineage must 
                    be tracked for in order to be kept.
                total_frames: int
                    Total number of frames in the video.
                current_frame: int
                    Current frame number indicating the progress of frame tracking.

            Returns:
                None
            """
            if current_frame == -1:
                effective_frames = total_frames
            else:
                effective_frames = current_frame
            min_frames = min_percent / 100 * effective_frames
            self.lineages = {
                id: lineage for id, lineage in self.lineages.items() if len(lineage) >= round(min_frames)
                }
    
    def remap_lineage_keys(self):
        """
        Remap the keys of the lineages dictionary to start from 1 and increment sequentially.

        Returns:
            None
        """
        new_lineages = {}
        new_key = 1
        for _, lineage in self.lineages.items():
            new_lineages[new_key] = lineage
            new_key += 1
        self.lineages = new_lineages