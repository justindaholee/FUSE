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
        self.lineages = []
        for cell in np.unique(init_mask):
            if cell != 0:
                cell_info = df[(df['Frame']==0) & (df['ROI']==(cell-1))]
                x = cell_info['x'].iloc[0]
                y = cell_info['y'].iloc[0]
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
        if cell.lineage_id > len(self.lineages):
            self.lineages.extend(
                deque() for _ in range(cell.lineage_id - len(self.lineages)))
        self.lineages[cell.lineage_id-1].append(cell)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the Library object to a pandas DataFrame.

        Returns:
            A pandas DataFrame containing information about each Cell object
            in the Library.
        """
        data = []
        for i, lineage in enumerate(self.lineages):
            for cell in lineage:
                data.append({
                    'cell_id': cell.cell_id,
                    'lineage_id': i+1,
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
        for i, lineage in enumerate(self.lineages):
            if len(lineage) > 0:
                cell = lineage[-1]
                recent_cells.append({
                    'cell_id': cell.cell_id,
                    'lineage_id': i + 1,
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
        for lineage_id, lineage in enumerate(self.lineages, start=1):
            if len(lineage) > 0:
                recent_cell = lineage[-1]
                if recent_cell.frame == frame and recent_cell.cell_id == cell_id:
                    return lineage_id
        return -1
    
    def identify_cells(self, current_frame: int, scores: List[Dict], 
                    iou_weights=0.6, visual_weights=0.4) -> None:
        """
        Find the best matching cell based on IoU and visual scores, and add it to the
        Lineage Library.

        Args:
            current_frame (int): Frame number of potential matching cells.
            cell (dict): Reference cell with its features.
            scores (list of dict): Potential matching cells with their
                features and scores.

        Returns:
            None
        """
        if not scores:
            return

        visual_scores = np.array([score['visual_score'] for score in scores])
        min_vis_score = np.min(visual_scores)
        max_vis_score = np.max(visual_scores)
        vis_score_range = max_vis_score - min_vis_score

        if vis_score_range != 0:
            visual_scores = (visual_scores - min_vis_score) / vis_score_range
        else:
            visual_scores = np.ones_like(visual_scores)

        normalized_scores = (
            iou_weights * np.array([score['iou_score'] for score in scores]) +
            visual_weights * visual_scores
        )

        while scores:
            match_index = np.argmax(normalized_scores)
            
            matched_cell = Cell(
                cell_id = scores[match_index]['next_cell_id'],
                lineage_id = scores[match_index]['lineage_id'],
                frame = current_frame,
                x = scores[match_index]['next_cell_x'],
                y = scores[match_index]['next_cell_y']
                )
            self.add_cell(matched_cell)

            matched_cell_id = scores[match_index]['next_cell_id']

            cell_id_indices = [i for i, score in enumerate(scores)
                               if score['next_cell_id'] == matched_cell_id]
            scores = [score for score in scores 
                      if score['next_cell_id'] != matched_cell_id]
            normalized_scores = np.delete(normalized_scores, cell_id_indices)
    
    def remove_short_lineages(self, min_percent: float, total_frames: int) -> None:
            """
            Remove lineages that have been tracked for fewer frames than 
            min_percent/100*total_frames.

            Parameters:
            
                min_percent: float
                    Minimum percentage of total frames that a lineage must 
                    be tracked for in order to be kept.
                total_frames: int
                    Total number of frames in the video.

            Returns:

                None
            """
            min_frames = min_percent / 100 * total_frames
            self.lineages = [
                lineage for lineage in self.lineages if len(lineage) >= min_frames
                ]