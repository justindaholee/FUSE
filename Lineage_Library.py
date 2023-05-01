from collections import deque
import pandas as pd
import math
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
        return f'Cell {self.cell_id} from Lineage {self.lineage_id} at Frame {self.frame} with centroid ({self.x}, {self.y})'

class Library:
    def __init__(self):
        """
        Initializes a new Library object.

        Returns:
            None
        """
        self.lineages = []

    def __init__(self, init_mask, df):
        """
        Initializes a new Library object and populates it with Cell objects based on an initial mask and DataFrame.

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

    def add_cell(self, cell):
        """
        Adds a Cell object to the Library object.

        Args:
            cell: A Cell object to be added to the Library.

        Returns:
            None
        """
        if cell.lineage_id > len(self.lineages):
            self.lineages.extend(deque() for _ in range(cell.lineage_id - len(self.lineages)))
        self.lineages[cell.lineage_id-1].append(cell)

    def recent(self, lineage_id):
        """
        Returns the most recent Cell object from a specific lineage.

        Args:
            lineage_id: The ID of the lineage to retrieve.

        Returns:
            The most recent Cell object from the specified lineage.
        """
        if lineage_id <= len(self.lineages):
            return self.lineages[lineage_id-1][-1]
    
    def to_dataframe(self):
        """
        Converts the Library object to a pandas DataFrame.

        Returns:
            A pandas DataFrame containing information about each Cell object in the Library.
        """
        data = []
        for i, lineage in enumerate(self.lineages):
            for cell in lineage:
                data.append({'cell_id': cell.cell_id, 'lineage_id': i+1, 'frame': cell.frame, 'x': cell.x, 'y': cell.y})
        return pd.DataFrame(data)
    
    def all_recent(self):
        """
        Returns a list of dictionaries representing the most recent Cell object in each lineage.

        Returns:
            A list of dictionaries, where each dictionary represents the most recent Cell object in a lineage.
            Each dictionary has the following keys: 'cell_id', 'lineage_id', 'frame', 'x', 'y',
            with corresponding values for each attribute of the Cell object.
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

    def is_recent_cell(self, frame: int, cell_id: int):
        """
        Checks if a cell is a recent cell based on the frame number and cell id.

        Args:
            frame: The frame number to check.
            cell_id: The cell id to check.

        Returns:
            The lineage number the cell was found in if it is a recent cell; otherwise, -1.
        """
        for lineage_id, lineage in enumerate(self.lineages, start=1):
            if len(lineage) > 0:
                recent_cell = lineage[-1]
                if recent_cell.frame == frame and recent_cell.cell_id == cell_id:
                    return lineage_id
        return -1
    
    # def make_assignment(self, current_frame, cell, scores):
    #     if len(scores) > 0:
            


    #     if len(scores) == 1 and (
    #         scores[0]['iou_score'] > 0.3) and (
    #         scores[0]['visual_score'] > 0.95):
    #         self.add_cell(Cell(
    #             cell_id = scores[0]['next_cell_id'],
    #             lineage_id = cell['lineage_id'],
    #             frame = current_frame,
    #             x = scores[0]['next_cell_x'],
    #             y = scores[0]['next_cell_y'],
    #         ))
    #         print("Only one match")