from collections import deque
import pandas as pd
import math

class Cell:
    def __init__(self, cell_id, lineage_id, frame, x, y):
        self.cell_id = cell_id
        self.lineage_id = lineage_id
        self.frame = frame
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'Cell {self.cell_id} from Lineage {self.lineage_id} at Frame {self.frame} with centroid ({self.x}, {self.y})'

class Library:
    def __init__(self):
        self.lineages = []

    def add_cell(self, cell):
        if cell.lineage_id > len(self.lineages):
            self.lineages.extend(deque() for _ in range(cell.lineage_id - len(self.lineages)))
        self.lineages[cell.lineage_id-1].append(cell)

    def recent(self, lineage_id):
        if lineage_id <= len(self.lineages):
            return self.lineages[lineage_id-1][-1]
    
    def to_dataframe(self):
        data = []
        for i, lineage in enumerate(self.lineages):
            for cell in lineage:
                data.append({'cellID': cell.cell_id, 'lineageID': i+1, 'frame': cell.frame, 'x': cell.x, 'y': cell.y})
        return pd.DataFrame(data)
    
    def all_recent(self):
        recent_cells = []
        for i, lineage in enumerate(self.lineages):
            if len(lineage) > 0:
                recent_cells.append((lineage[-1].cell_id, i+1, lineage[-1].frame, lineage[-1].x, lineage[-1].y))
        return recent_cells
    
    def lineage_ids_within_radius(self, x, y, radius):
        lineage_ids = []
        for i, lineage in enumerate(self.lineages):
            if len(lineage) > 0 and math.sqrt((lineage[-1].x - x)**2 + (lineage[-1].y - y)**2) <= radius:
                lineage_ids.append(i+1)
        return lineage_ids

