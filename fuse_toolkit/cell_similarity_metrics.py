
'''
Cell Similarity Metrics

This script defines functions for calculating similarity 
metrics between cells in different frames. 

Dependencies:

numpy

Functions:

calculate_iou:
    Calculates the Intersection over Union (IoU) of two cells in two frames.
cosine_similarity:
    Computes the cosine similarity between two vectors.

@author: Shani Zuniga
'''
import numpy as np

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
    dot_product = np.dot(np.squeeze(vec1), np.squeeze(vec2))
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    similarity = dot_product / (vec1_norm * vec2_norm)
    return similarity