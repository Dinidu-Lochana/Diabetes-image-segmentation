# utils/box_utils.py

import numpy as np

def boxes_to_centers(boxes):
    """
    Convert bounding boxes (x1, y1, x2, y2) to center points.
    
    Args:
        boxes (List or ndarray): Bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Center points [cx, cy] for each box.
    """
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append([cx, cy])
    return np.array(centers)
