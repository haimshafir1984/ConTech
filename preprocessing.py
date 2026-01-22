import numpy as np
from typing import Tuple, Optional, Dict

def clamp_bbox(bbox: Tuple[int,int,int,int], width: int, height: int) -> Tuple[int,int,int,int]:
    """Clamp bbox (x,y,w,h) into image bounds. Ensures w/h >= 1."""
    x,y,w,h = bbox
    x = int(max(0, min(x, width-1)))
    y = int(max(0, min(y, height-1)))
    w = int(max(1, min(w, width - x)))
    h = int(max(1, min(h, height - y)))
    return x,y,w,h

def apply_crop(image: np.ndarray, bbox: Optional[Tuple[int,int,int,int]]) -> Tuple[np.ndarray, Dict]:
    """Apply pixel crop on an image (H,W,C). Returns cropped image and crop meta."""
    h, w = image.shape[:2]
    if not bbox:
        return image, {
            "crop_applied": False,
            "crop_bbox_px": None,
            "crop_offset_px": {"x": 0, "y": 0},
            "crop_size_px": {"width": w, "height": h},
        }

    x,y,bw,bh = clamp_bbox(bbox, w, h)
    cropped = image[y:y+bh, x:x+bw].copy()
    return cropped, {
        "crop_applied": True,
        "crop_bbox_px": {"x": x, "y": y, "width": bw, "height": bh},
        "crop_offset_px": {"x": x, "y": y},
        "crop_size_px": {"width": bw, "height": bh},
    }
