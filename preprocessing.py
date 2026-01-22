"""
ConTech Pro - Preprocessing Module
מודול לטיפול מוקדם בתמונות: גזירה, ניקוי רעש, הסרת מסגרות
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    """
    מגביל bbox (x,y,w,h) לגבולות התמונה. מוודא שw/h >= 1.
    
    Args:
        bbox: (x, y, width, height)
        width: רוחב התמונה
        height: גובה התמונה
    
    Returns:
        bbox מוגבל: (x, y, width, height)
    """
    x, y, w, h = bbox
    x = int(max(0, min(x, width - 1)))
    y = int(max(0, min(y, height - 1)))
    w = int(max(1, min(w, width - x)))
    h = int(max(1, min(h, height - y)))
    return x, y, w, h


def apply_crop(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Dict]:
    """
    מבצע גזירה (crop) על תמונה לפי bbox.
    
    Args:
        image: תמונה (H, W, C) או (H, W)
        bbox: (x, y, width, height) או None לתמונה מלאה
    
    Returns:
        (cropped_image, crop_metadata)
        
    Metadata כולל:
        - crop_applied: True/False
        - crop_bbox_px: {x, y, width, height} או None
        - crop_offset_px: {x, y} - היסט מהפינה השמאלית עליונה
        - crop_size_px: {width, height} - גודל אזור הניתוח
    """
    h, w = image.shape[:2]
    
    if not bbox:
        return image, {
            "crop_applied": False,
            "crop_bbox_px": None,
            "crop_offset_px": {"x": 0, "y": 0},
            "crop_size_px": {"width": w, "height": h},
        }

    x, y, bw, bh = clamp_bbox(bbox, w, h)
    cropped = image[y : y + bh, x : x + bw].copy()
    
    return cropped, {
        "crop_applied": True,
        "crop_bbox_px": {"x": x, "y": y, "width": bw, "height": bh},
        "crop_offset_px": {"x": x, "y": y},
        "crop_size_px": {"width": bw, "height": bh},
    }


# ==========================================
# פונקציות נוספות (אופציונליות - לא מיושמות כרגע)
# ==========================================

def auto_crop_main_drawing(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    זיהוי אוטומטי של אזור השרטוט הראשי (ללא מסגרות/כותרות).
    
    Args:
        image: תמונת התוכנית (BGR או Grayscale)
    
    Returns:
        bbox (x, y, width, height) או None אם נכשל
    
    Note:
        TODO - פונקציה זו תמומש בעתיד לזיהוי אוטומטי.
        כרגע מחזירה None (לא מופעלת).
    """
    # TODO: להוסיף לוגיקה לזיהוי אוטומטי של אזור השרטוט
    # רעיונות:
    # 1. זיהוי מסגרת חיצונית (contours גדולים)
    # 2. הסרת אזורי טקסט צפופים בשוליים
    # 3. מציאת ה-bounding box של כל הקירות המזוהים
    return None


def remove_border_frame(image: np.ndarray, border_threshold: int = 50) -> np.ndarray:
    """
    מסיר מסגרת שחורה/עבה מסביב לתמונה.
    
    Args:
        image: תמונה (BGR או Grayscale)
        border_threshold: עובי מקסימלי של מסגרת להסרה (פיקסלים)
    
    Returns:
        תמונה לאחר הסרת מסגרת
    
    Note:
        TODO - פונקציה זו תמומש בעתיד.
        כרגע מחזירה את התמונה המקורית ללא שינוי.
    """
    # TODO: להוסיף לוגיקה להסרת מסגרות
    # רעיונות:
    # 1. זיהוי קווים אופקיים/אנכיים בשוליים
    # 2. ניתוח היסטוגרמה של עמודות/שורות
    # 3. חיתוך אוטומטי של אזורים אחידים
    return image


def remove_small_components(binary: np.ndarray, min_area_px: int = 50) -> np.ndarray:
    """
    מסיר רכיבים קטנים (רעש) מתמונה בינארית.
    
    Args:
        binary: תמונה בינארית (0/255)
        min_area_px: שטח מינימלי בפיקסלים - רכיבים קטנים יותר יוסרו
    
    Returns:
        תמונה בינארית מנוקה
    
    Note:
        TODO - פונקציה זו תמומש בעתיד.
        כרגע מחזירה את התמונה המקורית ללא שינוי.
    """
    # TODO: להוסיף לוגיקה להסרת רכיבים קטנים
    # רעיונות:
    # 1. connectedComponentsWithStats
    # 2. סינון לפי area
    # 3. יצירת מסכה נקייה
    return binary


def thickness_filter(binary: np.ndarray, level: str = "medium") -> np.ndarray:
    """
    מסנן קווים לפי עובי (מסיר קווים דקים מדי או עבים מדי).
    
    Args:
        binary: תמונה בינארית (0/255)
        level: רמת סינון - "light", "medium", "aggressive"
    
    Returns:
        תמונה בינארית מסוננת
    
    Note:
        TODO - פונקציה זו תמומש בעתיד.
        כרגע מחזירה את התמונה המקורית ללא שינוי.
    """
    # TODO: להוסיף לוגיקה לסינון לפי עובי
    # רעיונות:
    # 1. Morphological operations (erosion/dilation)
    # 2. Distance transform
    # 3. Skeleton analysis
    return binary


# ==========================================
# פונקציות עזר נוספות
# ==========================================

def get_crop_bbox_from_canvas_data(canvas_json_data, scale_factor: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
    """
    מחלץ bbox מנתוני st_canvas (Streamlit).
    
    Args:
        canvas_json_data: נתוני JSON מ-st_canvas
        scale_factor: גורם סקייל אם התמונה הוקטנה לתצוגה
    
    Returns:
        (x, y, width, height) או None אם אין ציור
    """
    if not canvas_json_data or "objects" not in canvas_json_data:
        return None
    
    objects = canvas_json_data["objects"]
    if not objects:
        return None
    
    # נקח את האובייקט האחרון (הציור האחרון)
    last_obj = objects[-1]
    
    # חילוץ נתונים
    x = int(last_obj.get("left", 0) / scale_factor)
    y = int(last_obj.get("top", 0) / scale_factor)
    width = int(last_obj.get("width", 0) / scale_factor)
    height = int(last_obj.get("height", 0) / scale_factor)
    
    # בדיקת תקינות
    if width < 10 or height < 10:
        return None
    
    return (x, y, width, height)


def visualize_crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], color=(0, 255, 0), thickness=3) -> np.ndarray:
    """
    מצייר bbox על התמונה (לתצוגה).
    
    Args:
        image: תמונה (BGR)
        bbox: (x, y, width, height)
        color: צבע (BGR)
        thickness: עובי קו
    
    Returns:
        תמונה עם bbox מצויר
    """
    x, y, w, h = bbox
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # הוספת תווית
    label = f"ROI: {w}x{h}px"
    cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result
