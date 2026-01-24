"""
ConTech Pro - Floor Area & Room Extraction Module
מודול חדש לחישוב שטחי רצפה, היקפים ופאנלים לפי חדרים

הגישה: סגמנטציה על בסיס מסכת קירות קיימת
Pipeline: walls → close gaps → inside segmentation → watershed/CC → metrics
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


# ==========================================
# שלב 1: סגירת פתחים במסכת קירות
# ==========================================

def close_walls_mask(walls_mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    סוגר פתחים קטנים (דלתות, חלונות) במסכת קירות
    כדי שחדרים יהיו אזורים סגורים לסגמנטציה
    
    Args:
        walls_mask: מסכת קירות (0/255)
        kernel_size: גודל kernel למורפולוגיה (ברירת מחדל 15)
    
    Returns:
        walls_closed: מסכת קירות סגורה
    """
    if walls_mask is None or walls_mask.size == 0:
        raise ValueError("walls_mask is empty or None")
    
    # וידוא פורמט נכון
    if len(walls_mask.shape) == 3:
        walls_mask = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2GRAY)
    
    # Closing למילוי פערים קטנים
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    walls_closed = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # ניקוי רעשים קטנים
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_closed = cv2.morphologyEx(walls_closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    return walls_closed


# ==========================================
# שלב 2: חישוב מסכת "פנים" (inside)
# ==========================================

def compute_inside_mask(walls_closed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    מחשב מסכת "פנים" - כל מה שנמצא בתוך המבנה (לא מחוץ לו)
    
    Pipeline:
    1. free = NOT(walls)
    2. outside = flood-fill מהגבולות על free
    3. inside = free AND NOT(outside)
    
    Args:
        walls_closed: מסכת קירות סגורה
    
    Returns:
        (inside_mask, outside_mask)
    """
    h, w = walls_closed.shape
    
    # 1. free space = NOT walls
    free_mask = cv2.bitwise_not(walls_closed)
    
    # 2. מצא את החוץ - flood fill מכל הגבולות
    outside_mask = np.zeros((h, w), dtype=np.uint8)
    
    # flood fill מארבע פינות הדף
    seed_points = [
        (0, 0),          # פינה שמאלית עליונה
        (w-1, 0),        # פינה ימנית עליונה
        (0, h-1),        # פינה שמאלית תחתונה
        (w-1, h-1)       # פינה ימנית תחתונה
    ]
    
    # הוסף גם נקודות לאורך השוליים
    step = max(w, h) // 20  # כל 5% מהצד
    for x in range(0, w, step):
        seed_points.append((x, 0))
        seed_points.append((x, h-1))
    for y in range(0, h, step):
        seed_points.append((0, y))
        seed_points.append((w-1, y))
    
    # ביצוע flood fill
    mask_flood = np.zeros((h+2, w+2), dtype=np.uint8)
    for seed in seed_points:
        x, y = seed
        if 0 <= x < w and 0 <= y < h:
            if free_mask[y, x] == 255:  # רק אם זה free space
                cv2.floodFill(free_mask, mask_flood, (x, y), 128)  # 128 = outside marker
    
    outside_mask = (free_mask == 128).astype(np.uint8) * 255
    
    # 3. inside = free AND NOT outside
    inside_mask = cv2.bitwise_and(cv2.bitwise_not(walls_closed), cv2.bitwise_not(outside_mask))
    
    return inside_mask, outside_mask


# ==========================================
# שלב 3: פירוק לחדרים (Segmentation)
# ==========================================

def segment_rooms(
    inside_mask: np.ndarray, 
    method: str = "watershed",
    min_room_area_px: int = 500
) -> List[Dict]:
    """
    מפרק את מסכת ה-inside לחדרים נפרדים
    
    Args:
        inside_mask: מסכת פנים (0/255)
        method: "watershed" או "cc" (connected components)
        min_room_area_px: שטח מינימלי לחדר (פיקסלים)
    
    Returns:
        list[room_region] כאשר room_region = {
            'id': int,
            'mask': np.ndarray (bool או uint8),
            'area_px': int,
            'contour': np.ndarray,
            'perimeter_px': float,
            'center': (cx, cy),
            'bbox': (x, y, w, h)
        }
    """
    if inside_mask is None or inside_mask.size == 0:
        return []
    
    rooms = []
    
    if method == "watershed":
        rooms = _segment_watershed(inside_mask, min_room_area_px)
    elif method == "cc":
        rooms = _segment_connected_components(inside_mask, min_room_area_px)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'watershed' or 'cc'")
    
    return rooms


def _segment_watershed(inside_mask: np.ndarray, min_area: int) -> List[Dict]:
    """
    פירוק באמצעות watershed על distance transform
    מומלץ כי זה מפריד חדרים מחוברים
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(inside_mask, cv2.DIST_L2, 5)
    
    # Threshold - רק מרכזים חזקים
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(inside_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background = 1
    markers[unknown == 255] = 0  # Unknown = 0
    
    # Watershed
    # צריך תמונה RGB
    img_rgb = cv2.cvtColor(inside_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_rgb, markers)
    
    # חילוץ חדרים
    rooms = []
    unique_labels = np.unique(markers)
    
    for label in unique_labels:
        if label <= 1:  # Skip background and border
            continue
        
        room_mask = (markers == label).astype(np.uint8) * 255
        area_px = np.count_nonzero(room_mask)
        
        if area_px < min_area:
            continue
        
        # מצא contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        perimeter_px = cv2.arcLength(contour, True)
        
        # מרכז
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        rooms.append({
            'id': int(label) - 2,  # Start from 0
            'mask': room_mask,
            'area_px': area_px,
            'contour': contour,
            'perimeter_px': perimeter_px,
            'center': (cx, cy),
            'bbox': (x, y, w, h)
        })
    
    return rooms


def _segment_connected_components(inside_mask: np.ndarray, min_area: int) -> List[Dict]:
    """
    פירוק פשוט באמצעות connected components
    טוב לחדרים מופרדים לגמרי
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inside_mask, connectivity=8
    )
    
    rooms = []
    
    for label in range(1, num_labels):  # Skip background (0)
        area_px = stats[label, cv2.CC_STAT_AREA]
        
        if area_px < min_area:
            continue
        
        room_mask = (labels == label).astype(np.uint8) * 255
        
        # מצא contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        perimeter_px = cv2.arcLength(contour, True)
        
        # מרכז
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        
        # Bounding box
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        
        rooms.append({
            'id': label - 1,
            'mask': room_mask,
            'area_px': area_px,
            'contour': contour,
            'perimeter_px': perimeter_px,
            'center': (cx, cy),
            'bbox': (x, y, w, h)
        })
    
    return rooms


# ==========================================
# שלב 4: חישוב מדדים לכל חדר
# ==========================================

def compute_room_metrics(
    room_regions: List[Dict],
    meters_per_pixel: Optional[float] = None,
    meters_per_pixel_x: Optional[float] = None,
    meters_per_pixel_y: Optional[float] = None,
) -> List[Dict]:
    """
    מחשב מדדים מפורטים לכל חדר
    
    Args:
        room_regions: רשימת חדרים מ-segment_rooms
        meters_per_pixel: יחס המרה איזוטרופי (אם None, רק פיקסלים)
        meters_per_pixel_x: יחס המרה בציר X (אופציונלי)
        meters_per_pixel_y: יחס המרה בציר Y (אופציונלי)
    
    Returns:
        list[room_metrics] עם שדות:
        - room_id, area_px, area_m2, perimeter_px, perimeter_m,
          baseboard_m, center, bbox, confidence, limitations
    """
    results = []
    
    for room in room_regions:
        metrics = {
            'room_id': room['id'],
            'area_px': room['area_px'],
            'perimeter_px': room['perimeter_px'],
            'center': room['center'],
            'bbox': room['bbox'],
            'limitations': []
        }
        
        # המרה למטרים (אם אפשר)
        if (
            meters_per_pixel_x is not None
            and meters_per_pixel_y is not None
            and meters_per_pixel_x > 0
            and meters_per_pixel_y > 0
        ):
            metrics['area_m2'] = room['area_px'] * meters_per_pixel_x * meters_per_pixel_y
            meters_per_pixel_eff = (meters_per_pixel_x + meters_per_pixel_y) / 2
            metrics['perimeter_m'] = room['perimeter_px'] * meters_per_pixel_eff
            metrics['baseboard_m'] = metrics['perimeter_m']  # MVP
            metrics['confidence'] = 0.85  # סבירות בסיסית
        elif meters_per_pixel is not None and meters_per_pixel > 0:
            metrics['area_m2'] = room['area_px'] * (meters_per_pixel ** 2)
            metrics['perimeter_m'] = room['perimeter_px'] * meters_per_pixel
            metrics['baseboard_m'] = metrics['perimeter_m']  # MVP
            metrics['confidence'] = 0.85  # סבירות בסיסית
        else:
            metrics['area_m2'] = None
            metrics['perimeter_m'] = None
            metrics['baseboard_m'] = None
            metrics['confidence'] = 0.0
            metrics['limitations'].append("אין קנה מידה - מציג בפיקסלים בלבד")
        
        # בדיקות איכות נוספות
        if room['area_px'] < 1000:
            metrics['limitations'].append("חדר קטן מאוד - ייתכן שזה לא חדר אמיתי")
            metrics['confidence'] *= 0.7
        
        # צורה מוזרה?
        if room['perimeter_px'] > 0:
            circularity = 4 * np.pi * room['area_px'] / (room['perimeter_px'] ** 2)
            if circularity < 0.1:  # צורה מאוד מוזרה
                metrics['limitations'].append("צורה לא סטנדרטית")
                metrics['confidence'] *= 0.8
        
        results.append(metrics)
    
    return results


# ==========================================
# שלב 5: התאמה לשמות חדרים מ-LLM
# ==========================================

def match_rooms_to_text(
    room_metrics: List[Dict],
    llm_rooms: List[Dict],
    max_area_diff_m2: float = 2.0
) -> List[Dict]:
    """
    מתאים בין חדרים גאומטריים לשמות חדרים מהטקסט
    
    Args:
        room_metrics: חדרים מזוהים גאומטרית
        llm_rooms: חדרים מהטקסט (LLM) עם {name.value, area_m2.value}
        max_area_diff_m2: סף הפרש שטח מקסימלי
    
    Returns:
        room_metrics מעודכן עם שדות:
        - matched_name, area_text_m2, diff_m2, match_confidence
    """
    if not llm_rooms or not room_metrics:
        return room_metrics
    
    # חלץ שמות ושטחים מ-LLM
    text_rooms = []
    for llm_room in llm_rooms:
        name_val = llm_room.get('name')
        area_val = llm_room.get('area_m2')
        
        # טיפול בפורמט {value, confidence}
        if isinstance(name_val, dict):
            name_val = name_val.get('value')
        if isinstance(area_val, dict):
            area_val = area_val.get('value')
        
        if name_val and area_val:
            try:
                text_rooms.append({
                    'name': str(name_val),
                    'area_m2': float(area_val)
                })
            except (ValueError, TypeError):
                continue
    
    if not text_rooms:
        return room_metrics
    
    # התאמה - greedy matching לפי מינימום הפרש שטח
    matched_text = set()
    
    for room in room_metrics:
        if room['area_m2'] is None:
            continue
        
        best_match = None
        min_diff = float('inf')
        
        for idx, text_room in enumerate(text_rooms):
            if idx in matched_text:
                continue
            
            diff = abs(room['area_m2'] - text_room['area_m2'])
            
            if diff < min_diff and diff <= max_area_diff_m2:
                min_diff = diff
                best_match = (idx, text_room, diff)
        
        if best_match:
            idx, text_room, diff = best_match
            matched_text.add(idx)
            
            room['matched_name'] = text_room['name']
            room['area_text_m2'] = text_room['area_m2']
            room['diff_m2'] = diff
            room['match_confidence'] = max(0, 1.0 - (diff / text_room['area_m2']))
        else:
            room['matched_name'] = None
            room['area_text_m2'] = None
            room['diff_m2'] = None
            room['match_confidence'] = 0.0
    
    return room_metrics


# ==========================================
# שלב 6: פונקציה מאחדת (High-level API)
# ==========================================

def analyze_floor_and_rooms(
    walls_mask: np.ndarray,
    original_image: np.ndarray,
    meters_per_pixel: Optional[float] = None,
    meters_per_pixel_x: Optional[float] = None,
    meters_per_pixel_y: Optional[float] = None,
    llm_rooms: Optional[List[Dict]] = None,
    segmentation_method: str = "watershed",
    min_room_area_px: int = 500
) -> Dict:
    """
    API ראשי לניתוח רצפות וחדרים
    
    Args:
        walls_mask: מסכת קירות (0/255)
        original_image: תמונה מקורית (לויזואליזציה)
        meters_per_pixel: יחס המרה איזוטרופי (optional)
        meters_per_pixel_x: יחס המרה בציר X (optional)
        meters_per_pixel_y: יחס המרה בציר Y (optional)
        llm_rooms: חדרים מהטקסט (optional)
        segmentation_method: "watershed" או "cc"
        min_room_area_px: שטח מינימלי לחדר
    
    Returns:
        {
            'success': bool,
            'rooms': List[Dict],  # room metrics
            'totals': {
                'num_rooms': int,
                'total_area_m2': float,
                'total_perimeter_m': float,
                'total_baseboard_m': float
            },
            'visualizations': {
                'overlay': np.ndarray,  # תמונה עם חדרים מסומנים
                'masks': Dict[int, np.ndarray]
            },
            'limitations': List[str],
            'debug': Dict
        }
    """
    result = {
        'success': False,
        'rooms': [],
        'totals': {},
        'visualizations': {},
        'limitations': [],
        'debug': {}
    }
    
    try:
        # 1. סגירת פתחים
        walls_closed = close_walls_mask(walls_mask, kernel_size=15)
        result['debug']['walls_closed'] = walls_closed
        
        # 2. חישוב inside
        inside_mask, outside_mask = compute_inside_mask(walls_closed)
        result['debug']['inside_mask'] = inside_mask
        result['debug']['outside_mask'] = outside_mask
        
        if np.count_nonzero(inside_mask) == 0:
            result['limitations'].append("לא נמצאו אזורים פנימיים - ייתכן שזיהוי הקירות לא מדויק")
            return result
        
        # 3. פירוק לחדרים
        room_regions = segment_rooms(inside_mask, method=segmentation_method, min_room_area_px=min_room_area_px)
        
        if not room_regions:
            result['limitations'].append("לא נמצאו חדרים - נסה להוריד את min_room_area_px")
            return result
        
        result['debug']['num_regions_found'] = len(room_regions)
        
        # 4. חישוב מדדים
        room_metrics = compute_room_metrics(
            room_regions,
            meters_per_pixel=meters_per_pixel,
            meters_per_pixel_x=meters_per_pixel_x,
            meters_per_pixel_y=meters_per_pixel_y,
        )
        
        # 5. התאמה לטקסט (אם יש)
        if llm_rooms:
            room_metrics = match_rooms_to_text(room_metrics, llm_rooms)
        
        # 6. חישוב סיכומים
        total_area_m2 = sum([r['area_m2'] for r in room_metrics if r['area_m2'] is not None])
        total_perimeter_m = sum([r['perimeter_m'] for r in room_metrics if r['perimeter_m'] is not None])
        total_baseboard_m = sum([r['baseboard_m'] for r in room_metrics if r['baseboard_m'] is not None])
        
        result['totals'] = {
            'num_rooms': len(room_metrics),
            'total_area_m2': total_area_m2 if total_area_m2 > 0 else None,
            'total_perimeter_m': total_perimeter_m if total_perimeter_m > 0 else None,
            'total_baseboard_m': total_baseboard_m if total_baseboard_m > 0 else None
        }
        
        # 7. ויזואליזציה
        try:
            overlay = create_rooms_overlay(original_image, room_regions, room_metrics)
            result['visualizations']['overlay'] = overlay
            result['visualizations']['masks'] = {r['room_id']: room_regions[i]['mask'] for i, r in enumerate(room_metrics)}
        except Exception as viz_err:
            result['limitations'].append(f"שגיאה ביצירת ויזואליזציה: {str(viz_err)}")
            result['visualizations']['overlay'] = None
            result['visualizations']['masks'] = {}
        
        result['rooms'] = room_metrics
        result['success'] = True
        
        # הוסף מגבלות כלליות
        if meters_per_pixel is None:
            result['limitations'].append("אין קנה מידה - כל המדידות בפיקסלים בלבד")
        
    except Exception as e:
        result['success'] = False
        result['limitations'].append(f"שגיאה: {str(e)}")
        result['debug']['error'] = str(e)
    
    return result


# ==========================================
# שלב 7: ויזואליזציה
# ==========================================

def create_rooms_overlay(
    original_image: np.ndarray,
    room_regions: List[Dict],
    room_metrics: List[Dict]
) -> np.ndarray:
    """
    יוצר תמונת overlay עם חדרים מסומנים בצבעים
    """
    if original_image is None or len(room_regions) == 0:
        return None
    
    try:
        # וידוא RGB
        if len(original_image.shape) == 2:
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = original_image.copy()
            if img_rgb.shape[2] == 3 and img_rgb.dtype == np.uint8:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        overlay = img_rgb.copy()
        
        # צבעים שונים לכל חדר
        colors = [
            (255, 200, 200),  # אדום בהיר
            (200, 255, 200),  # ירוק בהיר
            (200, 200, 255),  # כחול בהיר
            (255, 255, 200),  # צהוב בהיר
            (255, 200, 255),  # סגול בהיר
            (200, 255, 255),  # ציאן בהיר
        ]
        
        for idx, (region, metrics) in enumerate(zip(room_regions, room_metrics)):
            try:
                color = colors[idx % len(colors)]
                
                # מלא את החדר בצבע
                mask_bool = region['mask'] > 0
                overlay[mask_bool] = color
                
                # כתוב טקסט במרכז
                cx, cy = metrics['center']
                
                if metrics['area_m2'] is not None:
                    text = f"#{metrics['room_id']}\n{metrics['area_m2']:.1f} מ\"ר"
                else:
                    text = f"#{metrics['room_id']}\n{metrics['area_px']} px"
                
                if metrics.get('matched_name'):
                    text = f"{metrics['matched_name']}\n{metrics['area_m2']:.1f} מ\"ר"
                
                # רקע לבן לטקסט (קריא יותר)
                lines = text.split('\n')
                y_offset = cy - 15 * len(lines) // 2
                
                for line in lines:
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        overlay,
                        (cx - text_size[0]//2 - 5, y_offset - text_size[1] - 5),
                        (cx + text_size[0]//2 + 5, y_offset + 5),
                        (255, 255, 255),
                        -1
                    )
                    cv2.putText(
                        overlay, line,
                        (cx - text_size[0]//2, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                    )
                    y_offset += 20
            except Exception as e:
                # דלג על חדר זה אם יש שגיאה
                continue
        
        # שילוב עם התמונה המקורית (שקיפות)
        result = cv2.addWeighted(overlay, 0.5, img_rgb, 0.5, 0)
        
        return result
    
    except Exception as e:
        # במקרה של שגיאה כללית, החזר None
        return None
