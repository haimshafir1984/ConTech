"""
ConTech Pro - Floor Area & Room Extraction Module
מודול חדש לחישוב שטחי רצפה, היקפים ופאנלים לפי חדרים

הגישה: סגמנטציה על בסיס מסכת קירות קיימת
Pipeline: walls → close gaps → inside segmentation → watershed/CC → metrics

🆕 SAFE FIX v1.1:
- A) Adaptive wall closing (dynamic kernel)
- B) Frame seal before floodFill
- D) Don't discard large spaces as "non-standard"
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


# ==========================================
# שלב 1: סגירת פתחים במסכת קירות
# ==========================================


def close_walls_mask(walls_mask: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    סוגר פתחים קטנים (דלתות, חלונות) במסכת קירות
    כדי שחדרים יהיו אזורים סגורים לסגמנטציה

    🆕 FIX A: Adaptive kernel size

    Args:
        walls_mask: מסכת קירות (0/255)
        kernel_size: גודל kernel (אם None, יחושב אוטומטית)

    Returns:
        walls_closed: מסכת קירות סגורה
    """
    if walls_mask is None or walls_mask.size == 0:
        raise ValueError("walls_mask is empty or None")

    # וידוא פורמט נכון
    if len(walls_mask.shape) == 3:
        walls_mask = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2GRAY)

    h, w = walls_mask.shape

    # 🆕 A) Adaptive kernel size
    if kernel_size is None:
        # Formula: clamp(min(w,h) / 120, 15, 55)
        kernel_size = int(min(w, h) / 120)
        kernel_size = max(15, min(55, kernel_size))

    # Closing למילוי פערים קטנים
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    walls_closed = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ניקוי רעשים קטנים
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_closed = cv2.morphologyEx(
        walls_closed, cv2.MORPH_OPEN, kernel_small, iterations=1
    )

    return walls_closed


# ==========================================
# שלב 2: חישוב מסכת "פנים" (inside)
# ==========================================


def compute_inside_mask(walls_closed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    מחשב מסכת "פנים" - כל מה שנמצא בתוך המבנה (לא מחוץ לו)

    🆕 FIX B: Frame seal before floodFill

    Pipeline:
    1. free = NOT(walls)
    2. 🆕 seal frame borders
    3. outside = flood-fill מהגבולות על free
    4. inside = free AND NOT(outside)

    Args:
        walls_closed: מסכת קירות סגורה

    Returns:
        (inside_mask, outside_mask)
    """
    h, w = walls_closed.shape

    # 🆕 B) Frame seal - prevent edge leakage
    # Create a sealed copy with 3-pixel border
    sealed_walls = walls_closed.copy()
    sealed_walls[0:3, :] = 255  # top border
    sealed_walls[-3:, :] = 255  # bottom border
    sealed_walls[:, 0:3] = 255  # left border
    sealed_walls[:, -3:] = 255  # right border

    # 1. free space = NOT walls (use sealed version)
    free_mask = cv2.bitwise_not(sealed_walls)

    # 2. מצא את החוץ - flood fill מכל הגבולות
    outside_mask = np.zeros((h, w), dtype=np.uint8)

    # flood fill מארבע פינות הדף
    seed_points = [
        (0, 0),  # פינה שמאלית עליונה
        (w - 1, 0),  # פינה ימנית עליונה
        (0, h - 1),  # פינה שמאלית תחתונה
        (w - 1, h - 1),  # פינה ימנית תחתונה
    ]

    # הוסף גם נקודות לאורך השוליים
    step = max(w, h) // 20  # כל 5% מהצד
    for x in range(0, w, step):
        seed_points.append((x, 0))
        seed_points.append((x, h - 1))
    for y in range(0, h, step):
        seed_points.append((0, y))
        seed_points.append((w - 1, y))

    # ביצוע flood fill
    mask_flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in seed_points:
        x, y = seed
        if 0 <= x < w and 0 <= y < h:
            if free_mask[y, x] == 255:  # רק אם זה free space
                cv2.floodFill(
                    free_mask, mask_flood, (x, y), 128
                )  # 128 = outside marker

    outside_mask = (free_mask == 128).astype(np.uint8) * 255

    # 3. inside = free AND NOT outside (use original walls, not sealed)
    inside_mask = cv2.bitwise_and(
        cv2.bitwise_not(walls_closed), cv2.bitwise_not(outside_mask)
    )

    return inside_mask, outside_mask


# ==========================================
# שלב 3: פירוק לחדרים (Segmentation)
# ==========================================


def segment_rooms(
    inside_mask: np.ndarray,
    method: str = "watershed",
    min_room_area_px: int = 100,  # 🆕 lowered from 500 to 100
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
        contours, _ = cv2.findContours(
            room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter_px = cv2.arcLength(contour, True)

        # מרכז
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        rooms.append(
            {
                "id": int(label) - 2,  # Start from 0
                "mask": room_mask,
                "area_px": area_px,
                "contour": contour,
                "perimeter_px": perimeter_px,
                "center": (cx, cy),
                "bbox": (x, y, w, h),
            }
        )

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
        contours, _ = cv2.findContours(
            room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

        rooms.append(
            {
                "id": label - 1,
                "mask": room_mask,
                "area_px": area_px,
                "contour": contour,
                "perimeter_px": perimeter_px,
                "center": (cx, cy),
                "bbox": (x, y, w, h),
            }
        )

    return rooms


# ==========================================
# שלב 4: חישוב מדדים לכל חדר
# ==========================================


def compute_dynamic_min_room_area_px(
    inside_mask: np.ndarray,
    ratio: float = 0.0003,  # 🆕 lowered from 0.0008
    min_px: int = 100,  # 🆕 lowered from 300
    max_px: int = 8000,
) -> int:
    """
    מחשב סף מינימום דינאמי לחדר לפי גודל השטח הפנימי.
    """
    if inside_mask is None or inside_mask.size == 0:
        return min_px
    total_inside = int(np.count_nonzero(inside_mask))
    dynamic_px = int(total_inside * ratio)
    return int(max(min_px, min(max_px, dynamic_px)))


def compute_room_metrics(
    room_regions: List[Dict],
    meters_per_pixel: Optional[float] = None,
    meters_per_pixel_x: Optional[float] = None,
    meters_per_pixel_y: Optional[float] = None,
) -> List[Dict]:
    """
    מחשב מדדים מפורטים לכל חדר

    🆕 FIX D: Don't discard large spaces as "non-standard"

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
            "room_id": room["id"],
            "area_px": room["area_px"],
            "perimeter_px": room["perimeter_px"],
            "center": room["center"],
            "bbox": room["bbox"],
            "limitations": [],
        }

        # המרה למטרים (אם אפשר)
        if (
            meters_per_pixel_x is not None
            and meters_per_pixel_y is not None
            and meters_per_pixel_x > 0
            and meters_per_pixel_y > 0
        ):
            metrics["area_m2"] = (
                room["area_px"] * meters_per_pixel_x * meters_per_pixel_y
            )
            meters_per_pixel_eff = (meters_per_pixel_x + meters_per_pixel_y) / 2
            metrics["perimeter_m"] = room["perimeter_px"] * meters_per_pixel_eff
            metrics["baseboard_m"] = metrics["perimeter_m"]  # MVP
            metrics["confidence"] = 0.85  # סבירות בסיסית
        elif meters_per_pixel is not None and meters_per_pixel > 0:
            metrics["area_m2"] = room["area_px"] * (meters_per_pixel**2)
            metrics["perimeter_m"] = room["perimeter_px"] * meters_per_pixel
            metrics["baseboard_m"] = metrics["perimeter_m"]  # MVP
            metrics["confidence"] = 0.85  # סבירות בסיסית
        else:
            metrics["area_m2"] = None
            metrics["perimeter_m"] = None
            metrics["baseboard_m"] = None
            metrics["confidence"] = 0.0
            metrics["limitations"].append("אין קנה מידה - מציג בפיקסלים בלבד")

        # בדיקות איכות נוספות
        if room["area_px"] < 1000:
            metrics["limitations"].append("חדר קטן מאוד - ייתכן שזה לא חדר אמיתי")
            metrics["confidence"] *= 0.7

        # 🆕 FIX D: Don't discard large spaces
        # צורה מוזרה? רק אם החדר קטן
        if room["area_px"] < 15000:  # 🆕 Only check shape for small rooms
            if room["perimeter_px"] > 0:
                circularity = 4 * np.pi * room["area_px"] / (room["perimeter_px"] ** 2)
                if circularity < 0.1:  # צורה מאוד מוזרה
                    metrics["limitations"].append("צורה לא סטנדרטית")
                    metrics["confidence"] *= 0.8
        else:
            # 🆕 Large rooms: always keep, just note if unusual shape
            if room["perimeter_px"] > 0:
                circularity = 4 * np.pi * room["area_px"] / (room["perimeter_px"] ** 2)
                if circularity < 0.1:
                    metrics["limitations"].append("צורה לא סטנדרטית (חדר גדול)")
                    # Don't reduce confidence for large rooms!

        results.append(metrics)

    return results


# ==========================================
# שלב 5: התאמה לשמות חדרים מ-LLM
# ==========================================


def match_rooms_to_text(
    room_metrics: List[Dict],
    llm_rooms: List[Dict],
    max_area_diff_m2: float = 2.0,
    image_shape: Optional[Tuple[int, int]] = None,  # (height, width)
) -> List[Dict]:
    """
    מתאים בין חדרים גאומטריים לשמות חדרים מהטקסט.

    כאשר llm_rooms מכיל position_x_pct/position_y_pct (מ-Vision)
    ו-image_shape זמין, ההתאמה משלבת קרבה מרחבית + שטח.
    אחרת, התאמה על בסיס שטח בלבד (התנהגות קודמת).

    Args:
        room_metrics: חדרים מזוהים גאומטרית — כל חדר כולל center=(cx,cy)
        llm_rooms: חדרים מהטקסט / Vision
        max_area_diff_m2: סף הפרש שטח מקסימלי להתאמה לפי שטח בלבד
        image_shape: (height, width) של התמונה שעובדה ע"י OpenCV

    Returns:
        room_metrics מעודכן עם שדות matched_name, matched_confidence
    """
    if not llm_rooms or not room_metrics:
        return room_metrics

    img_h, img_w = (image_shape[0], image_shape[1]) if image_shape else (0, 0)

    # חילוץ שמות, שטחים ומיקומים מה-LLM/Vision
    text_rooms = []
    for lr in llm_rooms:
        if not isinstance(lr, dict):
            continue
        name = (
            lr.get("name", {}).get("value")
            if isinstance(lr.get("name"), dict)
            else lr.get("name")
        )
        area = (
            lr.get("area_m2", {}).get("value")
            if isinstance(lr.get("area_m2"), dict)
            else lr.get("area_m2")
        )
        if not name:
            continue
        entry: Dict = {"name": str(name)}
        if area is not None:
            try:
                entry["area_m2"] = float(area)
            except (ValueError, TypeError):
                pass
        # מיקום מ-Vision (0.0–1.0)
        x_pct = lr.get("position_x_pct")
        y_pct = lr.get("position_y_pct")
        if x_pct is not None and y_pct is not None:
            try:
                entry["x_pct"] = float(x_pct)
                entry["y_pct"] = float(y_pct)
            except (ValueError, TypeError):
                pass
        text_rooms.append(entry)

    if not text_rooms:
        return room_metrics

    # בדוק האם יש מידע מרחבי ב-llm_rooms וגם image_shape
    has_spatial = (
        img_w > 0 and img_h > 0
        and any("x_pct" in tr for tr in text_rooms)
    )

    _SQRT2 = 2 ** 0.5

    used_text_indices = set()

    for rm in room_metrics:
        # ציר גאומטרי — נרמול למרחב [0,1]
        cx, cy = rm.get("center", (0, 0))
        rm_x_pct = (cx / img_w) if img_w > 0 else 0.0
        rm_y_pct = (cy / img_h) if img_h > 0 else 0.0

        rm_area = rm.get("area_m2")

        best_idx = -1
        best_score = -1.0

        for idx, tr in enumerate(text_rooms):
            if idx in used_text_indices:
                continue

            score = 0.0
            weights = 0.0

            # ──── ציון שטח ────
            tr_area = tr.get("area_m2")
            if rm_area is not None and tr_area is not None:
                area_diff = abs(rm_area - tr_area)
                area_score = max(0.0, 1.0 - area_diff / max(max_area_diff_m2, 0.01))
                # סנן החוצה אם ההפרש גדול מדי ואין מידע מרחבי
                if not has_spatial and area_diff > max_area_diff_m2:
                    continue
                score += area_score * 1.0
                weights += 1.0

            # ──── ציון מרחבי ────
            if has_spatial and "x_pct" in tr:
                dx = rm_x_pct - tr["x_pct"]
                dy = rm_y_pct - tr["y_pct"]
                dist = (dx * dx + dy * dy) ** 0.5
                spatial_score = max(0.0, 1.0 - dist / _SQRT2)
                score += spatial_score * 1.5  # משקל גבוה יותר למרחב כשזמין
                weights += 1.5

            if weights == 0.0:
                continue

            normalized = score / weights
            if normalized > best_score:
                best_score = normalized
                best_idx = idx

        if best_idx >= 0 and best_score > 0.0:
            tr = text_rooms[best_idx]
            rm["matched_name"] = tr["name"]
            rm["matched_confidence"] = round(best_score, 3)
            if "area_m2" in tr:
                rm["area_text_m2"] = tr["area_m2"]
                if rm_area is not None:
                    rm["diff_m2"] = rm_area - tr["area_m2"]
            used_text_indices.add(best_idx)

    return room_metrics


# ==========================================
# שלב 5.5: ולידציה של קנה מידה מול טקסט
# ==========================================


def validate_scale_with_text_areas(
    room_metrics: List[Dict],
    min_samples: int = 3,
    tolerance_ratio: float = 0.25,
) -> Dict:
    """
    בודק סטייה של שטחי חדרים מול שטחים מהטקסט.
    מחזיר מידע סטטיסטי והמלצת תיקון אפשרית לסקלה.
    """
    ratios = []
    for room in room_metrics:
        text_area = room.get("area_text_m2")
        geom_area = room.get("area_m2")
        if text_area and geom_area and geom_area > 0 and text_area > 0:
            ratios.append(float(text_area) / float(geom_area))

    result = {
        "status": "insufficient_data",
        "samples": len(ratios),
        "median_ratio": None,
        "suggested_scale_factor": None,
        "warning": False,
    }

    if len(ratios) < min_samples:
        return result

    median_ratio = float(np.median(ratios))
    # יחס שטחים = (יחס סקלה)^2
    suggested_scale_factor = float(np.sqrt(median_ratio))
    deviation = abs(1.0 - median_ratio)

    result.update(
        {
            "status": "ok",
            "median_ratio": median_ratio,
            "suggested_scale_factor": suggested_scale_factor,
            "warning": deviation > tolerance_ratio,
        }
    )
    return result


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
    min_room_area_px: int = 100,  # 🆕 lowered from 500
) -> Dict:
    """
    API ראשי לניתוח רצפות וחדרים

    🆕 SAFE FIX v1.1 - includes fixes A, B, D

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
        "success": False,
        "rooms": [],
        "totals": {},
        "visualizations": {},
        "limitations": [],
        "debug": {},
    }

    try:
        # 1. סגור פערים בקירות
        # 🆕 FIX A: Adaptive kernel
        walls_closed = close_walls_mask(walls_mask, kernel_size=None)

        # 2. חשב inside/outside
        # 🆕 FIX B: Frame seal
        inside_mask, outside_mask = compute_inside_mask(walls_closed)

        # חישוב min_area דינמי אם נדרש
        if min_room_area_px == 0:
            min_room_area_px = compute_dynamic_min_room_area_px(inside_mask)
            result["debug"]["dynamic_min_area_px"] = min_room_area_px

        # 3. פרק לחדרים
        # 🆕 FIX D: Lower threshold (100 instead of 500)
        room_regions = segment_rooms(
            inside_mask, method=segmentation_method, min_room_area_px=min_room_area_px
        )

        if not room_regions:
            result["limitations"].append("לא נמצאו חדרים - בדוק את מסכת הקירות")
            return result

        # 4. חשב מדדים
        # 🆕 FIX D: Don't discard large spaces
        room_metrics = compute_room_metrics(
            room_regions, meters_per_pixel, meters_per_pixel_x, meters_per_pixel_y
        )

        # 5. התאם לשמות מהטקסט
        if llm_rooms:
            img_shape = original_image.shape[:2] if original_image is not None else None
            room_metrics = match_rooms_to_text(room_metrics, llm_rooms, image_shape=img_shape)

            # וולידציה של קנה מידה
            validation = validate_scale_with_text_areas(room_metrics)
            result["debug"]["scale_validation"] = validation

            if validation.get("warning"):
                result["limitations"].append("סטייה אפשרית בקנה המידה לפי שטחים מהטקסט")

        # 6. חישוב סיכומים
        total_area_m2 = sum(
            [r["area_m2"] for r in room_metrics if r["area_m2"] is not None]
        )
        total_perimeter_m = sum(
            [r["perimeter_m"] for r in room_metrics if r["perimeter_m"] is not None]
        )
        total_baseboard_m = sum(
            [r["baseboard_m"] for r in room_metrics if r["baseboard_m"] is not None]
        )

        result["totals"] = {
            "num_rooms": len(room_metrics),
            "total_area_m2": total_area_m2 if total_area_m2 > 0 else None,
            "total_perimeter_m": total_perimeter_m if total_perimeter_m > 0 else None,
            "total_baseboard_m": total_baseboard_m if total_baseboard_m > 0 else None,
        }

        # 7. ויזואליזציה
        try:
            overlay = create_rooms_overlay(original_image, room_regions, room_metrics)
            result["visualizations"]["overlay"] = overlay
            result["visualizations"]["masks"] = {
                r["room_id"]: room_regions[i]["mask"]
                for i, r in enumerate(room_metrics)
            }
        except Exception as viz_err:
            result["limitations"].append(f"שגיאה ביצירת ויזואליזציה: {str(viz_err)}")
            result["visualizations"]["overlay"] = None
            result["visualizations"]["masks"] = {}

        result["rooms"] = room_metrics
        result["success"] = True

        # הוסף מגבלות כלליות
        if meters_per_pixel is None:
            result["limitations"].append("אין קנה מידה - כל המדידות בפיקסלים בלבד")

    except Exception as e:
        result["success"] = False
        result["limitations"].append(f"שגיאה: {str(e)}")
        result["debug"]["error"] = str(e)

    return result


# ==========================================
# שלב 7: ויזואליזציה
# ==========================================


def create_rooms_overlay(
    original_image: np.ndarray, room_regions: List[Dict], room_metrics: List[Dict]
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
                mask_bool = region["mask"] > 0
                overlay[mask_bool] = color

                # כתוב טקסט במרכז
                cx, cy = metrics["center"]

                if metrics["area_m2"] is not None:
                    text = f"#{metrics['room_id']}\n{metrics['area_m2']:.1f} מ\"ר"
                else:
                    text = f"#{metrics['room_id']}\n{metrics['area_px']} px"

                if metrics.get("matched_name"):
                    text = f"{metrics['matched_name']}\n{metrics['area_m2']:.1f} מ\"ר"

                # רקע לבן לטקסט (קריא יותר)
                lines = text.split("\n")
                y_offset = cy - 15 * len(lines) // 2

                for line in lines:
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                        0
                    ]
                    cv2.rectangle(
                        overlay,
                        (cx - text_size[0] // 2 - 5, y_offset - text_size[1] - 5),
                        (cx + text_size[0] // 2 + 5, y_offset + 5),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        overlay,
                        line,
                        (cx - text_size[0] // 2, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
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
