import cv2
import numpy as np
import fitz
from typing import Tuple, Dict, Optional
from preprocessing import apply_crop
import os
import gc
import re

# ==========================================
# פונקציות עזר גלובליות לחישוב מדויק
# ==========================================


def parse_scale(scale_input) -> Optional[int]:
    """
    מנתח קנה מידה ומחזיר denominator (1:50 -> 50)

    🆕 v2.2: תמיכה מלאה במספרים, dict, וstrings

    Args:
        scale_input: יכול להיות:
            - str: "1:50", "קנ\\"מ 1:100", "SCALE 1:200"
            - int: 50, 100, 200
            - float: 50.0, 100.5
            - dict: {"value": 50} או {"value": "1:50"}
            - None

    Returns:
        int: denominator (10-500), או None אם לא תקין

    Examples:
        parse_scale("1:50") -> 50
        parse_scale(100) -> 100
        parse_scale({"value": 200}) -> 200
        parse_scale({"value": "1:75"}) -> 75
        parse_scale(0) -> None
        parse_scale("invalid") -> None
    """
    # Case 1: None או ערך ריק
    if scale_input is None:
        return None

    # Case 2: dict עם "value"
    if isinstance(scale_input, dict):
        if "value" in scale_input:
            # רקורסיה על הערך שבפנים
            return parse_scale(scale_input["value"])
        else:
            return None

    # Case 3: int או float
    if isinstance(scale_input, (int, float)):
        # בדיקת תקינות
        if scale_input <= 0:
            return None

        # המרה ל-int עם עיגול
        scale_int = int(round(scale_input))

        # בדיקת טווח סביר
        if 10 <= scale_int <= 500:
            return scale_int
        else:
            return None

    # Case 4: string - ניתוח עם regex
    if isinstance(scale_input, str):
        text = str(scale_input).strip()

        if not text:
            return None

        patterns = [
            r"1\s*:\s*(\d+)",  # 1:50
            r'קנ["\']מ\s*1\s*:\s*(\d+)',  # קנ"מ 1:50
            r"SCALE\s*1\s*:\s*(\d+)",  # SCALE 1:50
            r"(?:^|\s)1/(\d+)(?:\s|$)",  # 1/50
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    denominator = int(match.group(1))
                    if 10 <= denominator <= 500:
                        return denominator
                except (ValueError, IndexError):
                    continue

        return None

    # Case 5: טיפוס לא נתמך
    return None


def evaluate_flooring_mask_quality(mask: Optional[np.ndarray]) -> Dict:
    """
    מחשב מדד איכות בסיסי למסכת ריצוף (רעש/דלילות).
    """
    if mask is None or mask.size == 0:
        return {
            "quality_score": 0.0,
            "coverage_ratio": 0.0,
            "small_component_ratio": 1.0,
            "component_count": 0,
            "small_component_threshold_px": 0,
            "status": "empty",
        }

    area = int(cv2.countNonZero(mask))
    if area == 0:
        return {
            "quality_score": 0.0,
            "coverage_ratio": 0.0,
            "small_component_ratio": 1.0,
            "component_count": 0,
            "small_component_threshold_px": 0,
            "status": "empty",
        }

    h, w = mask.shape[:2]
    total_px = float(h * w)
    coverage_ratio = area / total_px

    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    component_count = max(0, num_labels - 1)
    small_threshold = max(20, int(0.0002 * total_px))
    small_area_total = 0
    for idx in range(1, num_labels):
        comp_area = stats[idx, cv2.CC_STAT_AREA]
        if comp_area < small_threshold:
            small_area_total += comp_area

    small_component_ratio = small_area_total / float(area) if area else 1.0

    quality_score = 1.0 - min(1.0, small_component_ratio * 1.2)
    if coverage_ratio < 0.001:
        quality_score *= 0.6
    if coverage_ratio > 0.6:
        quality_score *= 0.7

    return {
        "quality_score": float(max(0.0, min(1.0, quality_score))),
        "coverage_ratio": float(coverage_ratio),
        "small_component_ratio": float(small_component_ratio),
        "component_count": int(component_count),
        "small_component_threshold_px": int(small_threshold),
        "status": "ok",
    }


def detect_paper_size_mm(doc_page) -> Dict:
    """
    מזהה גודל נייר ISO (A0-A4) על בסיס גודל עמוד PDF

    Returns:
        {'detected_size': 'A1', 'width_mm': 594.0, 'height_mm': 841.0,
         'error_mm': 2.5, 'confidence': 0.95}
    """
    ISO_SIZES = {
        "A0": (841, 1189),
        "A1": (594, 841),
        "A2": (420, 594),
        "A3": (297, 420),
        "A4": (210, 297),
    }

    # המרה מ-points ל-mm
    rect = doc_page.rect
    width_mm = rect.width * 25.4 / 72
    height_mm = rect.height * 25.4 / 72

    # מיון (טיפול ב-landscape)
    dims_sorted = tuple(sorted([width_mm, height_mm]))

    # חיפוש התאמה קרובה
    best_match = None
    best_error = float("inf")

    for size_name, (w, h) in ISO_SIZES.items():
        iso_sorted = tuple(sorted([w, h]))
        error = np.sqrt(
            (dims_sorted[0] - iso_sorted[0]) ** 2
            + (dims_sorted[1] - iso_sorted[1]) ** 2
        )
        if error < best_error:
            best_error = error
            best_match = size_name

    confidence = max(0, 1 - (best_error / 50))

    return {
        "detected_size": best_match if best_error < 30 else "unknown",
        "width_mm": width_mm,
        "height_mm": height_mm,
        "error_mm": best_error,
        "confidence": confidence,
    }


def compute_skeleton_length_px(skeleton: np.ndarray) -> float:
    """
    🆕 v2.2: חישוב גיאומטרי מדויק של אורך skeleton

    שיטה: ספירת חיבורים לשכנים (neighbor-based)
    - חיבור אופקי/אנכי = 1.0 פיקסל
    - חיבור אלכסוני = √2 ≈ 1.414 פיקסלים

    בדיקה חד-כיוונית (ימין, למטה, אלכסונים למטה) כדי למנוע ספירה כפולה.

    Args:
        skeleton: תמונה בינארית של skeleton (grayscale או binary)

    Returns:
        float: אורך ב-פיקסלים (geometric)

    Note:
        גרסה קודמת (v2.1) השתמשה ב: count × 1.12
        גרסה זו (v2.2) מדויקת יותר לקירות אלכסוניים
    """
    if skeleton is None or skeleton.size == 0:
        return 0.0

    # המרה ל-grayscale אם צריך
    if len(skeleton.shape) == 3:
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)

    # המרה לבינארי
    skeleton_binary = (skeleton > 127).astype(np.uint8)

    h, w = skeleton_binary.shape

    # אופטימיזציה: NumPy vectorization במקום לולאה
    # זה פי 100 יותר מהיר מאשר לולאה ב-Python!

    # חיבורים אופקיים: (x, y) → (x+1, y)
    horizontal = np.sum(skeleton_binary[:, :-1] & skeleton_binary[:, 1:])

    # חיבורים אנכיים: (x, y) → (x, y+1)
    vertical = np.sum(skeleton_binary[:-1, :] & skeleton_binary[1:, :])

    # חיבורים אלכסוניים ימין-למטה: (x, y) → (x+1, y+1)
    diag_rd = np.sum(skeleton_binary[:-1, :-1] & skeleton_binary[1:, 1:])

    # חיבורים אלכסוניים שמאל-למטה: (x, y) → (x-1, y+1)
    # זה בעצם: (x+1, y) → (x, y+1) בכיוון הפוך
    diag_ld = np.sum(skeleton_binary[:-1, 1:] & skeleton_binary[1:, :-1])

    # חישוב אורך כולל
    import math

    sqrt_2 = math.sqrt(2)
    total_length = float(horizontal + vertical + (diag_rd + diag_ld) * sqrt_2)

    return total_length


class FloorPlanAnalyzer:
    """
    גרסה 2.0 - Multi-pass filtering עם confidence scoring
    """

    def __init__(self):
        self.debug_layers = {}  # לשמירת שכבות ביניים

    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """יצירת skeleton ללא צורך ב-ximgproc"""
        # Ensure we have a valid numpy array of an image
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(img).__name__}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        skeleton = np.zeros_like(binary, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            if cv2.countNonZero(binary) == 0:
                break

        return skeleton

    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 2000) -> np.ndarray:
        """
        המרת PDF לתמונה במלוא הרזולוציה (ללא חיתוך)

        Args:
            pdf_path: נתיב ל-PDF
            target_max_dim: מקסימום פיקסלים (4000 = רזולוציה גבוהה)

        Returns:
            תמונה BGR מלאה
        """
        doc = fitz.open(pdf_path)
        page = doc[0]

        # חישוב scale לרזולוציה גבוהה (עד 4000px)
        # מוגבל ל-2.0 כדי לא להגזים
        scale = min(2.0, target_max_dim / max(page.rect.width, page.rect.height))

        # יצירת pixmap ללא crop (clip=None)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
            clip=None,  # ← קריטי! לא לחתוך כלום
        )

        # המרה ל-numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # המרה ל-BGR (OpenCV format)
        if pix.n == 3:  # RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:  # Grayscale
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:  # RGBA או אחר
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        doc.close()
        del pix  # שחרור זיכרון

        return img_bgr

    # ==========================================
    # PASS 1: זיהוי טקסט ברור (Obvious Text)
    # ==========================================
    def _detect_obvious_text(self, gray: np.ndarray) -> np.ndarray:
        """
        מזהה רק טקסט שאנחנו 100% בטוחים שהוא טקסט
        קריטריונים נוקשים מאוד
        """
        h, w = gray.shape
        text_mask = np.zeros_like(gray)

        # Threshold גבוה יותר - רק דברים כהים מאוד
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # ניקוי רעשים זעירים
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # זיהוי רכיבים
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            aspect = max(bw, bh) / (min(bw, bh) + 1)
            density = area / (bw * bh) if (bw * bh) > 0 else 0

            is_obvious_text = False

            # כלל 1: תווים בודדים קטנים מאוד
            if area < 200 and 0.3 < aspect < 3.0 and density > 0.3:
                is_obvious_text = True

            # כלל 2: מילים קצרות
            if area < 800 and 2.0 < aspect < 8.0 and bh < 30:
                is_obvious_text = True

            # כלל 3: מספרים קטנים (מרובעים)
            if area < 150 and 0.7 < aspect < 1.5:
                is_obvious_text = True

            # חריגות - בטוח שזה לא טקסט
            if bw > w * 0.5 or bh > h * 0.5:  # דבר גדול מדי
                is_obvious_text = False

            if area > 5000:  # גדול מדי
                is_obvious_text = False

            if is_obvious_text:
                padding = 3  # ריפוד מינימלי
                text_mask[
                    max(0, y - padding) : min(h, y + bh + padding),
                    max(0, x - padding) : min(w, x + bw + padding),
                ] = 255

        return text_mask

    # ==========================================
    # PASS 2: זיהוי סמלים וכותרות
    # ==========================================
    def _detect_symbols_and_labels(self, gray: np.ndarray) -> np.ndarray:
        """
        מזהה סמלים, ציונים, וכותרות
        פחות נוקשה מ-Pass 1 אבל עדיין זהיר
        """
        h, w = gray.shape
        symbols_mask = np.zeros_like(gray)

        _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

        # חיבור אופקי בלבד (לכותרות)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        connected = cv2.dilate(binary, kernel_h, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            connected, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            aspect = max(bw, bh) / (min(bw, bh) + 1)

            is_symbol = False

            # כותרות ארוכות ודקות
            if 5.0 < aspect < 25.0 and bh < 50 and area < 4000:
                is_symbol = True

            # תיבות טקסט קטנות (סמלי מקרא וכו')
            if area < 1500 and 0.8 < aspect < 1.5:
                is_symbol = True

            # חריגות
            if bw > w * 0.7 or bh > h * 0.7:
                is_symbol = False

            if is_symbol:
                padding = 5
                symbols_mask[
                    max(0, y - padding) : min(h, y + bh + padding),
                    max(0, x - padding) : min(w, x + bw + padding),
                ] = 255

        return symbols_mask

    # ==========================================
    # PASS 3: זיהוי מספרי חדרים
    # ==========================================
    def _detect_room_numbers(
        self, gray: np.ndarray, walls_estimate: np.ndarray
    ) -> np.ndarray:
        """
        מזהה מספרי חדרים בלבד - באמצע חדרים
        משתמש במידע על הקירות כדי למצוא חדרים
        """
        h, w = gray.shape
        numbers_mask = np.zeros_like(gray)

        # מצא אזורים סגורים (חדרים)
        walls_dilated = cv2.dilate(
            walls_estimate, np.ones((3, 3), np.uint8), iterations=2
        )
        rooms = cv2.bitwise_not(walls_dilated)

        # מצא רכיבי חדרים
        num_rooms, room_labels, room_stats, room_centroids = (
            cv2.connectedComponentsWithStats(rooms, connectivity=8)
        )

        # זיהוי מספרים קטנים
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))  # אנכי
        connected = cv2.dilate(binary, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            connected, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cx, cy = centroids[i]

            aspect = max(bw, bh) / (min(bw, bh) + 1)

            # תנאים למספר חדר
            if area < 600 and bw < 50 and bh < 60:
                # בדוק אם זה באמצע חדר
                room_id = room_labels[int(cy), int(cx)]
                if room_id > 0:  # נמצא בתוך חדר
                    room_area = room_stats[room_id, cv2.CC_STAT_AREA]
                    if room_area > 5000:  # חדר גדול מספיק
                        padding = 6
                        numbers_mask[
                            max(0, y - padding) : min(h, y + bh + padding),
                            max(0, x - padding) : min(w, x + bw + padding),
                        ] = 255

        return numbers_mask

    # ==========================================
    # SMART WALL DETECTION
    # ==========================================
    def _smart_wall_detection(
        self, gray: np.ndarray, text_mask_combined: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        זיהוי קירות משופר עם confidence scoring
        מחזיר: (wall_mask, confidence_map)
        """
        h, w = gray.shape

        # זיהוי בסיסי
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        binary_no_text = cv2.subtract(binary, text_mask_combined)

        # ניקוי
        cleaned = cv2.morphologyEx(
            binary_no_text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # זיהוי רכיבים
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=4
        )

        wall_mask = np.zeros_like(gray)
        confidence_map = np.zeros_like(gray, dtype=np.float32)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            if bw == 0 or bh == 0:
                continue

            density = area / (bw * bh)
            aspect = max(bw, bh) / min(bw, bh)

            # חישוב confidence (0.0 - 1.0)
            confidence = 0.0

            # מדדים חיוביים
            if area > 100:  # ← יותר רגיש! (היה 150)
                confidence += 0.2

            if density > 0.3:  # ← יותר רגיש! (היה 0.35)
                confidence += 0.3

            if aspect > 2.5:  # ← יותר רגיש! (היה 3.0)
                confidence += 0.3

            if area > 400 and aspect > 4.0:  # ← יותר רגיש! (היה 500, 5.0)
                confidence += 0.2

            # מדדים שליליים
            # הוסרה בדיקת המסגרת - הייתה מסננת קירות חיצוניים!

            if area < 80:  # ← יותר סלחן! (היה 100)
                confidence *= 0.3

            if density < 0.2:  # ← יותר סלחן! (היה 0.25)
                confidence *= 0.5

            # שמירה רק אם confidence > 0.3 (← יותר סלחן! היה 0.4)
            if confidence > 0.3:
                mask = (labels == i).astype(np.uint8) * 255
                wall_mask = cv2.bitwise_or(wall_mask, mask)
                confidence_map[labels == i] = confidence

        # החלקה סופית
        final_walls = cv2.morphologyEx(
            wall_mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8)
        )

        return final_walls, confidence_map

    def _detect_walls_hough(
        self, gray: np.ndarray, text_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        זיהוי קירות באמצעות Hough Lines Transform
        מזהה קווים ישרים - עובד מצוין עם קירות מקוטעים!
        """
        h, w = gray.shape

        # הסרת טקסט
        clean = cv2.subtract(gray, text_mask)

        # הגברת ניגודיות
        clean = cv2.convertScaleAbs(clean, alpha=1.3, beta=10)

        # Sharpen
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        clean = cv2.filter2D(clean, -1, kernel_sharp)

        # זיהוי קצוות (Multi-scale)
        edges1 = cv2.Canny(clean, 30, 100)
        edges2 = cv2.Canny(clean, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        # הסרת טקסט מקצוות
        edges = cv2.subtract(edges, text_mask)

        # Hough Lines Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=25,
            maxLineGap=20,
        )

        # מסכת קירות + confidence
        walls_mask = np.zeros_like(gray)
        confidence_map = np.zeros_like(gray, dtype=np.float32)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # חישוב מאפיינים
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                # Confidence
                confidence = 0.0

                # אורך
                if length > 50:
                    confidence += 0.3
                elif length > 30:
                    confidence += 0.2
                else:
                    confidence += 0.1

                # כיוון
                is_horizontal = abs(angle) < 5 or abs(abs(angle) - 180) < 5
                is_vertical = abs(abs(angle) - 90) < 5

                if is_horizontal or is_vertical:
                    confidence += 0.5
                elif abs(angle) < 15 or abs(abs(angle) - 90) < 15:
                    confidence += 0.3

                # מיקום (קירות חיצוניים + טקסט בצדדים)
                margin = 80  # ← 30→80 (מרווח גדול יותר!)
                is_edge = (
                    x1 < margin
                    or x2 < margin
                    or x1 > w - margin
                    or x2 > w - margin
                    or y1 < margin
                    or y2 < margin
                    or y1 > h - margin
                    or y2 > h - margin
                )

                if is_edge:
                    confidence *= 0.3  # ← 0.7→0.3 (יותר קפדני!)

                # בדיקת צפיפות - סינון אזורי טקסט
                # באזורי טקסט יש הרבה קווים קטנים קרובים
                roi_size = 50
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                x_start = max(0, x_center - roi_size)
                x_end = min(w, x_center + roi_size)
                y_start = max(0, y_center - roi_size)
                y_end = min(h, y_center + roi_size)

                roi = edges[y_start:y_end, x_start:x_end]
                if roi.size > 0:
                    density = np.count_nonzero(roi) / roi.size
                    # אם צפיפות גבוהה (>0.15) = כנראה טקסט
                    if density > 0.15:
                        confidence *= 0.4

                # ציור אם confidence > 0.4
                if confidence > 0.4:
                    thickness = 3 if confidence > 0.7 else 2
                    cv2.line(walls_mask, (x1, y1), (x2, y2), 255, thickness)
                    cv2.line(confidence_map, (x1, y1), (x2, y2), confidence, thickness)

        # החלקה
        kernel = np.ones((4, 4), np.uint8)
        walls_mask = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel)

        return walls_mask, confidence_map

    # ==========================================
    # MAIN PROCESSING
    # ==========================================
    def process_file(self, pdf_path: str, save_debug=False, crop_bbox=None):
        """
        עיבוד מרכזי עם multi-pass filtering
        """
        image_full = self.pdf_to_image(pdf_path)
        full_gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
        full_h, full_w = full_gray.shape

        # אם המשתמש בחר אזור חיתוך – ננתח גיאומטריה רק עליו, אבל נשמור טקסט/מקרא מהדף המלא
        image_proc, crop_meta = apply_crop(image_full, crop_bbox)

        gray = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # === MULTI-PASS TEXT DETECTION ===

        # Pass 1: טקסט ברור
        text_obvious = self._detect_obvious_text(gray)
        self.debug_layers["text_obvious"] = text_obvious.copy()

        # Pass 2: סמלים וכותרות
        symbols = self._detect_symbols_and_labels(gray)
        self.debug_layers["symbols"] = symbols.copy()

        # הערכה ראשונית של קירות (לצורך Pass 3)
        _, binary_temp = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        walls_estimate = cv2.subtract(
            binary_temp, cv2.bitwise_or(text_obvious, symbols)
        )
        walls_estimate = cv2.morphologyEx(
            walls_estimate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        # Pass 3: מספרי חדרים
        room_numbers = self._detect_room_numbers(gray, walls_estimate)
        self.debug_layers["room_numbers"] = room_numbers.copy()

        # איחוד חכם של כל הטקסט
        text_mask_combined = cv2.bitwise_or(text_obvious, symbols)
        text_mask_combined = cv2.bitwise_or(text_mask_combined, room_numbers)

        # ניפוח סופי מתון
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask_combined = cv2.dilate(text_mask_combined, kernel_final, iterations=1)

        self.debug_layers["text_combined"] = text_mask_combined.copy()

        # === HOUGH LINES WALL DETECTION ===
        # החלפה: _smart_wall_detection → _detect_walls_hough
        final_walls, confidence_map = self._detect_walls_hough(gray, text_mask_combined)
        self.debug_layers["walls"] = final_walls.copy()
        self.debug_layers["confidence"] = confidence_map.copy()

        # === הפרדת חומרים ===
        kernel = np.ones((6, 6), np.uint8)
        concrete = cv2.dilate(
            cv2.erode(final_walls, kernel, iterations=1), kernel, iterations=2
        )
        blocks_mask = cv2.subtract(
            final_walls, concrete
        )  # Renamed to avoid shadowing with text blocks

        # === ריצוף ===
        edges = cv2.Canny(gray, 50, 150)
        flooring = cv2.subtract(
            cv2.subtract(edges, cv2.dilate(final_walls, np.ones((9, 9)))),
            text_mask_combined,
        )

        # === תמונת Debug משופרת ===
        debug_img = None
        if save_debug:
            debug_img = image_proc.copy()
            overlay = debug_img.copy()

            # שכבות צבעוניות
            overlay[text_obvious > 0] = [255, 100, 0]  # כתום - טקסט ברור
            overlay[symbols > 0] = [255, 200, 0]  # צהוב - סמלים
            overlay[room_numbers > 0] = [255, 0, 255]  # סגול - מספרי חדרים
            overlay[final_walls > 0] = [0, 255, 0]  # ירוק - קירות

            # הוספת confidence (קירות בהירים יותר = ביטחון גבוה)
            confidence_visual = (confidence_map * 255).astype(np.uint8)
            confidence_visual = cv2.applyColorMap(confidence_visual, cv2.COLORMAP_HOT)
            overlay = cv2.addWeighted(overlay, 0.7, confidence_visual, 0.3, 0)

            cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0, debug_img)
            # ← הוסף את זה כאן:
        # שמירת PDF bytes למטרות Google Vision OCR
        pdf_bytes_for_ocr = None
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes_for_ocr = f.read()
            except Exception as e:
                print(f"⚠️ Failed to read PDF bytes: {e}")
                pdf_bytes_for_ocr = None
        # === חישובים ונתונים ===
        skel = self._skeletonize(final_walls)
        pix = cv2.countNonZero(skel)

        meta = {
            "plan_name": os.path.basename(pdf_path),
            "raw_text": "",
            "pdf_bytes": pdf_bytes_for_ocr,
        }
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes_for_ocr = f.read()
            except:
                pdf_bytes_for_ocr = None
        else:
            pdf_bytes_for_ocr = None
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]

            # Extract full text (up to 20000 chars, not 3000)
            full_text = page.get_text()
            meta["raw_text_full"] = (
                full_text[:20000] if len(full_text) > 20000 else full_text
            )
            meta["raw_text"] = full_text[:3000]  # Keep for backward compatibility

            # Extract text blocks with bounding boxes
            try:
                text_blocks = page.get_text(
                    "blocks"
                )  # Renamed to avoid shadowing blocks_mask
                # Sort blocks by y position (top to bottom) then x (left to right)
                sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

                # Build structured block list
                meta["raw_blocks"] = [
                    {
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "text": b[4].strip(),
                        "block_type": int(b[5]) if len(b) > 5 else 0,
                        "block_no": int(b[6]) if len(b) > 6 else 0,
                    }
                    for b in sorted_blocks
                    if len(b) >= 5 and b[4].strip()
                ]

                # Build normalized text from sorted blocks
                normalized_text = "\n".join([b["text"] for b in meta["raw_blocks"]])
                meta["normalized_text"] = (
                    normalized_text[:20000]
                    if len(normalized_text) > 20000
                    else normalized_text
                )

            except Exception as block_err:
                # Fallback if block extraction fails
                meta["raw_blocks"] = []
                meta["normalized_text"] = meta["raw_text_full"]

            doc.close()
        except Exception as e:
            # Fallback to empty if PDF reading fails
            meta["raw_text_full"] = ""
            meta["raw_blocks"] = []
            meta["normalized_text"] = ""

        # Guard: Ensure masks are numpy arrays before skeletonization
        if not isinstance(blocks_mask, np.ndarray):
            raise TypeError(
                f"blocks_mask must be numpy array, got {type(blocks_mask).__name__}"
            )
        if not isinstance(concrete, np.ndarray):
            raise TypeError(
                f"concrete must be numpy array, got {type(concrete).__name__}"
            )

        flooring_quality = evaluate_flooring_mask_quality(flooring)

        meta.update(
            {
                "pixels_concrete": cv2.countNonZero(self._skeletonize(concrete)),
                "pixels_blocks": cv2.countNonZero(self._skeletonize(blocks_mask)),
                "pixels_flooring_area": cv2.countNonZero(flooring),
                "flooring_quality": flooring_quality,
                "flooring_confidence": flooring_quality.get("quality_score", 0.0),
                "confidence_avg": (
                    float(np.mean(confidence_map[final_walls > 0]))
                    if np.any(final_walls > 0)
                    else 0.0
                ),
                "text_removed_pixels": cv2.countNonZero(text_mask_combined),
            }
        )

        gc.collect()
        # === חישובי מדידה מדויקים (Stage 1 + 2) ===
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]

            # Stage 1: זיהוי גודל נייר
            paper_info = detect_paper_size_mm(page)
            meta["paper_size_detected"] = paper_info["detected_size"]
            meta["paper_mm"] = {
                "width": paper_info["width_mm"],
                "height": paper_info["height_mm"],
            }
            meta["paper_detection_error_mm"] = paper_info["error_mm"]
            meta["paper_detection_confidence"] = paper_info["confidence"]

            # חילוץ קנה מידה מטקסט
            scale_denom = None

            # 1. אם כבר חולץ scale_text קודם
            if meta.get("scale"):
                scale_denom = parse_scale(meta["scale"])

            # 2. אם לא, נסה מהטקסט הגולמי
            if not scale_denom and meta.get("raw_text"):
                scale_denom = parse_scale(meta["raw_text"])

            meta["scale_denominator"] = scale_denom

            # חישוב mm_per_pixel (מבוסס על גודל הדף המלא, לא על אזור החיתוך)
            if image_full is not None:
                mm_per_pixel_x = paper_info["width_mm"] / full_w
                mm_per_pixel_y = paper_info["height_mm"] / full_h
                mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2

                meta["mm_per_pixel"] = mm_per_pixel
                meta["mm_per_pixel_x"] = mm_per_pixel_x
                meta["mm_per_pixel_y"] = mm_per_pixel_y
                meta["image_size_px"] = {"width": full_w, "height": full_h}

                # מידע על אזור הניתוח (אם יש crop)
                meta["analysis_crop"] = crop_meta
                meta["analysis_image_size_px"] = {"width": w, "height": h}

                # חישוב meters_per_pixel
                if scale_denom:
                    meters_per_pixel = (mm_per_pixel * scale_denom) / 1000
                    meta["meters_per_pixel"] = meters_per_pixel
                    meta["meters_per_pixel_x"] = (mm_per_pixel_x * scale_denom) / 1000
                    meta["meters_per_pixel_y"] = (mm_per_pixel_y * scale_denom) / 1000
                    # 🆕 v2.2: Logical measurement confidence
                    # מבוסס על scale validity + aspect ratio (לא רק paper size!)

                    aspect_ratio = (
                        mm_per_pixel_x / mm_per_pixel_y if mm_per_pixel_y > 0 else 1.0
                    )

                    # aspect_ratio_ok: בטווח 0.95-1.05 (כמעט ריבועי)
                    aspect_ratio_ok = 0.95 <= aspect_ratio <= 1.05

                    # aspect_ratio_bad: מחוץ ל-0.90-1.10 (מעוות מאוד)
                    aspect_ratio_bad = aspect_ratio < 0.90 or aspect_ratio > 1.10

                    # scale_ok: יש denominator תקין
                    scale_ok = scale_denom is not None and scale_denom > 0

                    # חישוב confidence לוגי
                    if not scale_ok:
                        # אין scale → אי אפשר למדוד
                        measurement_confidence = 0.0
                    elif aspect_ratio_ok:
                        # aspect מעולה → confidence גבוה (גם ללא paper size)
                        measurement_confidence = 0.88
                        # modifier קטן מ-paper confidence (±0.02)
                        paper_modifier = (paper_info["confidence"] - 0.5) * 0.04
                        measurement_confidence += paper_modifier
                        measurement_confidence = max(
                            0.85, min(0.92, measurement_confidence)
                        )
                    elif aspect_ratio_bad:
                        # aspect גרוע → confidence נמוך
                        measurement_confidence = 0.30
                        paper_modifier = (paper_info["confidence"] - 0.5) * 0.02
                        measurement_confidence += paper_modifier
                        measurement_confidence = max(
                            0.20, min(0.40, measurement_confidence)
                        )
                    else:
                        # aspect בינוני → confidence בינוני
                        measurement_confidence = 0.60
                        paper_modifier = (paper_info["confidence"] - 0.5) * 0.03
                        measurement_confidence += paper_modifier
                        measurement_confidence = max(
                            0.50, min(0.70, measurement_confidence)
                        )

                    meta["measurement_confidence"] = measurement_confidence
                    meta["measurement_confidence_factors"] = {
                        "scale_ok": scale_ok,
                        "aspect_ratio": round(aspect_ratio, 3),
                        "aspect_ratio_ok": aspect_ratio_ok,
                        "aspect_ratio_bad": aspect_ratio_bad,
                        "paper_confidence": paper_info["confidence"],
                    }
                else:
                    meta["meters_per_pixel"] = None
                    meta["meters_per_pixel_x"] = None
                    meta["meters_per_pixel_y"] = None
                    meta["measurement_confidence"] = 0.0

            # Stage 2: מדידת אורך skeleton
            skeleton_length_px = compute_skeleton_length_px(skel)
            meta["wall_length_total_px"] = skeleton_length_px

            if meta.get("meters_per_pixel"):
                wall_length_m = skeleton_length_px * meta["meters_per_pixel"]
                meta["wall_length_total_m"] = wall_length_m
                meta["wall_length_method"] = "skeleton_based"
            else:
                meta["wall_length_total_m"] = None
                meta["wall_length_method"] = "insufficient_data"

            doc.close()

        except Exception as e:
            # אם נכשל - לא לשבור את כל התהליך
            meta["measurement_error"] = str(e)
            meta["meters_per_pixel"] = None
            meta["wall_length_total_m"] = None
        return (
            pix,
            skel,
            final_walls,
            image_proc,
            meta,
            concrete,
            blocks_mask,
            flooring,
            debug_img,
        )

    # ==========================================
    # פונקציות לזיהוי מקרא אוטומטי
    # ==========================================

    def auto_detect_legend(self, image: np.ndarray) -> Optional[tuple]:
        """
        מזהה אוטומטית את המקרא בתוכנית

        Args:
            image: תמונת התוכנית (BGR)

        Returns:
            (x, y, width, height) או None אם לא נמצא
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        h, w = gray.shape

        # מקרא בדרך כלל בפינה (למעלה או למטה)
        # נבדוק את 4 הפינות + 2 צדדים
        regions = {
            "top_left": (0, 0, w // 3, h // 4),
            "top_right": (2 * w // 3, 0, w // 3, h // 4),
            "bottom_left": (0, 3 * h // 4, w // 3, h // 4),
            "bottom_right": (2 * w // 3, 3 * h // 4, w // 3, h // 4),
            "left_middle": (0, h // 3, w // 4, h // 3),
            "right_middle": (3 * w // 4, h // 3, w // 4, h // 3),
        }

        best_region = None
        best_score = 0

        for name, (x, y, rw, rh) in regions.items():
            # וידוא שהאזור בגבולות
            if x + rw > w or y + rh > h:
                continue

            roi = gray[y : y + rh, x : x + rw]

            # חישוב ציון
            score = self._score_legend_region(roi)

            if score > best_score:
                best_score = score
                best_region = (x, y, rw, rh)

        # סף מינימום - אם הציון מעל 0.4 → סביר שזה מקרא
        if best_score > 0.4:
            return best_region

        return None

    def _score_legend_region(self, roi: np.ndarray) -> float:
        """
        מחשב ציון ל-ROI - האם זה מקרא?

        Args:
            roi: אזור לבדיקה (grayscale)

        Returns:
            ציון 0.0-1.0 (גבוה יותר = סיכוי גבוה יותר למקרא)
        """
        score = 0.0

        if roi.size == 0:
            return 0.0

        # 1. זיהוי מסגרת/קופסה (0.3 נקודות)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
        )

        if lines is not None and len(lines) > 8:
            # יש הרבה קווים → כנראה מסגרת
            score += 0.3

        # 2. צפיפות טקסט (0.4 נקודות)
        _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num_labels > 1:
            # ספירת רכיבים קטנים (טקסט/סמלים)
            areas = stats[1:, cv2.CC_STAT_AREA]
            small_components = np.sum((areas > 20) & (areas < 500))

            # מקרא בדרך כלל עם 20-100 רכיבים קטנים
            if small_components > 20:
                score += 0.4
            elif small_components > 10:
                score += 0.2

        # 3. בהירות ממוצעת (0.3 נקודות)
        # מקרא בדרך כלל בהיר (קווים דקים, הרבה רקע לבן)
        mean_brightness = np.mean(roi)
        if mean_brightness > 200:  # מאוד בהיר
            score += 0.3
        elif mean_brightness > 180:  # בהיר
            score += 0.2

        return min(1.0, score)  # מקסימום 1.0

    def extract_legend_region(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        חותך את אזור המקרא מהתמונה

        Args:
            image: תמונה מלאה
            bbox: (x, y, width, height)

        Returns:
            תמונת המקרא החתוכה
        """
        x, y, w, h = bbox
        return image[y : y + h, x : x + w].copy()

    # ==========================================
    # PHASE 2: זיהוי סוג תוכנית אוטומטי
    # ==========================================

    def detect_plan_type(self, image: np.ndarray, metadata: dict = None) -> dict:
        """
        מזהה אוטומטית את סוג התוכנית

        Args:
            image: תמונת התוכנית (BGR)
            metadata: מטא-דאטה (אם יש ניתוח מקרא)

        Returns:
            {
                'plan_type': 'קירות' / 'תקרה' / 'ריצוף' / 'חשמל' / 'אחר',
                'confidence': 0-100,
                'features': {...},
                'reasoning': 'הסבר'
            }
        """
        # שיטה 1: אם יש ניתוח מקרא - זו העדיפות הראשונה!
        if metadata and "legend_analysis" in metadata:
            legend = metadata["legend_analysis"]
            if legend.get("plan_type") and legend.get("plan_type") != "אחר":
                return {
                    "plan_type": legend["plan_type"],
                    "confidence": legend.get("confidence", 90),
                    "method": "legend",
                    "reasoning": f"זוהה מהמקרא: {legend.get('legend_title', '')}",
                }

        # שיטה 2: ניתוח ויזואלי
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # חישוב מאפיינים
        features = {
            "line_density": self._calculate_line_density(gray),
            "text_ratio": self._calculate_text_ratio(gray),
            "has_hatching": self._detect_hatching(gray),
            "has_tiles": self._detect_tiles(gray),
            "pattern_type": self._detect_pattern_type(gray),
            "small_components_ratio": self._calculate_small_components_ratio(gray),
        }

        # חוקי זיהוי מבוססי מאפיינים
        scores = {"ריצוף": 0, "תקרה": 0, "קירות": 0, "חשמל": 0, "אחר": 0}

        # ניקוד לפי מאפיינים

        # ריצוף
        if features["has_tiles"]:
            scores["ריצוף"] += 40
        if features["pattern_type"] == "grid":
            scores["ריצוף"] += 30
        if features["line_density"] > 0.4:  # הרבה קווים
            scores["ריצוף"] += 20

        # תקרה
        if features["has_hatching"]:
            scores["תקרה"] += 50
        if features["pattern_type"] == "diagonal":
            scores["תקרה"] += 30
        if 0.25 < features["line_density"] < 0.4:  # בינוני
            scores["תקרה"] += 20

        # קירות
        if features["line_density"] > 0.3 and features["text_ratio"] < 0.15:
            scores["קירות"] += 40
        if features["pattern_type"] == "lines" and not features["has_hatching"]:
            scores["קירות"] += 30
        if 0.1 < features["line_density"] < 0.35:
            scores["קירות"] += 20

        # חשמל
        if features["small_components_ratio"] > 0.3:  # הרבה סמלים קטנים
            scores["חשמל"] += 40
        if features["line_density"] < 0.2:  # מעט קווים
            scores["חשמל"] += 30
        if features["text_ratio"] > 0.2:  # הרבה טקסט/סמלים
            scores["חשמל"] += 20

        # מציאת הסוג עם הניקוד הגבוה ביותר
        plan_type = max(scores, key=scores.get)
        confidence = min(100, scores[plan_type])

        # אם הביטחון נמוך מדי - "אחר"
        if confidence < 40:
            plan_type = "אחר"
            confidence = 50

        return {
            "plan_type": plan_type,
            "confidence": confidence,
            "features": features,
            "scores": scores,
            "method": "visual",
            "reasoning": f"ניתוח ויזואלי: {plan_type} (ציון: {scores[plan_type]})",
        }

    def _calculate_line_density(self, gray: np.ndarray) -> float:
        """
        מחשב אחוז פיקסלים שחורים (קווים) בתמונה

        Returns:
            0.0-1.0 (אחוז פיקסלים שחורים)
        """
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        black_pixels = np.count_nonzero(binary)
        total_pixels = binary.size

        return black_pixels / total_pixels

    def _calculate_text_ratio(self, gray: np.ndarray) -> float:
        """
        מחשב אחוז רכיבים קטנים (טקסט/סמלים) בתמונה

        Returns:
            0.0-1.0 (אחוז רכיבים קטנים)
        """
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num_labels <= 1:
            return 0.0

        # ספירת רכיבים קטנים (area < 100px)
        small_components = np.sum(stats[1:, cv2.CC_STAT_AREA] < 100)

        return small_components / max(1, num_labels - 1)

    def _calculate_small_components_ratio(self, gray: np.ndarray) -> float:
        """
        מחשב אחוז רכיבים קטנים מאוד (סמלים)

        Returns:
            0.0-1.0
        """
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num_labels <= 1:
            return 0.0

        # רכיבים קטנים מאוד (10 < area < 50)
        tiny_components = np.sum(
            (stats[1:, cv2.CC_STAT_AREA] > 10) & (stats[1:, cv2.CC_STAT_AREA] < 50)
        )

        return tiny_components / max(1, num_labels - 1)

    def _detect_hatching(self, gray: np.ndarray) -> bool:
        """
        מזהה קווים אלכסוניים (hatching) - אופייני לתקרות

        Returns:
            True אם יש hatching
        """
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough Lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
        )

        if lines is None or len(lines) < 10:
            return False

        # בדיקת זווית קווים
        diagonal_count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # חישוב זווית
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # קווים אלכסוניים: 30-60 מעלות
            if 30 < angle < 60 or 120 < angle < 150:
                diagonal_count += 1

        # אם יותר מ-30% מהקווים אלכסוניים → hatching
        return diagonal_count / len(lines) > 0.3

    def _detect_tiles(self, gray: np.ndarray) -> bool:
        """
        מזהה אריחים (grid pattern) - אופייני לריצוף

        Returns:
            True אם יש grid
        """
        # חיפוש קווים אופקיים ואנכיים חזקים
        edges = cv2.Canny(gray, 50, 150)

        # קווים אופקיים
        horizontal = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20
        )

        # קווים אנכיים
        vertical = cv2.HoughLinesP(
            edges, 1, np.pi / 2, threshold=100, minLineLength=100, maxLineGap=20
        )

        h_count = len(horizontal) if horizontal is not None else 0
        v_count = len(vertical) if vertical is not None else 0

        # אם יש הרבה קווים אופקיים ואנכיים → grid
        return h_count > 20 and v_count > 20

    def _detect_pattern_type(self, gray: np.ndarray) -> str:
        """
        מזהה סוג pattern בתמונה

        Returns:
            'grid' / 'diagonal' / 'lines' / 'mixed'
        """
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10
        )

        if lines is None or len(lines) < 5:
            return "none"

        # סיווג קווים לפי זווית
        horizontal = 0  # 0±15°
        vertical = 0  # 90±15°
        diagonal = 0  # 30-60° או 120-150°

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 15 or angle > 165:
                horizontal += 1
            elif 75 < angle < 105:
                vertical += 1
            elif 30 < angle < 60 or 120 < angle < 150:
                diagonal += 1

        total = len(lines)

        # החלטה
        if horizontal / total > 0.3 and vertical / total > 0.3:
            return "grid"
        elif diagonal / total > 0.4:
            return "diagonal"
        elif (horizontal + vertical) / total > 0.7:
            return "lines"
        else:
            return "mixed"

    # ==========================================
    # PHASE 3: פרמטרים אדפטיביים לפי סוג תוכנית
    # ==========================================

    def get_adaptive_parameters(self, plan_type: str) -> dict:
        """
        מחזיר פרמטרים מותאמים לסוג התוכנית

        Args:
            plan_type: 'קירות' / 'תקרה' / 'ריצוף' / 'חשמל' / 'אחר'

        Returns:
            dict עם פרמטרים לעיבוד
        """

        if plan_type == "קירות":
            return {
                "text_threshold": 200,
                "min_wall_length": 50,
                "max_text_area": 200,
                "wall_thickness_kernel": (6, 6),
                "text_dilation_kernel": (5, 5),
                "confidence_threshold": 0.5,
                "ignore_hatching": False,
                "edge_sensitivity": "medium",
                "description": "אופטימלי לזיהוי קירות, דלתות וחלונות",
            }

        elif plan_type == "תקרה":
            return {
                "text_threshold": 190,
                "min_wall_length": 30,
                "max_text_area": 300,
                "wall_thickness_kernel": (4, 4),
                "text_dilation_kernel": (7, 7),
                "confidence_threshold": 0.4,
                "ignore_hatching": True,
                "edge_sensitivity": "low",
                "description": "אופטימלי לתקרות עם hatching וסמלים",
            }

        elif plan_type == "ריצוף":
            return {
                "text_threshold": 180,
                "min_wall_length": 20,
                "max_text_area": 500,
                "wall_thickness_kernel": (3, 3),
                "text_dilation_kernel": (6, 6),
                "confidence_threshold": 0.3,
                "ignore_hatching": False,
                "edge_sensitivity": "high",
                "description": "אופטימלי לריצוף עם grid וכיתובים",
            }

        elif plan_type == "חשמל":
            return {
                "text_threshold": 210,
                "min_wall_length": 60,
                "max_text_area": 100,
                "wall_thickness_kernel": (5, 5),
                "text_dilation_kernel": (3, 3),
                "confidence_threshold": 0.6,
                "ignore_hatching": False,
                "edge_sensitivity": "low",
                "description": "אופטימלי לתוכניות חשמל עם סמלים",
            }

        else:  # 'אחר' / ברירת מחדל
            return {
                "text_threshold": 200,
                "min_wall_length": 50,
                "max_text_area": 200,
                "wall_thickness_kernel": (6, 6),
                "text_dilation_kernel": (5, 5),
                "confidence_threshold": 0.5,
                "ignore_hatching": False,
                "edge_sensitivity": "medium",
                "description": "פרמטרים סטנדרטיים",
            }


# ==========================================
# METADATA EXPORT
# ==========================================


def export_walls_to_metadata(
    self,
    thick_walls: np.ndarray,
    pdf_path: str,
    pixels_per_meter: float,
    scale_text: str = "1:50",
) -> str:
    """
    יצירת קובץ metadata מתוצאת זיהוי

    Args:
        thick_walls: מסכת קירות (binary)
        pdf_path: נתיב ל-PDF המקורי
        pixels_per_meter: כיול
        scale_text: טקסט סקייל

    Returns:
        נתיב לקובץ metadata שנוצר
    """
    from contech_metadata import (
        ContechMetadata,
        calculate_pdf_checksum,
        extract_walls_from_opencv_mask,
        get_metadata_filepath,
    )

    # חישוב checksum
    checksum = calculate_pdf_checksum(pdf_path)

    # יצירת metadata
    import os

    filename = os.path.basename(pdf_path)
    metadata = ContechMetadata(filename, checksum)

    # הוספת כיול
    metadata.pixels_per_meter = pixels_per_meter
    metadata.scale_text = scale_text
    metadata.image_width = thick_walls.shape[1]
    metadata.image_height = thick_walls.shape[0]

    # חילוץ קירות
    walls = extract_walls_from_opencv_mask(thick_walls)
    for wall in walls:
        metadata.add_wall(wall)

    # שמירה
    metadata_path = get_metadata_filepath(pdf_path)
    metadata.save(metadata_path)

    return metadata_path
