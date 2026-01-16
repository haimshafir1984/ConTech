import cv2
import numpy as np
import fitz
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    גרסה 2.0 - Multi-pass filtering עם confidence scoring
    """
    
    def __init__(self):
        self.debug_layers = {}  # לשמירת שכבות ביניים
    
    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """יצירת skeleton ללא צורך ב-ximgproc"""
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
    
    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 3000) -> np.ndarray:
        """המרת PDF לתמונה"""
        doc = fitz.open(pdf_path)
        page = doc[0]
        scale = min(3.0, target_max_dim / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
        doc.close()
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
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
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
                    max(0, y-padding):min(h, y+bh+padding),
                    max(0, x-padding):min(w, x+bw+padding)
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
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
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
                    max(0, y-padding):min(h, y+bh+padding),
                    max(0, x-padding):min(w, x+bw+padding)
                ] = 255
        
        return symbols_mask
    
    # ==========================================
    # PASS 3: זיהוי מספרי חדרים
    # ==========================================
    def _detect_room_numbers(self, gray: np.ndarray, walls_estimate: np.ndarray) -> np.ndarray:
        """
        מזהה מספרי חדרים בלבד - באמצע חדרים
        משתמש במידע על הקירות כדי למצוא חדרים
        """
        h, w = gray.shape
        numbers_mask = np.zeros_like(gray)
        
        # מצא אזורים סגורים (חדרים)
        walls_dilated = cv2.dilate(walls_estimate, np.ones((3,3), np.uint8), iterations=2)
        rooms = cv2.bitwise_not(walls_dilated)
        
        # מצא רכיבי חדרים
        num_rooms, room_labels, room_stats, room_centroids = cv2.connectedComponentsWithStats(rooms, connectivity=8)
        
        # זיהוי מספרים קטנים
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))  # אנכי
        connected = cv2.dilate(binary, kernel, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
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
                            max(0, y-padding):min(h, y+bh+padding),
                            max(0, x-padding):min(w, x+bw+padding)
                        ] = 255
        
        return numbers_mask
    
    # ==========================================
    # SMART WALL DETECTION
    # ==========================================
    def _smart_wall_detection(self, gray: np.ndarray, text_mask_combined: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        זיהוי קירות משופר עם confidence scoring
        מחזיר: (wall_mask, confidence_map)
        """
        h, w = gray.shape
        
        # זיהוי בסיסי
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        binary_no_text = cv2.subtract(binary, text_mask_combined)
        
        # ניקוי
        cleaned = cv2.morphologyEx(binary_no_text, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        # זיהוי רכיבים
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=4)
        
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
            if area > 150:  # גודל סביר
                confidence += 0.2
            
            if density > 0.35:  # צפוף מספיק
                confidence += 0.3
            
            if aspect > 3.0:  # ארוך ודק (קיר)
                confidence += 0.3
            
            if area > 500 and aspect > 5.0:  # קיר ארוך וברור
                confidence += 0.2
            
            # מדדים שליליים
            if bw > w * 0.9 or bh > h * 0.9:  # מסגרת
                confidence = 0.0
            
            if area < 100:  # קטן מדי
                confidence *= 0.3
            
            if density < 0.25:  # דליל מדי
                confidence *= 0.5
            
            # שמירה רק אם confidence > 0.4
            if confidence > 0.4:
                mask = (labels == i).astype(np.uint8) * 255
                wall_mask = cv2.bitwise_or(wall_mask, mask)
                confidence_map[labels == i] = confidence
        
        # החלקה סופית
        final_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((4,4), np.uint8))
        
        return final_walls, confidence_map
    
    # ==========================================
    # MAIN PROCESSING
    # ==========================================
    def process_file(self, pdf_path: str, save_debug=False):
        """
        עיבוד מרכזי עם multi-pass filtering
        """
        image_proc = self.pdf_to_image(pdf_path)
        gray = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # === MULTI-PASS TEXT DETECTION ===
        
        # Pass 1: טקסט ברור
        text_obvious = self._detect_obvious_text(gray)
        self.debug_layers['text_obvious'] = text_obvious.copy()
        
        # Pass 2: סמלים וכותרות
        symbols = self._detect_symbols_and_labels(gray)
        self.debug_layers['symbols'] = symbols.copy()
        
        # הערכה ראשונית של קירות (לצורך Pass 3)
        _, binary_temp = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        walls_estimate = cv2.subtract(binary_temp, cv2.bitwise_or(text_obvious, symbols))
        walls_estimate = cv2.morphologyEx(walls_estimate, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # Pass 3: מספרי חדרים
        room_numbers = self._detect_room_numbers(gray, walls_estimate)
        self.debug_layers['room_numbers'] = room_numbers.copy()
        
        # איחוד חכם של כל הטקסט
        text_mask_combined = cv2.bitwise_or(text_obvious, symbols)
        text_mask_combined = cv2.bitwise_or(text_mask_combined, room_numbers)
        
        # ניפוח סופי מתון
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask_combined = cv2.dilate(text_mask_combined, kernel_final, iterations=1)
        
        self.debug_layers['text_combined'] = text_mask_combined.copy()
        
        # === SMART WALL DETECTION ===
        final_walls, confidence_map = self._smart_wall_detection(gray, text_mask_combined)
        self.debug_layers['walls'] = final_walls.copy()
        self.debug_layers['confidence'] = confidence_map.copy()
        
        # === הפרדת חומרים ===
        kernel = np.ones((6,6), np.uint8)
        concrete = cv2.dilate(cv2.erode(final_walls, kernel, iterations=1), kernel, iterations=2)
        blocks = cv2.subtract(final_walls, concrete)
        
        # === ריצוף ===
        edges = cv2.Canny(gray, 50, 150)
        flooring = cv2.subtract(cv2.subtract(edges, cv2.dilate(final_walls, np.ones((9,9)))), text_mask_combined)
        
        # === תמונת Debug משופרת ===
        debug_img = None
        if save_debug:
            debug_img = image_proc.copy()
            overlay = debug_img.copy()
            
            # שכבות צבעוניות
            overlay[text_obvious > 0] = [255, 100, 0]      # כתום - טקסט ברור
            overlay[symbols > 0] = [255, 200, 0]           # צהוב - סמלים
            overlay[room_numbers > 0] = [255, 0, 255]      # סגול - מספרי חדרים
            overlay[final_walls > 0] = [0, 255, 0]         # ירוק - קירות
            
            # הוספת confidence (קירות בהירים יותר = ביטחון גבוה)
            confidence_visual = (confidence_map * 255).astype(np.uint8)
            confidence_visual = cv2.applyColorMap(confidence_visual, cv2.COLORMAP_HOT)
            overlay = cv2.addWeighted(overlay, 0.7, confidence_visual, 0.3, 0)
            
            cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0, debug_img)
        
        # === חישובים ונתונים ===
        skel = self._skeletonize(final_walls)
        pix = cv2.countNonZero(skel)
        
        meta = {"plan_name": os.path.basename(pdf_path), "raw_text": ""}
        try:
            doc = fitz.open(pdf_path)
            meta["raw_text"] = doc[0].get_text()[:3000]
            doc.close()
        except:
            pass
        
        meta.update({
            "pixels_concrete": cv2.countNonZero(self._skeletonize(concrete)),
            "pixels_blocks": cv2.countNonZero(self._skeletonize(blocks)),
            "pixels_flooring_area": cv2.countNonZero(flooring),
            "confidence_avg": float(np.mean(confidence_map[final_walls > 0])) if np.any(final_walls > 0) else 0.0,
            "text_removed_pixels": cv2.countNonZero(text_mask_combined)
        })
        
        gc.collect()
        
        return pix, skel, final_walls, image_proc, meta, concrete, blocks, flooring, debug_img
