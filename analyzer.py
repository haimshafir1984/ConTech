import cv2
import numpy as np
import fitz
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    
    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """
        יצירת skeleton ללא צורך ב-ximgproc
        משתמש באלגוריתם מהיר של erosion iterative
        """
        # וידוא שהתמונה בינארית
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        skeleton = np.zeros_like(binary, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            # Erosion
            eroded = cv2.erode(binary, element)
            # Opening (erosion followed by dilation)
            temp = cv2.dilate(eroded, element)
            # Subtract
            temp = cv2.subtract(binary, temp)
            # Union
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            # Stop condition
            if cv2.countNonZero(binary) == 0:
                break
        
        return skeleton
    
    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 3000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        scale = min(3.0, target_max_dim / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
        doc.close()
        return img_bgr
    
    def _detect_text_regions(self, gray: np.ndarray) -> np.ndarray:
        """זיהוי טקסט אגרסיבי - מסיר כל טקסט, מספרים, סמלים"""
        h, w = gray.shape
        text_mask = np.zeros_like(gray)
        
        # שלב 1: זיהוי בסיסי עם threshold נמוך (תופס יותר טקסט)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # שלב 2: ניקוי רעשים קטנים
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        
        # שלב 3: חיבור תווים למילים (אופקי)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected_h = cv2.dilate(binary_clean, kernel_h, iterations=1)
        
        # שלב 4: חיבור תווים (אנכי) - למספרי חדרים
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        connected_v = cv2.dilate(binary_clean, kernel_v, iterations=1)
        
        # איחוד שתי הכיוונים
        connected = cv2.bitwise_or(connected_h, connected_v)
        
        # שלב 5: זיהוי רכיבים מחוברים
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            
            # חישוב aspect ratio
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            
            is_text = False
            
            # כלל 1: כל דבר קטן יחסית
            if area < 8000:  # הרחבנו מ-5000
                is_text = True
            
            # כלל 2: צורות מרובעות קטנות (סמלים)
            if 0.5 < aspect < 2.0 and area < 2000:
                is_text = True
            
            # כלל 3: קווי טקסט ארוכים ודקים
            if aspect > 3.0 and bh < 80:
                is_text = True
            
            # כלל 4: טקסט אנכי (מספרי חדרים)
            if aspect < 0.35 and bw < 80:
                is_text = True
            
            # חריגה: אל תסנן קירות (דברים ארוכים מאוד)
            if aspect > 15.0 and area > 1000:
                is_text = False
            
            # חריגה: מסגרת ראשית
            if bw > w * 0.9 or bh > h * 0.9:
                is_text = False
            
            if is_text:
                # הוספה עם ריפוד מוגבר
                padding = 8
                y1 = max(0, y - padding)
                y2 = min(h, y + bh + padding)
                x1 = max(0, x - padding)
                x2 = min(w, x + bw + padding)
                text_mask[y1:y2, x1:x2] = 255
        
        # שלב 6: ניפוח סופי כדי לוודא כיסוי מלא
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        text_mask = cv2.dilate(text_mask, kernel_final, iterations=2)
        
        return text_mask

    def process_file(self, pdf_path: str, save_debug=False):
        image_proc = self.pdf_to_image(pdf_path)
        gray = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. זיהוי טקסט
        text_mask = self._detect_text_regions(gray)
        
        # 2. זיהוי קירות - הסרה מוחלטת של הטקסט
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        binary_no_text = cv2.subtract(binary, text_mask)
        
        # ניקוי רעשים
        cleaned = cv2.morphologyEx(binary_no_text, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        walls_base = cv2.dilate(cleaned, np.ones((5,5), np.uint8), iterations=2)
        
        # 3. סינון קירות מתקדם
        num, labels, stats, _ = cv2.connectedComponentsWithStats(walls_base, connectivity=4)
        wall_mask = np.zeros_like(gray)
        
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # צפיפות: כמה פיקסלים שחורים יש בתוך הריבוע החוסם?
            density = area / (bw * bh)
            aspect = max(bw, bh) / min(bw, bh)
            
            is_wall = True
            
            # --- חוקים מחמירים לסינון רעש ---
            if area < 200: is_wall = False # גודל מינימלי עלה
            if density < 0.42: is_wall = False # צפיפות עלתה (קירות הם "בלוקים" מלאים)
            if bw > w * 0.95 or bh > h * 0.95: is_wall = False # מסגרת
            
            # הצלה: אם זה קיר ארוך ודק, הוא קיר גם אם הצפיפות נמוכה
            if aspect > 5.0 and area > 300: is_wall = True

            if is_wall:
                wall_mask[labels == i] = 255
        
        final_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # הפרדת חומרים
        kernel = np.ones((6,6), np.uint8)
        concrete = cv2.dilate(cv2.erode(final_walls, kernel, iterations=1), kernel, iterations=2)
        blocks = cv2.subtract(final_walls, concrete)
        
        # ריצוף (שאריות קווים)
        edges = cv2.Canny(gray, 50, 150)
        flooring = cv2.subtract(cv2.subtract(edges, cv2.dilate(final_walls, np.ones((9,9)))), text_mask)
        
        # הכנת תמונת דיבאג (זיכרון)
        debug_img = None
        if save_debug:
            debug_img = image_proc.copy()
            # צביעת טקסט באדום שקוף
            overlay = debug_img.copy()
            overlay[text_mask > 0] = [0, 0, 255]
            overlay[final_walls > 0] = [255, 0, 0]
            cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0, debug_img)

        # חישובים ונתונים
        skel = self._skeletonize(final_walls)
        pix = cv2.countNonZero(skel)
        
        meta = {"plan_name": os.path.basename(pdf_path), "raw_text": ""}
        try:
            doc = fitz.open(pdf_path)
            meta["raw_text"] = doc[0].get_text()[:3000]
            doc.close()
        except: pass
        
        meta.update({
            "pixels_concrete": cv2.countNonZero(self._skeletonize(concrete)),
            "pixels_blocks": cv2.countNonZero(self._skeletonize(blocks)),
            "pixels_flooring_area": cv2.countNonZero(flooring)
        })
        
        gc.collect()
        
        # החזרה: הוספנו את debug_img בסוף
        return pix, skel, final_walls, image_proc, meta, concrete, blocks, flooring, debug_img