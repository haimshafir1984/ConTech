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
        # זיהוי טקסט אגרסיבי יותר
        _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV) # סף נמוך יותר = תופס יותר טקסט אפור
        
        # חיבור אותיות למילים בצורה גסה יותר
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4)) 
        binary_connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_connect)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_connected, connectivity=8)
        text_mask = np.zeros_like(gray)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect = w / h if h > 0 else 0
            
            is_text = False
            
            # טקסט רגיל (מילים/משפטים)
            if area < 3500 and (h < 60 or w < 60): 
                is_text = True
                
            # כותרות או מסגרות טקסט
            if 3.0 < aspect < 20.0 and area < 5000:
                is_text = True

            if is_text:
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                # ניפוח אזור המחיקה כדי לוודא שאין שאריות
                text_mask[max(0, y-5):min(gray.shape[0], y+h+5), 
                          max(0, x-5):min(gray.shape[1], x+w+5)] = 255
                          
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