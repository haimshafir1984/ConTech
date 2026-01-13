import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    def pdf_to_image(self, pdf_path: str, max_size: int = 2000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # שמירה על זום נמוך כדי לא להעמיס על הזיכרון
        base_scale = 1.5 
        if rect.width > 2000 or rect.height > 2000:
            base_scale = 1.0 
            
        mat = fitz.Matrix(base_scale, base_scale)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            del pix 
            gc.collect() 
            return img_bgr
        except RuntimeError:
            mat = fitz.Matrix(0.8, 0.8)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if max(image.shape) > 2500:
            scale = 2500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- תיקון 1: סף מאוזן (150) ---
        # 85 היה מחמיר מדי. 200 היה רגיש מדי. 150 זה האמצע.
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- ביטלנו את השחיקה (Erosion) ---
        # זה מה שמחק את הקירות הדקים בגרסה הקודמת.
        
        # --- תיקון 2: סינון חכם לפי פרופורציה (Aspect Ratio) ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # חישוב יחס אורך/רוחב
            # קיר הוא ארוך (יחס גבוה). טקסט הוא מרובע (יחס נמוך, קרוב ל-1).
            longer_side = max(width, height)
            shorter_side = min(width, height) if min(width, height) > 0 else 1
            aspect_ratio = longer_side / shorter_side
            
            is_wall = True
            
            # 1. סינון רעש זעיר ממש (פיקסלים בודדים)
            if area < 15: 
                is_wall = False
            
            # 2. סינון טקסט: אם זה גם קטן וגם מרובע - זה כנראה אות
            # (שטח קטן מ-150 פיקסלים וגם יחס קטן מ-3)
            elif area < 150 and aspect_ratio < 3.0:
                is_wall = False
            
            # 3. הגנה על קירות: אם זה ארוך (יחס > 4), נשמור את זה גם אם זה דק/קטן
            if aspect_ratio > 4.0:
                is_wall = True

            # 4. סינון מסגרות ענק (כמו המסגרת של כל הדף)
            if width > image.shape[1] * 0.95 or height > image.shape[0] * 0.95:
                is_wall = False

            if is_wall:
                mask[labels == i] = 255
        
        # חיבור קירות (Closing)
        # קרנל בינוני לחיבור אלמנטים קרובים
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # החזרה לגודל מקורי
        if final_mask.shape[:2] != image.shape[:2]:
             final_mask = cv2.resize(final_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
             
        return final_mask
    
    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        try:
            return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except:
            return img 

    def extract_metadata(self, pdf_path: str) -> Dict[str, Optional[str]]:
        try:
            doc = fitz.open(pdf_path)
            text = doc[0].get_text()
            doc.close()
            return {"plan_name": os.path.basename(pdf_path), "scale": None, "raw_text": text[:2500]}
        except:
            return {}
    
    def process_file(self, pdf_path: str):
        image_proc = self.pdf_to_image(pdf_path)
        thick_walls = self.preprocess_image(image_proc)
        skeleton = self.skeletonize(thick_walls)
        pix = cv2.countNonZero(skeleton)
        metadata = self.extract_metadata(pdf_path)
        gc.collect()
        return pix, skeleton, thick_walls, image_proc, metadata