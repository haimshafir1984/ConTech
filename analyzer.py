import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    def pdf_to_image(self, pdf_path: str, max_size: int = 2000) -> np.ndarray:
        """
        המרה חסכונית בזיכרון.
        אנחנו מגבילים את הרזולוציה כדי לא להקריס את השרת החינמי.
        """
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # חישוב זום חכם שלא יפוצץ את הזיכרון
        # בודקים מה הגודל המקורי, ואם הוא ענק - לא עושים זום
        rect = page.rect
        base_scale = 1.5 # מורידים מ-2.0 ל-1.5 לטובת יציבות
        
        if rect.width > 2000 or rect.height > 2000:
            base_scale = 1.0 # אם הקובץ כבר ענק, לא נגדיל אותו
            
        mat = fitz.Matrix(base_scale, base_scale)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # המרה ישירה ל-NumPy בצורה יעילה
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3: 
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else: 
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            doc.close()
            del pix # שחרור זיכרון מידי
            gc.collect() # ניקוי זבל
            return img_bgr
            
        except RuntimeError:
            # אם נגמר הזיכרון, מנסים שוב ברזולוציה נמוכה
            print("Low memory warning: Retrying with lower resolution...")
            mat = fitz.Matrix(0.8, 0.8)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # הקטנה אם התמונה גדולה מדי (הגנה נוספת)
        if max(image.shape) > 2500:
            scale = 2500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # סף בינארי
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # --- ניקוי רעשים (טקסט) ---
        cleaned_binary = binary.copy()
        
        # שימוש ב-connectedComponents פשוט יותר שצורך פחות זיכרון
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_binary, connectivity=4)
        
        # יצירת מסכה חדשה במקום לערוך את הישנה (מהיר יותר ב-Numpy)
        mask = np.zeros_like(cleaned_binary)
        
        # סינון: רק אלמנטים בגודל סביר ייכנסו
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 100: # סינון רעש קטן
                mask[labels == i] = 255
        
        # חיבור קירות
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        
        detected_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        detected_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        
        combined = cv2.bitwise_or(detected_v, detected_h)
        
        block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, block_kernel, iterations=1)
        
        # החזרה לגודל מקורי אם הקטנו
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
        
        # ניקוי זיכרון אגרסיבי בסוף התהליך
        gc.collect()
        return pix, skeleton, thick_walls, image_proc, metadata