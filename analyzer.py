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
        
        # חישוב זום חסכוני בזיכרון
        rect = page.rect
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
            # Fallback למקרה של חוסר זיכרון
            mat = fitz.Matrix(0.8, 0.8)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # הקטנה בטיחותית
        if max(image.shape) > 2500:
            scale = 2500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- תיקון 1: סף מחמיר (Strict Threshold) ---
        # במקום 200 (שכולל אפור), ירדנו ל-85. רק שחור עמוק יעבור.
        # זה יעלים את רוב קווי הרשת והמסגרות האפורות.
        _, binary = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        
        # --- תיקון 2: שחיקה (Erosion) להעלמת טקסט ---
        # מוחקים אלמנטים דקיקים (כמו אותיות) לפני שמנסים לחבר אותם
        kernel_erode = np.ones((2,2), np.uint8)
        binary = cv2.erode(binary, kernel_erode, iterations=1)
        
        # ניקוי רעשים סטנדרטי
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        
        total_area = image.shape[0] * image.shape[1]
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # יחס רוחב/גובה (לזיהוי קווים ארוכים)
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            
            # --- תיקון 3: סינון גושים גדולים מדי או "מרובעים" מדי (כמו טקסט) ---
            # קיר הוא בדרך כלל ארוך ודק, או בעל שטח סביר.
            # גושי טקסט הם בדרך כלל מלבניים וגדולים אך לא ענקיים כמו כל המסגרת
            
            is_noise = False
            
            # סינון דברים קטנים מדי (לכלוך)
            if area < 50: is_noise = True
            
            # סינון מסגרות ענק (כמו המסגרת של כל הדף)
            if width > image.shape[1] * 0.9 or height > image.shape[0] * 0.9:
                is_noise = True

            if not is_noise:
                mask[labels == i] = 255
        
        # חיבור קירות (בזהירות יותר)
        # הקטנו את הקרנל כדי לא לחבר טקסטים רחוקים
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))   # היה (1,10)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)) # היה (10,1)
        
        detected_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        detected_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        
        combined = cv2.bitwise_or(detected_v, detected_h)
        
        # סגירה סופית עדינה
        block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, block_kernel, iterations=1)
        
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