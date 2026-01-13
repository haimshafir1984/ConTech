import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    def pdf_to_image(self, pdf_path: str, max_size: int = 3000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        # הגדלת רזולוציה כדי להבדיל בין טקסט לקווים דקים
        zoom = 2.0 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3: img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        doc.close()
        return img_bgr

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # המרה לאפור
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # סף בינארי (הפרדה לשחור/לבן)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # --- השינוי הגדול: ניקוי רעשים לפני חיבור ---
        # מחיקת אלמנטים קטנים מאוד (כמו אותיות בודדות) *לפני* שהן מתחברות לגושים
        # נניח שכל מה שקטן מ-50 פיקסלים הוא רעש/טקסט
        cleaned_binary = binary.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_binary, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # אם זה קטן מדי (טקסט) או גדול מדי בצורה קיצונית (מסגרת שחורה ענקית)
            if area < 100: 
                cleaned_binary[labels == i] = 0
        
        # עכשיו מחברים את מה שנשאר (הקירות)
        # שימוש בקרנל אנכי ואופקי כדי לשמור על קווים ישרים
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        
        # חיבור קווים אנכיים ואופקיים בנפרד
        detected_v = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        detected_h = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        
        # איחוד התוצאות
        combined = cv2.bitwise_or(detected_v, detected_h)
        
        # סגירה סופית למילוי חורים קטנים בקירות
        block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, block_kernel, iterations=1)
        
        return final_mask
    
    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        try:
            return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except:
            # גיבוי למקרה שאין ximgproc
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
        return pix, skeleton, thick_walls, image_proc, metadata