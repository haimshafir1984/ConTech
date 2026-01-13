import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    גרסת TURBO - מותאמת לשרתים עם 2GB RAM ומעלה.
    עובדת ברזולוציה גבוהה לדיוק מקסימלי בזיהוי קירות וסינון טקסט.
    """
    
    def pdf_to_image(self, pdf_path: str, max_size: int = 4000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # --- שדרוג 1: רזולוציה גבוהה (High Res) ---
        # בשרת חזק, אנחנו יכולים להרשות לעצמנו זום של 2.5 או 3.0.
        # זה קריטי כדי להפריד בין טקסט צפוף לקירות.
        base_scale = 2.5 
        
        # מנגנון הגנה: אם הקובץ המקורי ענק (כמו A0 מלא), נוריד טיפה את הזום
        if rect.width > 3000 or rect.height > 3000:
            base_scale = 1.5 
            
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
            # Fallback רק למקרה חירום קיצוני
            print("Warning: High-Res failed, falling back to Low-Res")
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # בגרסת הטורבו אנחנו כמעט לא מקטינים את התמונה
        if max(image.shape) > 4500:
            scale = 4500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- שדרוג 2: ניקוי רעשים חכם (Bilateral Filter) ---
        # הפילטר הזה מנקה "גרעיניות" בתמונה אבל שומר על קצוות חדים של קירות.
        # זה פעולה כבדה שדרשה זיכרון, ועכשיו אפשר להשתמש בה.
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # סף בינארי (Threshold)
        # 150 הוכיח את עצמו כאיזון טוב בין קירות אפורים לרקע לבן
        _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- שדרוג 3: לוגיקת סינון מתקדמת (High-Res Logic) ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # יחס אורך/רוחב
            longer = max(width, height)
            shorter = min(width, height) if min(width, height) > 0 else 1
            aspect_ratio = longer / shorter
            
            is_wall = True
            
            # בגלל שהרזולוציה גבוהה יותר, גם המספרים (שטח) גדלים.
            # אנחנו מעדכנים את הספים בהתאם.
            
            # 1. סינון רעש קטן (עכשיו 50 פיקסלים זה ממש כלום)
            if area < 50: 
                is_wall = False
            
            # 2. סינון טקסט (מרובע וקטן יחסית)
            # ברזולוציה גבוהה, אות יכולה להיות 200-300 פיקסלים
            elif area < 400 and aspect_ratio < 2.5:
                is_wall = False
            
            # 3. הגנה על קירות: אם זה ארוך מאוד (קיר) - שומרים
            if aspect_ratio > 5.0 and area > 100:
                is_wall = True

            # 4. סינון מסגרות ענק
            if width > image.shape[1] * 0.96 or height > image.shape[0] * 0.96:
                is_wall = False

            if is_wall:
                mask[labels == i] = 255
        
        # חיבור קירות (Morphological Closing)
        # ברזולוציה גבוהה, הקירות עבים יותר, אז צריך קרנל גדול יותר כדי לחבר אותם
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
        
        det_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)
        det_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        
        combined = cv2.bitwise_or(det_v, det_h)
        
        # סגירה סופית למילוי חורים
        final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, final_kernel, iterations=1)
        
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
            return {"plan_name": os.path.basename(pdf_path), "scale": None, "raw_text": text[:3000]}
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