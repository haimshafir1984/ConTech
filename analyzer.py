import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    גרסת Safe-Guard:
    משתמשת בחישוב זום דינמי כדי להבטיח שתמונת הפלט לעולם לא תחרוג
    מגבולות הזיכרון של השרת (2500px), לא משנה מה גודל ה-PDF המקורי.
    """
    
    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 2500) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # חישוב זום דינמי:
        # אנחנו בודקים כמה צריך להגדיל/להקטין כדי להגיע בדיוק ל-2500 פיקסלים
        # זה מונע מצב שקובץ ענק הופך למפלצת זיכרון
        scale = target_max_dim / max(rect.width, rect.height)
        
        # אם הקובץ המקורי קטן מאוד, לא ניתן לו לגדול מעבר ל-2.5 (כדי לא למרוח אותו)
        if scale > 3.0: scale = 3.0
        
        mat = fitz.Matrix(scale, scale)
        
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            del pix 
            gc.collect() 
            return img_bgr
        except RuntimeError:
            # במקרה חירום של קריסה - יורדים לרזולוציה נמוכה מאוד
            print("Memory Error detected. Retrying safely...")
            safe_scale = 1000 / max(rect.width, rect.height)
            mat = fitz.Matrix(safe_scale, safe_scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # וידוא נוסף: אם התמונה עדיין גדולה מדי, מקטינים בכוח
        # (זה קורה לעתים נדירות בגלל התיקון למעלה, אבל טוב שיש הגנה כפולה)
        if max(image.shape) > 2500:
            scale = 2500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        # שמירת הגודל המקורי להחזרה בסוף
        h_orig, w_orig = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # שימוש בפילטר מהיר
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # סף בינארי
        _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- לוגיקת זיהוי קירות ---
        # שימוש ב-connectivity=4 לחסכון בזיכרון
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            longer = max(width, height)
            shorter = min(width, height) if min(width, height) > 0 else 1
            aspect_ratio = longer / shorter
            
            is_wall = True
            
            # פרמטרים מותאמים לרזולוציה של 2500px
            if area < 20: is_wall = False
            elif area < 150 and aspect_ratio < 2.0: is_wall = False # סינון טקסט
            
            # הגנה על קירות דקים וארוכים
            if aspect_ratio > 4.0 and area > 50: is_wall = True
            
            # סינון מסגרות ענק
            if width > image.shape[1] * 0.95 or height > image.shape[0] * 0.95: is_wall = False

            if is_wall:
                mask[labels == i] = 255
        
        # חיבור קירות
        # קרנלים מותאמים ל-2500px
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        
        det_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)
        det_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        combined = cv2.bitwise_or(det_v, det_h)
        
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        # --- הפרדת בטון/בלוקים ---
        # ב-2500px, עובי קיר בטון הוא בערך 4-6 פיקסלים
        erosion_size = 3
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        
        concrete_core = cv2.erode(final_mask, element, iterations=1)
        concrete_mask = cv2.dilate(concrete_core, element, iterations=1)
        
        blocks_mask = cv2.subtract(final_mask, concrete_mask)
        
        # וידוא החזרה לגודל (למרות שאנחנו כבר בגודל הנכון, ליתר ביטחון)
        if final_mask.shape[:2] != (h_orig, w_orig):
             final_mask = cv2.resize(final_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             concrete_mask = cv2.resize(concrete_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             blocks_mask = cv2.resize(blocks_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             
        return final_mask, concrete_mask, blocks_mask
    
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
        # שימוש בשיטה הבטוחה עם גבול עליון של 2500 פיקסלים
        image_proc = self.pdf_to_image(pdf_path, target_max_dim=2500)
        
        thick_walls, concrete_mask, blocks_mask = self.preprocess_image(image_proc)
        
        skel_all = self.skeletonize(thick_walls)
        skel_concrete = self.skeletonize(concrete_mask)
        skel_blocks = self.skeletonize(blocks_mask)
        
        pix_all = cv2.countNonZero(skel_all)
        pix_concrete = cv2.countNonZero(skel_concrete)
        pix_blocks = cv2.countNonZero(skel_blocks)
        
        metadata = self.extract_metadata(pdf_path)
        metadata["pixels_concrete"] = pix_concrete
        metadata["pixels_blocks"] = pix_blocks
        
        gc.collect()
        return pix_all, skel_all, thick_walls, image_proc, metadata, concrete_mask, blocks_mask