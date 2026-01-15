import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    גרסה משולבת ומתוחכמת:
    1. זיהוי טקסט אקטיבי והסרתו.
    2. סינון לפי צפיפות (Density) להבדלה בין קיר לטקסט צפוף.
    3. הפרדה בין בטון לבלוקים.
    """
    
    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 3000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # זום חכם לשמירה על איכות
        scale = target_max_dim / max(rect.width, rect.height)
        if scale > 3.0: scale = 3.0 # הגבלה למניעת קריסת זיכרון
        
        mat = fitz.Matrix(scale, scale)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            # המרה ל-BGR (פורמט של OpenCV)
            if pix.n == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            doc.close()
            del pix 
            gc.collect() 
            return img_bgr
        except:
            # Fallback למקרה חירום (ללא זום)
            pix = page.get_pixmap(alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            doc.close()
            return img_bgr
    
    def _detect_text_regions(self, gray: np.ndarray) -> np.ndarray:
        """
        מזהה אזורי טקסט בתמונה ומחזיר מסכה (Mask) שלהם.
        """
        h, w = gray.shape
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Threshold גבוה כדי לתפוס רק טקסט שחור מובהק
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # ניקוי ראשוני של רעש
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # מציאת רכיבים (Connected Components)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # דילוג על רכיבים ענקיים (שבטוח הם קירות או מסגרת)
            if area > 3000: continue
            
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            
            is_text = False
            
            # 1. תווים בודדים (קטנים ופרופורציונליים)
            if 10 < area < 800 and 0.2 < aspect_ratio < 4.0:
                is_text = True
            
            # 2. קווי טקסט (מילים/משפטים) - רחבים וצרים
            if height < 50 and width > 20 and aspect_ratio > 2.0 and area < 1500:
                is_text = True
            
            # 3. טקסט אנכי (כמו מספרי חדרים צדדיים)
            if width < 50 and height > 20 and aspect_ratio < 0.5 and area < 1500:
                is_text = True
            
            if is_text:
                # סימון הטקסט במסכה עם "ריפוד" קטן (Dilate) כדי לוודא כיסוי
                x = max(0, stats[i, cv2.CC_STAT_LEFT] - 2)
                y = max(0, stats[i, cv2.CC_STAT_TOP] - 2)
                x2 = min(w, x + width + 4)
                y2 = min(h, y + height + 4)
                text_mask[y:y2, x:x2] = 255
        
        # איחוד אזורי טקסט קרובים
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel_final, iterations=1)
        
        return text_mask

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_orig, w_orig = image.shape[:2]
        
        # הקטנה אם התמונה ענקית (לביצועים)
        if max(image.shape) > 3500:
            scale = 3500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- שלב 1: זיהוי והסרת טקסט ---
        text_mask = self._detect_text_regions(gray)
        
        # --- שלב 2: זיהוי קירות ---
        # רגישות גבוהה (230) כדי לתפוס גם קירות בהירים
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        
        # הסרת הטקסט מהתמונה הבינארית! (פעולה קריטית)
        binary_no_text = cv2.subtract(binary, text_mask)
        
        # ניקוי רעשים דקים (רשתות/קווי מידה)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary_no_text, cv2.MORPH_OPEN, clean_kernel)
        
        # חיבור קירות (לסגירת חללים בקירות כפולים)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        walls_base = cv2.dilate(cleaned, dilate_kernel, iterations=2)
        
        # --- שלב 3: סינון מתקדם לפי לוגיקה של קיר ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walls_base, connectivity=4)
        wall_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # חישוב צפיפות (Density): כמה מהריבוע החוסם באמת מלא בפיקסלים?
            # טקסט הוא "אוורירי", קיר הוא "מלא"
            bbox_area = width * height
            density = area / bbox_area if bbox_area > 0 else 0
            
            is_wall = True
            
            # 1. קטן מדי? זבל.
            if area < 100: is_wall = False
            
            # 2. צפיפות נמוכה? כנראה טקסט שהתפספס בשלב 1
            if density < 0.35 and area < 1500: is_wall = False
            
            # 3. ריבוע מושלם קטן? כנראה סמל או עמוד בודד (אפשר להשאיר אם זה עמוד)
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            if area < 500 and aspect_ratio < 1.5: is_wall = False

            # 4. מסגרת של כל השרטוט? להעיף
            if width > image.shape[1] * 0.95 or height > image.shape[0] * 0.95: is_wall = False

            if is_wall:
                wall_mask[labels == i] = 255
        
        # החלקה סופית
        final_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        
        # --- שלב 4: הפרדת בטון / בלוקים ---
        # בטון = הליבה העבה. בלוקים = מה שנשאר.
        erosion_size = 6
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        concrete_core = cv2.erode(final_walls, element, iterations=1)
        # החזרת הבטון לגודל סביר
        concrete_mask = cv2.dilate(concrete_core, element, iterations=2)
        # הבלוקים הם ההפרש
        blocks_mask = cv2.subtract(final_walls, concrete_mask)
        
        # --- שלב 5: זיהוי ריצוף (אופציונלי) ---
        # ריצוף הוא בדרך כלל קווים דקים שנשארו אחרי שהורדנו את הקירות והטקסט
        edges = cv2.Canny(gray, 50, 150)
        # מרחיבים את הקירות כדי לא לתפוס את הקצוות שלהם כריצוף
        walls_expanded = cv2.dilate(final_walls, np.ones((9,9), np.uint8))
        flooring_candidates = cv2.subtract(edges, walls_expanded)
        # מוחקים גם את הטקסט מהריצוף
        flooring_candidates = cv2.subtract(flooring_candidates, text_mask)
        flooring_mask = cv2.morphologyEx(flooring_candidates, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        # החזרה לגודל המקורי
        if final_walls.shape[:2] != (h_orig, w_orig):
             final_walls = cv2.resize(final_walls, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             concrete_mask = cv2.resize(concrete_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             blocks_mask = cv2.resize(blocks_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             flooring_mask = cv2.resize(flooring_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
             
        return final_walls, concrete_mask, blocks_mask, flooring_mask
    
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
    
    def process_file(self, pdf_path: str, save_debug=False):
        image_proc = self.pdf_to_image(pdf_path, target_max_dim=3000)
        
        # העיבוד הראשי
        thick_walls, concrete_mask, blocks_mask, flooring_mask = self.preprocess_image(image_proc)
        
        # מצב דיבאג: שמירת תמונה שמראה מה זוהה כטקסט ומה כקיר
        if save_debug:
            gray = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)
            text_mask = self._detect_text_regions(gray)
            
            # יצירת תמונת השוואה (Overlay)
            debug_img = image_proc.copy()
            # צביעת טקסט באדום
            debug_img[text_mask > 0] = [0, 0, 255] 
            # צביעת קירות בכחול
            debug_img[thick_walls > 0] = [255, 0, 0]
            
            cv2.imwrite("debug_text_detection.png", debug_img)
        
        skel_all = self.skeletonize(thick_walls)
        skel_concrete = self.skeletonize(concrete_mask)
        skel_blocks = self.skeletonize(blocks_mask)
        
        pix_all = cv2.countNonZero(skel_all)
        pix_concrete = cv2.countNonZero(skel_concrete)
        pix_blocks = cv2.countNonZero(skel_blocks)
        pix_flooring = cv2.countNonZero(flooring_mask)
        
        metadata = self.extract_metadata(pdf_path)
        metadata["pixels_concrete"] = pix_concrete
        metadata["pixels_blocks"] = pix_blocks
        metadata["pixels_flooring_area"] = pix_flooring
        
        gc.collect()
        return pix_all, skel_all, thick_walls, image_proc, metadata, concrete_mask, blocks_mask, flooring_mask