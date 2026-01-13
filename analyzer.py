import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    Final Integrated Version:
    1. High Sensitivity (230) - לקווים צהובים/בהירים.
    2. Aggressive Dilation - לחיבור קירות חלולים (V4 Logic).
    3. Grid Removal - ניקוי רעשי ריצוף מהקירות.
    4. Flooring Detection - זיהוי שטח ריצוף כשכבה נפרדת.
    """
    
    def pdf_to_image(self, pdf_path: str, target_max_dim: int = 3000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # זום חכם
        scale = target_max_dim / max(rect.width, rect.height)
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
        except:
            # Fallback למקרה חירום
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_orig, w_orig = image.shape[:2]
        
        if max(image.shape) > 3500:
            scale = 3500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- שלב 1: זיהוי קירות (לוגיקה V4 - הרגישה והחזקה) ---
        
        # סף רגישות 230 (תופס קווים צהובים)
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        
        # ניקוי נקודות קטנות (רשתות)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)
        
        # ניפוח אגרסיבי (5x5) כדי לחבר קירות חלולים (כמו בתיקון שעבד לך)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        walls_base = cv2.dilate(cleaned, dilate_kernel, iterations=2)
        
        # פילטר לוגי לקירות
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walls_base, connectivity=4)
        wall_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            
            is_wall = True
            if area < 100: is_wall = False
            if area < 500 and aspect_ratio < 2.0: is_wall = False # סינון טקסט
            if aspect_ratio > 4.0 and area > 100: is_wall = True # שמירת קירות
            if width > image.shape[1] * 0.95: is_wall = False # סינון מסגרת ראשית

            if is_wall:
                wall_mask[labels == i] = 255
        
        # החלקה סופית של הקירות
        final_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        
        # --- שלב 2: זיהוי ריצוף (החדש) ---
        # מוצאים קווים (Canny) ומחסירים מהם את הקירות שמצאנו
        edges = cv2.Canny(gray, 50, 150)
        
        # מרחיבים את הקירות קצת כדי לא לתפוס את הקצה שלהם כריצוף
        walls_expanded = cv2.dilate(final_walls, np.ones((9,9), np.uint8), iterations=1)
        
        # הריצוף הוא כל הקווים שאינם קירות
        flooring_candidates = cv2.subtract(edges, walls_expanded)
        
        # ניקוי רעשים מהריצוף
        flooring_mask = cv2.morphologyEx(flooring_candidates, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        # --- שלב 3: הפרדת בטון/בלוקים ---
        erosion_size = 8 # ערך גבוה כי הקירות מנופחים
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        concrete_core = cv2.erode(final_walls, element, iterations=1)
        concrete_mask = cv2.dilate(concrete_core, element, iterations=2)
        blocks_mask = cv2.subtract(final_walls, concrete_mask)
        
        # החזרה לגודל מקורי
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
    
    def process_file(self, pdf_path: str):
        image_proc = self.pdf_to_image(pdf_path, target_max_dim=3000)
        # החזרת כל המסכות (כולל ריצוף)
        thick_walls, concrete_mask, blocks_mask, flooring_mask = self.preprocess_image(image_proc)
        
        skel_all = self.skeletonize(thick_walls)
        skel_concrete = self.skeletonize(concrete_mask)
        skel_blocks = self.skeletonize(blocks_mask)
        
        pix_all = cv2.countNonZero(skel_all)
        pix_concrete = cv2.countNonZero(skel_concrete)
        pix_blocks = cv2.countNonZero(skel_blocks)
        
        # שטח ריצוף (בפיקסלים)
        pix_flooring_area = cv2.countNonZero(flooring_mask)
        
        metadata = self.extract_metadata(pdf_path)
        metadata["pixels_concrete"] = pix_concrete
        metadata["pixels_blocks"] = pix_blocks
        metadata["pixels_flooring_area"] = pix_flooring_area
        
        gc.collect()
        return pix_all, skel_all, thick_walls, image_proc, metadata, concrete_mask, blocks_mask, flooring_mask