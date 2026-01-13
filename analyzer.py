import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    def pdf_to_image(self, pdf_path: str, max_size: int = 4000) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # עובדים ברזולוציה גבוהה (Turbo) כדי למדוד עובי קירות
        base_scale = 2.5 
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
            print("Fallback to Low-Res")
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max(image.shape) > 4500:
            scale = 4500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- זיהוי קירות בסיסי ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            longer = max(width, height)
            shorter = min(width, height) if min(width, height) > 0 else 1
            aspect_ratio = longer / shorter
            
            is_wall = True
            if area < 50: is_wall = False
            elif area < 400 and aspect_ratio < 2.5: is_wall = False
            if aspect_ratio > 5.0 and area > 100: is_wall = True
            if width > image.shape[1] * 0.96 or height > image.shape[0] * 0.96: is_wall = False

            if is_wall:
                mask[labels == i] = 255
        
        # חיבור וסגירה
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
        det_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)
        det_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        combined = cv2.bitwise_or(det_v, det_h)
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=1)
        
        # --- החדש: הפרדה בין בטון (עבה) לבלוקים (דק) ---
        # 1. מזהים בטון ע"י "שחיקה" (Erosion) אגרסיבית - רק העבים שורדים
        erosion_size = 6 # פרמטר לקביעת עובי בטון
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        concrete_core = cv2.erode(final_mask, element, iterations=1)
        # משחזרים את הצורה המקורית של הבטון
        concrete_mask = cv2.dilate(concrete_core, element, iterations=1)
        
        # 2. הבלוקים הם מה שנשאר: (הכל) פחות (בטון)
        blocks_mask = cv2.subtract(final_mask, concrete_mask)
        
        # החזרה לגודל מקורי
        if final_mask.shape[:2] != image.shape[:2]:
             final_mask = cv2.resize(final_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
             concrete_mask = cv2.resize(concrete_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
             blocks_mask = cv2.resize(blocks_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
             
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
        image_proc = self.pdf_to_image(pdf_path)
        # מקבלים חזרה 3 מסכות: הכל, בטון, בלוקים
        thick_walls, concrete_mask, blocks_mask = self.preprocess_image(image_proc)
        
        # חישוב שלד לכל סוג בנפרד (כדי לקבל מטר אורך)
        skel_all = self.skeletonize(thick_walls)
        skel_concrete = self.skeletonize(concrete_mask)
        skel_blocks = self.skeletonize(blocks_mask)
        
        pix_all = cv2.countNonZero(skel_all)
        pix_concrete = cv2.countNonZero(skel_concrete)
        pix_blocks = cv2.countNonZero(skel_blocks)
        
        metadata = self.extract_metadata(pdf_path)
        # שומרים את הנתונים החדשים במטא-דאטה
        metadata["pixels_concrete"] = pix_concrete
        metadata["pixels_blocks"] = pix_blocks
        
        gc.collect()
        # מחזירים את כל המסכות כדי שנוכל לצבוע אותן ב-App
        return pix_all, skel_all, thick_walls, image_proc, metadata, concrete_mask, blocks_mask