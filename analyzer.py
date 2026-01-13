import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import os
import gc

class FloorPlanAnalyzer:
    """
    גרסה מתוקנת: מתקנת את באג חוסר ההתאמה בגדלים (IndexError).
    כעת המסכות מוחזרות תמיד בגודל המקורי של התמונה, גם אם בוצעה הקטנה לצורך עיבוד.
    """
    
    def pdf_to_image(self, pdf_path: str, max_size: int = 3500) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        
        # זום חסכוני
        base_scale = 2.0 
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
            print("Fallback to Low-Res due to memory")
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n==3 else cv2.COLOR_GRAY2BGR)
            doc.close()
            return img_bgr

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # --- תיקון קריטי: שמירת הגודל המקורי לפני השינוי ---
        original_h, original_w = image.shape[:2]
        
        # הקטנה לצורך עיבוד מהיר וחסכוני
        if max(image.shape) > 3500:
            scale = 3500 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)
        
        # לוגיקת זיהוי קירות
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
            if area < 30: is_wall = False
            elif area < 250 and aspect_ratio < 2.0: is_wall = False
            if aspect_ratio > 4.5 and area > 80: is_wall = True
            if width > image.shape[1] * 0.95 or height > image.shape[0] * 0.95: is_wall = False

            if is_wall:
                mask[labels == i] = 255
        
        # חיבור וסגירה
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 6))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
        combined = cv2.bitwise_or(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=2),
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        )
        final_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        # הפרדת בטון/בלוקים
        erosion_size = 4 
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        concrete_core = cv2.erode(final_mask, element, iterations=1)
        concrete_mask = cv2.dilate(concrete_core, element, iterations=1)
        blocks_mask = cv2.subtract(final_mask, concrete_mask)
        
        # --- תיקון קריטי: החזרה לגודל המקורי שנשמר בהתחלה ---
        # אנחנו מוודאים שהמסכות תואמות בדיוק את התמונה המקורית שנכנסה לפונקציה
        if final_mask.shape[:2] != (original_h, original_w):
             final_mask = cv2.resize(final_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
             concrete_mask = cv2.resize(concrete_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
             blocks_mask = cv2.resize(blocks_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
             
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