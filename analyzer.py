import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import re
import os
import gc

class FloorPlanAnalyzer:
    """מחלקה לניתוח תוכניות בנייה - גרסה V2 (הכנה לזיהוי דפוסים)"""
    
    def __init__(self):
        # בעתיד: כאן נחזיק את המידע על הדפוסים שה-AI זיהה
        self.active_patterns = {} 
    
    def pdf_to_image(self, pdf_path: str, max_size: int = 2500) -> np.ndarray:
        # העלינו מעט את הרזולוציה (מ-1800 ל-2500) כדי לאפשר זיהוי טוב יותר של דפוסים עדינים
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        zoom = 1.0
        if max(rect.width, rect.height) > 0:
            zoom = max_size / max(rect.width, rect.height)
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3: img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        doc.close()
        del pix
        gc.collect()
        return img_bgr
    
    # --- פונקציות חדשות (Placeholders) לעתיד ---
    def detect_hatch_pattern(self, img_gray: np.ndarray, angle: int = 45) -> np.ndarray:
        """
        פונקציית עתיד: תזהה קיוקווים בזווית מסוימת (למשל לבטון).
        כרגע היא מחזירה תמונה ריקה.
        """
        # כאן יבוא קוד לשימוש בפילטרים (Gabor/Sobel) לזיהוי כיווניות
        return np.zeros_like(img_gray)

    def detect_solid_fill(self, img_gray: np.ndarray) -> np.ndarray:
        """
        פונקציית עתיד: תזהה מילוי מלא (למשל לבלוקים/עמודים).
        """
        # כאן יבוא קוד לזיהוי אזורים כהים גדולים
        return np.zeros_like(img_gray)
    # -------------------------------------------

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        הפונקציה המרכזית לזיהוי קירות - שופרה לסינון רעשים טוב יותר.
        """
        # הקטנת רזולוציה אם התמונה ענקית, לשיפור ביצועים
        proc_img = image.copy()
        if max(proc_img.shape) > 3000:
             scale = 3000 / max(proc_img.shape)
             proc_img = cv2.resize(proc_img, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        
        # שיפור 1: ניקוי רעשים אגרסיבי יותר לפני הסף
        # שימוש ב-Bilateral Filter שומר על קצוות חדים אבל מנקה רעש בתוך משטחים
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # סף בינארי (THRESH_OTSU מחשב אוטומטית את הסף האופטימלי)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # שיפור 2: פעולות מורפולוגיות לחיבור קווים מקווקווים וניקוי נקודות בודדות
        # Kernel מלבני עוזר לחבר קווים ארוכים
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        # שיפור 3: סינון לפי גודל (Area Filtering)
        # מחיקת אלמנטים קטנים מדי שאינם קירות (כמו טקסט קטן או לכלוך)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
        mask = np.zeros_like(processed)
        
        total_area = proc_img.shape[0] * proc_img.shape[1]
        # סף מינימלי: אלמנט חייב להיות לפחות 0.05% מהתמונה כדי להיחשב קיר
        min_area_threshold = total_area * 0.0005 
        
        for i in range(1, num_labels):
            # סינון נוסף: התעלמות ממסגרות דקות וארוכות מדי (כמו קווי גבול של השרטוט)
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # אם האלמנט גדול מספיק, ואינו "דק וארוך" בצורה קיצונית (כמו קו מסגרת)
            if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                if not (aspect_ratio > 20 or aspect_ratio < 0.05): # סינון קווים דקיקים מאוד
                    mask[labels == i] = 255
        
        # החזרת התמונה לגודל המקורי אם הקטנו אותה
        if mask.shape[:2] != image.shape[:2]:
             mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
             
        return mask
    
    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        # שימוש בפונקציה המובנית של OpenCV לביצועים טובים יותר (אם קיימת בגרסה)
        try:
            skel = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except:
            # Fallback לשיטה הישנה והאיטית יותר אם ximgproc לא קיים
            skel = np.zeros(img.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp_img = img.copy()
            while True:
                open_img = cv2.morphologyEx(temp_img, cv2.MORPH_OPEN, element)
                temp = cv2.subtract(temp_img, open_img)
                eroded = cv2.erode(temp_img, element)
                skel = cv2.bitwise_or(skel, temp)
                temp_img = eroded.copy()
                if cv2.countNonZero(temp_img) == 0: break
        return skel

    def extract_metadata(self, pdf_path: str) -> Dict[str, Optional[str]]:
        try:
            doc = fitz.open(pdf_path)
            text = doc[0].get_text()
            doc.close()
            # ניסיון בסיסי לחילוץ שם (ה-AI ב-brain.py יעשה עבודה טובה יותר)
            plan_name = os.path.basename(pdf_path).replace(".pdf", "")
            return {"plan_name": plan_name, "scale": None, "raw_text": text[:2500]}
        except Exception:
            return {"plan_name": os.path.basename(pdf_path), "scale": None, "raw_text": ""}
    
    def process_file(self, pdf_path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
        # 1. טעינה
        image_proc = self.pdf_to_image(pdf_path, max_size=2500)
        # 2. עיבוד (זיהוי קירות כללי)
        thick_walls = self.preprocess_image(image_proc)
        # 3. שלד (לחישוב אורך)
        skeleton = self.skeletonize(thick_walls)
        # 4. נתונים
        total_pixels = cv2.countNonZero(skeleton)
        metadata = self.extract_metadata(pdf_path)
        
        gc.collect()
        return total_pixels, skeleton, thick_walls, image_proc, metadata