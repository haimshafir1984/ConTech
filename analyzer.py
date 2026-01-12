import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import pandas as pd
import re
import os
import gc

class FloorPlanAnalyzer:
    """מחלקה לניתוח תוכניות בנייה - מותאמת לשרתים עם זיכרון מוגבל (Render Free Tier)"""
    
    def __init__(self):
        pass
    
    def pdf_to_image(self, pdf_path: str, max_size: int = 2000) -> np.ndarray:
        """
        המרה חכמה שחוסכת זיכרון: מחשבת את הזום הנדרש כדי לא לחרוג
        מגודל מקסימלי, ורק אז יוצרת את התמונה.
        """
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # חישוב יחס ההמרה כדי לא לחרוג מ-2000 פיקסלים (מונע קריסה)
        rect = page.rect
        zoom = 1.0
        if max(rect.width, rect.height) > 0:
            zoom = max_size / max(rect.width, rect.height)
        
        # יצירת המטריצה עם הזום המחושב מראש
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False) # alpha=False חוסך ערוץ שקיפות מיותר
        
        # המרה ישירה ל-numpy
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 3: # RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        doc.close()
        del pix # שחרור זיכרון מיידי
        gc.collect()
        
        return img_bgr
    
    def remove_margins(self, image: np.ndarray, margin_percent: float = 0.15) -> np.ndarray:
        h, w = image.shape[:2]
        m_t, m_b = int(h * margin_percent), int(h * margin_percent)
        m_l, m_r = int(w * margin_percent), int(w * margin_percent)
        
        cropped = image.copy()
        cropped[0:m_t, :] = 0
        cropped[h-m_b:h, :] = 0
        cropped[:, 0:m_l] = 0
        cropped[:, w-m_r:w] = 0
        return cropped

    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        # שיטת השלד הוחלפה לגרסה מהירה יותר (Thinning) אם אפשר,
        # אבל נשאיר את המקורית עם ניהול זיכרון טוב יותר
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_img = img.copy()
        
        while True:
            open_img = cv2.morphologyEx(temp_img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(temp_img, open_img)
            eroded = cv2.erode(temp_img, element)
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0:
                break
        
        return skel

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # הקטנת רזולוציה נוספת אם התמונה עדיין ענקית
        if max(image.shape) > 2000:
            scale = 2000 / max(image.shape)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # שימוש ב-Gaussian במקום Median למהירות
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # סף בינארי אדפטיבי
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = self.remove_margins(binary, margin_percent=0.10)
        
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # סינון רעשים
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
        mask = np.zeros_like(processed)
        
        # סינון חכם יותר לפי שטח יחסי
        min_area = (image.shape[0] * image.shape[1]) * 0.0001 # 0.01% מהשטח
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return result
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Optional[str]]:
        try:
            doc = fitz.open(pdf_path)
            # קריאת טקסט רק מהעמוד הראשון וללא פריסה מלאה לחסכון בזיכרון
            text = doc[0].get_text()
            doc.close()
            
            metadata = {"plan_name": None, "scale": None, "raw_text": text[:1000]} # הגדלת טווח החיפוש
            
            # חיפוש שם
            match = re.search(r"(?:תוכנית|שם\s*שרטוט|Project)[\s:]+([^\n\r]+)", text, re.IGNORECASE)
            metadata["plan_name"] = match.group(1).strip() if match else os.path.basename(pdf_path).replace(".pdf", "")
            
            # חיפוש קנה מידה
            match_s = re.search(r"(\d+)[\s:]*[:/][\s]*(\d+)", text)
            if match_s: metadata["scale"] = f"{match_s.group(1)}:{match_s.group(2)}"
            
            return metadata
        except Exception:
            return {"plan_name": os.path.basename(pdf_path), "scale": None, "raw_text": ""}
    
    def process_file(self, pdf_path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
        # 1. טעינה מותאמת זיכרון
        image_proc = self.pdf_to_image(pdf_path, max_size=1800) # הגבלה ל-1800 פיקסלים
        
        # 2. עיבוד
        thick_walls = self.preprocess_image(image_proc)
        skeleton = self.skeletonize(thick_walls)
        
        # 3. חישובים
        total_pixels = cv2.countNonZero(skeleton)
        metadata = self.extract_metadata(pdf_path)
        
        # ניקוי זיכרון אגרסיבי לפני החזרה
        gc.collect()
        
        return total_pixels, skeleton, thick_walls, image_proc, metadata