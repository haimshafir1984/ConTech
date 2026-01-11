import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Tuple, Dict, Optional
import pandas as pd
import re
import os

class FloorPlanAnalyzer:
    """מחלקה לניתוח תוכניות בנייה - אופטימיזציה למהירות ודיוק"""
    
    def __init__(self):
        pass
    
    def pdf_to_image(self, pdf_path: str, dpi: int = 200) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        doc.close()
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.medianBlur(gray, 5) 
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = self.remove_margins(binary, margin_percent=0.10)
        
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
        mask = np.zeros_like(processed)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 100:
                mask[labels == i] = 255
        
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Optional[str]]:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text()
        doc.close()
        metadata = {"plan_name": None, "scale": None, "raw_text": text[:500]}
        match = re.search(r"(?:תוכנית|שם\s*שרטוט|Project)[\s:]+([^\n\r]+)", text, re.IGNORECASE)
        metadata["plan_name"] = match.group(1).strip() if match else os.path.basename(pdf_path).replace(".pdf", "")
        match_s = re.search(r"(\d+)[\s:]*[:/][\s]*(\d+)", text)
        if match_s: metadata["scale"] = f"{match_s.group(1)}:{match_s.group(2)}"
        return metadata
    
    def process_file(self, pdf_path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
        image = self.pdf_to_image(pdf_path)
        h, w = image.shape[:2]
        max_dim = 2000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image_proc = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            image_proc = image
            
        thick_walls = self.preprocess_image(image_proc)
        skeleton = self.skeletonize(thick_walls)
        total_pixels = cv2.countNonZero(skeleton)
        metadata = self.extract_metadata(pdf_path)
        
        return total_pixels, skeleton, thick_walls, image_proc, metadata