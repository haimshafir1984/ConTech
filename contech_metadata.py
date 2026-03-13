"""
ConTech Metadata Schema
גרסה 1.0 - שמירת מטא-דאטה גיאומטרית של תוכניות בנייה
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

# ==========================================
# Schema Version
# ==========================================
METADATA_VERSION = "1.0"

# ==========================================
# Data Structures
# ==========================================


class Wall:
    """ייצוג קיר בודד"""

    def __init__(
        self,
        id: str,
        points: List[Tuple[int, int]],
        wall_type: str = "unknown",
        thickness: float = 0,
    ):
        """
        Args:
            id: מזהה ייחודי של הקיר
            points: רשימת נקודות (x, y) בפיקסלים
            wall_type: סוג קיר (concrete, blocks, partition)
            thickness: עובי קיר בס"מ
        """
        self.id = id
        self.points = points
        self.wall_type = wall_type
        self.thickness = thickness

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "points": self.points,
            "wall_type": self.wall_type,
            "thickness": self.thickness,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data["id"],
            points=[tuple(p) for p in data["points"]],
            wall_type=data.get("wall_type", "unknown"),
            thickness=data.get("thickness", 0),
        )

    def get_length_pixels(self) -> float:
        """מחשב אורך קיר בפיקסלים"""
        if len(self.points) < 2:
            return 0

        total_length = 0
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            total_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return total_length

    def distance_to_point(self, x: int, y: int) -> float:
        """מחשב מרחק מינימלי מנקודה לקיר"""
        if len(self.points) < 2:
            return float("inf")

        min_distance = float("inf")

        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]

            # חישוב מרחק מנקודה לקטע
            distance = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
            min_distance = min(min_distance, distance)

        return min_distance

    @staticmethod
    def _point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
        """מרחק מנקודה לקטע ישר"""
        # וקטור מ-p1 ל-p2
        dx = x2 - x1
        dy = y2 - y1

        # אורך הקטע בריבוע
        l2 = dx * dx + dy * dy

        if l2 == 0:
            # הקטע הוא נקודה
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # פרמטר t של ההקרנה
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))

        # נקודת ההקרנה
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        # מרחק מהנקודה להקרנה
        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


class ContechMetadata:
    """מבנה הנתונים המרכזי"""

    def __init__(self, filename: str, pdf_checksum: str):
        self.version = METADATA_VERSION
        self.filename = filename
        self.pdf_checksum = pdf_checksum
        self.created_at = datetime.now().isoformat()
        self.modified_at = datetime.now().isoformat()

        # כיול
        self.pixels_per_meter = 200.0
        self.scale_text = "1:50"
        self.is_scale_locked = False

        # גאומטריה
        self.walls: List[Wall] = []

        # תמונה
        self.image_width = 0
        self.image_height = 0

        # מטא-דאטה נוספת
        self.plan_name = ""
        self.plan_type = "walls"  # walls, ceiling, flooring, electrical
        self.notes = ""

    def add_wall(self, wall: Wall):
        """הוסף קיר"""
        self.walls.append(wall)
        self.modified_at = datetime.now().isoformat()

    def remove_wall(self, wall_id: str):
        """הסר קיר לפי ID"""
        self.walls = [w for w in self.walls if w.id != wall_id]
        self.modified_at = datetime.now().isoformat()

    def get_wall_by_id(self, wall_id: str) -> Optional[Wall]:
        """מצא קיר לפי ID"""
        for wall in self.walls:
            if wall.id == wall_id:
                return wall
        return None

    def find_nearest_wall(
        self, x: int, y: int, max_distance: float = 15
    ) -> Optional[Wall]:
        """מצא את הקיר הקרוב ביותר לנקודה"""
        nearest_wall = None
        min_distance = max_distance

        for wall in self.walls:
            distance = wall.distance_to_point(x, y)
            if distance < min_distance:
                min_distance = distance
                nearest_wall = wall

        return nearest_wall

    def get_total_length_meters(self) -> float:
        """חישוב אורך כולל של כל הקירות במטרים"""
        total_pixels = sum(w.get_length_pixels() for w in self.walls)
        return total_pixels / self.pixels_per_meter

    def to_dict(self) -> dict:
        """המרה למילון (לשמירה ל-JSON)"""
        return {
            "version": self.version,
            "filename": self.filename,
            "pdf_checksum": self.pdf_checksum,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "pixels_per_meter": self.pixels_per_meter,
            "scale_text": self.scale_text,
            "is_scale_locked": self.is_scale_locked,
            "walls": [w.to_dict() for w in self.walls],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "plan_name": self.plan_name,
            "plan_type": self.plan_type,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """טעינה ממילון (מ-JSON)"""
        metadata = cls(filename=data["filename"], pdf_checksum=data["pdf_checksum"])

        metadata.version = data.get("version", "1.0")
        metadata.created_at = data.get("created_at", datetime.now().isoformat())
        metadata.modified_at = data.get("modified_at", datetime.now().isoformat())
        metadata.pixels_per_meter = data.get("pixels_per_meter", 200.0)
        metadata.scale_text = data.get("scale_text", "1:50")
        metadata.is_scale_locked = data.get("is_scale_locked", False)

        metadata.walls = [Wall.from_dict(w) for w in data.get("walls", [])]

        metadata.image_width = data.get("image_width", 0)
        metadata.image_height = data.get("image_height", 0)
        metadata.plan_name = data.get("plan_name", "")
        metadata.plan_type = data.get("plan_type", "walls")
        metadata.notes = data.get("notes", "")

        return metadata

    def save(self, filepath: str):
        """שמירה לקובץ JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """טעינה מקובץ JSON"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ==========================================
# Helper Functions
# ==========================================


def calculate_pdf_checksum(pdf_path: str) -> str:
    """חישוב checksum של קובץ PDF"""
    hasher = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def extract_walls_from_opencv_mask(mask: np.ndarray) -> List[Wall]:
    """
    חילוץ קירות מתוצאת OpenCV (mask בינארי)

    Args:
        mask: מסכה בינארית (0/255) של קירות

    Returns:
        רשימת קירות עם קואורדינטות מדויקות
    """
    import cv2

    # מצא קונטורים
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    walls = []

    for idx, contour in enumerate(contours):
        # דלג על קונטורים קטנים מדי
        if cv2.contourArea(contour) < 50:
            continue

        # המרה לרשימת נקודות
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]

        # יצירת קיר
        wall = Wall(id=f"wall_{idx:04d}", points=points, wall_type="unknown")

        walls.append(wall)

    return walls


def get_metadata_filepath(pdf_path: str) -> str:
    """קבל נתיב לקובץ metadata תואם ל-PDF"""
    import os

    base = os.path.splitext(pdf_path)[0]
    return f"{base}_metadata.json"


def metadata_exists(pdf_path: str) -> bool:
    """בדוק אם קיים קובץ metadata"""
    import os

    return os.path.exists(get_metadata_filepath(pdf_path))


def validate_metadata_checksum(metadata: ContechMetadata, pdf_path: str) -> bool:
    """בדוק אם ה-checksum תואם"""
    current_checksum = calculate_pdf_checksum(pdf_path)
    return metadata.pdf_checksum == current_checksum
