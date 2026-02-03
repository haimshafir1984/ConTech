"""
Building Elements Data Model
מודל נתונים מונחה אובייקטים לאלמנטי בנייה

Phase 1: Smart Measurements + Quantity Calculator
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np


@dataclass
class Wall:
    """ייצוג קיר עם כל המאפיינים"""
    uid: str
    start: Tuple[float, float]  # (x, y) פיקסלים או מטרים
    end: Tuple[float, float]    # (x, y) פיקסלים או מטרים
    thickness: float = 0.20     # מטר (ברירת מחדל 20cm)
    height: float = 2.5         # מטר
    material: str = "בטון"      # בטון / בלוקים / גבס / אחר
    status: str = "planned"     # planned / in-progress / completed
    
    @property
    def length(self) -> float:
        """אורך הקיר במטרים"""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx**2 + dy**2)
    
    @property
    def area(self) -> float:
        """שטח הקיר (מ"ר)"""
        return self.length * self.height
    
    @property
    def volume(self) -> float:
        """נפח הקיר (מ"ק)"""
        return self.length * self.height * self.thickness
    
    def to_dict(self) -> Dict:
        """המרה למילון לשמירה ב-JSON"""
        return {
            'uid': self.uid,
            'start': list(self.start),
            'end': list(self.end),
            'thickness': self.thickness,
            'height': self.height,
            'material': self.material,
            'status': self.status,
            'length': round(self.length, 2),
            'area': round(self.area, 2),
            'volume': round(self.volume, 3)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Wall':
        """יצירת Wall ממילון"""
        return cls(
            uid=data['uid'],
            start=tuple(data['start']),
            end=tuple(data['end']),
            thickness=data.get('thickness', 0.20),
            height=data.get('height', 2.5),
            material=data.get('material', 'בטון'),
            status=data.get('status', 'planned')
        )


# ניתן להוסיף מחלקות נוספות בעתיד:
# @dataclass
# class Door:
#     pass
# 
# @dataclass
# class Window:
#     pass
# 
# @dataclass
# class Floor:
#     pass
