"""
Simple Snap Engine
מנוע הצמדה פשוט לנקודות - MVP

Phase 1: Smart Measurements + Quantity Calculator
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


class SimpleSnapEngine:
    """מנוע הצמדה בסיסי - רק לנקודות (MVP)"""
    
    def __init__(self, snap_points: List[Tuple[int, int]], tolerance_px: int = 15):
        """
        אתחול מנוע ההצמדה
        
        Args:
            snap_points: רשימת נקודות להצמדה [(x1,y1), (x2,y2), ...]
            tolerance_px: רדיוס הצמדה בפיקסלים (ברירת מחדל 15px)
        """
        self.points = snap_points
        self.tolerance = tolerance_px
    
    def find_snap(self, x: int, y: int) -> Optional[Tuple]:
        """
        מצא נקודת snap הכי קרובה
        
        Args:
            x, y: קואורדינטות הנקודה הנוכחית
        
        Returns:
            (snap_x, snap_y, distance) אם נמצאה נקודה קרובה
            None אם אין נקודה בטווח
        """
        min_dist = float('inf')
        nearest = None
        
        for px, py in self.points:
            # חישוב מרחק אוקלידי
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            
            # בדיקה אם בטווח ואם קרוב יותר
            if dist < self.tolerance and dist < min_dist:
                min_dist = dist
                nearest = (int(px), int(py), float(dist))
        
        return nearest
    
    def snap_if_close(self, x: int, y: int) -> Tuple[int, int, bool]:
        """
        החזר נקודה - snapped אם קרוב, אחרת מקורית
        
        Args:
            x, y: קואורדינטות מקוריות
        
        Returns:
            (final_x, final_y, was_snapped)
        """
        snap_result = self.find_snap(x, y)
        
        if snap_result:
            snap_x, snap_y, dist = snap_result
            return (snap_x, snap_y, True)
        else:
            return (x, y, False)
    
    def snap_line(self, x1: int, y1: int, x2: int, y2: int) -> Dict:
        """
        הצמדת שני קצוות של קו
        
        Args:
            x1, y1: נקודת התחלה
            x2, y2: נקודת סיום
        
        Returns:
            מילון עם התוצאות:
            {
                'start': (final_x1, final_y1),
                'end': (final_x2, final_y2),
                'start_snapped': bool,
                'end_snapped': bool,
                'snap_distances': [dist1, dist2]
            }
        """
        start_result = self.snap_if_close(x1, y1)
        end_result = self.snap_if_close(x2, y2)
        
        # חישוב מרחקי snap (למידע)
        snap_dist_start = 0
        snap_dist_end = 0
        
        if start_result[2]:  # אם נצמד
            snap_info = self.find_snap(x1, y1)
            if snap_info:
                snap_dist_start = snap_info[2]
        
        if end_result[2]:  # אם נצמד
            snap_info = self.find_snap(x2, y2)
            if snap_info:
                snap_dist_end = snap_info[2]
        
        return {
            'start': (start_result[0], start_result[1]),
            'end': (end_result[0], end_result[1]),
            'start_snapped': start_result[2],
            'end_snapped': end_result[2],
            'snap_distances': [snap_dist_start, snap_dist_end]
        }
    
    def add_point(self, x: int, y: int):
        """הוסף נקודה חדשה לרשימת נקודות הsnap"""
        if (x, y) not in self.points:
            self.points.append((x, y))
    
    def remove_point(self, x: int, y: int, tolerance: int = 5):
        """הסר נקודה (בטווח tolerance)"""
        to_remove = []
        for i, (px, py) in enumerate(self.points):
            if abs(px - x) < tolerance and abs(py - y) < tolerance:
                to_remove.append(i)
        
        # מחיקה הפוכה (מהסוף להתחלה) כדי לא לקלקל אינדקסים
        for i in reversed(to_remove):
            del self.points[i]
    
    def clear_points(self):
        """נקה את כל נקודות הsnap"""
        self.points.clear()
    
    def get_point_count(self) -> int:
        """מספר נקודות snap קיימות"""
        return len(self.points)
    
    def set_tolerance(self, new_tolerance: int):
        """עדכן רדיוס הצמדה"""
        if new_tolerance > 0:
            self.tolerance = new_tolerance
    
    def __repr__(self):
        return f"SimpleSnapEngine(points={len(self.points)}, tolerance={self.tolerance}px)"
