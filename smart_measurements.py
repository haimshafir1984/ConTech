"""
Smart Measurements System v2.0
מערכת מדידות חכמות על סמך segments קיימים (לא Hough כפול)

Phase 1: Smart Measurements + Quantity Calculator
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


class SmartMeasurements:
    """
    מערכת מדידות חכמות - משתמשת בגאומטריה שכבר זוהתה
    (לא מריצה Hough מחדש!)
    """
    
    def __init__(self, detected_segments: List[Dict], scale: float):
        """
        אתחול המערכת עם segments מוכנים
        
        Args:
            detected_segments: רשימת קטעים שכבר זוהו (מהאנליזר)
                              כל קטע: {'start': (x,y), 'end': (x,y), 'length_px': ...}
            scale: פיקסלים למטר
        """
        self.segments = detected_segments
        self.scale = scale
        
        # המרה למטרים (אם צריך)
        self._convert_to_meters()
    
    def _convert_to_meters(self):
        """המרת כל הקטעים למטרים"""
        for seg in self.segments:
            # אם אין length_m כבר - חשב אותו
            if 'length_m' not in seg:
                length_px = seg.get('length_px', 0)
                seg['length_m'] = length_px / self.scale if self.scale > 0 else 0
            
            # הוסף direction אם אין
            if 'direction' not in seg and 'angle' in seg:
                angle = seg['angle']
                if abs(angle) < 10 or abs(angle - 180) < 10:
                    seg['direction'] = 'horizontal'
                elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
                    seg['direction'] = 'vertical'
                else:
                    seg['direction'] = 'diagonal'
    
    def suggest_for_point(self, x: int, y: int, radius: int = 50) -> List[Dict]:
        """
        מציע מדידות לנקודה שנלחצה
        
        Args:
            x, y: קואורדינטות הנקודה (פיקסלים)
            radius: רדיוס חיפוש (פיקסלים)
        
        Returns:
            רשימת הצעות מדידה, ממוינת לפי רלוונטיות
        """
        suggestions = []
        
        # מצא קטעים קרובים לנקודה
        for seg in self.segments:
            # חשב מרחק מהנקודה לקטע
            dist = self._point_to_line_distance(
                (x, y),
                seg['start'],
                seg['end']
            )
            
            if dist < radius:
                # חשב ביטחון (confidence) - ככל שקרוב יותר, ביטחון גבוה יותר
                confidence = 1.0 - (dist / radius)
                
                direction = seg.get('direction', 'unknown')
                
                suggestions.append({
                    'type': 'wall_segment',
                    'description': f"קיר {direction}: {seg['length_m']:.2f}m",
                    'length_m': round(seg['length_m'], 2),
                    'start': seg['start'],
                    'end': seg['end'],
                    'direction': direction,
                    'confidence': round(confidence, 2),
                    'distance_from_click': round(dist, 1),
                    'source': 'analyzer'  # מקור: זוהה על ידי האנליזר
                })
        
        # מיון לפי ביטחון (גבוה לנמוך)
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # החזר עד 3 הצעות הכי טובות
        return suggestions[:3]
    
    def suggest_for_line(self, x1: int, y1: int, x2: int, y2: int) -> Dict:
        """
        מציע שיפורים לקו שצויר ידנית
        
        Args:
            x1, y1: נקודת התחלה (פיקסלים)
            x2, y2: נקודת סיום (פיקסלים)
        
        Returns:
            מידע על הקו + הצעות שיפור (snap to grid, alignment)
        """
        length_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        length_m = length_px / self.scale if self.scale > 0 else 0
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        
        # בדוק alignment עם קירות קיימים
        aligned_segments = []
        for seg in self.segments:
            seg_angle = seg.get('angle', 0)
            angle_diff = abs(seg_angle - angle)
            
            # בדוק אם מקביל (או כמעט מקביל)
            if angle_diff < 5 or angle_diff > 175:
                aligned_segments.append(seg)
        
        suggestion = {
            'measured_length_m': round(length_m, 2),
            'measured_length_px': round(length_px, 1),
            'angle': round(angle, 1),
            'is_aligned': len(aligned_segments) > 0,
            'aligned_with': aligned_segments[:2] if aligned_segments else []
        }
        
        # אם מיישר עם קירות קיימים - הצע snap
        if aligned_segments:
            avg_aligned_length = np.mean([s['length_m'] for s in aligned_segments])
            
            # אם ההפרש קטן (פחות מ-50cm) - הצע snap
            if abs(avg_aligned_length - length_m) < 0.5:
                suggestion['snap_suggestion'] = {
                    'length_m': round(avg_aligned_length, 2),
                    'reason': f'מיושר עם {len(aligned_segments)} קירות סמוכים',
                    'difference_cm': round(abs(avg_aligned_length - length_m) * 100, 1)
                }
        
        return suggestion
    
    def get_nearest_wall(self, x: int, y: int) -> Optional[Dict]:
        """
        מצא את הקיר הכי קרוב לנקודה
        
        Args:
            x, y: קואורדינטות (פיקסלים)
        
        Returns:
            מידע על הקיר הקרוב ביותר, או None
        """
        if not self.segments:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for seg in self.segments:
            dist = self._point_to_line_distance((x, y), seg['start'], seg['end'])
            
            if dist < min_dist:
                min_dist = dist
                nearest = seg
        
        if nearest:
            return {
                **nearest,
                'distance_px': round(min_dist, 1),
                'distance_m': round(min_dist / self.scale, 2) if self.scale > 0 else 0
            }
        
        return None
    
    def get_statistics(self) -> Dict:
        """
        סטטיסטיקות על הקירות שזוהו
        
        Returns:
            מילון עם נתונים סטטיסטיים
        """
        if not self.segments:
            return {
                'total_segments': 0,
                'total_length_m': 0,
                'avg_length_m': 0,
                'horizontal_count': 0,
                'vertical_count': 0,
                'diagonal_count': 0
            }
        
        total_length = sum(s.get('length_m', 0) for s in self.segments)
        horizontal = sum(1 for s in self.segments if s.get('direction') == 'horizontal')
        vertical = sum(1 for s in self.segments if s.get('direction') == 'vertical')
        diagonal = sum(1 for s in self.segments if s.get('direction') == 'diagonal')
        
        return {
            'total_segments': len(self.segments),
            'total_length_m': round(total_length, 2),
            'avg_length_m': round(total_length / len(self.segments), 2) if self.segments else 0,
            'horizontal_count': horizontal,
            'vertical_count': vertical,
            'diagonal_count': diagonal
        }
    
    @staticmethod
    def _point_to_line_distance(
        point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """
        מחשב מרחק מינימלי מנקודה לקטע קו
        
        Args:
            point: (x, y)
            line_start: (x1, y1)
            line_end: (x2, y2)
        
        Returns:
            מרחק בפיקסלים
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # נוסחת מרחק נקודה לקטע
        dx = x2 - x1
        dy = y2 - y1
        
        # אם הקטע הוא נקודה
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # מציאת הנקודה הקרובה ביותר על הקטע
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        # נקודת הקרנה
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # מרחק
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def export_segments_to_json(self) -> List[Dict]:
        """
        ייצוא כל הקטעים שזוהו ל-JSON
        
        Returns:
            רשימת קטעים במבנה JSON-friendly
        """
        return [
            {
                'start': list(seg['start']),
                'end': list(seg['end']),
                'length_m': round(seg.get('length_m', 0), 2),
                'direction': seg.get('direction', 'unknown'),
                'angle': round(seg.get('angle', 0), 1),
                'source': seg.get('source', 'analyzer')
            }
            for seg in self.segments
        ]
    
    def __repr__(self):
        return f"SmartMeasurements(segments={len(self.segments)}, scale={self.scale:.1f}px/m)"
