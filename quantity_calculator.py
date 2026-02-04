"""
Quantity Calculator
חישוב כמויות חומרי בנייה אוטומטי

Phase 1: Smart Measurements + Quantity Calculator
"""

from typing import List, Dict
import numpy as np

try:
    from building_elements import Wall
except ImportError:
    # Fallback אם building_elements לא זמין
    Wall = None


class QuantityCalculator:
    """מחשבון כמויות חומרי בנייה"""

    # ==========================================
    # קבועים - מבוססים על תקנים ישראליים
    # ==========================================

    # בלוקים
    config = {"blocks_per_sqm": 12.5, "waste_factor": 1.05}
    # בלוקים למ"ר (בלוק 20cm)
    MORTAR_PER_SQM = 0.015  # מ"ק מלט למ"ר קיר
    BLOCK_WEIGHT_KG = 18  # משקל בלוק ממוצע (ק"ג)
    BLOCKS_PER_PALLET = 72  # בלוקים בפלטה סטנדרטית

    # בטון
    CONCRETE_DENSITY = 2.4  # טון/מ"ק (בטון רגיל)
    CONCRETE_TRUCK_VOLUME = 6  # מ"ק למיקסר סטנדרטי

    # עלויות ברירת מחדל (ניתן לעדכון)
    DEFAULT_COST_PER_BLOCK = 8  # ₪ לבלוק
    DEFAULT_COST_PER_M3_CONCRETE = 450  # ₪ למ"ק בטון
    DEFAULT_COST_PER_M3_MORTAR = 350  # ₪ למ"ק מלט

    def __init__(self):
        self.walls: List[Wall] = []
        self.custom_costs = {}  # עלויות מותאמות אישית

    def add_wall(self, wall: Wall):
        """הוסף קיר לחישוב"""
        if Wall is None:
            raise ImportError("building_elements.Wall לא זמין")
        self.walls.append(wall)

    def set_custom_costs(self, **costs):
        """
        עדכן עלויות מותאמות אישית

        דוגמה:
            calc.set_custom_costs(
                cost_per_block=10,
                cost_per_m3_concrete=500
            )
        """
        self.custom_costs.update(costs)

    def _get_cost(self, key: str, default: float) -> float:
        """קבל עלות (מותאמת או ברירת מחדל)"""
        return self.custom_costs.get(key, default)

    def calculate_blocks(self) -> Dict:
    """חישוב בלוקים"""
    block_walls = [w for w in self.walls if w.material == "בלוקים"]
    
    if not block_walls:
        return {...}  # empty result
    
    total_area = sum(w.area for w in block_walls)
    
    # ← שימוש ב-config!
    blocks_per_sqm = self.config.get('blocks_per_sqm', 12.5)
    waste_factor = self.config.get('waste_factor', 1.05)
    
    # החל waste factor
    total_area_with_waste = total_area * waste_factor
    total_blocks = total_area_with_waste * blocks_per_sqm
    total_mortar = total_area_with_waste * self.MORTAR_PER_SQM
    
    return {
        'wall_count': len(block_walls),
        'total_area_sqm': round(total_area, 2),
        'total_area_with_waste': round(total_area_with_waste, 2),
        'waste_factor': waste_factor,
        'blocks_needed': int(np.ceil(total_blocks)),
        "mortar_cubic_meters": 0,
        "pallets": 0,
        "total_weight_tons": 0,
        "estimated_cost": 0,
    }

        total_area = sum(w.area for w in block_walls)
        total_blocks = total_area * self.BLOCKS_PER_SQM
        total_mortar = total_area * self.MORTAR_PER_SQM
        total_weight = (total_blocks * self.BLOCK_WEIGHT_KG) / 1000  # טון

        # עלויות
        cost_per_block = self._get_cost("cost_per_block", self.DEFAULT_COST_PER_BLOCK)
        cost_per_m3_mortar = self._get_cost(
            "cost_per_m3_mortar", self.DEFAULT_COST_PER_M3_MORTAR
        )

        blocks_cost = total_blocks * cost_per_block
        mortar_cost = total_mortar * cost_per_m3_mortar

        return {
            "wall_count": len(block_walls),
            "total_area_sqm": round(total_area, 2),
            "blocks_needed": int(np.ceil(total_blocks)),
            "mortar_cubic_meters": round(total_mortar, 3),
            "pallets": int(np.ceil(total_blocks / self.BLOCKS_PER_PALLET)),
            "total_weight_tons": round(total_weight, 2),
            "estimated_cost": round(blocks_cost + mortar_cost, 2),
        }

    def calculate_concrete(self) -> Dict:
        """
        חישוב כמויות בטון

        Returns:
            מילון עם כל הנתונים הדרושים
        """
        concrete_walls = [w for w in self.walls if w.material == "בטון"]

        if not concrete_walls:
            return {
                "wall_count": 0,
                "total_volume_cubic_meters": 0,
                "total_weight_tons": 0,
                "trucks_needed": 0,
                "estimated_cost": 0,
            }

        total_volume = sum(w.volume for w in concrete_walls)
        total_weight = total_volume * self.CONCRETE_DENSITY

        # עלויות
        cost_per_m3 = self._get_cost(
            "cost_per_m3_concrete", self.DEFAULT_COST_PER_M3_CONCRETE
        )

        return {
            "wall_count": len(concrete_walls),
            "total_volume_cubic_meters": round(total_volume, 2),
            "total_weight_tons": round(total_weight, 2),
            "trucks_needed": int(np.ceil(total_volume / self.CONCRETE_TRUCK_VOLUME)),
            "estimated_cost": round(total_volume * cost_per_m3, 2),
        }

    def calculate_all(self) -> Dict:
        """
        חישוב כללי של כל החומרים

        Returns:
            מילון מלא עם סיכומים וחישובים מפורטים
        """
        blocks = self.calculate_blocks()
        concrete = self.calculate_concrete()

        # סיכום כללי
        total_length = sum(w.length for w in self.walls)
        total_area = sum(w.area for w in self.walls)
        total_volume = sum(w.volume for w in self.walls)

        # עלות כוללת
        total_cost = blocks["estimated_cost"] + concrete["estimated_cost"]

        return {
            "summary": {
                "total_walls": len(self.walls),
                "total_length_m": round(total_length, 2),
                "total_area_sqm": round(total_area, 2),
                "total_volume_m3": round(total_volume, 2),
                "total_estimated_cost": round(total_cost, 2),
            },
            "blocks": blocks,
            "concrete": concrete,
            "flooring": self.calculate_flooring(),    # ← חדש
            "plaster": self.calculate_plaster(),      # ← חדש
            "paint": self.calculate_paint(),          # ← חדש
            "insulation": self.calculate_insulation(), # ← חדש
            "breakdown_by_material": self._breakdown_by_material(),
        }

    def _breakdown_by_material(self) -> Dict:
        """פירוק מפורט לפי סוג חומר"""
        breakdown = {}

        for wall in self.walls:
            material = wall.material
            if material not in breakdown:
                breakdown[material] = {
                    "count": 0,
                    "length_m": 0,
                    "area_sqm": 0,
                    "volume_m3": 0,
                }

            breakdown[material]["count"] += 1
            breakdown[material]["length_m"] += wall.length
            breakdown[material]["area_sqm"] += wall.area
            breakdown[material]["volume_m3"] += wall.volume

        # עיגול
        for material in breakdown:
            breakdown[material]["length_m"] = round(breakdown[material]["length_m"], 2)
            breakdown[material]["area_sqm"] = round(breakdown[material]["area_sqm"], 2)
            breakdown[material]["volume_m3"] = round(
                breakdown[material]["volume_m3"], 3
            )

        return breakdown

    def get_shopping_list(self) -> Dict:
        """
        רשימת קניות מפורטת

        Returns:
            רשימה של כל הפריטים שצריך לרכוש
        """
        blocks = self.calculate_blocks()
        concrete = self.calculate_concrete()

        shopping_list = []

        # בלוקים
        if blocks["blocks_needed"] > 0:
            shopping_list.append(
                {
                    "item": "בלוקי בנייה 20cm",
                    "quantity": blocks["blocks_needed"],
                    "unit": "יח'",
                    "note": f"{blocks['pallets']} פלטות",
                }
            )

            shopping_list.append(
                {
                    "item": "מלט",
                    "quantity": blocks["mortar_cubic_meters"],
                    "unit": 'מ"ק',
                    "note": "לחיבור בלוקים",
                }
            )

        # בטון
        if concrete["total_volume_cubic_meters"] > 0:
            shopping_list.append(
                {
                    "item": "בטון מוכן",
                    "quantity": concrete["total_volume_cubic_meters"],
                    "unit": 'מ"ק',
                    "note": f"{concrete['trucks_needed']} מיקסרים",
                }
            )

        return {"items": shopping_list, "total_items": len(shopping_list)}

    # ==========================================
    # פונקציות מורחבות - תיקון מהיר
    # ==========================================
    
    def calculate_flooring(self) -> Dict:
        """חישוב ריצוף/חיפוי"""
        flooring_walls = [w for w in self.walls if 'ריצוף' in w.material.lower() or 'חיפוי' in w.material.lower()]
        
        if not flooring_walls:
            return {'wall_count': 0, 'total_area_sqm': 0, 'tiles_needed': 0}
        
        total_area = sum([w.length * w.height for w in flooring_walls])
        waste_factor = self.config.get('waste_factor', 1.05)
        area_with_waste = total_area * waste_factor
        
        # אריחים - לפי גודל אריח (ברירת מחדל: 60x60cm)
        tile_size = self.config.get('tile_size_sqm', 0.36)
        tiles_needed = int(area_with_waste / tile_size) + 1
        
        # אריזות (בדרך כלל 4 אריחים באריזה)
        tiles_per_box = self.config.get('tiles_per_box', 4)
        boxes_needed = int(tiles_needed / tiles_per_box) + 1
        
        return {
            'wall_count': len(flooring_walls),
            'total_area_sqm': round(total_area, 2),
            'total_area_with_waste': round(area_with_waste, 2),
            'tiles_needed': tiles_needed,
            'boxes_needed': boxes_needed,
            'tile_size_sqm': tile_size,
            'waste_factor': waste_factor
        }
    
    def calculate_plaster(self) -> Dict:
        """חישוב טיח (שתי פאות)"""
        # טיח על שתי הפאות של כל קיר
        all_walls_area = sum([w.length * w.height * 2 for w in self.walls])
        
        if all_walls_area == 0:
            return {'wall_count': 0, 'total_area_sqm': 0, 'volume_cubic_meters': 0, 'bags_needed': 0}
        
        plaster_thickness = self.config.get('plaster_thickness_m', 0.015)  # 1.5 ס"מ
        waste_factor = self.config.get('waste_factor', 1.05)
        
        volume = all_walls_area * plaster_thickness * waste_factor
        
        # שקים של טיח (1 שק כיסוי ל-4 מ"ר בעובי 1.5 ס"מ)
        coverage_per_bag = self.config.get('plaster_coverage_sqm_per_bag', 4.0)
        bags_needed = int(all_walls_area / coverage_per_bag) + 1
        
        return {
            'wall_count': len(self.walls),
            'total_area_sqm': round(all_walls_area, 2),
            'volume_cubic_meters': round(volume, 3),
            'bags_needed': bags_needed,
            'thickness_m': plaster_thickness,
            'coverage_per_bag': coverage_per_bag
        }
    
    def calculate_paint(self) -> Dict:
        """חישוב צבע (שתי פאות, מספר שכבות)"""
        # צבע על שתי הפאות
        all_walls_area = sum([w.length * w.height * 2 for w in self.walls])
        
        if all_walls_area == 0:
            return {'wall_count': 0, 'total_area_sqm': 0, 'liters_needed': 0, 'buckets_needed': 0}
        
        # ברירת מחדל: 1 ליטר ל-10 מ"ר, 2 שכבות
        coverage_per_liter = self.config.get('paint_coverage_sqm_per_liter', 10.0)
        coats = self.config.get('paint_coats', 2)
        
        liters_needed = (all_walls_area * coats) / coverage_per_liter
        
        # דליים (בדרך כלל 18 ליטר לדלי)
        bucket_size = self.config.get('paint_bucket_size_liters', 18.0)
        buckets_needed = int(liters_needed / bucket_size) + 1
        
        return {
            'wall_count': len(self.walls),
            'total_area_sqm': round(all_walls_area, 2),
            'liters_needed': round(liters_needed, 1),
            'buckets_needed': buckets_needed,
            'coats': coats,
            'coverage_per_liter': coverage_per_liter
        }
    
    def calculate_insulation(self) -> Dict:
        """חישוב בידוד תרמי/אקוסטי"""
        # בידוד - לפי דרישה
        insulation_walls = [w for w in self.walls if 'בידוד' in w.material.lower()]
        
        if not insulation_walls:
            return {'wall_count': 0, 'total_area_sqm': 0, 'panels_needed': 0}
        
        total_area = sum([w.length * w.height for w in insulation_walls])
        waste_factor = self.config.get('waste_factor', 1.05)
        area_with_waste = total_area * waste_factor
        
        # פאנלים (ברירת מחדל: 1.2x0.6m = 0.72 מ"ר)
        panel_size = self.config.get('insulation_panel_sqm', 0.72)
        panels_needed = int(area_with_waste / panel_size) + 1
        
        return {
            'wall_count': len(insulation_walls),
            'total_area_sqm': round(total_area, 2),
            'total_area_with_waste': round(area_with_waste, 2),
            'panels_needed': panels_needed,
            'panel_size_sqm': panel_size
        }

    def reset(self):
        """אפס את כל הנתונים"""
        self.walls.clear()
        self.custom_costs.clear()

    def __repr__(self):
        return f"QuantityCalculator(walls={len(self.walls)})"
