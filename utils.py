import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports

def safe_process_metadata(raw_text):
    """
    Wrapper function for brain.process_plan_metadata
    Handles import errors gracefully
    """
    try:
        from brain import process_plan_metadata
        return process_plan_metadata(raw_text)
    except (ImportError, Exception) as e:
        return {
            "status": "error",
            "error": str(e),
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": [f"שגיאה בעיבוד: {str(e)}"],
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
        }

def safe_analyze_legend(image_bytes):
    """
    Wrapper function for brain.analyze_legend_image
    Handles import errors gracefully
    """
    try:
        from brain import analyze_legend_image
        return analyze_legend_image(image_bytes)
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def load_stats_df():
    """Load progress reports as DataFrame"""
    reports = get_progress_reports()
    if reports:
        df = pd.DataFrame(reports)
        return df.rename(columns={
            'date': 'תאריך', 'plan_name': 'שם תוכנית',
            'meters_built': 'כמות שבוצעה', 'note': 'הערה'
        })
    return pd.DataFrame()

def create_colored_overlay(original, concrete_mask, blocks_mask, flooring_mask=None):
    """
    יוצר תמונה צבעונית המשלבת את התוכנית המקורית עם השכבות שזוהו
    """
    # המרה ל-RGB (פורמט שהמסך יודע להציג)
    img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
    overlay = img_vis.copy()
    
    # צביעת בטון (כחול)
    if concrete_mask is not None:
        overlay[concrete_mask > 0] = [30, 144, 255] 
    
    # צביעת בלוקים (כתום)
    if blocks_mask is not None:
        overlay[blocks_mask > 0] = [255, 165, 0]
    
    # צביעת ריצוף (סגול בהיר) - אם נבחר להציג
    if flooring_mask is not None:
         overlay[flooring_mask > 0] = [200, 100, 255]
    
    # שילוב עם שקיפות (60% מקור, 40% צבע)
    cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, img_vis)
    return img_vis.astype(np.uint8)


def format_llm_metadata(llm_data):
    """
    ממיר את המטא-דאטה המלא למבנה פשוט יותר לתצוגה
    
    Args:
        llm_data: המילון המלא עם value/confidence/evidence
    
    Returns:
        מילון פשוט עם ערכים נקיים
    """
    if not llm_data or llm_data.get("status") in ["error", "no_api_key", "empty_text", "extraction_failed"]:
        return {
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": llm_data.get("limitations", []) if llm_data else [],
            "quantities_hint": llm_data.get("quantities_hint", {}) if llm_data else {}
        }
    
    def extract_value(field_obj):
        """חילוץ הערך מתוך אובייקט {value, confidence, evidence}"""
        if isinstance(field_obj, dict) and "value" in field_obj:
            return field_obj["value"]
        return field_obj  # אם זה כבר ערך פשוט
    
    # Document fields
    document = {}
    if "document" in llm_data and isinstance(llm_data["document"], dict):
        for key, field in llm_data["document"].items():
            document[key] = extract_value(field)
    
    # Rooms
    rooms = []
    if "rooms" in llm_data and isinstance(llm_data["rooms"], list):
        for room in llm_data["rooms"]:
            if isinstance(room, dict):
                simple_room = {}
                for key, field in room.items():
                    simple_room[key] = extract_value(field)
                rooms.append(simple_room)
    
    # Heights and levels
    heights_and_levels = {}
    if "heights_and_levels" in llm_data and isinstance(llm_data["heights_and_levels"], dict):
        for key, field in llm_data["heights_and_levels"].items():
            heights_and_levels[key] = extract_value(field)
    
    # Execution notes
    execution_notes = {}
    if "execution_notes" in llm_data and isinstance(llm_data["execution_notes"], dict):
        for key, field in llm_data["execution_notes"].items():
            execution_notes[key] = extract_value(field)
    
    # Limitations (already simple array)
    limitations = llm_data.get("limitations", [])
    
    # Quantities hint (already simple)
    quantities_hint = llm_data.get("quantities_hint", {
        "wall_types_mentioned": [],
        "material_hints": []
    })
    
    return {
        "document": document,
        "rooms": rooms,
        "heights_and_levels": heights_and_levels,
        "execution_notes": execution_notes,
        "limitations": limitations,
        "quantities_hint": quantities_hint
    }


def get_simple_metadata_values(llm_data):
    """
    מחלץ ערכים פשוטים למטא-דאטה הישנה (backward compatibility)
    
    Args:
        llm_data: המטא-דאטה המלא עם confidence
    
    Returns:
        מילון עם plan_name, scale וכו' כערכים פשוטים
    """
    if not llm_data or llm_data.get("status") in ["error", "no_api_key", "empty_text", "extraction_failed"]:
        return {}
    
    simple = {}
    
    # Extract from document
    if "document" in llm_data and isinstance(llm_data["document"], dict):
        doc = llm_data["document"]
        
        # plan_name
        if "plan_title" in doc and isinstance(doc["plan_title"], dict):
            title = doc["plan_title"].get("value")
            if title:
                simple["plan_name"] = title
        
        # scale
        if "scale" in doc and isinstance(doc["scale"], dict):
            scale = doc["scale"].get("value")
            if scale:
                simple["scale"] = scale
        
        # plan_type
        if "plan_type" in doc and isinstance(doc["plan_type"], dict):
            ptype = doc["plan_type"].get("value")
            if ptype:
                simple["plan_type"] = ptype
        
        # date
        if "date" in doc and isinstance(doc["date"], dict):
            date = doc["date"].get("value")
            if date:
                simple["date"] = date
        
        # floor_or_level
        if "floor_or_level" in doc and isinstance(doc["floor_or_level"], dict):
            floor = doc["floor_or_level"].get("value")
            if floor:
                simple["floor_or_level"] = floor
        
        # project_name
        if "project_name" in doc and isinstance(doc["project_name"], dict):
            proj = doc["project_name"].get("value")
            if proj:
                simple["project_name"] = proj
    
    return simple
