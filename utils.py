import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports

def safe_process_metadata(raw_text):
    try:
        from brain import process_plan_metadata
        return process_plan_metadata(raw_text)
    except (ImportError, Exception):
        return {}

def safe_analyze_legend(image_bytes):
    try:
        from brain import analyze_legend_image
        return analyze_legend_image(image_bytes)
    except Exception as e:
        return f"Error: {str(e)}"

def load_stats_df():
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


# ==========================================
# 🆕 פונקציות חדשות להצגת LLM Data
# ==========================================

def unwrap_field(field):
    """
    מחלץ את הערך מתוך מבנה {value, confidence, evidence}
    
    Args:
        field: יכול להיות dict עם value, או ערך רגיל
    
    Returns:
        הערך עצמו
    """
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def format_llm_metadata(llm_data):
    """
    ממיר LLM data מורכב לפורמט נקי ופשוט
    
    Args:
        llm_data: dict עם document, rooms, וכו'
    
    Returns:
        dict פשוט עם רק הערכים
    """
    if not llm_data or not isinstance(llm_data, dict):
        return {}
    
    # מבנה פשוט
    pretty = {}
    
    # חלק 1: Document metadata
    if "document" in llm_data:
        pretty["document"] = {
            k: unwrap_field(v) 
            for k, v in llm_data.get("document", {}).items()
        }
    
    # חלק 2: Rooms (אם יש)
    if "rooms" in llm_data and isinstance(llm_data["rooms"], list):
        pretty["rooms"] = [
            {
                "name": unwrap_field(r.get("name")),
                "area_m2": unwrap_field(r.get("area_m2")),
                "ceiling_height_m": unwrap_field(r.get("ceiling_height_m")),
                "ceiling_notes": unwrap_field(r.get("ceiling_notes")),
                "flooring_notes": unwrap_field(r.get("flooring_notes")),
                "other_notes": unwrap_field(r.get("other_notes")),
            }
            for r in llm_data.get("rooms", [])
        ]
    
    return pretty


def get_simple_metadata_values(llm_data):
    """
    מחלץ רק את הערכים הפשוטים (ללא confidence/evidence)
    מחזיר dict שטוח שאפשר לעדכן ישירות ל-metadata
    
    Args:
        llm_data: dict מלא מ-LLM
    
    Returns:
        dict שטוח עם ערכים בלבד
    """
    pretty = format_llm_metadata(llm_data)
    
    # שטח - רק הערכים החשובים
    simple = {}
    
    if "document" in pretty:
        doc = pretty["document"]
        
        # שדות שאנחנו רוצים להעתיק ישירות
        fields_to_copy = [
            "plan_title", "plan_name", "plan_type", 
            "scale", "date", "floor_or_level", 
            "project_name", "sheet_numbers"
        ]
        
        for field in fields_to_copy:
            if field in doc and doc[field] is not None:
                simple[field] = doc[field]
    
    return simple
