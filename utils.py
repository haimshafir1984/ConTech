import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports

def safe_process_metadata(raw_text):
    """
    wrapper ל-brain.safe_process_metadata
    מחזיר תמיד סכמה מלאה
    """
    try:
        from brain import safe_process_metadata as brain_extract
        result = brain_extract(raw_text)
        
        # אם זה פורמט ישן (flat), המר לסכמה חדשה
        if "document" not in result and ("plan_name" in result or "scale" in result):
            result = _wrap_legacy_to_schema(result)
        
        return result
    except (ImportError, Exception) as e:
        return {
            "status": "error",
            "error": str(e),
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": [f"שגיאה בעיבוד: {str(e)}"],
            "quantities_hint": {}
        }


def _wrap_legacy_to_schema(flat_data):
    """ממיר פורמט ישן לסכמה החדשה"""
    return {
        "status": "legacy_format",
        "document": {
            "plan_title": {
                "value": flat_data.get("plan_name"),
                "confidence": 50,
                "evidence": []
            },
            "plan_type": {
                "value": flat_data.get("plan_type", "other"),
                "confidence": 50,
                "evidence": []
            },
            "scale": {
                "value": flat_data.get("scale"),
                "confidence": 50,
                "evidence": []
            }
        },
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": ["המרה מפורמט ישן - אין נתוני חדרים"],
        "quantities_hint": {},
        "_model_used": flat_data.get("_model_used", "unknown")
    }


def safe_analyze_legend(image_bytes):
    try:
        from brain import analyze_legend_image
        return analyze_legend_image(image_bytes)
    except Exception as e:
        return {"error": str(e)}


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
    """יוצר תמונה צבעונית"""
    img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
    overlay = img_vis.copy()
    
    if concrete_mask is not None:
        overlay[concrete_mask > 0] = [30, 144, 255] 
    
    if blocks_mask is not None:
        overlay[blocks_mask > 0] = [255, 165, 0]
    
    if flooring_mask is not None:
         overlay[flooring_mask > 0] = [200, 100, 255]
    
    cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, img_vis)
    return img_vis.astype(np.uint8)


def unwrap_field(field):
    """מחלץ ערך מתוך {value, confidence, evidence}"""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def format_llm_metadata(llm_data):
    """ממיר לפורמט פשוט לתצוגה"""
    if not llm_data or not isinstance(llm_data, dict):
        return {}
    
    pretty = {}
    
    if "document" in llm_data:
        pretty["document"] = {
            k: unwrap_field(v) 
            for k, v in llm_data.get("document", {}).items()
        }
    
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
    
    # שדות נוספים
    for key in ["heights_and_levels", "execution_notes", "limitations", "quantities_hint"]:
        if key in llm_data:
            pretty[key] = llm_data[key]
    
    return pretty


def get_simple_metadata_values(llm_data):
    """מחלץ ערכים פשוטים לעדכון metadata"""
    pretty = format_llm_metadata(llm_data)
    simple = {}
    
    if "document" in pretty:
        doc = pretty["document"]
        fields_to_copy = [
            "plan_title", "plan_name", "plan_type", 
            "scale", "date", "floor_or_level", "project_name"
        ]
        
        for field in fields_to_copy:
            if field in doc and doc[field] is not None:
                simple[field] = doc[field]
    
    return simple
