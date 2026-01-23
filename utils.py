import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports

def safe_process_metadata(raw_text=None, raw_text_full=None, normalized_text=None, raw_blocks=None, candidates=None):
    """
    Enhanced wrapper for brain.process_plan_metadata
    Accepts multiple text sources and chooses the best one
    Always returns full schema structure
    
    Args:
        raw_text: Original text (legacy, 3000 chars)
        raw_text_full: Full text extraction
        normalized_text: Cleaned/normalized text
        raw_blocks: List of text blocks with metadata
        candidates: Alternative text candidates
    
    Returns:
        Dict with full schema: status, document, rooms, heights_and_levels, etc.
    """
    # Choose best available text source
    best_text = None
    
    # Priority order (best to worst)
    if normalized_text and normalized_text.strip():
        best_text = normalized_text
    elif raw_text_full and raw_text_full.strip():
        best_text = raw_text_full
    elif raw_text and raw_text.strip():
        best_text = raw_text
    elif raw_blocks and isinstance(raw_blocks, list):
        # Join text from blocks
        best_text = "\n".join([
            block.get("text", "") for block in raw_blocks 
            if isinstance(block, dict) and block.get("text")
        ])
    elif candidates and isinstance(candidates, list):
        # Join candidates
        best_text = "\n".join([str(c) for c in candidates if c])
    
    # If no text available, return empty schema
    if not best_text or not best_text.strip():
        return {
            "status": "empty_text",
            "error": "No text extracted from PDF",
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": ["No text found in PDF file"],
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
        }
    
    # Try to call brain extraction
    try:
        from brain import process_plan_metadata
        result = process_plan_metadata(best_text)
        
        # Legacy wrapper - if result is flat (old format), wrap it
        if isinstance(result, dict):
            # Check if it's the new format (has "document" key)
            if "document" in result:
                return result
            else:
                # Old format - wrap it
                return _wrap_legacy_format(result)
        
        return result
        
    except (ImportError, Exception) as e:
        return {
            "status": "error",
            "error": str(e),
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": [f"Processing error: {str(e)}"],
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
        }


def _wrap_legacy_format(old_data):
    """
    Wraps old flat format into new schema
    Old: {plan_name, scale, plan_type, ...}
    New: {document: {...}, rooms: [], ...}
    """
    document = {}
    
    # Map old keys to new document structure
    if "plan_name" in old_data:
        document["plan_title"] = {
            "value": old_data["plan_name"],
            "confidence": 50,
            "evidence": ["legacy data"]
        }
    
    if "scale" in old_data:
        document["scale"] = {
            "value": old_data["scale"],
            "confidence": 50,
            "evidence": ["legacy data"]
        }
    
    if "plan_type" in old_data:
        document["plan_type"] = {
            "value": old_data["plan_type"],
            "confidence": 50,
            "evidence": ["legacy data"]
        }
    
    return {
        "status": "success_legacy",
        "document": document,
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": ["Converted from legacy format"],
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
    # המרה ל-RGB
    img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
    overlay = img_vis.copy()
    
    # ===== תיקון קריטי: התאמת גדלים =====
    h, w = original.shape[:2]
    
    # וידוא שכל המסכות באותו גודל
    if concrete_mask is not None:
        if concrete_mask.shape[:2] != (h, w):
            concrete_mask = cv2.resize(concrete_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay[concrete_mask > 0] = [30, 144, 255]
    
    if blocks_mask is not None:
        if blocks_mask.shape[:2] != (h, w):
            blocks_mask = cv2.resize(blocks_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay[blocks_mask > 0] = [255, 165, 0]
    
    if flooring_mask is not None:
        if flooring_mask.shape[:2] != (h, w):
            flooring_mask = cv2.resize(flooring_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay[flooring_mask > 0] = [200, 100, 255]
    
    # שילוב עם שקיפות
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
