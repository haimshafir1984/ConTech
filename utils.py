import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports

def safe_process_metadata(raw_text=None, meta=None):
    """
    Enhanced wrapper for metadata processing with full text support.
    
    Args:
        raw_text: Legacy - short text for backward compatibility
        meta: Full metadata dict from analyzer with raw_text_full, normalized_text, raw_blocks
    """
    try:
        from brain import safe_process_metadata as brain_process
        from extractor import ArchitecturalTextExtractor
        
        # If meta dict provided, use enhanced extraction
        if meta and isinstance(meta, dict):
            raw_text_full = meta.get("raw_text_full")
            normalized_text = meta.get("normalized_text")
            raw_blocks = meta.get("raw_blocks")
            
            # Extract candidates if we have good text
            candidates = None
            text_to_extract = normalized_text or raw_text_full or meta.get("raw_text")
            if text_to_extract and len(text_to_extract) > 50:
                try:
                    extractor = ArchitecturalTextExtractor()
                    candidates = extractor.extract_candidates(text_to_extract)
                except:
                    candidates = None
            
            return brain_process(
                raw_text=meta.get("raw_text"),
                raw_text_full=raw_text_full,
                normalized_text=normalized_text,
                raw_blocks=raw_blocks,
                candidates=candidates
            )
        else:
            # Legacy mode - just use raw_text
            return brain_process(raw_text=raw_text)
            
    except (ImportError, Exception) as e:
        return {"error": str(e), "status": "extraction_failed"}

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