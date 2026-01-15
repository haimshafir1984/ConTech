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