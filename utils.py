import cv2
import numpy as np
import pandas as pd
from database import get_progress_reports
import streamlit as st
import traceback

# ==========================================
# Metadata Processing עם Error Handling מקיף
# ==========================================

def safe_process_metadata(raw_text=None, meta=None):
    """
    ✨ משופר: Error handling מקיף + fallback mechanisms
    
    מעבד מטא-דאטה מתוכנית בניה עם 3 שכבות הגנה:
    1. ניסיון עם קונטקסט מלא (20K chars)
    2. Fallback לטקסט בסיסי (3K chars)
    3. Fallback לערכים ברירת מחדל
    """
    
    # שכבה 1: בדיקת imports
    try:
        from brain_improved import process_plan_metadata, analyze_legend_image
    except ImportError:
        try:
            from brain import process_plan_metadata, analyze_legend_image
        except ImportError:
            st.error("❌ שגיאה קריטית: brain.py חסר!")
            return {
                "plan_name": "Unknown",
                "scale": None,
                "error": "Brain module not found"
            }
    
    # שכבה 2: ניסיון עיבוד
    try:
        # אם יש meta dict מלא - נסה עם קונטקסט מלא
        if meta and isinstance(meta, dict):
            
            # בדיקת זמינות נתונים
            has_full_text = meta.get("raw_text_full") and len(meta.get("raw_text_full", "")) > 100
            has_basic_text = meta.get("raw_text") and len(meta.get("raw_text", "")) > 50
            
            if not has_full_text and not has_basic_text:
                return {
                    "plan_name": meta.get("plan_name", "Unknown"),
                    "scale": None,
                    "error": "אין טקסט זמין לניתוח",
                    "warning": "ה-PDF לא הכיל טקסט קריא"
                }
            
            # ניסיון 1: עם קונטקסט מלא
            if has_full_text:
                try:
                    with st.spinner("🧠 מנתח מטא-דאטה עם AI..."):
                        result = process_plan_metadata(meta["raw_text_full"])
                        
                        # בדיקת תקינות התוצאה
                        if result and isinstance(result, dict) and not result.get("error"):
                            result["_processing_method"] = "full_context"
                            result["_text_length"] = len(meta["raw_text_full"])
                            return result
                        else:
                            # התוצאה לא תקינה - נסה fallback
                            st.warning("⚠️ ניתוח מלא נכשל, מנסה גרסה בסיסית...")
                            raise ValueError("Invalid result from full context")
                            
                except Exception as e:
                    st.warning(f"⚠️ ניתוח מלא נכשל: {str(e)[:100]}")
                    # ממשיכים ל-fallback למטה
            
            # ניסיון 2 (Fallback): עם טקסט בסיסי
            if has_basic_text:
                try:
                    with st.spinner("🔄 מנסה ניתוח בסיסי..."):
                        result = process_plan_metadata(meta["raw_text"])
                        
                        if result and isinstance(result, dict):
                            result["_processing_method"] = "basic_context"
                            result["_text_length"] = len(meta["raw_text"])
                            result["_warning"] = "נותח עם טקסט חלקי בלבד"
                            return result
                        else:
                            raise ValueError("Invalid result from basic context")
                            
                except Exception as e:
                    st.warning(f"⚠️ גם ניתוח בסיסי נכשל: {str(e)[:100]}")
                    # ממשיכים ל-fallback סופי למטה
        
        # אם הגענו לכאן עם meta - ניסינו הכל ונכשלנו
        # או שקיבלנו raw_text ישירות (legacy mode)
        elif raw_text and isinstance(raw_text, str) and len(raw_text) > 50:
            try:
                with st.spinner("🔄 מנתח טקסט..."):
                    result = process_plan_metadata(raw_text)
                    
                    if result and isinstance(result, dict):
                        result["_processing_method"] = "legacy"
                        return result
                    else:
                        raise ValueError("Invalid result")
                        
            except Exception as e:
                st.error(f"❌ ניתוח נכשל: {str(e)[:150]}")
                # ממשיכים ל-fallback למטה
        
        # אם הגענו לכאן - כל הניסיונות נכשלו
        st.error("❌ כל שיטות הניתוח נכשלו")
        
    except Exception as e:
        # שגיאה לא צפויה
        st.error(f"❌ שגיאה לא צפויה: {str(e)}")
        with st.expander("🔍 פרטי שגיאה מלאים"):
            st.code(traceback.format_exc())
    
    # שכבה 3: Fallback סופי - ערכים ברירת מחדל
    st.warning("⚠️ משתמש בערכי ברירת מחדל")
    
    fallback_result = {
        "plan_name": "Unknown Plan",
        "scale": None,
        "plan_type": "unknown",
        "_processing_method": "fallback",
        "_error": "כל שיטות הניתוח נכשלו",
        "_suggestion": "נסה להעלות תוכנית עם טקסט ברור יותר"
    }
    
    # נסה לחלץ שם מה-meta אם יש
    if meta and isinstance(meta, dict):
        if meta.get("plan_name"):
            fallback_result["plan_name"] = meta["plan_name"]
    
    return fallback_result


def safe_analyze_legend(image_bytes):
    """
    ✨ משופר: ניתוח מקרא עם Error Handling + Retry logic
    """
    
    # בדיקות קלט
    if not image_bytes:
        return {"error": "לא התקבלה תמונה"}
    
    if len(image_bytes) < 1000:
        return {"error": "התמונה קטנה מדי (פחות מ-1KB)"}
    
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
        return {"error": "התמונה גדולה מדי (מעל 10MB)"}
    
    # ניסיון טעינת המודול
    try:
        from brain_improved import analyze_legend_image
    except ImportError:
        try:
            from brain import analyze_legend_image
        except ImportError:
            return {"error": "Brain module not found"}
    
    # ניסיון ראשון
    try:
        with st.spinner("🔍 מנתח מקרא עם AI..."):
            result = analyze_legend_image(image_bytes)
            
            # בדיקת תקינות
            if result and isinstance(result, dict):
                if result.get("error"):
                    # יש שגיאה - נסה retry
                    st.warning("⚠️ ניסיון ראשון נכשל, מנסה שוב...")
                    raise ValueError(result["error"])
                else:
                    # הצלחה!
                    st.success("✅ ניתוח הושלם בהצלחה")
                    return result
            else:
                raise ValueError("Invalid result format")
                
    except Exception as e:
        st.warning(f"⚠️ ניסיון ראשון נכשל: {str(e)[:100]}")
        
        # ניסיון שני (Retry)
        try:
            with st.spinner("🔄 מנסה שוב..."):
                import time
                time.sleep(1)  # המתנה קצרה
                
                result = analyze_legend_image(image_bytes)
                
                if result and isinstance(result, dict) and not result.get("error"):
                    st.success("✅ ניתוח הושלם בניסיון השני")
                    result["_retry_count"] = 1
                    return result
                else:
                    raise ValueError("Second attempt failed")
                    
        except Exception as e2:
            st.error(f"❌ גם ניסיון שני נכשל: {str(e2)[:100]}")
            
            # החזרת שגיאה מפורטת
            return {
                "error": "ניתוח נכשל פעמיים",
                "first_error": str(e)[:200],
                "second_error": str(e2)[:200],
                "_suggestion": "נסה:\n1. לחתוך את המקרא ידנית\n2. להעלות תמונה באיכות גבוהה יותר\n3. לבדוק שהמקרא כולל טקסט ברור בעברית",
                "_fallback_action": "ניתן למלא את הנתונים ידנית"
            }


# ==========================================
# Utility Functions
# ==========================================

def load_stats_df():
    """טוען סטטיסטיקות עם Error Handling"""
    try:
        reports = get_progress_reports()
        if reports and len(reports) > 0:
            df = pd.DataFrame(reports)
            return df.rename(columns={
                'date': 'תאריך', 
                'plan_name': 'שם תוכנית',
                'meters_built': 'כמות שבוצעה', 
                'note': 'הערה'
            })
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ שגיאה בטעינת סטטיסטיקות: {str(e)}")
        return pd.DataFrame()


def create_colored_overlay(original, concrete_mask, blocks_mask, flooring_mask=None):
    """
    יוצר תמונה צבעונית המשלבת את התוכנית המקורית עם השכבות שזוהו
    
    ✨ משופר: Error Handling + validation
    """
    
    # בדיקות קלט
    if original is None or original.size == 0:
        st.error("❌ תמונה מקורית חסרה")
        return np.zeros((500, 500, 3), dtype=np.uint8)
    
    try:
        # המרה ל-RGB (פורמט שהמסך יודע להציג)
        if len(original.shape) == 2:
            img_vis = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB).astype(float)
        elif original.shape[2] == 4:
            img_vis = cv2.cvtColor(original, cv2.COLOR_BGRA2RGB).astype(float)
        else:
            img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
        
        overlay = img_vis.copy()
        
        # צביעת בטון (כחול) - רק אם יש מסכה תקינה
        if concrete_mask is not None and concrete_mask.size > 0:
            try:
                # ודא שהגדלים תואמים
                if concrete_mask.shape[:2] == img_vis.shape[:2]:
                    overlay[concrete_mask > 0] = [30, 144, 255]
                else:
                    concrete_mask_resized = cv2.resize(concrete_mask, 
                                                       (img_vis.shape[1], img_vis.shape[0]))
                    overlay[concrete_mask_resized > 0] = [30, 144, 255]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת בטון: {str(e)}")
        
        # צביעת בלוקים (כתום)
        if blocks_mask is not None and blocks_mask.size > 0:
            try:
                if blocks_mask.shape[:2] == img_vis.shape[:2]:
                    overlay[blocks_mask > 0] = [255, 165, 0]
                else:
                    blocks_mask_resized = cv2.resize(blocks_mask, 
                                                     (img_vis.shape[1], img_vis.shape[0]))
                    overlay[blocks_mask_resized > 0] = [255, 165, 0]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת בלוקים: {str(e)}")
        
        # צביעת ריצוף (סגול בהיר) - אם נבחר להציג
        if flooring_mask is not None and flooring_mask.size > 0:
            try:
                if flooring_mask.shape[:2] == img_vis.shape[:2]:
                    overlay[flooring_mask > 0] = [200, 100, 255]
                else:
                    flooring_mask_resized = cv2.resize(flooring_mask, 
                                                       (img_vis.shape[1], img_vis.shape[0]))
                    overlay[flooring_mask_resized > 0] = [200, 100, 255]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת ריצוף: {str(e)}")
        
        # שילוב עם שקיפות (60% מקור, 40% צבע)
        result = img_vis.copy()
        cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, result)
        
        return result.astype(np.uint8)
        
    except Exception as e:
        st.error(f"❌ שגיאה ביצירת overlay: {str(e)}")
        with st.expander("🔍 פרטי שגיאה"):
            st.code(traceback.format_exc())
        
        # fallback - החזר תמונה מקורית
        if len(original.shape) == 3:
            return cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
