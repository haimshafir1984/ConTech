import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from analyzer import FloorPlanAnalyzer
import tempfile
import os
import json
import base64
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# --- הגדרות מערכת וייבוא בטוח ---
try:
    from database import (
        init_database, save_plan, save_progress_report, 
        get_progress_reports, get_plan_by_filename, get_plan_by_id, get_all_plans,
        get_project_forecast, 
        calculate_material_estimates, get_project_financial_status, reset_all_data
    )
except ImportError as e:
    st.error(f"שגיאה בטעינת מסד הנתונים: {e}")
    st.stop()

from brain import learn_from_confirmation, process_plan_metadata
from datetime import datetime

# הגדרות תמונה
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- פונקציית עזר להמרת תמונה ל-Base64 (העקיפה) ---
def image_to_base64(image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return None

def load_stats_df():
    reports = get_progress_reports()
    if reports:
        df = pd.DataFrame(reports)
        return df.rename(columns={
            'date': 'תאריך', 'plan_name': 'שם תוכנית',
            'meters_built': 'מטרים שבוצעו', 'note': 'הערה'
        })
    return pd.DataFrame()

st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Heebo', sans-serif; direction: rtl; }
    .stCard { background-color: white; padding: 24px; border-radius: 12px; border: 1px solid #E0E0E0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 20px; }
    .kpi-container { display: flex; flex-direction: column; background: white; padding: 20px; border-radius: 12px; border: 1px solid #EAEAEA; box-shadow: 0 4px 12px rgba(0,0,0,0.03); height: 100%; }
    .mat-card { text-align: center; background: white; border: 1px solid #EEE; border-radius: 10px; padding: 15px; }
    .mat-val { font-size: 20px; font-weight: bold; color: #0F62FE; }
    .mat-lbl { font-size: 14px; color: #666; }
</style>
""", unsafe_allow_html=True)

if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

with st.sidebar:
    st.markdown("### **ConTech Pro**")
    mode = st.radio("בחר אזור עבודה:", ["🏢 מנהל פרויקט", "👷 דיווח שטח"])
    st.markdown("---")
    if st.button("🗑️ איפוס מערכת מלא"):
        reset_all_data()
        st.session_state.projects = {}
        st.rerun()

# --- לוגיקה ---

if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2 = st.tabs(["📂 העלאת תוכניות", "📊 דשבורד מנהלים"])
    
    with tab1:
        files = st.file_uploader("העלה תוכניות (PDF)", type="pdf", accept_multiple_files=True)
        if files:
            for f in files:
                if f.name not in st.session_state.projects:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        path = tmp.name
                    
                    analyzer = FloorPlanAnalyzer()
                    pix, skel, thick, orig, meta = analyzer.process_file(path)
                    
                    if not meta.get("plan_name"): meta["plan_name"] = f.name.replace(".pdf", "")
                    
                    st.session_state.projects[f.name] = {
                        "skeleton": skel, "thick_walls": thick, "original": orig,
                        "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                        "total_length": pix / 200.0
                    }
                    os.unlink(path)

        if st.session_state.projects:
            selected = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            c1, c2 = st.columns([1, 2])
            with c1:
                scale = st.slider("פיקסלים למטר", 10.0, 500.0, float(proj["scale"]))
                proj["scale"] = scale
                proj["total_length"] = proj["raw_pixels"] / scale
                st.info(f"אורך מזוהה: {proj['total_length']:.2f} מ'")
                
                if st.button("שמור"):
                    from database import save_plan
                    # יצירת מטא-דאטה בסיסי לשמירה
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    mats = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    save_plan(selected, proj["metadata"]["plan_name"], "", scale, proj["raw_pixels"], meta_json, None, 0, 0, json.dumps(mats))
                    st.success("נשמר")
            
            with c2:
                # שימוש בפרמטר הישן והבטוח
                st.image(proj["skeleton"], caption="זיהוי קירות", use_column_width=True)

    with tab2:
        plans = get_all_plans()
        if plans:
            p_opts = [f"{p['plan_name']} (ID: {p['id']})" for p in plans]
            sel = st.selectbox("בחר פרויקט:", p_opts)
            if sel:
                pid = int(sel.split("ID: ")[1].replace(")", ""))
                fc = get_project_forecast(pid)
                fin = get_project_financial_status(pid)
                
                k1, k2, k3 = st.columns(3)
                k1.metric("בוצע", f"{fc['cumulative_progress']:.1f} מ'")
                k2.metric("נותר", f"{fc['remaining_work']:.1f} מ'")
                # טיפול בטוח בערך "ימים לסיום"
                days = fc['days_to_finish']
                days_str = str(days) if isinstance(days, int) and days > 0 else "-"
                k3.metric("ימים לסיום", days_str)

elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    # --- אזור דיבאג ---
    with st.expander("🛠️ כלי דיבאג (פתח אם התמונה לא עולה)"):
        st.write(f"**Python Version:** {pd.__version__} (via pandas)")
        st.write(f"**Streamlit Version:** {st.__version__}")
        debug_mode = st.checkbox("הפעל מצב דיבאג מורחב")
    
    if not st.session_state.projects:
        st.info("אין תוכניות זמינות.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        # הכנת תמונה
        orig_rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = orig_rgb.shape[:2]
        
        # הדגשת קירות
        thick_walls = proj["thick_walls"]
        if thick_walls.shape[:2] != (h, w): 
            thick_walls = cv2.resize(thick_walls, (w, h), interpolation=cv2.INTER_NEAREST)
            
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate((thick_walls > 0).astype(np.uint8) * 255, kernel, iterations=2)
        
        opacity = st.slider("שקיפות הדגשה", 0.0, 1.0, 0.4)
        overlay = np.zeros_like(orig_rgb)
        overlay[dilated > 0] = [0, 120, 255]
        
        combined = cv2.addWeighted(orig_rgb, 1-opacity, overlay, opacity, 0).astype(np.uint8)
        bg_pil = Image.fromarray(combined).convert("RGB")
        
        # התאמה לגודל קנבס
        c_width = 1000
        factor = c_width / w
        c_height = int(h * factor)
        bg_resized = bg_pil.resize((c_width, c_height))
        
        # --- המרה ידנית ל-Base64 (הפתרון) ---
        bg_base64 = image_to_base64(bg_resized)
        
        if debug_mode:
            st.write("### נתוני דיבאג:")
            st.write(f"גודל תמונה מקורית: {w}x{h}")
            st.write(f"גודל קנבס: {c_width}x{c_height}")
            if bg_base64:
                st.success("✅ המרה ל-Base64 הצליחה (המחרוזת ארוכה מדי להצגה)")
                st.write(f"תחילת המחרוזת: {bg_base64[:50]}...")
            else:
                st.error("❌ נכשל בהמרת התמונה ל-Base64")
            
            st.write("תצוגת גיבוי (st.image):")
            st.image(bg_resized, use_column_width=True)

        st.markdown("**סמן את הקירות שבנית:**")
        
        try:
            # כאן אנחנו משתמשים במחרוזת ה-Base64 במקום באובייקט התמונה
            # זה מונע מ-st_canvas לנסות לעשות המרות שנכשלות
            canvas_result = st_canvas(
                stroke_width=5,
                stroke_color="#00FF00",
                background_image=bg_base64,  # מעבירים מחרוזת!
                width=c_width,
                height=c_height,
                drawing_mode="line",
                key=f"canvas_{plan_name}_{opacity}",
                update_streamlit=True
            )
            
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                # חישוב מטרים (זהה לקוד הקודם)
                w_mask = np.zeros((c_height, c_width), dtype=np.uint8)
                df_obj = pd.json_normalize(canvas_result.json_data["objects"])
                for _, obj in df_obj.iterrows():
                    if 'left' in obj:
                         p1 = (int(obj['left'] + obj.get('x1', 0)), int(obj['top'] + obj.get('y1', 0)))
                         p2 = (int(obj['left'] + obj.get('x2', 0)), int(obj['top'] + obj.get('y2', 0)))
                         cv2.line(w_mask, p1, p2, 255, 5)
                
                walls_res = cv2.resize(dilated, (c_width, c_height), interpolation=cv2.INTER_NEAREST)
                intersect = cv2.bitwise_and(w_mask, walls_res)
                meters = (cv2.countNonZero(intersect) / factor) / proj["scale"]
                
                st.success(f"✅ נמדדו: **{meters:.2f} מטר**")
                
                note = st.text_input("הערה:")
                if st.button("שלח דיווח"):
                     from database import get_plan_by_filename, save_plan
                     rec = get_plan_by_filename(plan_name)
                     # בדיקה ויצירה אם חסר (למניעת שגיאות)
                     if not rec:
                         meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                         pid = save_plan(plan_name, plan_name, "", proj["scale"], proj["raw_pixels"], meta_json)
                     else:
                         pid = rec['id']
                     
                     save_progress_report(pid, meters, note)
                     st.balloons()
        
        except Exception as e:
            st.error(f"❌ שגיאה בטעינת הקנבס: {e}")
            if debug_mode:
                st.exception(e)