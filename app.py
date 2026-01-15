import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
import io
from streamlit_drawable_canvas import st_canvas
from datetime import datetime

# ייבוא מהקבצים שלנו
from styles import setup_page, apply_css
from utils import safe_process_metadata, safe_analyze_legend, load_stats_df, create_colored_overlay
from analyzer import FloorPlanAnalyzer
from reporter import generate_status_pdf
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_all_plans,
    get_project_forecast, get_project_financial_status, 
    calculate_material_estimates, reset_all_data
)

# --- אתחול המערכת ---
setup_page()
apply_css()
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- Session State ---
if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

# --- תפריט צד (Sidebar) ---
with st.sidebar:
    st.markdown("## 🏗️")
    st.markdown("### **ConTech Pro**")
    st.caption("מערכת ניהול ובקרה חכמה")
    st.markdown("---")
    mode = st.radio("בחר אזור עבודה:", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות", expanded=False):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.success("בוצע איפוס")
            st.rerun()

# ==========================================
# 🏢 מצב מנהל פרויקט
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    
    tab1, tab2 = st.tabs(["📂 סדנת עבודה", "📊 דשבורד"])
    
    # --- טאב 1: העלאה ועריכה ---
    with tab1:
        with st.expander("העלאת קבצים חדשים", expanded=not st.session_state.projects):
            files = st.file_uploader("גרור PDF לכאן", type="pdf", accept_multiple_files=True)
            # צ'קבוקס למצב דיבאג (כאן השינוי המרכזי שרצית!)
            show_debug = st.checkbox("🔍 מצב דיבאג (הצג מה זוהה כטקסט)", value=False)

            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name}..."):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(f.getvalue())
                                    path = tmp.name
                                
                                analyzer = FloorPlanAnalyzer()
                                # קריאה לפונקציה עם פרמטר ה-Debug
                                pix, skel, thick, orig, meta, conc, blok, floor = analyzer.process_file(path, save_debug=show_debug)
                                
                                if not meta.get("plan_name"): meta["plan_name"] = f.name.replace(".pdf", "")
                                
                                # חילוץ מטא-דאטה (אופציונלי)
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(meta["raw_text"])
                                    meta.update({k: v for k, v in llm_data.items() if v})

                                st.session_state.projects[f.name] = {
                                    "skeleton": skel, "thick_walls": thick, "original": orig,
                                    "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                    "concrete_mask": conc, "blocks_mask": blok, "flooring_mask": floor,
                                    "total_length": pix/200.0
                                }
                                
                                # הצגת תמונת הדיבאג אם המשתמש ביקש
                                if show_debug and os.path.exists("debug_text_detection.png"):
                                    st.image("debug_text_detection.png", caption="אדום=טקסט שסונן | כחול=קירות שזוהו", use_column_width=True)
                                
                                os.unlink(path)
                                st.success(f"✅ {f.name} נטען")
                            except Exception as e: st.error(str(e))

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעבודה:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            c_edit, c_view = st.columns([1, 1.5], gap="large")
            
            with c_edit:
                st.markdown("#### הגדרות")
                p_name = st.text_input("שם התוכנית", value=proj["metadata"].get("plan_name", ""))
                
                # סליידרים לכיול
                scale_val = st.slider("קנה מידה (px/m)", 10.0, 1000.0, float(proj["scale"]))
                proj["scale"] = scale_val
                
                # חישובים
                total_len = proj["raw_pixels"] / scale_val
                conc_len = proj["metadata"].get("pixels_concrete", 0) / scale_val
                floor_sqm = proj["metadata"].get("pixels_flooring_area", 0) / (scale_val**2)
                proj["total_length"] = total_len
                
                st.info(f"📏 קירות: {total_len:.1f} מ' | 🔲 ריצוף: {floor_sqm:.1f} מ\"ר")
                
                # תקציב
                st.markdown("#### תקציב")
                budget = st.number_input("תקציב (₪)", value=0.0, step=1000.0)
                cost_m = st.number_input("מחיר למטר (₪)", value=st.session_state.default_cost_per_meter)
                
                if st.button("💾 שמור שינויים", type="primary", use_container_width=True):
                    proj["metadata"]["plan_name"] = p_name
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    mats = calculate_material_estimates(total_len, st.session_state.wall_height)
                    save_plan(selected, p_name, "1:50", scale_val, proj["raw_pixels"], meta_json, None, budget, cost_m, json.dumps(mats))
                    st.toast("הנתונים נשמרו!")

            with c_view:
                st.markdown("#### תצוגה")
                show_floor = st.checkbox("הצג ריצוף (סגול)", value=True)
                
                f_mask = proj["flooring_mask"] if show_floor else None
                overlay = create_colored_overlay(proj["original"], proj["concrete_mask"], proj["blocks_mask"], f_mask)
                st.image(overlay, use_column_width=True)

    # --- טאב 2: דשבורד ---
    with tab2:
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים שמורים.")
        else:
            sel_disp = st.selectbox("בחר פרויקט:", [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans])
            pid = int(sel_disp.split("(ID: ")[1].split(")")[0])
            
            fc = get_project_forecast(pid)
            fin = get_project_financial_status(pid)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='kpi-container'><div class='kpi-label'>בוצע</div><div class='kpi-value'>{fc['cumulative_progress']:.1f}</div><div class='kpi-sub'>מטרים</div></div>", unsafe_allow_html=True)
            
            pct = fc['completion_percentage'] if 'completion_percentage' in fc else 0
            k2.markdown(f"<div class='kpi-container'><div class='kpi-label'>הושלם</div><div class='kpi-value'>{pct:.1f}%</div><div class='kpi-sub'>מהיעד</div></div>", unsafe_allow_html=True)
            
            k3.markdown(f"<div class='kpi-container'><div class='kpi-label'>ימים לסיום</div><div class='kpi-value'>{fc['days_to_finish']}</div><div class='kpi-sub'>משוער</div></div>", unsafe_allow_html=True)
            
            k4.markdown(f"<div class='kpi-container'><div class='kpi-label'>עלות</div><div class='kpi-value'>{fin['current_cost']:,.0f}</div><div class='kpi-sub'>₪</div></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            df = load_stats_df()
            if not df.empty: st.bar_chart(df, x="תאריך", y="כמות שבוצעה")

# ==========================================
# 👷 מצב דיווח שטח
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.warning("אין תוכניות טעונות.")
    else:
        p_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[p_name]
        
        report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף"], horizontal=True)
        
        # הכנת תמונת רקע לקנבס
        orig_rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = orig_rgb.shape[:2]
        
        # התאמה למסך (הקטנה אם צריך)
        max_w = 800
        factor = max_w / w if w > max_w else 1.0
        new_w, new_h = int(w * factor), int(h * factor)
        
        # בחירת המסכה להדגשה
        if report_type == "🧱 בניית קירות":
            mask_to_show = proj["thick_walls"]
            draw_color = "#00FF00"
            stroke = 8
            msg = "סמן בירוק את הקירות שבנית"
        else:
            mask_to_show = proj["flooring_mask"]
            draw_color = "#FFFF00"
            stroke = 20
            msg = "צבע בצהוב את האזור שרוצף"
            
        # יצירת רקע מודגש
        mask_res = cv2.resize(mask_to_show, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(orig_rgb)
        overlay[mask_res > 0] = [0, 100, 255] # הדגשה כתומה עדינה
        bg = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)
        bg_pil = Image.fromarray(bg).resize((new_w, new_h))
        
        st.info(msg)
        
        canvas = st_canvas(
            fill_color="rgba(255, 255, 0, 0.3)" if "ריצוף" in report_type else "rgba(0,0,0,0)",
            stroke_width=stroke,
            stroke_color=draw_color,
            background_image=bg_pil,
            height=new_h, width=new_w,
            drawing_mode="freedraw",
            key=f"canv_{p_name}_{report_type}",
            update_streamlit=True
        )
        
        if canvas.json_data and canvas.json_data["objects"]:
            val = 0
            unit = ""
            
            # לוגיקת חישוב
            if report_type == "🧱 בניית קירות":
                # כאן נכנס החישוב המתוחכם (חיתוך)
                user_mask = np.zeros((new_h, new_w), dtype=np.uint8)
                if canvas.image_data is not None:
                    # הציור של המשתמש (שכבה 3 = אלפא או צבע)
                    user_draw = canvas.image_data[:, :, 3] > 0
                    
                    # הקירות המקוריים (מוקטנים לגודל הקנבס)
                    walls_small = cv2.resize(proj["thick_walls"], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    # ניפוח קל כדי שהחיתוך יעבוד גם אם המשתמש פספס קצת
                    walls_small = cv2.dilate(walls_small, np.ones((5,5), np.uint8))
                    
                    # חיתוך (Intersection)
                    intersect = np.logical_and(user_draw, walls_small > 0)
                    
                    # המרה למטרים
                    pixels = np.count_nonzero(intersect)
                    # זה חישוב גס, עדיף להשתמש ב-skeletonize על החיתוך לדיוק
                    val = (pixels / factor) / (proj["scale"] * 10) # פקטור אמפירי לעובי הקו
                    unit = "מטר"
            else:
                # ריצוף - פשוט שטח
                if canvas.image_data is not None:
                    px = np.count_nonzero(canvas.image_data[:, :, 3] > 0)
                    val = px / ((proj["scale"] * factor) ** 2)
                    unit = "מ\"ר"
            
            if val > 0:
                st.success(f"כמות מחושבת: {val:.2f} {unit}")
                if st.button("🚀 שלח דיווח"):
                    pid = save_plan(p_name, p_name, "1:50", proj["scale"], proj["raw_pixels"], "{}")
                    save_progress_report(pid, val, f"{report_type}")
                    st.balloons()