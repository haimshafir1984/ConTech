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

# ייבוא מהקבצים המסודרים שלך
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
if 'manual_corrections' not in st.session_state: st.session_state.manual_corrections = {}  # NEW!

# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro v2.0")
    st.caption("✨ Multi-pass Detection + Manual Corrections")
    mode = st.radio("ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות"):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1, key="global_wall_height")
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0, key="global_cost_per_meter")
    
    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.session_state.manual_corrections = {}
            st.success("המערכת אופסה")
            st.rerun()

# ==========================================
# 🏢 מצב מנהל
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2, tab3 = st.tabs(["📂 סדנת עבודה", "🎨 תיקונים ידניים", "📊 דשבורד"])  # ← טאב חדש!
    
    # --- טאב 1: העלאה ועריכה ---
    with tab1:
        with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
            files = st.file_uploader("גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
            debug_mode = st.selectbox("מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0)
            show_debug = debug_mode != "בסיסי"

            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name} עם Multi-Pass Detection..."):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(f.getvalue())
                                    path = tmp.name
                                
                                analyzer = FloorPlanAnalyzer()
                                pix, skel, thick, orig, meta, conc, blok, floor, debug_img = analyzer.process_file(path, save_debug=show_debug)
                                
                                if not meta.get("plan_name"): 
                                    meta["plan_name"] = f.name.replace(".pdf", "").replace("-", " ").strip()
                                
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(meta["raw_text"])
                                    meta.update({k: v for k, v in llm_data.items() if v})

                                st.session_state.projects[f.name] = {
                                    "skeleton": skel, "thick_walls": thick, "original": orig,
                                    "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                    "concrete_mask": conc, "blocks_mask": blok, "flooring_mask": floor,
                                    "total_length": pix/200.0, "llm_suggestions": llm_data if meta.get("raw_text") else {},
                                    "debug_layers": analyzer.debug_layers  # שמירת שכבות debug
                                }
                                
                                # תצוגת Debug משופרת
                                if show_debug and debug_img is not None:
                                    st.markdown("### 🔍 ניתוח Multi-Pass")
                                    
                                    if debug_mode == "מפורט - שכבות":
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.image(debug_img, caption="תוצאה משולבת", use_column_width=True)
                                        with col2:
                                            if 'text_combined' in analyzer.debug_layers:
                                                st.image(analyzer.debug_layers['text_combined'], caption="🔴 טקסט שהוסר", use_column_width=True)
                                        with col3:
                                            if 'walls' in analyzer.debug_layers:
                                                st.image(analyzer.debug_layers['walls'], caption="🟢 קירות שזוהו", use_column_width=True)
                                    
                                    elif debug_mode == "מלא - עם confidence":
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.image(debug_img, caption="תוצאה משולבת", use_column_width=True)
                                        with col2:
                                            # מקרא
                                            st.markdown("""
                                            **מקרא צבעים:**
                                            - 🟠 כתום = טקסט ברור
                                            - 🟡 צהוב = סמלים וכותרות
                                            - 🟣 סגול = מספרי חדרים
                                            - 🟢 ירוק = קירות
                                            - 🔥 אדום-צהוב = confidence גבוה
                                            - 🔵 כחול-שחור = confidence נמוך
                                            """)
                                            
                                            # סטטיסטיקות
                                            st.metric("Confidence ממוצע", f"{meta.get('confidence_avg', 0):.2f}")
                                            st.metric("פיקסלי טקסט שהוסרו", f"{meta.get('text_removed_pixels', 0):,}")
                                
                                os.unlink(path)
                                st.success(f"✅ {f.name} נותח בהצלחה!")
                            except Exception as e: 
                                st.error(f"שגיאה: {str(e)}")
                                import traceback
                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())

        # הקוד הרגיל של עריכת תוכניות ממשיך כרגיל...
        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            # (שאר הקוד זהה לגרסה הקודמת...)
            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state: st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state: st.session_state[scale_key] = proj["metadata"].get("scale", "")
            
            # תצוגה ועריכה פשוטה
            col_edit, col_preview = st.columns([1, 1.5], gap="large")
            
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                p_name = st.text_input("שם התוכנית", key=name_key)
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]), key=f"scale_slider_{selected}")
                proj["scale"] = scale_val
                
                total_len = proj["raw_pixels"] / scale_val
                st.info(f"📏 סה\"כ קירות: {total_len:.1f} מ'")
                
                if st.button("💾 שמור", key=f"save_{selected}"):
                    proj["metadata"]["plan_name"] = p_name
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    save_plan(selected, p_name, "1:50", scale_val, proj["raw_pixels"], meta_json)
                    st.toast("✅ נשמר!")
            
            with col_preview:
                st.markdown("### תצוגה מקדימה")
                show_flooring = st.checkbox("הצג ריצוף", value=True, key=f"show_flooring_{selected}")
                floor_mask = proj["flooring_mask"] if show_flooring else None
                overlay = create_colored_overlay(proj["original"], proj["concrete_mask"], 
                                                proj["blocks_mask"], floor_mask)
                st.image(overlay, use_column_width=True)
    
    # ==========================================
    # 🎨 טאב חדש: תיקונים ידניים (Approach C)
    # ==========================================
    with tab2:
        st.markdown("## 🎨 תיקונים ידניים")
        st.caption("הוסף או הסר קירות באופן ידני למדויקות מקסימלית")
        
        if not st.session_state.projects:
            st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        else:
            selected_plan = st.selectbox("בחר תוכנית לתיקון:", list(st.session_state.projects.keys()), key="correction_plan_select")
            proj = st.session_state.projects[selected_plan]
            
            correction_mode = st.radio("מצב תיקון:", 
                                      ["➕ הוסף קירות חסרים", "➖ הסר קירות מזויפים", "👁️ השוואה"], 
                                      horizontal=True)
            
            # הכנת תמונת רקע
            rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale_factor = 1000 / w if w > 1000 else 1.0
            img_display = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
            
            if correction_mode == "➕ הוסף קירות חסרים":
                st.info("🖌️ צייר בירוק על הקירות שהמערכת החמיצה")
                
                canvas_add = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=6,
                    stroke_color="#00FF00",
                    background_image=img_display,
                    height=int(h*scale_factor),
                    width=int(w*scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_add_{selected_plan}"
                )
                
                if canvas_add.image_data is not None and np.any(canvas_add.image_data[:, :, 3] > 0):
                    if st.button("✅ אשר הוספה", key="confirm_add"):
                        # שמירת התיקונים
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}
                        
                        # המרה לגודל מקורי
                        added_mask = cv2.resize(canvas_add.image_data[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
                        added_mask = (added_mask > 0).astype(np.uint8) * 255
                        
                        st.session_state.manual_corrections[selected_plan]['added_walls'] = added_mask
                        st.success("✅ קירות נוספו! עבור לטאב 'השוואה' לראות את התוצאה")
            
            elif correction_mode == "➖ הסר קירות מזויפים":
                st.info("🖌️ צייר באדום על קירות שהמערכת זיהתה בטעות")
                
                # תצוגה עם הקירות הקיימים
                walls_overlay = proj["thick_walls"].copy()
                walls_colored = cv2.cvtColor(walls_overlay, cv2.COLOR_GRAY2RGB)
                walls_colored[walls_overlay > 0] = [0, 255, 255]  # ציאן
                
                combined = cv2.addWeighted(rgb, 0.6, walls_colored, 0.4, 0)
                combined_resized = cv2.resize(combined, (int(w*scale_factor), int(h*scale_factor)))
                img_with_walls = Image.fromarray(combined_resized)
                
                canvas_remove = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=8,
                    stroke_color="#FF0000",
                    background_image=img_with_walls,
                    height=int(h*scale_factor),
                    width=int(w*scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_remove_{selected_plan}"
                )
                
                if canvas_remove.image_data is not None and np.any(canvas_remove.image_data[:, :, 3] > 0):
                    if st.button("✅ אשר הסרה", key="confirm_remove"):
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}
                        
                        removed_mask = cv2.resize(canvas_remove.image_data[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
                        removed_mask = (removed_mask > 0).astype(np.uint8) * 255
                        
                        st.session_state.manual_corrections[selected_plan]['removed_walls'] = removed_mask
                        st.success("✅ קירות הוסרו! עבור לטאב 'השוואה' לראות את התוצאה")
            
            elif correction_mode == "👁️ השוואה":
                st.markdown("### לפני ואחרי")
                
                if selected_plan in st.session_state.manual_corrections:
                    corrections = st.session_state.manual_corrections[selected_plan]
                    
                    # חישוב מסכת קירות מתוקנת
                    corrected_walls = proj["thick_walls"].copy()
                    
                    if 'added_walls' in corrections:
                        corrected_walls = cv2.bitwise_or(corrected_walls, corrections['added_walls'])
                    
                    if 'removed_walls' in corrections:
                        corrected_walls = cv2.subtract(corrected_walls, corrections['removed_walls'])
                    
                    # תצוגה לפני/אחרי
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🤖 זיהוי אוטומטי")
                        auto_overlay = rgb.copy()
                        auto_overlay[proj["thick_walls"] > 0] = [0, 255, 0]
                        st.image(auto_overlay, use_column_width=True)
                        
                        # סטטיסטיקות
                        auto_pixels = np.count_nonzero(proj["thick_walls"])
                        auto_length = auto_pixels / proj["scale"]
                        st.metric("אורך", f"{auto_length:.1f} מ'")
                    
                    with col2:
                        st.markdown("#### ✅ אחרי תיקון")
                        corrected_overlay = rgb.copy()
                        corrected_overlay[corrected_walls > 0] = [255, 165, 0]  # כתום
                        st.image(corrected_overlay, use_column_width=True)
                        
                        # סטטיסטיקות
                        corrected_pixels = np.count_nonzero(corrected_walls)
                        corrected_length = corrected_pixels / proj["scale"]
                        st.metric("אורך", f"{corrected_length:.1f} מ'", 
                                 delta=f"{corrected_length - auto_length:+.1f} מ'")
                    
                    # כפתור שמירה
                    st.markdown("---")
                    if st.button("💾 שמור גרסה מתוקנת", type="primary"):
                        # עדכון הפרויקט
                        proj["thick_walls"] = corrected_walls
                        proj["raw_pixels"] = corrected_pixels
                        proj["total_length"] = corrected_length
                        
                        # שמירה למסד נתונים
                        meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                        save_plan(selected_plan, proj["metadata"].get("plan_name"), "1:50", 
                                 proj["scale"], corrected_pixels, meta_json)
                        
                        st.success("✅ הגרסה המתוקנת נשמרה!")
                        st.balloons()
                else:
                    st.info("אין תיקונים ידניים עדיין. עבור לטאב 'הוסף קירות' או 'הסר קירות'")
    
    # --- טאב 3: דשבורד (ללא שינוי) ---
    with tab3:
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים במערכת")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan_dash = st.selectbox("בחר פרויקט:", plan_options)
            plan_id = int(selected_plan_dash.split("ID: ")[1].strip(")"))
            
            forecast = get_project_forecast(plan_id)
            financial = get_project_financial_status(plan_id)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🏗️ ביצוע", f"{forecast['cumulative_progress']:.1f} מ'")
            k2.metric("📊 התקדמות", f"{(forecast['cumulative_progress']/forecast['total_planned']*100):.1f}%" if forecast['total_planned'] > 0 else "0%")
            k3.metric("⏱️ תחזית", f"{forecast['days_to_finish']} ימים")
            k4.metric("💰 תקציב", f"{financial['current_cost']:,.0f} ₪")
            
            st.markdown("---")
            df_stats = load_stats_df()
            if not df_stats.empty:
                st.bar_chart(df_stats, x="תאריך", y="כמות שבוצעה", use_container_width=True)

# ==========================================
# 👷 מצב דיווח (ללא שינוי)
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.warning("אין תוכניות זמינות")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True)
        
        rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale_factor = 800 / w if w > 800 else 1.0
        img_resized = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
        
        if "קירות" in report_type:
            fill = "rgba(0,0,0,0)"
            stroke = "#00FF00"
            stroke_width = 8
        else:
            fill = "rgba(255,255,0,0.3)"
            stroke = "#FFFF00"
            stroke_width = 20
        
        canvas = st_canvas(
            fill_color=fill, stroke_color=stroke, stroke_width=stroke_width,
            background_image=img_resized,
            height=int(h*scale_factor), width=int(w*scale_factor),
            drawing_mode="freedraw",
            key=f"canvas_{plan_name}_{report_type}"
        )
        
        if canvas.json_data and canvas.json_data["objects"] and canvas.image_data is not None:
            measured = 0
            if "קירות" in report_type:
                user_draw = canvas.image_data[:, :, 3] > 0
                walls_resized = cv2.resize(proj["thick_walls"], (int(w*scale_factor), int(h*scale_factor)))
                intersection = np.logical_and(user_draw, walls_resized > 0)
                measured = np.count_nonzero(intersection) / scale_factor / proj["scale"]
            else:
                pixels = np.count_nonzero(canvas.image_data[:, :, 3] > 0)
                measured = pixels / ((proj["scale"] * scale_factor) ** 2)
            
            if measured > 0:
                st.success(f"✅ {measured:.2f} " + ('מ"ר' if 'ריצוף' in report_type else 'מטר'))
                if st.button("🚀 שלח דיווח", type="primary"):
                    rec = get_plan_by_filename(plan_name)
                    pid = rec['id'] if rec else save_plan(plan_name, plan_name, "1:50", proj["scale"], proj["raw_pixels"], "{}")
                    save_progress_report(pid, measured, report_type)
                    st.balloons()
