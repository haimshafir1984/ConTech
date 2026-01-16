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
    get_progress_reports, get_plan_by_filename, get_all_plans, get_plan_by_id,
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

# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro")
    mode = st.radio("ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות"):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
    
    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.success("המערכת אופסה")
            st.rerun()

# ==========================================
# 🏢 מצב מנהל
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2 = st.tabs(["📂 סדנת עבודה", "📊 דשבורד"])
    
    # --- טאב 1: העלאה ועריכה ---
    with tab1:
        with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
            files = st.file_uploader("גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
            show_debug = st.checkbox("🔍 הצג שכבות זיהוי (Debug)", value=False)

            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name}..."):
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
                                    "total_length": pix/200.0, "llm_suggestions": llm_data if meta.get("raw_text") else {}
                                }
                                
                                if show_debug and debug_img is not None:
                                    col_d1, col_d2 = st.columns([2, 1])
                                    with col_d1:
                                        st.image(debug_img, caption="🔴 אדום=טקסט מסונן | 🔵 כחול=קירות", use_column_width=True)
                                    with col_d2:
                                        st.info("""
                                        **מקרא Debug:**
                                        - 🔴 **אדום** = טקסט שזוהה והוסר
                                        - 🔵 **כחול** = קירות שזוהו
                                        
                                        אם עדיין יש טקסטים, ניתן לכוונן ב-analyzer.py
                                        """)
                                
                                os.unlink(path)
                                st.success(f"✅ {f.name} נטען בהצלחה")
                            except Exception as e: 
                                st.error(f"שגיאה: {str(e)}")
                                import traceback
                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            # יצירת מפתחות ייחודיים
            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state: st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state: st.session_state[scale_key] = proj["metadata"].get("scale", "")
            
            col_edit, col_preview = st.columns([1, 1.5], gap="large")
            
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                
                # סיווג תוכנית
                current_meta = proj.get("metadata", {})
                detected_type = current_meta.get("plan_type", "construction")
                type_map = {
                    "construction": "🧱 בנייה", "demolition": "🔨 הריסה",
                    "ceiling": "💡 תקרה", "electricity": "⚡ חשמל",
                    "plumbing": "💧 אינסטלציה", "other": "📋 אחר"
                }
                index_val = list(type_map.keys()).index(detected_type) if detected_type in type_map else 0
                selected_type = st.selectbox("סוג תוכנית", options=list(type_map.keys()), 
                                            format_func=lambda x: type_map[x], index=index_val)
                
                if selected_type == "ceiling": 
                    st.warning("⚠️ זו תוכנית תקרה - לא מתאימה למדידת קירות")
                elif selected_type == "demolition": 
                    st.error("🛑 זו תוכנית הריסה - שים לב לסימון")
                
                proj["metadata"]["plan_type"] = selected_type
                
                # שדות עריכה
                p_name = st.text_input("שם התוכנית", key=name_key)
                p_scale_text = st.text_input("קנה מידה (לתיעוד)", key=scale_key, placeholder="1:50")
                
                # לימוד מקרא AI
                with st.expander("📖 לימוד מקרא (AI Vision)"):
                    st.info("סמן את אזור המקרא בשרטוט")
                    zoom = st.slider("🔍 זום", 600, 1500, 800, step=50)
                    img_legend = Image.fromarray(cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB))
                    scale_factor = zoom / img_legend.width
                    img_resized = img_legend.resize((zoom, int(img_legend.height * scale_factor)))
                    
                    canvas_legend = st_canvas(
                        fill_color="rgba(255,165,0,0.3)", stroke_color="#FFA500", stroke_width=2,
                        background_image=img_resized, height=img_resized.height, width=zoom,
                        drawing_mode="rect", key=f"legend_{selected}_{zoom}"
                    )
                    
                    if canvas_legend.json_data and canvas_legend.json_data["objects"]:
                        if st.button("👁️ פענח"):
                            obj = canvas_legend.json_data["objects"][-1]
                            l, t, w, h = [int(x/scale_factor) for x in [obj["left"], obj["top"], obj["width"], obj["height"]]]
                            crop = np.array(img_legend)[t:t+h, l:l+w]
                            if crop.size > 0:
                                buf = io.BytesIO()
                                Image.fromarray(crop).save(buf, format="PNG")
                                with st.spinner("מנתח..."):
                                    result = safe_analyze_legend(buf.getvalue())
                                    st.text_area("תוצאה:", result, height=100)
                                    proj["metadata"]["legend_analysis"] = result
                
                # הגדרות תקציב
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    target_date = st.date_input("תאריך יעד")
                    target_date_str = target_date.strftime("%Y-%m-%d") if target_date else None
                with col_d2:
                    budget = st.number_input("תקציב (₪)", step=1000.0, min_value=0.0)
                
                cost_per_m = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
                
                # כיול סקייל
                st.markdown("#### כיול")
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]))
                proj["scale"] = scale_val
                
                # חישוב כמויות
                total_len = proj["raw_pixels"] / scale_val
                conc_len = proj["metadata"].get("pixels_concrete", 0) / scale_val
                block_len = proj["metadata"].get("pixels_blocks", 0) / scale_val
                floor_area = proj["metadata"].get("pixels_flooring_area", 0) / (scale_val ** 2)
                proj["total_length"] = total_len
                
                st.info(f"📏 קירות: {total_len:.1f}מ' | בטון: {conc_len:.1f}מ' | בלוקים: {block_len:.1f}מ' | ריצוף: {floor_area:.1f}מ\"ר")
                
                # מחשבון הצעת מחיר
                with st.expander("💰 מחשבון הצעת מחיר", expanded=True):
                    st.markdown("""<div style="background:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:10px;">
                    <strong>מחירון בסיס:</strong> בטון 1200₪/מ' | בלוקים 600₪/מ' | ריצוף 250₪/מ\"ר
                    </div>""", unsafe_allow_html=True)
                    
                    c_price = st.number_input("מחיר בטון (₪/מ')", value=1200.0, step=50.0)
                    b_price = st.number_input("מחיר בלוקים (₪/מ')", value=600.0, step=50.0)
                    f_price = st.number_input("מחיר ריצוף (₪/מ\"ר)", value=250.0, step=50.0)
                    
                    total_quote = (conc_len * c_price) + (block_len * b_price) + (floor_area * f_price)
                    st.markdown(f"#### 💵 סה\"כ הצעת מחיר: {total_quote:,.0f} ₪")
                    
                    # טבלת פירוט
                    quote_df = pd.DataFrame({
                        "פריט": ["קירות בטון", "קירות בלוקים", "ריצוף/חיפוי", "סה\"כ"],
                        "יחידה": ["מ'", "מ'", "מ\"ר", "-"],
                        "כמות": [f"{conc_len:.2f}", f"{block_len:.2f}", f"{floor_area:.2f}", "-"],
                        "מחיר יחידה": [f"{c_price:.0f}₪", f"{b_price:.0f}₪", f"{f_price:.0f}₪", "-"],
                        "סה\"כ": [f"{conc_len*c_price:,.0f}₪", f"{block_len*b_price:,.0f}₪", f"{floor_area*f_price:,.0f}₪", f"{total_quote:,.0f}₪"]
                    })
                    st.dataframe(quote_df, hide_index=True, use_container_width=True)
                
                # כפתור שמירה
                st.markdown("---")
                if st.button("💾 שמור תוכנית למערכת", type="primary"):
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale_text
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = json.dumps({
                        "concrete_length": conc_len,
                        "blocks_length": block_len,
                        "flooring_area": floor_area
                    }, ensure_ascii=False)
                    
                    plan_id = save_plan(selected, p_name, p_scale_text, scale_val, proj["raw_pixels"], 
                                       meta_json, target_date_str, budget, cost_per_m, materials)
                    st.toast("✅ נשמר למערכת!")
                    st.success(f"התוכנית נשמרה בהצלחה (ID: {plan_id})")
            
            with col_preview:
                st.markdown("### תצוגה מקדימה")
                show_flooring = st.checkbox("הצג ריצוף", value=True)
                floor_mask = proj["flooring_mask"] if show_flooring else None
                overlay = create_colored_overlay(proj["original"], proj["concrete_mask"], 
                                                proj["blocks_mask"], floor_mask)
                st.image(overlay, use_column_width=True)
                st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")
    
    # --- טאב 2: דשבורד ---
    with tab2:
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים במערכת. העלה תוכניות בטאב 'סדנת עבודה'")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan = st.selectbox("בחר פרויקט:", plan_options)
            plan_id = int(selected_plan.split("ID: ")[1].strip(")"))
            
            # חישוב נתונים
            forecast = get_project_forecast(plan_id)
            financial = get_project_financial_status(plan_id)
            
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🏗️ ביצוע", f"{forecast['cumulative_progress']:.1f} מ'")
            k2.metric("📊 התקדמות", f"{(forecast['cumulative_progress']/forecast['total_planned']*100):.1f}%" if forecast['total_planned'] > 0 else "0%")
            k3.metric("⏱️ תחזית", f"{forecast['days_to_finish']} ימים")
            k4.metric("💰 תקציב", f"{financial['current_cost']:,.0f} ₪")
            
            # גרף התקדמות
            st.markdown("---")
            st.markdown("### 📈 קצב ביצוע")
            df_stats = load_stats_df()
            if not df_stats.empty:
                st.bar_chart(df_stats, x="תאריך", y="כמות שבוצעה", use_container_width=True)
            else:
                st.info("אין דיווחי ביצוע עדיין")
            
            # טבלת דיווחים
            st.markdown("### 📋 דיווחים אחרונים")
            reports = get_progress_reports(plan_id)
            if reports:
                df_reports = pd.DataFrame(reports)
                st.dataframe(df_reports[['date', 'meters_built', 'note']], hide_index=True, use_container_width=True)
            else:
                st.info("אין דיווחים")

# ==========================================
# 👷 מצב דיווח
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.warning("אין תוכניות זמינות. עבור למנהל פרויקט להעלאת תוכניות.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True)
        
        # הכנת תמונה
        rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale_factor = 800 / w if w > 800 else 1.0
        img_resized = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
        
        # הגדרות canvas
        if "קירות" in report_type:
            fill = "rgba(0,0,0,0)"
            stroke = "#00FF00"
            stroke_width = 8
            msg = "סמן בירוק את הקירות שבוצעו"
        else:
            fill = "rgba(255,255,0,0.3)"
            stroke = "#FFFF00"
            stroke_width = 20
            msg = "צבע בצהוב את השטח שרוצף"
        
        st.caption(msg)
        canvas = st_canvas(
            fill_color=fill, stroke_color=stroke, stroke_width=stroke_width,
            background_image=img_resized,
            height=int(h*scale_factor), width=int(w*scale_factor),
            drawing_mode="freedraw",
            key=f"canvas_{plan_name}_{report_type}"
        )
        
        if canvas.json_data and canvas.json_data["objects"]:
            measured = 0
            unit = ""
            
            if "קירות" in report_type and canvas.image_data is not None:
                # חיתוך עם קירות
                user_draw = canvas.image_data[:, :, 3] > 0
                walls_resized = cv2.resize(proj["thick_walls"], (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_NEAREST)
                walls_dilated = cv2.dilate(walls_resized, np.ones((5,5), np.uint8))
                intersection = np.logical_and(user_draw, walls_dilated > 0)
                pixels = np.count_nonzero(intersection)
                measured = (pixels / scale_factor) / proj["scale"]
                unit = "מטר"
                
            elif "ריצוף" in report_type and canvas.image_data is not None:
                # חישוב שטח
                pixels = np.count_nonzero(canvas.image_data[:, :, 3] > 0)
                measured = pixels / ((proj["scale"] * scale_factor) ** 2)
                unit = "מ\"ר"
            
            if measured > 0:
                st.success(f"✅ כמות מחושבת: **{measured:.2f} {unit}**")
                note = st.text_input("הערה (אופציונלי)", placeholder=f"דיווח {report_type}")
                
                if st.button("🚀 שלח דיווח", type="primary"):
                    # שמירה
                    rec = get_plan_by_filename(plan_name)
                    if rec:
                        pid = rec['id']
                    else:
                        meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                        pid = save_plan(plan_name, proj["metadata"].get("plan_name", plan_name), 
                                      "1:50", proj["scale"], proj["raw_pixels"], meta_json)
                    
                    full_note = f"{note} ({measured:.2f} {unit})" if note else f"{report_type}: {measured:.2f} {unit}"
                    save_progress_report(pid, measured, full_note)
                    st.balloons()
                    st.success("הדיווח נשמר בהצלחה!")
            else:
                st.info("סמן על התמונה כדי לחשב כמות")
        else:
            st.info("התחל לצייר על התמונה")
