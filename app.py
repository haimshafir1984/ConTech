import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import json
import os
from streamlit_drawable_canvas import st_canvas

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

# --- אתחול ---
setup_page()
apply_css()
Image.MAX_IMAGE_PIXELS = None
init_database()

if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro")
    mode = st.radio("ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות"):
        st.session_state.wall_height = st.number_input("גובה קיר (מ')", value=st.session_state.wall_height)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter)
    
    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.rerun()

# ==========================================
# 🏢 מצב מנהל
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2 = st.tabs(["📂 סדנת עבודה", "📊 דשבורד"])
    
    # --- טאב 1: עבודה ---
    with tab1:
        with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
            files = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
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
                                # כאן התיקון: קבלת 9 משתנים במקום 8
                                pix, skel, thick, orig, meta, conc, blok, floor, debug_img = analyzer.process_file(path, save_debug=show_debug)
                                
                                if not meta.get("plan_name"): meta["plan_name"] = f.name.replace(".pdf", "")
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(meta["raw_text"])
                                    meta.update({k: v for k, v in llm_data.items() if v})

                                st.session_state.projects[f.name] = {
                                    "skeleton": skel, "thick_walls": thick, "original": orig,
                                    "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                    "concrete_mask": conc, "blocks_mask": blok, "flooring_mask": floor,
                                    "total_length": pix/200.0
                                }
                                
                                if show_debug and debug_img is not None:
                                    st.image(debug_img, caption="🔴 אדום=טקסט שסונן | 🔵 כחול=קירות שזוהו", use_column_width=True)
                                
                                os.unlink(path)
                                st.success(f"✅ {f.name} נטען")
                            except Exception as e: st.error(f"שגיאה: {str(e)}")

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            c1, c2 = st.columns([1, 1.5], gap="large")
            with c1:
                p_name = st.text_input("שם", value=proj["metadata"].get("plan_name", ""))
                scale_val = st.slider("סקייל (px/m)", 10.0, 1000.0, float(proj["scale"]))
                proj["scale"] = scale_val
                
                total_len = proj["raw_pixels"] / scale_val
                st.info(f"📏 קירות: {total_len:.1f} מ'")
                
                if st.button("💾 שמור"):
                    proj["metadata"]["plan_name"] = p_name
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    save_plan(selected, p_name, "1:50", scale_val, proj["raw_pixels"], meta_json, None, 0, 0, "{}")
                    st.toast("נשמר!")

                with st.expander("📖 מקרא (AI)"):
                    img_leg = Image.fromarray(cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB))
                    scale_leg = 800 / img_leg.width
                    img_leg_small = img_leg.resize((800, int(img_leg.height * scale_leg)))
                    
                    canvas = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)", stroke_color="#FFA500",
                        background_image=img_leg_small, height=img_leg_small.height, width=800,
                        drawing_mode="rect", key=f"leg_{selected}"
                    )
                    if st.button("פענח"):
                        if canvas.json_data["objects"]:
                            obj = canvas.json_data["objects"][-1]
                            l, t, w, h = [int(x/scale_leg) for x in [obj["left"], obj["top"], obj["width"], obj["height"]]]
                            crop = np.array(img_leg)[t:t+h, l:l+w]
                            if crop.size > 0:
                                buf = io.BytesIO()
                                Image.fromarray(crop).save(buf, format="PNG")
                                res = safe_analyze_legend(buf.getvalue())
                                st.info(res)

            with c2:
                show_floor = st.checkbox("הצג ריצוף", value=True)
                f_mask = proj["flooring_mask"] if show_floor else None
                overlay = create_colored_overlay(proj["original"], proj["concrete_mask"], proj["blocks_mask"], f_mask)
                st.image(overlay, use_column_width=True)

    # --- טאב 2: דשבורד ---
    with tab2:
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין נתונים.")
        else:
            sel_disp = st.selectbox("פרויקט:", [f"{p['plan_name']} ({p['id']})" for p in all_plans])
            pid = int(sel_disp.split("(")[1].strip(")"))
            fc = get_project_forecast(pid)
            fin = get_project_financial_status(pid)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("ביצוע", f"{fc['cumulative_progress']:.1f} מ'")
            k2.metric("תחזית", f"{fc['days_to_finish']} ימים")
            k3.metric("תקציב", f"{fin['current_cost']:,.0f} ₪")
            
            st.bar_chart(load_stats_df(), x="תאריך", y="כמות שבוצעה")
            
            if st.button("📄 PDF"):
                found = None
                for p in st.session_state.projects.values():
                    if p["metadata"].get("plan_name") == sel_disp.split(" (")[0]: found = p
                if found:
                    pdf = generate_status_pdf(sel_disp, found["original"], {"built": fc['cumulative_progress'], "total": fc['total_planned'], "percent": 0})
                    st.download_button("הורד", pdf, "report.pdf", "application/pdf")

# ==========================================
# 👷 מצב דיווח
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח שטח")
    if st.session_state.projects:
        sel = st.selectbox("תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[sel]
        rep_type = st.radio("סוג:", ["🧱 קירות", "🔲 ריצוף"], horizontal=True)
        
        rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        f = 800/w if w>800 else 1.0
        img_s = Image.fromarray(rgb).resize((int(w*f), int(h*f)))
        
        canv = st_canvas(
            fill_color="rgba(255,255,0,0.3)" if "ריצוף" in rep_type else "rgba(0,0,0,0)",
            stroke_width=8 if "קירות" in rep_type else 20,
            stroke_color="#00FF00" if "קירות" in rep_type else "#FFFF00",
            background_image=img_s,
            height=int(h*f), width=int(w*f),
            drawing_mode="freedraw", key=f"wk_{sel}_{rep_type}"
        )
        
        if st.button("🚀 שלח"):
             if canv.json_data["objects"]:
                 val = 0
                 # חישוב כמויות (חיתוך או שטח)
                 if "ריצוף" in rep_type and canv.image_data is not None:
                     px = np.count_nonzero(canv.image_data[:, :, 3] > 0)
                     val = px / ((proj["scale"]*f)**2)
                 elif "קירות" in rep_type and canv.image_data is not None:
                     user_draw = canv.image_data[:, :, 3] > 0
                     walls_small = cv2.resize(proj["thick_walls"], (int(w*f), int(h*f)), interpolation=cv2.INTER_NEAREST)
                     intersect = np.logical_and(user_draw, walls_small > 0)
                     val = (np.count_nonzero(intersect) / f) / (proj["scale"] * 10) # פקטור עובי

                 pid = save_plan(sel, sel, "1:50", proj["scale"], proj["raw_pixels"], "{}")
                 save_progress_report(pid, val, f"{rep_type}")
                 st.success(f"נשלח! ({val:.1f})")
    else: 
        st.error("אין תוכניות זמינות")