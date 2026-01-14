import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from analyzer import FloorPlanAnalyzer
import tempfile
import os
import json
import io
from streamlit_drawable_canvas import st_canvas
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_plan_by_id, get_all_plans,
    get_project_forecast, 
    calculate_material_estimates, get_project_financial_status, reset_all_data
)
from datetime import datetime
from reporter import generate_status_pdf

# --- אתחול ---
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- הגדרת עמוד ---
st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")

# --- פונקציות עזר ---
def safe_process_metadata(raw_text):
    try:
        from brain import process_plan_metadata
        return process_plan_metadata(raw_text)
    except: return {}

def safe_analyze_legend(image_bytes):
    try:
        from brain import analyze_legend_image
        return analyze_legend_image(image_bytes)
    except Exception as e: return f"Error: {str(e)}"

def load_stats_df():
    reports = get_progress_reports()
    if reports:
        df = pd.DataFrame(reports)
        return df.rename(columns={'date': 'תאריך', 'plan_name': 'שם תוכנית', 'meters_built': 'כמות שבוצעה', 'note': 'הערה'})
    return pd.DataFrame()

def create_colored_overlay(original, concrete_mask, blocks_mask, flooring_mask=None, noise_level=0):
    img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
    
    # יצירת עותקים כדי לא לדרוס את המקור
    c_clean = concrete_mask.copy() if concrete_mask is not None else np.zeros(original.shape[:2], dtype=np.uint8)
    b_clean = blocks_mask.copy() if blocks_mask is not None else np.zeros(original.shape[:2], dtype=np.uint8)
    
    # ניקוי רעשים חי (לפתרון בעיית הטקסט הכחול)
    if noise_level > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noise_level, noise_level))
        if c_clean.any(): c_clean = cv2.morphologyEx(c_clean, cv2.MORPH_OPEN, kernel)
        if b_clean.any(): b_clean = cv2.morphologyEx(b_clean, cv2.MORPH_OPEN, kernel)
    
    overlay = img_vis.copy()
    overlay[c_clean > 0] = [30, 144, 255] # כחול
    overlay[b_clean > 0] = [255, 165, 0]  # כתום
    if flooring_mask is not None:
         overlay[flooring_mask > 0] = [200, 100, 255] # סגול
    
    cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, img_vis)
    return img_vis.astype(np.uint8), c_clean, b_clean

# --- CSS עיצוב נקי ומופרד ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Heebo', sans-serif !important; direction: rtl; }
    .stApp { background-color: #F8F9FA; }
    
    /* הפרדה ויזואלית ברורה: צד שמאל אפור, צד ימין לבן */
    .visual-container {
        background-color: #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
        min-height: 85vh;
    }
    
    /* פאנל הגדרות צף ונקי - עם הפרדה ברורה */
    .settings-panel {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: -4px 0 15px rgba(0,0,0,0.03);
        height: 100%;
        min-height: 85vh;
    }

    /* כרטיסיות בתוך הפאנל */
    .panel-card {
        background: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 { color: #0F62FE; font-weight: 900 !important; }
    .stButton button { border-radius: 8px; font-weight: 700; border:none; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stButton button[kind="primary"] { background: #0F62FE; color: white; }
    .stButton button[kind="primary"]:hover { box-shadow: 0 5px 15px rgba(15, 98, 254, 0.3); }

    /* הסתרת כותרות דפדפן */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

# --- תפריט צד ---
with st.sidebar:
    st.markdown("### 🏗️ ConTech Pro")
    selected_view = st.radio("ניווט", ["🏢 מנהל פרויקט (סדנה)", "📊 דשבורד ניהולי", "👷 דיווח שטח"])
    st.markdown("---")
    if "מנהל" in selected_view:
         st.session_state.wall_height = st.number_input("גובה קיר (מ')", value=st.session_state.wall_height)
         if st.button("🗑️ איפוס מערכת"):
            if reset_all_data():
                st.session_state.projects = {}
                st.rerun()

# ==========================================
# VIEW 1: סדנת עבודה (מתוקן ומלא)
# ==========================================
if "סדנה" in selected_view:
    
    # כותרת עליונה
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1: st.title("סדנת עבודה")
    with col_head2:
        current_projects = list(st.session_state.projects.keys())
        selected_proj_name = st.selectbox("בחר תוכנית:", current_projects) if current_projects else None

    # חלוקה ראשית: 70% שמאל (ויזואלי), 30% ימין (הגדרות)
    col_vis, col_set = st.columns([2.5, 1], gap="large")
    
    # === צד ימין: פאנל הגדרות ===
    with col_set:
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        
        # 1. העלאת קבצים
        with st.expander("📂 ניהול קבצים", expanded=not selected_proj_name):
            files = st.file_uploader("העלה PDF", type="pdf", accept_multiple_files=True)
            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner("מעבד שרטוט..."):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(f.getvalue())
                                    path = tmp.name
                                analyzer = FloorPlanAnalyzer()
                                # קריאה לאנלייזר (הנחנו שהוא מחזיר את כל הפרמטרים)
                                pix, skel, thick, orig, meta, conc, blok, floor = analyzer.process_file(path)
                                
                                if meta.get("raw_text"): meta.update(safe_process_metadata(meta["raw_text"]))
                                
                                st.session_state.projects[f.name] = {
                                    "skeleton": skel, "thick_walls": thick, "original": orig,
                                    "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                    "concrete_mask": conc, "blocks_mask": blok, "flooring_mask": floor,
                                    "total_length": pix/200.0
                                }
                                os.unlink(path)
                                st.rerun()
                            except Exception as e: st.error(str(e))

        if selected_proj_name:
            proj = st.session_state.projects[selected_proj_name]
            
            # 2. הגדרות תוכנית
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("**🛠️ הגדרות וכיול**")
            new_name = st.text_input("שם התוכנית", value=proj["metadata"].get("plan_name", ""))
            
            # סליידר לסקייל
            new_scale = st.slider("קנה מידה (פיקסלים למטר)", 10.0, 500.0, float(proj["scale"]), step=1.0)
            proj["scale"] = new_scale
            
            # סליידר לניקוי רעשים
            st.markdown("**🧹 סינון רעשים** (הסרת טקסט)")
            noise_level = st.slider("רמת סינון", 0, 15, 0, help="אם טקסט נצבע כבטון, העלה את הערך הזה")
            st.markdown('</div>', unsafe_allow_html=True)

            # 3. לימוד מקרא
            with st.expander("📖 לימוד מקרא (AI)"):
                st.info("סמן את המקרא בתמונה למטה:")
                img_leg = Image.fromarray(cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB))
                
                # הקטנה לתצוגה כדי שייכנס בפאנל
                orig_w, orig_h = img_leg.size
                disp_w = 300
                factor = disp_w / orig_w
                disp_h = int(orig_h * factor)
                img_leg_small = img_leg.resize((disp_w, disp_h))
                
                canvas_leg = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2, stroke_color="#FFA500",
                    background_image=img_leg_small,
                    height=disp_h, width=disp_w,
                    drawing_mode="rect",
                    key="leg_canvas"
                )
                
                if st.button("👁️ פענח סימון"):
                    if canvas_leg.json_data and canvas_leg.json_data["objects"]:
                        obj = canvas_leg.json_data["objects"][-1]
                        # המרה חזרה לגודל מקורי
                        left = int(obj["left"]/factor)
                        top = int(obj["top"]/factor)
                        width = int(obj["width"]/factor)
                        height = int(obj["height"]/factor)
                        
                        crop = np.array(img_leg)[top:top+height, left:left+width]
                        if crop.size > 0:
                            buf = io.BytesIO()
                            Image.fromarray(crop).save(buf, format="PNG")
                            res = safe_analyze_legend(buf.getvalue())
                            st.success("פוענח!")
                            st.caption(res)
                    else: st.warning("נא לסמן אזור")

            # 4. עלויות
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("**💰 מחירון**")
            c1, c2 = st.columns(2)
            with c1: pc = st.number_input("בטון", value=1200.0)
            with c2: pb = st.number_input("בלוקים", value=600.0)
            pf = st.number_input("ריצוף", value=250.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("💾 שמור שינויים", type="primary", use_container_width=True):
                proj["metadata"]["plan_name"] = new_name
                save_plan(selected_proj_name, new_name, "1:50", new_scale, proj["raw_pixels"], json.dumps(proj["metadata"]))
                st.toast("נשמר בהצלחה!", icon="✅")

        st.markdown('</div>', unsafe_allow_html=True)

    # === צד שמאל: האזור הויזואלי ===
    with col_vis:
        st.markdown('<div class="visual-container">', unsafe_allow_html=True)
        
        if selected_proj_name:
            t1, t2, t3, t4 = st.columns([1,1,1,2])
            with t1: s_c = st.checkbox("בטון", True)
            with t2: s_b = st.checkbox("בלוקים", False)
            with t3: s_f = st.checkbox("ריצוף", False)
            
            c_mask = proj["concrete_mask"] if s_c else np.zeros_like(proj["concrete_mask"])
            b_mask = proj["blocks_mask"] if s_b else np.zeros_like(proj["blocks_mask"])
            f_mask = proj["flooring_mask"] if s_f else None
            
            # יצירת תמונה עם ניקוי רעשים חי
            final_img, clean_c, clean_b = create_colored_overlay(
                proj["original"], c_mask, b_mask, f_mask, noise_level=noise_level
            )
            
            st.image(final_img, use_column_width=True)
            
            # חישוב נתונים (על בסיס המסכות הנקיות)
            # שימוש ב-Skeletonize לחישוב מדויק יותר של אורך
            # שים לב: זה חישוב כבד, אם התמונה איטית אפשר להשתמש ב-countNonZero ישירות כהערכה
            try:
                skel_c = cv2.ximgproc.thinning(clean_c) if clean_c.max() > 0 else np.zeros_like(clean_c)
                skel_b = cv2.ximgproc.thinning(clean_b) if clean_b.max() > 0 else np.zeros_like(clean_b)
                len_c = cv2.countNonZero(skel_c) / new_scale
                len_b = cv2.countNonZero(skel_b) / new_scale
            except:
                # Fallback אם ximgproc לא מותקן
                len_c = (cv2.countNonZero(clean_c) / 10) / new_scale 
                len_b = (cv2.countNonZero(clean_b) / 10) / new_scale

            area_f = (proj["metadata"].get("pixels_flooring_area", 0)) / (new_scale**2)
            tot = (len_c * pc) + (len_b * pb) + (area_f * pf)
            
            st.info(f"📏 הערכה: בטון: {len_c:.1f} מ' | בלוקים: {len_b:.1f} מ' | ריצוף: {area_f:.1f} מ\"ר | סה\"כ: ₪{tot:,.0f}")
            
        else:
            st.info("👈 בחר או העלה תוכנית מהתפריט הימני")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VIEW 2: דשבורד
# ==========================================
elif "דשבורד" in selected_view:
    st.title("דשבורד ניהולי")
    all_plans = get_all_plans()
    if all_plans:
        sel = st.selectbox("פרויקט", [p['plan_name'] for p in all_plans])
        pid = int(get_plan_by_filename(sel)['id']) if get_plan_by_filename(sel) else 1
        
        fc = get_project_forecast(pid)
        fin = get_project_financial_status(pid)
        pct = (fc['cumulative_progress']/fc['total_planned']*100) if fc['total_planned']>0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("התקדמות", f"{pct:.1f}%", f"{fc['cumulative_progress']:.1f} מ'")
        k2.metric("תחזית חודשית", f"{fc['average_velocity']*30:.1f} מ'")
        k3.metric("תקציב", f"₪{fin['budget_limit']:,.0f}", f"{fin['budget_variance']:,.0f}")
        
        st.line_chart(load_stats_df(), x="תאריך", y="כמות שבוצעה")
        if st.button("📄 הפק דוח PDF"):
            found = None
            for p in st.session_state.projects.values():
                if p["metadata"].get("plan_name") == sel: found = p
            
            if found:
                pdf = generate_status_pdf(sel, found["original"], {"built": fc['cumulative_progress'], "total": fc['total_planned'], "percent": pct})
                st.download_button("📥 הורד PDF", pdf, "report.pdf", "application/pdf")
            else: st.warning("נא לטעון את הקובץ לזיכרון")
    else: st.info("אין נתונים")

# ==========================================
# VIEW 3: דיווח שטח (משוחזר עם לוגיקה)
# ==========================================
elif "שטח" in selected_view:
    st.title("דיווח שטח")
    if st.session_state.projects:
        sel_p = st.selectbox("תוכנית", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[sel_p]
        rep_type = st.radio("סוג", ["🧱 קירות", "🔲 ריצוף"], horizontal=True)
        
        rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        f = 600/w if w>600 else 1.0
        img_s = Image.fromarray(rgb).resize((int(w*f), int(h*f)))
        
        canv = st_canvas(
            fill_color="rgba(255,255,0,0.3)" if "ריצוף" in rep_type else "rgba(0,0,0,0)",
            stroke_width=8 if "קירות" in rep_type else 20,
            stroke_color="#00FF00" if "קירות" in rep_type else "#FFFF00",
            background_image=img_s,
            height=int(h*f), width=int(w*f),
            drawing_mode="freedraw", key=f"wk_{sel_p}_{rep_type}"
        )
        
        if st.button("🚀 שלח דיווח", type="primary"):
             if canv.json_data and canv.json_data["objects"]:
                 val = 0
                 unit = ""
                 
                 # --- לוגיקה אמיתית לחישוב ---
                 if rep_type == "🧱 קירות":
                     # יצירת מסכה ממה שהמשתמש צייר
                     user_mask = np.zeros((int(h*f), int(w*f)), dtype=np.uint8)
                     # (כאן אנו מניחים שימוש בספריה שרטוט פוליגונים, או פשוט ספירה גסה אם אין פוליגון מוגדר)
                     # לטובת היציבות נשתמש בספירת פיקסלים של הציור שהוחזר
                     if canv.image_data is not None:
                         # התאמת גודל הקירות המקוריים לגודל הקנבס
                         walls_resized = cv2.resize(proj["thick_walls"], (int(w*f), int(h*f)), interpolation=cv2.INTER_NEAREST)
                         # ציור המשתמש
                         user_draw = canv.image_data[:, :, 3] > 0
                         # חיתוך: איפה ציירתי וגם יש קיר
                         intersection = np.logical_and(user_draw, walls_resized > 0)
                         # חישוב אורך לפי סקלטון של החיתוך
                         inter_u8 = intersection.astype(np.uint8) * 255
                         try:
                            skel_inter = cv2.ximgproc.thinning(inter_u8)
                            val = cv2.countNonZero(skel_inter) / (proj["scale"] * f)
                         except:
                            val = cv2.countNonZero(inter_u8) / (proj["scale"] * f * 10) # הערכה
                         
                         unit = "מטר"
                 else:
                     # ריצוף
                     if canv.image_data is not None:
                         px = np.count_nonzero(canv.image_data[:, :, 3] > 0)
                         val = px / ((proj["scale"]*f)**2)
                         unit = "מ\"ר"
                 
                 if val > 0:
                     pid = save_plan(sel_p, sel_p, "1:50", proj["scale"], proj["raw_pixels"], "{}")
                     save_progress_report(pid, val, f"{rep_type}")
                     st.success(f"דיווח התקבל! ({val:.2f} {unit})")
                     st.balloons()
                 else:
                     st.warning("לא זוהה סימון על גבי האלמנטים הרלוונטיים")
    else: st.error("אין תוכניות")