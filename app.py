import streamlit as st
import sys
import types

# --- 🛠️ התיקון המנצח (The Magic Patch) 🛠️ ---
try:
    from streamlit.elements.lib.image_utils import image_to_url
    if "streamlit.elements.image" not in sys.modules:
        sys.modules["streamlit.elements.image"] = types.ModuleType("streamlit.elements.image")
    import streamlit.elements.image
    streamlit.elements.image.image_to_url = image_to_url
except ImportError:
    pass

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
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_plan_by_id, get_all_plans,
    get_project_forecast, 
    calculate_material_estimates, get_project_financial_status, reset_all_data
)
from datetime import datetime

# --- הגדרות ראשוניות ---
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- פונקציית עזר לטיפול בקובץ brain.py חסר ---
def safe_process_metadata(raw_text):
    """ניסיון לעבד מטא-דאטה עם fallback אם brain.py חסר"""
    try:
        from brain import process_plan_metadata
        return process_plan_metadata(raw_text)
    except (ImportError, Exception) as e:
        st.warning(f"AI metadata processing unavailable: {str(e)}")
        return {}

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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Heebo', sans-serif; direction: rtl; }
    :root { --primary-blue: #0F62FE; --bg-gray: #F4F7F6; --card-border: #E0E0E0; --text-dark: #161616; --text-meta: #6F6F6F; }
    .stCard { background-color: white; padding: 24px; border-radius: 12px; border: 1px solid var(--card-border); box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 20px; }
    .kpi-container { display: flex; flex-direction: column; background: white; padding: 20px; border-radius: 12px; border: 1px solid #EAEAEA; box-shadow: 0 4px 12px rgba(0,0,0,0.03); height: 100%; }
    .kpi-icon { font-size: 24px; margin-bottom: 12px; background: #F0F5FF; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 50%; }
    .kpi-label { font-size: 14px; color: var(--text-meta); font-weight: 500; }
    .kpi-value { font-size: 28px; font-weight: 700; color: var(--text-dark); margin-top: 4px; }
    .kpi-sub { font-size: 13px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #F0F0F0; }
    .mat-card { text-align: center; background: white; border: 1px solid #EEE; border-radius: 10px; padding: 15px; }
    .mat-val { font-size: 20px; font-weight: bold; color: var(--primary-blue); }
    .mat-lbl { font-size: 14px; color: #666; }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label { text-align: right !important; width: 100%; direction: rtl; }
    .stButton button { border-radius: 8px; font-weight: 500; height: 45px; }
    section[data-testid="stSidebar"] { background-color: #FAFAFA; border-left: 1px solid #EEE; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942823.png", width=50)
    st.markdown("### **ConTech Pro**")
    st.caption("מערכת ניהול ובקרה לקבלני שלד")
    st.markdown("---")
    mode = st.radio("בחר אזור עבודה:", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות", expanded=False):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ איפוס מערכת מלא", help="מוחק את כל הנתונים והפרויקטים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.success("המערכת אופסה")
            st.rerun()

if mode == "🏢 מנהל פרויקט":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("ניהול פרויקטים")
        st.caption("העלאת תוכניות, כיול ובקרת תקציב")
    
    tab1, tab2 = st.tabs(["📂 העלאת תוכניות", "📊 דשבורד מנהלים"])
    with tab1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        files = st.file_uploader("גרור לכאן קבצי PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if files:
            for f in files:
                if f.name not in st.session_state.projects:
                    with st.spinner(f"מפענח את {f.name}..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(f.getvalue())
                                path = tmp.name
                            analyzer = FloorPlanAnalyzer()
                            pix, skel, thick, orig, meta = analyzer.process_file(path)
                            if not meta.get("plan_name"): 
                                meta["plan_name"] = f.name.replace(".pdf", "").replace("-", " ").strip()
                            
                            raw_text = meta.get("raw_text", "")
                            llm_metadata = {}
                            if raw_text:
                                llm_metadata = safe_process_metadata(raw_text)
                                if llm_metadata.get("plan_name"): 
                                    meta["plan_name"] = llm_metadata["plan_name"]
                                if llm_metadata.get("scale"): 
                                    meta["scale"] = llm_metadata["scale"]
                            
                            st.session_state.projects[f.name] = {
                                "skeleton": skel, "thick_walls": thick, "original": orig,
                                "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                "total_length": pix / 200.0, "llm_suggestions": llm_metadata
                            }
                            os.unlink(path)
                            st.success(f"✅ {f.name} נטען בהצלחה")
                        except Exception as e:
                            st.error(f"❌ שגיאה בטעינת {f.name}: {str(e)}")

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", options=list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state: st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state: st.session_state[scale_key] = proj["metadata"].get("scale", "")

            col_edit, col_preview = st.columns([1, 1.5])
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                p_name = st.text_input("שם התוכנית", key=name_key)
                p_scale = st.text_input("קנה מידה", key=scale_key)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    target_date_val = st.date_input("תאריך יעד", key=f"td_{selected}")
                    target_date_str = target_date_val.strftime("%Y-%m-%d") if target_date_val else None
                with col_d2: budget_limit_val = st.number_input("תקציב (₪)", step=1000.0, key=f"bl_{selected}")
                cost_per_meter_val = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, key=f"cpm_{selected}")
                st.markdown("#### כיול")
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]), key=f"sl_{selected}")
                proj["scale"] = scale_val
                proj["total_length"] = proj["raw_pixels"] / scale_val
                st.info(f"📏 אורך קירות: **{proj['total_length']:.2f} מטר**")
                if st.button("💾 שמור נתונים", type="primary", use_container_width=True):
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale
                    metadata_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    save_plan(selected, p_name, p_scale, scale_val, proj["raw_pixels"], metadata_json, target_date_str, budget_limit_val, cost_per_meter_val, json.dumps(materials, ensure_ascii=False))
                    st.success("נשמר!")

            with col_preview:
                st.image(proj["skeleton"], caption="זיהוי קירות", use_column_width=True)
                if proj["total_length"] > 0:
                    mats = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    st.markdown("###### הערכה מהירה")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['block_count']:,}</div><div class='mat-lbl'>בלוקים</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['cement_cubic_meters']:.1f}</div><div class='mat-lbl'>מ\"ק מלט</div></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['wall_area_sqm']:.0f}</div><div class='mat-lbl'>מ\"ר קיר</div></div>", unsafe_allow_html=True)

    with tab2:
        all_plans = get_all_plans()
        if not all_plans: st.info("אנא שמור תוכנית אחת לפחות.")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_display = st.selectbox("בחר פרויקט:", plan_options)
            selected_id = int(selected_display.split("(ID: ")[1].split(")")[0])
            forecast = get_project_forecast(selected_id)
            fin = get_project_financial_status(selected_id)
            
            days_val = forecast['days_to_finish']
            days_left_display = days_val if days_val > 0 else "-"

            st.markdown("#### סטטוס ביצוע")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1: st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">🏗️</div><div class="kpi-label">בוצע בפועל</div><div class="kpi-value">{forecast['cumulative_progress']:.1f} מ'</div><div class="kpi-sub">מתוך {forecast['total_planned']:.1f} מ'</div></div>""", unsafe_allow_html=True)
            with kpi2:
                pct = (forecast['cumulative_progress'] / forecast['total_planned'] * 100) if forecast['total_planned'] > 0 else 0
                st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">📊</div><div class="kpi-label">אחוז השלמה</div><div class="kpi-value">{pct:.1f}%</div><div class="kpi-sub">נותרו {forecast['remaining_work']:.1f} מ'</div></div>""", unsafe_allow_html=True)
            with kpi3: st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">📅</div><div class="kpi-label">ימים לסיום</div><div class="kpi-value">{days_left_display}</div><div class="kpi-sub">קצב: {forecast['average_velocity']:.1f} מ'/יום</div></div>""", unsafe_allow_html=True)
            with kpi4:
                cost_color = "#ef4444" if fin['budget_variance'] < 0 else "#10b981"
                st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">💰</div><div class="kpi-label">עלות נוכחית</div><div class="kpi-value">{fin['current_cost']:,.0f} ₪</div><div class="kpi-sub" style="color: {cost_color}">תקציב: {fin['budget_limit']:,.0f} ₪</div></div>""", unsafe_allow_html=True)
            
            g_col, t_col = st.columns([2, 1])
            with g_col:
                st.markdown("##### קצב התקדמות")
                df = load_stats_df()
                if not df.empty: st.bar_chart(df, x="תאריך", y="מטרים שבוצעו", use_container_width=True)
            with t_col:
                st.markdown("##### דיווחים אחרונים")
                if not df.empty: st.dataframe(df[["תאריך", "מטרים שבוצעו", "הערה"]].head(5), hide_index=True, use_container_width=True)

elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    if not st.session_state.projects: 
        st.info("אין תוכניות זמינות. עבור למנהל פרויקט להעלאת תוכניות.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        try:
            orig_rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
            h, w = orig_rgb.shape[:2]
            thick_walls = proj["thick_walls"]
            
            # וידוא שהממדים תואמים
            if thick_walls.shape[:2] != (h, w): 
                thick_walls = cv2.resize(thick_walls, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # --- תיקון: הגדרת המשתנה dilated ---
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate((thick_walls > 0).astype(np.uint8) * 255, kernel, iterations=2)
            
            col_opacity, col_spacer = st.columns([2, 1])
            with col_opacity: 
                opacity = st.slider("עוצמת הדגשת קירות", 0.0, 1.0, 0.4)
            
            overlay = np.zeros_like(orig_rgb)
            overlay[dilated > 0] = [0, 120, 255]
            
            combined = cv2.addWeighted(orig_rgb, 1-opacity, overlay, opacity, 0).astype(np.uint8)
            
            # --- תיקון קריטי: הקטנת גודל הקנבס למניעת בעיות זיכרון ---
            max_canvas_width = 800
            if w > max_canvas_width:
                factor = max_canvas_width / w
                c_width = max_canvas_width
                c_height = int(h * factor)
            else:
                c_width = w
                c_height = h
                factor = 1.0
            
            # שמירת הפקטור בסשן סטייט לשימוש בחישובים
            if 'canvas_factor' not in st.session_state:
                st.session_state.canvas_factor = {}
            st.session_state.canvas_factor[plan_name] = factor
            
            # וידוא שהערכים הם integers
            c_width = int(c_width)
            c_height = int(c_height)
            
            # המרה ושינוי גודל - חשוב! combined כבר RGB
            combined_resized = cv2.resize(combined, (c_width, c_height), interpolation=cv2.INTER_AREA)
            
            # וידוא שהמערך הוא uint8
            if combined_resized.dtype != np.uint8:
                combined_resized = combined_resized.astype(np.uint8)
            
            # המרה ל-PIL Image
            bg_image_pil = Image.fromarray(combined_resized, mode='RGB')
            
            st.markdown("**סמן את הקירות שבנית היום (בירוק):**")
            st.caption(f"גודל קנבס: {c_width}x{c_height} | תמונה: {bg_image_pil.size} | מצב: {bg_image_pil.mode}")
            
            # --- מפתח יציב ---
            canvas_key = f"canvas_{plan_name}"
            
            try:
                # שימוש ב-PIL Image - הדרך התקנית
                canvas = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=8,
                    stroke_color="#00FF00", 
                    background_image=bg_image_pil,
                    height=c_height,  # חשוב: height לפני width
                    width=c_width,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key=canvas_key, 
                    update_streamlit=True
                )
                st.success("✅ הקנבס נטען בהצלחה")
            except AttributeError as e:
                if "'str' object has no attribute" in str(e):
                    st.error("השגיאה מצביעה על בעיית גרסה. מנסה numpy array...")
                    try:
                        canvas = st_canvas(
                            stroke_width=8,
                            stroke_color="#00FF00", 
                            background_image=combined_resized,  # numpy array
                            height=c_height,
                            width=c_width,
                            drawing_mode="freedraw",
                            key=f"{canvas_key}_numpy", 
                            update_streamlit=True
                        )
                        st.success("✅ הקנבס נטען עם numpy array")
                    except Exception as e2:
                        st.error(f"numpy גם לא עבד: {str(e2)}")
                        canvas = None
                else:
                    st.error(f"שגיאה: {str(e)}")
                    canvas = None
            except Exception as canvas_error:
                st.error(f"❌ שגיאה בטעינת הקנבס: {str(canvas_error)}")
                canvas = None
            
            if canvas and canvas.json_data is not None and canvas.json_data.get("objects"):
                try:
                    w_mask = np.zeros((c_height, c_width), dtype=np.uint8)
                    df_obj = pd.json_normalize(canvas.json_data["objects"])
                    
                    for _, obj in df_obj.iterrows():
                        # טיפול בשני סוגי אובייקטים: line ו-path
                        if 'path' in obj:  # freedraw
                            path = obj['path']
                            if isinstance(path, list) and len(path) > 0:
                                points = []
                                for p in path:
                                    if isinstance(p, list) and len(p) >= 3:
                                        points.append([int(p[1]), int(p[2])])
                                if len(points) > 1:
                                    points = np.array(points, dtype=np.int32)
                                    cv2.polylines(w_mask, [points], False, 255, 8)
                        
                        elif 'left' in obj and 'top' in obj and 'x1' in obj:  # line mode
                            l, t = int(obj['left']), int(obj['top'])
                            p1 = (l + int(obj['x1']), t + int(obj['y1']))
                            p2 = (l + int(obj['x2']), t + int(obj['y2']))
                            cv2.line(w_mask, p1, p2, 255, 8)
                    
                    # שינוי גודל של dilated לגודל הקנבס
                    walls_res = cv2.resize(dilated, (c_width, c_height), interpolation=cv2.INTER_NEAREST)
                    intersection = cv2.bitwise_and(w_mask, walls_res)
                    pixels = cv2.countNonZero(intersection)
                    
                    # חישוב מטרים - תיקון החישוב
                    if proj["scale"] > 0 and factor > 0:
                        meters = (pixels / factor) / proj["scale"]
                    else:
                        meters = 0
                    
                    if meters > 0:
                        st.success(f"✅ נמדדו: **{meters:.2f} מטר**")
                        note = st.text_input("הערה לדיווח (אופציונלי)")
                        
                        if st.button("🚀 שלח דיווח", type="primary", use_container_width=True):
                            rec = get_plan_by_filename(plan_name)
                            if rec:
                                pid = rec['id']
                            else:
                                # יצירת רשומה חדשה אם לא קיימת
                                pid = save_plan(
                                    plan_name, 
                                    proj["metadata"].get("plan_name", plan_name), 
                                    proj["metadata"].get("scale", ""), 
                                    proj["scale"], 
                                    proj["raw_pixels"], 
                                    json.dumps(proj["metadata"], ensure_ascii=False)
                                )
                            
                            save_progress_report(pid, meters, note)
                            st.balloons()
                            st.success("הדיווח נשלח בהצלחה!")
                    else:
                        st.info("⏳ סמן על הקירות כדי לחשב את המדידה")
                        
                except Exception as e:
                    st.error(f"שגיאה בעיבוד הציור: {str(e)}")
            else:
                st.info("⏳ התחל לצייר על הקנבס")
                
        except Exception as e:
            st.error(f"❌ שגיאה בטעינת הקנבס: {str(e)}")
            st.info("נסה לרענן את הדף או לבחור תוכנית אחרת")
