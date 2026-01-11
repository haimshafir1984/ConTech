import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from analyzer import FloorPlanAnalyzer
import tempfile
import os
import json
from streamlit_drawable_canvas import st_canvas
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_plan_by_id, get_all_plans,
    calculate_velocity, get_project_forecast, 
    calculate_material_estimates, get_project_financial_status, reset_all_data
)
from brain import learn_from_confirmation, process_plan_metadata
from datetime import datetime

# תיקון תאימות תמונות
try:
    import streamlit.elements.image as st_image
    from streamlit.elements.lib.image_utils import image_to_url
    st_image.image_to_url = image_to_url
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = None
init_database()

# פונקציית טעינת נתונים משופרת
def load_stats_df():
    reports = get_progress_reports()
    if reports:
        df = pd.DataFrame(reports)
        # המרה לפורמט עברי
        return df.rename(columns={
            'date': 'תאריך', 'plan_name': 'שם תוכנית',
            'meters_built': 'מטרים שבוצעו', 'note': 'הערה'
        })
    return pd.DataFrame()

st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")

# --- CSS עיצוב נקי ומוקפד (Clean UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Heebo', sans-serif;
        direction: rtl;
    }
    
    /* צבעים מוגדרים */
    :root {
        --primary-blue: #0F62FE; /* IBM Blue */
        --bg-gray: #F4F7F6;
        --card-border: #E0E0E0;
        --text-dark: #161616;
        --text-meta: #6F6F6F;
    }
    
    /* עיצוב כרטיסיות כללי - נקי ושטוח */
    .stCard {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid var(--card-border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* KPI Cards - עיצוב חדש */
    .kpi-container {
        display: flex;
        flex-direction: column;
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #EAEAEA;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        height: 100%;
        transition: all 0.2s ease;
    }
    
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border-color: var(--primary-blue);
    }
    
    .kpi-icon {
        font-size: 24px;
        margin-bottom: 12px;
        background: #F0F5FF;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }
    
    .kpi-label {
        font-size: 14px;
        color: var(--text-meta);
        font-weight: 500;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-dark);
        margin-top: 4px;
    }
    
    .kpi-sub {
        font-size: 13px;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #F0F0F0;
    }

    /* Material Cards Minimal */
    .mat-card {
        text-align: center;
        background: white;
        border: 1px solid #EEE;
        border-radius: 10px;
        padding: 15px;
    }
    .mat-val { font-size: 20px; font-weight: bold; color: var(--primary-blue); }
    .mat-lbl { font-size: 14px; color: #666; }

    /* RTL Override חזק */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        text-align: right !important;
        width: 100%;
        direction: rtl;
    }
    
    /* כפתורים */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        height: 45px;
    }

    /* Sidebar Clean */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-left: 1px solid #EEE;
    }
    
    /* הסתרת דקורציות של Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

if 'projects' not in st.session_state:
    st.session_state.projects = {}

if 'wall_height' not in st.session_state:
    st.session_state.wall_height = 2.5

if 'default_cost_per_meter' not in st.session_state:
    st.session_state.default_cost_per_meter = 0.0

# --- סרגל צד (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942823.png", width=50) # לוגו זמני
    st.markdown("### **ConTech Pro**")
    st.caption("מערכת ניהול ובקרה לקבלני שלד")
    
    st.markdown("---")
    mode = st.radio("בחר אזור עבודה:", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    
    # הגדרות גלובליות
    with st.expander("⚙️ הגדרות גלובליות", expanded=False):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
    
    # אזור מחיקה - עדין יותר
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ איפוס מערכת מלא", help="מוחק את כל הנתונים והפרויקטים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.success("המערכת אופסה")
            st.rerun()

# --- לוגיקה ראשית ---

if mode == "🏢 מנהל פרויקט":
    
    # כותרת ראשית מעוצבת
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("ניהול פרויקטים")
        st.caption("העלאת תוכניות, כיול ובקרת תקציב")
    
    tab1, tab2 = st.tabs(["📂 העלאת תוכניות", "📊 דשבורד מנהלים"])
    
    with tab1:
        # אזור העלאה מעוצב ככרטיסיה
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        files = st.file_uploader("גרור לכאן קבצי PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if files:
            for f in files:
                if f.name not in st.session_state.projects:
                    with st.spinner(f"מפענח את {f.name} באמצעות AI..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.getvalue())
                            path = tmp.name
                        
                        analyzer = FloorPlanAnalyzer()
                        pix, skel, thick, orig, meta = analyzer.process_file(path)
                        
                        # ניסיון חילוץ שם בסיסי משם הקובץ אם המטא-דאטה ריק
                        if not meta.get("plan_name"):
                            meta["plan_name"] = f.name.replace(".pdf", "").replace("-", " ").strip()

                        # חילוץ AI
                        raw_text = meta.get("raw_text", "")
                        llm_metadata = {}
                        if raw_text:
                            try:
                                llm_metadata = process_plan_metadata(raw_text)
                                # עדכון המטא רק אם ה-AI החזיר משהו הגיוני
                                if llm_metadata.get("plan_name"):
                                    meta["plan_name"] = llm_metadata["plan_name"]
                                if llm_metadata.get("scale"):
                                    meta["scale"] = llm_metadata["scale"]
                            except:
                                pass
                        
                        st.session_state.projects[f.name] = {
                            "skeleton": skel, "thick_walls": thick, "original": orig,
                            "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                            "total_length": pix / 200.0, "llm_suggestions": llm_metadata
                        }
                        os.unlink(path)

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", options=list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            # --- תיקון השדות הריקים ---
            # קביעת ערכי ברירת מחדל חזקים
            current_name = proj["metadata"].get("plan_name", "")
            if not current_name: 
                current_name = selected.replace(".pdf", "")
            
            current_scale = proj["metadata"].get("scale", "")
            
            # עדכון יזום של ה-Session State כדי שהשדות יתמלאו
            name_key = f"n_{selected}"
            scale_key = f"s_{selected}"
            
            if name_key not in st.session_state:
                st.session_state[name_key] = current_name
            if scale_key not in st.session_state:
                st.session_state[scale_key] = current_scale

            # אזור העריכה
            col_edit, col_preview = st.columns([1, 1.5])
            
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                
                # שדות טקסט
                p_name = st.text_input("שם התוכנית (למשל: קומה א')", key=name_key)
                p_scale = st.text_input("קנה מידה (למשל: 1:50)", key=scale_key)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    target_date_val = st.date_input("תאריך יעד", key=f"td_{selected}")
                    target_date_str = target_date_val.strftime("%Y-%m-%d") if target_date_val else None
                with col_d2:
                    budget_limit_val = st.number_input("תקציב (₪)", step=1000.0, key=f"bl_{selected}")

                cost_per_meter_val = st.number_input("עלות למטר רץ (₪)", value=st.session_state.default_cost_per_meter, key=f"cpm_{selected}")
                
                st.markdown("#### כיול")
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]), key=f"sl_{selected}")
                proj["scale"] = scale_val
                
                # חישוב אמת
                proj["total_length"] = proj["raw_pixels"] / scale_val
                st.info(f"📏 אורך קירות מזוהה: **{proj['total_length']:.2f} מטר**")

                if st.button("💾 שמור נתונים", type="primary", use_container_width=True):
                    # עדכון המטא דאטה הפנימי
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale
                    
                    from database import save_plan
                    metadata_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    
                    save_plan(
                        filename=selected,
                        plan_name=p_name,
                        extracted_scale=p_scale,
                        confirmed_scale=scale_val,
                        raw_pixel_count=proj["raw_pixels"],
                        metadata_json=metadata_json,
                        target_date=target_date_str,
                        budget_limit=budget_limit_val,
                        cost_per_meter=cost_per_meter_val,
                        material_estimate=json.dumps(materials, ensure_ascii=False)
                    )
                    st.success("הנתונים נשמרו בהצלחה!")

            with col_preview:
                st.image(proj["skeleton"], caption="זיהוי קירות (תצוגה מקדימה)", use_container_width=True)
                
                # כרטיסי חומרים מהירים מתחת לתמונה
                if proj["total_length"] > 0:
                    mats = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    st.markdown("###### הערכת חומרים מהירה")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['block_count']:,}</div><div class='mat-lbl'>בלוקים</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['cement_cubic_meters']:.1f}</div><div class='mat-lbl'>מ\"ק מלט</div></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['wall_area_sqm']:.0f}</div><div class='mat-lbl'>מ\"ר קיר</div></div>", unsafe_allow_html=True)

    with tab2:
        # דשבורד מנהלים
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אנא שמור תוכנית אחת לפחות כדי לראות נתונים.")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_display = st.selectbox("בחר פרויקט לניתוח:", plan_options)
            selected_id = int(selected_display.split("(ID: ")[1].split(")")[0])
            
            # שליפת נתונים מחושבים
            forecast = get_project_forecast(selected_id)
            fin = get_project_financial_status(selected_id)
            
            # --- שורת KPIs ראשית ---
            st.markdown("#### סטטוס ביצוע")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-icon">🏗️</div>
                    <div class="kpi-label">בוצע בפועל</div>
                    <div class="kpi-value">{forecast['cumulative_progress']:.1f} מ'</div>
                    <div class="kpi-sub">מתוך {forecast['total_planned']:.1f} מ'</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi2:
                pct = (forecast['cumulative_progress'] / forecast['total_planned'] * 100) if forecast['total_planned'] > 0 else 0
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-icon">📊</div>
                    <div class="kpi-label">אחוז השלמה</div>
                    <div class="kpi-value">{pct:.1f}%</div>
                    <div class="kpi-sub">נותרו {forecast['remaining_work']:.1f} מ'</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi3:
                days_left = forecast['days_to_finish'] if forecast['days_to_finish'] > 0 else "-"
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-icon">📅</div>
                    <div class="kpi-label">ימים לסיום</div>
                    <div class="kpi-value">{days_left}</div>
                    <div class="kpi-sub">קצב: {forecast['average_velocity']:.1f} מ'/יום</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi4:
                cost_color = "#ef4444" if fin['budget_variance'] < 0 else "#10b981"
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-label">עלות נוכחית</div>
                    <div class="kpi-value">{fin['current_cost']:,.0f} ₪</div>
                    <div class="kpi-sub" style="color: {cost_color}">תקציב: {fin['budget_limit']:,.0f} ₪</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("###") # ריווח
            
            # גרף וטבלה
            g_col, t_col = st.columns([2, 1])
            with g_col:
                st.markdown("##### קצב התקדמות יומי")
                df = load_stats_df()
                if not df.empty:
                    # סינון לפי התוכנית שנבחרה אם צריך, כרגע מציג הכל
                    st.bar_chart(df, x="תאריך", y="מטרים שבוצעו")
                else:
                    st.info("אין נתונים להצגה בגרף")
            
            with t_col:
                st.markdown("##### דיווחים אחרונים")
                if not df.empty:
                    st.dataframe(df[["תאריך", "מטרים שבוצעו", "הערה"]].head(5), hide_index=True)

elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.info("אין תוכניות זמינות. אנא פנה למנהל הפרויקט.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        # הכנת התצוגה (לוגיקה זהה למקור עם שיפורי Hitbox)
        orig_rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = orig_rgb.shape[:2]
        
        # התאמת מסכות
        thick_walls = proj["thick_walls"]
        if thick_walls.shape[:2] != (h, w):
            thick_walls = cv2.resize(thick_walls, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Hitbox מוגדל (15px) כדי להקל על העובד
        kernel = np.ones((15, 15), np.uint8)
        mask = (thick_walls > 0).astype(np.uint8) * 255
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # תצוגה
        col_opacity, col_spacer = st.columns([2, 1])
        with col_opacity:
            opacity = st.slider("עוצמת הדגשת קירות", 0.0, 1.0, 0.4)
            
        # יצירת Overlay
        overlay = np.zeros_like(orig_rgb)
        overlay[dilated_mask > 0] = [0, 120, 255] # כחול
        
        combined = cv2.addWeighted(orig_rgb, 1-opacity, overlay, opacity, 0)
        
        # קנבס
        c_width = 1000
        factor = c_width / w
        c_height = int(h * factor)
        
        combined_res = cv2.resize(combined, (c_width, c_height))
        
        st.markdown("**סמן את הקירות שבנית היום (בירוק):**")
        canvas = st_canvas(
            stroke_width=5,
            stroke_color="#00FF00",
            background_image=Image.fromarray(combined_res),
            width=c_width,
            height=c_height,
            drawing_mode="line",
            key=f"worker_{plan_name}"
        )
        
        # חישוב ביצוע
        if canvas.json_data and canvas.json_data["objects"]:
            # יצירת מסכת עובד בגודל קנבס
            w_mask = np.zeros((c_height, c_width), dtype=np.uint8)
            df_obj = pd.json_normalize(canvas.json_data["objects"])
            
            for _, obj in df_obj.iterrows():
                # טיפול בקואורדינטות בצורה בסיסית אך רובסטית
                p1 = (int(obj['left']), int(obj['top']))
                # בדיקה אם זה קו (בדרך כלל SVG path או x1/x2)
                # פישוט: נניח קווים ישרים לפי x1,y1,x2,y2 יחסיים ל-left/top
                if 'x1' in obj: 
                    p1 = (int(obj['left'] + obj['x1']), int(obj['top'] + obj['y1']))
                    p2 = (int(obj['left'] + obj['x2']), int(obj['top'] + obj['y2']))
                    cv2.line(w_mask, p1, p2, 255, 5)

            # בדיקת חפיפה (Resize את הקירות לגודל קנבס)
            walls_res = cv2.resize(dilated_mask, (c_width, c_height), interpolation=cv2.INTER_NEAREST)
            intersection = cv2.bitwise_and(w_mask, walls_res)
            
            # חישוב מטרים
            pixels = cv2.countNonZero(intersection)
            # פקטור המרה: פיקסלים בקנבס -> פיקסלים במקור -> מטרים
            # scale = פיקסלים במקור למטר
            # factor = יחס קנבס למקור
            
            # חישוב מדויק: (פיקסלים בקנבס / פקטור הקטנה) / סקלה
            meters = (pixels / factor) / proj["scale"]
            
            # תצוגה
            st.success(f"✅ נמדדו: **{meters:.2f} מטר**")
            
            note = st.text_input("הערה לדיווח")
            if st.button("🚀 שלח דיווח", type="primary", use_container_width=True):
                 # לוגיקת שמירה (בדיקה אם קיים ב-DB וכו')
                 from database import get_plan_by_filename, save_plan
                 rec = get_plan_by_filename(plan_name)
                 
                 # אם לא נשמר ב-DB עדיין, שומרים אוטומטית
                 pid = rec['id'] if rec else save_plan(
                     plan_name, 
                     proj["metadata"].get("plan_name", plan_name), 
                     "", proj["scale"], proj["raw_pixels"], "{}", proj["scale"]
                 )
                 
                 save_progress_report(pid, meters, note)
                 st.balloons()
                 st.success("הדיווח נשלח בהצלחה!")