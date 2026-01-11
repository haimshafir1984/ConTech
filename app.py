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