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

# CSS מותאם אישית לעיצוב מקצועי
st.markdown("""
<style>
    /* צבעי נושא - כחול בנייה/אפור הנדסי */
    :root {
        --construction-blue: #2563eb;
        --engineering-gray: #1e293b;
        --accent-orange: #f59e0b;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --planned-blue: #3b82f6;
        --completed-green: #10b981;
        --remaining-orange: #f59e0b;
    }
    
    /* KPI Cards מעוצבים - צבעים שונים לפי סוג */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    .kpi-card.planned {
        border-left: 4px solid var(--planned-blue);
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    .kpi-card.completed {
        border-left: 4px solid var(--completed-green);
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .kpi-card.remaining {
        border-left: 4px solid var(--remaining-orange);
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .kpi-card .kpi-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-card .kpi-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--engineering-gray);
        margin: 0;
    }
    
    .kpi-card .kpi-delta {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    /* Material Cards */
    .material-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .material-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-color: var(--construction-blue);
    }
    
    .material-card .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .material-card .label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    
    .material-card .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: var(--engineering-gray);
    }
    
    /* Dangerous Zone - רקע אדום דהוי קבוע */
    .danger-zone {
        background: #fef2f2 !important;
        border: 2px solid var(--danger-red);
        border-radius: 8px;
        padding: 1rem;
        margin-top: auto;
        margin-bottom: 0;
    }
    
    .danger-zone h3 {
        color: var(--danger-red);
        margin-bottom: 0.5rem;
    }
    
    /* RTL Support מוחלט - כל הטבלאות והכותרות */
    [dir="rtl"], 
    .stDataFrame,
    .stDataFrame *,
    .stTable,
    .stTable *,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown,
    .stText,
    .stMetric,
    .stSelectbox,
    .stTextInput,
    .stNumberInput,
    .stSlider,
    .stDateInput,
    .stRadio,
    .stCheckbox,
    .stButton,
    .stExpander,
    .stContainer,
    div[data-testid],
    .element-container,
    .stAlert,
    .stInfo,
    .stSuccess,
    .stWarning,
    .stError {
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: var(--engineering-gray);
        text-align: right !important;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--construction-blue), var(--success-green));
    }
    
    /* Sidebar Styling */
    .stSidebar {
        display: flex;
        flex-direction: column;
    }
    
    .stSidebar > div:first-child {
        flex-grow: 1;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid var(--success-green);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--success-green);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)

if 'projects' not in st.session_state:
    st.session_state.projects = {}

if 'wall_height' not in st.session_state:
    st.session_state.wall_height = 2.5

if 'default_cost_per_meter' not in st.session_state:
    st.session_state.default_cost_per_meter = 0.0

with st.sidebar:
    # בחירת משתמש בראש התפריט
    st.title("🏗️ ConTech Pro")
    st.divider()
    mode = st.radio("בחר משתמש:", ["מנהל פרויקט (Admin)", "דיווח ביצוע (Worker)"], key="user_mode")
    
    st.divider()
    
    # Project Settings
    with st.expander("⚙️ הגדרות פרויקט", expanded=False):
        wall_height = st.number_input("גובה קירות (מטר):", min_value=1.0, max_value=10.0, 
                                       value=st.session_state.wall_height, step=0.1, key="wall_height_setting")
        st.session_state.wall_height = wall_height
        
        default_cost = st.number_input("עלות למטר (₪):", min_value=0.0, value=st.session_state.default_cost_per_meter, 
                                       step=10.0, key="cost_per_meter_setting")
        st.session_state.default_cost_per_meter = default_cost
        
        st.info(f"💡 הגדרות אלה ישפיעו על חישובי החומרים והתקציב")
    
    # Dangerous Zone - בתחתית התפריט עם רקע אדום דהוי קבוע
    st.markdown('<div style="margin-top: auto; padding-top: 2rem;">', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
    st.markdown("### ⚠️ אזור מסוכן")
    st.markdown("**🗑️ איפוס נתוני פרויקט**")
    st.markdown("<small>פעולה זו תמחק את כל התוכניות והדיווחים. לא ניתן לבטל!</small>", unsafe_allow_html=True)
    
    # לוגיקת איפוס משופרת
    if 'reset_confirm' not in st.session_state:
        st.session_state.reset_confirm = False
    
    if not st.session_state.reset_confirm:
        if st.button("🗑️ איפוס כל הנתונים", type="secondary", use_container_width=True):
            st.session_state.reset_confirm = True
            st.rerun()
    else:
        st.warning("⚠️ **אזהרה:** פעולה זו תמחק את כל הנתונים מהמערכת!")
        confirm = st.checkbox("אני מבין שפעולה זו לא ניתנת לביטול", key="reset_checkbox")
        
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ אשר איפוס", type="primary", use_container_width=True):
                if reset_all_data():
                    st.success("✅ כל הנתונים נמחקו בהצלחה!")
                    st.session_state.projects = {}
                    st.session_state.reset_confirm = False
                    st.rerun()
                else:
                    st.error("❌ שגיאה במחיקת הנתונים. אנא נסה שוב.")
                    st.session_state.reset_confirm = False
        
        with col_cancel:
            if st.button("❌ ביטול", use_container_width=True):
                st.session_state.reset_confirm = False
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- מצב מנהל ---
if mode == "מנהל פרויקט (Admin)":
    tab1, tab2 = st.tabs(["📂 ניהול וכיול", "📊 דשבורד פרויקט"])
    
    with tab1:
        files = st.file_uploader("העלה תוכניות (PDF)", type="pdf", accept_multiple_files=True)
        if files:
            for f in files:
                if f.name not in st.session_state.projects:
                    with st.spinner(f"מעבד את {f.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.getvalue())
                            path = tmp.name
                        analyzer = FloorPlanAnalyzer()
                        pix, skel, thick, orig, meta = analyzer.process_file(path)
                        
                        # שימוש ב-LLM לחילוץ מטא-דאטה משופר
                        raw_text = meta.get("raw_text", "")
                        llm_metadata = {}
                        if raw_text:
                            try:
                                llm_metadata = process_plan_metadata(raw_text)
                                # עדכון המטא-דאטה עם התוצאות מה-LLM
                                meta.update(llm_metadata)
                            except Exception as e:
                                st.warning(f"⚠️ לא הצלחתי להשתמש ב-LLM לחילוץ מטא-דאטה: {e}")
                        
                        st.session_state.projects[f.name] = {
                            "skeleton": skel, "thick_walls": thick, "original": orig,
                            "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                            "total_length": pix / 200.0, "llm_suggestions": llm_metadata
                        }
                        os.unlink(path)

        if st.session_state.projects:
            selected = st.selectbox("בחר שרטוט:", options=list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            # הצגת המלצות LLM אם יש
            if proj.get("llm_suggestions"):
                suggestions = proj["llm_suggestions"]
                if any(suggestions.values()):
                    with st.expander("🤖 המלצות AI (LLM)", expanded=True):
                        if suggestions.get("plan_name"):
                            st.success(f"✅ שם תוכנית מזוהה: **{suggestions['plan_name']}**")
                        if suggestions.get("scale"):
                            st.info(f"📏 סקלה מזוהה: **{suggestions['scale']}**")
                        if suggestions.get("units"):
                            st.info(f"📐 יחידות: **{suggestions['units']}**")
            
            # עריכת מידע בסיסי
            col_in1, col_in2 = st.columns(2)
            default_name = proj["metadata"].get("plan_name") or proj.get("llm_suggestions", {}).get("plan_name") or ""
            default_scale = proj["metadata"].get("scale") or proj.get("llm_suggestions", {}).get("scale") or ""
            p_name = col_in1.text_input("שם התוכנית:", value=default_name, key=f"n_{selected}")
            p_scale = col_in2.text_input("סקלה בשרטוט:", value=default_scale, key=f"s_{selected}")

            # שדות חדשים: תאריך יעד, תקציב, עלות למטר
            st.subheader("📅 תכנון ופיננסים")
            col_date, col_budget, col_cost = st.columns(3)
            with col_date:
                target_date_val = st.date_input("תאריך יעד לסיום:", key=f"td_{selected}")
                target_date_str = target_date_val.strftime("%Y-%m-%d") if target_date_val else None
            with col_budget:
                budget_limit_val = st.number_input("תקציב כולל (₪):", min_value=0.0, value=0.0, step=1000.0, key=f"bl_{selected}")
            with col_cost:
                cost_per_meter_val = st.number_input("עלות למטר (₪):", min_value=0.0, 
                                                       value=st.session_state.default_cost_per_meter, 
                                                       step=10.0, key=f"cpm_{selected}")

            col_cal, col_view = st.columns([1, 2])
            with col_cal:
                scale_val = st.slider("כיול (פיקסלים למטר):", 10.0, 1000.0, float(proj["scale"]), key=f"sl_{selected}")
                proj["scale"] = scale_val
                proj["total_length"] = proj["raw_pixels"] / scale_val
                st.metric("סה'כ מתוכנן בקומה", f"{proj['total_length']:.2f} מטר")
                
                # חישוב והצגת הערכת חומרים עם עיצוב משופר
                if proj["total_length"] > 0:
                    materials = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    st.subheader("📦 הערכת חומרים")
                    
                    # Grid של 4 כרטיסי חומרים
                    mat_col1, mat_col2, mat_col3, mat_col4 = st.columns(4)
                    
                    with mat_col1:
                        st.markdown(f"""
                        <div class="material-card">
                            <div class="icon">🧱</div>
                            <div class="label">בלוקים</div>
                            <div class="value">{materials['block_count']:,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with mat_col2:
                        st.markdown(f"""
                        <div class="material-card">
                            <div class="icon">🪣</div>
                            <div class="label">מלט</div>
                            <div class="value">{materials['cement_cubic_meters']} מ"ק</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with mat_col3:
                        st.markdown(f"""
                        <div class="material-card">
                            <div class="icon">🏜️</div>
                            <div class="label">חול</div>
                            <div class="value">{materials['sand_cubic_meters']} מ"ק</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with mat_col4:
                        st.markdown(f"""
                        <div class="material-card">
                            <div class="icon">📐</div>
                            <div class="label">שטח קירות</div>
                            <div class="value">{materials['wall_area_sqm']:.1f} מ"ר</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.button("✅ אשר ושמור למערכת", use_container_width=True):
                    # עדכון המטא-דאטה עם הערכים המתוקנים
                    updated_meta = proj["metadata"].copy()
                    updated_meta["plan_name"] = p_name
                    updated_meta["scale"] = p_scale
                    
                    # שמירה עם השדות החדשים
                    from database import save_plan
                    import json
                    metadata_json = json.dumps(updated_meta, ensure_ascii=False)
                    
                    plan_id = save_plan(
                        filename=selected,
                        plan_name=p_name,
                        extracted_scale=p_scale,
                        confirmed_scale=scale_val,
                        raw_pixel_count=proj["raw_pixels"],
                        metadata_json=metadata_json,
                        target_date=target_date_str,
                        budget_limit=budget_limit_val if budget_limit_val > 0 else None,
                        cost_per_meter=cost_per_meter_val if cost_per_meter_val > 0 else None,
                        material_estimate=json.dumps(calculate_material_estimates(proj["total_length"], st.session_state.wall_height), ensure_ascii=False) if proj["total_length"] > 0 else None
                    )
                    st.success(f"✅ התוכנית נשמרה במסד הנתונים! (ID: {plan_id})")
                    st.balloons()

            with col_view:
                st.image(proj["skeleton"], caption="זיהוי קירות", use_container_width=True)

    with tab2:
        st.subheader("📊 דשבורד פרויקט")
        
        # בחירת תוכנית מהמסד נתונים
        all_plans = get_all_plans()
        if all_plans:
            plan_options = [f"{p['plan_name'] or p['filename']} (ID: {p['id']})" for p in all_plans]
            selected_plan_display = st.selectbox("בחר תוכנית:", options=plan_options)
            selected_plan_id = int(selected_plan_display.split("(ID: ")[1].split(")")[0])
            
            plan = get_plan_by_id(selected_plan_id)
            if plan:
                # KPIs: חיזוי וקצב עבודה - עם עיצוב מותאם אישית
                st.subheader("📈 חיזוי וקצב עבודה")
                forecast = get_project_forecast(selected_plan_id)
                
                # Progress Visualization: Execution vs Plan
                if forecast["total_planned"] > 0:
                    progress_pct = (forecast["cumulative_progress"] / forecast["total_planned"]) * 100
                    progress_pct = min(progress_pct, 100.0)
                    st.markdown(f"**התקדמות:** {forecast['cumulative_progress']:.2f} / {forecast['total_planned']:.2f} מטר ({progress_pct:.1f}%)")
                    st.progress(progress_pct / 100)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if forecast["average_velocity"] > 0:
                        st.markdown(f"""
                        <div class="kpi-card completed">
                            <div class="kpi-icon">⚡</div>
                            <div class="kpi-label">קצב עבודה ממוצע</div>
                            <div class="kpi-value">{forecast['average_velocity']:.2f} מטר/יום</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">⚡</div>
                            <div class="kpi-label">קצב עבודה ממוצע</div>
                            <div class="kpi-value">טרם חושב</div>
                            <div class="kpi-delta">נדרשים לפחות 2 ימי עבודה</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card remaining">
                        <div class="kpi-icon">📋</div>
                        <div class="kpi-label">עבודה נותרה</div>
                        <div class="kpi-value">{forecast['remaining_work']:.2f} מטר</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if forecast["days_to_finish"] > 0:
                        st.markdown(f"""
                        <div class="kpi-card planned">
                            <div class="kpi-icon">📅</div>
                            <div class="kpi-label">ימים לסיום משוער</div>
                            <div class="kpi-value">{forecast['days_to_finish']} ימי עבודה</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">📅</div>
                            <div class="kpi-label">ימים לסיום משוער</div>
                            <div class="kpi-value">טרם חושב</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    if forecast["estimated_completion_date"]:
                        date_str = forecast["estimated_completion_date"].strftime("%d/%m/%Y")
                        delta_html = ""
                        if plan.get("target_date"):
                            target_dt = datetime.strptime(plan["target_date"], "%Y-%m-%d").date()
                            delta_days = (forecast["estimated_completion_date"] - target_dt).days
                            delta_str = f"{abs(delta_days)} יום {'מאחר' if delta_days > 0 else 'מקדים'}"
                            delta_color = "color: #ef4444;" if delta_days > 0 else "color: #10b981;"
                            delta_html = f'<div class="kpi-delta" style="{delta_color}">{delta_str}</div>'
                        
                        st.markdown(f"""
                        <div class="kpi-card planned">
                            <div class="kpi-icon">🎯</div>
                            <div class="kpi-label">צפי סיום פרויקט</div>
                            <div class="kpi-value">{date_str}</div>
                            {delta_html}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">🎯</div>
                            <div class="kpi-label">צפי סיום פרויקט</div>
                            <div class="kpi-value">טרם חושב</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # פיננסים - עם עיצוב מותאם אישית ואייקונים
                st.subheader("💰 מצב פיננסי")
                financial = get_project_financial_status(selected_plan_id)
                
                fin_col1, fin_col2, fin_col3 = st.columns(3)
                
                with fin_col1:
                    if financial["budget_limit"] > 0:
                        st.markdown(f"""
                        <div class="kpi-card planned">
                            <div class="kpi-icon">💰</div>
                            <div class="kpi-label">תקציב כולל</div>
                            <div class="kpi-value">{financial['budget_limit']:,.0f} ₪</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">💰</div>
                            <div class="kpi-label">תקציב כולל</div>
                            <div class="kpi-value">לא הוגדר</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with fin_col2:
                    st.markdown(f"""
                    <div class="kpi-card completed">
                        <div class="kpi-icon">💸</div>
                        <div class="kpi-label">עלות נוכחית</div>
                        <div class="kpi-value">{financial['current_cost']:,.0f} ₪</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with fin_col3:
                    if financial["budget_limit"] > 0:
                        variance = financial["budget_variance"]
                        variance_icon = "📉" if variance < 0 else "📊"
                        variance_color = "color: #ef4444;" if variance < 0 else "color: #10b981;"
                        variance_label = f"{abs(variance):,.0f} ₪ {'יתרה' if variance >= 0 else 'חריגה'}"
                        st.markdown(f"""
                        <div class="kpi-card {'remaining' if variance < 0 else 'completed'}">
                            <div class="kpi-icon">{variance_icon}</div>
                            <div class="kpi-label">יתרה/חריגה</div>
                            <div class="kpi-value" style="{variance_color}">{variance_label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">📊</div>
                            <div class="kpi-label">יתרה/חריגה</div>
                            <div class="kpi-value">לא ניתן לחשב</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # גרף התקדמות תקציבית
                if financial["budget_limit"] > 0:
                    progress_pct = (financial["current_cost"] / financial["budget_limit"]) * 100
                    progress_pct = min(progress_pct, 100.0)
                    st.progress(progress_pct / 100)
                    st.caption(f"נוצל {progress_pct:.1f}% מהתקציב")
                
                # רכש וחומרים - עם עיצוב משופר
                if plan.get("material_estimate"):
                    st.subheader("📦 הערכת חומרים")
                    try:
                        materials = json.loads(plan["material_estimate"])
                        mat_col1, mat_col2, mat_col3, mat_col4 = st.columns(4)
                        
                        with mat_col1:
                            st.markdown(f"""
                            <div class="material-card">
                                <div class="icon">🧱</div>
                                <div class="label">בלוקים</div>
                                <div class="value">{materials.get('block_count', 0):,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with mat_col2:
                            st.markdown(f"""
                            <div class="material-card">
                                <div class="icon">🪣</div>
                                <div class="label">מלט</div>
                                <div class="value">{materials.get('cement_cubic_meters', 0):.2f} מ"ק</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with mat_col3:
                            st.markdown(f"""
                            <div class="material-card">
                                <div class="icon">🏜️</div>
                                <div class="label">חול</div>
                                <div class="value">{materials.get('sand_cubic_meters', 0):.2f} מ"ק</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with mat_col4:
                            st.markdown(f"""
                            <div class="material-card">
                                <div class="icon">📐</div>
                                <div class="label">שטח קירות</div>
                                <div class="value">{materials.get('wall_area_sqm', 0):.1f} מ"ר</div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        pass
        
        # טבלת דיווחים
        st.subheader("📋 דיווחי ביצוע")
        df = load_stats_df()
        
        # חישוב היקף כולל של כל הקומות שנטענו (למקרה שלא בחר תוכנית ספציפית)
        total_planned = sum(p["total_length"] for p in st.session_state.projects.values()) if st.session_state.projects else 0.0
        total_done = df["מטרים שבוצעו"].sum() if not df.empty else 0.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("סה'כ מתוכנן (מטר רץ)", f"{total_planned:.1f}")
        c2.metric("סה'כ בוצע (מטר רץ)", f"{total_done:.1f}")
        c3.metric("נותר לביצוע", f"{max(0, total_planned - total_done):.1f}")
        
        if not df.empty:
            st.markdown("---")
            st.caption("התקדמות לפי קומות")
            st.bar_chart(df.groupby("שם תוכנית")["מטרים שבוצעו"].sum())
            st.dataframe(df.sort_values(by="תאריך", ascending=False), use_container_width=True)
        else:
            st.info("עדיין אין דיווחי ביצוע במערכת.")

# --- מצב עובד ---
elif mode == "דיווח ביצוע (Worker)":
    st.header("👷 דיווח ביצוע יומי")
    if not st.session_state.projects:
        st.warning("המנהל טרם העלה תוכניות.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", options=list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        # קבלת תמונות
        orig_img = proj["original"]
        thick_walls = proj["thick_walls"]
        
        # המרת תמונה מקורית ל-RGB
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_rgb.shape[:2]
        
        # וידוא ש-thick_walls באותו גודל כמו original
        if thick_walls.shape[0] != orig_h or thick_walls.shape[1] != orig_w:
            thick_walls = cv2.resize(thick_walls, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # המרת thick_walls למסכה בינארית
        if len(thick_walls.shape) == 2:
            wall_mask_binary = (thick_walls > 0).astype(np.uint8) * 255
        else:
            wall_mask_binary = (thick_walls.sum(axis=2) > 0).astype(np.uint8) * 255
        
        # הגדרת שקיפות - סליידר למשתמש
        if 'wall_opacity' not in st.session_state:
            st.session_state.wall_opacity = 0.3
        
        opacity_col1, opacity_col2 = st.columns([3, 1])
        with opacity_col1:
            wall_opacity = st.slider("שקיפות שכבה כחולה (קירות מתוכננים):", 0.0, 1.0, 
                                     st.session_state.wall_opacity, 0.1, 
                                     help="התאם את השקיפות כדי לראות טוב יותר את הקירות המתוכננים")
            st.session_state.wall_opacity = wall_opacity
        with opacity_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # יישור
            st.caption(f"שקיפות: {int(wall_opacity * 100)}%")
        
        # הרחבת הקירות כדי שיהיו נראים יותר (עם hitbox גדול יותר)
        # הגדלת ה-hitbox ל-15 פיקסלים במקום 9
        kernel = np.ones((15, 15), np.uint8)
        wall_mask_thick = cv2.dilate(wall_mask_binary, kernel, iterations=2)
        
        # יצירת overlay כחול
        blue_overlay = np.zeros_like(orig_rgb)
        blue_overlay[:, :, 0] = 0    # R
        blue_overlay[:, :, 1] = 150  # G
        blue_overlay[:, :, 2] = 255  # B (כחול)
        
        # הכפלת התמונה הכחולה במסכת הקירות
        blue_mask_3d = wall_mask_thick[:, :, np.newaxis] / 255.0
        blue_overlay = (blue_overlay * blue_mask_3d).astype(np.uint8)
        
        # מיזוג התמונה המקורית עם overlay כחול (עם שקיפות משתנה)
        orig_weight = 1.0 - wall_opacity
        blue_weight = wall_opacity
        combined = cv2.addWeighted(orig_rgb, orig_weight, blue_overlay, blue_weight, 0)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # התאמת גודל קנבס
        c_width = 1000
        c_height = int(c_width * (orig_h / orig_w))
        if c_height > 600:
            c_height = 600
            c_width = int(c_height / (orig_h / orig_w))
        
        # שינוי גודל לגודל הקנבס
        combined_res = cv2.resize(combined, (c_width, c_height), interpolation=cv2.INTER_AREA)
        wall_mask_thick_resized = cv2.resize(wall_mask_thick, (c_width, c_height), interpolation=cv2.INTER_NEAREST)
        
        st.info("💡 הקווים הכחולים מציגים את הקירות המתוכננים. סמן קווים ירוקים על הקירות שבנית.")
        
        canvas_result = st_canvas(
            stroke_width=4, 
            stroke_color="#00FF00", 
            background_image=Image.fromarray(combined_res),
            width=c_width, 
            height=c_height, 
            drawing_mode="line", 
            key=f"canvas_{plan_name}",
            update_streamlit=True
        )

        meters_today = 0.0
        warning_message = None
        debug_info = ""
        scale_x = 1.0
        scale_y = 1.0
        total_worker_pixels = 0
        total_wall_pixels = 0
        intersection_pixels_canvas = 0
        intersection_pixels_orig = 0
        
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            if not objects.empty:
                # דיבאג: הדפסת הקואורדינטות הראשונות כדי לראות את הפורמט
                first_obj = objects.iloc[0]
                debug_coords_info = []
                if 'x1' in first_obj:
                    debug_coords_info.append(f"x1={first_obj['x1']}, y1={first_obj['y1']}, x2={first_obj['x2']}, y2={first_obj['y2']}")
                if 'left' in first_obj:
                    debug_coords_info.append(f"left={first_obj['left']}, top={first_obj['top']}, width={first_obj.get('width', 'N/A')}, height={first_obj.get('height', 'N/A')}")
                if 'scaledLeft' in first_obj:
                    debug_coords_info.append(f"scaledLeft={first_obj['scaledLeft']}, scaledTop={first_obj['scaledTop']}")
                
                if debug_coords_info and 'debug_coords_printed' not in st.session_state:
                    st.session_state['debug_coords_printed'] = True
                    with st.expander("🔍 דיבאג קואורדינטות (רק פעם אחת)", expanded=True):
                        st.write("**כל השדות באובייקט הראשון:**")
                        st.json(dict(first_obj))
                        st.write("**קואורדינטות מזוהות:**")
                        for info in debug_coords_info:
                            st.text(info)
                        st.write(f"**גודל קנבס:** {c_width} x {c_height}")
                        st.write(f"**גודל תמונה מקורית:** {orig_w} x {orig_h}")
                
                # יצירת מסכת worker בגודל הקנבס
                worker_mask = np.zeros((c_height, c_width), dtype=np.uint8)
                
                # שלב 1: חישוב מטרים לפי אורך קווים (לא שטח!)
                # נדגום נקודות לאורך כל קו ונבדוק כמה מהן על קירות
                total_line_length_canvas = 0.0
                overlapping_line_length_canvas = 0.0
                all_lines = []
                
                # בדיקת פורמט הקואורדינטות - יכול להיות x1/y1, left/top/width/height, או path
                lines_drawn = 0
                for _, obj in objects.iterrows():
                    x1, y1, x2, y2 = None, None, None, None
                    
                    # טיפול בקואורדינטות - streamlit-drawable-canvas משתמש ב-left/top + x1/y1/x2/y2 יחסיים
                    # x1/y1/x2/y2 הם קואורדינטות יחסיות (ממורכזות), לא אבסולוטיות!
                    # צריך להשתמש ב-left/top + x1/y1/x2/y2 או ב-left/top + width/height
                    
                    # ניסיון ראשון: left/top + width/height (הקואורדינטות האמיתיות)
                    if 'left' in obj and 'top' in obj:
                        left_val = float(obj['left'])
                        top_val = float(obj['top'])
                        
                        # אם יש width/height, נחשב את הקואורדינטות מהן
                        if 'width' in obj and 'height' in obj:
                            width_val = float(obj['width'])
                            height_val = float(obj['height'])
                            
                            # left/top זה מרכז האובייקט, לא פינה
                            # צריך לחשב את הנקודות לפי x1/y1/x2/y2 היחסיים + left/top
                            if 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
                                # x1/y1/x2/y2 הם יחסיים למרכז (left/top)
                                x1_rel = float(obj['x1'])
                                y1_rel = float(obj['y1'])
                                x2_rel = float(obj['x2'])
                                y2_rel = float(obj['y2'])
                                
                                # המרה לקואורדינטות אבסולוטיות
                                x1 = int(left_val + x1_rel)
                                y1 = int(top_val + y1_rel)
                                x2 = int(left_val + x2_rel)
                                y2 = int(top_val + y2_rel)
                                
                                # דיבאג: הדפסת הקואורדינטות המחושבות (רק פעם אחת)
                                if lines_drawn == 0 and 'debug_coords_calc_printed' not in st.session_state:
                                    st.session_state['debug_coords_calc_printed'] = True
                                    st.info(f"🔍 **דיבאג קואורדינטות:** left={left_val}, top={top_val}, x1_rel={x1_rel}, y1_rel={y1_rel}, x2_rel={x2_rel}, y2_rel={y2_rel} → x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            else:
                                # אם אין x1/y1/x2/y2, נשתמש ב-width/height (אבל זה נדיר לקווים)
                                x1 = int(left_val - width_val / 2)
                                y1 = int(top_val - height_val / 2)
                                x2 = int(left_val + width_val / 2)
                                y2 = int(top_val + height_val / 2)
                        elif 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
                            # יש left/top + x1/y1/x2/y2 יחסיים
                            x1_rel = float(obj['x1'])
                            y1_rel = float(obj['y1'])
                            x2_rel = float(obj['x2'])
                            y2_rel = float(obj['y2'])
                            
                            # המרה לקואורדינטות אבסולוטיות
                            x1 = int(left_val + x1_rel)
                            y1 = int(top_val + y1_rel)
                            x2 = int(left_val + x2_rel)
                            y2 = int(top_val + y2_rel)
                        elif 'x2' in obj and 'y2' in obj:
                            # יש left/top + x2/y2
                            x1 = int(left_val)
                            y1 = int(top_val)
                            x2 = int(float(obj['x2']))
                            y2 = int(float(obj['y2']))
                        else:
                            continue
                    # fallback: ניסיון ל-x1/y1/x2/y2 בלבד (אם אין left/top)
                    elif 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
                        x1_raw, y1_raw = float(obj['x1']), float(obj['y1'])
                        x2_raw, y2_raw = float(obj['x2']), float(obj['y2'])
                        
                        # בדיקה אם הקואורדינטות שליליות או קטנות - אם כן, הן כנראה יחסיות
                        if x1_raw < 0 or y1_raw < 0 or abs(x1_raw) < 10 or abs(y1_raw) < 10:
                            # אלה כנראה קואורדינטות יחסיות - צריך left/top
                            continue  # נדלג על זה, אין לנו left/top
                        
                        # בדיקה אם הקואורדינטות גדולות מהקנבס - אם כן, הן כנראה ביחס לגודל המקורי
                        if x1_raw > c_width or y1_raw > c_height or x2_raw > c_width or y2_raw > c_height:
                            # המרה מגודל מקורי לקנבס
                            scale_x_coord = c_width / orig_w
                            scale_y_coord = c_height / orig_h
                            x1 = int(x1_raw * scale_x_coord)
                            y1 = int(y1_raw * scale_y_coord)
                            x2 = int(x2_raw * scale_x_coord)
                            y2 = int(y2_raw * scale_y_coord)
                        else:
                            # הקואורדינטות כבר בגודל קנבס
                            x1, y1 = int(x1_raw), int(y1_raw)
                            x2, y2 = int(x2_raw), int(y2_raw)
                    # ניסיון לקואורדינטות scaled (אמורות להיות ביחס לקנבס, אבל בואו נבדוק)
                    elif 'scaledLeft' in obj and 'scaledTop' in obj:
                        scaled_left_raw, scaled_top_raw = float(obj['scaledLeft']), float(obj['scaledTop'])
                        
                        # בדיקה אם הקואורדינטות גדולות מהקנבס
                        if scaled_left_raw > c_width or scaled_top_raw > c_height:
                            scale_x_coord = c_width / orig_w
                            scale_y_coord = c_height / orig_h
                            x1 = int(scaled_left_raw * scale_x_coord)
                            y1 = int(scaled_top_raw * scale_y_coord)
                            if 'scaledWidth' in obj and 'scaledHeight' in obj:
                                x2 = x1 + int(float(obj['scaledWidth']) * scale_x_coord)
                                y2 = y1 + int(float(obj['scaledHeight']) * scale_y_coord)
                            elif 'width' in obj and 'height' in obj:
                                x2 = x1 + int(float(obj['width']) * scale_x_coord)
                                y2 = y1 + int(float(obj['height']) * scale_y_coord)
                            else:
                                continue
                        else:
                            x1, y1 = int(scaled_left_raw), int(scaled_top_raw)
                            if 'scaledWidth' in obj and 'scaledHeight' in obj:
                                x2 = x1 + int(obj['scaledWidth'])
                                y2 = y1 + int(obj['scaledHeight'])
                            elif 'width' in obj and 'height' in obj:
                                x2 = x1 + int(obj['width'])
                                y2 = y1 + int(obj['height'])
                            else:
                                continue
                    # fallback - הדפסת שדות זמינים (רק פעם אחת)
                    else:
                        if lines_drawn == 0:  # רק בקו הראשון
                            st.warning(f"⚠️ פורמט קואורדינטות לא מוכר. שדות זמינים: {list(obj.keys())}")
                        continue
                    
                    if x1 is None or y1 is None or x2 is None or y2 is None:
                        continue
                    
                    # וידוא שהקואורדינטות בתוך גבולות הקנבס
                    x1, y1 = max(0, min(x1, c_width-1)), max(0, min(y1, c_height-1))
                    x2, y2 = max(0, min(x2, c_width-1)), max(0, min(y2, c_height-1))
                    
                    # ציור על המסכה (לדיבאג)
                    cv2.line(worker_mask, (x1, y1), (x2, y2), 255, thickness=4)
                    lines_drawn += 1
                    
                    # חישוב אורך הקו בפיקסלים (בגודל קנבס)
                    line_length_canvas = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    total_line_length_canvas += line_length_canvas
                
                # וידוא ש-wall_mask_thick_resized הוא מסכה בינארית (0 או 255) ובגודל נכון
                if wall_mask_thick_resized.shape[0] != c_height or wall_mask_thick_resized.shape[1] != c_width:
                    # אם הגודל לא נכון, נשנה אותו
                    wall_mask_thick_resized = cv2.resize(wall_mask_thick_resized, (c_width, c_height), interpolation=cv2.INTER_NEAREST)
                
                # המרה למסכה בינארית (0 או 255)
                if wall_mask_thick_resized.max() > 1 or wall_mask_thick_resized.dtype != np.uint8:
                    wall_mask_thick_resized = (wall_mask_thick_resized > 127).astype(np.uint8) * 255
                
                # חישוב אורך שעל קירות - דגימת נקודות לכל קו
                overlapping_line_length_canvas = 0.0
                for idx, obj in objects.iterrows():
                    x1, y1, x2, y2 = None, None, None, None
                    
                    # חישוב קואורדינטות (שוב, כדי לבדוק אורך)
                    if 'left' in obj and 'top' in obj:
                        left_val = float(obj['left'])
                        top_val = float(obj['top'])
                        
                        if 'width' in obj and 'height' in obj and 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
                            x1_rel = float(obj['x1'])
                            y1_rel = float(obj['y1'])
                            x2_rel = float(obj['x2'])
                            y2_rel = float(obj['y2'])
                            x1 = int(left_val + x1_rel)
                            y1 = int(top_val + y1_rel)
                            x2 = int(left_val + x2_rel)
                            y2 = int(top_val + y2_rel)
                        elif 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
                            x1_rel = float(obj['x1'])
                            y1_rel = float(obj['y1'])
                            x2_rel = float(obj['x2'])
                            y2_rel = float(obj['y2'])
                            x1 = int(left_val + x1_rel)
                            y1 = int(top_val + y1_rel)
                            x2 = int(left_val + x2_rel)
                            y2 = int(top_val + y2_rel)
                        else:
                            continue
                    
                    if x1 is None or y1 is None or x2 is None or y2 is None:
                        continue
                    
                    # וידוא שהקואורדינטות בתוך גבולות הקנבס
                    x1, y1 = max(0, min(x1, c_width-1)), max(0, min(y1, c_height-1))
                    x2, y2 = max(0, min(x2, c_width-1)), max(0, min(y2, c_height-1))
                    
                    # חישוב אורך הקו (בפיקסלים בקנבס)
                    dx = x2 - x1
                    dy = y2 - y1
                    line_length_canvas = np.sqrt(dx*dx + dy*dy)
                    
                    # דגימת נקודות לאורך הקו (יותר נקודות לאורך ארוך יותר)
                    num_samples = max(int(line_length_canvas), 20)  # לפחות 20 נקודות, או אחת לכל פיקסל
                    on_wall_count = 0
                    
                    for i in range(num_samples):
                        t = i / max(num_samples - 1, 1)  # 0 עד 1
                        px = int(x1 + t * dx)
                        py = int(y1 + t * dy)
                        px = max(0, min(px, c_width - 1))
                        py = max(0, min(py, c_height - 1))
                        
                        # בדיקה אם הנקודה על קיר
                        if wall_mask_thick_resized[py, px] > 0:
                            on_wall_count += 1
                    
                    # חישוב האורך שעל קירות (אחוז נקודות על קירות * אורך הקו)
                    if num_samples > 0:
                        overlap_ratio = on_wall_count / num_samples
                        overlapping_line_length_canvas += line_length_canvas * overlap_ratio
                
                # המרת אורך מגודל קנבס לגודל מקורי
                scale_x = orig_w / c_width
                scale_y = orig_h / c_height
                
                # עבור קו, scale משקף את ההמרה במימד אחד (אורך)
                # הקנבס שומר על יחס גובה-רוחב, אז נשתמש בממוצע גיאומטרי
                # (זה מתאים יותר לחישוב scale ליניארי)
                scale_factor_length = np.sqrt(scale_x * scale_y)
                
                # המרת אורך מגודל קנבס לגודל מקורי
                overlapping_line_length_orig = overlapping_line_length_canvas * scale_factor_length
                
                # חישוב מטרים (scale הוא פיקסלים למטר בתמונה המקורית)
                if proj["scale"] > 0:
                    meters_today = overlapping_line_length_orig / proj["scale"]
                else:
                    meters_today = 0.0
                
                # דיבאג: שמירת מידע גם על השטח (לצורך השוואה)
                # הרחבת worker_mask למרווח טעות (רק לדיבאג)
                # הגדלת ה-hitbox ל-15 פיקסלים במקום 9
                kernel_worker = np.ones((15, 15), np.uint8)
                worker_mask_dilated = cv2.dilate(worker_mask, kernel_worker, iterations=3)
                wall_mask_dilated = cv2.dilate(wall_mask_thick_resized, kernel_worker, iterations=2)
                intersection = cv2.bitwise_and(worker_mask_dilated, wall_mask_dilated)
                intersection_pixels_canvas = cv2.countNonZero(intersection)
                total_worker_pixels = cv2.countNonZero(worker_mask_dilated)
                total_wall_pixels = cv2.countNonZero(wall_mask_dilated)
                
                scale_factor_area = scale_x * scale_y
                intersection_pixels_orig = int(intersection_pixels_canvas * scale_factor_area)
                
                # יצירת debug info (עם כל המידע כולל גדלים)
                scale_factor_display = scale_x * scale_y
                worker_shape = worker_mask.shape
                wall_shape = wall_mask_thick_resized.shape
                overlap_pct = (intersection_pixels_canvas / total_worker_pixels * 100) if total_worker_pixels > 0 else 0
                
                debug_info = f"[Debug: worker={total_worker_pixels} (shape={worker_shape}), walls={total_wall_pixels} (shape={wall_shape}), intersection={intersection_pixels_canvas} ({overlap_pct:.1f}% overlap), intersection_orig={intersection_pixels_orig}, line_length_canvas={overlapping_line_length_canvas:.1f}px, line_length_orig={overlapping_line_length_orig:.1f}px, scale_factor_length={scale_factor_length:.3f}, calibration={proj['scale']:.1f}px/m]"
                
                # יצירת תמונה ויזואלית לדיבאג אם אין intersection
                if intersection_pixels_canvas == 0 and total_worker_pixels > 0 and total_wall_pixels > 0:
                    # יצירת תמונת דיבאג שמציגה worker (ירוק), walls (כחול), intersection (אדום)
                    debug_img = np.zeros((c_height, c_width, 3), dtype=np.uint8)
                    debug_img[:, :, 1] = worker_mask  # ירוק
                    debug_img[:, :, 2] = wall_mask_thick_resized  # כחול
                    # intersection יהיה אדום (אם יש)
                    intersection_vis = cv2.bitwise_and(worker_mask, wall_mask_thick_resized)
                    debug_img[:, :, 0] = intersection_vis  # אדום
                    
                    # הוספת התמונה לדיבאג (נציג אותה למטה)
                    st.session_state['debug_vis'] = debug_img
                
                # בדיקת אזהרות
                if meters_today == 0.0 and total_worker_pixels > 0:
                    if total_wall_pixels == 0:
                        warning_message = "⚠️ לא זוהו קירות בתמונה. אנא בדוק את הניתוח במסך המנהל."
                    else:
                        warning_message = f"⚠️ הקווים שסומנו אינם חופפים לקירות המתוכננים. {debug_info}"
                elif total_worker_pixels > 0:
                    overlap_percent = (intersection_pixels_canvas / total_worker_pixels) * 100 if total_worker_pixels > 0 else 0
                    if overlap_percent < 50:
                        warning_message = f"⚠️ רק {overlap_percent:.0f}% מהסימון חופף לקירות. הקפד לסמן על הקירות הכחולים."
        
        # הצגת תוצאות
    col1, col2 = st.columns([2, 1])

    with col1:
            st.metric("נמדד לדיווח זה:", f"{meters_today:.2f} מטר")
            
            # הודעת עידוד בזמן אמת
            if meters_today > 0:
                st.markdown(f"""
                <div class="success-message">
                    ✅ סימנת בהצלחה {meters_today:.2f} מטרים!
                </div>
                """, unsafe_allow_html=True)
            
            # הצגת debug info תמיד (בגרסה מפורטת) אם יש קווים
            if canvas_result.json_data is not None and pd.json_normalize(canvas_result.json_data.get("objects", [])).shape[0] > 0:
                with st.expander("🔍 פרטי דיבאג", expanded=(meters_today == 0)):
                    st.text(debug_info if debug_info else "אין מידע דיבאג זמין")
                    st.text(f"גודל תמונה מקורית: {orig_w} x {orig_h}")
                    st.text(f"גודל קנבס: {c_width} x {c_height}")
                    st.text(f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
                    st.text(f"פיקסלי worker: {total_worker_pixels}")
                    st.text(f"פיקסלי walls: {total_wall_pixels}")
                    st.text(f"פיקסלי intersection (קנבס): {intersection_pixels_canvas}")
                    st.text(f"פיקסלי intersection (מקורי): {intersection_pixels_orig}")
                    st.text(f"**אורך קווים:**")
                    st.text(f"  - אורך בקנבס: {overlapping_line_length_canvas:.1f} פיקסלים")
                    st.text(f"  - אורך במקורי: {overlapping_line_length_orig:.1f} פיקסלים")
                    st.text(f"  - scale_factor_length: {scale_factor_length:.3f}")
                    st.text(f"כיול (פיקסלים למטר): {proj.get('scale', 'לא מוגדר')}")
                    st.text(f"**מטרים מחושבים (מאורך, לא מפיקסלים!): {meters_today:.2f}**")
                    
                    # הצגת תמונה ויזואלית לדיבאג אם אין intersection
                    if intersection_pixels_canvas == 0 and total_worker_pixels > 0 and total_wall_pixels > 0:
                        st.warning("🔍 **דיבאג ויזואלי:** למטה תראה תמונה שמציגה ירוק=worker, כחול=walls. אם אין אדום, זה אומר שאין חיתוך.")
                        if 'debug_vis' in st.session_state:
                            st.image(st.session_state['debug_vis'], caption="ירוק=קווים שציירת, כחול=קירות מזוהים, אדום=חיתוך", use_container_width=True)
            
            if warning_message:
                st.warning(warning_message)
            
            # הצגת התקדמות אם יש
            if proj.get("total_length", 0) > 0:
                progress = (meters_today / proj["total_length"]) * 100 if proj["total_length"] > 0 else 0
                progress = min(progress, 100.0)
                st.progress(progress / 100)
                st.caption(f"התקדמות: {progress:.1f}% מתוך {proj['total_length']:.1f} מטר מתוכנן")

    with col2:
            note = st.text_input("הערה (אופציונלי):", placeholder="למשל: קיר מסדרון צפוני")
            
            if st.button("✅ שלח דיווח סופי", type="primary", use_container_width=True):
                if meters_today > 0:
                    # בדיקה אם התוכנית קיימת ב-DB, אם לא - שומרים אותה עכשיו
                    plan_rec = get_plan_by_filename(plan_name)
                    if not plan_rec:
                        from database import save_plan
                        plan_id = save_plan(
                            filename=plan_name, 
                            plan_name=proj["metadata"].get("plan_name", plan_name.replace(".pdf", "")),
                            extracted_scale=proj["metadata"].get("scale"),
                            raw_pixel_count=proj["raw_pixels"],
                            metadata_json=json.dumps(proj["metadata"], ensure_ascii=False),
                            confirmed_scale=proj["scale"]
                        )
                    else:
                        plan_id = plan_rec['id']
                    
                    save_progress_report(plan_id, meters_today, note)
                    st.success(f"✅ דיווח על {meters_today:.2f} מטר נשלח בהצלחה!")
                    st.balloons()
else:
                    st.error("⚠️ לא סומנו קירות על גבי קירות מתוכננים. אנא סמן קווים ירוקים על הקירות הכחולים בלבד.")