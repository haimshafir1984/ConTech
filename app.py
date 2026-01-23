"""
ConTech Pro v2.0 - מערכת ניהול בנייה מקצועית
ארכיטקטורה מודולרית משופרת
"""

import streamlit as st
from PIL import Image

# ייבוא סגנונות ואתחול
from styles import apply_all_styles
from database import init_database, reset_all_data

# ייבוא דפים
from pages.manager import (
    render_workshop_tab,
    render_corrections_tab,
    render_dashboard_tab,
    render_invoices_tab,
    render_plan_data_tab,
    render_floor_analysis_tab  # ← הוסף את זה
)
from pages.worker import render_worker_page

# --- אתחול המערכת ---
apply_all_styles()
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- Session State ---
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "wall_height" not in st.session_state:
    st.session_state.wall_height = 2.5
if "default_cost_per_meter" not in st.session_state:
    st.session_state.default_cost_per_meter = 0.0
if "manual_corrections" not in st.session_state:
    st.session_state.manual_corrections = {}

# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro v2.0")
    st.caption("✨ Multi-pass Detection + Manual Corrections")
    mode = st.radio(
        "ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed"
    )
    st.markdown("---")

    with st.expander("⚙️ הגדרות גלובליות"):
        st.session_state.wall_height = st.number_input(
            "גובה קירות (מ')",
            value=st.session_state.wall_height,
            step=0.1,
            key="global_wall_height",
        )
        st.session_state.default_cost_per_meter = st.number_input(
            "עלות למטר (₪)",
            value=st.session_state.default_cost_per_meter,
            step=10.0,
            key="global_cost_per_meter",
        )

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 סדנת עבודה", 
    "🎨 תיקונים ידניים", 
    "📄 נתונים מהשרטוט",
    "📐 ניתוח שטחים",  # ← טאב חדש
    "📊 דשבורד", 
    "💰 חשבונות"
])

    with tab1:
    render_workshop_tab()

    with tab2:
    render_corrections_tab()

    with tab3:
    render_plan_data_tab()

    with tab4:
    render_floor_analysis_tab()  # ← הוסף את זה

    with tab5:
    render_dashboard_tab()

    with tab6:
    render_invoices_tab()

# סיום בלוק הטאבים - חזרה לרמה הראשית

# ==========================================
# 👷 מצב דיווח
# ==========================================
elif mode == "👷 דיווח שטח":
    render_worker_page()
