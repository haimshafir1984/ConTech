"""
ConTech Pro v2.0 - מערכת ניהול בנייה מקצועית
ארכיטקטורה מודולרית משופרת
"""

import streamlit as st
from PIL import Image

# ייבוא סגנונות ואתחול
from styles import apply_all_styles
from database import init_database, reset_all_data
from db_monitor import show_db_widget_sidebar

# --- אתחול המערכת ---
apply_all_styles()
Image.MAX_IMAGE_PIXELS = 250_000_000
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
        "ניווט",
        [
            "🏢 מנהל",
            "📂 נתונים",
            "💰 הנהלת חשבונות",
            "👷 דיווח שטח",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ווידג'ט DB בסיידבר
    show_db_widget_sidebar()

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


# ==========================================================
# 🏢 מצב: מנהל (סדנת עבודה + תכולה + דשבורד)
# ==========================================================
if mode == "🏢 מנהל":
    from pages.manager import (
        render_workshop_tab,
        render_dashboard_tab,
    )
    from pages.manager_planning import render_manager_planning_tab

    st.title("🏢 מנהל פרויקט")

    tab1, tab2, tab3 = st.tabs(
        [
            "📂 סדנת עבודה",
            "🧱 הגדרת תכולה",
            "📊 דשבורד",
        ]
    )

    with tab1:
        render_workshop_tab()

    with tab2:
        render_manager_planning_tab()

    with tab3:
        render_dashboard_tab()


# ==========================================================
# 📂 מצב: נתונים (תיקונים + נתונים מהשרטוט + ניתוח שטחים)
# ==========================================================
elif mode == "📂 נתונים":
    from pages.manager import (
        render_corrections_tab,
        render_plan_data_tab,
        render_floor_analysis_tab,
    )

    st.title("📂 נתונים")

    tab1, tab2, tab3 = st.tabs(
        [
            "🎨 תיקונים ידניים",
            "📄 נתונים מהשרטוט",
            "📐 ניתוח שטחים",
        ]
    )

    with tab1:
        render_corrections_tab()

    with tab2:
        render_plan_data_tab()

    with tab3:
        render_floor_analysis_tab()


# ==========================================================
# 💰 מצב: הנהלת חשבונות (חשבונות)
# ==========================================================
elif mode == "💰 הנהלת חשבונות":
    from pages.manager import render_invoices_tab

    st.title("💰 הנהלת חשבונות")

    tab1 = st.tabs(["💰 חשבונות"])[0]
    with tab1:
        render_invoices_tab()


# ==========================================================
# 👷 מצב: דיווח שטח (נשאר זהה)
# ==========================================================
elif mode == "👷 דיווח שטח":
    from pages.worker import render_worker_page

    render_worker_page()
