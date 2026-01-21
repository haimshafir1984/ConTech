"""
ConTech Pro - Dashboard Page
דשבורד ניהולי עם KPIs וגרפים
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import cv2
import numpy as np

from database import (
    get_all_plans, get_plan_by_id,
    get_project_forecast, get_project_financial_status,
    get_progress_reports
)
from reporter import generate_status_pdf
from utils import load_stats_df


def render_dashboard():
    """רנדור דשבורד מלא"""
    st.header("📊 דשבורד פרויקט")
    
    all_plans = get_all_plans()
    if not all_plans:
        st.info("🔍 אין פרויקטים במערכת. העלה תוכנית בסדנת עבודה.")
        return
    
    # בחירת פרויקט
    plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
    selected_plan_dash = st.selectbox("📂 בחר פרויקט:", plan_options, key="dashboard_plan_select")
    plan_id = int(selected_plan_dash.split("ID: ")[1].strip(")"))
    
    # טעינת נתונים
    forecast = get_project_forecast(plan_id)
    financial = get_project_financial_status(plan_id)
    plan_data = get_plan_by_id(plan_id)
    
    # === KPIs ===
    st.markdown("### 📈 מדדי ביצוע")
    
    k1, k2, k3, k4 = st.columns(4)
    
    total = forecast.get('total_planned', 0)
    built = forecast.get('cumulative_progress', 0)
    percent = (built / total * 100) if total > 0 else 0
    remaining = total - built
    
    with k1:
        st.metric(
            label="📏 סך הכל מתוכנן",
            value=f"{total:.1f} מ'",
            help="סך כל הקירות שזוהו בתוכנית"
        )
    
    with k2:
        st.metric(
            label="✅ בוצע בפועל",
            value=f"{built:.1f} מ'",
            delta=f"{percent:.1f}%",
            delta_color="normal",
            help="סך כל הדיווחים מצטבר"
        )
    
    with k3:
        st.metric(
            label="⏳ נותר לביצוע",
            value=f"{remaining:.1f} מ'",
            delta=f"{forecast.get('days_to_finish', 0)} ימים",
            delta_color="inverse",
            help="תחזית עפ\"י קצב ביצוע נוכחי"
        )
    
    with k4:
        budget = financial.get('budget_limit', 0)
        cost = financial.get('current_cost', 0)
        variance = budget - cost
        st.metric(
            label="💰 עלות מצטברת",
            value=f"{cost:,.0f} ₪",
            delta=f"{variance:,.0f} ₪ {'תקציב' if variance >= 0 else 'חריגה'}",
            delta_color="normal" if variance >= 0 else "inverse",
            help=f"תקציב: {budget:,.0f} ₪"
        )
    
    # === Progress Bar ===
    st.markdown("---")
    st.markdown("### 📊 התקדמות כללית")
    
    if percent < 30:
        color = "#EF4444"
    elif percent < 70:
        color = "#F59E0B"
    else:
        color = "#10B981"
    
    progress_html = f"""
    <div style="margin: 1.5rem 0;">
        <div style="width: 100%; background: #e5e7eb; border-radius: 12px; height: 40px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="
                width: {percent}%; 
                background: linear-gradient(90deg, {color}, {color}dd); 
                height: 100%; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                color: white; 
                font-weight: bold; 
                font-size: 18px; 
                transition: width 0.5s ease;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
            ">
                {percent:.1f}%
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.875rem; color: #6b7280; font-weight: 500;">
            <span>🚀 התחלה</span>
            <span>📍 {built:.1f} מ' מתוך {total:.1f} מ'</span>
            <span>🎯 סיום</span>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # === גרף התקדמות ===
    st.markdown("---")
    st.markdown("### 📈 גרף התקדמות לאורך זמן")
    
    df_stats = load_stats_df()
    if not df_stats.empty:
        df_current = df_stats[df_stats['שם תוכנית'] == plan_data['plan_name']]
        
        if not df_current.empty:
            st.bar_chart(df_current, x="תאריך", y="כמות שבוצעה", use_container_width=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📋 מספר דיווחים", len(df_current))
            with col_b:
                avg_daily = df_current['כמות שבוצעה'].mean()
                st.metric("📊 ממוצע יומי", f"{avg_daily:.1f} מ'")
            with col_c:
                max_day = df_current['כמות שבוצעה'].max()
                st.metric("⭐ יום שיא", f"{max_day:.1f} מ'")
        else:
            st.info("📭 אין דיווחים לפרויקט זה עדיין")
    else:
        st.info("📭 אין דיווחים במערכת")
    
    # === כפתורי פעולה ===
    st.markdown("---")
    st.markdown("### 🎯 פעולות ודוחות")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 הדפס דוח PDF", use_container_width=True, type="primary", key="pdf_button_dash"):
            with st.spinner("🔄 מכין דוח מפורט..."):
                try:
                    # ניסיון למצוא תמונה
                    if st.session_state.projects:
                        first_proj = list(st.session_state.projects.values())[0]
                        rgb = cv2.cvtColor(first_proj['original'], cv2.COLOR_BGR2RGB)
                    else:
                        rgb = np.ones((800, 1200, 3), dtype=np.uint8) * 255
                        cv2.putText(rgb, "Image Not Available", (350, 400), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 3)
                    
                    stats = {
                        'built': built,
                        'total': total,
                        'percent': percent,
                        'remaining': remaining,
                        'cost': cost,
                        'budget': budget
                    }
                    
                    pdf_buffer = generate_status_pdf(plan_data['plan_name'], rgb, stats)
                    
                    st.download_button(
                        label="⬇️ הורד דוח PDF",
                        data=pdf_buffer,
                        file_name=f"status_report_{plan_data['plan_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="secondary",
                        key="download_pdf_dash"
                    )
                    st.success("✅ הדוח מוכן להורדה!")
                
                except Exception as e:
                    st.error(f"❌ שגיאה ביצירת דוח: {str(e)}")
    
    with col2:
        if st.button("📊 ייצא נתונים", use_container_width=True, key="export_button_dash"):
            st.info("💡 תכונה בפיתוח: ייצוא ל-Excel")
    
    with col3:
        if st.button("📧 שלח דוא\"ל", use_container_width=True, key="email_button_dash"):
            st.info("💡 תכונה בפיתוח: שליחת דוח באימייל")
    
    # === טבלת דיווחים ===
    st.markdown("---")
    st.markdown("### 📋 דיווחים אחרונים")
    
    reports = get_progress_reports(plan_id)
    if reports:
        recent = reports[:5]
        
        for i, r in enumerate(recent, 1):
            meters = r['meters_built']
            if meters > 20:
                icon = "🟢"
            elif meters > 10:
                icon = "🟡"
            else:
                icon = "🔴"
            
            with st.expander(f"{icon} {r['date']} - {meters:.1f} מ' - {r.get('note', 'אין הערה')}"):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**📏 כמות:** {meters:.1f} מטרים")
                    if r.get('note'):
                        st.write(f"**📝 הערה:** {r['note']}")
                    st.caption(f"תאריך: {r['date']}")
                with col_b:
                    st.metric("דיווח #", i)
                    st.caption(f"ID: {r['id']}")
        
        total_reports = len(reports)
        if total_reports > 5:
            st.caption(f"📌 מציג 5 מתוך {total_reports} דיווחים")
    else:
        st.info("📭 אין דיווחים לפרויקט זה. התחל לדווח בסדנת עבודה!")
