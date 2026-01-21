"""
ConTech Pro - Invoices Page
מחולל חשבונות חלקיים לקבלנים
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from database import (
    get_all_plans, get_progress_reports,
    get_all_work_types_for_plan,
    get_progress_summary_by_date_range,
    get_payment_invoice_data
)
from reporter import generate_payment_invoice_pdf


def render_invoices():
    """רנדור מחולל חשבונות"""
    st.markdown("## 💰 מחולל חשבונות חלקיים")
    st.caption("הפקת חשבונית לתשלום לקבלן על בסיס ביצוע בפועל")
    
    all_plans = get_all_plans()
    if not all_plans:
        st.info("אין פרויקטים במערכת")
        return
    
    # בחירת פרויקט
    plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
    selected_plan_invoice = st.selectbox("בחר פרויקט:", plan_options, key="invoice_plan_select")
    plan_id = int(selected_plan_invoice.split("ID: ")[1].strip(")"))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📅 בחר טווח תאריכים")
        
        # טווח מהיר
        quick_range = st.radio(
            "בחירה מהירה:",
            ["שבוע אחרון", "חודש אחרון", "טווח מותאם אישית"],
            horizontal=True
        )
        
        if quick_range == "שבוע אחרון":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
        elif quick_range == "חודש אחרון":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        else:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "מתאריך:",
                    value=datetime.now() - timedelta(days=30),
                    key="start_date_picker"
                )
            with col_date2:
                end_date = st.date_input(
                    "עד תאריך:",
                    value=datetime.now(),
                    key="end_date_picker"
                )
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        st.info(f"📊 תקופת החשבון: {start_str} עד {end_str}")
        
        # הגדרת מחירי יחידה
        st.markdown("### 💵 מחירי יחידה")
        
        work_types = get_all_work_types_for_plan(plan_id)
        
        if not work_types:
            st.warning("אין דיווחים לפרויקט זה עדיין")
        else:
            st.caption("ערוך את המחירים לפי הצורך")
            
            unit_prices = {}
            
            for work_type in work_types:
                if 'ריצוף' in work_type.lower() or 'חיפוי' in work_type.lower():
                    default_price = 250
                    unit = 'מ"ר'
                elif 'בטון' in work_type.lower():
                    default_price = 1200
                    unit = "מ'"
                elif 'בלוק' in work_type.lower():
                    default_price = 600
                    unit = "מ'"
                else:
                    default_price = 800
                    unit = "מ'"
                
                col_type, col_price = st.columns([2, 1])
                with col_type:
                    st.markdown(f"**{work_type}** ({unit})")
                with col_price:
                    price = st.number_input(
                        "מחיר:",
                        value=float(default_price),
                        step=50.0,
                        key=f"price_{work_type}",
                        label_visibility="collapsed"
                    )
                    unit_prices[work_type] = price
    
    with col2:
        st.markdown("### 👷 פרטי קבלן")
        st.caption("שדות אלה יופיעו בחשבונית")
        
        contractor_name = st.text_input(
            "שם הקבלן:",
            value="",
            placeholder="ישראל ישראלי",
            key="contractor_name"
        )
        
        contractor_company = st.text_input(
            "שם חברה:",
            value="",
            placeholder='בניית ישראל בע"מ',
            key="contractor_company"
        )
        
        contractor_vat = st.text_input(
            "ח.פ / ע.מ:",
            value="",
            placeholder="123456789",
            key="contractor_vat"
        )
        
        contractor_address = st.text_area(
            "כתובת:",
            value="",
            placeholder="רחוב הבניינים 1, תל אביב",
            height=80,
            key="contractor_address"
        )
        
        st.markdown("---")
        
        # כפתור יצירת חשבונית
        if st.button("🧾 צור חשבונית", type="primary", use_container_width=True):
            if not contractor_name or not contractor_vat:
                st.error("❌ יש למלא שם קבלן ומספר עוסק")
            else:
                with st.spinner("מכין חשבונית..."):
                    try:
                        invoice_data = get_payment_invoice_data(
                            plan_id,
                            start_str,
                            end_str,
                            unit_prices
                        )
                        
                        if invoice_data.get('error'):
                            st.error(f"❌ {invoice_data['error']}")
                        elif not invoice_data['items']:
                            st.warning("⚠️ אין דיווחים בטווח התאריכים הזה")
                        else:
                            contractor_info = {
                                'name': contractor_name,
                                'company': contractor_company,
                                'vat_id': contractor_vat,
                                'address': contractor_address
                            }
                            
                            pdf_buffer = generate_payment_invoice_pdf(
                                invoice_data,
                                contractor_info
                            )
                            
                            st.success("✅ החשבונית הוכנה בהצלחה!")
                            
                            st.markdown("### 📋 סיכום החשבונית")
                            
                            df_items = pd.DataFrame([
                                {
                                    'סוג עבודה': item['work_type'],
                                    'כמות': f"{item['quantity']:.2f}",
                                    'יחידה': item['unit'],
                                    'מחיר יחידה': f"{item['unit_price']:,.0f} ₪",
                                    'סה"כ': f"{item['subtotal']:,.2f} ₪"
                                }
                                for item in invoice_data['items']
                            ])
                            
                            st.dataframe(df_items, use_container_width=True, hide_index=True)
                            
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                st.metric("סכום ביניים", f"{invoice_data['total_amount']:,.2f} ₪")
                            with col_sum2:
                                st.metric('מע"מ (17%)', f"{invoice_data['vat']:,.2f} ₪")
                            with col_sum3:
                                st.metric("**סה\"כ לתשלום**", f"{invoice_data['total_with_vat']:,.2f} ₪")
                            
                            st.download_button(
                                label="📥 הורד חשבונית (PDF)",
                                data=pdf_buffer,
                                file_name=f"invoice_{invoice_data['plan']['plan_name']}_{start_str}_{end_str}.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ שגיאה ביצירת חשבונית: {str(e)}")
                        import traceback
                        with st.expander("פרטי שגיאה"):
                            st.code(traceback.format_exc())
    
    # תצוגה מקדימה של דיווחים
    st.markdown("---")
    with st.expander("📊 דיווחים בטווח התאריכים"):
        summary = get_progress_summary_by_date_range(plan_id, start_str, end_str)
        if summary:
            df_summary = pd.DataFrame([
                {
                    'סוג עבודה': item['work_type'],
                    'כמות כוללת': f"{item['total_quantity']:.2f}",
                    'יחידה': item['unit'],
                    'מספר דיווחים': item['report_count']
                }
                for item in summary
            ])
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            st.info("אין דיווחים בטווח זה")
