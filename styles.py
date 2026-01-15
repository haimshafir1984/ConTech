import streamlit as st

def setup_page():
    st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")

def apply_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
        
        /* הגדרות בסיס */
        html, body, [class*="css"] { font-family: 'Heebo', sans-serif; direction: rtl; }
        :root { --primary-blue: #0F62FE; --bg-gray: #F4F7F6; --card-border: #E0E0E0; }
        
        /* כרטיסיות */
        .stCard { 
            background-color: white; 
            padding: 24px; 
            border-radius: 12px; 
            border: 1px solid var(--card-border); 
            box-shadow: 0 2px 8px rgba(0,0,0,0.04); 
            margin-bottom: 20px; 
        }
        
        /* KPI Dashboard */
        .kpi-container { 
            display: flex; 
            flex-direction: column; 
            background: white; 
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid #EAEAEA; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); 
            height: 100%; 
            text-align: center;
        }
        .kpi-icon { font-size: 24px; margin-bottom: 10px; }
        .kpi-label { font-size: 14px; color: #666; }
        .kpi-value { font-size: 28px; font-weight: 800; color: #0F62FE; margin: 5px 0; }
        .kpi-sub { font-size: 12px; color: #999; }
        
        /* כרטיסי חומרים */
        .mat-card { text-align: center; background: white; border: 1px solid #EEE; border-radius: 10px; padding: 15px; }
        .mat-val { font-size: 20px; font-weight: bold; color: #0F62FE; }
        .mat-lbl { font-size: 14px; color: #666; }
        
        /* תיבת מחיר */
        .price-box { 
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            border-right: 4px solid #0F62FE; 
            margin-bottom: 10px; 
        }
        
        /* התאמות כלליות */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label { 
            text-align: right !important; 
            width: 100%; 
            direction: rtl; 
        }
        .stButton button { border-radius: 8px; font-weight: 500; height: 45px; }
        section[data-testid="stSidebar"] { background-color: #FAFAFA; border-left: 1px solid #EEE; }
    </style>
    """, unsafe_allow_html=True)