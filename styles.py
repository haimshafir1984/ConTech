"""
ConTech Pro - מערכת עיצוב משולבת
קובץ styles.py עם הקוד הישן + רכיבים חדשים
"""

import streamlit as st
from datetime import datetime

# ==========================================
# פונקציות ישנות (מהקובץ המקורי) ✅
# ==========================================

def setup_page():
    """הגדרת עמוד - מהקוד המקורי"""
    st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")


def apply_css():
    """CSS מקורי - נשאר בדיוק כמו שהיה"""
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


# ==========================================
# צבעים חדשים (תוספת) 🆕
# ==========================================

COLORS = {
    # Primary (כחול - משתמש בצבע הקיים)
    'primary': '#0F62FE',
    'primary_dark': '#0043CE',
    'primary_light': '#4589FF',
    
    # Secondary (ירוק הצלחה)
    'secondary': '#10B981',
    'secondary_dark': '#059669',
    'secondary_light': '#34D399',
    
    # Status
    'success': '#10B981',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'info': '#3B82F6',
    
    # Neutrals
    'gray_50': '#F9FAFB',
    'gray_100': '#F3F4F6',
    'gray_200': '#E5E7EB',
    'gray_300': '#D1D5DB',
    'gray_500': '#6B7280',
    'gray_700': '#374151',
    'gray_900': '#111827',
    
    # Text
    'text_primary': '#111827',
    'text_secondary': '#6B7280',
}


# ==========================================
# רכיבים חדשים (תוספת) 🆕
# ==========================================

def render_header_new(user_name="מנהל פרויקט", show_date=True):
    """Header חדש מקצועי - אופציונלי!"""
    
    date_str = datetime.now().strftime('%d/%m/%Y') if show_date else ""
    
    st.markdown(f"""
        <style>
        .new-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .new-header-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .new-logo {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .new-logo-icon {{
            font-size: 2.5rem;
        }}
        .new-logo-text h1 {{
            color: white;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }}
        .new-logo-text p {{
            color: rgba(255,255,255,0.9);
            margin: 0;
            font-size: 0.875rem;
        }}
        .new-user-section {{
            color: white;
            text-align: right;
        }}
        
        @media (max-width: 768px) {{
            .new-header {{
                padding: 1rem;
            }}
            .new-header-content {{
                flex-direction: column;
                gap: 1rem;
            }}
            .new-logo-text h1 {{
                font-size: 1.5rem;
            }}
        }}
        </style>
        
        <div class="new-header">
            <div class="new-header-content">
                <div class="new-logo">
                    <div class="new-logo-icon">🏗️</div>
                    <div class="new-logo-text">
                        <h1>ConTech Pro</h1>
                        <p>מערכת ניהול פרויקטים חכמה לבנייה</p>
                    </div>
                </div>
                <div class="new-user-section">
                    <div>👤 {user_name}</div>
                    {f'<div style="font-size: 0.875rem; opacity: 0.9;">📅 {date_str}</div>' if show_date else ''}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_alert(message, variant="info"):
    """התראה מעוצבת - חדש! 🆕"""
    
    variants = {
        'success': {'bg': '#F0FDF4', 'border': COLORS['success'], 'icon': '✅', 'text': '#065F46'},
        'warning': {'bg': '#FFFBEB', 'border': COLORS['warning'], 'icon': '⚠️', 'text': '#92400E'},
        'error': {'bg': '#FEF2F2', 'border': COLORS['error'], 'icon': '❌', 'text': '#991B1B'},
        'info': {'bg': '#EFF6FF', 'border': COLORS['info'], 'icon': 'ℹ️', 'text': '#1E40AF'}
    }
    
    style = variants[variant]
    
    st.markdown(f"""
        <style>
        .alert {{
            background: {style['bg']};
            border-right: 4px solid {style['border']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .alert-icon {{
            font-size: 1.5rem;
            flex-shrink: 0;
        }}
        .alert-message {{
            color: {style['text']};
            font-weight: 500;
            flex: 1;
        }}
        </style>
        
        <div class="alert">
            <div class="alert-icon">{style['icon']}</div>
            <div class="alert-message">{message}</div>
        </div>
    """, unsafe_allow_html=True)


def render_progress(value, max_value=100, label="", show_percentage=True):
    """Progress bar משופר - חדש! 🆕"""
    
    percentage = (value / max_value) * 100
    
    # בחירת צבע לפי אחוז
    if percentage < 30:
        color = COLORS['error']
    elif percentage < 70:
        color = COLORS['warning']
    else:
        color = COLORS['success']
    
    st.markdown(f"""
        <style>
        .progress-container {{
            width: 100%;
            margin: 1rem 0;
        }}
        .progress-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            font-weight: 500;
        }}
        .progress-bar {{
            width: 100%;
            height: 24px;
            background: {COLORS['gray_200']};
            border-radius: 12px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
            width: {percentage}%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-left: 0.5rem;
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        </style>
        
        <div class="progress-container">
            {f'<div class="progress-label"><span>{label}</span><span>{percentage:.1f}%</span></div>' if label or show_percentage else ''}
            <div class="progress-bar">
                <div class="progress-fill">
                    {f'{percentage:.0f}%' if show_percentage and percentage > 15 else ''}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_stat_card_new(title, value, change=None, icon="", trend="neutral"):
    """KPI Card משופר - חדש! 🆕"""
    
    trend_colors = {
        'up': {'color': COLORS['success'], 'icon': '↗'},
        'down': {'color': COLORS['error'], 'icon': '↘'},
        'neutral': {'color': COLORS['text_secondary'], 'icon': '→'}
    }
    
    trend_style = trend_colors[trend]
    
    st.markdown(f"""
        <style>
        .stat-card-new {{
            background: white;
            border: 1px solid {COLORS['gray_200']};
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s;
            height: 100%;
        }}
        .stat-card-new:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .stat-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .stat-title {{
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            font-weight: 500;
        }}
        .stat-icon-new {{
            font-size: 1.5rem;
        }}
        .stat-value-new {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS['primary']};
            margin-bottom: 0.5rem;
        }}
        .stat-change {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.875rem;
            color: {trend_style['color']};
            font-weight: 500;
        }}
        </style>
        
        <div class="stat-card-new">
            <div class="stat-header">
                <div class="stat-title">{title}</div>
                {f'<div class="stat-icon-new">{icon}</div>' if icon else ''}
            </div>
            <div class="stat-value-new">{value}</div>
            {f'<div class="stat-change">{trend_style["icon"]} {change}</div>' if change else ''}
        </div>
    """, unsafe_allow_html=True)


def render_empty_state(title, message, icon="📭"):
    """Empty state - חדש! 🆕"""
    
    st.markdown(f"""
        <style>
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            text-align: center;
            background: {COLORS['gray_50']};
            border-radius: 12px;
            border: 2px dashed {COLORS['gray_300']};
        }}
        .empty-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}
        .empty-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 0.5rem;
        }}
        .empty-message {{
            font-size: 1rem;
            color: {COLORS['text_secondary']};
        }}
        </style>
        
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-message">{message}</div>
        </div>
    """, unsafe_allow_html=True)


def render_loading(message="טוען..."):
    """Loading spinner - חדש! 🆕"""
    
    st.markdown(f"""
        <style>
        .loading-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            gap: 1rem;
        }}
        .spinner {{
            border: 4px solid {COLORS['gray_200']};
            border-top: 4px solid {COLORS['primary']};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            color: {COLORS['text_secondary']};
            font-size: 1rem;
            font-weight: 500;
        }}
        </style>
        
        <div class="loading-container">
            <div class="spinner"></div>
            <div class="loading-text">{message}</div>
        </div>
    """, unsafe_allow_html=True)


def render_card_new(title, content, icon="", variant="default"):
    """Card משופר - חדש! 🆕"""
    
    variants = {
        'default': {'border': COLORS['gray_200'], 'bg': 'white'},
        'success': {'border': COLORS['success'], 'bg': '#F0FDF4'},
        'warning': {'border': COLORS['warning'], 'bg': '#FFFBEB'},
        'error': {'border': COLORS['error'], 'bg': '#FEF2F2'},
        'info': {'border': COLORS['info'], 'bg': '#EFF6FF'}
    }
    
    style = variants[variant]
    
    st.markdown(f"""
        <style>
        .card-new {{
            background: {style['bg']};
            border: 2px solid {style['border']};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s;
        }}
        .card-new:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .card-header-new {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }}
        .card-icon-new {{
            font-size: 2rem;
        }}
        .card-title-new {{
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
            color: {COLORS['text_primary']};
        }}
        .card-content-new {{
            color: {COLORS['text_secondary']};
            line-height: 1.6;
        }}
        </style>
        
        <div class="card-new">
            <div class="card-header-new">
                {f'<div class="card-icon-new">{icon}</div>' if icon else ''}
                <h3 class="card-title-new">{title}</h3>
            </div>
            <div class="card-content-new">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_divider(text=""):
    """מפריד עם טקסט - חדש! 🆕"""
    
    st.markdown(f"""
        <style>
        .divider {{
            display: flex;
            align-items: center;
            text-align: center;
            margin: 2rem 0;
        }}
        .divider::before,
        .divider::after {{
            content: '';
            flex: 1;
            border-bottom: 1px solid {COLORS['gray_300']};
        }}
        .divider-text {{
            padding: 0 1rem;
            color: {COLORS['text_secondary']};
            font-weight: 500;
            font-size: 0.875rem;
        }}
        </style>
        
        <div class="divider">
            {f'<span class="divider-text">{text}</span>' if text else ''}
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# פונקציה משולבת - מומלץ! ⭐
# ==========================================

def apply_all_styles():
    """
    מריץ את כל הסגנונות - ישן + חדש
    השתמש בזה בתחילת app.py!
    """
    setup_page()
    apply_css()


# ==========================================
# מדריך שימוש 📖
# ==========================================

"""
איך להשתמש בקובץ החדש?

1️⃣ בתחילת app.py:
   
   from styles import *
   
   apply_all_styles()  # ← זה מריץ הכל!

2️⃣ להשתמש ב-CSS הישן (עובד כמו קודם):
   
   st.markdown('<div class="kpi-container">...</div>', unsafe_allow_html=True)

3️⃣ להשתמש ברכיבים החדשים:
   
   render_alert("הפרויקט נשמר!", "success")
   render_progress(45, 100, "התקדמות")
   render_stat_card_new("ביצוע", "145 מ'", "+12", "🏗️", "up")

4️⃣ Header חדש (אופציונלי):
   
   render_header_new(user_name="אליהו כהן")


📊 דוגמה מלאה:

from styles import *

# בתחילת האפליקציה
apply_all_styles()
render_header_new()

# התראות
render_alert("הפרויקט נשמר בהצלחה!", "success")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_stat_card_new("ביצוע", "145.5 מ'", "+12.5", "🏗️", "up")

# Progress
render_progress(45, 100, "התקדמות כללית")

# Empty state
if not projects:
    render_empty_state("אין פרויקטים", "העלה תוכנית חדשה")
"""
