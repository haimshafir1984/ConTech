"""
ConTech Pro - מערכת עיצוב מלאה ומשופרת
קובץ styles.py עם עיצוב חדש + תאימות מלאה לאחור
"""

import streamlit as st
from datetime import datetime

# ==========================================
# Design System - צבעים
# ==========================================

COLORS = {
    # Primary (כחול בנייה - כמו הישן!)
    'primary': '#0F62FE',
    'primary_dark': '#0043CE',
    'primary_light': '#4589FF',
    
    # Secondary (ירוק הצלחה)
    'secondary': '#10B981',
    'secondary_dark': '#059669',
    'secondary_light': '#34D399',
    
    # Accent (כתום)
    'accent': '#F59E0B',
    'accent_dark': '#D97706',
    'accent_light': '#FBBF24',
    
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
    'gray_400': '#9CA3AF',
    'gray_500': '#6B7280',
    'gray_600': '#4B5563',
    'gray_700': '#374151',
    'gray_800': '#1F2937',
    'gray_900': '#111827',
    
    # Background
    'bg_primary': '#FFFFFF',
    'bg_secondary': '#F4F7F6',  # ← כמו הישן!
    'bg_tertiary': '#F3F4F6',
    
    # Text
    'text_primary': '#111827',
    'text_secondary': '#6B7280',
    'text_tertiary': '#9CA3AF',
    
    # Borders
    'border': '#E0E0E0',  # ← כמו הישן!
}


# ==========================================
# פונקציות ישנות - תאימות מלאה! ✅
# ==========================================

def setup_page():
    """
    הגדרת עמוד - פונקציה מקורית
    עובדת בדיוק כמו קודם! ✅
    """
    st.set_page_config(
        page_title="ConTech Pro", 
        layout="wide", 
        page_icon="🏗️"
    )


def apply_css():
    """
    CSS מקורי + שיפורים
    שומר על כל הקלאסים הישנים + מוסיף חדשים!
    """
    st.markdown(f"""
    <style>
        /* =============================================== */
        /* CSS מקורי - עובד בדיוק כמו קודם! ✅ */
        /* =============================================== */
        
        @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;800&display=swap');
        
        /* הגדרות בסיס */
        html, body, [class*="css"] {{ 
            font-family: 'Heebo', sans-serif; 
            direction: rtl; 
        }}
        
        :root {{ 
            --primary-blue: {COLORS['primary']}; 
            --bg-gray: {COLORS['bg_secondary']}; 
            --card-border: {COLORS['border']}; 
        }}
        
        /* ============================================= */
        /* כרטיסיות - ישן (עובד!) */
        /* ============================================= */
        .stCard {{ 
            background-color: white; 
            padding: 24px; 
            border-radius: 12px; 
            border: 1px solid var(--card-border); 
            box-shadow: 0 2px 8px rgba(0,0,0,0.04); 
            margin-bottom: 20px; 
        }}
        
        /* ============================================= */
        /* KPI Dashboard - ישן (עובד!) */
        /* ============================================= */
        .kpi-container {{ 
            display: flex; 
            flex-direction: column; 
            background: white; 
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid #EAEAEA; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); 
            height: 100%; 
            text-align: center;
        }}
        .kpi-icon {{ font-size: 24px; margin-bottom: 10px; }}
        .kpi-label {{ font-size: 14px; color: #666; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; color: {COLORS['primary']}; margin: 5px 0; }}
        .kpi-sub {{ font-size: 12px; color: #999; }}
        
        /* ============================================= */
        /* כרטיסי חומרים - ישן (עובד!) */
        /* ============================================= */
        .mat-card {{ 
            text-align: center; 
            background: white; 
            border: 1px solid #EEE; 
            border-radius: 10px; 
            padding: 15px; 
        }}
        .mat-val {{ font-size: 20px; font-weight: bold; color: {COLORS['primary']}; }}
        .mat-lbl {{ font-size: 14px; color: #666; }}
        
        /* ============================================= */
        /* תיבת מחיר - ישן (עובד!) */
        /* ============================================= */
        .price-box {{ 
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            border-right: 4px solid {COLORS['primary']}; 
            margin-bottom: 10px; 
        }}
        
        /* ============================================= */
        /* התאמות כלליות - ישן (עובד!) */
        /* ============================================= */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {{ 
            text-align: right !important; 
            width: 100%; 
            direction: rtl; 
            font-weight: 500;
            color: {COLORS['text_primary']};
        }}
        
        .stButton button {{ 
            border-radius: 8px; 
            font-weight: 500; 
            height: 45px;
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            transition: all 0.2s;
        }}
        
        .stButton button:hover {{
            background-color: {COLORS['primary_dark']};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        section[data-testid="stSidebar"] {{ 
            background-color: #FAFAFA; 
            border-left: 1px solid #EEE; 
        }}
        
        /* =============================================== */
        /* שיפורים חדשים - נוספים! 🆕 */
        /* =============================================== */
        
        /* Background כללי */
        .main {{
            background-color: {COLORS['bg_secondary']};
        }}
        
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}
        
        /* Headers משופרים */
        h1 {{
            color: {COLORS['text_primary']};
            font-weight: 700;
            margin-bottom: 1.5rem;
            font-size: 2.5rem;
        }}
        
        h2 {{
            color: {COLORS['text_primary']};
            font-weight: 600;
            margin-bottom: 1rem;
            font-size: 2rem;
        }}
        
        h3 {{
            color: {COLORS['text_secondary']};
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 1.5rem;
        }}
        
        /* Streamlit Components משופרים */
        .stTextInput>div>div>input, 
        .stSelectbox>div>div>select,
        .stNumberInput>div>div>input {{
            border: 2px solid {COLORS['gray_300']};
            border-radius: 8px;
            padding: 0.75rem;
            font-size: 1rem;
            transition: all 0.2s;
        }}
        
        .stTextInput>div>div>input:focus, 
        .stSelectbox>div>div>select:focus,
        .stNumberInput>div>div>input:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px {COLORS['primary_light']}33;
            outline: none;
        }}
        
        /* Tabs משופרים */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background-color: white;
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            color: {COLORS['text_secondary']};
            transition: all 0.2s;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
        }}
        
        /* Expander משופר */
        .streamlit-expanderHeader {{
            background-color: white;
            border: 1px solid {COLORS['gray_200']};
            border-radius: 8px;
            font-weight: 500;
            color: {COLORS['text_primary']};
            transition: all 0.2s;
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: {COLORS['gray_50']};
            border-color: {COLORS['primary']};
        }}
        
        /* Metrics משופרים */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS['primary']};
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            font-weight: 500;
        }}
        
        /* Success/Error messages */
        .stSuccess {{
            background-color: #F0FDF4;
            border-right: 4px solid {COLORS['success']};
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .stError {{
            background-color: #FEF2F2;
            border-right: 4px solid {COLORS['error']};
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .stWarning {{
            background-color: #FFFBEB;
            border-right: 4px solid {COLORS['warning']};
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .stInfo {{
            background-color: #EFF6FF;
            border-right: 4px solid {COLORS['info']};
            border-radius: 8px;
            padding: 1rem;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem;
            }}
            
            h1 {{ font-size: 1.75rem !important; }}
            h2 {{ font-size: 1.5rem !important; }}
            h3 {{ font-size: 1.25rem !important; }}
            
            .kpi-container {{
                margin-bottom: 1rem;
            }}
            
            .stButton button {{
                height: 40px;
                font-size: 0.9rem;
            }}
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Spinner customization */
        .stSpinner > div {{
            border-color: {COLORS['primary']} transparent {COLORS['primary']} transparent;
        }}
    </style>
    """, unsafe_allow_html=True)


def apply_all_styles():
    """
    פונקציה משולבת - מריצה הכל!
    עובדת בדיוק כמו קודם! ✅
    """
    setup_page()
    apply_css()


# ==========================================
# רכיבים חדשים - תוספות! 🆕
# ==========================================

def render_header(user_name="מנהל פרויקט", show_date=True):
    """Header מקצועי חדש - אופציונלי!"""
    
    date_str = datetime.now().strftime('%d/%m/%Y') if show_date else ""
    
    st.markdown(f"""
        <style>
        .contech-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .logo {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .logo-icon {{
            font-size: 2.5rem;
            animation: float 3s ease-in-out infinite;
        }}
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
        }}
        .logo-text h1 {{
            color: white;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .logo-text p {{
            color: rgba(255,255,255,0.95);
            margin: 0;
            font-size: 0.875rem;
            font-weight: 400;
        }}
        .user-section {{
            color: white;
            text-align: right;
            background: rgba(255,255,255,0.1);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }}
        .user-name {{
            font-weight: 500;
            margin-bottom: 0.25rem;
        }}
        .user-date {{
            font-size: 0.875rem;
            opacity: 0.9;
        }}
        
        @media (max-width: 768px) {{
            .contech-header {{
                padding: 1rem;
            }}
            .header-content {{
                flex-direction: column;
                gap: 1rem;
            }}
            .logo-text h1 {{
                font-size: 1.5rem;
            }}
            .user-section {{
                width: 100%;
                text-align: center;
            }}
        }}
        </style>
        
        <div class="contech-header">
            <div class="header-content">
                <div class="logo">
                    <div class="logo-icon">🏗️</div>
                    <div class="logo-text">
                        <h1>ConTech Pro</h1>
                        <p>מערכת ניהול פרויקטים חכמה לבנייה</p>
                    </div>
                </div>
                <div class="user-section">
                    <div class="user-name">👤 {user_name}</div>
                    {f'<div class="user-date">📅 {date_str}</div>' if show_date else ''}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_alert(message, variant="info", icon=None):
    """התראה מעוצבת"""
    
    variants = {
        'success': {
            'bg': '#F0FDF4', 
            'border': COLORS['success'], 
            'icon': icon or '✅', 
            'text': '#065F46'
        },
        'warning': {
            'bg': '#FFFBEB', 
            'border': COLORS['warning'], 
            'icon': icon or '⚠️', 
            'text': '#92400E'
        },
        'error': {
            'bg': '#FEF2F2', 
            'border': COLORS['error'], 
            'icon': icon or '❌', 
            'text': '#991B1B'
        },
        'info': {
            'bg': '#EFF6FF', 
            'border': COLORS['info'], 
            'icon': icon or 'ℹ️', 
            'text': '#1E40AF'
        }
    }
    
    style = variants.get(variant, variants['info'])
    
    st.markdown(f"""
        <div style="
            background: {style['bg']};
            border-right: 4px solid {style['border']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 1.5rem; flex-shrink: 0;">{style['icon']}</div>
            <div style="color: {style['text']}; font-weight: 500; flex: 1;">{message}</div>
        </div>
    """, unsafe_allow_html=True)


def render_progress(value, max_value=100, label="", show_percentage=True, height=24):
    """Progress bar מעוצב"""
    
    percentage = (value / max_value) * 100
    
    # בחירת צבע לפי אחוז
    if percentage < 30:
        color = COLORS['error']
    elif percentage < 70:
        color = COLORS['warning']
    else:
        color = COLORS['success']
    
    st.markdown(f"""
        <div style="width: 100%; margin: 1rem 0;">
            {f'''<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.875rem; color: {COLORS['text_secondary']}; font-weight: 500;">
                <span>{label}</span>
                <span>{percentage:.1f}%</span>
            </div>''' if label or show_percentage else ''}
            <div style="
                width: 100%;
                height: {height}px;
                background: {COLORS['gray_200']};
                border-radius: {height//2}px;
                overflow: hidden;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
            ">
                <div style="
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
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    {f'{percentage:.0f}%' if show_percentage and percentage > 15 else ''}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_stat_card(title, value, change=None, icon="", trend="neutral"):
    """KPI Card מעוצב - שם חדש כדי לא להתנגש עם הישן"""
    
    trend_colors = {
        'up': {'color': COLORS['success'], 'icon': '↗'},
        'down': {'color': COLORS['error'], 'icon': '↘'},
        'neutral': {'color': COLORS['text_secondary'], 'icon': '→'}
    }
    
    trend_style = trend_colors.get(trend, trend_colors['neutral'])
    
    st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid {COLORS['gray_200']};
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s;
            height: 100%;
        " onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)';" 
           onmouseout="this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 0.875rem; color: {COLORS['text_secondary']}; font-weight: 500;">
                    {title}
                </div>
                {f'<div style="font-size: 1.5rem;">{icon}</div>' if icon else ''}
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: {COLORS['primary']}; margin-bottom: 0.5rem;">
                {value}
            </div>
            {f'''<div style="display: flex; align-items: center; gap: 0.25rem; font-size: 0.875rem; color: {trend_style['color']}; font-weight: 500;">
                {trend_style['icon']} {change}
            </div>''' if change else ''}
        </div>
    """, unsafe_allow_html=True)


def render_empty_state(title, message, icon="📭"):
    """Empty state מעוצב"""
    
    st.markdown(f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            text-align: center;
            background: {COLORS['gray_50']};
            border-radius: 12px;
            border: 2px dashed {COLORS['gray_300']};
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.5; animation: pulse 2s ease-in-out infinite;">
                {icon}
            </div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 1rem; color: {COLORS['text_secondary']};">
                {message}
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 0.8; }}
        }}
        </style>
    """, unsafe_allow_html=True)


def render_loading(message="טוען..."):
    """Loading spinner מעוצב"""
    
    st.markdown(f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            gap: 1rem;
        ">
            <div style="
                border: 4px solid {COLORS['gray_200']};
                border-top: 4px solid {COLORS['primary']};
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
            "></div>
            <div style="
                color: {COLORS['text_secondary']};
                font-size: 1rem;
                font-weight: 500;
            ">{message}</div>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
    """, unsafe_allow_html=True)


def render_card(title, content, icon="", variant="default"):
    """Card מעוצב"""
    
    variants = {
        'default': {'border': COLORS['gray_200'], 'bg': 'white'},
        'success': {'border': COLORS['success'], 'bg': '#F0FDF4'},
        'warning': {'border': COLORS['warning'], 'bg': '#FFFBEB'},
        'error': {'border': COLORS['error'], 'bg': '#FEF2F2'},
        'info': {'border': COLORS['info'], 'bg': '#EFF6FF'}
    }
    
    style = variants.get(variant, variants['default'])
    
    st.markdown(f"""
        <div style="
            background: {style['bg']};
            border: 2px solid {style['border']};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s;
        " onmouseover="this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)';" 
           onmouseout="this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                {f'<div style="font-size: 2rem;">{icon}</div>' if icon else ''}
                <h3 style="font-size: 1.25rem; font-weight: 600; margin: 0; color: {COLORS['text_primary']};">
                    {title}
                </h3>
            </div>
            <div style="color: {COLORS['text_secondary']}; line-height: 1.6;">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_divider(text=""):
    """מפריד עם טקסט אופציונלי"""
    
    st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            text-align: center;
            margin: 2rem 0;
        ">
            <div style="flex: 1; border-bottom: 1px solid {COLORS['gray_300']};"></div>
            {f'<span style="padding: 0 1rem; color: {COLORS["text_secondary"]}; font-weight: 500; font-size: 0.875rem;">{text}</span>' if text else ''}
            <div style="flex: 1; border-bottom: 1px solid {COLORS['gray_300']};"></div>
        </div>
    """, unsafe_allow_html=True)


def render_badge(text, variant="default"):
    """תג/badge קטן"""
    
    variants = {
        'default': {'bg': COLORS['gray_200'], 'text': COLORS['text_primary']},
        'success': {'bg': COLORS['success'], 'text': 'white'},
        'warning': {'bg': COLORS['warning'], 'text': 'white'},
        'error': {'bg': COLORS['error'], 'text': 'white'},
        'info': {'bg': COLORS['info'], 'text': 'white'},
        'primary': {'bg': COLORS['primary'], 'text': 'white'}
    }
    
    style = variants.get(variant, variants['default'])
    
    st.markdown(f"""
        <span style="
            display: inline-block;
            background: {style['bg']};
            color: {style['text']};
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        ">{text}</span>
    """, unsafe_allow_html=True)


# ==========================================
# Helper Functions
# ==========================================

def show_success(message):
    """Shortcut for success alert"""
    render_alert(message, "success")

def show_error(message):
    """Shortcut for error alert"""
    render_alert(message, "error")

def show_warning(message):
    """Shortcut for warning alert"""
    render_alert(message, "warning")

def show_info(message):
    """Shortcut for info alert"""
    render_alert(message, "info")
