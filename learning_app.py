import streamlit as st
import pandas as pd

# הגדרת דף
st.set_page_config(
    page_title="פלטפורמת למידה לבתי ספר",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# עיצוב מותאם אישית
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .selection-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# כותר ראשי
st.markdown('<h1 class="main-header">📚 פלטפורמת למידה לבתי ספר</h1>', unsafe_allow_html=True)

# נתונים ראשוניים
CLASSES = {
    "כיתה א'": 1,
    "כיתה ב'": 2,
    "כיתה ג'": 3,
    "כיתה ד'": 4,
    "כיתה ה'": 5,
    "כיתה ו'": 6,
    "כיתה ז'": 7,
    "כיתה ח'": 8,
    "כיתה ט'": 9,
    "כיתה י'": 10,
    "כיתה יא'": 11,
    "כיתה יב'": 12
}

SUBJECTS = {
    "מתמטיקה": "math",
    "עברית": "hebrew",
    "אנגלית": "english",
    "מדעים": "science",
    "היסטוריה": "history",
    "גיאוגרפיה": "geography",
    "ספרות": "literature",
    "פיזיקה": "physics",
    "כימיה": "chemistry",
    "ביולוגיה": "biology"
}

LEARNING_LEVELS = {
    "רמה בסיסית": "basic",
    "רמה בינונית": "intermediate",
    "רמה מתקדמת": "advanced",
    "רמה מצוינת": "excellent"
}

# סיידבר לבחירות
st.sidebar.header("⚙️ בחירת פרמטרי למידה")

selected_class = st.sidebar.selectbox(
    "בחר כיתה:",
    options=list(CLASSES.keys()),
    index=5  # ברירת מחדל: כיתה ו'
)

selected_subject = st.sidebar.selectbox(
    "בחר מקצוע:",
    options=list(SUBJECTS.keys()),
    index=0  # ברירת מחדל: מתמטיקה
)

selected_level = st.sidebar.selectbox(
    "בחר רמת לימוד:",
    options=list(LEARNING_LEVELS.keys()),
    index=1  # ברירת מחדל: רמה בינונית
)

# תוכן ראשי
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="selection-box">', unsafe_allow_html=True)
    st.subheader("📋 סיכום הבחירות שלך")
    
    # הצגת הבחירות
    info_data = {
        "פרמטר": ["כיתה", "מקצוע", "רמת לימוד"],
        "ערך": [selected_class, selected_subject, selected_level]
    }
    info_df = pd.DataFrame(info_data)
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # אזור תוכן הלמידה
    st.markdown("---")
    st.subheader(f"📖 תוכן למידה - {selected_subject}")
    
    # הודעת מותאמת אישית לפי הבחירות
    class_num = CLASSES[selected_class]
    subject_code = SUBJECTS[selected_subject]
    level_code = LEARNING_LEVELS[selected_level]
    
    # תוכן דינמי לפי הבחירות
    if class_num <= 3:
        grade_category = "יסודי נמוך"
    elif class_num <= 6:
        grade_category = "יסודי גבוה"
    elif class_num <= 9:
        grade_category = "חטיבת ביניים"
    else:
        grade_category = "תיכון"
    
    st.info(f"""
    **קטגוריית כיתה:** {grade_category}
    
    **מקצוע נבחר:** {selected_subject}
    
    **רמת קושי:** {selected_level}
    
    **מזהה ייחודי:** {class_num}-{subject_code}-{level_code}
    """)
    
    # דוגמאות תוכן לפי מקצוע
    content_examples = {
        "מתמטיקה": {
            "basic": "📐 חיבור וחיסור עד 20, זיהוי צורות בסיסיות",
            "intermediate": "📊 כפל וחילוק, שברים פשוטים, בעיות מילוליות",
            "advanced": "📈 אלגברה בסיסית, משוואות, גיאומטריה",
            "excellent": "🔢 אלגברה מתקדמת, טריגונומטריה, חשבון דיפרנציאלי"
        },
        "עברית": {
            "basic": "🔤 אותיות וצלילים, קריאה בסיסית",
            "intermediate": "📝 הבנת הנקרא, כתיבה נכונה, דקדוק בסיסי",
            "advanced": "📚 ניתוח טקסטים, חיבור, דקדוק מתקדם",
            "excellent": "✍️ כתיבה יצירתית, ניתוח ספרותי, ביטוי עצמי"
        },
        "אנגלית": {
            "basic": "🔤 ABC, מילים בסיסיות, משפטים קצרים",
            "intermediate": "📖 קריאה בסיסית, דקדוק, שיחה יומיומית",
            "advanced": "📚 קריאה מתקדמת, כתיבה, הבנת הנשמע",
            "excellent": "💬 שיחה שוטפת, כתיבה אקדמית, ספרות אנגלית"
        }
    }
    
    if selected_subject in content_examples:
        st.success(f"**תוכן מומלץ:** {content_examples[selected_subject].get(level_code, 'תוכן מותאם אישית')}")
    else:
        st.success(f"**תוכן מותאם אישית** עבור {selected_subject} ברמה {selected_level}")

with col2:
    st.subheader("📊 סטטיסטיקות")
    
    # מטריקות
    st.metric("מספר כיתות זמינות", len(CLASSES))
    st.metric("מספר מקצועות", len(SUBJECTS))
    st.metric("מספר רמות", len(LEARNING_LEVELS))
    
    st.markdown("---")
    
    # גרף התפלגות כיתות
    st.subheader("📈 התפלגות כיתות")
    class_dist = pd.DataFrame({
        "כיתה": list(CLASSES.keys()),
        "מספר": list(CLASSES.values())
    })
    st.bar_chart(class_dist.set_index("כיתה")["מספר"])

# כפתור התחלת למידה
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🚀 התחל למידה", use_container_width=True, type="primary"):
        st.balloons()
        st.success(f"✅ התחלת למידה ב-{selected_subject} לכיתה {selected_class} ברמה {selected_level}!")
        st.info("💡 כאן תוכל להוסיף תוכן למידה, תרגילים, ומשחקים מותאמים אישית")

# הערות למפתח
with st.expander("ℹ️ מידע למפתח"):
    st.markdown("""
    **מבנה האפליקציה:**
    - בחירת כיתה, מקצוע ורמת לימוד
    - תצוגה דינמית של תוכן לפי הבחירות
    - אפשר להרחיב עם:
        - מסד נתונים של תרגילים
        - מערכת ניקוד והתקדמות
        - משחקים אינטראקטיביים
        - דוחות התקדמות
    """)
