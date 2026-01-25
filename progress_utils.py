# progress_utils.py
"""
✨ Progress Indicators משופרים
מציג progress bar מפורט לכל שלב בעיבוד התוכנית
"""

import streamlit as st
import time
from contextlib import contextmanager

@contextmanager
def progress_tracker(title="מעבד...", steps=None):
    """
    Context manager לניהול progress bar
    
    שימוש:
    ------
    steps = ["טעינת PDF", "זיהוי קירות", "חילוץ מטא-דאטה", "שמירה"]
    
    with progress_tracker("מעבד תוכנית", steps) as progress:
        progress.update(0, "טעינת PDF...")
        # קוד לטעינה
        
        progress.update(1, "מזהה קירות...")
        # קוד לזיהוי
        
        progress.update(2, "מחלץ מטא-דאטה...")
        # קוד לחילוץ
        
        progress.complete("✅ הושלם!")
    """
    
    class ProgressManager:
        def __init__(self, title, steps):
            self.title = title
            self.steps = steps or []
            self.total_steps = len(self.steps) if self.steps else 100
            self.current_step = 0
            
            # יצירת UI elements
            self.title_placeholder = st.empty()
            self.progress_bar = st.progress(0)
            self.status_placeholder = st.empty()
            
            # הצגת כותרת
            self.title_placeholder.markdown(f"### {title}")
        
        def update(self, step_index, message=""):
            """
            מעדכן את ה-progress bar
            
            Args:
                step_index: אינדקס השלב (0-based)
                message: הודעת סטטוס
            """
            self.current_step = step_index
            
            # חישוב אחוז
            if self.steps:
                progress = (step_index + 1) / self.total_steps
                step_name = self.steps[step_index] if step_index < len(self.steps) else message
            else:
                progress = step_index / 100.0
                step_name = message
            
            # עדכון UI
            self.progress_bar.progress(min(progress, 1.0))
            
            # הודעת סטטוס עם אייקון
            if message:
                self.status_placeholder.info(f"🔄 {message}")
            elif step_name:
                self.status_placeholder.info(f"🔄 {step_name}")
            
            # המתנה קצרה לאנימציה
            time.sleep(0.1)
        
        def complete(self, message="✅ הושלם בהצלחה!"):
            """סיום עם הודעת הצלחה"""
            self.progress_bar.progress(1.0)
            self.status_placeholder.success(message)
        
        def error(self, message="❌ שגיאה"):
            """סיום עם שגיאה"""
            self.status_placeholder.error(message)
        
        def cleanup(self):
            """מנקה את ה-UI elements"""
            time.sleep(1.5)  # המתנה להצגת סטטוס סופי
            self.title_placeholder.empty()
            self.progress_bar.empty()
            self.status_placeholder.empty()
    
    # יצירת manager
    manager = ProgressManager(title, steps)
    
    try:
        yield manager
    finally:
        # ניקוי אוטומטי
        manager.cleanup()


# ==========================================
# דוגמאות שימוש
# ==========================================

def process_pdf_with_progress(pdf_path):
    """
    דוגמה: עיבוד PDF עם progress indicators
    """
    from analyzer import FloorPlanAnalyzer
    
    steps = [
        "📄 טעינת PDF וניתוח",
        "🔍 זיהוי קירות ומבנים",
        "🧠 חילוץ מטא-דאטה עם AI",
        "💾 שמירה למסד נתונים",
        "✅ סיום עיבוד"
    ]
    
    with progress_tracker("מעבד תוכנית", steps) as progress:
        
        # שלב 1: טעינה
        progress.update(0)
        analyzer = FloorPlanAnalyzer()
        
        # שלב 2: זיהוי
        progress.update(1)
        result = analyzer.process_file(pdf_path)
        
        # שלב 3: מטא-דאטה
        progress.update(2)
        metadata = extract_metadata(result)
        
        # שלב 4: שמירה
        progress.update(3)
        save_to_database(metadata)
        
        # שלב 5: סיום
        progress.update(4)
        time.sleep(0.5)
        progress.complete("🎉 התוכנית עובדה בהצלחה!")
    
    return result


def upload_files_with_progress(files):
    """
    דוגמה: העלאת קבצים מרובים עם progress
    """
    total_files = len(files)
    
    with progress_tracker("מעלה קבצים", None) as progress:
        
        for i, file in enumerate(files):
            # עדכון progress
            percent = int((i / total_files) * 100)
            progress.update(percent, f"מעבד קובץ {i+1}/{total_files}: {file.name}")
            
            # עיבוד הקובץ
            process_file(file)
        
        progress.complete(f"✅ הועלו {total_files} קבצים בהצלחה!")


def analyze_with_substeps(image):
    """
    דוגמה: ניתוח עם sub-steps
    """
    main_steps = [
        "זיהוי טקסט",
        "זיהוי קירות",
        "זיהוי חומרים",
        "חישוב כמויות"
    ]
    
    with progress_tracker("מנתח תוכנית", main_steps) as progress:
        
        # שלב 1: טקסט
        progress.update(0, "מחלץ טקסט מ-PDF...")
        text = extract_text(image)
        
        # שלב 2: קירות
        progress.update(1, "מזהה קירות בתוכנית...")
        walls = detect_walls(image)
        
        # Sub-step
        progress.status_placeholder.info("🔄 מזהה קירות - מסנן רעשים...")
        time.sleep(0.5)
        
        progress.status_placeholder.info("🔄 מזהה קירות - מחשב אורכים...")
        time.sleep(0.5)
        
        # שלב 3: חומרים
        progress.update(2, "מזהה סוגי חומרים...")
        materials = detect_materials(walls)
        
        # שלב 4: כמויות
        progress.update(3, "מחשב כמויות...")
        quantities = calculate_quantities(materials)
        
        progress.complete("✅ ניתוח הושלם!")
    
    return quantities


# ==========================================
# Helper Functions (דוגמאות בלבד)
# ==========================================

def extract_text(image):
    time.sleep(1)
    return "dummy text"

def detect_walls(image):
    time.sleep(1.5)
    return "dummy walls"

def detect_materials(walls):
    time.sleep(1)
    return "dummy materials"

def calculate_quantities(materials):
    time.sleep(0.5)
    return "dummy quantities"

def extract_metadata(result):
    time.sleep(0.8)
    return {}

def save_to_database(metadata):
    time.sleep(0.5)
    pass

def process_file(file):
    time.sleep(0.3)
    pass
