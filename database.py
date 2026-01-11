import sqlite3
import os
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

DB_FILE = "project_data.db"

def init_database():
    """יוצר את מסד הנתונים ואת הטבלאות אם הן לא קיימות"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # טבלת plans - תוכניות בנייה
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            plan_name TEXT,
            extracted_scale TEXT,
            confirmed_scale REAL,
            raw_pixel_count INTEGER,
            metadata_json TEXT,
            target_date DATE,
            budget_limit REAL,
            cost_per_meter REAL,
            material_estimate TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # הוספת עמודות חדשות לטבלה קיימת (migration)
    try:
        cursor.execute("ALTER TABLE plans ADD COLUMN target_date DATE")
    except sqlite3.OperationalError:
        pass  # העמודה כבר קיימת
    
    try:
        cursor.execute("ALTER TABLE plans ADD COLUMN budget_limit REAL")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE plans ADD COLUMN cost_per_meter REAL")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE plans ADD COLUMN material_estimate TEXT")
    except sqlite3.OperationalError:
        pass
    
    # טבלת progress_reports - דיווחי ביצוע
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS progress_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id INTEGER NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            meters_built REAL NOT NULL,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
        )
    """)
    
    # אינדקסים לשיפור ביצועים
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_filename ON plans(filename)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_plan_id ON progress_reports(plan_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_date ON progress_reports(date)")
    
    conn.commit()
    conn.close()

def get_plan_by_filename(filename: str) -> Optional[Dict]:
    """מביא תוכנית לפי שם קובץ"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # מאפשר גישה בשם עמודה
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM plans WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return dict(row)
    return None

def save_plan(filename: str, plan_name: Optional[str], extracted_scale: Optional[str], 
              raw_pixel_count: int, metadata_json: str = "", confirmed_scale: Optional[float] = None,
              target_date: Optional[str] = None, budget_limit: Optional[float] = None,
              cost_per_meter: Optional[float] = None, material_estimate: Optional[str] = None) -> int:
    """
    שומר או מעדכן תוכנית במסד הנתונים
    מחזירה: plan_id
    """
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # בדיקה אם התוכנית כבר קיימת
    existing = get_plan_by_filename(filename)
    
    if existing:
        # עדכון תוכנית קיימת
        cursor.execute("""
            UPDATE plans 
            SET plan_name = ?, extracted_scale = ?, confirmed_scale = ?, 
                raw_pixel_count = ?, metadata_json = ?, target_date = ?, 
                budget_limit = ?, cost_per_meter = ?, material_estimate = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE filename = ?
        """, (plan_name, extracted_scale, confirmed_scale, raw_pixel_count, metadata_json,
              target_date, budget_limit, cost_per_meter, material_estimate, filename))
        plan_id = existing['id']
    else:
        # יצירת תוכנית חדשה
        cursor.execute("""
            INSERT INTO plans (filename, plan_name, extracted_scale, confirmed_scale, 
                             raw_pixel_count, metadata_json, target_date, budget_limit,
                             cost_per_meter, material_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, plan_name, extracted_scale, confirmed_scale, raw_pixel_count, metadata_json,
              target_date, budget_limit, cost_per_meter, material_estimate))
        plan_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return plan_id

def save_progress_report(plan_id: int, meters_built: float, note: str = "") -> int:
    """
    שומר דיווח ביצוע חדש
    מחזירה: report_id
    """
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    cursor.execute("""
        INSERT INTO progress_reports (plan_id, date, time, meters_built, note)
        VALUES (?, ?, ?, ?, ?)
    """, (plan_id, date_str, time_str, meters_built, note))
    
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def get_progress_reports(plan_id: Optional[int] = None) -> List[Dict]:
    """מביא דיווחי ביצוע - אפשר לסנן לפי plan_id"""
    init_database()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if plan_id:
        cursor.execute("""
            SELECT pr.*, p.filename, p.plan_name 
            FROM progress_reports pr
            JOIN plans p ON pr.plan_id = p.id
            WHERE pr.plan_id = ?
            ORDER BY pr.date DESC, pr.time DESC
        """, (plan_id,))
    else:
        cursor.execute("""
            SELECT pr.*, p.filename, p.plan_name 
            FROM progress_reports pr
            JOIN plans p ON pr.plan_id = p.id
            ORDER BY pr.date DESC, pr.time DESC
        """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_all_plans() -> List[Dict]:
    """מביא את כל התוכניות"""
    init_database()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM plans ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_plan_stats(plan_id: int) -> Dict:
    """מביא סטטיסטיקות לתוכנית מסוימת"""
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # סך הכל מטרים שנבנו
    cursor.execute("SELECT SUM(meters_built) as total FROM progress_reports WHERE plan_id = ?", (plan_id,))
    total_built = cursor.fetchone()[0] or 0.0
    
    # מספר דיווחים
    cursor.execute("SELECT COUNT(*) FROM progress_reports WHERE plan_id = ?", (plan_id,))
    report_count = cursor.fetchone()[0] or 0
    
    # תוכנית עצמה
    plan = get_plan_by_id(plan_id)
    
    conn.close()
    
    return {
        "plan": plan,
        "total_built": total_built,
        "report_count": report_count
    }

def get_plan_by_id(plan_id: int) -> Optional[Dict]:
    """מביא תוכנית לפי ID"""
    init_database()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM plans WHERE id = ?", (plan_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return dict(row)
    return None

def calculate_velocity(plan_id: int) -> float:
    """
    מחשב את קצב העבודה הממוצע (מטרים ליום) עבור תוכנית מסוימת
    מחזירה: מטרים ליום (float) או 0 אם אין מספיק נתונים
    """
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # קבלת כל הדיווחים של התוכנית, ממוינים לפי תאריך
    cursor.execute("""
        SELECT date, SUM(meters_built) as daily_total
        FROM progress_reports
        WHERE plan_id = ?
        GROUP BY date
        ORDER BY date ASC
    """, (plan_id,))
    
    reports = cursor.fetchall()
    conn.close()
    
    if len(reports) < 2:
        # צריך לפחות 2 ימים כדי לחשב velocity
        return 0.0
    
    # חישוב סך הכל ימים שבין הדיווח הראשון לאחרון
    first_date = datetime.strptime(reports[0][0], "%Y-%m-%d")
    last_date = datetime.strptime(reports[-1][0], "%Y-%m-%d")
    days_diff = (last_date - first_date).days + 1  # +1 כי כולל את היום הראשון
    
    if days_diff == 0:
        return 0.0
    
    # חישוב סך הכל מטרים שנבנו
    total_meters = sum(r[1] for r in reports)
    
    # velocity = סך מטרים / סך ימים
    velocity = total_meters / days_diff
    
    return velocity

def get_project_forecast(plan_id: int) -> Dict:
    """
    מחשב תחזית סיום עבור תוכנית מסוימת
    מחזירה: Dict עם remaining_work, average_velocity, days_to_finish, estimated_completion_date
    """
    plan = get_plan_by_id(plan_id)
    if not plan:
        return {
            "remaining_work": 0.0,
            "average_velocity": 0.0,
            "days_to_finish": 0,
            "estimated_completion_date": None,
            "total_planned": 0.0,
            "cumulative_progress": 0.0
        }
    
    # קבלת סך הכל מתוכנן (מפיקסלים וסקלה)
    total_planned = 0.0
    if plan.get('raw_pixel_count') and plan.get('confirmed_scale') and plan['confirmed_scale'] > 0:
        total_planned = plan['raw_pixel_count'] / plan['confirmed_scale']
    
    # קבלת סך הכל שבוצע
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(meters_built) FROM progress_reports WHERE plan_id = ?", (plan_id,))
    cumulative_progress = cursor.fetchone()[0] or 0.0
    conn.close()
    
    remaining_work = max(0.0, total_planned - cumulative_progress)
    average_velocity = calculate_velocity(plan_id)
    
    days_to_finish = 0
    estimated_completion_date = None
    
    if average_velocity > 0:
        days_to_finish = int(remaining_work / average_velocity)
        
        # חישוב תאריך סיום משוער (לא כולל סופ"ש)
        today = datetime.now().date()
        working_days_added = 0
        days_to_add = days_to_finish
        
        while days_to_add > 0:
            current_date = today + timedelta(days=working_days_added)
            # בדיקה אם זה יום עבודה (0=Monday, 4=Friday)
            if current_date.weekday() < 5:  # יום ראשון עד חמישי
                days_to_add -= 1
            working_days_added += 1
            
            # הגנה מפני לולאה אינסופית
            if working_days_added > days_to_finish * 2:
                break
        
        estimated_completion_date = today + timedelta(days=working_days_added)
    
    return {
        "remaining_work": remaining_work,
        "average_velocity": average_velocity,
        "days_to_finish": days_to_finish,
        "estimated_completion_date": estimated_completion_date,
        "total_planned": total_planned,
        "cumulative_progress": cumulative_progress
    }

def calculate_material_estimates(total_length: float, wall_height: float = 2.5) -> Dict:
    """
    מחשב הערכת חומרים לבנייה
    Block Count = Total_Length * Wall_Height * 10 (בלוקים למ"ר)
    מחזירה: Dict עם block_count, cement_estimate, sand_estimate
    """
    # חישוב שטח קירות (מ"ר)
    wall_area = total_length * wall_height
    
    # חישוב מספר בלוקים (10 בלוקים למ"ר)
    block_count = int(wall_area * 10)
    
    # הערכת מלט (כ-0.07 מ"ק מלט למ"ר)
    cement_cubic_meters = wall_area * 0.07
    
    # הערכת חול (כ-0.14 מ"ק חול למ"ר)
    sand_cubic_meters = wall_area * 0.14
    
    return {
        "block_count": block_count,
        "cement_cubic_meters": round(cement_cubic_meters, 2),
        "sand_cubic_meters": round(sand_cubic_meters, 2),
        "wall_area_sqm": round(wall_area, 2),
        "wall_height": wall_height
    }

def get_project_financial_status(plan_id: int) -> Dict:
    """
    מחשב מצב פיננסי של פרויקט
    מחזירה: Dict עם budget_limit, cost_per_meter, current_cost, budget_variance
    """
    plan = get_plan_by_id(plan_id)
    if not plan:
        return {
            "budget_limit": 0.0,
            "cost_per_meter": 0.0,
            "current_cost": 0.0,
            "budget_variance": 0.0,
            "progress_meters": 0.0
        }
    
    budget_limit = plan.get('budget_limit') or 0.0
    cost_per_meter = plan.get('cost_per_meter') or 0.0
    
    # קבלת סך הכל שבוצע
    init_database()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(meters_built) FROM progress_reports WHERE plan_id = ?", (plan_id,))
    progress_meters = cursor.fetchone()[0] or 0.0
    conn.close()
    
    current_cost = progress_meters * cost_per_meter
    budget_variance = budget_limit - current_cost
    
    return {
        "budget_limit": budget_limit,
        "cost_per_meter": cost_per_meter,
        "current_cost": current_cost,
        "budget_variance": budget_variance,
        "progress_meters": progress_meters
    }

def reset_all_data() -> bool:
    """
    איפוס כל הנתונים - מוחק את כל הרשומות מטבלאות plans ו-progress_reports
    ⚠️ פעולה מסוכנת - לא ניתן לבטל!
    מחזירה: True אם הצליח, False אחרת
    """
    try:
        init_database()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # מחיקת כל הדיווחים
        cursor.execute("DELETE FROM progress_reports")
        
        # מחיקת כל התוכניות
        cursor.execute("DELETE FROM plans")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ שגיאה באיפוס נתונים: {e}")
        return False
