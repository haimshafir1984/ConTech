import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

# בדיקה האם אנחנו בענן (Postgres) או מקומי (SQLite)
DB_URL = os.environ.get("DATABASE_URL")
# שימוש בשם קובץ קבוע וברור יותר למסד הנתונים המקומי
DB_FILE = os.environ.get("DB_FILE_PATH", "project_data.db")

def get_connection():
    """יוצר חיבור למסד הנתונים המתאים (Postgres או SQLite)"""
    if DB_URL:
        try:
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            return conn
        except Exception as e:
            print(f"Error connecting to Postgres: {e}")
            return None
    else:
        # מצב מקומי - שימוש בקובץ
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # מאפשר גישה לשדות לפי שם
        return conn

def init_database():
    """יצירת הטבלאות אם לא קיימות"""
    conn = get_connection()
    if not conn: return
    
    # שאילתות יצירה - מותאמות לשני סוגי המסדים
    if DB_URL:
        # Syntax for PostgreSQL
        id_type = "SERIAL PRIMARY KEY"
        json_type = "TEXT" # פשטות
    else:
        # Syntax for SQLite
        id_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        json_type = "TEXT"

    queries = [
        f"""CREATE TABLE IF NOT EXISTS plans (
            id {id_type},
            filename TEXT UNIQUE,
            plan_name TEXT,
            scale_text TEXT,
            scale_value REAL,
            raw_pixels INTEGER,
            metadata {json_type},
            target_date TEXT,
            budget_limit REAL,
            cost_per_meter REAL,
            materials_json {json_type},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        
        f"""CREATE TABLE IF NOT EXISTS progress_reports (
            id {id_type},
            plan_id INTEGER,
            meters_built REAL,
            note TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(plan_id) REFERENCES plans(id)
        );"""
    ]
    
    try:
        cur = conn.cursor()
        for q in queries:
            cur.execute(q)
        conn.commit()
    except Exception as e:
        print(f"DB Init Error: {e}")
    finally:
        conn.close()

# --- פונקציות עזר לשאילתות (Wrapper) ---
def run_query(query, params=(), fetch="all"):
    conn = get_connection()
    if not conn: return None
    
    # המרת Placeholder מ-? (SQLite) ל-%s (Postgres) אם צריך
    if DB_URL:
        query = query.replace("?", "%s")
        
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        if fetch == "all":
            res = cur.fetchall()
            # המרה למילון רגיל
            return [dict(row) for row in res]
        elif fetch == "one":
            res = cur.fetchone()
            return dict(res) if res else None
        elif fetch == "insert":
            conn.commit()
            if DB_URL:
                # ב-Postgres צריך לבקש את ה-ID בחזרה
                try:
                    return cur.fetchone()['id']
                except:
                    return cur.lastrowid
            else:
                return cur.lastrowid
        else:
            conn.commit()
    except Exception as e:
        print(f"Query Error: {e} | Query: {query}")
        return None
    finally:
        conn.close()

# --- פונקציות האפליקציה ---

def save_plan(filename, plan_name, scale_text, scale_val, pixels, metadata, target_date=None, budget=0, cost=0, materials="{}"):
    # בדיקה אם קיים כבר
    existing = get_plan_by_filename(filename)
    if existing:
        query = """UPDATE plans SET plan_name=?, scale_text=?, scale_value=?, raw_pixels=?, metadata=?, 
                   target_date=?, budget_limit=?, cost_per_meter=?, materials_json=? WHERE id=?"""
        run_query(query, (plan_name, scale_text, scale_val, pixels, metadata, target_date, budget, cost, materials, existing['id']), fetch="commit")
        return existing['id']
    else:
        if DB_URL:
            # Postgres דורש RETURNING כדי לקבל ID
            query = """INSERT INTO plans (filename, plan_name, scale_text, scale_value, raw_pixels, metadata, target_date, budget_limit, cost_per_meter, materials_json) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id"""
            
            # ביצוע ישיר כדי לקבל את ה-ID ב-Postgres
            conn = get_connection()
            try:
                cur = conn.cursor()
                cur.execute(query.replace("?", "%s"), (filename, plan_name, scale_text, scale_val, pixels, metadata, target_date, budget, cost, materials))
                new_id = cur.fetchone()['id']
                conn.commit()
                return new_id
            finally:
                conn.close()
        else:
            query = """INSERT INTO plans (filename, plan_name, scale_text, scale_value, raw_pixels, metadata, target_date, budget_limit, cost_per_meter, materials_json) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            return run_query(query, (filename, plan_name, scale_text, scale_val, pixels, metadata, target_date, budget, cost, materials), fetch="insert")

def save_progress_report(plan_id, meters, note):
    query = "INSERT INTO progress_reports (plan_id, meters_built, note) VALUES (?, ?, ?)"
    run_query(query, (plan_id, meters, note), fetch="insert")

def get_all_plans():
    return run_query("SELECT * FROM plans ORDER BY created_at DESC")

def get_plan_by_filename(filename):
    return run_query("SELECT * FROM plans WHERE filename = ?", (filename,), fetch="one")

def get_plan_by_id(pid):
    return run_query("SELECT * FROM plans WHERE id = ?", (pid,), fetch="one")

def get_progress_reports(plan_id=None):
    if plan_id:
        return run_query("SELECT r.*, p.plan_name FROM progress_reports r JOIN plans p ON r.plan_id = p.id WHERE r.plan_id = ? ORDER BY r.date DESC", (plan_id,))
    else:
        return run_query("SELECT r.*, p.plan_name FROM progress_reports r JOIN plans p ON r.plan_id = p.id ORDER BY r.date DESC")

def reset_all_data():
    run_query("DELETE FROM progress_reports", fetch="commit")
    run_query("DELETE FROM plans", fetch="commit")
    return True

# --- פונקציות חישוב (Business Logic) ---
def get_project_forecast(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan: return {}
    
    reports = get_progress_reports(plan_id)
    total_built = sum([r['meters_built'] for r in reports]) if reports else 0
    
    # --- תיקון קריטי: חישוב אורך מתוכנן ---
    # הנוסחה הנכונה: פיקסלים / (פיקסלים למטר) = מטרים
    if plan['scale_value'] and plan['scale_value'] > 0:
        total_len_meters = plan['raw_pixels'] / plan['scale_value']
    else:
        total_len_meters = 0
    
    # נשמור גם את הערך השגוי הישן כ"planned" למקרה שמישהו מסתמך עליו, אבל נשתמש בחדש
    total_planned = total_len_meters 
    
    # חישוב ימים
    days_passed = 0
    velocity = 0
    if reports:
        dates = [datetime.strptime(str(r['date'])[:10], "%Y-%m-%d") for r in reports]
        if len(dates) >= 2:
            days_passed = (max(dates) - min(dates)).days + 1
            velocity = total_built / days_passed if days_passed > 0 else total_built
        else:
            days_passed = 1
            velocity = total_built
            
    remaining = max(0, total_len_meters - total_built)
    days_to_finish = remaining / velocity if velocity > 0 else 999
    
    return {
        "cumulative_progress": total_built,
        "total_planned": total_planned,
        "remaining_work": remaining,
        "average_velocity": velocity,
        "days_to_finish": int(days_to_finish)
    }

def get_project_financial_status(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan: return {}
    
    reports = get_progress_reports(plan_id)
    total_built = sum([r['meters_built'] for r in reports]) if reports else 0
    
    cost_per_m = plan['cost_per_meter'] if plan['cost_per_meter'] else 0
    current_cost = total_built * cost_per_m
    budget = plan['budget_limit'] if plan['budget_limit'] else 0
    
    return {
        "current_cost": current_cost,
        "budget_limit": budget,
        "budget_variance": budget - current_cost
    }

def calculate_material_estimates(total_length_meters, height_meters=2.5):
    # הערכה גסה: 10 בלוקים למ"ר, עובי 20 ס"מ
    wall_area = total_length_meters * height_meters
    blocks = wall_area * 10 
    cement = wall_area * 0.02 # הערכה: 0.02 קוב למ"ר
    
    return {
        "block_count": int(blocks),
        "cement_cubic_meters": round(cement, 1),
        "wall_area_sqm": round(wall_area, 1)
    }
# ==========================================
# פונקציות לחשבונות חלקיים
# ==========================================

def get_progress_summary_by_date_range(plan_id, start_date, end_date):
    """
    מסכם את כל הדיווחים בטווח תאריכים ומקבץ לפי סוג עבודה
    
    Args:
        plan_id: מזהה התוכנית
        start_date: תאריך התחלה (YYYY-MM-DD)
        end_date: תאריך סיום (YYYY-MM-DD)
    
    Returns:
        רשימה של מילונים עם: work_type, total_quantity, unit
    """
    query = """
        SELECT 
            note as work_type,
            SUM(meters_built) as total_quantity,
            COUNT(*) as report_count
        FROM progress_reports 
        WHERE plan_id = ? 
        AND date >= ? 
        AND date <= ?
        GROUP BY note
        ORDER BY total_quantity DESC
    """
    
    results = run_query(query, (plan_id, start_date, end_date))
    
    if not results:
        return []
    
    # זיהוי יחידות על בסיס סוג העבודה
    for item in results:
        work_type = item.get('work_type', '').lower()
        if 'ריצוף' in work_type or 'חיפוי' in work_type:
            item['unit'] = 'מ"ר'
        else:
            item['unit'] = "מ'"
    
    return results

def get_payment_invoice_data(plan_id, start_date, end_date, unit_prices=None):
    """
    מכין את כל הנתונים הדרושים לחשבון חלקי
    
    Args:
        plan_id: מזהה התוכנית
        start_date: תאריך התחלה
        end_date: תאריך סיום
        unit_prices: מילון של מחירים לפי סוג עבודה {work_type: price}
    
    Returns:
        מילון עם כל הנתונים לחשבון
    """
    plan = get_plan_by_id(plan_id)
    if not plan:
        return None
    
    # קבלת סיכום הדיווחים
    summary = get_progress_summary_by_date_range(plan_id, start_date, end_date)
    
    if not summary:
        return {
            'plan': plan,
            'start_date': start_date,
            'end_date': end_date,
            'items': [],
            'total_amount': 0,
            'error': 'אין דיווחים בטווח התאריכים הזה'
        }
    
    # אם לא סופקו מחירים, השתמש ברנדומליים
    if not unit_prices:
        unit_prices = {}
    
    # חישוב סכומים
    items = []
    total_amount = 0
    
    for item in summary:
        work_type = item['work_type']
        quantity = item['total_quantity']
        unit = item['unit']
        
        # קבלת מחיר יחידה
        if work_type in unit_prices:
            price = unit_prices[work_type]
        else:
            # מחיר רנדומלי לפי סוג
            if unit == 'מ"ר':
                price = 250  # ריצוף/חיפוי
            elif 'בטון' in work_type.lower():
                price = 1200  # קירות בטון
            elif 'בלוק' in work_type.lower():
                price = 600  # קירות בלוקים
            else:
                price = 800  # ברירת מחדל
        
        subtotal = quantity * price
        total_amount += subtotal
        
        items.append({
            'work_type': work_type,
            'quantity': quantity,
            'unit': unit,
            'unit_price': price,
            'subtotal': subtotal,
            'report_count': item['report_count']
        })
    
    return {
        'plan': plan,
        'start_date': start_date,
        'end_date': end_date,
        'items': items,
        'total_amount': total_amount,
        'vat': total_amount * 0.17,  # מע"מ
        'total_with_vat': total_amount * 1.17
    }

def get_all_work_types_for_plan(plan_id):
    """
    מחזיר רשימה של כל סוגי העבודות שדווחו בפרויקט
    שימושי להגדרת מחירי יחידה
    """
    query = """
        SELECT DISTINCT note as work_type
        FROM progress_reports
        WHERE plan_id = ?
        ORDER BY work_type
    """
    results = run_query(query, (plan_id,))
    return [r['work_type'] for r in results] if results else []
