import sqlite3
import json
from datetime import datetime

DB_NAME = "contech.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        plan_name TEXT,
        extracted_scale TEXT,
        confirmed_scale REAL,
        raw_pixel_count INTEGER,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        target_date TEXT,
        budget_limit REAL DEFAULT 0,
        cost_per_meter REAL DEFAULT 0,
        material_estimate TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS progress_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id INTEGER,
        report_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        meters_built REAL,
        worker_name TEXT,
        note TEXT,
        FOREIGN KEY(plan_id) REFERENCES plans(id)
    )''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def save_plan(filename, plan_name, extracted_scale, confirmed_scale, raw_pixel_count, metadata_json, target_date=None, budget_limit=0, cost_per_meter=0, material_estimate="{}"):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''INSERT OR REPLACE INTO plans 
            (filename, plan_name, extracted_scale, confirmed_scale, raw_pixel_count, metadata_json, target_date, budget_limit, cost_per_meter, material_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (filename, plan_name, extracted_scale, confirmed_scale, raw_pixel_count, metadata_json, target_date, budget_limit, cost_per_meter, material_estimate))
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()

def save_progress_report(plan_id, meters, note=""):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO progress_reports (plan_id, meters_built, note) VALUES (?, ?, ?)",
                  (plan_id, meters, note))
        conn.commit()
    finally:
        conn.close()

def get_all_plans():
    conn = get_db_connection()
    plans = conn.execute("SELECT * FROM plans ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(p) for p in plans]

def get_plan_by_filename(filename):
    conn = get_db_connection()
    plan = conn.execute("SELECT * FROM plans WHERE filename = ?", (filename,)).fetchone()
    conn.close()
    return dict(plan) if plan else None

def get_plan_by_id(plan_id):
    conn = get_db_connection()
    plan = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    conn.close()
    return dict(plan) if plan else None

def get_progress_reports(plan_id=None):
    conn = get_db_connection()
    query = """
        SELECT r.*, p.plan_name 
        FROM progress_reports r
        JOIN plans p ON r.plan_id = p.id
    """
    params = []
    if plan_id:
        query += " WHERE r.plan_id = ?"
        params.append(plan_id)
    
    query += " ORDER BY r.report_date DESC"
    
    reports = conn.execute(query, params).fetchall()
    conn.close()
    
    results = []
    for r in reports:
        row = dict(r)
        try:
            dt = datetime.fromisoformat(row['report_date'])
        except:
            try:
                dt = datetime.strptime(row['report_date'], "%Y-%m-%d %H:%M:%S")
            except:
                dt = row['report_date']
        
        if isinstance(dt, datetime):
            row['date'] = dt.strftime("%d/%m/%Y %H:%M")
        else:
            row['date'] = str(dt)
            
        results.append(row)
    return results

def calculate_material_estimates(total_length_meters, wall_height_meters=2.5):
    total_area = total_length_meters * wall_height_meters
    blocks_per_sqm = 10 
    blocks = total_area * blocks_per_sqm * 1.05
    mortar_volume = total_area * 0.02
    cement = mortar_volume * 0.3
    sand = mortar_volume * 0.7
    
    return {
        "wall_area_sqm": total_area,
        "block_count": int(blocks),
        "cement_cubic_meters": cement,
        "sand_cubic_meters": sand
    }

def get_project_forecast(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan: return {}
    
    reports = get_progress_reports(plan_id)
    
    try: confirmed_scale = float(plan.get('confirmed_scale', 0))
    except: confirmed_scale = 0.0
        
    try: raw_pixels = float(plan.get('raw_pixel_count', 0))
    except: raw_pixels = 0.0
    
    total_planned_meters = 0
    if confirmed_scale > 0:
        total_planned_meters = raw_pixels / confirmed_scale
        
    cumulative = sum([float(r['meters_built']) for r in reports])
    
    days_passed = 0
    velocity = 0
    if reports:
        try:
            first = reports[-1]['report_date']
            last = reports[0]['report_date']
            # ניקוי פורמט תאריך למקרה שיש מילישניות
            d1_str = str(first).split('.')[0]
            d2_str = str(last).split('.')[0]
            
            d1 = datetime.strptime(d1_str, "%Y-%m-%d %H:%M:%S")
            d2 = datetime.strptime(d2_str, "%Y-%m-%d %H:%M:%S")
            delta = (d2 - d1).days
            days_passed = delta if delta > 0 else 1
            velocity = cumulative / days_passed
        except:
            days_passed = 1
            velocity = cumulative
            
    remaining = total_planned_meters - cumulative
    days_to_finish = (remaining / velocity) if velocity > 0 else -1
    
    return {
        "total_planned": total_planned_meters,
        "cumulative_progress": cumulative,
        "remaining_work": max(0, remaining),
        "average_velocity": velocity,
        "days_to_finish": int(days_to_finish) # מחזיר תמיד מספר (-1 אם לא ידוע)
    }

def get_project_financial_status(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan: return {}
    
    reports = get_progress_reports(plan_id)
    cumulative_meters = sum([float(r['meters_built']) for r in reports])
    
    try:
        cost_per_meter = float(plan.get('cost_per_meter', 0))
        budget_limit = float(plan.get('budget_limit', 0))
    except:
        cost_per_meter = 0
        budget_limit = 0
        
    current_cost = cumulative_meters * cost_per_meter
    variance = budget_limit - current_cost
    
    return {
        "budget_limit": budget_limit,
        "current_cost": current_cost,
        "budget_variance": variance
    }

def reset_all_data():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM progress_reports")
    c.execute("DELETE FROM plans")
    c.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    conn.close()
    return True