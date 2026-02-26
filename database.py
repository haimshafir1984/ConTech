import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

# בדיקה האם אנחנו בענן (Postgres) או מקומי (SQLite)
_raw_db_url = os.environ.get("DATABASE_URL", "")
# קבל רק URL של Postgres אמיתי (מתחיל ב-postgres:// או postgresql://)
# URL של Prisma כגון "file:./prisma/dev.db" אינו Postgres — נתעלם ממנו
_is_real_postgres = _raw_db_url.startswith(("postgres://", "postgresql://"))
DB_URL = _raw_db_url if _is_real_postgres else ""
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
        # מצב מקומי - שימוש בקובץ SQLite
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # מאפשר גישה לשדות לפי שם
        return conn


def init_database():
    """יצירת הטבלאות אם לא קיימות"""
    conn = get_connection()
    if not conn:
        return

    # שאילתות יצירה - מותאמות לשני סוגי המסדים
    if DB_URL:
        # Syntax for PostgreSQL
        id_type = "SERIAL PRIMARY KEY"
        json_type = "TEXT"  # פשטות
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
        );""",
    ]

    try:
        cur = conn.cursor()
        for q in queries:
            cur.execute(q)
        conn.commit()
    except Exception as e:
        print(f"DB Init Error: {e}")

    # ── הוסף עמודות חסרות לטבלה קיימת (migration בטוחה) ──
    # תומך בשני סוגי DB: SQLite (PRAGMA) ו-Postgres (information_schema)
    _missing_cols_postgres = [
        ("materials_json", "TEXT"),
        ("img_original", "BYTEA"),
        ("img_thick_walls", "BYTEA"),
        ("img_flooring_mask", "BYTEA"),
        ("img_skeleton", "BYTEA"),
        ("img_concrete_mask", "BYTEA"),
        ("img_blocks_mask", "BYTEA"),
    ]
    _missing_cols_sqlite = [
        ("materials_json", "TEXT"),
        ("img_original", "BLOB"),
        ("img_thick_walls", "BLOB"),
        ("img_flooring_mask", "BLOB"),
        ("img_skeleton", "BLOB"),
        ("img_concrete_mask", "BLOB"),
        ("img_blocks_mask", "BLOB"),
    ]
    try:
        conn2 = get_connection()
        if conn2:
            cur2 = conn2.cursor()
            if DB_URL:
                cur2.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='plans'"
                )
                existing_cols = {row["column_name"] for row in cur2.fetchall()}
                for col, col_type in _missing_cols_postgres:
                    if col not in existing_cols:
                        cur2.execute(f"ALTER TABLE plans ADD COLUMN IF NOT EXISTS {col} {col_type}")
            else:
                cur2.execute("PRAGMA table_info(plans)")
                existing_cols = {row[1] for row in cur2.fetchall()}
                for col, col_type in _missing_cols_sqlite:
                    if col not in existing_cols:
                        cur2.execute(f"ALTER TABLE plans ADD COLUMN {col} {col_type}")
            conn2.commit()
            conn2.close()
    except Exception as e:
        print(f"DB migration warning: {e}")
    finally:
        conn.close()


# --- פונקציות עזר לשאילתות (Wrapper) ---
def run_query(query, params=(), fetch="all"):
    conn = get_connection()
    if not conn:
        return None

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
                    return cur.fetchone()["id"]
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


def save_plan(
    filename,
    plan_name,
    scale_text,
    scale_val,
    pixels,
    metadata,
    target_date=None,
    budget=0,
    cost=0,
    materials="{}",
):
    """
    שומר / מעדכן תוכנית ב-DB.
    תומך בשתי סכמות:
      - סכמה חדשה: extracted_scale, confirmed_scale, raw_pixel_count, metadata_json, material_estimate
      - סכמה ישנה: scale_text, scale_value, raw_pixels, metadata, materials_json
    """
    # זהה איזו סכמה קיימת
    conn_check = get_connection()
    if conn_check:
        try:
            cur_check = conn_check.cursor()
            if _is_real_postgres:
                # Postgres: use information_schema instead of PRAGMA
                cur_check.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='plans'"
                )
                cols = {row["column_name"] for row in cur_check.fetchall()}
            else:
                cur_check.execute("PRAGMA table_info(plans)")
                cols = {row[1] for row in cur_check.fetchall()}
        except Exception:
            cols = set()
        finally:
            conn_check.close()
    else:
        cols = set()

    # אם Postgres ו-cols ריק (שגיאה בשאילתה) — נניח סכמה ישנה כברירת מחדל
    if _is_real_postgres and not cols:
        cols = {"metadata", "scale_text", "scale_value", "raw_pixels"}

    # new_schema requires BOTH metadata_json AND extracted_scale columns to exist
    # (some DBs have metadata_json but not extracted_scale — don't treat as new schema)
    new_schema = "metadata_json" in cols and "extracted_scale" in cols
    old_schema = "metadata" in cols       # סכמה ישנה (database.py init)

    existing = get_plan_by_filename(filename)

    if existing:
        # ── UPDATE ──
        rid = existing["id"]
        if new_schema:
            query = """UPDATE plans SET plan_name=?, extracted_scale=?, confirmed_scale=?,
                       raw_pixel_count=?, metadata_json=?, target_date=?, budget_limit=?,
                       cost_per_meter=?, material_estimate=?
                       WHERE id=?"""
            run_query(query, (plan_name, scale_text, scale_val, pixels, metadata,
                               target_date, budget, cost, materials, rid), fetch="commit")
        elif old_schema:
            query = """UPDATE plans SET plan_name=?, scale_text=?, scale_value=?, raw_pixels=?,
                       metadata=?, target_date=?, budget_limit=?, cost_per_meter=?, materials_json=?
                       WHERE id=?"""
            run_query(query, (plan_name, scale_text, scale_val, pixels, metadata,
                               target_date, budget, cost, materials, rid), fetch="commit")
        return rid
    else:
        # ── INSERT ──
        if new_schema:
            query = """INSERT INTO plans (filename, plan_name, extracted_scale, confirmed_scale,
                       raw_pixel_count, metadata_json, target_date, budget_limit, cost_per_meter,
                       material_estimate) VALUES (?,?,?,?,?,?,?,?,?,?)"""
        elif old_schema:
            query = """INSERT INTO plans (filename, plan_name, scale_text, scale_value, raw_pixels,
                       metadata, target_date, budget_limit, cost_per_meter, materials_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?)"""
        else:
            # סכמה לא מזוהה — צור עמודות מינימליות
            query = """INSERT INTO plans (filename, plan_name) VALUES (?,?)"""
            return run_query(query, (filename, plan_name), fetch="insert")

        params = (filename, plan_name, scale_text, scale_val, pixels, metadata,
                  target_date, budget, cost, materials)

        if DB_URL:
            conn = get_connection()
            if conn is None:
                return None
            try:
                cur = conn.cursor()
                cur.execute(query.replace("?", "%s") + " RETURNING id", params)
                new_id = cur.fetchone()["id"]
                conn.commit()
                return new_id
            finally:
                conn.close()
        else:
            return run_query(query, params, fetch="insert")


def update_plan_metadata(plan_id, metadata_json_str):
    """עדכון metadata של תוכנית — תומך בשתי הסכמות."""
    # זהה איזו עמודה קיימת — ברירת מחדל: "metadata" (סכמה ישנה)
    conn_check = get_connection()
    meta_col = "metadata"
    if conn_check:
        try:
            cur_check = conn_check.cursor()
            if _is_real_postgres:
                cur_check.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='plans'"
                )
                cols = {row["column_name"] for row in cur_check.fetchall()}
            else:
                cur_check.execute("PRAGMA table_info(plans)")
                cols = {row[1] for row in cur_check.fetchall()}
            # Use metadata_json only if the new schema is fully present
            if "metadata_json" in cols and "extracted_scale" in cols:
                meta_col = "metadata_json"
        except Exception:
            pass
        finally:
            conn_check.close()

    run_query(f"UPDATE plans SET {meta_col}=? WHERE id=?",
              (metadata_json_str, plan_id), fetch="commit")
    return True


def save_progress_report(plan_id, meters, note):
    query = (
        "INSERT INTO progress_reports (plan_id, meters_built, note) VALUES (?, ?, ?)"
    )
    run_query(query, (plan_id, meters, note), fetch="insert")


def _normalize_plan_row(row):
    """
    ממפה שמות עמודות בין הסכמה הישנה לחדשה כך שהקוד תמיד רואה שמות אחידים.
    סכמה חדשה (project_data.db): extracted_scale, confirmed_scale, raw_pixel_count, metadata_json, material_estimate
    סכמה ישנה (init_database):   scale_text, scale_value, raw_pixels, metadata, materials_json
    """
    if not row:
        return row
    d = dict(row)
    # מיפוי חדש→ישן (הוסף aliases כדי שהקוד הישן ימשיך לעבוד)
    if "extracted_scale" in d and "scale_text" not in d:
        d["scale_text"] = d["extracted_scale"]
    if "confirmed_scale" in d and "scale_value" not in d:
        d["scale_value"] = d["confirmed_scale"]
    if "raw_pixel_count" in d and "raw_pixels" not in d:
        d["raw_pixels"] = d["raw_pixel_count"]
    if "metadata_json" in d and "metadata" not in d:
        d["metadata"] = d["metadata_json"]
    if "material_estimate" in d and "materials_json" not in d:
        # material_estimate column was created as NUMERIC — coerce to string for JSON parsing
        _me = d["material_estimate"]
        d["materials_json"] = _me if isinstance(_me, str) else None
    # מיפוי ישן→חדש
    if "scale_text" in d and "extracted_scale" not in d:
        d["extracted_scale"] = d["scale_text"]
    if "scale_value" in d and "confirmed_scale" not in d:
        d["confirmed_scale"] = d["scale_value"]
    if "raw_pixels" in d and "raw_pixel_count" not in d:
        d["raw_pixel_count"] = d["raw_pixels"]
    if "metadata" in d and "metadata_json" not in d:
        d["metadata_json"] = d["metadata"]
    if "materials_json" in d and "material_estimate" not in d:
        d["material_estimate"] = d["materials_json"]
    return d


def get_all_plans():
    rows = run_query("SELECT * FROM plans ORDER BY created_at DESC")
    return [_normalize_plan_row(r) for r in rows] if rows else []


def get_plan_by_filename(filename):
    row = run_query("SELECT * FROM plans WHERE filename = ?", (filename,), fetch="one")
    return _normalize_plan_row(row)


def get_plan_by_id(pid):
    row = run_query("SELECT * FROM plans WHERE id = ?", (pid,), fetch="one")
    return _normalize_plan_row(row)


def get_progress_reports(plan_id=None):
    if plan_id:
        return run_query(
            "SELECT r.*, p.plan_name FROM progress_reports r JOIN plans p ON r.plan_id = p.id WHERE r.plan_id = ? ORDER BY r.date DESC",
            (plan_id,),
        )
    else:
        return run_query(
            "SELECT r.*, p.plan_name FROM progress_reports r JOIN plans p ON r.plan_id = p.id ORDER BY r.date DESC"
        )


def reset_all_data():
    run_query("DELETE FROM progress_reports", fetch="commit")
    run_query("DELETE FROM plans", fetch="commit")
    return True


# --- פונקציות חישוב (Business Logic) ---
def get_project_forecast(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan:
        return {}

    reports = get_progress_reports(plan_id)
    total_built = sum([r["meters_built"] for r in reports]) if reports else 0

    # --- תיקון קריטי: חישוב אורך מתוכנן ---
    # הנוסחה הנכונה: פיקסלים / (פיקסלים למטר) = מטרים
    if plan["scale_value"] and plan["scale_value"] > 0:
        total_len_meters = plan["raw_pixels"] / plan["scale_value"]
    else:
        total_len_meters = 0

    # נשמור גם את הערך השגוי הישן כ"planned" למקרה שמישהו מסתמך עליו, אבל נשתמש בחדש
    total_planned = total_len_meters

    # חישוב ימים
    days_passed = 0
    velocity = 0
    if reports:
        dates = [datetime.strptime(str(r["date"])[:10], "%Y-%m-%d") for r in reports]
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
        "days_to_finish": int(days_to_finish),
    }


def get_project_financial_status(plan_id):
    plan = get_plan_by_id(plan_id)
    if not plan:
        return {}

    reports = get_progress_reports(plan_id)
    total_built = sum([r["meters_built"] for r in reports]) if reports else 0

    cost_per_m = plan["cost_per_meter"] if plan["cost_per_meter"] else 0
    current_cost = total_built * cost_per_m
    budget = plan["budget_limit"] if plan["budget_limit"] else 0

    return {
        "current_cost": current_cost,
        "budget_limit": budget,
        "budget_variance": budget - current_cost,
    }


def calculate_material_estimates(total_length_meters, height_meters=2.5):
    # הערכה גסה: 10 בלוקים למ"ר, עובי 20 ס"מ
    wall_area = total_length_meters * height_meters
    blocks = wall_area * 10
    cement = wall_area * 0.02  # הערכה: 0.02 קוב למ"ר

    return {
        "block_count": int(blocks),
        "cement_cubic_meters": round(cement, 1),
        "wall_area_sqm": round(wall_area, 1),
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
        AND DATE(date) >= DATE(?)  -- רק תאריך!
        AND DATE(date) <= DATE(?)  -- רק תאריך!
        GROUP BY note
        ORDER BY total_quantity DESC
    """

    results = run_query(query, (plan_id, start_date, end_date))

    if not results:
        return []

    # זיהוי יחידות על בסיס סוג העבודה
    for item in results:
        work_type = item.get("work_type", "").lower()
        if "ריצוף" in work_type or "חיפוי" in work_type:
            item["unit"] = 'מ"ר'
        else:
            item["unit"] = "מ'"

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
            "plan": plan,
            "start_date": start_date,
            "end_date": end_date,
            "items": [],
            "total_amount": 0,
            "error": "אין דיווחים בטווח התאריכים הזה",
        }

    # אם לא סופקו מחירים, השתמש ברנדומליים
    if not unit_prices:
        unit_prices = {}

    # חישוב סכומים
    items = []
    total_amount = 0

    for item in summary:
        work_type = item["work_type"] or "בניית קירות"
        quantity = item["total_quantity"]
        unit = item["unit"]

        # קבלת מחיר יחידה
        if work_type in unit_prices:
            price = unit_prices[work_type]
        else:
            # מחיר רנדומלי לפי סוג
            if unit == 'מ"ר':
                price = 250  # ריצוף/חיפוי
            elif "בטון" in work_type.lower():
                price = 1200  # קירות בטון
            elif "בלוק" in work_type.lower():
                price = 600  # קירות בלוקים
            else:
                price = 800  # ברירת מחדל

        subtotal = quantity * price
        total_amount += subtotal

        items.append(
            {
                "work_type": work_type,
                "quantity": quantity,
                "unit": unit,
                "unit_price": price,
                "subtotal": subtotal,
                "report_count": item["report_count"],
            }
        )

    return {
        "plan": plan,
        "start_date": start_date,
        "end_date": end_date,
        "items": items,
        "total_amount": total_amount,
        "vat": total_amount * 0.17,  # מע"מ
        "total_with_vat": total_amount * 1.17,
    }


def save_plan_images(
    filename: str,
    original_jpg: bytes,
    thick_walls_png: bytes,
    flooring_mask_png: bytes = b"",
    skeleton_png: bytes = b"",
    concrete_mask_png: bytes = b"",
    blocks_mask_png: bytes = b"",
) -> bool:
    """
    שומר את כל ה-masks של תוכנית כ-BLOB ב-DB (שרידות בין restarts ו-workers).
    נקרא אחרי _persist_plan_to_database בזמן upload.
    """
    conn = get_connection()
    if not conn:
        return False

    # וודא שהעמודות קיימות (migration בטוחה)
    _blob_cols_pg = [
        ("img_original", "BYTEA"), ("img_thick_walls", "BYTEA"),
        ("img_flooring_mask", "BYTEA"), ("img_skeleton", "BYTEA"),
        ("img_concrete_mask", "BYTEA"), ("img_blocks_mask", "BYTEA"),
    ]
    try:
        cur = conn.cursor()
        if DB_URL:
            for col, col_type in _blob_cols_pg:
                cur.execute(f"ALTER TABLE plans ADD COLUMN IF NOT EXISTS {col} {col_type};")
        else:
            cur.execute("PRAGMA table_info(plans)")
            existing = {row[1] for row in cur.fetchall()}
            for col, _ in _blob_cols_pg:
                if col not in existing:
                    cur.execute(f"ALTER TABLE plans ADD COLUMN {col} BLOB")
        conn.commit()
    except Exception as e:
        print(f"[DB] add blob columns warning: {e}")

    try:
        cur = conn.cursor()
        if DB_URL:
            cur.execute(
                """UPDATE plans SET
                   img_original=%s, img_thick_walls=%s,
                   img_flooring_mask=%s, img_skeleton=%s,
                   img_concrete_mask=%s, img_blocks_mask=%s
                   WHERE filename=%s""",
                (original_jpg or None, thick_walls_png or None,
                 flooring_mask_png or None, skeleton_png or None,
                 concrete_mask_png or None, blocks_mask_png or None,
                 filename)
            )
        else:
            cur.execute(
                """UPDATE plans SET
                   img_original=?, img_thick_walls=?,
                   img_flooring_mask=?, img_skeleton=?,
                   img_concrete_mask=?, img_blocks_mask=?
                   WHERE filename=?""",
                (original_jpg or None, thick_walls_png or None,
                 flooring_mask_png or None, skeleton_png or None,
                 concrete_mask_png or None, blocks_mask_png or None,
                 filename)
            )
        conn.commit()
        saved = cur.rowcount > 0
        print(f"[DB-BLOB] save_plan_images: filename={filename} rowcount={cur.rowcount} "
              f"orig={len(original_jpg or b'')}B walls={len(thick_walls_png or b'')}B "
              f"flooring={len(flooring_mask_png or b'')}B")
        return saved
    except Exception as e:
        print(f"[DB] save_plan_images error: {e}")
        return False
    finally:
        conn.close()


def load_plan_images(filename: str):
    """
    טוען את כל ה-masks מה-DB.
    מחזיר tuple של 6: (original, thick_walls, flooring_mask, skeleton, concrete_mask, blocks_mask).
    כל ערך הוא bytes או None.
    """
    conn = get_connection()
    if not conn:
        return None, None, None, None, None, None

    # וודא שכל עמודות ה-BLOB קיימות לפני ה-SELECT (migration בטוחה)
    _blob_cols_pg = [
        ("img_original", "BYTEA"), ("img_thick_walls", "BYTEA"),
        ("img_flooring_mask", "BYTEA"), ("img_skeleton", "BYTEA"),
        ("img_concrete_mask", "BYTEA"), ("img_blocks_mask", "BYTEA"),
    ]
    try:
        cur = conn.cursor()
        if DB_URL:
            for col, col_type in _blob_cols_pg:
                cur.execute(f"ALTER TABLE plans ADD COLUMN IF NOT EXISTS {col} {col_type};")
        else:
            cur.execute("PRAGMA table_info(plans)")
            existing = {row[1] for row in cur.fetchall()}
            for col, _ in _blob_cols_pg:
                if col not in existing:
                    cur.execute(f"ALTER TABLE plans ADD COLUMN {col} BLOB")
        conn.commit()
    except Exception as e:
        print(f"[DB] load_plan_images add-columns warning: {e}")

    try:
        cur = conn.cursor()
        query = """SELECT img_original, img_thick_walls,
                          img_flooring_mask, img_skeleton,
                          img_concrete_mask, img_blocks_mask
                   FROM plans WHERE filename={ph}""".format(ph="%s" if DB_URL else "?")
        cur.execute(query, (filename,))
        row = cur.fetchone()
        if not row:
            return None, None, None, None, None, None

        def _b(v):
            return bytes(v) if v else None

        if DB_URL:
            return (_b(row["img_original"]), _b(row["img_thick_walls"]),
                    _b(row["img_flooring_mask"]), _b(row["img_skeleton"]),
                    _b(row["img_concrete_mask"]), _b(row["img_blocks_mask"]))
        else:
            return (_b(row[0]), _b(row[1]), _b(row[2]), _b(row[3]), _b(row[4]), _b(row[5]))
    except Exception as e:
        print(f"[DB] load_plan_images error: {e}")
        return None, None, None, None, None, None
    finally:
        conn.close()


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
    return [r["work_type"] for r in results] if results else []


def upsert_document_profile(profile):
    """
    יוצר או מחזיר document_profile לפי filename + aspect_ratio
    """
    query = """
        INSERT INTO document_profiles
        (project_name, filename, page_width_px, page_height_px,
         aspect_ratio, has_grid, avg_wall_thickness,
         detected_scale, final_scale, signature)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
    """

    return run_query(
        query,
        (
            profile.get("project_name"),
            profile.get("filename"),
            profile.get("page_width_px"),
            profile.get("page_height_px"),
            profile.get("aspect_ratio"),
            profile.get("has_grid"),
            profile.get("avg_wall_thickness"),
            profile.get("detected_scale"),
            profile.get("final_scale"),
            json.dumps(profile.get("signature"), ensure_ascii=False),
        ),
        fetch="insert",
    )


def insert_learning_event(profile_id, event_type, payload, confidence=1.0):
    query = """
        INSERT INTO learning_events
        (profile_id, event_type, payload, confidence)
        VALUES (?, ?, ?, ?)
    """
    run_query(
        query,
        (profile_id, event_type, json.dumps(payload), confidence),
        fetch="commit",
    )


def get_learning_overrides_by_aspect(aspect_ratio):
    query = """
        SELECT le.event_type, le.payload, le.confidence
        FROM document_profiles dp
        JOIN learning_events le ON dp.id = le.profile_id
        WHERE ABS(dp.aspect_ratio - ?) < 0.05
        ORDER BY le.created_at DESC
        LIMIT 20
    """
    return run_query(query, (aspect_ratio,))
