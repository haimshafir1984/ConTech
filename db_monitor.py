"""
ConTech Pro - Database Monitor
מערכת ניטור למסד נתונים עם התראות וניתוח שימוש
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

from database import get_connection, DB_FILE, DB_URL

# PostgreSQL דרך SQLAlchemy (פותר אזהרות pandas ומונע עומסים)
try:
    import sqlalchemy as sa
except Exception:
    sa = None


# ==========================================
# Engine ל-PostgreSQL (משאב משותף, לא נוצר כל rerun)
# ==========================================


@st.cache_resource
def _get_pg_engine():
    """
    יוצר Engine ל-PostgreSQL פעם אחת וממחזר חיבורים.
    """
    if not DB_URL:
        return None
    if sa is None:
        # אם SQLAlchemy לא מותקן, נחזור לשיטה הישנה (אבל מומלץ להתקין)
        return None

    return sa.create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
    )


def _read_sql_df(query: str) -> pd.DataFrame:
    """
    קורא DataFrame מ-SQL בצורה בטוחה:
    - בפוסטגרס: SQLAlchemy engine
    - ב-SQLite: get_connection() קיים
    """
    if DB_URL:
        engine = _get_pg_engine()
        if engine is not None:
            with engine.connect() as conn:
                return pd.read_sql(query, conn)

        # fallback (לא מומלץ, אבל כדי לא לשבור)
        conn = get_connection()
        try:
            return pd.read_sql(query, conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    else:
        conn = get_connection()
        if not conn:
            raise RuntimeError("לא ניתן להתחבר למסד נתונים")
        try:
            return pd.read_sql(query, conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ==========================================
# פונקציות ניטור (נשמרו באותם שמות)
# ==========================================


@st.cache_data(ttl=60)
def get_db_size():
    """מחזיר את גודל מסד הנתונים ב-MB"""
    try:
        if DB_URL:  # PostgreSQL
            query = "SELECT pg_database_size(current_database()) as size;"
            result = _read_sql_df(query)
            size_bytes = result["size"][0]
            size_mb = size_bytes / (1024 * 1024)
            return size_mb, None
        else:  # SQLite
            if os.path.exists(DB_FILE):
                size_bytes = os.path.getsize(DB_FILE)
                size_mb = size_bytes / (1024 * 1024)
                return size_mb, None
            else:
                return 0, "⚠️ קובץ DB לא קיים"
    except Exception as e:
        return None, f"❌ שגיאה: {str(e)}"


@st.cache_data(ttl=60)
def get_table_stats():
    """מחזיר סטטיסטיקות טבלאות"""
    try:
        if DB_URL:  # PostgreSQL
            query = """
                SELECT 
                    schemaname || '.' || tablename AS table_name,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                    pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """
            df = _read_sql_df(query)
            return df
        else:  # SQLite
            tables = ["plans", "progress_reports"]
            data = []

            for table in tables:
                try:
                    query = f"SELECT COUNT(*) as count FROM {table};"
                    result = _read_sql_df(query)
                    count = result["count"][0]
                    data.append(
                        {
                            "table_name": table,
                            "records": count,
                            "size": "N/A",
                        }
                    )
                except Exception:
                    pass

            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ שגיאה בשאילתה: {str(e)}")
        return None


@st.cache_data(ttl=60)
def get_record_counts():
    """מחזיר מספר רשומות בכל טבלה"""
    counts = {}
    try:
        # Plans
        query = "SELECT COUNT(*) as count FROM plans;"
        result = _read_sql_df(query)
        counts["plans"] = result["count"][0]

        # Progress reports
        query = "SELECT COUNT(*) as count FROM progress_reports;"
        result = _read_sql_df(query)
        counts["reports"] = result["count"][0]

        return counts
    except Exception as e:
        st.error(f"❌ שגיאה: {str(e)}")
        return {}


def get_storage_forecast(current_size_mb, growth_rate_mb_per_month=10):
    """מחשב תחזית שימוש עתידי"""
    max_storage_mb = 1024  # 1 GB
    remaining_mb = max_storage_mb - current_size_mb

    if growth_rate_mb_per_month <= 0:
        return None, remaining_mb

    months_to_full = remaining_mb / growth_rate_mb_per_month
    return months_to_full, remaining_mb


def show_db_monitor():
    """
    מציג דשבורד ניטור מלא למסד נתונים
    """
    st.markdown("## 📊 ניטור מסד נתונים")
    st.caption(f"עדכון אחרון: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ==========================================
    # מידע בסיסי
    # ==========================================

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("### 🔧 מידע טכני")
        if DB_URL:
            st.info("**סוג:** PostgreSQL (ענן)")
            try:
                st.code(f"Host: {DB_URL.split('@')[1].split('/')[0]}")
            except Exception:
                st.code("Host: (לא זמין)")
        else:
            st.info("**סוג:** SQLite (מקומי)")
            st.code(f"Path: {DB_FILE}")

    with col_info2:
        st.markdown("### 💾 קיבולת")
        max_storage = "1 GB"
        max_ram = "256 MB"
        st.metric("Storage מקסימלי", max_storage)
        st.metric("RAM", max_ram)

    st.markdown("---")

    # ==========================================
    # גודל DB
    # ==========================================

    st.markdown("### 💿 שימוש ב-Storage")

    size_mb, error = get_db_size()

    if error:
        st.error(error)
    elif size_mb is not None:

        # חישוב אחוזים
        max_storage_mb = 1024  # 1 GB
        usage_percent = (size_mb / max_storage_mb) * 100
        remaining_mb = max_storage_mb - size_mb

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📊 גודל נוכחי",
                f"{size_mb:.2f} MB",
                delta=f"{usage_percent:.1f}%",
                delta_color="inverse" if usage_percent > 80 else "off",
            )

        with col2:
            st.metric(
                "💾 נותר",
                f"{remaining_mb:.2f} MB",
                delta=f"{100 - usage_percent:.1f}%",
                delta_color="normal" if usage_percent < 80 else "inverse",
            )

        with col3:
            # תחזית
            months_to_full, _ = get_storage_forecast(size_mb)
            if months_to_full:
                st.metric(
                    "📅 חודשים עד מלא",
                    f"{months_to_full:.1f}",
                    delta="בקצב נוכחי" if months_to_full > 6 else "⚠️ מהר!",
                    delta_color="normal" if months_to_full > 6 else "inverse",
                )
            else:
                st.metric("📅 חודשים עד מלא", "N/A")

        with col4:
            counts = get_record_counts()
            total_records = counts.get("plans", 0) + counts.get("reports", 0)
            st.metric('📋 סה"כ רשומות', f"{total_records:,}")

        # Progress bar
        st.markdown("#### שימוש מפורט:")

        # בחירת צבע לפי אחוז
        if usage_percent < 50:
            bar_color = "#10B981"  # ירוק
        elif usage_percent < 80:
            bar_color = "#F59E0B"  # כתום
        else:
            bar_color = "#EF4444"  # אדום

        progress_html = f"""
        <div style="margin: 1rem 0;">
            <div style="width: 100%; background: #e5e7eb; border-radius: 12px; height: 30px; overflow: hidden;">
                <div style="
                    width: {usage_percent}%;
                    background: {bar_color};
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    transition: width 0.5s ease;
                ">
                    {usage_percent:.1f}%
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.875rem; color: #666;">
                <span>0 MB</span>
                <span>{size_mb:.2f} MB / {max_storage_mb} MB</span>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

        # התראות
        if usage_percent > 90:
            st.error("🚨 **התראה קריטית!** מסד הנתונים כמעט מלא!")
            st.warning("**פעולות מומלצות:**")
            st.markdown(
                """
            - 🗑️ מחק פרויקטים ישנים
            - 📦 יצא גיבוי וארכב
            - 💰 שדרג ל-plan גדול יותר (Render: $7/חודש ל-10GB)
            """
            )
        elif usage_percent > 80:
            st.warning("⚠️ **אזהרה:** מסד הנתונים מעל 80% תפוסה")
            st.info("💡 שקול לנקות נתונים ישנים או לשדרג")
        elif usage_percent > 60:
            st.info("ℹ️ מסד הנתונים בשימוש בריא")

    st.markdown("---")

    # ==========================================
    # סטטיסטיקות טבלאות
    # ==========================================

    st.markdown("### 📊 פירוט טבלאות")

    df_tables = get_table_stats()
    if df_tables is not None and not df_tables.empty:
        st.dataframe(df_tables, use_container_width=True, hide_index=True)
    else:
        counts = get_record_counts()
        if counts:
            st.markdown(
                f"""
            - **Plans:** {counts.get('plans', 0):,} רשומות
            - **Progress Reports:** {counts.get('reports', 0):,} רשומות
            """
            )

    st.markdown("---")

    # ==========================================
    # פעולות ניקוי
    # ==========================================

    st.markdown("### 🧹 פעולות תחזוקה")

    col_clean1, col_clean2 = st.columns(2)

    with col_clean1:
        if st.button("🔄 רענן נתונים", use_container_width=True):
            st.rerun()

    with col_clean2:
        if st.button("📥 יצא גיבוי SQL", use_container_width=True):
            st.info("💡 תכונה בפיתוח")

    # אזור מסוכן
    with st.expander("🔴 אזור מסוכן - ניקוי נתונים"):
        st.warning("⚠️ פעולות אלה בלתי הפיכות!")

        col_danger1, col_danger2 = st.columns(2)

        with col_danger1:
            if st.button("🗑️ מחק דיווחים ישנים מעל שנה", type="secondary"):
                st.info("💡 תכונה בפיתוח")

        with col_danger2:
            if st.button("💣 איפוס מלא", type="secondary"):
                st.error("❌ תכונה לא זמינה בממשק זה")
                st.caption("השתמש בדשבורד הראשי")


# ==========================================
# Sidebar Widget (קטן)
# ==========================================


def show_db_widget_sidebar():
    """
    ווידג'ט קטן לסיידבר
    """
    with st.sidebar.expander("💾 מסד נתונים"):
        size_mb, error = get_db_size()

        if error:
            st.error(error)
        elif size_mb is not None:
            max_storage_mb = 1024
            usage_percent = (size_mb / max_storage_mb) * 100

            st.metric(
                "שימוש",
                f"{size_mb:.1f} MB",
                delta=f"{usage_percent:.0f}%",
            )

            # Progress mini
            st.progress(usage_percent / 100.0)

            # כפתור לדשבורד מלא
            if st.button("📊 דשבורד מלא", use_container_width=True):
                st.session_state["show_db_monitor"] = True


# ==========================================
# שימוש
# ==========================================

if __name__ == "__main__":
    # הרצה עצמאית לבדיקה
    st.set_page_config(page_title="DB Monitor", layout="wide")
    show_db_monitor()
