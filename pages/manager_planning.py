import streamlit as st
import json


def _safe_get_metadata(proj):
    meta = proj.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta


def render_manager_planning_tab():
    """
    🧱 הגדרת תכולה לבנייה (שלד UI)
    בשלב הזה: רק מבנה/UX. לוגיקה תתווסף לאחר אישור.
    """

    st.header("🧱 הגדרת תכולה לבנייה")
    st.caption("המנהל מגדיר מה צריך להיבנות; העובד ידווח מה בוצע בפועל.")

    # --- Guard: אין פרויקטים ---
    if "projects" not in st.session_state or not st.session_state.projects:
        st.info("אין תוכניות זמינות. קודם העלה תוכנית ב'📂 סדנת עבודה'.")
        return

    st.markdown("---")

    # ======================================================
    # שלב 0: בחירת תוכנית
    # ======================================================
    st.subheader("שלב 0: בחירת תוכנית")
    plan_name = st.selectbox(
        "בחר תוכנית לעבוד עליה:",
        list(st.session_state.projects.keys()),
        key="planning_plan_select",
    )
    proj = st.session_state.projects[plan_name]
    meta = _safe_get_metadata(proj)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "קיים Crop ROI?",
            "כן" if proj.get("_is_cropped") or proj.get("crop_bbox") else "לא",
        )
    with col_b:
        scale_val = proj.get("scale", 0)
        st.metric("סקייל (px/m)", f"{scale_val:.1f}" if scale_val else "לא הוגדר")
    with col_c:
        planned = meta.get("planned_items_count", 0)
        st.metric("פריטים מתוכננים", f"{planned}")

    st.markdown("---")

    # ======================================================
    # שלב 1: כיול (UI בלבד כרגע)
    # ======================================================
    st.subheader("שלב 1: כיול סקייל")
    with st.expander("📏 כיול (יעבור למנהל כחלק מהשינוי הגדול)", expanded=True):
        st.info(
            "בשלב הזה זה שלד. בהמשך נוסיף כאן את כלי הכיול (קו ידוע + אורך אמיתי) ונשמור ל־DB/metadata."
        )
        st.radio(
            "סטטוס כיול:",
            ["יש סקייל מוגדר", "אין סקייל – נדרש כיול"],
            index=0 if scale_val else 1,
            key="planning_calibration_status",
        )
        st.caption("בהמשך: Canvas לציור קו + הזנת אורך אמיתי + כפתור 'שמור כיול'.")

    st.markdown("---")

    # ======================================================
    # שלב 2: סוג תוכנית
    # ======================================================
    st.subheader("שלב 2: סוג תוכנית")
    detected_type = (
        meta.get("legend_analysis", {}).get("plan_type")
        if isinstance(meta.get("legend_analysis"), dict)
        else None
    )
    if detected_type:
        st.success(f"✅ זוהה אוטומטית (מהמקרא): {detected_type}")

    plan_type = st.selectbox(
        "בחר סוג תוכנית:",
        ["קירות", "ריצוף", "גג"],
        index=0,
        key="planning_plan_type",
        help="בהמשך: נשתמש בזה כדי להפעיל כלים וחישובים מתאימים.",
    )

    st.markdown("---")

    # ======================================================
    # שלב 3: ניהול אזורים (Zones)
    # ======================================================
    st.subheader("שלב 3: חלוקת שרטוט לאזורים (Zones)")
    st.caption(
        "הלקוח ביקש: אפשר לחלק את אותו שרטוט למספר אזורים עם נתונים שונים (חומר/גובה וכו')."
    )

    with st.expander("🗺️ יצירה/ניהול אזורים", expanded=True):
        st.info("שלד UI. בהמשך נוסיף Canvas לאזור (rect/polygon) + שמירה לאזורי תכולה.")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("שם אזור", value="אזור 1", key="zone_name_input")
            st.selectbox(
                "סוג עבודה באזור", ["קירות", "ריצוף", "גג"], key="zone_work_type"
            )
        with col2:
            st.selectbox(
                "חומר ברירת מחדל (קירות)",
                ["בטון", "בלוקים", "גבס"],
                key="zone_default_material",
            )
            st.number_input(
                "גובה ברירת מחדל (מ')", value=2.6, step=0.1, key="zone_default_height"
            )

        st.button("➕ הוסף אזור (שלד)", key="add_zone_stub")

        st.markdown("#### רשימת אזורים (שלד)")
        st.dataframe(
            [
                {
                    "שם": "אזור 1",
                    "סוג": "קירות",
                    "חומר": "בטון",
                    "גובה": 2.6,
                    "פריטים": 0,
                },
                {
                    "שם": "אזור 2",
                    "סוג": "קירות",
                    "חומר": "גבס",
                    "גובה": 2.8,
                    "פריטים": 0,
                },
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ======================================================
    # שלב 4: סימון תכולה בתוך אזור
    # ======================================================
    st.subheader("שלב 4: סימון תכולה (Planned Items)")
    st.caption(
        "כאן נמחזר את ה־Canvas מה־Worker: line / path / rect / polygon + Snap לקירות."
    )

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("### 🎨 אזור ציור")
        st.info(
            "שלד UI. בהמשך נכניס כאן st_canvas עם כל המצבים (line/path/rect/polygon) + Snap."
        )
        st.checkbox("✅ הצמדה לקירות (Snap)", value=True, key="planning_snap_on")
        st.checkbox(
            "✅ תיקון אוטומטי לקו שיצא מהקיר", value=True, key="planning_auto_correct"
        )
        st.caption("בהמשך: תצוגת שרטוט + ציור.")

    with col_right:
        st.markdown("### ⚙️ מאפייני פריט")
        st.selectbox("סוג קיר", ["בטון", "בלוקים", "גבס"], key="planning_wall_type")
        st.number_input("גובה (מ')", value=2.6, step=0.1, key="planning_wall_height")
        st.checkbox(
            "החל על כל הפריטים המסומנים", value=False, key="planning_apply_to_selected"
        )
        st.button("➕ הוסף לתכולה (שלד)", key="planning_add_items_stub")
        st.button(
            "💾 שמור תכולה לשרטוט (שלד)", type="primary", key="planning_save_stub"
        )

    st.markdown("---")

    # ======================================================
    # שלב 5: טבלת פריטים מתוכננים (שלד)
    # ======================================================
    st.subheader("שלב 5: פריטים מתוכננים")
    st.dataframe(
        [
            {
                "#": 1,
                "אזור": "אזור 1",
                "סוג": plan_type,
                "חומר": "בטון",
                "גובה": 2.6,
                "כמות": "12.4 מ'",
                "סטטוס": "מתוכנן",
            },
            {
                "#": 2,
                "אזור": "אזור 2",
                "סוג": plan_type,
                "חומר": "גבס",
                "גובה": 2.8,
                "כמות": "8.1 מ'",
                "סטטוס": "מתוכנן",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ======================================================
    # שלב 6: סיכום + נעילה (שלד)
    # ======================================================
    st.subheader("שלב 6: סיכום")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("סה״כ קירות בטון", "12.4 מ׳")
    with k2:
        st.metric("סה״כ קירות גבס", "8.1 מ׳")
    with k3:
        st.metric("סה״כ פריטים", "2")

    st.button("✅ נעל תכולה והעבר לדיווח שטח (שלד)", key="planning_lock_stub")
    st.caption(
        "בהמשך: נציג ב־Worker overlay של התכולה + התאמת סימון ביצוע לפריטים מתוכננים."
    )
