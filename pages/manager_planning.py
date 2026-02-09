import streamlit as st
import json
import io
import traceback
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from pages.worker import compute_line_length_px, px_to_m  # שימוש 1:1 כמו Worker
import cv2


def _build_bg_image_from_proj(proj, max_width=900):
    """יוצר background_image יציב ל-st_canvas + ממדי תצוגה + scale_factor"""
    img0 = proj.get("original")
    if img0 is None:
        raise ValueError("proj['original'] missing")

    if not isinstance(img0, np.ndarray):
        raise ValueError(f"proj['original'] must be np.ndarray, got {type(img0)}")

    # אם זה BGR (ברוב המקרים אצלך), נהפוך ל-RGB
    if img0.ndim == 3 and img0.shape[2] == 3:
        rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    elif img0.ndim == 2:
        rgb = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img0.shape}")

    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image size")

    scale_factor = min(1.0, float(max_width) / float(w))
    disp_w = max(1, int(w * scale_factor))
    disp_h = max(1, int(h * scale_factor))

    pil = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    # טריק יציב כמו Worker: BytesIO + reopen + load()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    bg = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    bg.load()

    return bg, disp_w, disp_h, scale_factor


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
    🧱 הגדרת תכולה לבנייה
    בשלב הזה: מעבירים 1:1 את כיול + ציור מה-Worker (לוגיקה בסיסית).
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
    # שלב 1: כיול סקייל (הועתק מה-Worker 1:1)
    # ======================================================
    st.subheader("שלב 1: כיול סקייל")

    is_fallback = proj.get("_scale_is_fallback", False) or (
        proj.get("scale", 0) in (0, None)
    )

    try:
        bg_img, disp_w, disp_h, scale_factor = _build_bg_image_from_proj(
            proj, max_width=900
        )
    except Exception:
        st.error("❌ לא הצלחתי להכין תמונת רקע לכיול (בדוק proj['original']).")
        st.code(traceback.format_exc())
        bg_img, disp_w, disp_h, scale_factor = None, 0, 0, 1.0

    with st.expander("📏 כיול סקלה (חשוב לדיוק!)", expanded=True):
        st.caption(
            "צייר קו על אלמנט עם אורך ידוע בתוכנית → הזן אורך אמיתי → הסקייל יחושב."
        )

        if bg_img is None:
            st.info("אין תמונה זמינה לכיול כרגע.")
        else:
            cal_canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_color="#FF00FF",
                stroke_width=3,
                background_image=bg_img,
                height=disp_h,
                width=disp_w,
                drawing_mode="line",
                key=f"planning_cal_canvas_{plan_name}",
                update_streamlit=True,
            )

            cal_px = 0.0
            if cal_canvas.json_data and cal_canvas.json_data.get("objects"):
                cal_lines = [
                    o
                    for o in cal_canvas.json_data["objects"]
                    if o.get("type") == "line"
                ]
                if cal_lines:
                    cal_px = compute_line_length_px(cal_lines[-1])

            if cal_px > 0:
                st.info(f"📐 אורך הקו שציירת: {cal_px:.0f} px (על הקנבס)")
                col_real, col_btn = st.columns([2, 1])
                with col_real:
                    real_length_m = st.number_input(
                        "אורך אמיתי (מטר):",
                        value=1.0,
                        min_value=0.1,
                        max_value=100.0,
                        step=0.5,
                        key=f"planning_cal_real_{plan_name}",
                    )
                with col_btn:
                    st.write("")
                    if st.button(
                        "✅ תקן סקלה",
                        type="primary",
                        use_container_width=True,
                        key=f"planning_fix_scale_{plan_name}",
                    ):
                        cal_px_original = cal_px / scale_factor
                        proj["scale"] = cal_px_original / real_length_m
                        st.success(f"✅ סקייל עודכן: {proj['scale']:.1f} px/m")
                        st.rerun()
            else:
                st.info("👆 צייר קו ישר על אלמנט עם אורך ידוע")
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
    # storage per plan
    zones_key = f"planning_zones_{plan_name}"
    if zones_key not in st.session_state:
        st.session_state[zones_key] = []

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

        if st.button("➕ הוסף אזור", key=f"add_zone_{plan_name}"):
            st.session_state[zones_key].append(
                {
                    "שם": st.session_state.zone_name_input,
                    "סוג": st.session_state.zone_work_type,
                    "חומר": st.session_state.zone_default_material,
                    "גובה": st.session_state.zone_default_height,
                    "פריטים": 0,
                }
            )
            st.toast("✅ אזור נוסף")
            st.rerun()

        st.markdown("#### רשימת אזורים")
        st.dataframe(
            st.session_state[zones_key], use_container_width=True, hide_index=True
        )
    st.markdown("---")

    # ======================================================
    # שלב 4: סימון תכולה בתוך אזור
    # ======================================================
    st.subheader("שלב 4: סימון תכולה (Planned Items)")
    st.caption(
        "כאן נמחזר את ה־Canvas מה־Worker: line / freedraw / rect / polygon + Snap לקירות."
    )

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("### 🎨 אזור ציור")

        drawing_mode = st.selectbox(
            "מצב ציור:",
            ["line", "freedraw", "rect", "polygon"],
            index=0,
            key=f"planning_drawing_mode_{plan_name}",
        )

        fill = "rgba(0,0,0,0)"
        stroke = "#00FF00"
        stroke_width = 6

        try:
            bg_img2, disp_w2, disp_h2, scale_factor2 = _build_bg_image_from_proj(
                proj, max_width=900
            )
            canvas = st_canvas(
                fill_color=fill,
                stroke_color=stroke,
                stroke_width=stroke_width,
                background_image=bg_img2,
                height=disp_h2,
                width=disp_w2,
                drawing_mode=drawing_mode,
                key=f"planning_canvas_{plan_name}_{disp_w2}x{disp_h2}_{drawing_mode}",
                update_streamlit=True,
            )
        except Exception:
            st.error("❌ לא הצלחתי להכין רקע לקנבס (סימון תכולה).")
            st.code(traceback.format_exc())
            canvas = None

        st.checkbox("✅ הצמדה לקירות (Snap)", value=True, key="planning_snap_on")
        st.checkbox(
            "✅ תיקון אוטומטי לקו שיצא מהקיר", value=True, key="planning_auto_correct"
        )

    with col_right:
        st.markdown("### ⚙️ מאפייני פריט")
        st.selectbox("סוג קיר", ["בטון", "בלוקים", "גבס"], key="planning_wall_type")
        st.number_input("גובה (מ')", value=2.6, step=0.1, key="planning_wall_height")
        st.checkbox(
            "החל על כל הפריטים המסומנים", value=False, key="planning_apply_to_selected"
        )
        st.button("➕ הוסף לתכולה (שלד)", key="planning_add_items_stub")

        if st.button(
            "💾 שמור תכולה לשרטוט", type="primary", key=f"planning_save_{plan_name}"
        ):
            meta = _safe_get_metadata(proj)
            meta.setdefault("planning", {})
            meta["planning"]["canvas_json"] = canvas.json_data if canvas else None
            meta["planning"]["plan_type"] = st.session_state.get(
                "planning_plan_type", "קירות"
            )
            meta["planned_items_count"] = (
                len((canvas.json_data or {}).get("objects", [])) if canvas else 0
            )
            proj["metadata"] = meta
            st.success(
                "✅ נשמר! (כרגע רק JSON של הקנבס; בהמשך נחשב כמויות ונבנה פריטים אמיתיים)"
            )
            st.rerun()

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
