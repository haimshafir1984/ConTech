"""
ConTech Pro - Manager Planning Tab
לשונית הגדרת תכולה מתוכננת (Planned Items + BOQ)
"""

import streamlit as st
import json
import io
import traceback
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
from datetime import datetime

# ייבוא פונקציות קיימות מ-worker
try:
    from .worker import (
        compute_line_length_px,
        compute_rect_area_px,
        px_to_m,
        px2_to_m2,
        get_scale_with_fallback,
    )

    WORKER_FUNCS_AVAILABLE = True
except ImportError:
    WORKER_FUNCS_AVAILABLE = False


# ==========================================
# פונקציות עזר
# ==========================================


def _safe_get_metadata(proj):
    """מחזיר metadata בצורה בטוחה"""
    meta = proj.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta


def _build_bg_image_from_proj(proj, max_width=900):
    """
    יוצר background_image ל-st_canvas + ממדים + scale_factor
    """
    img0 = proj.get("original")
    if img0 is None:
        raise ValueError("proj['original'] missing")

    if not isinstance(img0, np.ndarray):
        raise ValueError(f"proj['original'] must be np.ndarray, got {type(img0)}")

    # המרה ל-RGB
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

    # טריק יציבות
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    bg = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    bg.load()

    return bg, disp_w, disp_h, scale_factor


def _get_drawing_bbox(proj):
    """
    מחזיר bbox של השרטוט (x, y, w, h)
    משתמש במסכת הקירות או fallback ל-bbox מלא
    """
    thick_walls = proj.get("thick_walls")
    if thick_walls is not None and np.any(thick_walls > 0):
        # מצא bbox של כל הקירות
        coords = cv2.findNonZero(thick_walls)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return (x, y, w, h)

    # Fallback: כל התמונה
    img = proj.get("original")
    if img is not None:
        h, w = img.shape[:2]
        return (0, 0, w, h)

    return (0, 0, 1, 1)


def _is_point_in_bbox(x, y, bbox):
    """בדיקה אם נקודה בתוך bbox"""
    bx, by, bw, bh = bbox
    return bx <= x <= (bx + bw) and by <= y <= (by + bh)


def _clip_object_to_bbox(obj, bbox, scale_factor):
    """
    מחתך אובייקט לתוך bbox (במקרה של line/path)
    מחזיר None אם האובייקט כולו מחוץ ל-bbox
    """
    obj_type = obj.get("type", "")
    bx, by, bw, bh = bbox

    # המרת bbox לקואורדינטות קנבס
    bx_canvas = int(bx * scale_factor)
    by_canvas = int(by * scale_factor)
    bw_canvas = int(bw * scale_factor)
    bh_canvas = int(bh * scale_factor)

    if obj_type == "line":
        x1 = obj.get("x1", 0)
        y1 = obj.get("y1", 0)
        x2 = obj.get("x2", 0)
        y2 = obj.get("y2", 0)

        # בדוק אם לפחות נקודה אחת בתוך bbox
        p1_in = _is_point_in_bbox(x1, y1, (bx_canvas, by_canvas, bw_canvas, bh_canvas))
        p2_in = _is_point_in_bbox(x2, y2, (bx_canvas, by_canvas, bw_canvas, bh_canvas))

        if not p1_in and not p2_in:
            return None  # כולו מחוץ

        # Clip
        x1 = max(bx_canvas, min(x1, bx_canvas + bw_canvas))
        y1 = max(by_canvas, min(y1, by_canvas + bh_canvas))
        x2 = max(bx_canvas, min(x2, bx_canvas + bw_canvas))
        y2 = max(by_canvas, min(y2, by_canvas + bh_canvas))

        obj_clipped = obj.copy()
        obj_clipped["x1"] = x1
        obj_clipped["y1"] = y1
        obj_clipped["x2"] = x2
        obj_clipped["y2"] = y2
        return obj_clipped

    elif obj_type == "rect":
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0)
        height = obj.get("height", 0)

        # בדוק חיתוך
        rect_right = left + width
        rect_bottom = top + height
        bbox_right = bx_canvas + bw_canvas
        bbox_bottom = by_canvas + bh_canvas

        if (
            left > bbox_right
            or rect_right < bx_canvas
            or top > bbox_bottom
            or rect_bottom < by_canvas
        ):
            return None  # אין חיפוף

        # Clip
        new_left = max(left, bx_canvas)
        new_top = max(top, by_canvas)
        new_right = min(rect_right, bbox_right)
        new_bottom = min(rect_bottom, bbox_bottom)

        obj_clipped = obj.copy()
        obj_clipped["left"] = new_left
        obj_clipped["top"] = new_top
        obj_clipped["width"] = new_right - new_left
        obj_clipped["height"] = new_bottom - new_top
        return obj_clipped

    # עבור path/polygon - פשוט נבדוק אם יש נקודה אחת לפחות בפנים
    elif obj_type == "path":
        path = obj.get("path", [])
        any_inside = False
        for p in path:
            if len(p) >= 3:
                x, y = p[1], p[2]
                if _is_point_in_bbox(
                    x, y, (bx_canvas, by_canvas, bw_canvas, bh_canvas)
                ):
                    any_inside = True
                    break

        return obj if any_inside else None

    return obj  # אחר - נשאיר


def _calculate_boq_from_items(planned_items, categories_config):
    """
    מחשב כתב כמויות (BOQ) מרשימת פריטים מתוכננים

    Returns:
        dict: {
            category_key: {
                "type": ...,
                "subtype": ...,
                "total_length_m": ...,
                "total_area_m2": ...,
                "count": ...,
                "params": {...}
            }
        }
    """
    boq = {}

    for item in planned_items:
        cat_key = item.get("category", "unknown")
        length_m = item.get("length_m", 0)
        area_m2 = item.get("area_m2", 0)

        # קבלת פרמטרים מהקטגוריה
        cat_info = categories_config.get(cat_key, {})

        if cat_key not in boq:
            boq[cat_key] = {
                "type": cat_info.get("type", "unknown"),
                "subtype": cat_info.get("subtype", ""),
                "total_length_m": 0,
                "total_area_m2": 0,
                "count": 0,
                "params": cat_info.get("params", {}),
            }

        boq[cat_key]["total_length_m"] += length_m
        boq[cat_key]["total_area_m2"] += area_m2
        boq[cat_key]["count"] += 1

    return boq


# ==========================================
# רנדור הלשונית הראשית
# ==========================================


def render_manager_planning_tab():
    """
    🧱 הגדרת תכולה לבנייה - Wizard בן 4 שלבים
    """

    st.header("🧱 הגדרת תכולה לבנייה")
    st.caption("Wizard להגדרת תכולה מתוכננת (Planned Items) + כתב כמויות (BOQ)")

    # בדיקת זמינות פונקציות
    if not WORKER_FUNCS_AVAILABLE:
        st.error("⚠️ לא ניתן לייבא פונקציות מ-worker.py - חלק מהפונקציונליות לא תעבוד")

    # --- Guard: אין פרויקטים ---
    if "projects" not in st.session_state or not st.session_state.projects:
        st.warning("📂 אין תוכניות זמינות במערכת")
        st.info("👉 קודם העלה תוכנית ב'📂 סדנת עבודה'")
        return

    st.markdown("---")

    # מצב wizard - שמירת שלב נוכחי
    wizard_key = "planning_wizard_step"
    if wizard_key not in st.session_state:
        st.session_state[wizard_key] = 1

    current_step = st.session_state[wizard_key]

    # Progress bar
    progress_pct = (current_step - 1) / 3  # 4 שלבים = 0, 0.33, 0.66, 1.0
    st.progress(progress_pct, text=f"שלב {current_step} מתוך 4")

    # ==========================================
    # שלב 1: בחירת תוכנית
    # ==========================================
    if current_step == 1:
        st.subheader("📂 שלב 1: בחירת תוכנית")

        # בחירת תוכנית
        plan_options = list(st.session_state.projects.keys())

        # ברירת מחדל: אחרונה
        default_idx = len(plan_options) - 1

        plan_name = st.selectbox(
            "בחר תוכנית לעבוד עליה:",
            plan_options,
            index=default_idx,
            key="planning_selected_plan",
        )

        proj = st.session_state.projects[plan_name]
        meta = _safe_get_metadata(proj)

        # תצוגת מידע על התוכנית
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("קובץ", plan_name)
        with col2:
            scale_val = proj.get("scale", 0)
            st.metric("סקייל", f"{scale_val:.1f} px/m" if scale_val else "❌ לא הוגדר")
        with col3:
            planned = meta.get("planning", {}).get("planned_items_count", 0)
            st.metric("פריטים מתוכננים", f"{planned}")

        # תצוגת thumbnail
        try:
            img = proj.get("original")
            if img is not None:
                rgb = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.ndim == 3
                    else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                )
                h, w = rgb.shape[:2]
                thumb_w = min(400, w)
                scale = thumb_w / w
                thumb_h = int(h * scale)
                thumb = cv2.resize(rgb, (thumb_w, thumb_h))
                st.image(thumb, caption=f"תוכנית: {plan_name}", use_column_width=False)
        except:
            st.info("לא ניתן להציג תצוגה מקדימה")

        st.markdown("---")

        # כפתור המשך
        if st.button(
            "➡️ המשך לשלב 2: כיול וקטגוריות", type="primary", use_container_width=True
        ):
            st.session_state[wizard_key] = 2
            st.rerun()

    # ==========================================
    # שלב 2: כיול סקייל + הגדרת קטגוריות
    # ==========================================
    elif current_step == 2:
        plan_name = st.session_state.get("planning_selected_plan")
        if not plan_name or plan_name not in st.session_state.projects:
            st.error("❌ תוכנית לא נבחרה - חזור לשלב 1")
            if st.button("⬅️ חזור לשלב 1"):
                st.session_state[wizard_key] = 1
                st.rerun()
            return

        proj = st.session_state.projects[plan_name]
        meta = _safe_get_metadata(proj)

        st.subheader("📏 שלב 2: כיול סקייל והגדרת קטגוריות")

        # === כיול סקייל ===
        st.markdown("### 📐 כיול סקייל")
        st.caption("צייר קו על אלמנט עם אורך ידוע → הזן אורך אמיתי → הסקייל יחושב")

        try:
            bg_img, disp_w, disp_h, scale_factor = _build_bg_image_from_proj(
                proj, max_width=800
            )
        except Exception as e:
            st.error(f"❌ שגיאה בהכנת תמונה: {str(e)}")
            bg_img, disp_w, disp_h, scale_factor = None, 0, 0, 1.0

        if bg_img:
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
                if cal_lines and WORKER_FUNCS_AVAILABLE:
                    cal_px = compute_line_length_px(cal_lines[-1])

            if cal_px > 0:
                st.success(f"✅ אורך הקו: {cal_px:.0f} px (על הקנבס)")

                col_input, col_btn = st.columns([3, 1])
                with col_input:
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
                        "✅ עדכן סקייל",
                        type="primary",
                        key=f"planning_fix_scale_{plan_name}",
                    ):
                        cal_px_original = cal_px / scale_factor
                        proj["scale"] = cal_px_original / real_length_m
                        st.success(f"✅ סקייל עודכן: {proj['scale']:.1f} px/m")
                        st.rerun()
            else:
                st.info("👆 צייר קו ישר על אלמנט עם אורך ידוע")
        else:
            st.warning("לא ניתן להציג כלי כיול")

        # הצגת סקייל נוכחי
        current_scale = proj.get("scale", 0)
        if current_scale > 0:
            st.info(f"📏 **סקייל נוכחי:** {current_scale:.1f} px/m")
        else:
            st.warning("⚠️ סקייל לא הוגדר - מומלץ לכייל לפני המשך")

        st.markdown("---")

        # === הגדרת קטגוריות ===
        st.markdown("### 🗂️ הגדרת קטגוריות עבודה")
        st.caption("הגדר את סוגי העבודות והחומרים שיופיעו בפרויקט")

        # מאגר קטגוריות ב-session
        categories_key = f"planning_categories_{plan_name}"
        if categories_key not in st.session_state:
            st.session_state[categories_key] = {}

        categories = st.session_state[categories_key]

        # טופס הוספת קטגוריה
        with st.expander("➕ הוסף קטגוריה חדשה", expanded=len(categories) == 0):
            col1, col2 = st.columns(2)

            with col1:
                new_type = st.selectbox(
                    "סוג עבודה:", ["קירות", "ריצוף", "תקרה"], key="new_cat_type"
                )

            with col2:
                # תתי-סוגים לפי סוג
                if new_type == "קירות":
                    subtypes = ["בטון", "בלוקים", "גבס", "אחר"]
                elif new_type == "ריצוף":
                    subtypes = ["קרמיקה", "פרקט", "גרניט פורצלן", "אחר"]
                else:  # תקרה
                    subtypes = ["גבס", "מינרלים", "אקוסטית", "אחר"]

                new_subtype = st.selectbox("תת-סוג:", subtypes, key="new_cat_subtype")

            # פרמטרים נוספים
            col3, col4 = st.columns(2)
            with col3:
                if new_type == "קירות":
                    new_param1 = st.number_input(
                        "גובה (מ'):", value=2.6, step=0.1, key="new_cat_param1"
                    )
                elif new_type == "ריצוף":
                    new_param1 = st.number_input(
                        'עובי (ס"מ):', value=1.0, step=0.1, key="new_cat_param1"
                    )
                else:
                    new_param1 = st.number_input(
                        "גובה תקרה (מ'):", value=2.8, step=0.1, key="new_cat_param1"
                    )

            with col4:
                new_param2 = st.text_input("הערה:", value="", key="new_cat_param2")

            if st.button("➕ הוסף קטגוריה", use_container_width=True):
                cat_key = f"{new_type}_{new_subtype}_{len(categories)}"
                categories[cat_key] = {
                    "type": new_type,
                    "subtype": new_subtype,
                    "params": {"height_or_thickness": new_param1, "note": new_param2},
                }
                st.success(f"✅ קטגוריה נוספה: {new_type} - {new_subtype}")
                st.rerun()

        # הצגת קטגוריות קיימות
        if categories:
            st.markdown("#### רשימת קטגוריות:")
            for cat_key, cat_data in categories.items():
                with st.expander(
                    f"🔹 {cat_data['type']} - {cat_data['subtype']}", expanded=False
                ):
                    st.json(cat_data)
                    if st.button(f"🗑️ מחק", key=f"del_cat_{cat_key}"):
                        del categories[cat_key]
                        st.rerun()
        else:
            st.info("אין קטגוריות עדיין - הוסף לפחות קטגוריה אחת")

        st.markdown("---")

        # כפתורי ניווט
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅️ חזור לשלב 1", use_container_width=True):
                st.session_state[wizard_key] = 1
                st.rerun()

        with col_next:
            # Guard: חייבים scale וקטגוריה אחת לפחות
            can_proceed = (proj.get("scale", 0) > 0) and len(categories) > 0

            if not can_proceed:
                st.warning("⚠️ חובה: כיול סקייל + לפחות קטגוריה אחת")

            if st.button(
                "➡️ המשך לשלב 3: סימון על קנבס",
                type="primary",
                disabled=not can_proceed,
                use_container_width=True,
            ):
                st.session_state[wizard_key] = 3
                st.rerun()

    # ==========================================
    # שלב 3: סימון על קנבס
    # ==========================================
    elif current_step == 3:
        plan_name = st.session_state.get("planning_selected_plan")
        if not plan_name or plan_name not in st.session_state.projects:
            st.error("❌ תוכנית לא נבחרה")
            if st.button("⬅️ חזור לשלב 1"):
                st.session_state[wizard_key] = 1
                st.rerun()
            return

        proj = st.session_state.projects[plan_name]
        meta = _safe_get_metadata(proj)

        categories_key = f"planning_categories_{plan_name}"
        categories = st.session_state.get(categories_key, {})

        if not categories:
            st.error("❌ אין קטגוריות מוגדרות")
            if st.button("⬅️ חזור לשלב 2"):
                st.session_state[wizard_key] = 2
                st.rerun()
            return

        st.subheader("🎨 שלב 3: סימון תכולה על הקנבס")

        col_canvas, col_config = st.columns([2, 1])

        with col_config:
            st.markdown("### ⚙️ הגדרות ציור")

            # בחירת קטגוריה פעילה
            cat_labels = {
                k: f"{v['type']} - {v['subtype']}" for k, v in categories.items()
            }
            active_cat = st.selectbox(
                "קטגוריה פעילה:",
                list(cat_labels.keys()),
                format_func=lambda x: cat_labels[x],
                key="planning_active_category",
            )

            st.info(f"✏️ כל ציור חדש יקבל: **{cat_labels[active_cat]}**")

            # מצב ציור
            drawing_mode = st.selectbox(
                "מצב ציור:",
                ["line", "freedraw", "rect", "polygon"],
                index=0,
                key="planning_drawing_mode",
            )

            st.markdown("---")

            # הצגת מידע על הקטגוריה הפעילה
            st.markdown("**פרמטרים:**")
            active_cat_data = categories[active_cat]
            st.write(f"סוג: {active_cat_data['type']}")
            st.write(f"תת-סוג: {active_cat_data['subtype']}")
            params = active_cat_data.get("params", {})
            for pk, pv in params.items():
                st.write(f"{pk}: {pv}")

            st.markdown("---")

            # אופציות נוספות
            st.checkbox(
                "✅ הצמדה לגבולות שרטוט", value=True, key="planning_clip_to_bounds"
            )
            st.checkbox("🔍 הצג bbox של שרטוט", value=False, key="planning_show_bbox")

        with col_canvas:
            st.markdown("### 🖼️ קנבס")

            try:
                bg_img, disp_w, disp_h, scale_factor = _build_bg_image_from_proj(
                    proj, max_width=700
                )
            except Exception as e:
                st.error(f"❌ שגיאה: {str(e)}")
                bg_img = None

            if bg_img:
                # הצגת bbox אם מבוקש
                if st.session_state.get("planning_show_bbox", False):
                    bbox = _get_drawing_bbox(proj)
                    bx, by, bw, bh = bbox
                    st.caption(f"📏 bbox של שרטוט: x={bx}, y={by}, w={bw}, h={bh}")

                canvas = st_canvas(
                    fill_color=(
                        "rgba(0,255,0,0.1)"
                        if drawing_mode == "rect"
                        else "rgba(0,0,0,0)"
                    ),
                    stroke_color="#00FF00",
                    stroke_width=5,
                    background_image=bg_img,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode=drawing_mode,
                    key=f"planning_main_canvas_{plan_name}_{drawing_mode}",
                    update_streamlit=True,
                )
            else:
                canvas = None
                st.error("לא ניתן להציג קנבס")

        st.markdown("---")

        # ניתוח ושמירת פריטים
        items_key = f"planning_items_{plan_name}"
        if items_key not in st.session_state:
            st.session_state[items_key] = []

        planned_items = st.session_state[items_key]

        # עיבוד ציורים חדשים
        if canvas and canvas.json_data and canvas.json_data.get("objects"):
            new_objects = canvas.json_data["objects"]

            # בדוק אם יש פריטים חדשים שטרם נשמרו
            existing_count = len(planned_items)

            if len(new_objects) > existing_count:
                st.info(f"🆕 זוהו {len(new_objects) - existing_count} פריטים חדשים")

                if st.button("💾 שמור פריטים חדשים", type="primary"):
                    scale = proj.get("scale", 200)
                    bbox = _get_drawing_bbox(proj)
                    clip_enabled = st.session_state.get("planning_clip_to_bounds", True)

                    for obj in new_objects[existing_count:]:
                        # Clip אם מבוקש
                        if clip_enabled:
                            obj_clipped = _clip_object_to_bbox(obj, bbox, scale_factor)
                            if obj_clipped is None:
                                st.warning(f"⚠️ פריט מחוץ לגבולות - נדלג")
                                continue
                            obj = obj_clipped

                        # חישוב מידות
                        obj_type = obj.get("type", "")
                        length_px = 0
                        area_px = 0

                        if obj_type == "line" and WORKER_FUNCS_AVAILABLE:
                            length_px = compute_line_length_px(obj)
                        elif obj_type == "rect" and WORKER_FUNCS_AVAILABLE:
                            area_px = compute_rect_area_px(obj)
                        elif obj_type == "path" and WORKER_FUNCS_AVAILABLE:
                            length_px = compute_line_length_px(obj)

                        # המרה למטרים
                        if length_px > 0 and WORKER_FUNCS_AVAILABLE:
                            length_m = px_to_m(length_px / scale_factor, 1.0, scale)
                        else:
                            length_m = 0

                        if area_px > 0 and WORKER_FUNCS_AVAILABLE:
                            area_m2 = px2_to_m2(area_px / (scale_factor**2), 1.0, scale)
                        else:
                            area_m2 = 0

                        # שמירת פריט
                        item = {
                            "uid": f"item_{len(planned_items)}_{datetime.now().strftime('%H%M%S')}",
                            "type": obj_type,
                            "category": active_cat,
                            "raw_object": obj,
                            "length_m": round(length_m, 2),
                            "area_m2": round(area_m2, 2),
                            "timestamp": datetime.now().isoformat(),
                        }

                        planned_items.append(item)

                    st.success(f"✅ נשמרו {len(new_objects) - existing_count} פריטים!")
                    st.rerun()

        # הצגת פריטים קיימים
        if planned_items:
            st.markdown("### 📋 פריטים מתוכננים")
            st.caption(f'סה"כ: {len(planned_items)} פריטים')

            for idx, item in enumerate(planned_items):
                cat_data = categories.get(item["category"], {})
                cat_label = (
                    f"{cat_data.get('type', '?')} - {cat_data.get('subtype', '?')}"
                )

                with st.expander(
                    f"#{idx+1} | {cat_label} | {item['type']}", expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("אורך", f"{item['length_m']:.2f} מ'")
                    with col2:
                        st.metric("שטח", f"{item['area_m2']:.2f} מ\"ר")
                    with col3:
                        if st.button("🗑️", key=f"del_item_{item['uid']}"):
                            planned_items.remove(item)
                            st.rerun()
        else:
            st.info("אין פריטים עדיין - התחל לצייר!")

        st.markdown("---")

        # ניווט
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅️ חזור לשלב 2", use_container_width=True):
                st.session_state[wizard_key] = 2
                st.rerun()

        with col_next:
            can_proceed = len(planned_items) > 0

            if not can_proceed:
                st.warning("⚠️ נדרש לפחות פריט אחד")

            if st.button(
                "➡️ המשך לשלב 4: שמירה + BOQ",
                type="primary",
                disabled=not can_proceed,
                use_container_width=True,
            ):
                st.session_state[wizard_key] = 4
                st.rerun()

    # ==========================================
    # שלב 4: שמירה וכתב כמויות
    # ==========================================
    elif current_step == 4:
        plan_name = st.session_state.get("planning_selected_plan")
        if not plan_name or plan_name not in st.session_state.projects:
            st.error("❌ תוכנית לא נבחרה")
            if st.button("⬅️ חזור לשלב 1"):
                st.session_state[wizard_key] = 1
                st.rerun()
            return

        proj = st.session_state.projects[plan_name]
        meta = _safe_get_metadata(proj)

        categories_key = f"planning_categories_{plan_name}"
        items_key = f"planning_items_{plan_name}"

        categories = st.session_state.get(categories_key, {})
        planned_items = st.session_state.get(items_key, [])

        st.subheader("📊 שלב 4: סיכום ושמירה")

        # חישוב BOQ
        boq = _calculate_boq_from_items(planned_items, categories)

        # הצגת BOQ
        st.markdown("### 📋 כתב כמויות (BOQ)")

        if boq:
            import pandas as pd

            boq_rows = []
            for cat_key, data in boq.items():
                row = {
                    "סוג": data["type"],
                    "תת-סוג": data["subtype"],
                    "כמות פריטים": data["count"],
                    "אורך כולל (מ')": f"{data['total_length_m']:.2f}",
                    'שטח כולל (מ"ר)': f"{data['total_area_m2']:.2f}",
                }

                # פרמטרים
                params = data.get("params", {})
                if params.get("height_or_thickness"):
                    row["פרמטר"] = f"{params['height_or_thickness']}"

                boq_rows.append(row)

            df_boq = pd.DataFrame(boq_rows)
            st.dataframe(df_boq, use_column_width=True, hide_index=True)

            # סיכום סופי
            st.markdown("---")
            st.markdown("### 📊 סיכום כללי")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('סה"כ פריטים', len(planned_items))
            with col2:
                total_length = sum([item["length_m"] for item in planned_items])
                st.metric('סה"כ אורך', f"{total_length:.2f} מ'")
            with col3:
                total_area = sum([item["area_m2"] for item in planned_items])
                st.metric('סה"כ שטח', f'{total_area:.2f} מ"ר')
        else:
            st.warning("אין נתונים ל-BOQ")

        st.markdown("---")

        # שמירה
        st.markdown("### 💾 שמירה סופית")
        st.caption("שמירת התכנון למטא-דאטה של התוכנית")

        col_back, col_save = st.columns(2)

        with col_back:
            if st.button("⬅️ חזור לשלב 3", use_container_width=True):
                st.session_state[wizard_key] = 3
                st.rerun()

        with col_save:
            if st.button("✅ שמור ל וסיים", type="primary", use_container_width=True):
                # שמירה למטא-דאטה
                meta["planning"] = {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "scale": proj.get("scale", 0),
                    "categories": categories,
                    "planned_items": planned_items,
                    "planned_items_count": len(planned_items),
                    "boq": boq,
                    "totals": {
                        "total_length_m": sum(
                            [item["length_m"] for item in planned_items]
                        ),
                        "total_area_m2": sum(
                            [item["area_m2"] for item in planned_items]
                        ),
                    },
                }

                proj["metadata"] = meta

                st.success("✅ התכנון נשמר בהצלחה!")
                st.balloons()

                st.info("💡 כעת העובד יוכל לראות את התכולה המתוכננת ולדווח על ביצוע")

                # איפוס wizard
                if st.button("🔄 תכנן תוכנית נוספת"):
                    st.session_state[wizard_key] = 1
                    st.rerun()


# ==========================================
# נקודת כניסה (אם מריצים ישירות)
# ==========================================
if __name__ == "__main__":
    st.set_page_config(page_title="Planning Test", layout="wide")
    render_manager_planning_tab()
