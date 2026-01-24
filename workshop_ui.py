import streamlit as st
import cv2
import numpy as np
import json
import pandas as pd
import tempfile
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ייבוא פונקציות קיימות מהקבצים שלך
from analyzer import FloorPlanAnalyzer
from database import save_plan
from utils import (
    safe_process_metadata,
    create_colored_overlay,
    get_simple_metadata_values,
)
from floor_extractor import analyze_floor_and_rooms
from preprocessing import get_crop_bbox_from_canvas_data

# ==========================================
# רכיבי UI מודולריים (Cards & Components)
# ==========================================


def _ui_header_and_upload():
    """חלק עליון: בחירת פרויקט וכפתור העלאה דיסקרטי"""

    col_title, col_upload = st.columns([4, 1])

    with col_title:
        # אם יש פרויקטים, הצג סרגל בחירה מעוצב
        if st.session_state.projects:
            project_names = list(st.session_state.projects.keys())
            selected = st.selectbox(
                "📂 בחר תוכנית לעבודה:",
                project_names,
                index=(
                    0
                    if "ws_active_plan" not in st.session_state
                    else project_names.index(st.session_state.ws_active_plan)
                ),
                key="ws_main_selector",
                label_visibility="collapsed",
            )
            st.session_state.ws_active_plan = selected
        else:
            st.info("👈 התחל בהעלאת תוכנית חדשה")

    with col_upload:
        # תיקון תאימות לגרסה 1.28: שימוש ב-expander במקום popover
        with st.expander("➕ תוכנית חדשה", expanded=False):
            st.markdown("### העלאת תוכנית")
            files = st.file_uploader(
                "", type="pdf", accept_multiple_files=True, key="ws_modern_upload"
            )

            # --- לוגיקת העלאה (זהה למקור) ---
            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name}..."):
                            try:
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=".pdf"
                                ) as tmp:
                                    tmp.write(f.getvalue())
                                    path = tmp.name

                                analyzer = FloorPlanAnalyzer()
                                (
                                    pix,
                                    skel,
                                    thick,
                                    orig,
                                    meta,
                                    conc,
                                    blok,
                                    floor,
                                    debug_img,
                                ) = analyzer.process_file(path, save_debug=False)

                                # מילוי ברירות מחדל
                                if not meta.get("plan_name"):
                                    meta["plan_name"] = f.name.replace(
                                        ".pdf", ""
                                    ).strip()

                                # חילוץ LLM
                                llm_data = {}
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(
                                        raw_text=meta.get("raw_text"),
                                        raw_text_full=meta.get("raw_text_full"),
                                        normalized_text=meta.get("normalized_text"),
                                        raw_blocks=meta.get("raw_blocks"),
                                    )
                                    meta.update(get_simple_metadata_values(llm_data))

                                st.session_state.projects[f.name] = {
                                    "skeleton": skel,
                                    "thick_walls": thick,
                                    "original": orig,
                                    "raw_pixels": pix,
                                    "scale": 200.0,
                                    "metadata": meta,
                                    "concrete_mask": conc,
                                    "blocks_mask": blok,
                                    "flooring_mask": floor,
                                    "total_length": pix / 200.0,
                                    "llm_data": llm_data,
                                    "floor_analysis": None,  # מקום לתוצאות ניתוח עתידי
                                }
                                os.unlink(path)
                                st.session_state.ws_active_plan = f.name
                                st.rerun()
                            except Exception as e:
                                st.error(f"שגיאה: {e}")


def _ui_control_panel(plan_key, proj):
    """פאנל צד שמאל: הגדרות, סקלה ופעולות - כולל מדידות מתקדמות"""

    with st.container(border=True):
        st.markdown("### ⚙️ הגדרות")

        # עריכת שם
        new_name = st.text_input(
            "שם התוכנית",
            value=proj["metadata"].get("plan_name", plan_key),
            key="ui_name_edit",
        )
        proj["metadata"]["plan_name"] = new_name

        # --- מדידות מתקדמות (החלק החשוב שהוחזר) ---
        with st.expander("📐 מדידות מתקדמות (נייר/סקלה)", expanded=False):
            meta = proj.get("metadata", {})

            # 1. בחירת גודל נייר (Override)
            current_detected = meta.get("paper_size_detected", "unknown")
            paper_options = ["זיהוי אוטומטי", "A0", "A1", "A2", "A3", "A4"]

            # מציאת האינדקס הנוכחי
            default_idx = 0
            if "paper_override" in st.session_state:
                if st.session_state.paper_override in paper_options:
                    default_idx = paper_options.index(st.session_state.paper_override)

            selected_paper = st.selectbox(
                f"גודל נייר (זוהה: {current_detected}):",
                options=paper_options,
                index=default_idx,
                key=f"paper_select_{plan_key}",
            )

            # לוגיקת עדכון גודל נייר (נלקח מ-manager.py)
            if selected_paper != "זיהוי אוטומטי" and selected_paper != current_detected:
                ISO_SIZES = {
                    "A0": (841, 1189),
                    "A1": (594, 841),
                    "A2": (420, 594),
                    "A3": (297, 420),
                    "A4": (210, 297),
                }
                paper_w_mm, paper_h_mm = ISO_SIZES[selected_paper]

                # התאמת כיוון (Landscape/Portrait)
                if meta.get("image_size_px"):
                    w_px = meta["image_size_px"]["width"]
                    h_px = meta["image_size_px"]["height"]
                    if w_px > h_px and paper_w_mm < paper_h_mm:
                        paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm

                # עדכון המטא-דאטה
                meta["paper_size_detected"] = selected_paper
                meta["paper_mm"] = {"width": paper_w_mm, "height": paper_h_mm}

                # חישוב מחדש של יחסים
                if meta.get("image_size_px"):
                    mm_per_pixel = (paper_w_mm / w_px + paper_h_mm / h_px) / 2.0
                    meta["mm_per_pixel"] = float(mm_per_pixel)

                    # אם יש סקלה ידועה, נעדכן גם את המטרים
                    if meta.get("scale_denominator"):
                        meta["meters_per_pixel"] = float(
                            (mm_per_pixel * meta["scale_denominator"]) / 1000.0
                        )
                        proj["scale"] = 1.0 / meta["meters_per_pixel"]  # סנכרון הפוך

            # 2. תצוגת נתונים טכניים (החישוב)
            if all(k in meta for k in ["paper_mm", "image_size_px", "mm_per_pixel"]):
                st.markdown("---")
                st.caption("נתוני חישוב:")
                st.code(
                    f"""
נייר: {meta['paper_mm']['width']:.0f}x{meta['paper_mm']['height']:.0f} מ"מ
תמונה: {meta['image_size_px']['width']}x{meta['image_size_px']['height']} px
יחס: {meta['mm_per_pixel']:.4f} מ"מ/px
                 """,
                    language="text",
                )

            # 3. הזנה ידנית של קנה מידה (Fallback)
            st.markdown("---")
            col_man1, col_man2 = st.columns([2, 1])
            with col_man1:
                manual_scale = st.text_input(
                    "קנה מידה (טקסט)",
                    value=meta.get("scale", "1:50"),
                    key=f"txt_scale_{plan_key}",
                )
            with col_man2:
                if st.button("עדכן", key=f"btn_scale_{plan_key}"):
                    from analyzer import parse_scale

                    parsed = parse_scale(manual_scale)
                    if parsed:
                        meta["scale_denominator"] = parsed
                        meta["scale"] = manual_scale
                        # עדכון מטרים לפיקסל
                        if meta.get("mm_per_pixel"):
                            meta["meters_per_pixel"] = (
                                meta["mm_per_pixel"] * parsed
                            ) / 1000.0
                            proj["scale"] = 1.0 / meta["meters_per_pixel"]
                        st.success("עודכן!")

        # סליידר סקלה (ויזואלי ומהיר)
        st.markdown("---")
        st.caption("כיוונון עדין (פיקסלים למטר)")

        # הגנה מפני ערך 0 או None
        current_scale = float(proj["scale"]) if proj.get("scale") else 200.0

        scale_val = st.slider(
            "Scale",
            10.0,
            1000.0,
            current_scale,
            key="ui_scale_slider",
            label_visibility="collapsed",
        )
        proj["scale"] = scale_val

        # חישוב אורך מהיר
        pixels = proj.get("raw_pixels", 0)
        total_m = pixels / scale_val

        col_m, col_px = st.columns(2)
        with col_m:
            st.metric("אורך קירות", f"{total_m:.1f} מ'")
        with col_px:
            st.metric("פיקסלים", f"{pixels:,}")

    # כפתור פעולה ראשי - ניתוח חדרים
    st.markdown("### 🧠 פעולות חכמות")
    if st.button("🔍 נתח חדרים ושטחים", use_container_width=True, type="primary"):
        with st.spinner("מבצע סגמנטציה וניתוח גיאומטרי..."):
            _run_floor_analysis(plan_key, proj)

    # שמירה
    st.markdown("---")
    if st.button("💾 שמור למערכת", use_container_width=True):
        _save_project_logic(plan_key, proj)


def _ui_visualization_area(plan_key, proj):
    """איזור ויזואליזציה מרכזי + טאבים של נתונים"""

    # --- שכבות תצוגה ---
    col_vis_toggles = st.columns([1, 1, 1, 3])
    with col_vis_toggles[0]:
        show_flooring = st.toggle("ריצוף", value=True)
    with col_vis_toggles[1]:
        show_rooms = st.toggle("חדרים", value=True)  # יופיע רק אם יש ניתוח

    # יצירת התמונה להצגה
    # 1. בסיס (קירות)
    overlay = create_colored_overlay(
        proj["original"],
        proj["concrete_mask"],
        proj["blocks_mask"],
        proj["flooring_mask"] if show_flooring else None,
    )

    # 2. שכבת חדרים (אם בוצע ניתוח וביקשו להציג)
    if (
        show_rooms
        and proj.get("floor_analysis")
        and proj["floor_analysis"].get("success")
    ):
        viz_overlay = proj["floor_analysis"]["visualizations"].get("overlay")
        if viz_overlay is not None:
            # שילוב עדין בין הויזואליזציות
            overlay = cv2.addWeighted(overlay, 0.4, viz_overlay, 0.6, 0)

    st.image(overlay, use_container_width=True, channels="BGR")

    # --- טאבים תחתונים לנתונים (במקום לעבור עמודים) ---
    st.markdown("---")
    tab_ai, tab_rooms, tab_calc = st.tabs(
        ["🤖 נתוני AI (טקסט)", "📐 טבלת חדרים", "💰 מחשבון"]
    )

    with tab_ai:
        _render_ai_data_tab(proj)

    with tab_rooms:
        _render_rooms_table(proj)

    with tab_calc:
        _render_calculator(proj)


# ==========================================
# לוגיקה פנימית ופונקציות עזר (Logic Helpers)
# ==========================================


def _run_floor_analysis(plan_key, proj):
    """מפעיל את הלוגיקה מ-floor_extractor.py"""
    walls_mask = proj.get("thick_walls")
    if walls_mask is None:
        st.error("חסרה מסכת קירות")
        return

    # חילוץ נתונים
    meta = proj.get("metadata", {})
    meters_per_pixel = meta.get(
        "meters_per_pixel", 1.0 / proj["scale"]
    )  # Fallback to manual scale

    llm_rooms = None
    if proj.get("llm_data") and "rooms" in proj["llm_data"]:
        llm_rooms = proj["llm_data"]["rooms"]

    # הפעלת הפונקציה מהקובץ הקיים
    result = analyze_floor_and_rooms(
        walls_mask=walls_mask,
        original_image=proj["original"],
        meters_per_pixel=meters_per_pixel,
        llm_rooms=llm_rooms,
        segmentation_method="watershed",
        min_room_area_px=500,
    )

    # שמירת תוצאה
    proj["floor_analysis"] = result

    if result["success"]:
        st.success(f"נמצאו {result['totals']['num_rooms']} חדרים")
    else:
        st.warning("הניתוח לא הצליח במלואו")


def _render_ai_data_tab(proj):
    """מציג את הנתונים מ-Utils בצורה נקייה"""
    llm = proj.get("llm_data", {})
    if not llm or llm.get("status") == "error":
        st.caption("לא נמצא מידע טקסטואלי")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**פרטי מסמך**")
        doc = llm.get("document", {})
        if doc:
            st.write(f"🏷️ פרויקט: {doc.get('project_name', {}).get('value', '-')}")
            st.write(f"📅 תאריך: {doc.get('date', {}).get('value', '-')}")
            st.write(f"👷 אדריכל: {doc.get('architect_name', {}).get('value', '-')}")

    with col2:
        st.markdown("**הערות ביצוע**")
        notes = llm.get("execution_notes", {})
        if notes:
            with st.container(height=100):
                st.write(notes.get("general_notes", {}).get("value", "-"))


def _render_rooms_table(proj):
    """מציג טבלת חדרים מהניתוח הגאומטרי"""
    analysis = proj.get("floor_analysis")
    if not analysis or not analysis.get("success"):
        st.info("לחץ על 'נתח חדרים ושטחים' בפאנל הצדדי כדי לראות נתונים כאן.")
        return

    rooms = analysis["rooms"]
    if not rooms:
        st.write("לא זוהו חדרים.")
        return

    # הכנת דאטה לטבלה
    data = []
    for r in rooms:
        data.append(
            {
                "מזהה": r["room_id"],
                "שם (AI)": r.get("matched_name", "-"),
                'שטח (מ"ר)': f"{r['area_m2']:.2f}" if r["area_m2"] else "-",
                "היקף (מ')": f"{r['perimeter_m']:.2f}" if r["perimeter_m"] else "-",
            }
        )

    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    # סיכומים
    tot = analysis["totals"]
    st.caption(
        f"סה\"כ שטח: {tot['total_area_m2']:.1f} מ\"ר | סה\"כ היקף: {tot['total_perimeter_m']:.1f} מ'"
    )


def _render_calculator(proj):
    """מחשבון הצעת מחיר פשוט"""
    # חישוב כמויות (לוגיקה בסיסית)
    scale = proj["scale"]
    pixels = proj.get("raw_pixels", 0)

    # אם יש ניתוח חדרים מדויק, נשתמש בו לריצוף
    floor_area = 0
    if proj.get("floor_analysis") and proj["floor_analysis"].get("success"):
        floor_area = proj["floor_analysis"]["totals"]["total_area_m2"] or 0
    else:
        # Fallback לחישוב פיקסלים גס
        floor_area = proj["metadata"].get("pixels_flooring_area", 0) / (scale**2)

    total_len_m = pixels / scale

    # UI
    c1, c2, c3 = st.columns(3)
    p_wall = c1.number_input("מחיר קיר (₪/מ')", value=1200)
    p_floor = c2.number_input('מחיר ריצוף (₪/מ"ר)', value=250)

    cost_walls = total_len_m * p_wall
    cost_floor = floor_area * p_floor
    total = cost_walls + cost_floor

    st.markdown(f'### סה"כ משוער: **{total:,.0f} ₪**')
    st.caption(f"קירות: {cost_walls:,.0f}₪ | ריצוף: {cost_floor:,.0f}₪")


def _save_project_logic(plan_key, proj):
    """לוגיקת שמירה למסד הנתונים"""
    try:
        meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
        # חישוב חומרים בסיסי
        conc_len = proj.get("total_length", 0)  # פישוט

        save_plan(
            plan_key,
            proj["metadata"].get("plan_name", "ללא שם"),
            "1:50",  # Placeholder
            float(proj["scale"]),
            int(proj["raw_pixels"]),
            meta_json,
            None,
            0,
            0,
            "{}",
        )
        st.toast("✅ הפרויקט נשמר בהצלחה!")
    except Exception as e:
        st.error(f"שגיאה בשמירה: {e}")


# ==========================================
# הפונקציה הראשית (Main Export)
# ==========================================


def render_modern_workshop():
    """זו הפונקציה היחידה שתקרא לה מ-manager.py"""

    # אתחול Session State אם לא קיים
    if "projects" not in st.session_state:
        st.session_state.projects = {}

    st.markdown("## 🛠️ סדנת עבודה")

    # 1. חלק עליון - בחירה והעלאה
    _ui_header_and_upload()
    st.divider()

    # 2. אזור עבודה ראשי (רק אם יש פרויקט)
    if st.session_state.get("ws_active_plan") and st.session_state.projects:
        active_key = st.session_state.ws_active_plan
        if active_key in st.session_state.projects:
            proj = st.session_state.projects[active_key]

            # פריסת מסך: צד שמאל צר (פקדים), צד ימין רחב (תצוגה)
            col_ctrl, col_view = st.columns([1, 2.5], gap="medium")

            with col_ctrl:
                _ui_control_panel(active_key, proj)

            with col_view:
                _ui_visualization_area(active_key, proj)
        else:
            st.error("התוכנית שנבחרה לא נמצאה בזיכרון. נסה להעלות מחדש.")
    else:
        st.info("אנא בחר או העלה תוכנית כדי להתחיל לעבוד.")
