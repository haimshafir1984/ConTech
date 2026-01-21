import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
import io
import gc  # 🆕 לניהול זיכרון
from streamlit_drawable_canvas import st_canvas
from datetime import datetime

# ייבוא מהקבצים המסודרים שלך
from styles import *
from utils import (
    safe_process_metadata,
    safe_analyze_legend,
    load_stats_df,
    create_colored_overlay,
    format_llm_metadata,
    get_simple_metadata_values,
)
from analyzer import FloorPlanAnalyzer, compute_skeleton_length_px
from reporter import generate_status_pdf, generate_payment_invoice_pdf
from database import (
    init_database,
    save_plan,
    save_progress_report,
    get_progress_reports,
    get_plan_by_filename,
    get_all_plans,
    get_plan_by_id,
    get_project_forecast,
    get_project_financial_status,
    calculate_material_estimates,
    reset_all_data,
    get_payment_invoice_data,
    get_all_work_types_for_plan,
    get_progress_summary_by_date_range,
)

# --- אתחול המערכת ---
apply_all_styles()
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- Session State ---
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "wall_height" not in st.session_state:
    st.session_state.wall_height = 2.5
if "default_cost_per_meter" not in st.session_state:
    st.session_state.default_cost_per_meter = 0.0
if "manual_corrections" not in st.session_state:
    st.session_state.manual_corrections = {}


# 🆕 פונקציות ייעול זיכרון
def compress_image(img, max_size=1200):
    """מקטין תמונה גדולה לחיסכון בזיכרון"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# --- פונקציה לחישוב קירות מתוקנים ---
def get_corrected_walls(selected_plan, proj):
    """מחזיר את מסכת הקירות המתוקנת (אם יש תיקונים)"""
    if selected_plan in st.session_state.manual_corrections:
        corrections = st.session_state.manual_corrections[selected_plan]
        corrected = proj["thick_walls"].copy()

        if "added_walls" in corrections:
            corrected = cv2.bitwise_or(corrected, corrections["added_walls"])

        if "removed_walls" in corrections:
            corrected = cv2.subtract(corrected, corrections["removed_walls"])

        return corrected
    else:
        return proj["thick_walls"]


# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro v2.0")
    st.caption("✨ Multi-pass Detection + Manual Corrections")
    mode = st.radio(
        "ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed"
    )
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות"):
        st.session_state.wall_height = st.number_input(
            "גובה קירות (מ')",
            value=st.session_state.wall_height,
            step=0.1,
            key="global_wall_height",
        )
        st.session_state.default_cost_per_meter = st.number_input(
            "עלות למטר (₪)",
            value=st.session_state.default_cost_per_meter,
            step=10.0,
            key="global_cost_per_meter",
        )

    # 🆕 ניטור וניקוי זיכרון
    if st.session_state.projects:
        num_proj = len(st.session_state.projects)
        st.caption(f"📊 {num_proj} פרויקט{'ים' if num_proj > 1 else ''} טעונים")
        if st.button("🧹 נקה זיכרון"):
            for proj in st.session_state.projects.values():
                if "debug_layers" in proj:
                    proj["debug_layers"] = {}
            gc.collect()
            st.success("הזיכרון נוקה!")
            st.rerun()

    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.session_state.manual_corrections = {}
            gc.collect()  # 🆕 שחרור זיכרון
            st.success("המערכת אופסה")
            st.rerun()

# ==========================================
# 🏢 מצב מנהל
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📂 סדנת עבודה", "🎨 תיקונים ידניים", "📊 דשבורד", "💰 חשבונות"]
    )

    # --- טאב 1: העלאה ועריכה ---
    with tab1:
        with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
            files = st.file_uploader(
                "גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True
            )
            debug_mode = st.selectbox(
                "מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0
            )
            show_debug = debug_mode != "בסיסי"

            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name} עם Multi-Pass Detection..."):
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
                                ) = analyzer.process_file(path, save_debug=show_debug)

                                if not meta.get("plan_name"):
                                    meta["plan_name"] = (
                                        f.name.replace(".pdf", "")
                                        .replace("-", " ")
                                        .strip()
                                    )

                                # 🆕 עיבוד LLM משופר עם סכמה מלאה
                                llm_data = {}
                                if meta.get("raw_text"):
                                    # 🆕 הגבלת raw_text לחיסכון בזיכרון
                                    meta["raw_text"] = meta["raw_text"][:4000]
                                    llm_data = safe_process_metadata(meta["raw_text"])

                                    # חילוץ ערכים פשוטים לשדות ישנים
                                    simple_data = get_simple_metadata_values(llm_data)
                                    meta.update(simple_data)

                                # 🆕 דחיסת תמונה מקורית לחיסכון בזיכרון
                                orig_compressed = compress_image(orig, max_size=1200)

                                st.session_state.projects[f.name] = {
                                    "skeleton": None,  # 🆕 לא שומרים - חוסך ~10MB
                                    "thick_walls": thick,
                                    "original": orig_compressed,  # 🆕 גרסה דחוסה
                                    "raw_pixels": pix,
                                    "scale": 200.0,
                                    "metadata": meta,
                                    "concrete_mask": conc,
                                    "blocks_mask": blok,
                                    "flooring_mask": floor,
                                    "total_length": pix / 200.0,
                                    "llm_data": llm_data,  # ← שם מפתח נכון!
                                    "debug_layers": {},  # 🆕 מערך ריק - חוסך ~20-30MB
                                }

                                # תצוגת Debug משופרת
                                if show_debug and debug_img is not None:
                                    st.markdown("### 🔍 ניתוח Multi-Pass")

                                    if debug_mode == "מפורט - שכבות":
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.image(
                                                debug_img,
                                                caption="תוצאה משולבת",
                                                use_column_width=True,
                                            )
                                        with col2:
                                            if (
                                                hasattr(analyzer, "debug_layers")
                                                and "text_combined"
                                                in analyzer.debug_layers
                                            ):
                                                st.image(
                                                    analyzer.debug_layers[
                                                        "text_combined"
                                                    ],
                                                    caption="🔴 טקסט שהוסר",
                                                    use_column_width=True,
                                                )
                                        with col3:
                                            if (
                                                hasattr(analyzer, "debug_layers")
                                                and "walls" in analyzer.debug_layers
                                            ):
                                                st.image(
                                                    analyzer.debug_layers["walls"],
                                                    caption="🟢 קירות שזוהו",
                                                    use_column_width=True,
                                                )

                                    elif debug_mode == "מלא - עם confidence":
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.image(
                                                debug_img,
                                                caption="תוצאה משולבת",
                                                use_column_width=True,
                                            )
                                        with col2:
                                            st.markdown(
                                                """
                                            **מקרא צבעים:**
                                            - 🟠 כתום = טקסט ברור
                                            - 🟡 צהוב = סמלים וכותרות
                                            - 🟣 סגול = מספרי חדרים
                                            - 🟢 ירוק = קירות
                                            - 🔥 אדום-צהוב = confidence גבוה
                                            - 🔵 כחול-שחור = confidence נמוך
                                            """
                                            )

                                            st.metric(
                                                "Confidence ממוצע",
                                                f"{meta.get('confidence_avg', 0):.2f}",
                                            )
                                            st.metric(
                                                "פיקסלי טקסט שהוסרו",
                                                f"{meta.get('text_removed_pixels', 0):,}",
                                            )

                                # 🆕 ניקוי זיכרון מיידי
                                del orig, skel
                                if debug_img is not None:
                                    del debug_img
                                if hasattr(analyzer, "debug_layers"):
                                    del analyzer.debug_layers
                                gc.collect()

                                os.unlink(path)
                                st.success(f"✅ {f.name} נותח בהצלחה!")
                            except Exception as e:
                                st.error(f"שגיאה: {str(e)}")
                                import traceback

                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox(
                "בחר תוכנית לעריכה:", list(st.session_state.projects.keys())
            )
            proj = st.session_state.projects[selected]

            # 🆕 תצוגת מטא-דאטה - תמיד מוצג!
            llm_data = proj.get("llm_data", {})

            st.markdown("---")
            st.markdown("### 🤖 מטא-דאטה שחולץ מה-PDF")

            # הצגת שגיאות אם יש
            if llm_data.get("status") in [
                "no_api_key",
                "empty_text",
                "extraction_failed",
            ]:
                st.error(f"⚠️ {llm_data.get('error', 'שגיאה לא ידועה')}")

                # פירוט שגיאות לפי מודל
                if "errors_by_model" in llm_data:
                    with st.expander("🔍 שגיאות לפי מודל"):
                        for model, err in llm_data["errors_by_model"].items():
                            st.text(f"{model}: {err}")

            # תצוגת JSON - תמיד!
            if llm_data:
                tab1_meta, tab2_meta = st.tabs(["📄 תצוגה פשוטה", "🔍 JSON מלא"])

                with tab1_meta:
                    # תצוגה ידידותית
                    pretty = format_llm_metadata(llm_data)

                    doc = pretty.get("document", {})
                    if doc:
                        cols = st.columns(3)

                        with cols[0]:
                            if doc.get("plan_title"):
                                st.metric("שם תוכנית", doc["plan_title"])
                            if doc.get("plan_type"):
                                st.metric("סוג תוכנית", doc["plan_type"])

                        with cols[1]:
                            if doc.get("scale"):
                                st.metric("קנה מידה", doc["scale"])
                            if doc.get("date"):
                                st.metric("תאריך", doc["date"])

                        with cols[2]:
                            if doc.get("floor_or_level"):
                                st.metric("קומה", doc["floor_or_level"])
                            if doc.get("project_name"):
                                st.metric("פרויקט", doc["project_name"])

                    # חדרים
                    rooms = pretty.get("rooms", [])
                    if rooms:
                        st.markdown("---")
                        st.markdown("#### 🏠 חדרים שזוהו")

                        # 🆕 הגבלה ל-10 חדרים לחיסכון בזיכרון
                        for i, room in enumerate(rooms[:10], 1):
                            with st.expander(f"חדר {i}: {room.get('name', 'ללא שם')}"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    if room.get("area_m2"):
                                        st.write(f"**שטח:** {room['area_m2']} מ\"ר")
                                    if room.get("ceiling_height_m"):
                                        st.write(
                                            f"**גובה תקרה:** {room['ceiling_height_m']} מ'"
                                        )

                                with col2:
                                    if room.get("flooring_notes"):
                                        st.caption(
                                            f"**ריצוף:** {room['flooring_notes']}"
                                        )
                                    if room.get("other_notes"):
                                        st.caption(f"**הערות:** {room['other_notes']}")

                        if len(rooms) > 10:
                            st.caption(f"מציג 10 מתוך {len(rooms)} חדרים")

                    # מגבלות
                    if pretty.get("limitations"):
                        with st.expander("⚠️ מגבלות ובעיות"):
                            for lim in pretty["limitations"]:
                                st.warning(lim)

                with tab2_meta:
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        show_confidence = st.checkbox(
                            "הצג confidence scores", value=False
                        )

                    with col2:
                        import json as json_module

                        json_str = json_module.dumps(
                            llm_data, ensure_ascii=False, indent=2
                        )
                        st.download_button(
                            label="📥 הורד",
                            data=json_str,
                            file_name="metadata.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                    st.json(llm_data)
            else:
                st.warning("אין מטא-דאטה זמין")

            st.markdown("---")

            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state:
                st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state:
                st.session_state[scale_key] = proj["metadata"].get("scale", "")

            col_edit, col_preview = st.columns([1, 1.5], gap="large")

            with col_edit:
                st.markdown("### הגדרות תוכנית")

                # אינדיקטור תיקונים
                if selected in st.session_state.manual_corrections:
                    st.success("✏️ תוכנית זו תוקנה ידנית")

                p_name = st.text_input("שם התוכנית", key=name_key)
                p_scale_text = st.text_input(
                    "קנה מידה (לתיעוד)", key=scale_key, placeholder="1:50"
                )

                st.markdown("#### כיול")
                scale_val = st.slider(
                    "פיקסלים למטר",
                    10.0,
                    1000.0,
                    float(proj["scale"]),
                    key=f"scale_slider_{selected}",
                )
                proj["scale"] = scale_val

                # שימוש בגרסה המתוקנת
                corrected_walls = get_corrected_walls(selected, proj)
                corrected_pixels = np.count_nonzero(corrected_walls)
                total_len = corrected_pixels / scale_val

                # חישוב חומרים מהגרסה המתוקנת
                kernel = np.ones((6, 6), np.uint8)
                conc_corrected = cv2.dilate(
                    cv2.erode(corrected_walls, kernel, iterations=1),
                    kernel,
                    iterations=2,
                )
                block_corrected = cv2.subtract(corrected_walls, conc_corrected)

                conc_len = np.count_nonzero(conc_corrected) / scale_val
                block_len = np.count_nonzero(block_corrected) / scale_val
                floor_area = proj["metadata"].get("pixels_flooring_area", 0) / (
                    scale_val**2
                )

                proj["total_length"] = total_len

                st.info(
                    f"📏 קירות: {total_len:.1f}מ' | בטון: {conc_len:.1f}מ' | בלוקים: {block_len:.1f}מ' | ריצוף: {floor_area:.1f}מ\"ר"
                )
                # === 📐 מדידות מתקדמות ===
                with st.expander("📐 מדידות מתקדמות (Stage 1+2)", expanded=False):
                    meta = proj.get("metadata", {})

                    # תצוגת גודל נייר
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        paper_size = meta.get("paper_size_detected", "unknown")
                        st.metric("גודל נייר", paper_size)
                    with col_b:
                        conf = meta.get("paper_detection_confidence", 0) * 100
                        st.metric("ביטחון זיהוי", f"{conf:.0f}%")
                    with col_c:
                        scale_denom = meta.get("scale_denominator")
                        st.metric(
                            "קנה מידה", f"1:{scale_denom}" if scale_denom else "לא זוהה"
                        )
    # Debug - למה לא זוהה?
                    if not scale_denom:
                        st.markdown("---")
                        st.markdown("#### 🔍 למה קנה מידה לא זוהה?")

                        with st.container():
                            st.write("**מקורות שנבדקו:**")
                            st.write(f"1. meta['scale'] = `{meta.get('scale', 'לא קיים')}`")
                            st.write(f"2. meta['raw_text'][:200] = `{meta.get('raw_text', 'לא קיים')[:200]}`")

                            # ניסיון ידני
                            st.markdown("**ניסיון ידני:**")
                            manual_scale_text = st.text_input(
                                "הזן קנה מידה ידנית (לדוגמה: 1:50):", 
                                key=f"manual_scale_{selected}"
                            )

                            if manual_scale_text and st.button("החל", key=f"apply_scale_{selected}"):
                                from analyzer import parse_scale
                                parsed = parse_scale(manual_scale_text)
                                if parsed:
                                    meta["scale_denominator"] = parsed
                                    meta["scale"] = manual_scale_text
                                    st.success(f"✅ קנה מידה עודכן ל-1:{parsed}")
                                    st.rerun()
                                else:
                                    st.error("❌ לא הצלחתי לפרסר את הקנה מידה")

                    # תצוגת יחס המרה
                    if meta.get("meters_per_pixel"):
                    st.success(
                        f"✅ יחס המרה: **{meta['meters_per_pixel']*1000:.3f} מ\"מ/פיקסל** → **{meta['meters_per_pixel']:.6f} מ'/פיקסל**"
                    )

                    # אורך חישובי
                    if meta.get("wall_length_total_m"):
                        st.info(
                            f"📏 אורך קירות (מבוסס skeleton): **{meta['wall_length_total_m']:.2f} מטר**"
                        )
                else:
                    st.warning("⚠️ לא ניתן לחשב יחס המרה - חסר קנה מידה או גודל נייר")

                    # הצגת פרטים טכניים
                    if st.checkbox("הצג פרטים טכניים", key=f"show_tech_{selected}"):
                        st.json(
                            {
                                "paper_mm": meta.get("paper_mm"),
                                "image_size_px": meta.get("image_size_px"),
                                "mm_per_pixel": meta.get("mm_per_pixel"),
                                "skeleton_length_px": meta.get("wall_length_total_px"),
                                "method": meta.get("wall_length_method"),
                            }
                        )
                    # === בחירת גודל נייר ידנית ===
                    st.markdown("---")
                    st.markdown("### 📄 Override גודל נייר")
                    st.caption("אם הזיהוי האוטומטי לא מדויק, בחר ידנית:")

                    paper_override_key = f"paper_override_{selected}"

                    current_detected = meta.get("paper_size_detected", "unknown")

                    paper_options = ["זיהוי אוטומטי", "A0", "A1", "A2", "A3", "A4"]
                    default_idx = 0
                    if paper_override_key in st.session_state:
                        try:
                            default_idx = paper_options.index(
                                st.session_state[paper_override_key]
                            )
                        except ValueError:
                            default_idx = 0

                    selected_paper = st.selectbox(
                        f"גודל נייר (זוהה: {current_detected}):",
                        options=paper_options,
                        index=default_idx,
                        key=f"paper_select_{selected}",
                    )

                    if selected_paper != "זיהוי אוטומטי":
                        st.session_state[paper_override_key] = selected_paper

                        # חישוב מחדש עם override
                        ISO_SIZES = {
                            "A0": (841, 1189),
                            "A1": (594, 841),
                            "A2": (420, 594),
                            "A3": (297, 420),
                            "A4": (210, 297),
                        }

                        paper_w_mm, paper_h_mm = ISO_SIZES[selected_paper]

                        # עדכון metadata
                        meta["paper_size_detected"] = selected_paper
                        meta["paper_mm"] = {"width": paper_w_mm, "height": paper_h_mm}
                        meta["paper_detection_confidence"] = 1.0  # ידני = 100%

                        # חישוב mm_per_pixel מחדש
                        if meta.get("image_size_px"):
                            w_px = meta["image_size_px"]["width"]
                            h_px = meta["image_size_px"]["height"]

                            mm_per_pixel_x = paper_w_mm / w_px
                            mm_per_pixel_y = paper_h_mm / h_px
                            mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2

                            meta["mm_per_pixel"] = mm_per_pixel

                            # חישוב meters_per_pixel מחדש
                            scale_denom = meta.get("scale_denominator")
                            if scale_denom:
                                meters_per_pixel = (mm_per_pixel * scale_denom) / 1000
                                meta["meters_per_pixel"] = meters_per_pixel

                                # חישוב אורך קירות מחדש
                                if meta.get("wall_length_total_px"):
                                    wall_length_m = (
                                        meta["wall_length_total_px"] * meters_per_pixel
                                    )
                                    meta["wall_length_total_m"] = wall_length_m

                        st.success(f"✅ גודל נייר עודכן ל-{selected_paper}")
                        st.rerun()
                    elif paper_override_key in st.session_state:
                        # אם חזר ל-"זיהוי אוטומטי", נקה את ה-override
                        del st.session_state[paper_override_key]

                    # === Stage 3: כיול ידני ===
                    st.markdown("---")
                    st.markdown("### 🎯 Stage 3: כיול ידני (Override)")
                    st.caption("צייר קו על מימד ידוע בתוכנית והזן את האורך האמיתי")

                    # בדיקה אם יש כבר כיול
                    calibration_key = f"calibration_{selected}"
                    if calibration_key in st.session_state:
                        cal = st.session_state[calibration_key]
                        st.success(
                            f"✅ כיול פעיל: **{cal['true_m']:.2f} מ'** = {cal['pixel_len']:.1f} פיקסלים"
                        )
                        st.metric(
                            "יחס מכויל", f"{cal['meters_per_pixel']:.6f} מ'/פיקסל"
                        )

                        if st.button("🔄 אפס כיול", key=f"reset_cal_{selected}"):
                            del st.session_state[calibration_key]
                            st.rerun()
                    else:
                        # קנבס לציור קו
                        st.markdown("**1️⃣ צייר קו על מימד ידוע:**")

                        rgb_cal = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                        h_cal, w_cal = rgb_cal.shape[:2]
                        scale_cal = min(1.0, 800 / max(w_cal, h_cal))

                        img_cal = Image.fromarray(rgb_cal).resize(
                            (int(w_cal * scale_cal), int(h_cal * scale_cal))
                        )

                        canvas_cal = st_canvas(
                            fill_color="rgba(0,0,0,0)",
                            stroke_width=3,
                            stroke_color="#FF0000",
                            background_image=img_cal,
                            height=int(h_cal * scale_cal),
                            width=int(w_cal * scale_cal),
                            drawing_mode="line",
                            key=f"calibration_canvas_{selected}",
                            update_streamlit=True,
                        )

                        if canvas_cal.json_data and canvas_cal.json_data["objects"]:
                            # יש קו מצויר
                            line = canvas_cal.json_data["objects"][-1]

                            x1 = line["x1"] / scale_cal
                            y1 = line["y1"] / scale_cal
                            x2 = line["x2"] / scale_cal
                            y2 = line["y2"] / scale_cal

                            pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            st.info(f"📏 אורך הקו: **{pixel_distance:.1f} פיקסלים**")

                            # קלט אורך אמיתי
                            st.markdown("**2️⃣ הזן אורך אמיתי (מטרים):**")
                            true_meters = st.number_input(
                                "אורך במטרים:",
                                min_value=0.1,
                                max_value=100.0,
                                value=1.0,
                                step=0.1,
                                key=f"true_m_{selected}",
                            )

                            if st.button(
                                "✅ החל כיול",
                                type="primary",
                                key=f"apply_cal_{selected}",
                            ):
                                # שמירת כיול
                                meters_per_px_cal = true_meters / pixel_distance

                                st.session_state[calibration_key] = {
                                    "method": "manual",
                                    "true_m": true_meters,
                                    "pixel_len": pixel_distance,
                                    "meters_per_pixel": meters_per_px_cal,
                                }

                                # עדכון metadata
                                proj["metadata"]["calibration"] = st.session_state[
                                    calibration_key
                                ]

                                # חישוב מחדש עם כיול
                                corrected_walls_cal = get_corrected_walls(
                                    selected, proj
                                )
                                skeleton_cal = (
                                    cv2.ximgproc.thinning(corrected_walls_cal)
                                    if hasattr(cv2, "ximgproc")
                                    else corrected_walls_cal
                                )
                                skeleton_len_cal = compute_skeleton_length_px(
                                    skeleton_cal
                                )
                                wall_len_cal = skeleton_len_cal * meters_per_px_cal

                                proj["metadata"][
                                    "wall_length_calibrated_m"
                                ] = wall_len_cal

                                st.success(
                                    f"✅ כיול הוחל! אורך חדש: **{wall_len_cal:.2f} מטר**"
                                )
                                st.rerun()

                # מחשבון הצעת מחיר
                with st.expander("💰 מחשבון הצעת מחיר", expanded=False):
                    st.markdown(
                        """<div style="background:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:10px;">
                    <strong>מחירון בסיס:</strong> בטון 1200₪/מ' | בלוקים 600₪/מ' | ריצוף 250₪/מ\"ר
                    </div>""",
                        unsafe_allow_html=True,
                    )

                    c_price = st.number_input(
                        "מחיר בטון (₪/מ')",
                        value=1200.0,
                        step=50.0,
                        key=f"c_price_{selected}",
                    )
                    b_price = st.number_input(
                        "מחיר בלוקים (₪/מ')",
                        value=600.0,
                        step=50.0,
                        key=f"b_price_{selected}",
                    )
                    f_price = st.number_input(
                        'מחיר ריצוף (₪/מ"ר)',
                        value=250.0,
                        step=50.0,
                        key=f"f_price_{selected}",
                    )

                    total_quote = (
                        (conc_len * c_price)
                        + (block_len * b_price)
                        + (floor_area * f_price)
                    )
                    st.markdown(f'#### 💵 סה"כ הצעת מחיר: {total_quote:,.0f} ₪')

                    quote_df = pd.DataFrame(
                        {
                            "פריט": [
                                "קירות בטון",
                                "קירות בלוקים",
                                "ריצוף/חיפוי",
                                'סה"כ',
                            ],
                            "יחידה": ["מ'", "מ'", 'מ"ר', "-"],
                            "כמות": [
                                f"{conc_len:.2f}",
                                f"{block_len:.2f}",
                                f"{floor_area:.2f}",
                                "-",
                            ],
                            "מחיר יחידה": [
                                f"{c_price:.0f}₪",
                                f"{b_price:.0f}₪",
                                f"{f_price:.0f}₪",
                                "-",
                            ],
                            'סה"כ': [
                                f"{conc_len*c_price:,.0f}₪",
                                f"{block_len*b_price:,.0f}₪",
                                f"{floor_area*f_price:,.0f}₪",
                                f"{total_quote:,.0f}₪",
                            ],
                        }
                    )
                    st.dataframe(quote_df, hide_index=True, use_container_width=True)

                st.markdown("---")
                if st.button(
                    "💾 שמור תוכנית למערכת", type="primary", key=f"save_{selected}"
                ):
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale_text
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = json.dumps(
                        {
                            "concrete_length": conc_len,
                            "blocks_length": block_len,
                            "flooring_area": floor_area,
                        },
                        ensure_ascii=False,
                    )

                    plan_id = save_plan(
                        selected,
                        p_name,
                        p_scale_text,
                        scale_val,
                        corrected_pixels,
                        meta_json,
                        None,
                        0,
                        0,
                        materials,
                    )
                    st.toast("✅ נשמר למערכת!")
                    st.success(f"התוכנית נשמרה בהצלחה (ID: {plan_id})")

            with col_preview:
                st.markdown("### תצוגה מקדימה")

                if selected in st.session_state.manual_corrections:
                    st.caption("✏️ גרסה מתוקנת ידנית")

                show_flooring = st.checkbox(
                    "הצג ריצוף", value=True, key=f"show_flooring_{selected}"
                )

                # שימוש בגרסה המתוקנת
                corrected_walls_display = get_corrected_walls(selected, proj)

                kernel_display = np.ones((6, 6), np.uint8)
                concrete_corrected = cv2.dilate(
                    cv2.erode(corrected_walls_display, kernel_display, iterations=1),
                    kernel_display,
                    iterations=2,
                )
                blocks_corrected = cv2.subtract(
                    corrected_walls_display, concrete_corrected
                )

                floor_mask = proj["flooring_mask"] if show_flooring else None

                overlay = create_colored_overlay(
                    proj["original"], concrete_corrected, blocks_corrected, floor_mask
                )
                st.image(overlay, use_column_width=True)
                st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")

                # ========== תכונה: ניתוח מקרא ==========
                st.markdown("---")
                with st.expander("🎨 נתח מקרא (AI)", expanded=False):
                    st.caption(
                        "המערכת תנסה למצוא את המקרא אוטומטית, או שאתה יכול לחתוך ידנית"
                    )

                    # כפתור זיהוי אוטומטי
                    col_auto, col_manual = st.columns([1, 1])

                    with col_auto:
                        if st.button(
                            "🔍 מצא מקרא אוטומטית",
                            key=f"auto_legend_{selected}",
                            use_container_width=True,
                        ):
                            with st.spinner("מחפש מקרא..."):
                                try:
                                    analyzer_temp = FloorPlanAnalyzer()
                                    legend_bbox = analyzer_temp.auto_detect_legend(
                                        proj["original"]
                                    )

                                    if legend_bbox:
                                        x, y, w, h = legend_bbox

                                        # חיתוך והצגה
                                        cropped = proj["original"][y : y + h, x : x + w]
                                        cropped_rgb = cv2.cvtColor(
                                            cropped, cv2.COLOR_BGR2RGB
                                        )

                                        st.success("✅ נמצא מקרא!")
                                        st.image(
                                            cropped_rgb,
                                            caption=f"מקרא שזוהה (גודל: {w}x{h}px)",
                                            width=400,
                                        )

                                        # שמירה ב-session
                                        if "auto_legend" not in st.session_state:
                                            st.session_state.auto_legend = {}
                                        st.session_state.auto_legend[selected] = cropped

                                        # כפתור ניתוח
                                        if st.button(
                                            "📝 נתח מקרא זה",
                                            key=f"analyze_auto_{selected}",
                                        ):
                                            with st.spinner("מנתח עם Claude AI..."):
                                                _, buffer = cv2.imencode(
                                                    ".png", cropped
                                                )
                                                image_bytes = buffer.tobytes()

                                                result = safe_analyze_legend(
                                                    image_bytes
                                                )

                                                if (
                                                    isinstance(result, dict)
                                                    and "error" not in result
                                                ):
                                                    st.success("✅ ניתוח הושלם!")

                                                    col_a, col_b = st.columns(2)
                                                    with col_a:
                                                        st.metric(
                                                            "סוג תוכנית",
                                                            result.get(
                                                                "plan_type", "לא זוהה"
                                                            ),
                                                        )
                                                        st.metric(
                                                            "רמת ביטחון",
                                                            f"{result.get('confidence', 0)}%",
                                                        )

                                                    with col_b:
                                                        if result.get(
                                                            "materials_found"
                                                        ):
                                                            st.markdown(
                                                                "**חומרים שזוהו:**"
                                                            )
                                                            for material in result[
                                                                "materials_found"
                                                            ]:
                                                                st.markdown(
                                                                    f"- {material}"
                                                                )

                                                    if result.get("symbols"):
                                                        st.markdown("**סמלים:**")
                                                        for symbol in result["symbols"][
                                                            :5
                                                        ]:
                                                            st.markdown(
                                                                f"- **{symbol.get('symbol', '')}**: {symbol.get('meaning', '')}"
                                                            )

                                                    if result.get("notes"):
                                                        st.info(f"💡 {result['notes']}")

                                                    proj["metadata"][
                                                        "legend_analysis"
                                                    ] = result
                                                else:
                                                    st.error(
                                                        f"❌ {result.get('error', 'שגיאה לא ידועה')}"
                                                    )
                                    else:
                                        st.warning(
                                            "⚠️ לא נמצא מקרא אוטומטית. נסה לחתוך ידנית למטה."
                                        )
                                        st.caption(
                                            "💡 טיפ: המקרא בדרך כלל בפינה או בצד של התוכנית"
                                        )

                                except Exception as e:
                                    st.error(f"❌ שגיאה: {str(e)}")

                    with col_manual:
                        st.markdown("**או:**")
                        st.caption("צייר ריבוע סביב המקרא ידנית ↓")

                    st.markdown("---")
                    st.markdown("### חיתוך ידני")

                    # המרה נכונה של התמונה
                    rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                    h_img, w_img = rgb.shape[:2]

                    # רזולוציה גבוהה יותר לחיתוך מדויק
                    scale_factor = min(1.0, 1200 / max(w_img, h_img))

                    new_w = int(w_img * scale_factor)
                    new_h = int(h_img * scale_factor)

                    # המרה ל-PIL ושינוי גודל
                    pil_image = Image.fromarray(rgb.astype("uint8"), "RGB")
                    pil_image_resized = pil_image.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS
                    )

                    legend_canvas = st_canvas(
                        fill_color="rgba(255,0,0,0.1)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=pil_image_resized,
                        height=new_h,
                        width=new_w,
                        drawing_mode="rect",
                        key=f"legend_canvas_{selected}",
                        update_streamlit=True,
                    )

                    if legend_canvas.json_data and legend_canvas.json_data["objects"]:
                        if st.button(
                            "🔍 נתח מקרא עם AI", key=f"analyze_legend_{selected}"
                        ):
                            with st.spinner("מנתח מקרא..."):
                                try:
                                    # חילוץ הריבוע שצויר
                                    rect = legend_canvas.json_data["objects"][-1]
                                    x = int(rect["left"] / scale_factor)
                                    y = int(rect["top"] / scale_factor)
                                    rect_w = int(rect["width"] / scale_factor)
                                    rect_h = int(rect["height"] / scale_factor)

                                    # חיתוך האזור מהתמונה המקורית
                                    cropped = proj["original"][
                                        y : y + rect_h, x : x + rect_w
                                    ]

                                    # המרה ל-bytes
                                    _, buffer = cv2.imencode(".png", cropped)
                                    image_bytes = buffer.tobytes()

                                    # ניתוח עם Claude
                                    result = safe_analyze_legend(image_bytes)

                                    if (
                                        isinstance(result, dict)
                                        and "error" not in result
                                    ):
                                        # הצגת תוצאות
                                        st.success("✅ ניתוח הושלם!")

                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric(
                                                "סוג תוכנית",
                                                result.get("plan_type", "לא זוהה"),
                                            )
                                            st.metric(
                                                "רמת ביטחון",
                                                f"{result.get('confidence', 0)}%",
                                            )

                                        with col_b:
                                            if result.get("materials_found"):
                                                st.markdown("**חומרים שזוהו:**")
                                                for material in result[
                                                    "materials_found"
                                                ]:
                                                    st.markdown(f"- {material}")

                                        if result.get("symbols"):
                                            st.markdown("**סמלים:**")
                                            for symbol in result["symbols"][:5]:
                                                st.markdown(
                                                    f"- **{symbol.get('symbol', '')}**: {symbol.get('meaning', '')}"
                                                )

                                        if result.get("notes"):
                                            st.info(f"💡 {result['notes']}")

                                        # שמירה למטא-דאטה
                                        proj["metadata"]["legend_analysis"] = result

                                    elif isinstance(result, dict) and "error" in result:
                                        st.error(f"שגיאה: {result['error']}")
                                    else:
                                        st.warning(f"תשובה לא צפויה: {result}")

                                except Exception as e:
                                    st.error(f"שגיאה בניתוח: {str(e)}")
                                    import traceback

                                    with st.expander("פרטי שגיאה"):
                                        st.code(traceback.format_exc())
                    else:
                        st.info("👆 צייר ריבוע סביב המקרא בתוכנית ולחץ על הכפתור")

    # ==========================================
    # 🎨 טאב 2: תיקונים ידניים
    # ==========================================
    with tab2:
        st.markdown("## 🎨 תיקונים ידניים")
        st.caption("הוסף או הסר קירות באופן ידני למדויקות מקסימלית")

        if not st.session_state.projects:
            st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        else:
            selected_plan = st.selectbox(
                "בחר תוכנית לתיקון:",
                list(st.session_state.projects.keys()),
                key="correction_plan_select",
            )
            proj = st.session_state.projects[selected_plan]

            correction_mode = st.radio(
                "מצב תיקון:",
                ["➕ הוסף קירות חסרים", "➖ הסר קירות מזויפים", "👁️ השוואה"],
                horizontal=True,
            )

            rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
            h_img, w_img = rgb.shape[:2]
            scale_factor = 1000 / w_img if w_img > 1000 else 1.0
            img_display = Image.fromarray(rgb).resize(
                (int(w_img * scale_factor), int(h_img * scale_factor))
            )

            if correction_mode == "➕ הוסף קירות חסרים":
                st.info("🖌️ צייר בירוק על הקירות שהמערכת החמיצה")

                canvas_add = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=6,
                    stroke_color="#00FF00",
                    background_image=img_display,
                    height=int(h_img * scale_factor),
                    width=int(w_img * scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_add_{selected_plan}",
                    update_streamlit=True,
                )

                if canvas_add.image_data is not None and np.any(
                    canvas_add.image_data[:, :, 3] > 0
                ):
                    if st.button("✅ אשר הוספה", key="confirm_add"):
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}

                        added_mask = cv2.resize(
                            canvas_add.image_data[:, :, 3],
                            (w_img, h_img),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        added_mask = (added_mask > 0).astype(np.uint8) * 255

                        st.session_state.manual_corrections[selected_plan][
                            "added_walls"
                        ] = added_mask
                        st.success("✅ קירות נוספו! עבור לטאב 'השוואה' לראות את התוצאה")
                        st.rerun()

            elif correction_mode == "➖ הסר קירות מזויפים":
                st.info("🖌️ צייר באדום על קירות שהמערכת זיהתה בטעות")

                walls_overlay = proj["thick_walls"].copy()
                walls_colored = cv2.cvtColor(walls_overlay, cv2.COLOR_GRAY2RGB)
                walls_colored[walls_overlay > 0] = [0, 255, 255]

                combined = cv2.addWeighted(rgb, 0.6, walls_colored, 0.4, 0)
                combined_resized = cv2.resize(
                    combined, (int(w_img * scale_factor), int(h_img * scale_factor))
                )
                img_with_walls = Image.fromarray(combined_resized)

                canvas_remove = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=8,
                    stroke_color="#FF0000",
                    background_image=img_with_walls,
                    height=int(h_img * scale_factor),
                    width=int(w_img * scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_remove_{selected_plan}",
                    update_streamlit=True,
                )

                if canvas_remove.image_data is not None and np.any(
                    canvas_remove.image_data[:, :, 3] > 0
                ):
                    if st.button("✅ אשר הסרה", key="confirm_remove"):
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}

                        removed_mask = cv2.resize(
                            canvas_remove.image_data[:, :, 3],
                            (w_img, h_img),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        removed_mask = (removed_mask > 0).astype(np.uint8) * 255

                        st.session_state.manual_corrections[selected_plan][
                            "removed_walls"
                        ] = removed_mask
                        st.success("✅ קירות הוסרו! עבור לטאב 'השוואה' לראות את התוצאה")
                        st.rerun()

            elif correction_mode == "👁️ השוואה":
                st.markdown("### לפני ואחרי")

                if selected_plan in st.session_state.manual_corrections:
                    corrected_walls = get_corrected_walls(selected_plan, proj)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 🤖 זיהוי אוטומטי")
                        auto_overlay = rgb.copy()
                        auto_overlay[proj["thick_walls"] > 0] = [0, 255, 0]
                        st.image(auto_overlay, use_column_width=True)

                        auto_pixels = np.count_nonzero(proj["thick_walls"])
                        auto_length = auto_pixels / proj["scale"]
                        st.metric("אורך", f"{auto_length:.1f} מ'")

                    with col2:
                        st.markdown("#### ✅ אחרי תיקון")
                        corrected_overlay = rgb.copy()
                        corrected_overlay[corrected_walls > 0] = [255, 165, 0]
                        st.image(corrected_overlay, use_column_width=True)

                        corrected_pixels = np.count_nonzero(corrected_walls)
                        corrected_length = corrected_pixels / proj["scale"]
                        st.metric(
                            "אורך",
                            f"{corrected_length:.1f} מ'",
                            delta=f"{corrected_length - auto_length:+.1f} מ'",
                        )

                    st.markdown("---")
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("💾 שמור גרסה מתוקנת", type="primary"):
                            proj["thick_walls"] = corrected_walls
                            proj["raw_pixels"] = corrected_pixels
                            proj["total_length"] = corrected_length

                            meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                            save_plan(
                                selected_plan,
                                proj["metadata"].get("plan_name"),
                                "1:50",
                                proj["scale"],
                                corrected_pixels,
                                meta_json,
                            )

                            st.success("✅ הגרסה המתוקנת נשמרה!")

                    with col_btn2:
                        if st.button("🔄 אפס תיקונים", key="reset_corrections"):
                            del st.session_state.manual_corrections[selected_plan]
                            st.success("התיקונים אופסו")
                            st.rerun()
                else:
                    st.info(
                        "אין תיקונים ידניים עדיין. עבור לטאב 'הוסף קירות' או 'הסר קירות'"
                    )

    # --- טאב 3: דשבורד ---
    with tab3:
        st.header("📊 דשבורד פרויקט")

        all_plans = get_all_plans()
        if not all_plans:
            st.info("🔍 אין פרויקטים במערכת. העלה תוכנית בסדנת עבודה.")
        else:
            # בחירת פרויקט
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan_dash = st.selectbox(
                "📂 בחר פרויקט:", plan_options, key="dashboard_plan_select"
            )
            plan_id = int(selected_plan_dash.split("ID: ")[1].strip(")"))

            # טעינת נתונים
            forecast = get_project_forecast(plan_id)
            financial = get_project_financial_status(plan_id)
            plan_data = get_plan_by_id(plan_id)

            # === KPIs מורחבים ===
            st.markdown("### 📈 מדדי ביצוע")

            k1, k2, k3, k4 = st.columns(4)

            total = forecast.get("total_planned", 0)
            built = forecast.get("cumulative_progress", 0)
            percent = (built / total * 100) if total > 0 else 0
            remaining = total - built

            with k1:
                st.metric(
                    label="📏 סך הכל מתוכנן",
                    value=f"{total:.1f} מ'",
                    help="סך כל הקירות שזוהו בתוכנית",
                )

            with k2:
                st.metric(
                    label="✅ בוצע בפועל",
                    value=f"{built:.1f} מ'",
                    delta=f"{percent:.1f}%",
                    delta_color="normal",
                    help="סך כל הדיווחים מצטבר",
                )

            with k3:
                st.metric(
                    label="⏳ נותר לביצוע",
                    value=f"{remaining:.1f} מ'",
                    delta=f"{forecast.get('days_to_finish', 0)} ימים",
                    delta_color="inverse",
                    help='תחזית עפ"י קצב ביצוע נוכחי',
                )

            with k4:
                budget = financial.get("budget_limit", 0)
                cost = financial.get("current_cost", 0)
                variance = budget - cost
                st.metric(
                    label="💰 עלות מצטברת",
                    value=f"{cost:,.0f} ₪",
                    delta=f"{variance:,.0f} ₪ {'תקציב' if variance >= 0 else 'חריגה'}",
                    delta_color="normal" if variance >= 0 else "inverse",
                    help=f"תקציב: {budget:,.0f} ₪",
                )

            # === Progress Bar ויזואלי ===
            st.markdown("---")
            st.markdown("### 📊 התקדמות כללית")

            # צבע דינמי
            if percent < 30:
                color = "#EF4444"  # אדום
            elif percent < 70:
                color = "#F59E0B"  # כתום
            else:
                color = "#10B981"  # ירוק

            progress_html = f"""
            <div style="margin: 1.5rem 0;">
                <div style="width: 100%; background: #e5e7eb; border-radius: 12px; height: 40px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="
                        width: {percent}%; 
                        background: linear-gradient(90deg, {color}, {color}dd); 
                        height: 100%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        color: white; 
                        font-weight: bold; 
                        font-size: 18px; 
                        transition: width 0.5s ease;
                        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
                    ">
                        {percent:.1f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.875rem; color: #6b7280; font-weight: 500;">
                    <span>🚀 התחלה</span>
                    <span>📍 {built:.1f} מ' מתוך {total:.1f} מ'</span>
                    <span>🎯 סיום</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)

            # === גרף התקדמות ===
            st.markdown("---")
            st.markdown("### 📈 גרף התקדמות לאורך זמן")

            df_stats = load_stats_df()
            if not df_stats.empty:
                # סינון לפרויקט הנוכחי
                df_current = df_stats[df_stats["שם תוכנית"] == plan_data["plan_name"]]

                if not df_current.empty:
                    # הצגת הגרף
                    st.bar_chart(
                        df_current, x="תאריך", y="כמות שבוצעה", use_container_width=True
                    )

                    # סטטיסטיקות נוספות
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("📋 מספר דיווחים", len(df_current))
                    with col_b:
                        avg_daily = df_current["כמות שבוצעה"].mean()
                        st.metric("📊 ממוצע יומי", f"{avg_daily:.1f} מ'")
                    with col_c:
                        max_day = df_current["כמות שבוצעה"].max()
                        st.metric("⭐ יום שיא", f"{max_day:.1f} מ'")
                else:
                    st.info("📭 אין דיווחים לפרויקט זה עדיין")
            else:
                st.info("📭 אין דיווחים במערכת")

            # === כפתורי פעולה ===
            st.markdown("---")
            st.markdown("### 🎯 פעולות ודוחות")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📄 הדפס דוח PDF",
                    use_container_width=True,
                    type="primary",
                    key="pdf_button_dash",
                ):
                    with st.spinner("🔄 מכין דוח מפורט..."):
                        try:
                            # ניסיון למצוא תמונה
                            if (
                                selected_plan
                                and selected_plan in st.session_state.projects
                            ):
                                proj = st.session_state.projects[selected_plan]
                                rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                            else:
                                # תמונה ריקה אם אין
                                rgb = np.ones((800, 1200, 3), dtype=np.uint8) * 255
                                cv2.putText(
                                    rgb,
                                    "Image Not Available",
                                    (350, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (150, 150, 150),
                                    3,
                                )

                            # הכנת נתונים
                            stats = {
                                "built": built,
                                "total": total,
                                "percent": percent,
                                "remaining": remaining,
                                "cost": cost,
                                "budget": budget,
                            }

                            # יצירת PDF
                            pdf_buffer = generate_status_pdf(
                                plan_data["plan_name"], rgb, stats
                            )

                            # כפתור הורדה
                            st.download_button(
                                label="⬇️ הורד דוח PDF",
                                data=pdf_buffer,
                                file_name=f"status_report_{plan_data['plan_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                type="secondary",
                                key="download_pdf_dash",
                            )
                            st.success("✅ הדוח מוכן להורדה!")

                        except Exception as e:
                            st.error(f"❌ שגיאה ביצירת דוח: {str(e)}")
                            st.info("💡 ודא שהקובץ reporter.py קיים ותקין")

            with col2:
                if st.button(
                    "📊 ייצא נתונים", use_container_width=True, key="export_button_dash"
                ):
                    st.info("💡 תכונה בפיתוח: ייצוא ל-Excel")

            with col3:
                if st.button(
                    '📧 שלח דוא"ל', use_container_width=True, key="email_button_dash"
                ):
                    st.info("💡 תכונה בפיתוח: שליחת דוח באימייל")

            # === טבלת דיווחים אחרונים ===
            st.markdown("---")
            st.markdown("### 📋 דיווחים אחרונים")

            reports = get_progress_reports(plan_id)
            if reports:
                # הצג 5 אחרונים
                recent = reports[:5]

                for i, r in enumerate(recent, 1):
                    # צבע לפי גודל דיווח
                    meters = r["meters_built"]
                    if meters > 20:
                        icon = "🟢"
                    elif meters > 10:
                        icon = "🟡"
                    else:
                        icon = "🔴"

                    with st.expander(
                        f"{icon} {r['date']} - {meters:.1f} מ' - {r.get('note', 'אין הערה')}"
                    ):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**📏 כמות:** {meters:.1f} מטרים")
                            if r.get("note"):
                                st.write(f"**📝 הערה:** {r['note']}")
                            st.caption(f"תאריך: {r['date']}")
                        with col_b:
                            st.metric("דיווח #", i)
                            st.caption(f"ID: {r['id']}")

                # סיכום
                total_reports = len(reports)
                if total_reports > 5:
                    st.caption(f"📌 מציג 5 מתוך {total_reports} דיווחים")
            else:
                st.info("📭 אין דיווחים לפרויקט זה. התחל לדווח בסדנת עבודה!")

    # ==========================================
    # 💰 טאב 4: חשבונות חלקיים
    # ==========================================
    with tab4:
        st.markdown("## 💰 מחולל חשבונות חלקיים")
        st.caption("הפקת חשבונית לתשלום לקבלן על בסיס ביצוע בפועל")

        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים במערכת")
        else:
            # בחירת פרויקט
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan_invoice = st.selectbox(
                "בחר פרויקט:", plan_options, key="invoice_plan_select"
            )
            plan_id = int(selected_plan_invoice.split("ID: ")[1].strip(")"))

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 📅 בחר טווח תאריכים")

                # טווח מהיר
                quick_range = st.radio(
                    "בחירה מהירה:",
                    ["שבוע אחרון", "חודש אחרון", "טווח מותאם אישית"],
                    horizontal=True,
                )

                from datetime import datetime, timedelta

                if quick_range == "שבוע אחרון":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                elif quick_range == "חודש אחרון":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                else:  # טווח מותאם אישית
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input(
                            "מתאריך:",
                            value=datetime.now() - timedelta(days=30),
                            key="start_date_picker",
                        )
                    with col_date2:
                        end_date = st.date_input(
                            "עד תאריך:", value=datetime.now(), key="end_date_picker"
                        )

                # המרה ל-string
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                st.info(f"📊 תקופת החשבון: {start_str} עד {end_str}")

                # === הגדרת מחירי יחידה ===
                st.markdown("### 💵 מחירי יחידה")

                # קבלת סוגי עבודות
                work_types = get_all_work_types_for_plan(plan_id)

                if not work_types:
                    st.warning("אין דיווחים לפרויקט זה עדיין")
                else:
                    st.caption(
                        "ערוך את המחירים לפי הצורך. המחירים המוצגים הם ברירות מחדל."
                    )

                    unit_prices = {}

                    for work_type in work_types:
                        # מחיר ברירת מחדל
                        if "ריצוף" in work_type.lower() or "חיפוי" in work_type.lower():
                            default_price = 250
                            unit = 'מ"ר'
                        elif "בטון" in work_type.lower():
                            default_price = 1200
                            unit = "מ'"
                        elif "בלוק" in work_type.lower():
                            default_price = 600
                            unit = "מ'"
                        else:
                            default_price = 800
                            unit = "מ'"

                        col_type, col_price = st.columns([2, 1])
                        with col_type:
                            st.markdown(f"**{work_type}** ({unit})")
                        with col_price:
                            price = st.number_input(
                                "מחיר:",
                                value=float(default_price),
                                step=50.0,
                                key=f"price_{work_type}",
                                label_visibility="collapsed",
                            )
                            unit_prices[work_type] = price

            with col2:
                st.markdown("### 👷 פרטי קבלן")
                st.caption("שדות אלה יופיעו בחשבונית")

                contractor_name = st.text_input(
                    "שם הקבלן:",
                    value="",
                    placeholder="ישראל ישראלי",
                    key="contractor_name",
                )

                contractor_company = st.text_input(
                    "שם חברה:",
                    value="",
                    placeholder='בניית ישראל בע"מ',
                    key="contractor_company",
                )

                contractor_vat = st.text_input(
                    "ח.פ / ע.מ:",
                    value="",
                    placeholder="123456789",
                    key="contractor_vat",
                )

                contractor_address = st.text_area(
                    "כתובת:",
                    value="",
                    placeholder="רחוב הבניינים 1, תל אביב",
                    height=80,
                    key="contractor_address",
                )

                st.markdown("---")

                # כפתור יצירת חשבונית
                if st.button(
                    "🧾 צור חשבונית", type="primary", use_container_width=True
                ):
                    # בדיקת שדות חובה
                    if not contractor_name or not contractor_vat:
                        st.error("❌ יש למלא שם קבלן ומספר עוסק")
                    else:
                        with st.spinner("מכין חשבונית..."):
                            try:
                                # קבלת נתוני חשבון
                                invoice_data = get_payment_invoice_data(
                                    plan_id, start_str, end_str, unit_prices
                                )

                                if invoice_data.get("error"):
                                    st.error(f"❌ {invoice_data['error']}")
                                elif not invoice_data["items"]:
                                    st.warning("⚠️ אין דיווחים בטווח התאריכים הזה")
                                else:
                                    # פרטי קבלן
                                    contractor_info = {
                                        "name": contractor_name,
                                        "company": contractor_company,
                                        "vat_id": contractor_vat,
                                        "address": contractor_address,
                                    }

                                    # יצירת PDF
                                    pdf_buffer = generate_payment_invoice_pdf(
                                        invoice_data, contractor_info
                                    )

                                    # הצגת סיכום
                                    st.success("✅ החשבונית הוכנה בהצלחה!")

                                    st.markdown("### 📋 סיכום החשבונית")

                                    # טבלת פריטים
                                    import pandas as pd

                                    df_items = pd.DataFrame(
                                        [
                                            {
                                                "סוג עבודה": item["work_type"],
                                                "כמות": f"{item['quantity']:.2f}",
                                                "יחידה": item["unit"],
                                                "מחיר יחידה": f"{item['unit_price']:,.0f} ₪",
                                                'סה"כ': f"{item['subtotal']:,.2f} ₪",
                                            }
                                            for item in invoice_data["items"]
                                        ]
                                    )

                                    st.dataframe(
                                        df_items,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    # סיכום סופי
                                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                                    with col_sum1:
                                        st.metric(
                                            "סכום ביניים",
                                            f"{invoice_data['total_amount']:,.2f} ₪",
                                        )
                                    with col_sum2:
                                        st.metric(
                                            'מע"מ (17%)',
                                            f"{invoice_data['vat']:,.2f} ₪",
                                        )
                                    with col_sum3:
                                        st.metric(
                                            '**סה"כ לתשלום**',
                                            f"{invoice_data['total_with_vat']:,.2f} ₪",
                                        )

                                    # כפתור הורדה
                                    st.download_button(
                                        label="📥 הורד חשבונית (PDF)",
                                        data=pdf_buffer,
                                        file_name=f"invoice_{invoice_data['plan']['plan_name']}_{start_str}_{end_str}.pdf",
                                        mime="application/pdf",
                                        type="primary",
                                        use_container_width=True,
                                    )

                            except Exception as e:
                                st.error(f"❌ שגיאה ביצירת חשבונית: {str(e)}")
                                import traceback

                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())

            # תצוגה מקדימה של דיווחים
            st.markdown("---")
            with st.expander("📊 דיווחים בטווח התאריכים"):
                summary = get_progress_summary_by_date_range(
                    plan_id, start_str, end_str
                )
                if summary:
                    import pandas as pd

                    df_summary = pd.DataFrame(
                        [
                            {
                                "סוג עבודה": item["work_type"],
                                "כמות כוללת": f"{item['total_quantity']:.2f}",
                                "יחידה": item["unit"],
                                "מספר דיווחים": item["report_count"],
                            }
                            for item in summary
                        ]
                    )
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("אין דיווחים בטווח זה")

# ==========================================
# 👷 מצב דיווח
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")

    if not st.session_state.projects:
        st.warning("אין תוכניות זמינות")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]

        report_type = st.radio(
            "סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True
        )

        # שימוש בגרסה המתוקנת
        corrected_walls = get_corrected_walls(plan_name, proj)

        rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        scale_factor = 800 / w_img if w_img > 800 else 1.0
        img_resized = Image.fromarray(rgb).resize(
            (int(w_img * scale_factor), int(h_img * scale_factor))
        )

        if "קירות" in report_type:
            fill = "rgba(0,0,0,0)"
            stroke = "#00FF00"
            stroke_width = 8
        else:
            fill = "rgba(255,255,0,0.3)"
            stroke = "#FFFF00"
            stroke_width = 20

        canvas = st_canvas(
            fill_color=fill,
            stroke_color=stroke,
            stroke_width=stroke_width,
            background_image=img_resized,
            height=int(h_img * scale_factor),
            width=int(w_img * scale_factor),
            drawing_mode="freedraw",
            key=f"canvas_{plan_name}_{report_type}",
            update_streamlit=True,
        )

        if (
            canvas.json_data
            and canvas.json_data["objects"]
            and canvas.image_data is not None
        ):
            measured = 0
            if "קירות" in report_type:
                user_draw = canvas.image_data[:, :, 3] > 0
                walls_resized = cv2.resize(
                    corrected_walls,
                    (int(w_img * scale_factor), int(h_img * scale_factor)),
                )
                intersection = np.logical_and(user_draw, walls_resized > 0)
                measured = np.count_nonzero(intersection) / scale_factor / proj["scale"]
            else:
                pixels = np.count_nonzero(canvas.image_data[:, :, 3] > 0)
                measured = pixels / ((proj["scale"] * scale_factor) ** 2)

            if measured > 0:
                unit = 'מ"ר' if "ריצוף" in report_type else "מטר"
                st.success(f"✅ {measured:.2f} {unit}")

                if st.button("🚀 שלח דיווח", type="primary"):
                    rec = get_plan_by_filename(plan_name)
                    pid = (
                        rec["id"]
                        if rec
                        else save_plan(
                            plan_name,
                            plan_name,
                            "1:50",
                            proj["scale"],
                            proj["raw_pixels"],
                            "{}",
                        )
                    )
                    save_progress_report(pid, measured, report_type)
                    st.success("הדיווח נשמר בהצלחה!")
