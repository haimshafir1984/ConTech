"""
ConTech Pro - Manager Pages
מכיל את כל הטאבים של מצב מנהל (גרסה מקורית משוחזרת)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ייבוא פונקציות לוגיקה
from analyzer import FloorPlanAnalyzer, compute_skeleton_length_px
from reporter import generate_status_pdf, generate_payment_invoice_pdf
from database import (
    save_plan,
    get_plan_by_id,
    get_progress_reports,
    get_project_forecast,
    get_project_financial_status,
    get_payment_invoice_data,
    get_all_work_types_for_plan,
    get_progress_summary_by_date_range,
    get_all_plans,
)
from utils import (
    safe_process_metadata,
    safe_analyze_legend,
    load_stats_df,
    create_colored_overlay,
    get_simple_metadata_values,
)

# ייבוא פונקציות preprocessing לגזירה
from preprocessing import get_crop_bbox_from_canvas_data


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
    return proj["thick_walls"]


def render_workshop_tab():
    """טאב 1: סדנת עבודה - העלאה ועריכה (עם תמיכה ב-Crop ROI)"""

    # ==========================================
    # שלב 0: Crop ROI (אופציונלי)
    # ==========================================
    st.markdown("### ✂️ שלב 0: גזירת אזור שרטוט (אופציונלי)")

    enable_crop = st.checkbox(
        "🎯 הפעל גזירה ידנית לפני ניתוח",
        value=False,
        help="אפשר לסמן אזור מסוים בתוכנית לניתוח (ROI). שאר התוכנית תתעלם.",
    )

    if enable_crop:
        st.info(
            "💡 במצב זה, תוכל לסמן מלבן על התוכנית לפני הניתוח. רק האזור בתוך המלבן ינותח."
        )

        # אתחול session state ל-crop
        if "crop_mode_data" not in st.session_state:
            st.session_state.crop_mode_data = {}

        # העלאת קובץ למצב Crop
        crop_file = st.file_uploader(
            "📂 העלה PDF לגזירה",
            type="pdf",
            key="crop_file_uploader",
            help="העלה תוכנית אחת לפעם עבור גזירה",
        )

        if crop_file:
            file_key = crop_file.name

            # אם זה קובץ חדש, נאתחל
            if file_key not in st.session_state.crop_mode_data:
                with st.spinner("טוען תצוגה מקדימה..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(crop_file.getvalue())
                            temp_path = tmp.name

                        analyzer = FloorPlanAnalyzer()
                        preview_img = analyzer.pdf_to_image(temp_path)

                        st.session_state.crop_mode_data[file_key] = {
                            "preview_img": preview_img,
                            "pdf_path": temp_path,
                            "crop_bbox": None,
                            "processed": False,
                        }

                        os.unlink(temp_path)

                    except Exception as e:
                        st.error(f"❌ שגיאה בטעינת PDF: {str(e)}")
                        crop_file = None

            # הצגת Canvas לציור ROI
            if file_key in st.session_state.crop_mode_data:
                data = st.session_state.crop_mode_data[file_key]
                preview_img = data["preview_img"]

                preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                h, w = preview_rgb.shape[:2]

                max_width = 800
                scale_factor = min(1.0, max_width / w)
                display_w = int(w * scale_factor)
                display_h = int(h * scale_factor)

                pil_preview = Image.fromarray(preview_rgb)
                pil_preview_resized = pil_preview.resize(
                    (display_w, display_h), Image.Resampling.LANCZOS
                )

                st.markdown("#### 🎨 צייר מלבן סביב אזור השרטוט:")
                st.caption(f"גודל מקורי: {w}x{h}px | תצוגה: {display_w}x{display_h}px")

                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.1)",
                    stroke_width=3,
                    stroke_color="#00FF00",
                    background_image=pil_preview_resized,
                    height=display_h,
                    width=display_w,
                    drawing_mode="rect",
                    key=f"crop_canvas_{file_key}",
                    update_streamlit=True,
                )

                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    bbox = get_crop_bbox_from_canvas_data(
                        canvas_result.json_data, scale_factor
                    )

                    if bbox:
                        x, y, bw, bh = bbox
                        st.success(f"✅ אזור נבחר: {bw}x{bh}px (מיקום: x={x}, y={y})")

                        if st.button(
                            "🚀 נתח תוכנית עם גזירה",
                            type="primary",
                            key=f"analyze_crop_{file_key}",
                        ):
                            with st.spinner(f"מנתח {file_key} עם Crop ROI..."):
                                try:
                                    with tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".pdf"
                                    ) as tmp:
                                        tmp.write(crop_file.getvalue())
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
                                    ) = analyzer.process_file(
                                        path, save_debug=False, crop_bbox=bbox
                                    )

                                    if not meta.get("plan_name"):
                                        meta["plan_name"] = (
                                            file_key.replace(".pdf", "")
                                            .replace("-", " ")
                                            .strip()
                                        )

                                    llm_data = {}
                                    if meta.get("raw_text"):
                                        llm_data = safe_process_metadata(
                                            raw_text=meta.get("raw_text"),
                                            raw_text_full=meta.get("raw_text_full"),
                                            normalized_text=meta.get("normalized_text"),
                                            raw_blocks=meta.get("raw_blocks"),
                                        )
                                        meta.update(
                                            get_simple_metadata_values(llm_data)
                                        )

                                    st.session_state.projects[file_key] = {
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
                                        "llm_suggestions": (
                                            llm_data if meta.get("raw_text") else {}
                                        ),
                                        "debug_layers": getattr(
                                            analyzer, "debug_layers", {}
                                        ),
                                    }

                                    os.unlink(path)
                                    del st.session_state.crop_mode_data[file_key]

                                    st.success(f"✅ {file_key} נותח בהצלחה עם Crop!")
                                    st.info(
                                        "💾 עכשיו תוכל למצוא את התוכנית ברשימה למטה"
                                    )
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"❌ שגיאה: {str(e)}")
                                    import traceback

                                    with st.expander("פרטי שגיאה"):
                                        st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ צייר מלבן על התמונה")
                else:
                    st.info("👆 צייר מלבן על אזור השרטוט")
        else:
            st.info("📂 העלה קובץ PDF למעלה")

        st.markdown("---")

    # ==========================================
    # העלאה רגילה (ללא Crop)
    # ==========================================

    with st.expander(
        "העלאת קבצים (מצב רגיל)",
        expanded=not st.session_state.projects and not enable_crop,
    ):
        if enable_crop:
            st.warning("⚠️ מצב גזירה פעיל - השתמש בהעלאה למעלה")

        files = st.file_uploader(
            "גרור PDF או לחץ לבחירה",
            type="pdf",
            accept_multiple_files=True,
            key="main_file_uploader",
        )
        debug_mode = st.selectbox(
            "מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0
        )
        show_debug = debug_mode != "בסיסי"

        if files:
            for f in files:
                if f.name in st.session_state.projects:
                    continue

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
                                f.name.replace(".pdf", "").replace("-", " ").strip()
                            )

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
                            "llm_suggestions": (
                                llm_data if meta.get("raw_text") else {}
                            ),
                            "debug_layers": getattr(analyzer, "debug_layers", {}),
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
                                        and "text_combined" in analyzer.debug_layers
                                    ):
                                        st.image(
                                            analyzer.debug_layers["text_combined"],
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

                        os.unlink(path)
                        st.success(f"✅ {f.name} נותח בהצלחה!")

                    except Exception as e:
                        st.error(f"שגיאה: {str(e)}")
                        import traceback

                        show_trace = st.checkbox(
                            "פרטי שגיאה (Traceback)", value=False, key=f"trace_{f.name}"
                        )
                        if show_trace:
                            st.code(traceback.format_exc())

    if st.session_state.projects:
        st.markdown("---")
        selected = st.selectbox(
            "בחר תוכנית לעריכה:", list(st.session_state.projects.keys())
        )
        proj = st.session_state.projects[selected]

        name_key = f"name_{selected}"
        scale_key = f"scale_{selected}"
        if name_key not in st.session_state:
            st.session_state[name_key] = proj["metadata"].get("plan_name", "")
        if scale_key not in st.session_state:
            st.session_state[scale_key] = proj["metadata"].get("scale", "")

        col_edit, col_preview = st.columns([1, 1.5], gap="large")

        with col_edit:
            st.markdown("### הגדרות תוכנית")

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

            corrected_walls = get_corrected_walls(selected, proj)
            corrected_pixels = int(np.count_nonzero(corrected_walls))
            total_len = corrected_pixels / scale_val

            kernel = np.ones((6, 6), np.uint8)
            conc_corrected = cv2.dilate(
                cv2.erode(corrected_walls, kernel, iterations=1), kernel, iterations=2
            )
            block_corrected = cv2.subtract(corrected_walls, conc_corrected)

            conc_len = float(np.count_nonzero(conc_corrected) / scale_val)
            block_len = float(np.count_nonzero(block_corrected) / scale_val)
            floor_area = float(
                proj["metadata"].get("pixels_flooring_area", 0) / (scale_val**2)
            )

            proj["total_length"] = total_len

            st.info(
                f"📏 קירות: {total_len:.1f}מ' | בטון: {conc_len:.1f}מ' | בלוקים: {block_len:.1f}מ' | ריצוף: {floor_area:.1f}מ\"ר"
            )

            # מדידות מתקדמות
            with st.expander("📐 מדידות מתקדמות (Stage 1+2)", expanded=False):
                meta = proj.get("metadata", {})

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

                st.markdown("---")
                st.markdown("#### 📄 Override גודל נייר")

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

                prev_override = st.session_state.get(paper_override_key)

                if selected_paper != "זיהוי אוטומטי":
                    if prev_override != selected_paper:
                        st.session_state[paper_override_key] = selected_paper

                        ISO_SIZES = {
                            "A0": (841, 1189),
                            "A1": (594, 841),
                            "A2": (420, 594),
                            "A3": (297, 420),
                            "A4": (210, 297),
                        }

                        paper_w_mm, paper_h_mm = ISO_SIZES[selected_paper]

                        if meta.get("image_size_px"):
                            w_px = meta["image_size_px"]["width"]
                            h_px = meta["image_size_px"]["height"]
                            if w_px > h_px and paper_w_mm < paper_h_mm:
                                paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm

                        meta["paper_size_detected"] = selected_paper
                        meta["paper_mm"] = {"width": paper_w_mm, "height": paper_h_mm}
                        meta["paper_detection_confidence"] = 1.0

                        if meta.get("image_size_px"):
                            w_px = meta["image_size_px"]["width"]
                            h_px = meta["image_size_px"]["height"]

                            mm_per_pixel_x = paper_w_mm / w_px
                            mm_per_pixel_y = paper_h_mm / h_px
                            mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2.0

                            meta["mm_per_pixel"] = float(mm_per_pixel)

                            scale_denom = meta.get("scale_denominator")
                            if scale_denom:
                                meters_per_pixel = (mm_per_pixel * scale_denom) / 1000.0
                                meta["meters_per_pixel"] = float(meters_per_pixel)

                                if meta.get("wall_length_total_px"):
                                    meta["wall_length_total_m"] = float(
                                        meta["wall_length_total_px"] * meters_per_pixel
                                    )

                        st.success(f"✅ גודל נייר עודכן ל-{selected_paper}")
                        st.rerun()
                else:
                    if prev_override is not None:
                        del st.session_state[paper_override_key]
                        st.rerun()

                if not meta.get("scale_denominator"):
                    st.markdown("---")
                    st.markdown("#### 🔍 למה קנה מידה לא זוהה?")

                    st.write("**מקורות שנבדקו:**")
                    st.write(f"1. meta['scale'] = `{meta.get('scale', 'לא קיים')}`")
                    st.write(
                        f"2. meta['raw_text'][:200] = `{meta.get('raw_text', 'לא קיים')[:200]}`"
                    )

                    st.markdown("**ניסיון ידני:**")
                    manual_scale_text = st.text_input(
                        "הזן קנה מידה ידנית (לדוגמה: 1:50):",
                        key=f"manual_scale_{selected}",
                    )

                    if manual_scale_text and st.button(
                        "החל", key=f"apply_scale_{selected}"
                    ):
                        from analyzer import parse_scale

                        parsed = parse_scale(manual_scale_text)
                        if parsed:
                            meta["scale_denominator"] = parsed
                            meta["scale"] = manual_scale_text
                            st.success(f"✅ קנה מידה עודכן ל-1:{parsed}")
                            st.rerun()
                        else:
                            st.error("❌ לא הצלחתי לפרסר את הקנה מידה")

                st.markdown("---")
                st.markdown("#### 📊 נתוני חישוב מה-PDF")

                has_data = all(
                    [
                        meta.get("paper_size_detected"),
                        meta.get("image_size_px"),
                        meta.get("scale_denominator"),
                        meta.get("mm_per_pixel"),
                        meta.get("meters_per_pixel"),
                    ]
                )

                if has_data:
                    paper_w = meta["paper_mm"]["width"]
                    paper_h = meta["paper_mm"]["height"]
                    img_w = meta["image_size_px"]["width"]
                    img_h = meta["image_size_px"]["height"]

                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown(
                            f"**📄 נייר:** `{meta['paper_size_detected']}`  \n`{paper_w:.0f}×{paper_h:.0f}` מ\"מ"
                        )
                    with col_p2:
                        st.markdown(f"**🖼️ תמונה:** \n`{img_w}×{img_h}` px")

                    mm_per_px = float(meta["mm_per_pixel"])
                    m_per_px = float(meta["meters_per_pixel"])
                    scale_denom = int(meta["scale_denominator"])

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.markdown(f'**מ"מ/px** \n`{mm_per_px:.4f}`')
                    with col_r2:
                        st.markdown(f"**קנה מידה** \n`1:{scale_denom}`")
                    with col_r3:
                        st.markdown(f"**מטר/px** \n`{m_per_px:.6f}`")

                    st.markdown("**3️⃣ חישוב צעד אחר צעד:**")
                    show_formulas = st.checkbox(
                        "👁️ הצג נוסחאות",
                        value=True,
                        key=f"show_formulas_{selected}",
                    )
                    if show_formulas:
                        st.code(
                            f"""
נוסחאות החישוב:

1. מ"מ/פיקסל = גודל נייר במ"מ / גודל תמונה בפיקסלים
   mm_per_pixel_x = {paper_w} / {img_w} = {paper_w/img_w:.4f}
   mm_per_pixel_y = {paper_h} / {img_h} = {paper_h/img_h:.4f}
   mm_per_pixel = ממוצע = {mm_per_px:.4f}

2. מטר/פיקסל = (מ"מ/פיקסל × קנה מידה) / 1000
   meters_per_pixel = ({mm_per_px:.4f} × {scale_denom}) / 1000
   meters_per_pixel = {m_per_px:.6f}

3. אורך קירות במטרים = פיקסלי קירות × מטר/פיקסל
""",
                            language="text",
                        )

                    # אם לא התקבל wall_length_total_px מהאנלייזר, ננסה לחשב מה-skeleton
                    if (
                        not meta.get("wall_length_total_px")
                        and proj.get("skeleton") is not None
                    ):
                        try:
                            meta["wall_length_total_px"] = float(
                                compute_skeleton_length_px(proj["skeleton"])
                            )
                        except Exception:
                            pass

                    if meta.get("wall_length_total_px"):
                        wall_px = float(meta["wall_length_total_px"])
                        wall_m = float(
                            meta.get("wall_length_total_m", wall_px * m_per_px)
                        )

                        st.markdown("---")
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.success(f"📏 `{wall_px:,.0f}` פיקסלים")
                        with col_w2:
                            st.success(f"📐 `{wall_m:.2f}` מטר")

                else:
                    st.warning("⚠️ חסרים נתונים - בחר גודל נייר וקנה מידה למעלה")

            # מחשבון הצעת מחיר
            with st.expander("💰 מחשבון הצעת מחיר", expanded=False):
                st.markdown(
                    """<div style="background:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:10px;">
<strong>מחירון בסיס:</strong> בטון 1200₪/מ' | בלוקים 600₪/מ' | ריצוף 250₪/מ"ר
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
                        "פריט": ["קירות בטון", "קירות בלוקים", "ריצוף/חיפוי", 'סה"כ'],
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
                    float(scale_val),
                    int(corrected_pixels),
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

            corrected_walls_display = get_corrected_walls(selected, proj)

            kernel_display = np.ones((6, 6), np.uint8)
            concrete_corrected = cv2.dilate(
                cv2.erode(corrected_walls_display, kernel_display, iterations=1),
                kernel_display,
                iterations=2,
            )
            blocks_corrected = cv2.subtract(corrected_walls_display, concrete_corrected)

            floor_mask = proj["flooring_mask"] if show_flooring else None
            overlay = create_colored_overlay(
                proj["original"], concrete_corrected, blocks_corrected, floor_mask
            )
            st.image(overlay, use_column_width=True)
            st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")


def render_corrections_tab():
    """טאב 2: תיקונים ידניים"""
    st.markdown("## 🎨 תיקונים ידניים")
    st.caption("הוסף או הסר קירות באופן ידני למדויקות מקסימלית")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

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
    h, w = rgb.shape[:2]
    scale_factor = 1000 / w if w > 1000 else 1.0
    img_display = Image.fromarray(rgb).resize(
        (int(w * scale_factor), int(h * scale_factor))
    )

    if correction_mode == "➕ הוסף קירות חסרים":
        st.info("🖌️ צייר בירוק על הקירות שהמערכת החמיצה")

        canvas_add = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=6,
            stroke_color="#00FF00",
            background_image=img_display,
            height=int(h * scale_factor),
            width=int(w * scale_factor),
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
                    (w, h),
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
            combined, (int(w * scale_factor), int(h * scale_factor))
        )
        img_with_walls = Image.fromarray(combined_resized)

        canvas_remove = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=8,
            stroke_color="#FF0000",
            background_image=img_with_walls,
            height=int(h * scale_factor),
            width=int(w * scale_factor),
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
                    (w, h),
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
                    proj["raw_pixels"] = int(corrected_pixels)
                    proj["total_length"] = float(corrected_length)

                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    save_plan(
                        selected_plan,
                        proj["metadata"].get("plan_name"),
                        "1:50",
                        float(proj["scale"]),
                        int(corrected_pixels),
                        meta_json,
                    )
                    st.success("✅ הגרסה המתוקנת נשמרה!")

            with col_btn2:
                if st.button("🔄 אפס תיקונים", key="reset_corrections"):
                    del st.session_state.manual_corrections[selected_plan]
                    st.success("התיקונים אופסו")
                    st.rerun()
        else:
            st.info("אין תיקונים ידניים עדיין. עבור לטאב 'הוסף קירות' או 'הסר קירות'")


def render_dashboard_tab():
    """טאב 3: דשבורד"""
    from pages.dashboard import render_dashboard

    render_dashboard()


def render_invoices_tab():
    """טאב 4: חשבונות"""
    from pages.invoices import render_invoices

    render_invoices()


def render_plan_data_tab():
    """טאב חדש: נתונים מהשרטוט - הצגת חדרים ושטחים"""

    st.markdown("## 📄 נתונים מהשרטוט")
    st.caption("מידע שחולץ מהטקסט בתוכנית באמצעות AI")

    if not st.session_state.projects:
        st.info("📂 אין תוכניות במערכת. העלה תוכנית בטאב 'סדנת עבודה'")
        return

    # בחירת תוכנית
    selected_plan = st.selectbox(
        "בחר תוכנית:", list(st.session_state.projects.keys()), key="plan_data_select"
    )

    if not selected_plan:
        return

    proj = st.session_state.projects[selected_plan]

    # Load LLM data from multiple possible locations
    llm_data = None

    # Priority 1: Direct llm_data key
    if "llm_data" in proj:
        llm_data = proj["llm_data"]
    # Priority 2: llm_suggestions key (legacy)
    elif "llm_suggestions" in proj:
        llm_data = proj["llm_suggestions"]
    # Priority 3: Nested in metadata
    elif "metadata" in proj and isinstance(proj["metadata"], dict):
        if "llm_data" in proj["metadata"]:
            llm_data = proj["metadata"]["llm_data"]

    # If no data, offer to extract
    if not llm_data or llm_data.get("status") in ["error", "empty_text", "no_api_key"]:
        st.warning("⚠️ לא נמצא מידע מחולץ לתוכנית זו")

        # Show reason
        if llm_data:
            if llm_data.get("error"):
                st.error(f"שגיאה: {llm_data['error']}")
            if llm_data.get("limitations"):
                st.info("מגבלות:")
                for limit in llm_data["limitations"]:
                    st.markdown(f"- {limit}")

        # Offer re-extraction
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**האם לנסות לחלץ שוב?**")
            st.caption("המערכת תנסה לחלץ מידע מהטקסט של התוכנית")

        with col2:
            if st.button("🔄 חלץ מחדש", type="primary", use_container_width=True):
                with st.spinner("מחלץ נתונים..."):
                    try:
                        from utils import safe_process_metadata

                        # Get best text source
                        meta = proj.get("metadata", {})
                        llm_data = safe_process_metadata(
                            raw_text=meta.get("raw_text"),
                            raw_text_full=meta.get("raw_text_full"),
                            normalized_text=meta.get("normalized_text"),
                            raw_blocks=meta.get("raw_blocks"),
                        )

                        # Store in all locations for compatibility
                        proj["llm_data"] = llm_data
                        proj["llm_suggestions"] = llm_data  # Backward compat

                        st.success("✅ חילוץ הושלם!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ שגיאה: {str(e)}")

        st.markdown("---")

    # Display extracted data
    if llm_data and llm_data.get("status") != "error":

        # Status badge
        status = llm_data.get("status", "unknown")
        if status == "success":
            st.success(f"✅ סטטוס: הופק בהצלחה")
        elif status == "success_legacy":
            st.info(f"ℹ️ סטטוס: הומר מפורמט ישן")

        if llm_data.get("model_used"):
            st.caption(f"🤖 מודל: {llm_data['model_used']}")

        st.markdown("---")

        # === 1. Document Information ===
        st.markdown("### 📋 פרטי מסמך")

        document = llm_data.get("document", {})
        if document:
            doc_data = []

            field_labels = {
                "plan_title": "שם תוכנית",
                "plan_type": "סוג תוכנית",
                "scale": "סקלה",
                "date": "תאריך",
                "floor_or_level": "קומה/מפלס",
                "project_name": "שם פרויקט",
                "project_address": "כתובת",
                "architect_name": "אדריכל",
                "drawing_number": "מספר שרטוט",
            }

            for field_key, field_label in field_labels.items():
                if field_key in document:
                    field_obj = document[field_key]

                    # Extract value
                    if isinstance(field_obj, dict):
                        value = field_obj.get("value")
                        confidence = field_obj.get("confidence", 0)

                        if value:
                            doc_data.append(
                                {
                                    "שדה": field_label,
                                    "ערך": str(value),
                                    "ביטחון": f"{confidence}%",
                                }
                            )
                    elif field_obj:  # Simple value
                        doc_data.append(
                            {"שדה": field_label, "ערך": str(field_obj), "ביטחון": "N/A"}
                        )

            if doc_data:
                df_doc = pd.DataFrame(doc_data)
                st.dataframe(df_doc, use_container_width=True, hide_index=True)
            else:
                st.info("לא נמצאו פרטי מסמך")
        else:
            st.info("לא נמצאו פרטי מסמך")

        st.markdown("---")

        # === 2. Rooms Table ===
        st.markdown("### 🏠 חדרים ושטחים")

        rooms = llm_data.get("rooms", [])
        if rooms:
            st.success(f"✅ נמצאו {len(rooms)} חדרים")

            rooms_data = []
            for idx, room in enumerate(rooms, 1):
                # Extract values from each field
                def get_val(field_obj, default=""):
                    if isinstance(field_obj, dict):
                        return field_obj.get("value", default)
                    return field_obj if field_obj else default

                def get_conf(field_obj):
                    if isinstance(field_obj, dict):
                        return field_obj.get("confidence", 0)
                    return 0

                room_row = {
                    "#": idx,
                    "שם חדר": get_val(room.get("name", {})),
                    'שטח (מ"ר)': get_val(room.get("area_m2", {}), 0),
                    "גובה תקרה (מ')": get_val(room.get("ceiling_height_m", {}), ""),
                    "ריצוף": get_val(room.get("flooring_notes", {})),
                    "תקרה": get_val(room.get("ceiling_notes", {})),
                    "הערות": get_val(room.get("other_notes", {})),
                }

                # Add confidence for area (most important)
                area_conf = get_conf(room.get("area_m2", {}))
                if area_conf > 0:
                    room_row["ביטחון שטח"] = f"{area_conf}%"

                rooms_data.append(room_row)

            df_rooms = pd.DataFrame(rooms_data)
            st.dataframe(df_rooms, use_container_width=True, hide_index=True)

            # Total area
            total_area = sum(
                [
                    float(r['שטח (מ"ר)'])
                    for r in rooms_data
                    if r['שטח (מ"ר)'] and str(r['שטח (מ"ר)']).replace(".", "").isdigit()
                ]
            )

            if total_area > 0:
                st.metric("סך כל שטח החדרים", f'{total_area:.2f} מ"ר')
        else:
            st.warning("⚠️ לא נמצאו חדרים בטקסט")
            st.caption("💡 ייתכן שהתוכנית לא כוללת טבלת חדרים או שהטקסט לא חולץ כראוי")

        st.markdown("---")

        # === 3. Heights and Levels ===
        heights = llm_data.get("heights_and_levels", {})
        if heights:
            st.markdown("### 📏 גבהים ומפלסים")

            height_data = []
            height_labels = {
                "default_ceiling_height_m": "גובה תקרה סטנדרטי (מ')",
                "default_floor_height_m": "גובה רצפה ממפלס 0 (מ')",
                "construction_level_m": "מפלס בנייה (מ')",
            }

            for key, label in height_labels.items():
                if key in heights:
                    field_obj = heights[key]
                    value = (
                        field_obj.get("value")
                        if isinstance(field_obj, dict)
                        else field_obj
                    )

                    if value:
                        height_data.append({"פרמטר": label, "ערך": value})

            if height_data:
                df_heights = pd.DataFrame(height_data)
                st.dataframe(df_heights, use_container_width=True, hide_index=True)
            else:
                st.info("לא נמצאו נתוני גבהים")

        # === 4. Execution Notes ===
        notes = llm_data.get("execution_notes", {})
        if notes and any(notes.values()):
            st.markdown("### 📝 הערות ביצוע")

            note_labels = {
                "general_notes": "הערות כלליות",
                "structural_notes": "הערות קונסטרוקציה",
                "hvac_notes": "מיזוג אוויר",
                "electrical_notes": "חשמל",
                "plumbing_notes": "אינסטלציה",
            }

            for key, label in note_labels.items():
                if key in notes:
                    field_obj = notes[key]
                    value = (
                        field_obj.get("value")
                        if isinstance(field_obj, dict)
                        else field_obj
                    )

                    if value:
                        st.markdown(f"**{label}:** {value}")

        # === 5. Quantities Hint ===
        quantities = llm_data.get("quantities_hint", {})
        if quantities:
            wall_types = quantities.get("wall_types_mentioned", [])
            materials = quantities.get("material_hints", [])

            if wall_types or materials:
                st.markdown("### 🔨 רמזים לכמויות")

                if wall_types:
                    st.markdown("**סוגי קירות שהוזכרו:**")
                    for wt in wall_types:
                        st.markdown(f"- {wt}")

                if materials:
                    st.markdown("**חומרי גמר שהוזכרו:**")
                    for mat in materials:
                        st.markdown(f"- {mat}")

        # === 6. Limitations ===
        limitations = llm_data.get("limitations", [])
        if limitations:
            st.markdown("---")
            st.markdown("### ⚠️ מגבלות וזיהוי בעיות")
            for limit in limitations:
                st.warning(limit)

        # === 7. Raw JSON Debug ===
        st.markdown("---")
        with st.expander("🔍 JSON מלא (Debug)", expanded=False):
            st.json(llm_data)


def render_floor_analysis_tab():
    """
    טאב חדש: ניתוח שטחי רצפה והיקפים
    מבוסס על סגמנטציה של חדרים מתוך מסכת קירות
    """
    import pandas as pd
    import cv2
    import numpy as np
    from floor_extractor import analyze_floor_and_rooms

    st.markdown("## 📐 ניתוח שטחי רצפה והיקפים")
    st.caption("חישוב אוטומטי של שטחי חדרים, היקפים ופאנלים על בסיס זיהוי קירות")

    if not st.session_state.projects:
        st.info("📂 אין תוכניות במערכת. העלה תוכנית בטאב 'סדנת עבודה'")
        return

    # בחירת תוכנית
    selected_plan = st.selectbox(
        "בחר תוכנית:",
        list(st.session_state.projects.keys()),
        key="floor_analysis_plan_select",
    )

    if not selected_plan:
        return

    proj = st.session_state.projects[selected_plan]

    st.markdown("---")

    # הגדרות ניתוח
    with st.expander("⚙️ הגדרות מתקדמות", expanded=False):
        col_set1, col_set2 = st.columns(2)

        with col_set1:
            seg_method = st.radio(
                "שיטת סגמנטציה:",
                ["watershed", "cc"],
                index=0,
                help="watershed מומלץ - מפריד חדרים מחוברים | cc - פשוט יותר",
            )

        with col_set2:
            min_area = st.number_input(
                "שטח מינימלי לחדר (פיקסלים):",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
                help="חדרים קטנים מזה יתעלמו",
            )

    # כפתור ניתוח
    if st.button(
        "🔍 חשב שטחים והיקפים מהשרטוט", type="primary", use_container_width=True
    ):

        with st.spinner("מנתח... זה עשוי לקחת מספר שניות"):
            try:
                # שלב 1: הכן נתונים
                walls_mask = proj.get("thick_walls")
                original_img = proj.get("original")

                if walls_mask is None:
                    st.error("❌ לא נמצאה מסכת קירות. נסה לעבד את התוכנית מחדש.")
                    return

                # שלב 2: חלץ meters_per_pixel
                meta = proj.get("metadata", {})
                meters_per_pixel = meta.get("meters_per_pixel")

                if meters_per_pixel is None:
                    st.warning("⚠️ אין קנה מידה מוגדר - התוצאות יהיו בפיקסלים בלבד")

                # שלב 3: חלץ LLM rooms (אם יש)
                llm_data = proj.get("llm_data") or proj.get("llm_suggestions")
                llm_rooms = None
                if llm_data and isinstance(llm_data, dict):
                    llm_rooms = llm_data.get("rooms", [])

                # שלב 4: ניתוח!
                result = analyze_floor_and_rooms(
                    walls_mask=walls_mask,
                    original_image=original_img,
                    meters_per_pixel=meters_per_pixel,
                    llm_rooms=llm_rooms,
                    segmentation_method=seg_method,
                    min_room_area_px=int(min_area),
                )

                # שמור בפרויקט
                proj["floor_analysis"] = result

                # שלב 5: הצג תוצאות
                if not result["success"]:
                    st.error("❌ הניתוח נכשל")
                    if result.get("limitations"):
                        for lim in result["limitations"]:
                            st.warning(f"⚠️ {lim}")
                    return

                st.success(
                    f"✅ הניתוח הושלם! נמצאו {result['totals']['num_rooms']} אזורים/חדרים"
                )

                # תקציר
                st.markdown("### 📊 תקציר")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("מספר חדרים", result["totals"]["num_rooms"])

                with col2:
                    if result["totals"]["total_area_m2"] is not None:
                        st.metric(
                            'סה"כ שטח רצפה',
                            f'{result["totals"]["total_area_m2"]:.2f} מ"ר',
                        )
                    else:
                        st.metric('סה"כ שטח רצפה', "N/A")

                with col3:
                    if result["totals"]["total_perimeter_m"] is not None:
                        st.metric(
                            'סה"כ היקף',
                            f'{result["totals"]["total_perimeter_m"]:.1f} מ\'',
                        )
                    else:
                        st.metric('סה"כ היקף', "N/A")

                with col4:
                    if result["totals"]["total_baseboard_m"] is not None:
                        st.metric(
                            'סה"כ פאנלים (MVP)',
                            f'{result["totals"]["total_baseboard_m"]:.1f} מ\'',
                        )
                    else:
                        st.metric('סה"כ פאנלים', "N/A")

                # טבלת חדרים
                st.markdown("---")
                st.markdown("### 🏠 פירוט לפי חדרים")

                if result["rooms"]:
                    rooms_data = []
                    for room in result["rooms"]:
                        row = {
                            "מזהה": f"#{room['room_id']}",
                        }

                        # שם (אם matched)
                        if room.get("matched_name"):
                            row["שם חדר"] = room["matched_name"]
                        else:
                            row["שם חדר"] = "-"

                        # שטחים
                        if room["area_m2"] is not None:
                            row['שטח (מ"ר)'] = f"{room['area_m2']:.2f}"

                            if room.get("area_text_m2"):
                                row["שטח מטקסט"] = f"{room['area_text_m2']:.2f}"
                                row["הפרש"] = f"{room['diff_m2']:+.2f}"
                            else:
                                row["שטח מטקסט"] = "-"
                                row["הפרש"] = "-"
                        else:
                            row["שטח (פיקסלים)"] = room["area_px"]

                        # היקף
                        if room["perimeter_m"] is not None:
                            row["היקף (מ')"] = f"{room['perimeter_m']:.1f}"
                        else:
                            row["היקף (פיקסלים)"] = f"{room['perimeter_px']:.0f}"

                        # פאנלים
                        if room["baseboard_m"] is not None:
                            row["פאנלים (מ')"] = f"{room['baseboard_m']:.1f}"

                        # ביטחון
                        if (
                            room.get("match_confidence") is not None
                            and room["match_confidence"] > 0
                        ):
                            row["התאמה"] = f"{room['match_confidence']:.0%}"

                        rooms_data.append(row)

                    df = pd.DataFrame(rooms_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # מגבלות
                    if result.get("limitations"):
                        st.markdown("---")
                        st.markdown("### ⚠️ מגבלות וזיהוי בעיות")
                        for lim in result["limitations"]:
                            st.warning(lim)

                    # ויזואליזציה
                    st.markdown("---")
                    st.markdown("### 🎨 ויזואליזציה")

                    overlay = result["visualizations"].get("overlay")
                    if overlay is not None:
                        st.image(
                            overlay,
                            caption="חדרים מסומנים בצבעים",
                            use_column_width=True,
                        )

                    # Debug data
                    with st.expander("🔍 JSON מלא (Debug)", expanded=False):
                        # הכן גרסה JSON-safe
                        result_json = {
                            "success": result["success"],
                            "totals": result["totals"],
                            "rooms": [
                                {
                                    k: v for k, v in room.items() if k not in ["mask"]
                                }  # הסר numpy arrays
                                for room in result["rooms"]
                            ],
                            "limitations": result["limitations"],
                        }
                        st.json(result_json)

                else:
                    st.info("לא נמצאו חדרים")

            except Exception as e:
                st.error(f"❌ שגיאה בניתוח: {str(e)}")
                import traceback

                with st.expander("פרטי שגיאה מפורטים"):
                    st.code(traceback.format_exc())

    # הצג תוצאות קיימות (אם יש)
    elif "floor_analysis" in proj:
        st.info("💾 יש ניתוח קיים. לחץ על הכפתור למעלה לניתוח מחדש.")

        result = proj["floor_analysis"]

        if result.get("success"):
            st.markdown("### 📊 תוצאות אחרונות")

            # תקציר מהיר
            col1, col2 = st.columns(2)
            with col1:
                st.metric("חדרים שנמצאו", result["totals"]["num_rooms"])
            with col2:
                if result["totals"]["total_area_m2"]:
                    st.metric(
                        'סה"כ שטח', f'{result["totals"]["total_area_m2"]:.1f} מ"ר'
                    )

            # טבלה מקוצרת
            if result["rooms"]:
                quick_data = []
                for room in result["rooms"][:5]:  # רק 5 ראשונים
                    row = {"#": room["room_id"]}
                    if room.get("matched_name"):
                        row["שם"] = room["matched_name"]
                    if room["area_m2"]:
                        row["שטח"] = f"{room['area_m2']:.1f} מ\"ר"
                    quick_data.append(row)

                st.dataframe(pd.DataFrame(quick_data), hide_index=True)

                if len(result["rooms"]) > 5:
                    st.caption(
                        f"מציג 5 מתוך {len(result['rooms'])} חדרים. לחץ 'חשב מחדש' לתצוגה מלאה."
                    )
