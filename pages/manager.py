"""
ConTech Pro v2.0 - Manager Pages
כל הטאבים של מצב מנהל פרויקט
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
import io
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
from PIL import Image

from analyzer import FloorPlanAnalyzer
from contech_metadata import (
    metadata_exists,
    ContechMetadata,
    get_metadata_filepath,
    validate_metadata_checksum,
)
from reporter import generate_status_pdf, generate_payment_invoice_pdf
from database import (
    save_plan,
    save_progress_report,
    get_progress_reports,
    get_plan_by_filename,
    get_all_plans,
    get_plan_by_id,
    get_project_forecast,
    get_project_financial_status,
    calculate_material_estimates,
    get_payment_invoice_data,
    get_all_work_types_for_plan,
    get_progress_summary_by_date_range,
)
from utils import (
    safe_process_metadata,
    safe_analyze_legend,
    load_stats_df,
    create_colored_overlay,
    extract_segments_from_mask,
)


def get_corrected_walls(selected_plan, proj):
    """
    מחזיר מסכת קירות מתוקנת (אם יש תיקונים ידניים)

    Args:
        selected_plan: שם התוכנית
        proj: אובייקט הפרויקט מ-session_state

    Returns:
        מסכת קירות מתוקנת (numpy array)
    """
    import cv2

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


# ==========================================
# TAB 1: סדנת עבודה
# ==========================================
def render_workshop_tab():
    """
    ✨ טאב העלאה ועריכת תוכניות
    עם תמיכה ב-Google Vision OCR
    """
    # ==========================================
    # 🆕 CHANGE 1: הוספת checkbox ל-Google Vision OCR
    # ==========================================
    with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
        # ✨ בורר Google Vision OCR
        col_ocr1, col_ocr2 = st.columns([3, 1])

        with col_ocr1:
            files = st.file_uploader(
                "גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True
            )

        with col_ocr2:
            use_google_ocr = st.checkbox(
                "🔍 Google Vision",
                value=True,
                help="OCR מדויק יותר (במיוחד עברית)",
                key="use_google_ocr",
            )

        debug_mode = st.selectbox(
            "מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0
        )
        show_debug = debug_mode != "בסיסי"
    # ==========================================
    # 🎯 גזירה ידנית (Crop ROI)
    # ==========================================
    enable_crop = st.checkbox(
        "🎯 הפעל גזירה ידנית לפני ניתוח",
        value=False,
        help="אפשר לסמן אזור מסוים בתוכנית לניתוח (ROI). שאר התוכנית תתעלם.",
        key="enable_crop_checkbox",
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
                        st.error(f"❌ לא ניתן לפתוח את הקובץ - ודא שזה PDF תקין")
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
                    # פונקציה פשוטה לחילוץ bbox (תחליף ל-preprocessing)
                    def get_crop_bbox_simple(json_data, scale):
                        """חילוץ bbox מה-canvas"""
                        if not json_data or not json_data.get("objects"):
                            return None

                        rect = json_data["objects"][-1]  # הריבוע האחרון
                        x = int(rect["left"] / scale)
                        y = int(rect["top"] / scale)
                        w = int(rect["width"] / scale)
                        h = int(rect["height"] / scale)

                        return (x, y, w, h)

                    bbox = get_crop_bbox_simple(canvas_result.json_data, scale_factor)

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
                                        llm_data = safe_process_metadata(meta=meta)
                                        meta.update(
                                            {k: v for k, v in llm_data.items() if v}
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
                            # ========== METADATA CHECK - תוספת חדשה ==========
                            metadata_path = get_metadata_filepath(path)
                            metadata_loaded = False

                            if metadata_exists(path):
                                try:
                                    metadata = ContechMetadata.load(metadata_path)

                                    if validate_metadata_checksum(metadata, path):
                                        st.info(
                                            f"✅ נמצא metadata (נוצר {metadata.created_at[:10]})"
                                        )

                                        use_metadata = st.checkbox(
                                            f"🔒 טען מ-metadata [{f.name}]",
                                            value=True,
                                            key=f"use_meta_{f.name}",
                                            help="נתונים מדויקים מהפעם הקודמת",
                                        )

                                        if use_metadata:
                                            st.success("📥 טוען מ-metadata...")

                                            analyzer = FloorPlanAnalyzer()
                                            img_temp = analyzer.pdf_to_image(path)
                                            h, w = img_temp.shape[:2]

                                            thick_walls = np.zeros(
                                                (h, w), dtype=np.uint8
                                            )

                                            for wall in metadata.walls:
                                                points = np.array(
                                                    wall.points, dtype=np.int32
                                                )
                                                cv2.polylines(
                                                    thick_walls,
                                                    [points],
                                                    False,
                                                    255,
                                                    thickness=5,
                                                )

                                            pix = int(
                                                metadata.get_total_length_meters()
                                                * metadata.pixels_per_meter
                                            )

                                            meta_dict = {
                                                "plan_name": metadata.plan_name
                                                or f.name.replace(".pdf", ""),
                                                "scale": metadata.scale_text,
                                                "raw_text": "",
                                            }

                                            kernel = np.ones((6, 6), np.uint8)
                                            conc = cv2.dilate(
                                                cv2.erode(
                                                    thick_walls, kernel, iterations=1
                                                ),
                                                kernel,
                                                iterations=2,
                                            )
                                            blok = cv2.subtract(thick_walls, conc)
                                            floor = np.zeros_like(thick_walls)

                                            st.session_state.projects[f.name] = {
                                                "skeleton": thick_walls,
                                                "thick_walls": thick_walls,
                                                "original": img_temp,
                                                "raw_pixels": pix,
                                                "scale": metadata.pixels_per_meter,
                                                "metadata": meta_dict,
                                                "concrete_mask": conc,
                                                "blocks_mask": blok,
                                                "flooring_mask": floor,
                                                "total_length": metadata.get_total_length_meters(),
                                                "llm_suggestions": {},
                                                "debug_layers": {},
                                                "_from_metadata": True,
                                                "_metadata_object": metadata,
                                            }

                                            st.success(
                                                f"✅ טעינה מ-metadata ({len(metadata.walls)} קירות)"
                                            )
                                            metadata_loaded = True

                                    else:
                                        st.warning("⚠️ PDF השתנה. מריץ זיהוי מחדש.")

                                except Exception as e:
                                    st.error(f"❌ שגיאה בטעינת metadata: {str(e)}")
                                    metadata_loaded = False
                            # ========== אם לא טענו מ-metadata, ממשיכים לקוד הקיים ==========
                            if not metadata_loaded:

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

                            # ==========================================
                            # 🆕 CHANGE 2: העברת pdf_bytes לפונקציית metadata
                            # ==========================================
                            # חילוץ מטא-דאטה + ניתוח חכם עם Google Vision OCR
                            llm_data = {}

                            run_ai = st.button(
                                f"🧠 נתח מטא-דאטה עם AI עבור {f.name}",
                                key=f"ai_{f.name}",
                            )
                            if run_ai and meta.get("raw_text"):
                                llm_data = safe_process_metadata(
                                    meta=meta,
                                    pdf_bytes=meta.get("pdf_bytes"),  # ← העברת ה-bytes
                                )
                                meta.update({k: v for k, v in llm_data.items() if v})
                            meta.pop("pdf_bytes", None)

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

                            # ==========================================
                            # 🆕 CHANGE 3: הוספת אינדיקטור OCR
                            # ==========================================
                            # הצגת מקור ה-OCR
                            if llm_data and llm_data.get("_ocr_source"):
                                ocr_icons = {
                                    "google_vision": "🔍 Google Vision OCR",
                                    "pymupdf": "📄 PyMuPDF",
                                    "pymupdf_fallback": "📄 PyMuPDF (fallback)",
                                }
                                ocr_source = llm_data.get("_ocr_source", "unknown")
                                ocr_label = ocr_icons.get(ocr_source, ocr_source)

                                # צבע לפי מקור
                                if ocr_source == "google_vision":
                                    st.success(f"✨ {ocr_label}")
                                else:
                                    st.info(f"ℹ️ {ocr_label}")

                        except Exception as e:
                            st.error(f"שגיאה: {str(e)}")
                            import traceback

                            st.error("פרטים נוספים:")
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
            # ========== אינדיקטורים - תוספת חדשה ==========
            if proj.get("_from_metadata"):
                st.info("🔒 תוכנית נטענה מ-metadata - דיוק גבוה!")
            # אינדיקטור תיקונים
            if selected in st.session_state.manual_corrections:
                st.success("✏️ תוכנית זו תוקנה ידנית")

            p_name = st.text_input("שם התוכנית", key=name_key)
            p_scale_text = st.text_input(
                "קנה מידה (לתיעוד)", key=scale_key, placeholder="1:50"
            )

            st.markdown("#### כיול")

            # ========== נעילת סקייל אם מ-metadata - תוספת חדשה ==========
            if proj.get("_from_metadata"):
                st.warning("🔒 הסקייל נעול (טעון מ-metadata)")
                scale_val = proj["scale"]
                st.metric("פיקסלים למטר", f"{scale_val:.1f}")
            else:
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
            segments = extract_segments_from_mask(corrected_walls, scale_val)
            total_len = sum(seg.get("length_px", 0) for seg in segments) / scale_val

            # חישוב חומרים מהגרסה המתוקנת
            kernel = np.ones((6, 6), np.uint8)
            conc_corrected = cv2.dilate(
                cv2.erode(corrected_walls, kernel, iterations=1), kernel, iterations=2
            )
            block_corrected = cv2.subtract(corrected_walls, conc_corrected)

            conc_segments = extract_segments_from_mask(conc_corrected, scale_val)
            block_segments = extract_segments_from_mask(block_corrected, scale_val)

            conc_len = sum(seg.get("length_px", 0) for seg in conc_segments) / scale_val
            block_len = (
                sum(seg.get("length_px", 0) for seg in block_segments) / scale_val
            )

            floor_area = proj["metadata"].get("pixels_flooring_area", 0) / (
                scale_val**2
            )

            proj["total_length"] = total_len

            st.info(
                f"📏 קירות: {total_len:.1f}מ' | בטון: {conc_len:.1f}מ' | בלוקים: {block_len:.1f}מ' | ריצוף: {floor_area:.1f}מ\"ר"
            )

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
                meta_json = json.dumps(
                    {
                        k: v
                        for k, v in proj["metadata"].items()
                        if not isinstance(v, bytes)
                    },
                    ensure_ascii=False,
                )
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
                # ========== METADATA EXPORT - תוספת חדשה ==========
                if not proj.get("_from_metadata"):
                    try:
                        analyzer = FloorPlanAnalyzer()

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as temp_pdf:
                            temp_path = temp_pdf.name

                        metadata_filepath = analyzer.export_walls_to_metadata(
                            corrected_walls, temp_path, scale_val, p_scale_text
                        )

                        st.info(
                            f"📦 נוצר metadata: {os.path.basename(metadata_filepath)}"
                        )

                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

                    except Exception as e:
                        st.warning(f"⚠️ לא ניתן ליצור metadata: {str(e)}")
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
            blocks_corrected = cv2.subtract(corrected_walls_display, concrete_corrected)

            floor_mask = proj["flooring_mask"] if show_flooring else None

            overlay = create_colored_overlay(
                proj["original"], concrete_corrected, blocks_corrected, floor_mask
            )
            st.image(overlay, use_column_width=True)
            st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")

            # ניתוח מקרא
            st.markdown("---")
            with st.expander("🎨 נתח מקרא (AI)", expanded=False):
                st.caption(
                    "המערכת תנסה למצוא את המקרא אוטומטית, או שאתה יכול לחתוך ידנית"
                )

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

                                    if "auto_legend" not in st.session_state:
                                        st.session_state.auto_legend = {}
                                    st.session_state.auto_legend[selected] = cropped

                                    if st.button(
                                        "📝 נתח מקרא זה",
                                        key=f"analyze_auto_{selected}",
                                    ):
                                        with st.spinner("מנתח עם Claude AI..."):
                                            _, buffer = cv2.imencode(".png", cropped)
                                            image_bytes = buffer.tobytes()

                                            result = safe_analyze_legend(image_bytes)

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

                rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                scale_factor = min(1.0, 1200 / max(w, h))

                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)

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
                    if st.button("🔍 נתח מקרא עם AI", key=f"analyze_legend_{selected}"):
                        with st.spinner("מנתח מקרא..."):
                            try:
                                rect = legend_canvas.json_data["objects"][-1]
                                x = int(rect["left"] / scale_factor)
                                y = int(rect["top"] / scale_factor)
                                rect_w = int(rect["width"] / scale_factor)
                                rect_h = int(rect["height"] / scale_factor)

                                cropped = proj["original"][
                                    y : y + rect_h, x : x + rect_w
                                ]

                                _, buffer = cv2.imencode(".png", cropped)
                                image_bytes = buffer.tobytes()

                                result = safe_analyze_legend(image_bytes)

                                if isinstance(result, dict) and "error" not in result:
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
                                            for material in result["materials_found"]:
                                                st.markdown(f"- {material}")

                                    if result.get("symbols"):
                                        st.markdown("**סמלים:**")
                                        for symbol in result["symbols"][:5]:
                                            st.markdown(
                                                f"- **{symbol.get('symbol', '')}**: {symbol.get('meaning', '')}"
                                            )

                                    if result.get("notes"):
                                        st.info(f"💡 {result['notes']}")

                                    proj["metadata"]["legend_analysis"] = result

                                elif isinstance(result, dict) and "error" in result:
                                    st.error(f"שגיאה: {result['error']}")
                                else:
                                    st.warning(f"תשובה לא צפויה: {result}")

                            except Exception as e:
                                st.error(f"שגיאה בניתוח: {str(e)}")
                                import traceback

                                st.markdown("פרטי שגיאה")
                                st.code(traceback.format_exc())
                else:
                    st.info("👆 צייר ריבוע סביב המקרא בתוכנית ולחץ על הכפתור")


# ==========================================
# TAB 2: תיקונים ידניים
# ==========================================
def render_corrections_tab():
    """טאב תיקונים ידניים לקירות"""
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
                        proj["raw_pixels"] = corrected_pixels
                        proj["total_length"] = corrected_length

                        from utils import clean_metadata_for_json

                        meta_json = json.dumps(
                            clean_metadata_for_json(proj["metadata"]),
                            ensure_ascii=False,
                        )
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


# ==========================================
# TAB 3: נתונים מהשרטוט (Placeholder)
# ==========================================
def render_plan_data_tab():
    """טאב חישוב נתונים לפי גודל דף וסקייל"""
    st.markdown("## 📄 נתונים מהשרטוט")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected = st.selectbox(
        "בחר תוכנית לניתוח:",
        list(st.session_state.projects.keys()),
        key="plan_data_selector",
    )

    proj = st.session_state.projects[selected]

    st.markdown("---")

    # ========== חלק 1: מידע בסיסי ==========
    st.markdown("### 📊 מידע בסיסי")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "גודל תמונה",
            f"{proj['original'].shape[1]} × {proj['original'].shape[0]} px",
        )

    with col2:
        scale = proj.get("scale", 200.0)
        st.metric("סקייל", f"{scale:.1f} px/מ'")

    with col3:
        scale_text = proj["metadata"].get("scale", "לא ידוע")
        st.metric("קנה מידה", scale_text)

    # ========== חלק 2: חישוב לפי גודל דף ==========
    st.markdown("---")
    st.markdown("### 📐 חישוב אורך קירות לפי גודל דף פיזי")

    st.caption(
        """
    💡 אם אתה יודע את גודל הדף המודפס (למשל A3, A2), 
    ניתן לחשב את הסקייל האמיתי ולקבל מדידה מדויקת.
    """
    )

    with st.expander("🔧 הגדרות חישוב", expanded=True):
        col_size, col_orient = st.columns(2)

        with col_size:
            paper_sizes = {
                "A4": (210, 297),  # מ"מ
                "A3": (297, 420),
                "A2": (420, 594),
                "A1": (594, 841),
                "A0": (841, 1189),
                "מותאם אישית": None,
            }

            paper_choice = st.selectbox(
                "גודל דף:", list(paper_sizes.keys()), key="paper_size_choice"
            )

        with col_orient:
            orientation = st.radio(
                "כיוון:", ["לאורך", "לרוחב"], horizontal=True, key="paper_orientation"
            )

        # קבלת מידות
        if paper_choice == "מותאם אישית":
            col_w, col_h = st.columns(2)
            with col_w:
                paper_width_mm = st.number_input(
                    'רוחב (מ"מ):', min_value=100, max_value=2000, value=420, step=10
                )
            with col_h:
                paper_height_mm = st.number_input(
                    'גובה (מ"מ):', min_value=100, max_value=2000, value=594, step=10
                )
        else:
            w, h = paper_sizes[paper_choice]
            if orientation == "לרוחב":
                paper_width_mm = max(w, h)
                paper_height_mm = min(w, h)
            else:
                paper_width_mm = min(w, h)
                paper_height_mm = max(w, h)

        st.info(f'📄 גודל דף: {paper_width_mm} × {paper_height_mm} מ"מ')

    # ========== חישובים ==========
    if st.button("🧮 חשב סקייל אמיתי", type="primary"):
        # המרה ממ"מ למטרים
        paper_width_m = paper_width_mm / 1000
        paper_height_m = paper_height_mm / 1000

        # גודל תמונה בפיקסלים
        img_width_px = proj["original"].shape[1]
        img_height_px = proj["original"].shape[0]

        # חישוב פיקסלים למטר של הדף
        pixels_per_meter_width = img_width_px / paper_width_m
        pixels_per_meter_height = img_height_px / paper_height_m

        # ממוצע
        calculated_scale = (pixels_per_meter_width + pixels_per_meter_height) / 2

        st.markdown("---")
        st.markdown("### 📊 תוצאות חישוב")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric(
                "סקייל מחושב",
                f"{calculated_scale:.1f} px/מ'",
                help="מבוסס על גודל הדף הפיזי",
            )

        with col_r2:
            current_scale = proj.get("scale", 200.0)
            diff = calculated_scale - current_scale
            st.metric(
                "סקייל נוכחי",
                f"{current_scale:.1f} px/מ'",
                delta=f"{diff:+.1f}",
                delta_color="off",
            )

        with col_r3:
            error_pct = (
                abs(diff / calculated_scale * 100) if calculated_scale > 0 else 0
            )
            st.metric(
                "סטייה", f"{error_pct:.1f}%", help="הפרש בין הסקייל הנוכחי למחושב"
            )

        # חישוב אורכים מחדש
        st.markdown("---")
        st.markdown("### 📏 אורכי קירות מתוקנים")

        from pages.manager import get_corrected_walls

        corrected_walls = get_corrected_walls(selected, proj)

        # עם הסקייל הנוכחי
        pixels_current = np.count_nonzero(corrected_walls)
        length_current = pixels_current / current_scale

        # עם הסקייל המחושב
        length_calculated = pixels_current / calculated_scale

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.info(
                f"""
            **עם סקייל נוכחי ({current_scale:.1f}):**
            - אורך כולל: **{length_current:.2f} מ'**
            """
            )

        with col_l2:
            st.success(
                f"""
            **עם סקייל מחושב ({calculated_scale:.1f}):**
            - אורך כולל: **{length_calculated:.2f} מ'**
            - הפרש: **{(length_calculated - length_current):.2f} מ'**
            """
            )

        # אפשרות לעדכון
        st.markdown("---")

        if st.button("✅ עדכן סקייל לערך המחושב", type="secondary"):
            proj["scale"] = calculated_scale
            st.success(f"✅ הסקייל עודכן ל-{calculated_scale:.1f} px/מ'")
            st.balloons()
            st.rerun()

    # ========== חלק 3: נתוני מטא-דאטה ==========
    st.markdown("---")
    st.markdown("### 🗂️ מטא-דאטה")

    metadata = proj.get("metadata", {})

    if metadata:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**מידע מהתוכנית:**")
            st.write(f"- שם: {metadata.get('plan_name', 'לא ידוע')}")
            st.write(f"- קנה מידה: {metadata.get('scale', 'לא ידוע')}")

        with col_m2:
            st.markdown("**מקור:**")
            if proj.get("_from_metadata"):
                st.success("✅ נטען מ-Metadata JSON")
                metadata_obj = proj.get("_metadata_object")
                if metadata_obj:
                    st.write(f"- מספר קירות: {len(metadata_obj.walls)}")
                    st.write(f"- נוצר: {metadata_obj.created_at[:10]}")
            else:
                st.info("ℹ️ זיהוי OpenCV")

    # ========== חלק 4: חומרים ==========
    st.markdown("---")
    st.markdown("### 🧱 פירוט חומרים")

    # קבלת נתוני חומרים
    from pages.manager import get_corrected_walls

    corrected_walls = get_corrected_walls(selected, proj)

    scale = proj.get("scale", 200.0)

    # חישוב חלוקה לחומרים
    kernel = np.ones((6, 6), np.uint8)
    concrete = cv2.dilate(
        cv2.erode(corrected_walls, kernel, iterations=1), kernel, iterations=2
    )
    blocks = cv2.subtract(corrected_walls, concrete)

    concrete_len = np.count_nonzero(concrete) / scale
    blocks_len = np.count_nonzero(blocks) / scale
    total_len = concrete_len + blocks_len

    col_mat1, col_mat2, col_mat3 = st.columns(3)

    with col_mat1:
        st.metric("🔵 בטון", f"{concrete_len:.1f} מטר")

    with col_mat2:
        st.metric("🟠 בלוקים", f"{blocks_len:.1f} מטר")

    with col_mat3:
        st.metric('📏 סה"כ', f"{total_len:.1f} מטר")

    # תרשים
    import pandas as pd

    df_materials = pd.DataFrame(
        {"חומר": ["בטון", "בלוקים"], "אורך במטרים": [concrete_len, blocks_len]}
    )

    st.bar_chart(df_materials.set_index("חומר"))

    # ========== חלק 5: ייצוא ==========
    st.markdown("---")
    st.markdown("### 📤 ייצוא נתונים")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        # CSV
        csv_data = f"""סוג,כמות,יחידה
קירות בטון,{concrete_len:.2f},מטר
קירות בלוקים,{blocks_len:.2f},מטר
סה"כ קירות,{total_len:.2f},מטר
"""
        st.download_button(
            "📥 הורד CSV",
            data=csv_data,
            file_name=f"{selected}_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_exp2:
        # JSON
        import json

        json_data = json.dumps(
            {
                "plan_name": metadata.get("plan_name", selected),
                "scale": scale,
                "scale_text": metadata.get("scale", ""),
                "materials": {
                    "concrete_meters": concrete_len,
                    "blocks_meters": blocks_len,
                    "total_meters": total_len,
                },
                "image_size": {
                    "width": proj["original"].shape[1],
                    "height": proj["original"].shape[0],
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        st.download_button(
            "📥 הורד JSON",
            data=json_data,
            file_name=f"{selected}_data.json",
            mime="application/json",
            use_container_width=True,
        )


# ==========================================
# TAB 4: ניתוח שטחים (Placeholder)
# ==========================================
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
            auto_min_area = st.checkbox(
                "סף חדרים אוטומטי",
                value=True,
                help="מחשב סף דינאמי לפי גודל השטח הפנימי",
            )
            min_area = st.number_input(
                "שטח מינימלי לחדר (פיקסלים):",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
                help="חדרים קטנים מזה יתעלמו",
                disabled=auto_min_area,
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
                meters_per_pixel_x = meta.get("meters_per_pixel_x")
                meters_per_pixel_y = meta.get("meters_per_pixel_y")

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
                    meters_per_pixel_x=meters_per_pixel_x,
                    meters_per_pixel_y=meters_per_pixel_y,
                    llm_rooms=llm_rooms,
                    segmentation_method=seg_method,
                    min_room_area_px=0 if auto_min_area else int(min_area),
                )

                # שמור בפרויקט
                proj["floor_analysis"] = result

                # שיפור מסכת ריצוף לפי מסכות חדרים (אם קיימות)
                try:
                    refined_flooring = refine_flooring_mask_with_rooms(
                        proj.get("flooring_mask"),
                        result.get("visualizations", {}).get("masks"),
                    )
                    if refined_flooring is not None:
                        proj["flooring_mask_refined"] = refined_flooring
                        meta = proj.get("metadata", {})
                        meta["pixels_flooring_area_refined"] = int(
                            np.count_nonzero(refined_flooring)
                        )
                except Exception:
                    pass

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


# ==========================================
# TAB 5: דשבורד
# ==========================================
def render_dashboard_tab():
    """טאב דשבורד פרויקט"""
    st.header("📊 דשבורד פרויקט")

    all_plans = get_all_plans()
    if not all_plans:
        st.info("🔍 אין פרויקטים במערכת. העלה תוכנית בסדנת עבודה.")
    else:
        plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
        selected_plan_dash = st.selectbox(
            "📂 בחר פרויקט:", plan_options, key="dashboard_plan_select"
        )
        plan_id = int(selected_plan_dash.split("ID: ")[1].strip(")"))

        forecast = get_project_forecast(plan_id)
        financial = get_project_financial_status(plan_id)
        plan_data = get_plan_by_id(plan_id)

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

        st.markdown("---")
        st.markdown("### 📊 התקדמות כללית")

        if percent < 30:
            color = "#EF4444"
        elif percent < 70:
            color = "#F59E0B"
        else:
            color = "#10B981"

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

        st.markdown("---")
        st.markdown("### 📈 גרף התקדמות לאורך זמן")

        df_stats = load_stats_df()
        if not df_stats.empty:
            df_current = df_stats[df_stats["שם תוכנית"] == plan_data["plan_name"]]

            if not df_current.empty:
                st.bar_chart(
                    df_current,
                    x="תאריך",
                    y="כמות שבוצעה",
                    use_container_width=True,
                )

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
                        if (
                            selected_plan_dash
                            and selected_plan_dash in st.session_state.projects
                        ):
                            proj = st.session_state.projects[selected_plan_dash]
                            rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                        else:
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

                        stats = {
                            "built": built,
                            "total": total,
                            "percent": percent,
                            "remaining": remaining,
                            "cost": cost,
                            "budget": budget,
                        }

                        pdf_buffer = generate_status_pdf(
                            plan_data["plan_name"], rgb, stats
                        )

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
                "📊 ייצא נתונים",
                use_container_width=True,
                key="export_button_dash",
            ):
                st.info("💡 תכונה בפיתוח: ייצוא ל-Excel")

        with col3:
            if st.button(
                '📧 שלח דוא"ל', use_container_width=True, key="email_button_dash"
            ):
                st.info("💡 תכונה בפיתוח: שליחת דוח באימייל")

        st.markdown("---")
        st.markdown("### 📋 דיווחים אחרונים")

        reports = get_progress_reports(plan_id)
        if reports:
            recent = reports[:5]

            for i, r in enumerate(recent, 1):
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

            total_reports = len(reports)
            if total_reports > 5:
                st.caption(f"📌 מציג 5 מתוך {total_reports} דיווחים")
        else:
            st.info("📭 אין דיווחים לפרויקט זה. התחל לדווח בסדנת עבודה!")


# ==========================================
# TAB 6: חשבונות חלקיים
# ==========================================
def render_invoices_tab():
    """טאב חשבונות חלקיים"""
    st.markdown("## 💰 מחולל חשבונות חלקיים")
    st.caption("הפקת חשבונית לתשלום לקבלן על בסיס ביצוע בפועל")

    all_plans = get_all_plans()
    if not all_plans:
        st.info("אין פרויקטים במערכת")
    else:
        plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
        selected_plan_invoice = st.selectbox(
            "בחר פרויקט:", plan_options, key="invoice_plan_select"
        )
        plan_id = int(selected_plan_invoice.split("ID: ")[1].strip(")"))

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📅 בחר טווח תאריכים")

            quick_range = st.radio(
                "בחירה מהירה:",
                ["שבוע אחרון", "חודש אחרון", "טווח מותאם אישית"],
                horizontal=True,
            )

            from datetime import timedelta

            if quick_range == "שבוע אחרון":
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
            elif quick_range == "חודש אחרון":
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
            else:
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

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            st.info(f"📊 תקופת החשבון: {start_str} עד {end_str}")

            st.markdown("### 💵 מחירי יחידה")

            work_types = get_all_work_types_for_plan(plan_id)

            if not work_types:
                st.warning("אין דיווחים לפרויקט זה עדיין")
            else:
                st.caption("ערוך את המחירים לפי הצורך. המחירים המוצגים הם ברירות מחדל.")

                unit_prices = {}

                for work_type in work_types:
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
                "ח.פ / ע.מ:", value="", placeholder="123456789", key="contractor_vat"
            )

            contractor_address = st.text_area(
                "כתובת:",
                value="",
                placeholder="רחוב הבניינים 1, תל אביב",
                height=80,
                key="contractor_address",
            )

            st.markdown("---")

            if st.button("🧾 צור חשבונית", type="primary", use_container_width=True):
                if not contractor_name or not contractor_vat:
                    st.error("❌ יש למלא שם קבלן ומספר עוסק")
                else:
                    with st.spinner("מכין חשבונית..."):
                        try:
                            invoice_data = get_payment_invoice_data(
                                plan_id, start_str, end_str, unit_prices
                            )

                            if invoice_data.get("error"):
                                st.error(f"❌ {invoice_data['error']}")
                            elif not invoice_data["items"]:
                                st.warning("⚠️ אין דיווחים בטווח התאריכים הזה")
                            else:
                                contractor_info = {
                                    "name": contractor_name,
                                    "company": contractor_company,
                                    "vat_id": contractor_vat,
                                    "address": contractor_address,
                                }

                                pdf_buffer = generate_payment_invoice_pdf(
                                    invoice_data, contractor_info
                                )

                                st.success("✅ החשבונית הוכנה בהצלחה!")

                                st.markdown("### 📋 סיכום החשבונית")

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
                                    df_items, use_container_width=True, hide_index=True
                                )

                                col_sum1, col_sum2, col_sum3 = st.columns(3)
                                with col_sum1:
                                    st.metric(
                                        "סכום ביניים",
                                        f"{invoice_data['total_amount']:,.2f} ₪",
                                    )
                                with col_sum2:
                                    st.metric(
                                        'מע"מ (17%)', f"{invoice_data['vat']:,.2f} ₪"
                                    )
                                with col_sum3:
                                    st.metric(
                                        '**סה"כ לתשלום**',
                                        f"{invoice_data['total_with_vat']:,.2f} ₪",
                                    )

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

        st.markdown("---")
        with st.expander("📊 דיווחים בטווח התאריכים"):
            summary = get_progress_summary_by_date_range(plan_id, start_str, end_str)
            if summary:
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
