"""
ConTech Pro - Manager Pages
מכיל את כל הטאבים של מצב מנהל
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
from datetime import datetime

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
)


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


def render_workshop_tab():
    """טאב 1: סדנת עבודה - העלאה ועריכה"""

    # במקום expander (כדי למנוע nested expander כשיש בפנים expander של "פרטי שגיאה")
    st.markdown("### 📤 העלאת קבצים")

    files_container = st.container()
    with files_container:
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
                                    f.name.replace(".pdf", "").replace("-", " ").strip()
                                )

                            if meta.get("raw_text"):
                                llm_data = safe_process_metadata(meta["raw_text"])
                                meta.update({k: v for k, v in llm_data.items() if v})

                            # שמירה ל-session
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
                                "debug_image": debug_img,
                                "debug_layers": analyzer.debug_layers,
                                "confidence_map": analyzer.confidence_map,
                            }

                            try:
                                os.unlink(path)
                            except Exception:
                                pass

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
        if selected is None or selected not in st.session_state.projects:
            st.warning("בחר תוכנית כדי להמשיך.")
            return
        proj = st.session_state.projects[selected]

        name_key = f"name_{selected}"
        scale_key = f"scale_text_{selected}"
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

            p_scale_text = st.text_input("קנה מידה (לדוגמה 1:50)", key=scale_key)

            # Legacy scale slider (px per meter)
            scale_val = st.slider(
                "פיקסלים למטר (Legacy)",
                10.0,
                1000.0,
                float(proj.get("scale", 200.0)),
                key=f"scale_slider_{selected}",
            )
            proj["scale"] = scale_val

            corrected_walls = get_corrected_walls(selected, proj)
            corrected_pixels = int(np.count_nonzero(corrected_walls))

            total_len = corrected_pixels / scale_val

            st.write(f"🧱 פיקסלים (קירות): {corrected_pixels:,}")
            st.write(f"📏 אורך קירות (Legacy, מטר): {total_len:.2f}")

            st.markdown("---")

            # ------------------------------------------------------------
            # 📐 מדידות מתקדמות (Stage 1 + 2)
            # ------------------------------------------------------------
            with st.expander("📐 מדידות מתקדמות (Stage 1 + 2)", expanded=True):
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

                # בחירת גודל נייר ידנית
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

                if selected_paper != "זיהוי אוטומטי":
                    st.session_state[paper_override_key] = selected_paper

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
                    meta["paper_detection_confidence"] = 1.0

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
                    del st.session_state[paper_override_key]

                # במקום expander מקונן: checkbox
                show_formulas = st.checkbox(
                    "👁️ הצג נוסחאות", value=True, key=f"show_formulas_{selected}"
                )
                if show_formulas:
                    st.markdown("#### 🔎 חישוב צעד-אחר-צעד")
                    if meta.get("paper_mm") and meta.get("image_size_px"):
                        pw = meta["paper_mm"]["width"]
                        ph = meta["paper_mm"]["height"]
                        wpx = meta["image_size_px"]["width"]
                        hpx = meta["image_size_px"]["height"]
                        st.code(
                            f"mm_per_pixel = average({pw}/{wpx}, {ph}/{hpx})\n"
                            f"mm_per_pixel = {meta.get('mm_per_pixel')}"
                        )
                        if meta.get("scale_denominator"):
                            sd = meta["scale_denominator"]
                            st.code(
                                f"meters_per_pixel = (mm_per_pixel * scale_denominator) / 1000\n"
                                f"meters_per_pixel = ({meta.get('mm_per_pixel')} * {sd}) / 1000 = {meta.get('meters_per_pixel')}"
                            )

                    if meta.get("wall_length_total_px") and meta.get(
                        "meters_per_pixel"
                    ):
                        st.code(
                            f"wall_length_m = wall_length_total_px * meters_per_pixel\n"
                            f"wall_length_m = {meta['wall_length_total_px']} * {meta['meters_per_pixel']} = {meta.get('wall_length_total_m')}"
                        )

            st.markdown("---")

            # מחירון (קיים אצלך)
            with st.expander("💰 תמחור אוטומטי (Beta)", expanded=False):
                st.caption("הערכה אוטומטית בסיסית – לפי אורכי קירות ושטח ריצוף אם זוהו")

                c_price = st.number_input("מחיר בטון למטר", 0.0, 2000.0, 350.0)
                b_price = st.number_input("מחיר בלוקים למטר", 0.0, 2000.0, 250.0)
                f_price = st.number_input("מחיר ריצוף למ״ר", 0.0, 2000.0, 180.0)

                meta = proj.get("metadata", {})

                conc_len = float(meta.get("concrete_length_m", 0) or 0)
                block_len = float(meta.get("blocks_length_m", 0) or 0)
                floor_area = float(meta.get("floor_area_m2", 0) or 0)

                total_quote = (
                    conc_len * c_price + block_len * b_price + floor_area * f_price
                )

                quote_df = pd.DataFrame(
                    {
                        "סוג": ["בטון", "בלוקים", "ריצוף", "סה״כ"],
                        "כמות": [conc_len, block_len, floor_area, ""],
                        "יחידה": ["מ׳", "מ׳", "מ״ר", ""],
                        "מחיר יחידה": [c_price, b_price, f_price, ""],
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
                        "concrete": float(
                            proj["metadata"].get("concrete_length_m", 0) or 0
                        ),
                        "blocks": float(
                            proj["metadata"].get("blocks_length_m", 0) or 0
                        ),
                        "flooring": float(
                            proj["metadata"].get("floor_area_m2", 0) or 0
                        ),
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
                corrected = get_corrected_walls(selected, proj)
                overlay = create_colored_overlay(proj["original"], corrected)
            else:
                overlay = proj.get("overlay")

            # חשוב: ב-streamlit 1.28–1.29 משתמשים ב-use_column_width
            if overlay is not None:
                st.image(
                    overlay,
                    caption="Overlay",
                    use_column_width=True,
                )
            elif proj.get("original") is not None:
                st.image(
                    proj["original"],
                    caption="מקור",
                    use_column_width=True,
                )

            # הצגת אורך קירות "סופי" אם קיים
            meta = proj.get("metadata", {})
            if meta.get("wall_length_total_m") is not None:
                st.success(f"🧱 אורך קירות (מ׳): {meta['wall_length_total_m']:.2f}")
            elif (
                meta.get("wall_length_total_px") is not None
                and meta.get("meters_per_pixel") is not None
            ):
                st.success(
                    f"🧱 אורך קירות (מ׳): {meta['wall_length_total_px'] * meta['meters_per_pixel']:.2f}"
                )
            else:
                st.info(
                    "ℹ️ עדיין אין חישוב מטרי מלא (בדוק קנה מידה/גודל דף וזיהוי קירות)."
                )


def render_corrections_tab():
    """טאב 2: תיקונים ידניים"""
    st.markdown("## 🎨 תיקונים ידניים")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected_plan = st.selectbox(
        "בחר תוכנית לתיקון:",
        list(st.session_state.projects.keys()),
        key="correction_plan_select",
    )
    proj = st.session_state.projects[selected_plan]

    st.info("טאב תיקונים – נשאר כמו במערכת המקורית. (אפשר להרחיב לפי צורך)")


def render_dashboard_tab():
    """טאב 3: דשבורד"""
    st.markdown("## 📊 דשבורד")

    try:
        df = load_stats_df()
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("אין עדיין נתונים להצגה.")
    except Exception:
        st.info("אין עדיין נתונים להצגה.")


def render_invoices_tab():
    """טאב 4: חשבונות"""
    st.markdown("## 💰 חשבונות")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected_plan = st.selectbox(
        "בחר תוכנית לחשבונית/דוח:",
        list(st.session_state.projects.keys()),
        key="invoices_plan_select",
    )
    proj = st.session_state.projects[selected_plan]

    st.markdown("### 📄 דוח סטטוס")
    if st.button("צור דוח PDF", key="btn_status_pdf"):
        meta = proj.get("metadata", {})
        pdf_bytes = generate_status_pdf(
            plan_name=meta.get("plan_name", selected_plan),
            metadata=meta,
        )
        st.download_button(
            "⬇️ הורד דוח סטטוס",
            data=pdf_bytes,
            file_name=f"status_{selected_plan}.pdf",
            mime="application/pdf",
        )

    st.markdown("---")
    st.markdown("### 🧾 חשבונית תשלום")
    if st.button("צור חשבונית PDF", key="btn_invoice_pdf"):
        meta = proj.get("metadata", {})
        invoice_data = get_payment_invoice_data()
        pdf_bytes = generate_payment_invoice_pdf(
            plan_name=meta.get("plan_name", selected_plan),
            invoice_data=invoice_data,
        )
        st.download_button(
            "⬇️ הורד חשבונית",
            data=pdf_bytes,
            file_name=f"invoice_{selected_plan}.pdf",
            mime="application/pdf",
        )
     except Exception as e:
         st.error(f"❌ שגיאה ביצירת חשבונית: {e}")