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

    with st.container():
        st.subheader("העלאת קבצים")
        st.caption("העלאה, בחירת אזור ניתוח (Crop), והרצה על קבצי PDF")
        files = st.file_uploader(
            "גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True
        )
        debug_mode = st.selectbox(
            "מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0
        )
        show_debug = debug_mode != "בסיסי"

        if files:
            st.markdown("### ✂️ בחירת אזור ניתוח (אופציונלי)")
            st.caption(
                "סמן מלבן סביב אזור השרטוט (בלי הטקסטים בצד). החיתוך משפיע **רק** על חישוב קירות/רצפה/סקלטון. "
                "הטקסט מה-PDF נשמר במטאדאטה עבור מקרא/מידע."
            )

            for f in files:
                if f.name in st.session_state.projects:
                    continue

                with st.container():
                    st.markdown(f"#### 📄 {f.name}")
                    st.markdown("---")
                    # שמירת ה-PDF לקובץ זמני
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        path = tmp.name

                    # Preview (עמוד 1) כדי שהמשתמש יסמן ROI
                    analyzer_preview = FloorPlanAnalyzer()
                    try:
                        img_bgr_full = analyzer_preview.pdf_to_image(path)
                        img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        st.error(f"❌ לא הצלחתי ליצור Preview ל-crop: {e}")
                        continue

                    full_h, full_w = img_rgb_full.shape[:2]

                    # הצגה ברזולוציה ידידותית (שומר יחס)
                    disp_w = min(1000, full_w)
                    disp_scale = disp_w / full_w
                    disp_h = int(full_h * disp_scale)

                    img_rgb_disp = cv2.resize(
                        img_rgb_full, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    )
                    bg_pil = Image.fromarray(img_rgb_disp)

                    st.markdown("#### 1) סמן מלבן סביב השרטוט")
                    st.caption("טיפ: אם לא מסמנים כלום – המערכת תנתח את כל הדף (כמו היום).")

                    canvas_key = f"crop_canvas_{f.name}"
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 0)",
                        stroke_width=2,
                        stroke_color="#00FF00",
                        background_image=bg_pil,
                        update_streamlit=True,
                        height=disp_h,
                        width=disp_w,
                        drawing_mode="rect",
                        key=canvas_key,
                    )

                    # חילוץ bbox מה-canvas (בפיקסלים של התמונה המקורית)
                    crop_bbox = None
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        obj = canvas_result.json_data["objects"][-1]
                        left = float(obj.get("left", 0))
                        top = float(obj.get("top", 0))
                        width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                        height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

                        x = int(left / disp_scale)
                        y = int(top / disp_scale)
                        w = int(width / disp_scale)
                        h = int(height / disp_scale)

                        x = max(0, min(x, full_w - 1))
                        y = max(0, min(y, full_h - 1))
                        w = max(1, min(w, full_w - x))
                        h = max(1, min(h, full_h - y))
                        crop_bbox = (x, y, w, h)

                    st.session_state[f"crop_bbox_{f.name}"] = crop_bbox
                    if crop_bbox:
                        st.success(
                            f"✅ אזור ניתוח נבחר: x={crop_bbox[0]}, y={crop_bbox[1]}, w={crop_bbox[2]}, h={crop_bbox[3]}"
                        )
                    else:
                        st.info("ℹ️ לא נבחר אזור – ניתוח יתבצע על כל הדף.")

                    st.markdown("#### 2) הרץ ניתוח")
                    analyze_btn = st.button(
                        f"🚀 נתח והוסף את {f.name}", key=f"analyze_btn_{f.name}"
                    )

                    if analyze_btn:
                        with st.spinner(f"מעבד {f.name} עם Multi-Pass Detection..."):
                            try:
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
                                    path,
                                    save_debug=show_debug,
                                    crop_bbox=crop_bbox,
                                )

                                if not meta.get("plan_name"):
                                    meta["plan_name"] = (
                                        f.name.replace(".pdf", "").replace("-", " ").strip()
                                    )

                                llm_data = {}
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(meta["raw_text"])
                                    meta.update({k: v for k, v in llm_data.items() if v})

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
                                    "debug_image": (debug_img if show_debug else None),
                                }

                                st.toast("✅ הניתוח הושלם והתווסף לרשימה")
                                st.success(f"✅ {f.name} נוסף לתוכניות")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ שגיאה בעיבוד: {e}")
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
            corrected_pixels = np.count_nonzero(corrected_walls)
            total_len = corrected_pixels / scale_val

            kernel = np.ones((6, 6), np.uint8)
            conc_corrected = cv2.dilate(
                cv2.erode(corrected_walls, kernel, iterations=1), kernel, iterations=2
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

                prev_override = st.session_state.get(paper_override_key)

                # ⚠️ חשוב: להימנע מ-loop אינסופי של rerun.
                # נבצע rerun רק אם המשתמש באמת שינה את הבחירה (או אם מוחקים override).
                if selected_paper != "זיהוי אוטומטי":
                    if prev_override != selected_paper:
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

                        # התאמת כיוון (portrait/landscape) לפי יחס התמונה
                        if meta.get("image_size_px"):
                            w_px = meta["image_size_px"]["width"]
                            h_px = meta["image_size_px"]["height"]
                            if w_px > h_px and paper_w_mm < paper_h_mm:
                                paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm

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
                else:
                    if prev_override is not None:
                        del st.session_state[paper_override_key]
                        st.rerun()

                # Debug - למה לא זוהה?
                if not scale_denom:
                    st.markdown("---")
                    st.markdown("#### 🔍 למה קנה מידה לא זוהה?")

                    with st.container():
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
                                # ============================================
                # 📊 תצוגת נתוני חישוב מה-PDF
                # ============================================
                st.markdown("---")
                st.markdown("#### 📊 נתוני חישוב מה-PDF")

                # בדיקה האם יש נתונים
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
                    # תצוגה קומפקטית עם מספרים קטנים
                    st.markdown("---")
                    st.markdown("#### 📊 נתוני חישוב")

                    # 1. גודל נייר ותמונה
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
                        st.markdown(f"**🖼️ תמונה:**  \n`{img_w}×{img_h}` px")

                    # 2. יחסי המרה
                    mm_per_px = meta["mm_per_pixel"]
                    m_per_px = meta["meters_per_pixel"]
                    scale_denom = meta["scale_denominator"]

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.markdown(f'**מ"מ/px**  \n`{mm_per_px:.4f}`')
                    with col_r2:
                        st.markdown(f"**קנה מידה**  \n`1:{scale_denom}`")
                    with col_r3:
                        st.markdown(f"**מטר/px**  \n`{m_per_px:.6f}`")

                    # 3. חישוב מפורט
                    st.markdown("**3️⃣ חישוב צעד אחר צעד:**")
                    show_formulas = st.checkbox("👁️ הצג נוסחאות", value=True, key=f"show_formulas_{selected}")
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
                    if not meta.get("wall_length_total_px") and proj.get("skeleton") is not None:
                        try:
                            meta["wall_length_total_px"] = float(compute_skeleton_length_px(proj["skeleton"]))
                        except Exception:
                            pass

                    # 4. תוצאות סופיות
                    if meta.get("wall_length_total_px"):
                        wall_px = meta["wall_length_total_px"]
                        wall_m = meta.get("wall_length_total_m", wall_px * m_per_px)

                        st.markdown("---")
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.success(f"📏 `{wall_px:,.0f}` פיקסלים")
                        with col_w2:
                            st.success(f"📐 `{wall_m:.2f}` מטר")

                    st.markdown("---")

                else:
                    # אין נתונים מספיקים
                    st.markdown("---")
                    st.warning("⚠️ חסרים נתונים - בחר גודל נייר וקנה מידה למעלה")

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
            st.info("אין תיקונים ידניים עדיין. עבור לטאב 'הוסף קירות' או 'הסר קירות'")


def render_dashboard_tab():
    """טאב 3: דשבורד"""
    from pages.dashboard import render_dashboard

    render_dashboard()


def render_invoices_tab():
    """טאב 4: חשבונות"""
    from pages.invoices import render_invoices

    render_invoices()