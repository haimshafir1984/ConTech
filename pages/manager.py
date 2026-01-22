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
    create_colored_overlay,
    overlay_masks_alpha,
    generate_project_overview_html,
    render_sidebar_header,
    safe_float,
    iso_paper_mm,
)

# ------------------------------------------------------------
# Session State Init
# ------------------------------------------------------------
def ensure_session_state():
    if "projects" not in st.session_state:
        st.session_state.projects = {}
    if "selected_plan_id" not in st.session_state:
        st.session_state.selected_plan_id = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _npimg_from_pil(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return _rgb_to_bgr(rgb)


def _pil_from_npimg_bgr(img_bgr: np.ndarray) -> Image.Image:
    rgb = _bgr_to_rgb(img_bgr)
    return Image.fromarray(rgb)


def _clamp_bbox(x, y, w, h, W, H):
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))
    return x, y, w, h


def _extract_bbox_from_canvas(canvas_result):
    """
    מחלץ bbox (x,y,w,h) מתוך streamlit-drawable-canvas אם המשתמש צייר מלבן אחד.
    מחזיר None אם לא צויר/אין מידע תקין.
    """
    if not canvas_result:
        return None
    if not getattr(canvas_result, "json_data", None):
        return None
    data = canvas_result.json_data
    if not data or "objects" not in data or not data["objects"]:
        return None

    # נחפש אובייקט מסוג rect (המלבן האחרון)
    rects = [o for o in data["objects"] if o.get("type") in ("rect", "rectangle")]
    if not rects:
        return None
    r = rects[-1]
    x = r.get("left")
    y = r.get("top")
    w = r.get("width")
    h = r.get("height")
    if x is None or y is None or w is None or h is None:
        return None

    # Fabric.js width/height יכולים להיות מושפעים מ-scaleX/scaleY
    sx = r.get("scaleX", 1.0) or 1.0
    sy = r.get("scaleY", 1.0) or 1.0
    w = w * sx
    h = h * sy

    return int(x), int(y), int(w), int(h)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
def render_workshop_tab():
    """טאב 1: סדנת עבודה - העלאה, ניתוח, תצוגה"""
    ensure_session_state()

    st.markdown("## 🧰 סדנת עבודה")
    st.caption("העלה PDF, סמן אזור שרטוט (אופציונלי), ונתח את התוכנית")

    # --- העלאת קבצים (ללא expander חיצוני כדי להימנע מ-nested expander)
    upload_container = st.container()
    with upload_container:
        st.markdown("### 📤 העלאת תוכניות (PDF)")
        uploaded_files = st.file_uploader(
            "בחר קובץ/ים PDF", type=["pdf"], accept_multiple_files=True
        )

        show_debug = st.checkbox("הצג דיבאג (תמונות ביניים)", value=False)
        st.markdown("---")

        if uploaded_files:
            for f in uploaded_files:
                file_container = st.container()
                with file_container:
                    st.markdown(f"### 📄 {f.name}")

                    # אם הקובץ כבר נותח, נציג הודעה קצרה
                    if f.name in st.session_state.projects:
                        st.info("כבר נותח ונמצא ברשימת התוכניות (למטה).")

                    # שמירה זמנית לקובץ כדי שהאנלייזר יקרא אותו
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name

                    # Preview תמונת PDF (עמוד ראשון)
                    preview_img_bgr = None
                    try:
                        analyzer_preview = FloorPlanAnalyzer(debug=False)
                        preview_img_bgr = analyzer_preview.render_pdf_to_image(tmp_path)
                        st.image(_bgr_to_rgb(preview_img_bgr), use_container_width=True)
                    except Exception as e:
                        st.warning(f"לא הצלחתי להציג פריוויו: {e}")

                    # ROI crop (אופציונלי)
                    crop_bbox = None
                    if preview_img_bgr is not None:
                        st.markdown("#### ✂️ חיתוך אזור השרטוט (אופציונלי)")
                        st.caption(
                            "צייר מלבן סביב אזור השרטוט כדי שהניתוח יתבצע רק עליו (הטקסטים בצד לא ייכנסו למדידה)."
                        )
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 0, 0, 0.05)",
                            stroke_width=2,
                            stroke_color="rgba(255, 0, 0, 0.8)",
                            background_image=_pil_from_npimg_bgr(preview_img_bgr),
                            update_streamlit=True,
                            height=min(650, preview_img_bgr.shape[0]),
                            width=min(1200, preview_img_bgr.shape[1]),
                            drawing_mode="rect",
                            key=f"canvas_{f.name}",
                        )
                        bbox = _extract_bbox_from_canvas(canvas_result)
                        if bbox:
                            x, y, w, h = bbox
                            H, W = preview_img_bgr.shape[:2]
                            x, y, w, h = _clamp_bbox(x, y, w, h, W, H)
                            crop_bbox = {"x": x, "y": y, "w": w, "h": h}
                            st.success(f"✅ אזור נבחר: x={x}, y={y}, w={w}, h={h}")
                        else:
                            st.info("לא נבחר אזור חיתוך (הניתוח יתבצע על כל הדף).")

                    # כפתור הרצה
                    run_btn = st.button(
                        "🚀 נתח והוסף לרשימה",
                        key=f"analyze_{f.name}",
                        use_container_width=True,
                    )

                    if run_btn:
                        with st.spinner("מנתח את התוכנית..."):
                            try:
                                analyzer = FloorPlanAnalyzer(debug=show_debug)
                                results = analyzer.process_file(
                                    tmp_path,
                                    crop_bbox=crop_bbox,
                                )

                                original = results.get("original")
                                overlay = results.get("overlay")
                                meta = results.get("metadata", {}) or {}
                                conc = results.get("concrete_mask")
                                blok = results.get("blocks_mask")
                                floor = results.get("flooring_mask")
                                debug_img = results.get("debug_image")

                                # חישוב אורך סקלטון בפיקסלים (במידה וקיים)
                                skeleton = results.get("skeleton")
                                pix = 0.0
                                if skeleton is not None:
                                    try:
                                        pix = compute_skeleton_length_px(skeleton)
                                    except Exception:
                                        pix = 0.0

                                # שמירה ל-session
                                st.session_state.projects[f.name] = {
                                    "file_name": f.name,
                                    "original": original,
                                    "overlay": overlay,
                                    "metadata": meta,
                                    "concrete_mask": conc,
                                    "blocks_mask": blok,
                                    "flooring_mask": floor,
                                    "total_length": pix / 200.0,
                                    "llm_suggestions": (
                                        meta.get("llm_suggestions", {})
                                        if meta.get("raw_text")
                                        else {}
                                    ),
                                    "debug_image": (debug_img if show_debug else None),
                                    # שמירה של ROI למעקב
                                    "analysis_crop": crop_bbox,
                                    "skeleton": skeleton,
                                }

                                st.toast("✅ הניתוח הושלם והתווסף לרשימה")
                                st.success(f"✅ {f.name} נוסף לתוכניות")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ שגיאה בעיבוד: {e}")
                    st.markdown("---")

        # ✅ Guard: אם אין עדיין תוכניות, ה-selectbox יחזיר None ויפיל את האפליקציה
        projects = st.session_state.get("projects", {})
        if not projects:
            st.info("📂 עדיין לא נטענו תוכניות. העלה PDF כדי להתחיל.")
            return

        selected = st.selectbox(
            "בחר תוכנית לעריכה:",
            options=list(projects.keys()),
            key="workshop_selected_plan",
        )
        if selected is None or selected not in projects:
            st.warning("בחר תוכנית כדי להמשיך.")
            return

        proj = projects[selected]

        name_key = f"name_{selected}"
        scale_key = f"scale_{selected}"
        if name_key not in st.session_state:
            st.session_state[name_key] = proj["metadata"].get("plan_name", "")
        if scale_key not in st.session_state:
            st.session_state[scale_key] = proj["metadata"].get("scale", "")

        col_edit, col_preview = st.columns([1, 1.5], gap="large")

        # ------------------------------------------------------------
        # Edit column
        # ------------------------------------------------------------
        with col_edit:
            st.markdown("### ✏️ פרטי תוכנית")
            st.text_input("שם תוכנית", key=name_key)

            # Scale input
            st.text_input("קנה מידה (למשל 1:50)", key=scale_key)

            # Paper size manual override
            st.markdown("### 📄 גודל נייר")
            current_paper = proj["metadata"].get("paper_size", None)

            paper_options = ["אוטומטי"] + list(iso_paper_mm.keys())
            default_index = 0
            if current_paper in iso_paper_mm:
                default_index = paper_options.index(current_paper)

            paper_select_key = f"paper_select_{selected}"
            chosen_paper = st.selectbox(
                "בחר גודל נייר",
                options=paper_options,
                index=default_index,
                key=paper_select_key,
            )

            # Apply paper (manual override)
            apply_paper = st.button("✅ החל גודל נייר", use_container_width=True)

            if apply_paper:
                if chosen_paper == "אוטומטי":
                    proj["metadata"].pop("paper_size", None)
                else:
                    proj["metadata"]["paper_size"] = chosen_paper
                    wmm, hmm = iso_paper_mm[chosen_paper]
                    proj["metadata"]["paper_width_mm"] = wmm
                    proj["metadata"]["paper_height_mm"] = hmm
                st.toast("📄 גודל נייר נשמר")
                st.rerun()

            st.markdown("---")
            st.markdown("### 📌 שמירה")
            if st.button("💾 שמור תוכנית לבסיס נתונים", use_container_width=True):
                try:
                    save_plan(
                        file_name=selected,
                        plan_name=st.session_state[name_key],
                        metadata=proj["metadata"],
                    )
                    st.success("✅ נשמר בהצלחה")
                except Exception as e:
                    st.error(f"❌ שגיאה בשמירה: {e}")

            st.markdown("---")

            # Debug formulas - no nested expander, use checkbox
            st.markdown("### 🧮 חישובי מדידה (בדיקה)")
            show_formulas = st.checkbox("👁️ הצג נוסחאות וחישוב צעד-אחר-צעד", value=True)

            meta = proj.get("metadata", {}) or {}
            paper = meta.get("paper_size") or "לא ידוע"
            pw = meta.get("paper_width_mm")
            ph = meta.get("paper_height_mm")
            img_w = meta.get("image_width_px")
            img_h = meta.get("image_height_px")
            mm_per_px = meta.get("mm_per_px")
            scale = meta.get("scale")
            meters_per_px = meta.get("meters_per_px")

            st.write("📊 נתוני חישוב מה-PDF")
            if pw and ph:
                st.write(f"📄 נייר: {paper} {pw}×{ph} מ\"מ")
            if img_w and img_h:
                st.write(f"🖼️ תמונה: {img_w}×{img_h} px")
            if mm_per_px:
                st.write(f"מ\"מ/px {mm_per_px:.4f}")
            if scale:
                st.write(f"קנה מידה {scale}")
            if meters_per_px:
                st.write(f"מטר/px {meters_per_px:.6f}")

            if show_formulas:
                st.write("3️⃣ חישוב צעד אחר צעד:")
                if pw and ph and img_w and img_h:
                    st.code(
                        f"mm_per_px = average({pw}/{img_w}, {ph}/{img_h}) = {mm_per_px}"
                    )
                if mm_per_px and scale:
                    try:
                        denom = int(str(scale).split(":")[1])
                        st.code(
                            f"meters_per_px = (mm_per_px * scale_denominator) / 1000\n"
                            f"= ({mm_per_px} * {denom}) / 1000 = {meters_per_px}"
                        )
                    except Exception:
                        pass

            st.markdown("---")

        # ------------------------------------------------------------
        # Preview column
        # ------------------------------------------------------------
        with col_preview:
            st.markdown("### 👀 תצוגה")
            if proj.get("overlay") is not None:
                st.image(_bgr_to_rgb(proj["overlay"]), use_container_width=True)
                st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")
            elif proj.get("original") is not None:
                st.image(_bgr_to_rgb(proj["original"]), use_container_width=True)
            else:
                st.info("אין תמונה להצגה")

            # סיכום אורך קירות
            st.markdown("### 📏 אורך קירות (סופי)")

            # ננסה להביא wall_length_total_px מתוך metadata
            wall_px = meta.get("wall_length_total_px")
            if wall_px is None:
                # fallback: לחשב מה-skeleton אם קיים
                skeleton = proj.get("skeleton")
                if skeleton is not None:
                    try:
                        wall_px = compute_skeleton_length_px(skeleton)
                    except Exception:
                        wall_px = None

            if wall_px is None:
                st.warning("לא הצלחתי לחשב אורך קירות בפיקסלים (ייתכן שלא נוצר skeleton).")
            else:
                st.write(f"🧱 אורך קירות (px): {wall_px:.1f}")
                if meters_per_px:
                    st.success(f"🧱 אורך קירות (מ׳): {wall_px * meters_per_px:.2f}")
                else:
                    st.info("חסר meters_per_px (בדוק קנה מידה וגודל נייר).")


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

    st.image(rgb, use_container_width=True)
    st.info("טאב זה הוא בסיס לתיקונים — אפשר להרחיב בהמשך לפי הצורך.")


def render_reports_tab():
    """טאב 3: דוחות וסטטוסים"""
    st.markdown("## 📑 דוחות")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected_plan = st.selectbox(
        "בחר תוכנית לדוח:",
        list(st.session_state.projects.keys()),
        key="reports_plan_select",
    )
    proj = st.session_state.projects[selected_plan]

    st.markdown("### 🧾 יצוא דוח סטטוס")
    if st.button("📄 צור דוח PDF", use_container_width=True):
        try:
            pdf_bytes = generate_status_pdf(
                plan_name=proj_
