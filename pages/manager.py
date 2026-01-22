"""
ConTech Pro - Manager Pages
חייב לכלול את הפונקציות שהאפליקציה מייבאת ב-app.py:
- render_workshop_tab
- render_corrections_tab
- render_dashboard_tab
- render_invoices_tab
"""

import os
import json
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from analyzer import FloorPlanAnalyzer, compute_skeleton_length_px
from reporter import generate_status_pdf, generate_payment_invoice_pdf
from database import (
    save_plan,
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


# --------------------------------------------------------------------------------------
# Session State
# --------------------------------------------------------------------------------------
def _ensure_state():
    if "projects" not in st.session_state:
        st.session_state.projects = {}
    if "manual_corrections" not in st.session_state:
        st.session_state.manual_corrections = {}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _pil_from_bgr(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(_bgr_to_rgb(img_bgr))


def _clamp_bbox(x, y, w, h, W, H):
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))
    return x, y, w, h


def _extract_bbox_from_canvas(canvas_result):
    """
    מחלץ bbox (x,y,w,h) מתוך streamlit-drawable-canvas אם צויר rect.
    """
    if not canvas_result or not getattr(canvas_result, "json_data", None):
        return None
    data = canvas_result.json_data or {}
    objs = data.get("objects") or []
    rects = [o for o in objs if o.get("type") in ("rect", "rectangle")]
    if not rects:
        return None
    r = rects[-1]
    x = r.get("left")
    y = r.get("top")
    w = r.get("width")
    h = r.get("height")
    if x is None or y is None or w is None or h is None:
        return None
    sx = r.get("scaleX", 1.0) or 1.0
    sy = r.get("scaleY", 1.0) or 1.0
    w = w * sx
    h = h * sy
    return int(x), int(y), int(w), int(h)


def _normalize_analyzer_result(res):
    """
    תומך גם בפורמט tuple של האנלייזר המקורי וגם בפורמט dict (אם שינית).
    מחזיר:
      pix, skel, thick, orig, meta, conc, blok, floor, debug_img
    """
    if isinstance(res, dict):
        pix = res.get("raw_pixels") or res.get("pix") or 0.0
        skel = res.get("skeleton")
        thick = res.get("thick_walls") or res.get("final_walls") or res.get("walls")
        orig = res.get("original") or res.get("image") or res.get("image_proc")
        meta = res.get("metadata") or {}
        conc = res.get("concrete_mask")
        blok = res.get("blocks_mask")
        floor = res.get("flooring_mask")
        debug_img = res.get("debug_image")
        return pix, skel, thick, orig, meta, conc, blok, floor, debug_img

    # tuple המקורי: (pix, skel, final_walls, image_proc, meta, concrete, blocks_mask, flooring, debug_img)
    if isinstance(res, tuple) and len(res) >= 9:
        pix, skel, thick, orig, meta, conc, blok, floor, debug_img = res[:9]
        return pix, skel, thick, orig, meta, conc, blok, floor, debug_img

    # fallback
    return 0.0, None, None, None, {}, None, None, None, None


def get_corrected_walls(selected_plan, proj):
    """מחזיר את מסכת הקירות המתוקנת (אם יש תיקונים)"""
    if selected_plan in st.session_state.manual_corrections:
        corrections = st.session_state.manual_corrections[selected_plan]
        corrected = proj["thick_walls"].copy()

        if "added_walls" in corrections and corrections["added_walls"] is not None:
            corrected = cv2.bitwise_or(corrected, corrections["added_walls"])

        if "removed_walls" in corrections and corrections["removed_walls"] is not None:
            corrected = cv2.subtract(corrected, corrections["removed_walls"])

        return corrected
    return proj["thick_walls"]


# --------------------------------------------------------------------------------------
# TAB 1: Workshop
# --------------------------------------------------------------------------------------
def render_workshop_tab():
    _ensure_state()

    st.markdown("## 🧰 סדנת עבודה")

    # ⛔ אין expander חיצוני כדי לא ליפול על nested expander
    files = st.file_uploader(
        "גרור PDF או לחץ לבחירה",
        type="pdf",
        accept_multiple_files=True,
        key="mgr_uploader",
    )

    colA, colB = st.columns([1, 1], gap="medium")
    with colA:
        show_debug = st.checkbox("הצג Debug", value=False)
    with colB:
        debug_mode = st.selectbox(
            "מצב Debug",
            ["בסיסי", "מפורט - שכבות"],
            index=0,
            disabled=not show_debug,
        )

    st.markdown("---")

    # ניתוח קבצים
    if files:
        for f in files:
            st.markdown(f"### 📄 {f.name}")

            # קובץ זמני
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name

            # Preview + Crop ROI
            crop_bbox = None
            preview_img = None
            try:
                analyzer_preview = FloorPlanAnalyzer()
                preview_img = analyzer_preview.pdf_to_image(path)
                st.image(
                    _bgr_to_rgb(preview_img),
                    caption="תצוגה מקדימה",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"לא הצלחתי להציג תצוגה מקדימה: {e}")

            if preview_img is not None:
                st.markdown("#### ✂️ חיתוך אזור שרטוט (אופציונלי)")
                st.caption(
                    "צייר מלבן סביב השרטוט כדי למדוד רק את האזור הזה (הטקסטים בצד לא ייכנסו לחישוב)."
                )

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.05)",
                    stroke_width=2,
                    stroke_color="rgba(255, 0, 0, 0.8)",
                    background_image=_pil_from_bgr(preview_img),
                    update_streamlit=True,
                    drawing_mode="rect",
                    height=min(650, preview_img.shape[0]),
                    width=min(1200, preview_img.shape[1]),
                    key=f"crop_canvas_{f.name}",
                )
                bbox = _extract_bbox_from_canvas(canvas_result)
                if bbox:
                    x, y, w, h = bbox
                    H, W = preview_img.shape[:2]
                    x, y, w, h = _clamp_bbox(x, y, w, h, W, H)
                    crop_bbox = {"x": x, "y": y, "w": w, "h": h}
                    st.success(f"✅ ROI: x={x}, y={y}, w={w}, h={h}")
                else:
                    st.info("לא נבחר ROI — הניתוח יתבצע על כל הדף.")

            run_btn = st.button(
                "🚀 נתח והוסף", key=f"run_{f.name}", use_container_width=True
            )

            if run_btn:
                try:
                    analyzer = FloorPlanAnalyzer()

                    # נסה להריץ עם crop_bbox אם האנלייזר תומך. אם לא — fallback.
                    try:
                        res = analyzer.process_file(
                            path, save_debug=show_debug, crop_bbox=crop_bbox
                        )
                    except TypeError:
                        res = analyzer.process_file(path, save_debug=show_debug)

                    pix, skel, thick, orig, meta, conc, blok, floor, debug_img = (
                        _normalize_analyzer_result(res)
                    )

                    # שילוב LLM מטא־דאטה אם קיים raw_text
                    llm_data = {}
                    try:
                        if isinstance(meta, dict) and meta.get("raw_text"):
                            llm_data = safe_process_metadata(meta["raw_text"])
                            if isinstance(llm_data, dict):
                                meta.update({k: v for k, v in llm_data.items() if v})
                    except Exception:
                        llm_data = {}

                    st.session_state.projects[f.name] = {
                        "skeleton": skel,
                        "thick_walls": thick,
                        "original": orig,
                        "raw_pixels": float(pix or 0.0),
                        "scale": 200.0,  # fallback legacy px/meter
                        "metadata": meta if isinstance(meta, dict) else {},
                        "concrete_mask": conc,
                        "blocks_mask": blok,
                        "flooring_mask": floor,
                        "total_length": (float(pix or 0.0) / 200.0) if pix else 0.0,
                        "llm_suggestions": (
                            llm_data if meta and isinstance(meta, dict) else {}
                        ),
                        "debug_layers": getattr(analyzer, "debug_layers", {}),
                        "analysis_crop": crop_bbox,
                    }

                    # Debug תצוגה (ללא expanders מקוננים)
                    if show_debug and debug_img is not None:
                        st.markdown("#### 🔍 Debug")
                        st.image(
                            debug_img, caption="תוצאה משולבת", use_container_width=True
                        )

                    st.success(f"✅ {f.name} נותח ונוסף לרשימה")
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

                    st.rerun()

                except Exception as e:
                    st.error(f"שגיאה: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

            st.markdown("---")

    # ✅ Guard: אם אין פרויקטים — לא מציגים selectbox שמחזיר None
    if not st.session_state.projects:
        st.info("📂 עדיין לא נטענו תוכניות. העלה PDF כדי להתחיל.")
        return

    st.markdown("## 📌 תוכניות שנטענו")

    selected = st.selectbox(
        "בחר תוכנית לעריכה:",
        list(st.session_state.projects.keys()),
        key="mgr_select_plan",
    )
    if selected is None or selected not in st.session_state.projects:
        st.warning("בחר תוכנית כדי להמשיך.")
        return

    proj = st.session_state.projects[selected]

    name_key = f"name_{selected}"
    scale_key = f"scale_text_{selected}"
    if name_key not in st.session_state:
        st.session_state[name_key] = proj.get("metadata", {}).get("plan_name", "")
    if scale_key not in st.session_state:
        st.session_state[scale_key] = proj.get("metadata", {}).get("scale", "")

    col_edit, col_preview = st.columns([1, 1.5], gap="large")

    # ---------------------------
    # Edit
    # ---------------------------
    with col_edit:
        st.markdown("### הגדרות תוכנית")

        if selected in st.session_state.manual_corrections:
            st.success("✏️ תוכנית זו תוקנה ידנית")

        p_name = st.text_input("שם התוכנית", key=name_key)
        p_scale_text = st.text_input(
            "קנה מידה (לתיעוד)", key=scale_key, placeholder="1:50"
        )

        # Legacy calibration (px per meter) - נשאר
        st.markdown("#### כיול (Legacy)")
        scale_val = st.slider(
            "פיקסלים למטר",
            10.0,
            1000.0,
            float(proj.get("scale", 200.0)),
            key=f"scale_slider_{selected}",
        )
        proj["scale"] = scale_val

        corrected_walls = get_corrected_walls(selected, proj)
        corrected_pixels = (
            int(np.count_nonzero(corrected_walls)) if corrected_walls is not None else 0
        )
        total_len_legacy = (corrected_pixels / scale_val) if scale_val else 0.0

        st.write(f"🧱 פיקסלים (קירות): {corrected_pixels:,}")
        st.write(f"📏 אורך קירות (Legacy, מטר): {total_len_legacy:.2f}")

        # אם יש meters_per_px מהמטא — נחשב גם “השיטה החדשה”
        meta = proj.get("metadata", {}) or {}
        meters_per_px = meta.get("meters_per_px")
        wall_px = meta.get("wall_length_total_px")

        # fallback ל-skeleton אם חסר wall_length_total_px
        if wall_px is None and proj.get("skeleton") is not None:
            try:
                wall_px = compute_skeleton_length_px(proj["skeleton"])
            except Exception:
                wall_px = None

        st.markdown("#### חישוב לפי דף + קנה מידה (אם זמין)")
        if meters_per_px and wall_px:
            st.success(
                f"📏 אורך קירות (מ׳): {(float(wall_px) * float(meters_per_px)):.2f}"
            )
            st.caption(
                f"(wall_px={float(wall_px):.1f}, meters_per_px={float(meters_per_px):.6f})"
            )
        else:
            st.info(
                "לא נמצאו meters_per_px / wall_length_total_px במטאדאטה — מוצג חישוב Legacy בלבד."
            )

        # שמירה ל-DB (קיים בפרויקט)
        st.markdown("---")
        if st.button("💾 שמור תוכנית לבסיס נתונים", use_container_width=True):
            try:
                # שמירה בסיסית: שם + metadata
                save_plan(
                    file_name=selected,
                    plan_name=p_name,
                    metadata=meta,
                )
                st.success("✅ נשמר בהצלחה")
            except Exception as e:
                st.error(f"❌ שגיאה בשמירה: {e}")

    # ---------------------------
    # Preview
    # ---------------------------
    with col_preview:
        st.markdown("### תצוגה")
        if proj.get("original") is not None:
            overlay = None
            try:
                # overlay פשוט אם יש masks
                overlay = create_colored_overlay(
                    proj.get("original"),
                    proj.get("concrete_mask"),
                    proj.get("blocks_mask"),
                    proj.get("flooring_mask"),
                )
            except Exception:
                overlay = None

            if overlay is not None:
                st.image(
                    _bgr_to_rgb(overlay), caption="Overlay", use_container_width=True
                )
            else:
                st.image(
                    _bgr_to_rgb(proj["original"]),
                    caption="מקור",
                    use_container_width=True,
                )
        else:
            st.info("אין תמונה להצגה")


# --------------------------------------------------------------------------------------
# TAB 2: Corrections
# --------------------------------------------------------------------------------------
def render_corrections_tab():
    _ensure_state()

    st.markdown("## 🎨 תיקונים ידניים")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected_plan = st.selectbox(
        "בחר תוכנית לתיקון:",
        list(st.session_state.projects.keys()),
        key="correction_plan_select",
    )
    if selected_plan is None or selected_plan not in st.session_state.projects:
        st.warning("בחר תוכנית כדי להמשיך.")
        return

    proj = st.session_state.projects[selected_plan]
    if proj.get("original") is None or proj.get("thick_walls") is None:
        st.warning("חסרים נתונים לתיקון (תמונה/מסכת קירות).")
        return

    st.caption("מימוש תיקון ידני בסיסי (אפשר להרחיב).")

    mode = st.radio(
        "מצב:",
        ["➕ הוסף קירות", "➖ הסר קירות"],
        horizontal=True,
        key="corr_mode",
    )

    base_img = proj["original"].copy()
    st.image(_bgr_to_rgb(base_img), use_container_width=True)

    st.info(
        "טאב זה נשאר מינימלי כדי לא להפיל את המערכת. אם תרצה, נרחיב אותו עם ציור על canvas."
    )


# --------------------------------------------------------------------------------------
# TAB 3: Dashboard (App expects this in manager.py)
# --------------------------------------------------------------------------------------
def render_dashboard_tab():
    _ensure_state()

    st.markdown("## 📊 דשבורד")

    # דוגמה: טבלת סטטיסטיקות אם קיימת
    try:
        df = load_stats_df()
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("אין עדיין נתוני סטטיסטיקה להצגה.")
    except Exception as e:
        st.info("אין עדיין נתוני סטטיסטיקה להצגה.")
        st.caption(f"(debug: {e})")

    # דוגמאות לקריאות DB (לא חובה)
    try:
        plans = get_all_plans()
        st.caption(f"מספר תוכניות בבסיס: {len(plans) if plans else 0}")
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# TAB 4: Invoices (App expects this in manager.py)
# --------------------------------------------------------------------------------------
def render_invoices_tab():
    _ensure_state()

    st.markdown("## 🧾 חשבוניות ודוחות")

    if not st.session_state.projects:
        st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        return

    selected_plan = st.selectbox(
        "בחר תוכנית:",
        list(st.session_state.projects.keys()),
        key="invoice_plan_select",
    )
    if selected_plan is None or selected_plan not in st.session_state.projects:
        st.warning("בחר תוכנית כדי להמשיך.")
        return

    proj = st.session_state.projects[selected_plan]
    meta = proj.get("metadata", {}) or {}

    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown("### 📄 דוח סטטוס")
        if st.button("צור דוח PDF", use_container_width=True, key="btn_status_pdf"):
            try:
                pdf_bytes = generate_status_pdf(
                    plan_name=meta.get("plan_name", selected_plan),
                    metadata=meta,
                )
                st.download_button(
                    label="⬇️ הורד דוח סטטוס",
                    data=pdf_bytes,
                    file_name=f"status_{selected_plan}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"❌ שגיאה ביצירת דוח: {e}")

    with c2:
        st.markdown("### 🧾 חשבונית תשלום")
        if st.button(
            "צור חשבונית PDF", use_container_width=True, key="btn_invoice_pdf"
        ):
            try:
                invoice_data = get_payment_invoice_data()
                pdf_bytes = generate_payment_invoice_pdf(
                    plan_name=meta.get("plan_name", selected_plan),
                    invoice_data=invoice_data,
                )
                st.download_button(
                    label="⬇️ הורד חשבונית",
                    data=pdf_bytes,
                    file_name=f"invoice_{selected_plan}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"❌ שגיאה ביצירת חשבונית: {e}")
