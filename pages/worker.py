"""
ConTech Pro - Worker Page v2.2
מצב דיווח שטח מתקדם עם ברירות מחדל A4+1:50, שמירת תשובות, ומצב 2 נקודות
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
import json
from datetime import datetime
import uuid
import re
import io
import traceback
from PIL import Image

from database import (
    save_progress_report,
    save_plan,
    get_plan_by_filename,
    get_plan_by_id,
    update_plan_metadata,
)
from utils import extract_segments_from_mask


try:
    from smart_measurements import SmartMeasurements
    from quantity_calculator import QuantityCalculator
    from building_elements import Wall
    from snap_engine import SimpleSnapEngine

    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    SmartMeasurements = None
    QuantityCalculator = None
    Wall = None
    SimpleSnapEngine = None

# ==========================================
# פונקציות המרה - מקור אמת יחיד
# ==========================================


def get_scale_with_fallback(proj):
    """מחזיר scale עם fallback ל-A4 + 1:50 אם לא מוגדר"""
    scale = proj.get("scale", 0)

    if scale and scale > 0:
        return scale, False  # scale תקין, לא fallback

    # Fallback: A4 + scale_denominator
    metadata = proj.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    defaults = metadata.get("defaults", {})
    scale_denominator = defaults.get("scale_denominator", 50)
    paper = defaults.get("paper", "A4")

    # A4 במטרים
    if paper == "A4":
        width_m = 0.210
        height_m = 0.297
    else:
        # ברירת מחדל A4
        width_m = 0.210
        height_m = 0.297

    # חישוב real_width_m
    real_width_m = width_m * scale_denominator

    # חישוב scale_px_per_m
    original_img = proj.get("original")
    if original_img is not None:
        image_width_px = original_img.shape[1]
        scale_px_per_m = image_width_px / real_width_m
        return scale_px_per_m, True  # fallback

    # אם אין תמונה, החזר ברירת מחדל גסה
    return 200.0, True


def px_to_m(px_value, scale_factor, scale):
    """המרת פיקסלים למטרים"""
    if scale <= 0:
        return 0.0
    return px_value / scale_factor / scale


def px2_to_m2(px2_value, scale_factor, scale):
    """המרת פיקסלים בריבוע למ"ר"""
    if scale <= 0:
        return 0.0
    return px2_value / ((scale * scale_factor) ** 2)


# ==========================================
# פונקציות עזר
# ==========================================


def get_corrected_walls(selected_plan, proj):
    """מחזיר את מסכת הקירות המתוקנת"""
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


def build_snap_points(segments):
    """
    בונה רשימת נקודות להצמדה מתוך segments

    Args:
        segments: רשימת קטעים

    Returns:
        רשימת נקודות: [(x1, y1), (x2, y2), ...]
    """
    points = set()  # set למניעת כפילויות

    for seg in segments:
        # קצוות הקטע
        points.add(seg["start"])
        points.add(seg["end"])

        # אמצע הקטע (אופציונלי)
        mid_x = (seg["start"][0] + seg["end"][0]) // 2
        mid_y = (seg["start"][1] + seg["end"][1]) // 2
        points.add((mid_x, mid_y))

    return list(points)


def generate_uid():
    """יוצר uuid קצר וייחודי"""
    return str(uuid.uuid4())[:8]


def compute_line_length_px(obj):
    """מחשב אורך קו בפיקסלים - לא תלוי בעובי"""
    if obj.get("type") == "line":
        x1 = obj.get("x1", 0)
        y1 = obj.get("y1", 0)
        x2 = obj.get("x2", 0)
        y2 = obj.get("y2", 0)
        dx = x2 - x1
        dy = y2 - y1
        return np.sqrt(dx * dx + dy * dy)
    elif obj.get("type") == "path":
        path = obj.get("path", [])
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if len(p1) >= 3 and len(p2) >= 3:
                dx = p2[1] - p1[1]
                dy = p2[2] - p1[2]
                total += np.sqrt(dx * dx + dy * dy)
        return total
    return 0.0


def compute_rect_area_px(obj):
    """מחשב שטח ריבוע בפיקסלים"""
    if obj.get("type") == "rect":
        w = obj.get("width", 0)
        h = obj.get("height", 0)
        return abs(w * h)
    return 0.0


def create_single_object_mask(obj, canvas_width, canvas_height):
    """יוצר מסכה לאובייקט בודד - רק לצורך auto-enrichment"""
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    obj_type = obj.get("type", "")

    if obj_type == "line":
        x1 = int(obj.get("x1", 0))
        y1 = int(obj.get("y1", 0))
        x2 = int(obj.get("x2", 0))
        y2 = int(obj.get("y2", 0))
        stroke_width = int(obj.get("strokeWidth", 5))
        cv2.line(mask, (x1, y1), (x2, y2), 255, stroke_width)

    elif obj_type == "rect":
        left = int(obj.get("left", 0))
        top = int(obj.get("top", 0))
        width = int(obj.get("width", 0))
        height = int(obj.get("height", 0))
        cv2.rectangle(mask, (left, top), (left + width, top + height), 255, -1)

    elif obj_type == "path":
        path = obj.get("path", [])
        if len(path) > 1:
            points = []
            for p in path:
                if len(p) >= 3:
                    x = int(p[1])
                    y = int(p[2])
                    points.append((x, y))
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(
                        mask,
                        points[i],
                        points[i + 1],
                        255,
                        int(obj.get("strokeWidth", 5)),
                    )

    return mask


def auto_enrich_item(item, mask, corrected_walls, proj):
    """Auto-enrichment: מציע is_wall וחומר לפי overlap - רק לצורך הצעות, לא מדידה"""
    if mask is None:
        return item

    item_pixels = np.count_nonzero(mask)
    if item_pixels == 0:
        return item

    # בדיקת overlap עם קירות (אם קיים)
    if corrected_walls is not None and corrected_walls.shape == mask.shape:
        intersection = cv2.bitwise_and(mask, corrected_walls)
        overlap_pixels = np.count_nonzero(intersection)

        overlap_ratio = overlap_pixels / item_pixels
        item["wall_overlap_ratio"] = round(overlap_ratio, 2)
        item["is_wall_suggested"] = overlap_ratio > 0.5

    # ניסיון לזהות חומר (safe - לא קורס אם חסר)
    concrete_mask = proj.get("concrete_mask")
    blocks_mask = proj.get("blocks_mask")

    suggested_material = None

    if concrete_mask is not None and concrete_mask.shape == mask.shape:
        try:
            conc_overlap = np.count_nonzero(cv2.bitwise_and(mask, concrete_mask))
            if conc_overlap > item_pixels * 0.5:
                suggested_material = "בטון"
        except:
            pass

    if blocks_mask is not None and blocks_mask.shape == mask.shape:
        try:
            block_overlap = np.count_nonzero(cv2.bitwise_and(mask, blocks_mask))
            if block_overlap > item_pixels * 0.5:
                suggested_material = "בלוקים"
        except:
            pass

    item["material_suggested"] = suggested_material

    return item


def create_annotated_preview(rgb_image, items_data, selected_uid=None):
    """יוצר תמונת preview עם מספרים ומדידות והדגשת פריט נבחר"""
    annotated = rgb_image.copy()

    for idx, item in enumerate(items_data, 1):
        cx = int(item.get("center_x", 0))
        cy = int(item.get("center_y", 0))
        measurement = item.get("measurement", 0.0)
        unit = item.get("unit", "m")
        uid = item.get("uid", "")

        text = f"{idx}: {measurement:.2f}{unit}"

        # הדגשה אם נבחר
        is_selected = selected_uid and uid == selected_uid
        text_color = (255, 0, 0) if is_selected else (0, 0, 255)
        box_color = (255, 0, 0) if is_selected else (0, 0, 0)
        box_thickness = 3 if is_selected else 2

        # רקע לטקסט
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if is_selected else 0.6
        thickness = 3 if is_selected else 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (cx - 5, cy - text_h - 10),
            (cx + text_w + 5, cy + 5),
            (255, 255, 255),
            -1,
        )

        cv2.rectangle(
            annotated,
            (cx - 5, cy - text_h - 10),
            (cx + text_w + 5, cy + 5),
            box_color,
            box_thickness,
        )

        # טקסט
        cv2.putText(annotated, text, (cx, cy), font, font_scale, text_color, thickness)

    return annotated


def load_form_schema(plan_name, proj):
    """טוען את ה-schema של הטופס מהמטא-דאטה"""
    metadata = proj.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    schema = metadata.get("worker_form_schema", [])

    # Default schema אם אין
    if not schema:
        schema = [
            {
                "type": "checkbox",
                "label": "האם זה קיר?",
                "key": "is_wall",
                "default": True,
            },
            {
                "type": "checkbox",
                "label": "האם זה קיר גבס?",
                "key": "is_gypsum",
                "default": False,
            },
            {
                "type": "select",
                "label": "חומר:",
                "key": "material",
                "options": ["בטון", "בלוקים", "גבס", "אחר"],
                "default": "בטון",
            },
            {
                "type": "number",
                "label": "גובה (מ'):",
                "key": "height",
                "default": 2.60,
                "step": 0.1,
            },
        ]

    return schema


def save_form_schema(plan_name, proj, schema):
    """שומר את ה-schema בחזרה למטא-דאטה"""
    rec = get_plan_by_filename(plan_name)
    if not rec:
        return False

    plan_id = rec["id"]

    metadata = proj.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    metadata["worker_form_schema"] = schema
    metadata_json_str = json.dumps(metadata, ensure_ascii=False)

    update_plan_metadata(plan_id, metadata_json_str)
    proj["metadata"] = metadata

    return True


def validate_schema_field(field):
    """ולידציה לשדה schema"""
    errors = []

    # בדיקת key
    key = field.get("key", "").strip()
    if not key:
        errors.append("Key לא יכול להיות ריק")
    elif not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
        errors.append(f"Key '{key}' לא תקין (חייב להתחיל באות/_, רק אותיות/מספרים/_)")

    # בדיקת label
    if not field.get("label", "").strip():
        errors.append("Label לא יכול להיות ריק")

    # בדיקת type
    if field.get("type") not in ["checkbox", "select", "number", "text"]:
        errors.append("Type חייב להיות checkbox/select/number/text")

    # בדיקת options ל-select
    if field.get("type") == "select":
        options = field.get("options", [])
        if not options or len(options) == 0:
            errors.append("Select חייב לכלול לפחות אפשרות אחת")

    return errors


def render_schema_editor(plan_name, proj):
    """מסך הגדרת schema לטופס עם ולידציה"""
    st.markdown("### ⚙️ הגדרת טופס (למנהל)")
    st.caption("הגדר שדות ושאלות שיופיעו לכל פריט")

    # טעינת schema מה-DB או session_state
    schema_key = f"schema_editing_{plan_name}"

    if schema_key not in st.session_state:
        st.session_state[schema_key] = load_form_schema(plan_name, proj)

    schema = st.session_state[schema_key]

    # UI לעריכת schema
    new_schema = []
    fields_to_delete = []
    all_keys = []
    validation_errors = []

    st.markdown("---")

    for idx, field in enumerate(schema):
        # Card פשוט
        st.markdown(f"#### שדה #{idx+1}: {field.get('label', 'ללא שם')}")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            field_type = st.selectbox(
                "סוג שדה:",
                ["checkbox", "select", "number", "text"],
                index=["checkbox", "select", "number", "text"].index(
                    field.get("type", "text")
                ),
                key=f"schema_type_{idx}",
            )

            field_label = st.text_input(
                "תווית:",
                value=field.get("label", ""),
                key=f"schema_label_{idx}",
            )

        with col2:
            field_key = st.text_input(
                "Key (משתנה):",
                value=field.get("key", ""),
                key=f"schema_key_{idx}",
            )

            if field_type == "checkbox":
                field_default = st.checkbox(
                    "ברירת מחדל:",
                    value=field.get("default", False),
                    key=f"schema_default_{idx}",
                )
            elif field_type == "number":
                field_default = st.number_input(
                    "ברירת מחדל:",
                    value=float(field.get("default", 0)),
                    key=f"schema_default_{idx}",
                )
            elif field_type == "select":
                options_str = st.text_input(
                    "אפשרויות (מופרד בפסיק):",
                    value=",".join(field.get("options", [])),
                    key=f"schema_options_{idx}",
                )
                field_options = [o.strip() for o in options_str.split(",") if o.strip()]
                if field_options:
                    current_default = field.get("default", "")
                    if current_default not in field_options:
                        current_default = field_options[0]
                    field_default = st.selectbox(
                        "ברירת מחדל:",
                        field_options,
                        index=(
                            field_options.index(current_default)
                            if current_default in field_options
                            else 0
                        ),
                        key=f"schema_default_select_{idx}",
                    )
                else:
                    field_default = ""
            else:
                field_default = st.text_input(
                    "ברירת מחדל:",
                    value=field.get("default", ""),
                    key=f"schema_default_text_{idx}",
                )

        with col3:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("🗑️", key=f"delete_field_{idx}", help="מחק שדה"):
                fields_to_delete.append(idx)

        # בניית השדה החדש
        new_field = {
            "type": field_type,
            "label": field_label,
            "key": field_key,
            "default": field_default,
        }

        if field_type == "select" and field_options:
            new_field["options"] = field_options
        elif field_type == "number":
            new_field["step"] = field.get("step", 0.1)

        # ולידציה
        field_errors = validate_schema_field(new_field)
        if field_errors:
            validation_errors.extend([f"שדה #{idx+1}: {err}" for err in field_errors])

        # בדיקת כפילויות key
        if field_key.strip():
            if field_key in all_keys:
                validation_errors.append(f"שדה #{idx+1}: Key '{field_key}' כבר קיים")
            all_keys.append(field_key)

        new_schema.append(new_field)

        st.markdown("---")

    # מחיקת שדות
    if fields_to_delete:
        new_schema = [f for i, f in enumerate(new_schema) if i not in fields_to_delete]
        st.session_state[schema_key] = new_schema
        st.rerun()

    # עדכון session_state
    st.session_state[schema_key] = new_schema

    # הצגת שגיאות ולידציה
    if validation_errors:
        st.error("❌ שגיאות ולידציה:")
        for err in validation_errors:
            st.write(f"- {err}")

    # כפתורי פעולה
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("➕ הוסף שדה", use_container_width=True):
            st.session_state[schema_key].append(
                {
                    "type": "text",
                    "label": "שדה חדש",
                    "key": f"new_field_{len(st.session_state[schema_key])+1}",
                    "default": "",
                }
            )
            st.rerun()

    with col_btn2:
        save_disabled = len(validation_errors) > 0
        if st.button(
            "💾 שמור Schema",
            type="primary",
            use_container_width=True,
            disabled=save_disabled,
        ):
            if save_form_schema(plan_name, proj, st.session_state[schema_key]):
                st.success("✅ Schema נשמר!")
                del st.session_state[schema_key]
                st.rerun()
            else:
                st.error("❌ שגיאה בשמירה")

    with col_btn3:
        if st.button("🔄 איפוס", use_container_width=True):
            if schema_key in st.session_state:
                del st.session_state[schema_key]
            st.rerun()


def render_item_questions(uid, item, schema, answers_key):
    """מציג שאלות לפריט לפי ה-schema עם שמירת תשובות"""
    # אתחול answers אם לא קיים
    if answers_key not in st.session_state:
        st.session_state[answers_key] = {}

    if uid not in st.session_state[answers_key]:
        st.session_state[answers_key][uid] = {}

    answers = st.session_state[answers_key][uid]

    for field in schema:
        field_type = field.get("type", "text")
        field_label = field.get("label", "שדה")
        field_key = field.get("key", "field")
        field_default = field.get("default", None)

        # Auto-enrichment suggestions
        if field_key == "is_wall" and "is_wall_suggested" in item:
            if item["is_wall_suggested"]:
                field_label += (
                    f" (מומלץ: כן - {item.get('wall_overlap_ratio', 0)*100:.0f}%)"
                )

        if field_key == "material" and "material_suggested" in item:
            if item["material_suggested"]:
                field_label += f" (מומלץ: {item['material_suggested']})"

        # קבלת ערך נוכחי
        current_value = answers.get(field_key, field_default)

        if field_type == "checkbox":
            if current_value is None:
                current_value = False
            new_value = st.checkbox(
                field_label,
                value=bool(current_value),
                key=f"{field_key}_{uid}",
            )
            answers[field_key] = new_value
            item[field_key] = new_value

        elif field_type == "select":
            options = field.get("options", [""])
            if current_value not in options and options:
                current_value = options[0]

            new_value = st.selectbox(
                field_label,
                options,
                index=options.index(current_value) if current_value in options else 0,
                key=f"{field_key}_{uid}",
            )
            answers[field_key] = new_value
            item[field_key] = new_value

        elif field_type == "number":
            if current_value is None:
                current_value = 0
            new_value = st.number_input(
                field_label,
                value=float(current_value),
                step=field.get("step", 0.1),
                key=f"{field_key}_{uid}",
            )
            answers[field_key] = new_value
            item[field_key] = new_value

        elif field_type == "text":
            if current_value is None:
                current_value = ""
            new_value = st.text_input(
                field_label,
                value=str(current_value),
                key=f"{field_key}_{uid}",
            )
            answers[field_key] = new_value
            item[field_key] = new_value


# ==========================================
# Worker Page Main
# ==========================================


def render_worker_page():
    """
    מצב דיווח שטח - גרסה 2.3 UX משופר
    חלוקה ל-3 שלבים ברורים: הכנה → סימון → שמירה
    """
    st.title("👷 דיווח ביצוע מהשטח")
    st.caption("מערכת דיווח מקצועית | גרסה 2.3")

    # אתחול projects אם לא קיים
    if "projects" not in st.session_state:
        st.session_state.projects = {}

    if not st.session_state.projects:
        st.error("📂 אין תוכניות זמינות במערכת")
        st.info("💡 **מה עושים עכשיו?** לך למצב 'מנהל פרויקט' והעלה קובץ PDF של תוכנית")
        st.caption("אחרי ההעלאה, תוכל לחזור לכאן ולהתחיל דיווח")
        return

    # ==========================================
    # 📋 שלב 1 מתוך 3: הכנת עבודה
    # ==========================================
    st.divider()
    st.info(
        "🔹 **שלב 1 מתוך 3:** בחירת תוכנית והגדרות ראשוניות | **הבא:** סימון ביצוע ↓"
    )

    st.header("📋 שלב 1: הכנת עבודה")

    # === בחירת פרויקט ===
    st.markdown("### 1️⃣ בחר תוכנית")
    plan_name = st.selectbox(
        "איזו תוכנית אתה עובד עליה היום?",
        list(st.session_state.projects.keys()),
        help="בחר את התוכנית שאתה רוצה לדווח עליה",
    )
    proj = st.session_state.projects[plan_name]
    # ========== METADATA AWARENESS - תוספת חדשה ==========
    with st.expander("🔒 הגדרות Metadata", expanded=False):
        if proj.get("_from_metadata"):
            st.success("✅ תוכנית נטענה מ-Metadata - דיוק גבוה!")
            md = proj.get("_metadata_object", {})
            # אם זה אובייקט (לא dict) – ננסה לספור קירות בצורה בטוחה
            walls_count = 0
            try:
                if hasattr(md, "walls"):
                    walls_count = len(md.walls)
                elif isinstance(md, dict):
                    walls_count = len(md.get("walls", []))
            except:
                walls_count = 0

            st.caption(f"📦 {walls_count} קירות מוגדרים")

            show_metadata_overlay = st.checkbox(
                "🎨 הצג קירות כ-overlay",
                value=True,
                help="מציג את הקירות שנטענו מ-metadata כקווים על השרטוט",
                key=f"show_metadata_overlay_{plan_name}",
            )
            st.session_state[f"show_metadata_overlay_{plan_name}"] = (
                show_metadata_overlay
            )
        else:
            st.info("ℹ️ תוכנית לא נטענה מ-metadata - זיהוי OpenCV רגיל")
            st.session_state[f"show_metadata_overlay_{plan_name}"] = False
    # ========== סוף תוספת ==========

    # === בדיקת scale עם fallback ===
    scale_value, is_fallback = get_scale_with_fallback(proj)
    proj["scale"] = scale_value  # עדכון לשימוש

    if is_fallback:
        st.warning(
            "⚠️ לא נמצאה סקלה מדויקת - המערכת משתמשת בברירת מחדל (דף A4, קנה 1:50)"
        )
        st.caption("💡 לתוצאות מדויקות יותר, בקש ממנהל הפרויקט להגדיר סקלה")
    else:
        st.success("✅ הסקלה מוגדרת ומדויקת")

    # אתחול report_objects (מקור אמת יחיד)
    report_key = f"report_objects_{plan_name}"
    if report_key not in st.session_state:
        st.session_state[report_key] = []

    # אתחול answers
    answers_key = f"item_answers_{plan_name}"
    if answers_key not in st.session_state:
        st.session_state[answers_key] = {}

    # ===== PHASE 1: אתחול Smart Measurements + Snap =====
    if PHASE1_AVAILABLE:
        smart_key = f"smart_measurements_{plan_name}"
        snap_key = f"snap_engine_{plan_name}"

        # אתחול רק אם לא קיים
        if smart_key not in st.session_state or snap_key not in st.session_state:
            try:
                corrected_walls_temp = get_corrected_walls(plan_name, proj)
                segments = extract_segments_from_mask(corrected_walls_temp, scale_value)

                # SmartMeasurements (אם המודול קיים)
                st.session_state[smart_key] = SmartMeasurements(
                    detected_segments=segments, scale=scale_value
                )

                # בניית snap points (תומך בשני פורמטים של segments)
                snap_points = []
                for seg in segments:
                    if isinstance(seg, dict):
                        if "start" in seg and "end" in seg:
                            x1, y1 = seg["start"]
                            x2, y2 = seg["end"]
                        elif all(k in seg for k in ("x1", "y1", "x2", "y2")):
                            x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
                        else:
                            continue

                        snap_points.append((int(x1), int(y1)))
                        snap_points.append((int(x2), int(y2)))

                st.session_state[snap_key] = SimpleSnapEngine(
                    snap_points=snap_points, tolerance_px=15
                )

            except Exception as e:
                st.warning(f"⚠️ Phase 1 init failed: {str(e)}")
                st.session_state[smart_key] = None
                st.session_state[snap_key] = None

    # === Schema Editor (Expander למנהל) ===
    with st.expander("🔧 הגדרות מתקדמות (למנהלים)", expanded=False):
        st.caption(
            "⚠️ אזור זה מיועד למנהלי פרויקט - עובדי שטח רגילים לא צריכים לגעת כאן"
        )
        render_schema_editor(plan_name, proj)

    # === תאריך ומשמרת ===
    st.markdown("### 2️⃣ פרטי העבודה")
    col_date, col_shift = st.columns(2)
    with col_date:
        report_date = st.date_input(
            "תאריך הדיווח:",
            value=datetime.now().date(),
            help="באיזה תאריך בוצעה העבודה?",
        )
    with col_shift:
        shift = st.selectbox(
            "משמרת:", ["בוקר", "צהריים", "לילה"], help="באיזו משמרת עבדת?"
        )

    st.success("✅ שלב 1 הושלם | עבור למטה לשלב 2 ↓")

    # ==========================================
    # 🛠️ שלב 2 מתוך 3: סימון ביצוע
    # ==========================================
    st.divider()
    st.info(
        "🔹 **שלב 2 מתוך 3:** סמן את העבודה שביצעת על התוכנית | **הבא:** בדיקה ושמירה ↓"
    )

    st.header("🛠️ שלב 2: סימון ביצוע")

    # === בחירת מצב עבודה ===
    st.markdown("### 1️⃣ מה ביצעת?")
    report_type = st.radio(
        "בחר את סוג העבודה:",
        ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"],
        horizontal=True,
        help="בחר את סוג העבודה שביצעת היום",
    )

    # === בחירת מצב ציור ===
    st.markdown("### 2️⃣ כיצד לסמן?")
    col_mode1, col_mode2 = st.columns([4, 1])

    with col_mode1:
        drawing_mode_display = st.radio(
            "בחר כלי ציור:",
            ["✏️ קו ישר", "🖊️ ציור חופשי", "▭ ריבוע"],
            horizontal=True,
            help="בחר את הכלי המתאים",
        )

    with col_mode2:
        with st.expander("⚙️"):
            two_point_mode = st.checkbox("מצב 2 נקודות", value=False)
            st.caption("למתקדמים")

    if two_point_mode:
        drawing_mode = "point"
        st.info("📍 **מצב 2 נקודות:** לחץ על נקודה ראשונה, אחר כך על נקודה שנייה")
    else:
        if "ישר" in drawing_mode_display:
            drawing_mode = "line"
            st.info("💡 לחץ והחזק, גרור לכיוון הרצוי, ושחרר ליצירת קו")
        elif "חופשי" in drawing_mode_display:
            drawing_mode = "freedraw"
            st.info("💡 צייר בחופשיות על התוכנית")
        else:
            drawing_mode = "rect"
            st.info("💡 צייר ריבוע סביב האזור")

    st.markdown("---")
    st.markdown("### 3️⃣ סמן על התוכנית")

    # === הכנת תמונה ===
    corrected_walls = get_corrected_walls(plan_name, proj)
    rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # === DEBUG - הוסף כאן ===
    st.write("🔍 Debug:")
    st.write(f"proj keys: {list(proj.keys())}")

    if "thick_walls" in proj:
        st.write(f"✅ thick_walls: {proj['thick_walls'].shape}")
        st.write(f"יש פיקסלים: {np.any(proj['thick_walls'] > 0)}")
    else:
        st.error("❌ thick_walls לא קיים!")

    if corrected_walls is not None:
        st.write(f"✅ corrected_walls: {corrected_walls.shape}")
        st.write(f"יש פיקסלים: {np.any(corrected_walls > 0)}")
    else:
        st.error("❌ corrected_walls הוא None!")
    st.write("---")
    # === סוף DEBUG ==
    MAX_W = 700
    MAX_H = 650  # אפשר לשנות לפי הטעם
    scale_w = MAX_W / w if w > MAX_W else 1.0
    scale_h = MAX_H / h if h > MAX_H else 1.0
    scale_factor = min(scale_w, scale_h, 1.0)

    img_resized = Image.fromarray(rgb).resize(
        (int(w * scale_factor), int(h * scale_factor))
    )
    # ========== METADATA OVERLAY - תוספת חדשה ==========

    img_resized_with_overlay = img_resized

    try:
        if proj.get("_from_metadata") and st.session_state.get(
            f"show_metadata_overlay_{plan_name}", False
        ):
            # נצייר על תמונת הרקע (RGB) ואז נהפוך ל-PIL
            bg_rgb = np.array(img_resized.convert("RGB")).copy()

            md = proj.get("_metadata_object")
            # scale לתצוגה (תואם ל-resize שעשית)
            display_w = int(w * scale_factor)
            display_h = int(h * scale_factor)

            # פונקציה בטוחה להבאת walls
            walls = None
            if hasattr(md, "walls"):
                walls = md.walls
            elif isinstance(md, dict):
                walls = md.get("walls")

            if walls:
                for wall in walls:
                    # תומך בשני פורמטים:
                    # 1) wall.points (רשימת (x,y))
                    # 2) dict עם key 'points'
                    pts = None
                    if hasattr(wall, "points"):
                        pts = wall.points
                    elif isinstance(wall, dict):
                        pts = wall.get("points")

                    if not pts:
                        continue

                    scaled_points = [
                        (int(p[0] * scale_factor), int(p[1] * scale_factor))
                        for p in pts
                    ]
                    points_array = np.array(scaled_points, dtype=np.int32)

                    # שים לב: OpenCV מצייר BGR, אבל bg_rgb הוא RGB.
                    # לא נוגעים בצבעים “מדויקים” כדי לא להסתבך — העיקר שיופיע.
                    cv2.polylines(
                        bg_rgb, [points_array], False, (0, 0, 255), thickness=2
                    )

                img_resized_with_overlay = Image.fromarray(bg_rgb)
    except Exception:
        # אם משהו לא תואם – לא שוברים כלום, פשוט בלי overlay
        img_resized_with_overlay = img_resized
    # ========== סוף תוספת ==========

    # === כלי כיול סקלה ===
    # כיון שה-scale הנוכחי הוא FALLBACK (A4+1:50) ולא נכון לתוכנית,
    # נתת לעובד אפשרות לכיול ישר: צייר קו, כתוב אורך אמיתי, scale מחשב
    with st.expander("📏 כיול סקלה (חשוב לדיוק!)", expanded=is_fallback):
        st.caption(
            "צייר קו על קיר שיש לו אורך ידווע, כתוב את האורך האמיתי שלו → ה-scale יחשב אוטומטית"
        )

        cal_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_color="#FF00FF",
            stroke_width=3,
            background_image=img_resized_with_overlay,
            height=int(h * scale_factor),
            width=int(w * scale_factor),
            drawing_mode="line",
            key=f"cal_canvas_{plan_name}_{proj['scale']}",
            update_streamlit=True,
        )

        # חישוב אורך הקו האחרון שציירת
        cal_px = 0.0
        if cal_canvas.json_data and cal_canvas.json_data.get("objects"):
            cal_lines = [
                o for o in cal_canvas.json_data["objects"] if o.get("type") == "line"
            ]
            if cal_lines:
                cal_px = compute_line_length_px(cal_lines[-1])

        if cal_px > 0:
            st.info(f"📐 אורך הקו שציירת: {cal_px:.0f} פיקסלים (על הקנבס)")

            col_real, col_btn = st.columns([2, 1])
            with col_real:
                real_length_m = st.number_input(
                    "אורך האמיתי של הקו הזה (מטר):",
                    value=1.0,
                    min_value=0.1,
                    max_value=100.0,
                    step=0.5,
                    key=f"cal_real_length_{plan_name}",
                )
            with col_btn:
                st.write("")
                if st.button("✅ תקן סקלה", type="primary", use_container_width=True):
                    # cal_px הוא על הקנבס הקטן → חזרה למקורי
                    cal_px_original = cal_px / scale_factor
                    new_scale = cal_px_original / real_length_m
                    proj["scale"] = new_scale

                    verify = px_to_m(cal_px, scale_factor, new_scale)
                    st.success(
                        f"✅ סקלה תיקנה! {new_scale:.1f} px/m (וריפיקציה: {verify:.2f}m)"
                    )
                    st.rerun()
        else:
            st.info("👆 צייר קו ישר על קיר שיש לו אורך ידווע בתוכנית")

        st.caption(
            f"Scale הנוכחי: {proj['scale']:.1f} px/m {'⚠️ FALLBACK' if is_fallback else '✅ הוגדר ידנית'}"
        )

    # === הגדרות ציור ===
    if "קירות" in report_type:
        fill = "rgba(0,0,0,0)"
        stroke = "#00FF00"
        stroke_width = 6
    else:
        fill = "rgba(255,255,0,0.3)"
        stroke = "#FFFF00"
        stroke_width = 20

    # === Layout: שתי עמודות ===
    col_left, col_right = st.columns([1.5, 1], gap="medium")

    with col_left:
        st.markdown("### 🎨 אזור ציור")

        overlay_on = st.session_state.get(f"show_metadata_overlay_{plan_name}", False)

        try:
            # ודא PIL RGB
            if img_resized_with_overlay is None:
                raise ValueError("img_resized_with_overlay is None")

            img_rgb = img_resized_with_overlay.convert("RGB")

            # טריק יציב: להפוך לבייטים ולפתוח מחדש (מונע מצבים שה-PIL לא נטען כמו שצריך)
            buf = io.BytesIO()
            img_rgb.save(buf, format="PNG")
            bg_img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            bg_img.load()

            canvas = st_canvas(
                fill_color=fill,
                stroke_color=stroke,
                stroke_width=stroke_width if not two_point_mode else 1,
                background_image=bg_img,
                height=int(h * scale_factor),
                width=int(w * scale_factor),
                drawing_mode=drawing_mode,
                point_display_radius=5 if two_point_mode else 0,
                key=f"canvas_{plan_name}_{w}x{h}_sf{scale_factor:.4f}_ov{int(overlay_on)}_{report_type}_{drawing_mode}_{two_point_mode}_bgfix1",
                update_streamlit=True,
            )

        except Exception:
            # לא דיבאג חופר – רק הודעת כשל + traceback כדי שלא "ייעלם" בשקט
            st.error("❌ לא הצלחתי להכין רקע לקנבס. בדוק תמונה/חיתוך/scale.")
            st.code(traceback.format_exc())
            canvas = None

        # === הוסף כאן ===
        # Snap Indicator (אינדיקציה ויזואלית)
        if PHASE1_AVAILABLE and f"snap_engine_{plan_name}" in st.session_state:
            snap_engine = st.session_state[f"snap_engine_{plan_name}"]

            # בדיקה אם יש אובייקטים בקנבס
            if canvas.json_data and canvas.json_data.get("objects"):
                last_obj = canvas.json_data["objects"][-1]

                # אם זה קו - בדוק snap בנקודות הקצה
                if last_obj.get("type") == "line":
                    x1, y1 = last_obj.get("x1", 0), last_obj.get("y1", 0)
                    x2, y2 = last_obj.get("x2", 0), last_obj.get("y2", 0)

                    # בדיקת snap
                    snap1 = snap_engine.find_snap(int(x1), int(y1))
                    snap2 = snap_engine.find_snap(int(x2), int(y2))

                    # תצוגה
                    if snap1 or snap2:
                        snapped_text = []
                        if snap1:
                            snapped_text.append(f"התחלה: ✅ נצמד ({snap1[2]:.0f}px)")
                        if snap2:
                            snapped_text.append(f"סיום: ✅ נצמד ({snap2[2]:.0f}px)")

                        st.success("🎯 " + " | ".join(snapped_text))
        # כפתורי ניהול
        if two_point_mode:
            # מצב 2 נקודות - כפתורים מיוחדים
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 המר לקווים", use_container_width=True):
                    if canvas.json_data and canvas.json_data.get("objects"):
                        points = [
                            obj
                            for obj in canvas.json_data["objects"]
                            if obj.get("type") in ["circle", "rect"]
                            and obj.get("width", 0) < 20
                        ]

                        if len(points) >= 2:
                            for i in range(0, len(points) - 1, 2):
                                p1 = points[i]
                                p2 = points[i + 1]

                                x1 = p1.get("left", 0) + p1.get("width", 0) / 2
                                y1 = p1.get("top", 0) + p1.get("height", 0) / 2
                                x2 = p2.get("left", 0) + p2.get("width", 0) / 2
                                y2 = p2.get("top", 0) + p2.get("height", 0) / 2

                                line_obj = {
                                    "type": "line",
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "stroke": stroke,
                                    "strokeWidth": stroke_width,
                                    "uid": generate_uid(),
                                }
                                st.session_state[report_key].append(line_obj)

                            st.success(f"✅ נוצרו {len(points)//2} קווים!")
                            st.rerun()
                        else:
                            st.warning("יש צורך ב-2 נקודות לפחות")

            with col_btn2:
                if st.button("🗑️ נקה נקודות", use_container_width=True):
                    st.rerun()
        else:
            # מצב רגיל
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🗑️ נקה הכל", use_container_width=True):
                    st.session_state[report_key] = []
                    st.session_state[answers_key] = {}
                    st.rerun()

            with col_btn2:
                if st.button("↩️ בטל אחרון", use_container_width=True):
                    if st.session_state[report_key]:
                        removed = st.session_state[report_key].pop()
                        # מחק גם תשובות
                        removed_uid = removed.get("uid")
                        if removed_uid in st.session_state[answers_key]:
                            del st.session_state[answers_key][removed_uid]
                        st.rerun()

    with col_right:
        st.markdown("### 📋 פרטי פריטים")

        items_data = []

        # === בניית report_objects מקנבס (מקור אמת יחיד) ===
        if canvas.json_data and canvas.json_data.get("objects"):
            canvas_objects = canvas.json_data["objects"]

            if two_point_mode:
                # סינון נקודות
                canvas_objects = [
                    obj
                    for obj in canvas_objects
                    if not (
                        obj.get("type") in ["circle", "rect"]
                        and obj.get("width", 0) < 20
                    )
                ]

            # סנכרון
            current_count = len(st.session_state[report_key])
            canvas_count = len(canvas_objects)

            if canvas_count > current_count:
                for i in range(current_count, canvas_count):
                    new_obj = canvas_objects[i].copy()
                    new_obj["uid"] = generate_uid()
                    st.session_state[report_key].append(new_obj)

        objects = st.session_state[report_key]

        if len(objects) == 0:
            st.info("🖌️ התחל לצייר על התוכנית")
        else:
            # === חישוב מדידות ===
            items_data = []  # ← תיקון: אתחול כאן!
            total_length = 0.0
            total_area = 0.0

            for obj in objects:
                uid = obj.get("uid", generate_uid())

                # חישוב מדידה
                if "קירות" in report_type:
                    # מדידת אורך רק לפי compute_line_length_px - לא תלוי בעובי
                    length_px = compute_line_length_px(obj)
                    if length_px > 0:
                        length_m = px_to_m(length_px, scale_factor, proj["scale"])
                        total_length += length_m

                        # מרכז
                        if obj.get("type") == "line":
                            cx = int((obj.get("x1", 0) + obj.get("x2", 0)) / 2)
                            cy = int((obj.get("y1", 0) + obj.get("y2", 0)) / 2)
                        else:
                            cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                            cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2

                        item = {
                            "uid": uid,
                            "type": obj.get("type", "unknown"),
                            "measurement": length_m,
                            "unit": "m",
                            "center_x": cx,
                            "center_y": cy,
                        }

                        # Auto-enrichment רק להצעות
                        mask = create_single_object_mask(
                            obj, int(w * scale_factor), int(h * scale_factor)
                        )
                        walls_resized = cv2.resize(
                            corrected_walls,
                            (int(w * scale_factor), int(h * scale_factor)),
                        )
                        # ========== CONFIDENCE (SNAP-LIKE) - תוספת חדשה ==========
                        try:
                            # רק לקירות
                            drawn_px = int(np.count_nonzero(mask))
                            if drawn_px > 0 and walls_resized is not None:
                                # tolerance במטרים -> לפיקסלים לפי scale
                                snap_tolerance_m = (
                                    0.20  # אפשר לשנות בהמשך, כרגע קבוע כדי לא לשבור UI
                                )
                                snap_radius_px = max(
                                    1,
                                    int(
                                        snap_tolerance_m * proj["scale"] * scale_factor
                                    ),
                                )

                                kernel = np.ones(
                                    (snap_radius_px * 2 + 1, snap_radius_px * 2 + 1),
                                    np.uint8,
                                )
                                snap_mask = cv2.dilate(
                                    (walls_resized > 0).astype(np.uint8) * 255, kernel
                                )

                                snapped_drawn = np.logical_and(mask > 0, snap_mask > 0)
                                intersection = np.logical_and(
                                    snapped_drawn, walls_resized > 0
                                )

                                matched_px = int(np.count_nonzero(intersection))
                                confidence = (matched_px / drawn_px) * 100.0

                                st.session_state[f"last_confidence_{plan_name}"] = (
                                    confidence
                                )
                        except Exception:
                            pass
                        # ========== סוף תוספת ==========

                        item = auto_enrich_item(item, mask, walls_resized, proj)

                        items_data.append(item)
                else:
                    # ריצוף
                    if obj.get("type") == "rect":
                        area_px = compute_rect_area_px(obj)
                        if area_px > 0:
                            area_m2 = px2_to_m2(area_px, scale_factor, proj["scale"])
                            total_area += area_m2

                            cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                            cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2

                            item = {
                                "uid": uid,
                                "type": "rect",
                                "measurement": area_m2,
                                "unit": "m²",
                                "center_x": cx,
                                "center_y": cy,
                            }

                            # Auto-enrichment
                            mask = create_single_object_mask(
                                obj, int(w * scale_factor), int(h * scale_factor)
                            )
                            item = auto_enrich_item(item, mask, None, proj)

                            items_data.append(item)
                    else:
                        mask = create_single_object_mask(
                            obj, int(w * scale_factor), int(h * scale_factor)
                        )
                        pixels = np.count_nonzero(mask)
                        if pixels > 0:
                            area_m2 = px2_to_m2(pixels, scale_factor, proj["scale"])
                            total_area += area_m2

                            cy_arr, cx_arr = np.where(mask > 0)
                            if len(cx_arr) > 0:
                                cx = int(np.mean(cx_arr))
                                cy = int(np.mean(cy_arr))
                            else:
                                cx = int(obj.get("left", 0))
                                cy = int(obj.get("top", 0))

                            item = {
                                "uid": uid,
                                "type": obj.get("type", "unknown"),
                                "measurement": area_m2,
                                "unit": "m²",
                                "center_x": cx,
                                "center_y": cy,
                            }

                            # Auto-enrichment
                            item = auto_enrich_item(item, mask, None, proj)

                            items_data.append(item)

            # === סיכום + שלב 3 ===
            st.success("✅ שלב 2 הושלם | עבור למטה לשלב 3 ↓")

            st.divider()
            st.info("🔹 **שלב 3 מתוך 3:** בדוק את הדיווח ושמור למערכת")

            st.header("💾 שלב 3: בדיקה ושמירה")

            # סיכום מספרי
            if "קירות" in report_type:
                st.success(f"📏 סה\"כ: {total_length:.2f} מ'")
            else:
                st.success(f'📐 סה"כ: {total_area:.2f} מ"ר')

            # תיאור סיפורי
            st.markdown(f"### 👁️ תצוגה מקדימה")
            if len(items_data) == 1:
                st.write(f"✓ נמצא **פריט אחד** לדיווח")
            else:
                st.write(f"✓ נמצאו **{len(items_data)} פריטים** לדיווח")

            st.metric("פריטים", len(items_data))
            # ========== CONFIDENCE DISPLAY - תוספת חדשה ==========
            confidence = st.session_state.get(f"last_confidence_{plan_name}", 0)

            if proj.get("_from_metadata") and confidence > 0:
                col_c1, col_c2 = st.columns([3, 1])
                with col_c1:
                    if confidence >= 80:
                        st.success(f"🟢 דיוק מצוין: {confidence:.1f}%")
                    elif confidence >= 60:
                        st.warning(f"🟡 דיוק טוב: {confidence:.1f}%")
                    else:
                        st.error(f"🔴 דיוק נמוך: {confidence:.1f}%")
                        st.caption("⚠️ צייר מעל הקירות המסומנים")
                with col_c2:
                    st.metric("דיוק", f"{confidence:.0f}%")
            # ========== סוף תוספת ==========

            # === טעינת schema ===
            schema = load_form_schema(plan_name, proj)
            st.caption(f"📋 מספר שדות בטופס: {len(schema)}")

            # === פריט נבחר ===
            selected_key = f"selected_item_{plan_name}"
            if selected_key not in st.session_state:
                st.session_state[selected_key] = None

            # === רשימת פריטים קומפקטית ===
            if items_data:
                st.markdown("#### 🔧 בחר פריט:")

                for idx, item in enumerate(items_data, 1):
                    uid = item.get("uid")
                    measurement = item["measurement"]
                    unit = item["unit"]

                    col_num, col_select = st.columns([3, 1])
                    with col_num:
                        st.write(f"**#{idx}** - {measurement:.2f} {unit}")
                    with col_select:
                        if st.button("📝", key=f"select_{uid}", help="ערוך"):
                            st.session_state[selected_key] = uid
                            st.rerun()

                st.markdown("---")

                # === טופס לפריט נבחר ===
                selected_uid = st.session_state[selected_key]

                if selected_uid:
                    selected_item = next(
                        (
                            item
                            for item in items_data
                            if item.get("uid") == selected_uid
                        ),
                        None,
                    )

                    if selected_item:
                        idx = items_data.index(selected_item) + 1
                        st.markdown(f"### ✏️ עריכת פריט #{idx}")
                        st.caption(
                            f"מדידה: {selected_item['measurement']:.2f} {selected_item['unit']}"
                        )

                        # שאלות (ללא container - תאימות לגרסאות ישנות)
                        render_item_questions(
                            selected_uid, selected_item, schema, answers_key
                        )

                        if st.button("✅ סיים עריכה", key="done_editing"):
                            st.session_state[selected_key] = None
                            st.rerun()
                    else:
                        st.warning("פריט לא נמצא")
                else:
                    st.info("👆 בחר פריט מהרשימה לעריכה")

                # ===== PHASE 1: חישוב כמויות (expander) =====
                with st.expander("📊 חישוב כמויות מפורט", expanded=False):
                    if PHASE1_AVAILABLE and items_data:
                        st.caption(
                            "💡 חישוב מקיף: בלוקים, בטון, ריצוף, טיח, צבע ובידוד"
                        )

                        st.markdown("---")
                        st.markdown("#### ⚙️ הגדרות חישוב")

                        col_set1, col_set2, col_set3 = st.columns(3)

                        with col_set1:
                            st.markdown("**מידות קירות:**")
                            cfg_h = st.number_input(
                                "גובה (מ')",
                                value=2.5,
                                min_value=0.1,
                                max_value=10.0,
                                step=0.1,
                                key=f"h_{plan_name}",
                            )
                            cfg_t = st.number_input(
                                "עובי (מ')",
                                value=0.20,
                                min_value=0.05,
                                max_value=1.0,
                                step=0.05,
                                key=f"t_{plan_name}",
                            )

                        with col_set2:
                            st.markdown("**בלוקים ובטון:**")
                            cfg_b = st.number_input(
                                'בלוקים/מ"ר',
                                value=12.5,
                                min_value=1.0,
                                max_value=50.0,
                                step=0.5,
                                key=f"b_{plan_name}",
                            )
                            cfg_w = st.number_input(
                                "בזבוז %",
                                value=5.0,
                                min_value=0.0,
                                max_value=50.0,
                                step=1.0,
                                key=f"w_{plan_name}",
                            )

                        with col_set3:
                            st.markdown("**ריצוף וטיח:**")
                            cfg_tile = st.number_input(
                                'גודל אריח (מ"ר)',
                                value=0.36,
                                min_value=0.01,
                                max_value=2.0,
                                step=0.01,
                                key=f"tile_{plan_name}",
                                help="ברירת מחדל: 60x60cm = 0.36",
                            )
                            cfg_plaster = st.number_input(
                                'עובי טיח (ס"מ)',
                                value=1.5,
                                min_value=0.5,
                                max_value=5.0,
                                step=0.1,
                                key=f"plaster_{plan_name}",
                            )

                        config = {
                            "blocks_per_sqm": cfg_b,
                            "waste_factor": 1.0 + (cfg_w / 100.0),
                            "default_wall_height": cfg_h,
                            "default_wall_thickness": cfg_t,
                            "tile_size_sqm": cfg_tile,
                            "plaster_thickness_m": cfg_plaster / 100.0,  # המרה לmeter
                        }

                        from building_elements import Wall

                        try:
                            try:
                                calc = QuantityCalculator(config=config)
                            except TypeError:
                                calc = QuantityCalculator(config)

                            # הוספת קירות
                            for item in items_data:
                                if item.get("material") and item.get("measurement"):
                                    wall = Wall(
                                        uid=item["uid"],
                                        start=(0, 0),
                                        end=(item["measurement"], 0),
                                        thickness=cfg_t,
                                        height=cfg_h,
                                        material=item["material"],
                                        status="planned",
                                    )
                                    calc.add_wall(wall)

                            # חישוב
                            quantities = calc.calculate_all()

                            # ===== תצוגה מורחבת =====
                            st.markdown("#### 📋 תוצאות חישוב")

                            # שורה 1: בסיסי
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "🧱 קירות", quantities["summary"]["total_walls"]
                                )
                            with col2:
                                blocks = quantities["blocks"]
                                if blocks["wall_count"] > 0:
                                    st.metric(
                                        "🔲 בלוקים", f"{blocks['blocks_needed']:,}"
                                    )
                                    st.caption(
                                        f"🚛 {blocks['blocks_needed']/60:.1f} פלטות"
                                    )
                                else:
                                    st.info("אין בלוקים")
                            with col3:
                                concrete = quantities["concrete"]
                                if concrete["wall_count"] > 0:
                                    st.metric(
                                        "🏗️ בטון",
                                        f"{concrete['total_volume_cubic_meters']:.2f} מ\"ק",
                                    )
                                else:
                                    st.info("אין בטון")

                            # שורה 2: גימור
                            st.markdown("---")
                            st.markdown("**גימור ותשטיחים:**")
                            col4, col5, col6 = st.columns(3)

                            with col4:
                                flooring = quantities["flooring"]
                                if flooring["wall_count"] > 0:
                                    st.metric(
                                        "🔳 ריצוף", f"{flooring['total_area_sqm']} מ\"ר"
                                    )
                                    st.caption(
                                        f"📦 {flooring['boxes_needed']} אריזות ({flooring['tiles_needed']} אריחים)"
                                    )
                                else:
                                    st.info("אין ריצוף")

                            with col5:
                                plaster = quantities["plaster"]
                                if plaster["wall_count"] > 0:
                                    st.metric(
                                        "🧱 טיח", f"{plaster['total_area_sqm']} מ\"ר"
                                    )
                                    st.caption(f"📦 {plaster['bags_needed']} שקים")
                                else:
                                    st.info("טיח: מחושב על כל הקירות")

                            with col6:
                                paint = quantities["paint"]
                                if paint["wall_count"] > 0:
                                    st.metric(
                                        "🎨 צבע", f"{paint['liters_needed']} ליטר"
                                    )
                                    st.caption(
                                        f"🪣 {paint['buckets_needed']} דליים ({paint['coats']} שכבות)"
                                    )
                                else:
                                    st.info("צבע: מחושב על כל הקירות")

                            # שורה 3: בידוד (אם יש)
                            insulation = quantities["insulation"]
                            if insulation["wall_count"] > 0:
                                st.markdown("---")
                                st.markdown("**בידוד:**")
                                col7, col8, col9 = st.columns(3)
                                with col7:
                                    st.metric(
                                        "🛡️ שטח בידוד",
                                        f"{insulation['total_area_sqm']} מ\"ר",
                                    )
                                with col8:
                                    st.metric("📋 פאנלים", insulation["panels_needed"])
                                with col9:
                                    st.caption(
                                        f"גודל פאנל: {insulation['panel_size_sqm']} מ\"ר"
                                    )

                            # פירוט מלא
                            with st.expander("📊 פירוט מלא (JSON)", expanded=False):
                                st.json(quantities)

                            # הערות תחתונות
                            st.caption(
                                f"⚙️ מבוסס על: גובה {cfg_h}m, עובי {cfg_t}m, בזבוז {cfg_w}%, טיח {cfg_plaster}cm"
                            )
                            st.caption("💡 החישוב כולל שתי פאות לטיח וצבע")

                        except Exception as e:
                            st.error(f"❌ שגיאה בחישוב כמויות: {str(e)}")

                            show_debug_q = st.checkbox(
                                "🐛 הצג Debug", value=False, key=f"dbg_q_{plan_name}"
                            )
                            if show_debug_q:
                                st.code(str(e))
                                import traceback

                                st.code(traceback.format_exc())

            # === כפתור שליחה ===
            st.markdown("---")
            st.markdown("### ✅ שמור דיווח")

            if len(items_data) == 0:
                st.warning("⚠️ אין פריטים לשמירה - צייר על התוכנית קודם")
            elif st.button(
                "🚀 שמור דיווח",
                type="primary",
                use_container_width=True,
                help="שמור את כל הפריטים למערכת",
            ):
                # JSON סופי
                json_final = {
                    "project_name": plan_name,
                    "date": report_date.strftime("%Y-%m-%d"),
                    "shift": shift,
                    "mode": "walls" if "קירות" in report_type else "floor",
                    "drawing_mode": drawing_mode,
                    "items": items_data,
                    "totals": {
                        "length_m": (
                            round(total_length, 2) if "קירות" in report_type else 0
                        ),
                        "area_m2": (
                            round(total_area, 2) if "ריצוף" in report_type else 0
                        ),
                    },
                }

                # הצגת JSON
                with st.expander("📄 נתונים מפורטים", expanded=False):
                    st.json(json_final)

                # שמירה
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

                measured = total_length if "קירות" in report_type else total_area
                note_text = f"{report_type} | {shift} | {len(items_data)} פריטים"

                try:
                    final_note = note_text
                    confidence = st.session_state.get(f"last_confidence_{plan_name}", 0)
                    if confidence > 0:
                        final_note += f" (דיוק: {confidence:.0f}%)"

                    save_progress_report(pid, measured, final_note)

                    st.success(f"✅ **הדיווח נשמר בהצלחה!**")
                    st.info(
                        f"📋 **סיכום:** {len(items_data)} פריטים | {report_date.strftime('%d/%m/%Y')} | {shift}"
                    )
                    st.balloons()
                    st.caption("💡 תוכל למצוא את הדיווח בדשבורד של המנהל")

                    # ניקוי
                    st.session_state[report_key] = []
                    st.session_state[answers_key] = {}
                    st.session_state.pop(f"last_confidence_{plan_name}", None)
                    if selected_key in st.session_state:
                        st.session_state[selected_key] = None

                except Exception as e:
                    st.error(f"❌ שגיאה: {str(e)}")

        # === Preview מסומן ===
        if items_data:
            st.markdown("---")
            st.markdown("#### 🔍 Preview")
            selected_uid = st.session_state.get(selected_key)

            # בסיס ה-preview: שמוע הקנבס RGBA על ה-background
            # (canvas.image_data הוא שקיף + קווים בלבד, לא כולל תמונה)
            bg = cv2.resize(rgb, (int(w * scale_factor), int(h * scale_factor)))
            if canvas.image_data is not None:
                overlay = canvas.image_data.copy().astype("uint8")  # RGBA
                if overlay.shape[-1] == 4:
                    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                    fg = overlay[:, :, :3].astype(np.float32)
                    base = (fg * alpha + bg.astype(np.float32) * (1 - alpha)).astype(
                        np.uint8
                    )
                else:
                    base = overlay
            else:
                base = bg

            annotated = create_annotated_preview(
                base,
                items_data,
                selected_uid,
            )
            st.image(annotated, caption="פריטים מסומנים (אדום = נבחר)")
