"""
ConTech Pro - Worker Page v2.2
מצב דיווח שטח מתקדם עם ברירות מחדל A4+1:50, שמירת תשובות, ומצב 2 נקודות
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from datetime import datetime
import uuid
import re

from database import (
    save_progress_report,
    save_plan,
    get_plan_by_filename,
    get_plan_by_id,
    update_plan_metadata,
)

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


def extract_segments_from_mask(walls_mask, scale):
    """
    מחלץ segments מתוך מסכה קיימת ללא Hough כפול

    Args:
        walls_mask: מסכת קירות (numpy array)
        scale: פיקסלים למטר

    Returns:
        רשימת segments: [{'start': (x,y), 'end': (x,y), 'length_px': ...}]
    """
    segments = []

    # שיטה 1: שימוש ב-contours (יותר יציב מ-Hough)
    contours, _ = cv2.findContours(
        walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # פישוט הקונטור לקווים
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # המרה לקטעים
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]

            length_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            # סינון קטעים קצרים מדי
            if length_px > 20:  # מינימום 20 פיקסלים
                segments.append(
                    {
                        "start": tuple(p1),
                        "end": tuple(p2),
                        "length_px": length_px,
                        "length_m": length_px / scale,
                        "source": "contours",
                    }
                )

    return segments


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
    """מצב דיווח שטח מתקדם v2.2"""
    st.title("👷 דיווח ביצוע - מתקדם v2.2")
    st.caption("✨ ברירות מחדל A4+1:50, שמירת תשובות, מצב 2 נקודות")
    
    # אתחול projects אם לא קיים
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    
    if not st.session_state.projects:
        st.warning("📂 אין תוכניות זמינות. אנא העלה תוכנית במצב מנהל.")
        st.info("💡 לך ל'מנהל פרויקט' והעלה קובץ PDF של תוכנית")
        return

    # === בחירת פרויקט ===
    plan_name = st.selectbox("📋 בחר תוכנית:", list(st.session_state.projects.keys()))
    proj = st.session_state.projects[plan_name]

    # === בדיקת scale עם fallback ===
    scale_value, is_fallback = get_scale_with_fallback(proj)
    proj["scale"] = scale_value  # עדכון לשימוש

    if is_fallback:
        st.warning("⚠️ משתמש בברירת מחדל A4 + 1:50 (לא הוגדר במנהל)")
    else:
        st.success("✅ סקלה מוגדרת לפי מנהל")

    # אתחול report_objects (מקור אמת יחיד)
    report_key = f"report_objects_{plan_name}"
    if report_key not in st.session_state:
        st.session_state[report_key] = []

    # אתחול answers
    answers_key = f"item_answers_{plan_name}"
    if answers_key not in st.session_state:
        st.session_state[answers_key] = {}

    # === Schema Editor (Expander למנהל) ===
    with st.expander("⚙️ הגדרת טופס (למנהל)", expanded=False):
        render_schema_editor(plan_name, proj)

    st.markdown("---")

    # === תאריך ומשמרת ===
    col_date, col_shift = st.columns(2)
    with col_date:
        report_date = st.date_input("📅 תאריך דיווח:", value=datetime.now().date())
    with col_shift:
        shift = st.selectbox("⏰ משמרת:", ["בוקר", "צהריים", "לילה"])

    st.markdown("---")

    # === בחירת מצב עבודה ===
    report_type = st.radio(
        "🎯 סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True
    )

    # === בחירת מצב ציור ===
    col_mode1, col_mode2 = st.columns([3, 1])

    with col_mode1:
        drawing_mode_display = st.radio(
            "🖌️ מצב ציור:",
            ["✏️ קו ישר (line)", "🖊️ ציור חופשי (freedraw)", "▭ ריבוע (rect)"],
            horizontal=True,
        )

    with col_mode2:
        two_point_mode = st.checkbox("🎯 2 נקודות", value=False)

    if two_point_mode:
        drawing_mode = "point"
        st.info("📍 לחץ על 2 נקודות ליצירת קו. לאחר מכן לחץ 'המר לקווים'")
    else:
        if "קו" in drawing_mode_display:
            drawing_mode = "line"
            st.info("💡 לחץ והחזק, גרור לכיוון הרצוי, ושחרר ליצירת קו ישר מדויק")
        elif "חופשי" in drawing_mode_display:
            drawing_mode = "freedraw"
        else:
            drawing_mode = "rect"

    st.markdown("---")

    # === הכנת תמונה ===
    corrected_walls = get_corrected_walls(plan_name, proj)
    rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale_factor = 800 / w if w > 800 else 1.0
    img_resized = Image.fromarray(rgb).resize(
        (int(w * scale_factor), int(h * scale_factor))
    )

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
            background_image=img_resized,
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

        # Canvas ציור
        canvas = st_canvas(
            fill_color=fill,
            stroke_color=stroke,
            stroke_width=stroke_width if not two_point_mode else 1,
            background_image=img_resized,
            height=int(h * scale_factor),
            width=int(w * scale_factor),
            drawing_mode=drawing_mode,
            point_display_radius=5 if two_point_mode else 0,
            key=f"canvas_{plan_name}_{report_type}_{drawing_mode}_{two_point_mode}",
            update_streamlit=True,
        )
        # === הוסף כאן ===
        # Snap Indicator (אינדיקציה ויזואלית)
        if PHASE1_AVAILABLE and f"snap_{plan_name}" in st.session_state:
            snap_engine = st.session_state[f"snap_{plan_name}"]

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

            # === סיכום ===
            if "קירות" in report_type:
                st.success(f"✅ סה\"כ: {total_length:.2f} מ'")
            else:
                st.success(f'✅ סה"כ: {total_area:.2f} מ"ר')

            st.metric("פריטים", len(items_data))

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

            # === כפתור שליחה ===
            st.markdown("---")
            if st.button("🚀 שלח דיווח", type="primary", use_container_width=True):
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
                    save_progress_report(pid, measured, note_text)
                    st.success("✅ הדיווח נשמר!")

                    # ניקוי
                    st.session_state[report_key] = []
                    st.session_state[answers_key] = {}
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
