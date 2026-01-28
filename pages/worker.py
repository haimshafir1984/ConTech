"""
ConTech Pro - Worker Page v2.0
מצב דיווח שטח מתקדם עם Schema Editor ו-2-Point Mode
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from datetime import datetime

from database import (
    save_progress_report,
    save_plan,
    get_plan_by_filename,
    get_plan_by_id,
    update_plan_metadata,
)


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


def compute_line_length_px(obj):
    """מחשב אורך קו בפיקסלים"""
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
    """יוצר מסכה לאובייקט בודד"""
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
    """Auto-enrichment: מציע is_wall וחומר לפי overlap"""
    if mask is None or corrected_walls is None:
        return item

    # בדיקת overlap עם קירות
    if corrected_walls.shape == mask.shape:
        intersection = cv2.bitwise_and(mask, corrected_walls)
        overlap_pixels = np.count_nonzero(intersection)
        item_pixels = np.count_nonzero(mask)

        if item_pixels > 0:
            overlap_ratio = overlap_pixels / item_pixels
            item["wall_overlap_ratio"] = round(overlap_ratio, 2)
            item["is_wall_suggested"] = overlap_ratio > 0.5
        else:
            item["wall_overlap_ratio"] = 0
            item["is_wall_suggested"] = False

    # ניסיון לזהות חומר
    concrete_mask = proj.get("concrete_mask")
    blocks_mask = proj.get("blocks_mask")

    suggested_material = None
    if concrete_mask is not None and concrete_mask.shape == mask.shape:
        conc_overlap = np.count_nonzero(cv2.bitwise_and(mask, concrete_mask))
        if conc_overlap > item_pixels * 0.5:
            suggested_material = "בטון"

    if blocks_mask is not None and blocks_mask.shape == mask.shape:
        block_overlap = np.count_nonzero(cv2.bitwise_and(mask, blocks_mask))
        if block_overlap > item_pixels * 0.5:
            suggested_material = "בלוקים"

    item["material_suggested"] = suggested_material

    return item


def create_annotated_preview(rgb_image, items_data):
    """יוצר תמונת preview עם מספרים ומדידות"""
    annotated = rgb_image.copy()

    for idx, item in enumerate(items_data, 1):
        cx = int(item.get("center_x", 0))
        cy = int(item.get("center_y", 0))
        measurement = item.get("measurement", 0.0)
        unit = item.get("unit", "m")

        text = f"{idx}: {measurement:.2f}{unit}"

        # רקע לטקסט
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
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
            (0, 0, 0),
            2,
        )

        # טקסט
        cv2.putText(annotated, text, (cx, cy), font, font_scale, (0, 0, 255), thickness)

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


def render_schema_editor(plan_name, proj):
    """מסך הגדרת schema לטופס"""
    st.markdown("### ⚙️ הגדרת טופס (למנהל)")
    st.caption("הגדר שדות ושאלות שיופיעו לכל פריט")

    schema = load_form_schema(plan_name, proj)

    # UI לעריכת schema - ללא expanders מקוננים!
    new_schema = []

    st.markdown("---")

    for idx, field in enumerate(schema):
        # Card פשוט במקום expander
        st.markdown(f"#### שדה #{idx+1}: {field.get('label', 'ללא שם')}")

        col1, col2 = st.columns(2)

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
                "תווית:", value=field.get("label", ""), key=f"schema_label_{idx}"
            )

        with col2:
            field_key = st.text_input(
                "Key (משתנה):", value=field.get("key", ""), key=f"schema_key_{idx}"
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
                    field_default = st.selectbox(
                        "ברירת מחדל:",
                        field_options,
                        index=0,
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

        new_schema.append(new_field)

        if st.button("🗑️ מחק שדה", key=f"delete_field_{idx}"):
            # סימון למחיקה
            new_schema = [f for i, f in enumerate(new_schema) if i != idx]

        st.markdown("---")

    # כפתור הוספת שדה
    if st.button("➕ הוסף שדה חדש"):
        new_schema.append(
            {"type": "text", "label": "שדה חדש", "key": "new_field", "default": ""}
        )
        st.rerun()

    # שמירה
    if st.button("💾 שמור Schema", type="primary"):
        if save_form_schema(plan_name, proj, new_schema):
            st.success("✅ Schema נשמר!")
            st.rerun()
        else:
            st.error("❌ שגיאה בשמירה")


def render_item_questions(item_id, item, schema):
    """מציג שאלות לפריט לפי ה-schema"""
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

        if field_type == "checkbox":
            item[field_key] = st.checkbox(
                field_label,
                value=item.get(
                    field_key, field_default if field_default is not None else False
                ),
                key=f"{field_key}_{item_id}",
            )

        elif field_type == "select":
            options = field.get("options", [""])
            current_val = item.get(field_key, field_default)
            if current_val not in options and options:
                current_val = options[0]

            item[field_key] = st.selectbox(
                field_label,
                options,
                index=options.index(current_val) if current_val in options else 0,
                key=f"{field_key}_{item_id}",
            )

        elif field_type == "number":
            item[field_key] = st.number_input(
                field_label,
                value=float(
                    item.get(
                        field_key, field_default if field_default is not None else 0
                    )
                ),
                step=field.get("step", 0.1),
                key=f"{field_key}_{item_id}",
            )

        elif field_type == "text":
            item[field_key] = st.text_input(
                field_label,
                value=item.get(
                    field_key, field_default if field_default is not None else ""
                ),
                key=f"{field_key}_{item_id}",
            )


# ==========================================
# Worker Page Main
# ==========================================


def render_worker_page():
    """מצב דיווח שטח מתקדם v2.0"""
    st.title("👷 דיווח ביצוע - מתקדם v2.0")
    st.caption("✨ Schema Editor, Auto-enrichment, 2-Point Mode")

    if not st.session_state.projects:
        st.warning("📂 אין תוכניות זמינות. אנא העלה תוכנית במצב מנהל.")
        return

    # === בחירת פרויקט ===
    plan_name = st.selectbox("📋 בחר תוכנית:", list(st.session_state.projects.keys()))
    proj = st.session_state.projects[plan_name]

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
    col_mode1, col_mode2 = st.columns([2, 1])

    with col_mode1:
        drawing_mode_display = st.radio(
            "🖌️ מצב ציור:",
            ["✏️ קו (line)", "🖊️ ציור חופשי (freedraw)", "▭ ריבוע (rect)"],
            horizontal=True,
        )

    with col_mode2:
        two_point_mode = st.checkbox("🎯 מצב 2 נקודות", value=False)

    if two_point_mode:
        drawing_mode = "point"
        st.info("📍 לחץ על 2 נקודות על התמונה ליצירת קו")
    else:
        if "קו" in drawing_mode_display:
            drawing_mode = "line"
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

    # === הגדרות ציור ===
    if "קירות" in report_type:
        fill = "rgba(0,0,0,0)"
        stroke = "#00FF00"
        stroke_width = 8
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

        # 2-Point Mode Logic
        if two_point_mode and canvas.json_data and canvas.json_data.get("objects"):
            objects = canvas.json_data["objects"]
            points = [
                obj
                for obj in objects
                if obj.get("type") in ["circle", "rect"] and obj.get("width", 0) < 20
            ]

            if len(points) >= 2:
                st.info(f"✅ נאספו {len(points)} נקודות. לחץ 'המר לקווים' ליצירת קווים")

                if st.button("🔄 המר לקווים", key="convert_points"):
                    # יצירת line objects מנקודות
                    if "manual_lines" not in st.session_state:
                        st.session_state.manual_lines = []

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
                        }
                        st.session_state.manual_lines.append(line_obj)

                    st.success(f"✅ נוצרו {len(points)//2} קווים!")
                    st.rerun()

    with col_right:
        st.markdown("### 📋 פרטי פריטים")

        # עיבוד נתונים
        objects = []

        # הוספת אובייקטים רגילים
        if canvas.json_data and canvas.json_data.get("objects"):
            if two_point_mode:
                # סינון נקודות
                objects = [
                    obj
                    for obj in canvas.json_data["objects"]
                    if not (
                        obj.get("type") in ["circle", "rect"]
                        and obj.get("width", 0) < 20
                    )
                ]
            else:
                objects = canvas.json_data["objects"]

        # הוספת קווים ידניים
        if "manual_lines" in st.session_state:
            objects.extend(st.session_state.manual_lines)

        if len(objects) == 0 or canvas.image_data is None:
            st.info("🖌️ התחל לצייר על התוכנית")
        else:
            # === חישוב מדידות ===
            items_data = []
            total_length = 0.0
            total_area = 0.0

            for idx, obj in enumerate(objects):
                # חישוב מדידה
                if "קירות" in report_type:
                    length_px = compute_line_length_px(obj)
                    if length_px > 0:
                        length_m = length_px / scale_factor / proj["scale"]
                        total_length += length_m

                        # מרכז
                        if obj.get("type") == "line":
                            cx = int((obj.get("x1", 0) + obj.get("x2", 0)) / 2)
                            cy = int((obj.get("y1", 0) + obj.get("y2", 0)) / 2)
                        else:
                            cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                            cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2

                        item = {
                            "item_id": idx + 1,
                            "type": obj.get("type", "unknown"),
                            "measurement": length_m,
                            "unit": "m",
                            "center_x": cx,
                            "center_y": cy,
                        }

                        # Auto-enrichment
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
                            area_m2 = area_px / ((proj["scale"] * scale_factor) ** 2)
                            total_area += area_m2

                            cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                            cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2

                            items_data.append(
                                {
                                    "item_id": idx + 1,
                                    "type": "rect",
                                    "measurement": area_m2,
                                    "unit": "m²",
                                    "center_x": cx,
                                    "center_y": cy,
                                }
                            )
                    else:
                        mask = create_single_object_mask(
                            obj, int(w * scale_factor), int(h * scale_factor)
                        )
                        pixels = np.count_nonzero(mask)
                        if pixels > 0:
                            area_m2 = pixels / ((proj["scale"] * scale_factor) ** 2)
                            total_area += area_m2

                            cy_arr, cx_arr = np.where(mask > 0)
                            if len(cx_arr) > 0:
                                cx = int(np.mean(cx_arr))
                                cy = int(np.mean(cy_arr))
                            else:
                                cx = int(obj.get("left", 0))
                                cy = int(obj.get("top", 0))

                            items_data.append(
                                {
                                    "item_id": idx + 1,
                                    "type": obj.get("type", "unknown"),
                                    "measurement": area_m2,
                                    "unit": "m²",
                                    "center_x": cx,
                                    "center_y": cy,
                                }
                            )

            # === סיכום ===
            if "קירות" in report_type:
                st.success(f"✅ סה\"כ: {total_length:.2f} מ'")
            else:
                st.success(f'✅ סה"כ: {total_area:.2f} מ"ר')

            st.metric("פריטים", len(items_data))

            # === טעינת schema ===
            schema = load_form_schema(plan_name, proj)

            # === רשימת פריטים ===
            if items_data:
                st.markdown("#### 🔧 פריטים:")

                for item in items_data:
                    item_id = item["item_id"]
                    measurement = item["measurement"]
                    unit = item["unit"]

                    # Card קומפקטי
                    with st.expander(
                        f"#{item_id} - {measurement:.2f} {unit}", expanded=False
                    ):
                        render_item_questions(item_id, item, schema)

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
                    "two_point_mode": two_point_mode,
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
                    st.balloons()

                    # ניקוי
                    if "manual_lines" in st.session_state:
                        del st.session_state.manual_lines

                except Exception as e:
                    st.error(f"❌ שגיאה: {str(e)}")

        # Preview מסומן (למטה)
        if items_data:
            st.markdown("---")
            st.markdown("#### 🔍 Preview")
            annotated = create_annotated_preview(
                cv2.resize(rgb, (int(w * scale_factor), int(h * scale_factor))),
                items_data,
            )
            st.image(annotated, caption="פריטים מסומנים")
