"""
ConTech Pro - Worker Page MVP
מצב דיווח שטח מתקדם עם פריטים נפרדים
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from datetime import datetime

from database import save_progress_report, save_plan, get_plan_by_filename


def get_corrected_walls(selected_plan, proj):
    """מחזיר את מסכת הקירות המתוקנת"""
    if selected_plan in st.session_state.manual_corrections:
        corrections = st.session_state.manual_corrections[selected_plan]
        corrected = proj["thick_walls"].copy()
        
        if 'added_walls' in corrections:
            corrected = cv2.bitwise_or(corrected, corrections['added_walls'])
        
        if 'removed_walls' in corrections:
            corrected = cv2.subtract(corrected, corrections['removed_walls'])
        
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
        return np.sqrt(dx*dx + dy*dy)
    elif obj.get("type") == "path":
        path = obj.get("path", [])
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            if len(p1) >= 2 and len(p2) >= 2:
                dx = p2[1] - p1[1]
                dy = p2[2] - p1[2]
                total += np.sqrt(dx*dx + dy*dy)
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
        cv2.rectangle(mask, (left, top), (left+width, top+height), 255, -1)
    
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
                    cv2.line(mask, points[i], points[i+1], 255, int(obj.get("strokeWidth", 5)))
    
    return mask


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
        
        cv2.rectangle(annotated, 
                     (cx - 5, cy - text_h - 10),
                     (cx + text_w + 5, cy + 5),
                     (255, 255, 255), -1)
        
        cv2.rectangle(annotated,
                     (cx - 5, cy - text_h - 10),
                     (cx + text_w + 5, cy + 5),
                     (0, 0, 0), 2)
        
        # טקסט
        cv2.putText(annotated, text, (cx, cy), font, font_scale, (0, 0, 255), thickness)
    
    return annotated


def render_worker_page():
    """מצב דיווח שטח מתקדם"""
    st.title("👷 דיווח ביצוע - מתקדם")
    st.caption("✨ דיווח מפורט עם פריטים נפרדים")
    
    if not st.session_state.projects:
        st.warning("📂 אין תוכניות זמינות. אנא העלה תוכנית במצב מנהל.")
        return
    
    # === בחירת פרויקט ===
    plan_name = st.selectbox("📋 בחר תוכנית:", list(st.session_state.projects.keys()))
    proj = st.session_state.projects[plan_name]
    
    # === תאריך ומשמרת ===
    col_date, col_shift = st.columns(2)
    with col_date:
        report_date = st.date_input("📅 תאריך דיווח:", value=datetime.now().date())
    with col_shift:
        shift = st.selectbox("⏰ משמרת:", ["בוקר", "צהריים", "לילה"])
    
    st.markdown("---")
    
    # === בחירת מצב עבודה ===
    report_type = st.radio(
        "🎯 סוג עבודה:",
        ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"],
        horizontal=True
    )
    
    # === בחירת מצב ציור ===
    drawing_mode_display = st.radio(
        "🖌️ מצב ציור:",
        ["✏️ קו (line)", "🖊️ ציור חופשי (freedraw)", "▭ ריבוע (rect)"],
        horizontal=True
    )
    
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
    img_resized = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
    
    # === הגדרות ציור ===
    if "קירות" in report_type:
        fill = "rgba(0,0,0,0)"
        stroke = "#00FF00"
        stroke_width = 8
    else:
        fill = "rgba(255,255,0,0.3)"
        stroke = "#FFFF00"
        stroke_width = 20
    
    # === Canvas ציור ===
    st.markdown("### 🎨 אזור ציור")
    canvas = st_canvas(
        fill_color=fill,
        stroke_color=stroke,
        stroke_width=stroke_width,
        background_image=img_resized,
        height=int(h*scale_factor),
        width=int(w*scale_factor),
        drawing_mode=drawing_mode,
        key=f"canvas_{plan_name}_{report_type}_{drawing_mode}",
        update_streamlit=True
    )
    
    # === עיבוד נתונים ===
    if canvas.json_data and canvas.json_data.get("objects") and canvas.image_data is not None:
        objects = canvas.json_data["objects"]
        
        if len(objects) == 0:
            st.info("🖌️ התחל לצייר על התוכנית")
            return
        
        # === חישוב מדידות לכל פריט ===
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
                    
                    # מרכז לאנוטציה
                    if obj.get("type") == "line":
                        cx = int((obj.get("x1", 0) + obj.get("x2", 0)) / 2)
                        cy = int((obj.get("y1", 0) + obj.get("y2", 0)) / 2)
                    else:
                        cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                        cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2
                    
                    items_data.append({
                        "item_id": idx + 1,
                        "type": obj.get("type", "unknown"),
                        "measurement": length_m,
                        "unit": "m",
                        "center_x": cx,
                        "center_y": cy,
                        "is_wall": True,
                        "is_gypsum": False,
                        "material": "בטון",
                        "height": st.session_state.get("wall_height", 2.60)
                    })
            else:
                # ריצוף/חיפוי
                if obj.get("type") == "rect":
                    area_px = compute_rect_area_px(obj)
                    if area_px > 0:
                        area_m2 = area_px / ((proj["scale"] * scale_factor) ** 2)
                        total_area += area_m2
                        
                        cx = int(obj.get("left", 0)) + int(obj.get("width", 0)) // 2
                        cy = int(obj.get("top", 0)) + int(obj.get("height", 0)) // 2
                        
                        items_data.append({
                            "item_id": idx + 1,
                            "type": "rect",
                            "measurement": area_m2,
                            "unit": "m²",
                            "center_x": cx,
                            "center_y": cy,
                            "is_wall": False,
                            "is_gypsum": False,
                            "material": "ריצוף",
                            "height": 0
                        })
                else:
                    # freedraw - חישוב שטח מסכה
                    mask = create_single_object_mask(obj, int(w*scale_factor), int(h*scale_factor))
                    pixels = np.count_nonzero(mask)
                    if pixels > 0:
                        area_m2 = pixels / ((proj["scale"] * scale_factor) ** 2)
                        total_area += area_m2
                        
                        cy, cx = np.where(mask > 0)
                        if len(cx) > 0:
                            cx_avg = int(np.mean(cx))
                            cy_avg = int(np.mean(cy))
                        else:
                            cx_avg = int(obj.get("left", 0))
                            cy_avg = int(obj.get("top", 0))
                        
                        items_data.append({
                            "item_id": idx + 1,
                            "type": obj.get("type", "unknown"),
                            "measurement": area_m2,
                            "unit": "m²",
                            "center_x": cx_avg,
                            "center_y": cy_avg,
                            "is_wall": False,
                            "is_gypsum": False,
                            "material": "ריצוף",
                            "height": 0
                        })
        
        # === Preview מסומן ===
        if items_data:
            st.markdown("---")
            st.markdown("### 🔍 תצוגה מקדימה")
            
            annotated = create_annotated_preview(
                cv2.resize(rgb, (int(w*scale_factor), int(h*scale_factor))),
                items_data
            )
            st.image(annotated, use_container_width=True, caption="פריטים מסומנים")
            
            # === רשימת פריטים + שאלות ===
            st.markdown("---")
            st.markdown("### 📋 פרטי פריטים")
            
            for item in items_data:
                item_id = item["item_id"]
                measurement = item["measurement"]
                unit = item["unit"]
                
                with st.expander(f"🔧 פריט #{item_id} - {measurement:.2f} {unit}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        item["is_wall"] = st.checkbox(
                            "האם זה קיר?",
                            value=item.get("is_wall", False),
                            key=f"is_wall_{item_id}"
                        )
                        
                        item["is_gypsum"] = st.checkbox(
                            "האם זה קיר גבס?",
                            value=item.get("is_gypsum", False),
                            key=f"is_gypsum_{item_id}"
                        )
                    
                    with col2:
                        item["material"] = st.selectbox(
                            "חומר:",
                            ["בטון", "בלוקים", "גבס", "אחר"],
                            index=["בטון", "בלוקים", "גבס", "אחר"].index(item.get("material", "בטון")),
                            key=f"material_{item_id}"
                        )
                        
                        if item.get("is_wall", False):
                            item["height"] = st.number_input(
                                "גובה (מ'):",
                                value=float(item.get("height", 2.60)),
                                step=0.1,
                                key=f"height_{item_id}"
                            )
            
            # === סיכום כולל ===
            st.markdown("---")
            st.markdown("### 📊 סיכום דיווח")
            
            if "קירות" in report_type:
                st.success(f"✅ **סך הכל:** {total_length:.2f} מטרים")
            else:
                st.success(f"✅ **סך הכל:** {total_area:.2f} מ\"ר")
            
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.metric("מספר פריטים", len(items_data))
            with col_summary2:
                st.metric("תאריך", report_date.strftime("%d/%m/%Y"))
            
            # === כפתור שליחה ===
            if st.button("🚀 שלח דיווח", type="primary", use_container_width=True):
                # בניית JSON סופי
                json_final = {
                    "project_name": plan_name,
                    "date": report_date.strftime("%Y-%m-%d"),
                    "shift": shift,
                    "mode": "walls" if "קירות" in report_type else "floor",
                    "drawing_mode": drawing_mode,
                    "items": items_data,
                    "totals": {
                        "length_m": round(total_length, 2) if "קירות" in report_type else 0,
                        "area_m2": round(total_area, 2) if "ריצוף" in report_type else 0
                    }
                }
                
                # הצגת JSON
                st.markdown("### 📄 נתונים מפורטים")
                st.json(json_final)
                
                # שמירה בדאטאבייס (קיים)
                rec = get_plan_by_filename(plan_name)
                pid = rec['id'] if rec else save_plan(
                    plan_name, plan_name, "1:50",
                    proj["scale"], proj["raw_pixels"], "{}"
                )
                
                measured = total_length if "קירות" in report_type else total_area
                note_text = f"{report_type} | {shift} | {len(items_data)} פריטים"
                
                # שמירה (כולל JSON אם אפשר)
                try:
                    save_progress_report(pid, measured, note_text)
                    st.success("✅ הדיווח נשמר בהצלחה!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ שגיאה בשמירה: {str(e)}")
        else:
            st.info("🖌️ צייר פריטים על התוכנית כדי להתחיל")
    else:
        st.info("🖌️ התחל לצייר על התוכנית")