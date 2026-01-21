"""
ConTech Pro - Worker Page
מצב דיווח שטח לעובדים
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from database import save_progress_report, save_plan, get_plan_by_filename
from analyzer import compute_skeleton_length_px


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


def render_worker_page():
    """מצב דיווח שטח"""
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.warning("אין תוכניות זמינות")
        return
    
    plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
    proj = st.session_state.projects[plan_name]
    
    report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True)
    
    corrected_walls = get_corrected_walls(plan_name, proj)
    
    rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale_factor = 800 / w if w > 800 else 1.0
    img_resized = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
    
    if "קירות" in report_type:
        fill = "rgba(0,0,0,0)"
        stroke = "#00FF00"
        stroke_width = 8
    else:
        fill = "rgba(255,255,0,0.3)"
        stroke = "#FFFF00"
        stroke_width = 20
    
    canvas = st_canvas(
        fill_color=fill, stroke_color=stroke, stroke_width=stroke_width,
        background_image=img_resized,
        height=int(h*scale_factor), width=int(w*scale_factor),
        drawing_mode="freedraw",
        key=f"canvas_{plan_name}_{report_type}",
        update_streamlit=True
    )
    
    if canvas.json_data and canvas.json_data["objects"] and canvas.image_data is not None:
        measured = 0
        if "קירות" in report_type:
            user_draw = canvas.image_data[:, :, 3] > 0
            walls_resized = cv2.resize(corrected_walls, (int(w*scale_factor), int(h*scale_factor)))
            intersection = np.logical_and(user_draw, walls_resized > 0)
            measured = np.count_nonzero(intersection) / scale_factor / proj["scale"]
        else:
            pixels = np.count_nonzero(canvas.image_data[:, :, 3] > 0)
            measured = pixels / ((proj["scale"] * scale_factor) ** 2)
        
        if measured > 0:
            unit = 'מ"ר' if 'ריצוף' in report_type else 'מטר'
            st.success(f"✅ {measured:.2f} {unit}")
            
            if st.button("🚀 שלח דיווח", type="primary"):
                rec = get_plan_by_filename(plan_name)
                pid = rec['id'] if rec else save_plan(plan_name, plan_name, "1:50", proj["scale"], proj["raw_pixels"], "{}")
                save_progress_report(pid, measured, report_type)
                st.success("הדיווח נשמר בהצלחה!")
