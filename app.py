import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
import io
from streamlit_drawable_canvas import st_canvas
from datetime import datetime

# ייבוא מהקבצים המסודרים שלך
from styles import *
from utils import safe_process_metadata, safe_analyze_legend, load_stats_df, create_colored_overlay
from analyzer import FloorPlanAnalyzer
from reporter import generate_status_pdf, generate_payment_invoice_pdf
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_all_plans,
    get_project_forecast, get_project_financial_status, 
    calculate_material_estimates, reset_all_data,
    get_payment_invoice_data, get_all_work_types_for_plan,
    get_progress_summary_by_date_range
)

# --- אתחול המערכת ---
apply_all_styles() 
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- Session State ---
if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0
if 'manual_corrections' not in st.session_state: st.session_state.manual_corrections = {}

# --- פונקציה לחישוב קירות מתוקנים ---
def get_corrected_walls(selected_plan, proj):
    """מחזיר את מסכת הקירות המתוקנת (אם יש תיקונים)"""
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

# --- תפריט צד ---
with st.sidebar:
    st.markdown("## 🏗️ ConTech Pro v2.0")
    st.caption("✨ Multi-pass Detection + Manual Corrections")
    mode = st.radio("ניווט", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות"):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1, key="global_wall_height")
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0, key="global_cost_per_meter")
    
    if st.button("🗑️ איפוס נתונים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.session_state.manual_corrections = {}
            st.success("המערכת אופסה")
            st.rerun()

# ==========================================
# 🏢 מצב מנהל
# ==========================================
if mode == "🏢 מנהל פרויקט":
    st.title("ניהול פרויקטים")
    tab1, tab2, tab3, tab4 = st.tabs(["📂 סדנת עבודה", "🎨 תיקונים ידניים", "📊 דשבורד", "💰 חשבונות"])
    
    # --- טאב 1: העלאה ועריכה ---
    with tab1:
        with st.expander("העלאת קבצים", expanded=not st.session_state.projects):
            files = st.file_uploader("גרור PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
            debug_mode = st.selectbox("מצב Debug", ["בסיסי", "מפורט - שכבות", "מלא - עם confidence"], index=0)
            show_debug = debug_mode != "בסיסי"

            if files:
                for f in files:
                    if f.name not in st.session_state.projects:
                        with st.spinner(f"מעבד {f.name} עם Multi-Pass Detection..."):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(f.getvalue())
                                    path = tmp.name
                                
                                analyzer = FloorPlanAnalyzer()
                                pix, skel, thick, orig, meta, conc, blok, floor, debug_img = analyzer.process_file(path, save_debug=show_debug)
                                
                                if not meta.get("plan_name"): 
                                    meta["plan_name"] = f.name.replace(".pdf", "").replace("-", " ").strip()
                                
                                if meta.get("raw_text"):
                                    llm_data = safe_process_metadata(meta["raw_text"])
                                    meta.update({k: v for k, v in llm_data.items() if v})

                                st.session_state.projects[f.name] = {
                                    "skeleton": skel, "thick_walls": thick, "original": orig,
                                    "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                    "concrete_mask": conc, "blocks_mask": blok, "flooring_mask": floor,
                                    "total_length": pix/200.0, "llm_suggestions": llm_data if meta.get("raw_text") else {},
                                    "debug_layers": getattr(analyzer, 'debug_layers', {})
                                }
                                
                                # תצוגת Debug משופרת
                                if show_debug and debug_img is not None:
                                    st.markdown("### 🔍 ניתוח Multi-Pass")
                                    
                                    if debug_mode == "מפורט - שכבות":
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.image(debug_img, caption="תוצאה משולבת", use_column_width=True)
                                        with col2:
                                            if hasattr(analyzer, 'debug_layers') and 'text_combined' in analyzer.debug_layers:
                                                st.image(analyzer.debug_layers['text_combined'], caption="🔴 טקסט שהוסר", use_column_width=True)
                                        with col3:
                                            if hasattr(analyzer, 'debug_layers') and 'walls' in analyzer.debug_layers:
                                                st.image(analyzer.debug_layers['walls'], caption="🟢 קירות שזוהו", use_column_width=True)
                                    
                                    elif debug_mode == "מלא - עם confidence":
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.image(debug_img, caption="תוצאה משולבת", use_column_width=True)
                                        with col2:
                                            st.markdown("""
                                            **מקרא צבעים:**
                                            - 🟠 כתום = טקסט ברור
                                            - 🟡 צהוב = סמלים וכותרות
                                            - 🟣 סגול = מספרי חדרים
                                            - 🟢 ירוק = קירות
                                            - 🔥 אדום-צהוב = confidence גבוה
                                            - 🔵 כחול-שחור = confidence נמוך
                                            """)
                                            
                                            st.metric("Confidence ממוצע", f"{meta.get('confidence_avg', 0):.2f}")
                                            st.metric("פיקסלי טקסט שהוסרו", f"{meta.get('text_removed_pixels', 0):,}")
                                
                                os.unlink(path)
                                st.success(f"✅ {f.name} נותח בהצלחה!")
                            except Exception as e: 
                                st.error(f"שגיאה: {str(e)}")
                                import traceback
                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            
            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state: st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state: st.session_state[scale_key] = proj["metadata"].get("scale", "")
            
            col_edit, col_preview = st.columns([1, 1.5], gap="large")
            
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                
                # אינדיקטור תיקונים
                if selected in st.session_state.manual_corrections:
                    st.success("✏️ תוכנית זו תוקנה ידנית")
                
                p_name = st.text_input("שם התוכנית", key=name_key)
                p_scale_text = st.text_input("קנה מידה (לתיעוד)", key=scale_key, placeholder="1:50")
                
                st.markdown("#### כיול")
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]), key=f"scale_slider_{selected}")
                proj["scale"] = scale_val
                
                # שימוש בגרסה המתוקנת
                corrected_walls = get_corrected_walls(selected, proj)
                corrected_pixels = np.count_nonzero(corrected_walls)
                total_len = corrected_pixels / scale_val
                
                # חישוב חומרים מהגרסה המתוקנת
                kernel = np.ones((6,6), np.uint8)
                conc_corrected = cv2.dilate(cv2.erode(corrected_walls, kernel, iterations=1), kernel, iterations=2)
                block_corrected = cv2.subtract(corrected_walls, conc_corrected)
                
                conc_len = np.count_nonzero(conc_corrected) / scale_val
                block_len = np.count_nonzero(block_corrected) / scale_val
                floor_area = proj["metadata"].get("pixels_flooring_area", 0) / (scale_val ** 2)
                
                proj["total_length"] = total_len
                
                st.info(f"📏 קירות: {total_len:.1f}מ' | בטון: {conc_len:.1f}מ' | בלוקים: {block_len:.1f}מ' | ריצוף: {floor_area:.1f}מ\"ר")
                
                # מחשבון הצעת מחיר
                with st.expander("💰 מחשבון הצעת מחיר", expanded=False):
                    st.markdown("""<div style="background:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:10px;">
                    <strong>מחירון בסיס:</strong> בטון 1200₪/מ' | בלוקים 600₪/מ' | ריצוף 250₪/מ\"ר
                    </div>""", unsafe_allow_html=True)
                    
                    c_price = st.number_input("מחיר בטון (₪/מ')", value=1200.0, step=50.0, key=f"c_price_{selected}")
                    b_price = st.number_input("מחיר בלוקים (₪/מ')", value=600.0, step=50.0, key=f"b_price_{selected}")
                    f_price = st.number_input("מחיר ריצוף (₪/מ\"ר)", value=250.0, step=50.0, key=f"f_price_{selected}")
                    
                    total_quote = (conc_len * c_price) + (block_len * b_price) + (floor_area * f_price)
                    st.markdown(f"#### 💵 סה\"כ הצעת מחיר: {total_quote:,.0f} ₪")
                    
                    quote_df = pd.DataFrame({
                        "פריט": ["קירות בטון", "קירות בלוקים", "ריצוף/חיפוי", "סה\"כ"],
                        "יחידה": ["מ'", "מ'", "מ\"ר", "-"],
                        "כמות": [f"{conc_len:.2f}", f"{block_len:.2f}", f"{floor_area:.2f}", "-"],
                        "מחיר יחידה": [f"{c_price:.0f}₪", f"{b_price:.0f}₪", f"{f_price:.0f}₪", "-"],
                        "סה\"כ": [f"{conc_len*c_price:,.0f}₪", f"{block_len*b_price:,.0f}₪", f"{floor_area*f_price:,.0f}₪", f"{total_quote:,.0f}₪"]
                    })
                    st.dataframe(quote_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                if st.button("💾 שמור תוכנית למערכת", type="primary", key=f"save_{selected}"):
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale_text
                    meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = json.dumps({
                        "concrete_length": conc_len,
                        "blocks_length": block_len,
                        "flooring_area": floor_area
                    }, ensure_ascii=False)
                    
                    plan_id = save_plan(selected, p_name, p_scale_text, scale_val, corrected_pixels, 
                                       meta_json, None, 0, 0, materials)
                    st.toast("✅ נשמר למערכת!")
                    st.success(f"התוכנית נשמרה בהצלחה (ID: {plan_id})")
            
            with col_preview:
                st.markdown("### תצוגה מקדימה")
                
                if selected in st.session_state.manual_corrections:
                    st.caption("✏️ גרסה מתוקנת ידנית")
                
                show_flooring = st.checkbox("הצג ריצוף", value=True, key=f"show_flooring_{selected}")
                
                # שימוש בגרסה המתוקנת
                corrected_walls_display = get_corrected_walls(selected, proj)
                
                kernel_display = np.ones((6,6), np.uint8)
                concrete_corrected = cv2.dilate(cv2.erode(corrected_walls_display, kernel_display, iterations=1), kernel_display, iterations=2)
                blocks_corrected = cv2.subtract(corrected_walls_display, concrete_corrected)
                
                floor_mask = proj["flooring_mask"] if show_flooring else None
                overlay = create_colored_overlay(proj["original"], concrete_corrected, 
                                                blocks_corrected, floor_mask)
                st.image(overlay, use_column_width=True)
                st.caption("🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף")
                
                # ========== תכונה חדשה: ניתוח מקרא ==========
                st.markdown("---")
                with st.expander("🎨 נתח מקרא (AI)", expanded=False):
                    st.caption("המערכת תנסה למצוא את המקרא אוטומטית, או שאתה יכול לחתוך ידנית")
                    
                    # כפתור זיהוי אוטומטי
                    col_auto, col_manual = st.columns([1, 1])
                    
                    with col_auto:
                        if st.button("🔍 מצא מקרא אוטומטית", key=f"auto_legend_{selected}", use_container_width=True):
                            with st.spinner("מחפש מקרא..."):
                                try:
                                    analyzer_temp = FloorPlanAnalyzer()
                                    legend_bbox = analyzer_temp.auto_detect_legend(proj["original"])
                                    
                                    if legend_bbox:
                                        x, y, w, h = legend_bbox
                                        
                                        # חיתוך והצגה
                                        cropped = proj["original"][y:y+h, x:x+w]
                                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                        
                                        st.success("✅ נמצא מקרא!")
                                        st.image(cropped_rgb, caption=f"מקרא שזוהה (גודל: {w}x{h}px)", width=400)
                                        
                                        # שמירה ב-session
                                        if 'auto_legend' not in st.session_state:
                                            st.session_state.auto_legend = {}
                                        st.session_state.auto_legend[selected] = cropped
                                        
                                        # כפתור ניתוח
                                        if st.button("📝 נתח מקרא זה", key=f"analyze_auto_{selected}"):
                                            with st.spinner("מנתח עם Claude AI..."):
                                                _, buffer = cv2.imencode('.png', cropped)
                                                image_bytes = buffer.tobytes()
                                                
                                                result = safe_analyze_legend(image_bytes)
                                                
                                                if isinstance(result, dict) and "error" not in result:
                                                    st.success("✅ ניתוח הושלם!")
                                                    
                                                    col_a, col_b = st.columns(2)
                                                    with col_a:
                                                        st.metric("סוג תוכנית", result.get("plan_type", "לא זוהה"))
                                                        st.metric("רמת ביטחון", f"{result.get('confidence', 0)}%")
                                                    
                                                    with col_b:
                                                        if result.get("materials_found"):
                                                            st.markdown("**חומרים שזוהו:**")
                                                            for material in result["materials_found"]:
                                                                st.markdown(f"- {material}")
                                                    
                                                    if result.get("symbols"):
                                                        st.markdown("**סמלים:**")
                                                        for symbol in result["symbols"][:5]:
                                                            st.markdown(f"- **{symbol.get('symbol', '')}**: {symbol.get('meaning', '')}")
                                                    
                                                    if result.get("notes"):
                                                        st.info(f"💡 {result['notes']}")
                                                    
                                                    proj["metadata"]["legend_analysis"] = result
                                                else:
                                                    st.error(f"❌ {result.get('error', 'שגיאה לא ידועה')}")
                                    else:
                                        st.warning("⚠️ לא נמצא מקרא אוטומטית. נסה לחתוך ידנית למטה.")
                                        st.caption("💡 טיפ: המקרא בדרך כלל בפינה או בצד של התוכנית")
                                        
                                except Exception as e:
                                    st.error(f"❌ שגיאה: {str(e)}")
                    
                    with col_manual:
                        st.markdown("**או:**")
                        st.caption("צייר ריבוע סביב המקרא ידנית ↓")
                    
                    st.markdown("---")
                    st.markdown("### חיתוך ידני")
                    
                    # המרה נכונה של התמונה
                    rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    
                    # רזולוציה גבוהה יותר לחיתוך מדויק
                    scale_factor = min(1.0, 800 / max(w, h))  # ← הגדלנו מ-1000 ל-1200
                    
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    
                    # המרה ל-PIL ושינוי גודל
                    pil_image = Image.fromarray(rgb.astype('uint8'), 'RGB')
                    pil_image_resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    legend_canvas = st_canvas(
                        fill_color="rgba(255,0,0,0.1)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=pil_image_resized,
                        height=new_h,
                        width=new_w,
                        drawing_mode="rect",
                        key=f"legend_canvas_{selected}",
                        update_streamlit=True
                    )
                    
                    if legend_canvas.json_data and legend_canvas.json_data["objects"]:
                        if st.button("🔍 נתח מקרא עם AI", key=f"analyze_legend_{selected}"):
                            with st.spinner("מנתח מקרא..."):
                                try:
                                    # חילוץ הריבוע שצויר
                                    rect = legend_canvas.json_data["objects"][-1]  # הריבוע האחרון
                                    x = int(rect["left"] / scale_factor)
                                    y = int(rect["top"] / scale_factor)
                                    rect_w = int(rect["width"] / scale_factor)
                                    rect_h = int(rect["height"] / scale_factor)
                                    
                                    # חיתוך האזור מהתמונה המקורית
                                    cropped = proj["original"][y:y+rect_h, x:x+rect_w]
                                    
                                    # המרה ל-bytes
                                    _, buffer = cv2.imencode('.png', cropped)
                                    image_bytes = buffer.tobytes()
                                    
                                    # ניתוח עם Claude
                                    result = safe_analyze_legend(image_bytes)
                                    
                                    if isinstance(result, dict) and "error" not in result:
                                        # הצגת תוצאות
                                        st.success("✅ ניתוח הושלם!")
                                        
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("סוג תוכנית", result.get("plan_type", "לא זוהה"))
                                            st.metric("רמת ביטחון", f"{result.get('confidence', 0)}%")
                                        
                                        with col_b:
                                            if result.get("materials_found"):
                                                st.markdown("**חומרים שזוהו:**")
                                                for material in result["materials_found"]:
                                                    st.markdown(f"- {material}")
                                        
                                        if result.get("symbols"):
                                            st.markdown("**סמלים:**")
                                            for symbol in result["symbols"][:5]:
                                                st.markdown(f"- **{symbol.get('symbol', '')}**: {symbol.get('meaning', '')}")
                                        
                                        if result.get("notes"):
                                            st.info(f"💡 {result['notes']}")
                                        
                                        # שמירה למטא-דאטה
                                        proj["metadata"]["legend_analysis"] = result
                                        
                                    elif isinstance(result, dict) and "error" in result:
                                        st.error(f"שגיאה: {result['error']}")
                                    else:
                                        st.warning(f"תשובה לא צפויה: {result}")
                                        
                                except Exception as e:
                                    st.error(f"שגיאה בניתוח: {str(e)}")
                                    import traceback
                                    with st.expander("פרטי שגיאה"):
                                        st.code(traceback.format_exc())
                    else:
                        st.info("👆 צייר ריבוע סביב המקרא בתוכנית ולחץ על הכפתור")

    
    # ==========================================
    # 🎨 טאב 2: תיקונים ידניים
    # ==========================================
    with tab2:
        st.markdown("## 🎨 תיקונים ידניים")
        st.caption("הוסף או הסר קירות באופן ידני למדויקות מקסימלית")
        
        if not st.session_state.projects:
            st.info("📂 אנא העלה תוכנית תחילה בטאב 'סדנת עבודה'")
        else:
            selected_plan = st.selectbox("בחר תוכנית לתיקון:", list(st.session_state.projects.keys()), key="correction_plan_select")
            proj = st.session_state.projects[selected_plan]
            
            correction_mode = st.radio("מצב תיקון:", 
                                      ["➕ הוסף קירות חסרים", "➖ הסר קירות מזויפים", "👁️ השוואה"], 
                                      horizontal=True)
            
            rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale_factor = 800 / w if w > 800 else 1.0
            img_display = Image.fromarray(rgb).resize((int(w*scale_factor), int(h*scale_factor)))
            
            if correction_mode == "➕ הוסף קירות חסרים":
                st.info("🖌️ צייר בירוק על הקירות שהמערכת החמיצה")
                
                canvas_add = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=6,
                    stroke_color="#00FF00",
                    background_image=img_display,
                    height=int(h*scale_factor),
                    width=int(w*scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_add_{selected_plan}",
                    update_streamlit=True
                )
                
                if canvas_add.image_data is not None and np.any(canvas_add.image_data[:, :, 3] > 0):
                    if st.button("✅ אשר הוספה", key="confirm_add"):
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}
                        
                        added_mask = cv2.resize(canvas_add.image_data[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
                        added_mask = (added_mask > 0).astype(np.uint8) * 255
                        
                        st.session_state.manual_corrections[selected_plan]['added_walls'] = added_mask
                        st.success("✅ קירות נוספו! עבור לטאב 'השוואה' לראות את התוצאה")
                        st.rerun()
            
            elif correction_mode == "➖ הסר קירות מזויפים":
                st.info("🖌️ צייר באדום על קירות שהמערכת זיהתה בטעות")
                
                walls_overlay = proj["thick_walls"].copy()
                walls_colored = cv2.cvtColor(walls_overlay, cv2.COLOR_GRAY2RGB)
                walls_colored[walls_overlay > 0] = [0, 255, 255]
                
                combined = cv2.addWeighted(rgb, 0.6, walls_colored, 0.4, 0)
                combined_resized = cv2.resize(combined, (int(w*scale_factor), int(h*scale_factor)))
                img_with_walls = Image.fromarray(combined_resized)
                
                canvas_remove = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=8,
                    stroke_color="#FF0000",
                    background_image=img_with_walls,
                    height=int(h*scale_factor),
                    width=int(w*scale_factor),
                    drawing_mode="freedraw",
                    key=f"canvas_remove_{selected_plan}",
                    update_streamlit=True
                )
                
                if canvas_remove.image_data is not None and np.any(canvas_remove.image_data[:, :, 3] > 0):
                    if st.button("✅ אשר הסרה", key="confirm_remove"):
                        if selected_plan not in st.session_state.manual_corrections:
                            st.session_state.manual_corrections[selected_plan] = {}
                        
                        removed_mask = cv2.resize(canvas_remove.image_data[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
                        removed_mask = (removed_mask > 0).astype(np.uint8) * 255
                        
                        st.session_state.manual_corrections[selected_plan]['removed_walls'] = removed_mask
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
                        st.metric("אורך", f"{corrected_length:.1f} מ'", 
                                 delta=f"{corrected_length - auto_length:+.1f} מ'")
                    
                    st.markdown("---")
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("💾 שמור גרסה מתוקנת", type="primary"):
                            proj["thick_walls"] = corrected_walls
                            proj["raw_pixels"] = corrected_pixels
                            proj["total_length"] = corrected_length
                            
                            meta_json = json.dumps(proj["metadata"], ensure_ascii=False)
                            save_plan(selected_plan, proj["metadata"].get("plan_name"), "1:50", 
                                     proj["scale"], corrected_pixels, meta_json)
                            
                            st.success("✅ הגרסה המתוקנת נשמרה!")
                            
                    
                    with col_btn2:
                        if st.button("🔄 אפס תיקונים", key="reset_corrections"):
                            del st.session_state.manual_corrections[selected_plan]
                            st.success("התיקונים אופסו")
                            st.rerun()
                else:
                    st.info("אין תיקונים ידניים עדיין. עבור לטאב 'הוסף קירות' או 'הסר קירות'")
    
    # --- טאב 3: דשבורד ---
    with tab3:
        st.header("📊 דשבורד פרויקט")
        
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים במערכת")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan_dash = st.selectbox("בחר פרויקט:", plan_options)
            plan_id = int(selected_plan_dash.split("ID: ")[1].strip(")"))
            
            # קבלת נתונים
            forecast = get_project_forecast(plan_id)
            financial = get_project_financial_status(plan_id)
            plan_data = get_plan_by_id(plan_id)
            
            # === KPIs משופרים ===
            k1, k2, k3, k4 = st.columns(4)
            
            total = forecast.get('total_planned', 0)
            built = forecast.get('cumulative_progress', 0)
            percent = (built / total * 100) if total > 0 else 0
            
            with k1:
                st.metric(
                    label="📏 סך הכל",
                    value=f"{total:.1f} מ'",
                    help="סך הכל מטרים בתוכנית"
                )
            
            with k2:
                st.metric(
                    label="✅ בוצע",
                    value=f"{built:.1f} מ'",
                    delta=f"{percent:.1f}%",
                    delta_color="normal"
                )
            
            with k3:
                remaining = total - built
                st.metric(
                    label="⏳ נותר",
                    value=f"{remaining:.1f} מ'",
                    delta=f"{forecast.get('days_to_finish', 0)} ימים",
                    delta_color="inverse"
                )
            
            with k4:
                st.metric(
                    label="💰 עלות נוכחית",
                    value=f"{financial.get('current_cost', 0):,.0f} ₪",
                    help=f"תקציב: {financial.get('budget_limit', 0):,.0f} ₪"
                )
            
            # === Progress Bar ויזואלי ===
            st.markdown("---")
            st.subheader("📊 התקדמות כללית")
            
            # צבע לפי התקדמות
            if percent < 30:
                color = "#EF4444"  # אדום
            elif percent < 70:
                color = "#F59E0B"  # כתום
            else:
                color = "#10B981"  # ירוק
            
            progress_html = f"""
            <div style="margin: 1rem 0;">
                <div style="width: 100%; background: #e0e0e0; border-radius: 10px; height: 35px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="width: {percent}%; background: linear-gradient(90deg, {color}, {color}dd); 
                                height: 100%; display: flex; align-items: center; justify-content: center; 
                                color: white; font-weight: bold; font-size: 16px; transition: width 0.5s;">
                        {percent:.1f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.875rem; color: #666;">
                    <span>התחלה</span>
                    <span>{built:.1f} מ' מתוך {total:.1f} מ'</span>
                    <span>סיום</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
            # === גרף התקדמות ===
            st.markdown("---")
            st.subheader("📈 גרף התקדמות לאורך זמן")
            
            df_stats = load_stats_df()
            if not df_stats.empty:
                # סינון לפרויקט הנוכחי
                df_current = df_stats[df_stats['שם תוכנית'] == plan_data['plan_name']]
                if not df_current.empty:
                    st.bar_chart(df_current, x="תאריך", y="כמות שבוצעה", use_container_width=True)
                else:
                    st.info("אין דיווחים לפרויקט זה")
            else:
                st.info("אין דיווחים במערכת")
            
            # === כפתורי פעולה ===
            st.markdown("---")
            st.subheader("🎯 פעולות ודוחות")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 הדפס דוח PDF", use_container_width=True, type="primary"):
                    with st.spinner("מכין דוח..."):
                        # נסה למצוא את התמונה המקורית
                        if selected_plan and selected_plan in st.session_state.projects:
                            proj = st.session_state.projects[selected_plan]
                            rgb = cv2.cvtColor(proj['original'], cv2.COLOR_BGR2RGB)
                        else:
                            # אם אין תמונה - צור תמונה ריקה
                            rgb = np.ones((800, 1000, 3), dtype=np.uint8) * 255
                            cv2.putText(rgb, "No Image Available", (300, 400), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 3)
                        
                        # נתונים לדוח
                        stats = {
                            'built': built,
                            'total': total,
                            'percent': percent,
                            'remaining': remaining,
                            'cost': financial.get('current_cost', 0),
                            'budget': financial.get('budget_limit', 0)
                        }
                        
                        # יצירת PDF
                        pdf_buffer = generate_status_pdf(plan_data['plan_name'], rgb, stats)
                        
                        st.download_button(
                            label="⬇️ הורד דוח PDF",
                            data=pdf_buffer,
                            file_name=f"status_report_{plan_data['plan_name']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="secondary"
                        )
            
            with col2:
                if st.button("📊 יצא נתונים", use_container_width=True):
                    st.info("💡 בקרוב: ייצוא ל-Excel")
            
            with col3:
                if st.button("📧 שלח דוא\"ל", use_container_width=True):
                    st.info("💡 בקרוב: שליחת דוח באימייל")
            
            # === טבלת דיווחים אחרונים ===
            st.markdown("---")
            st.subheader("📋 דיווחים אחרונים")
            
            reports = get_progress_reports(plan_id)
            if reports:
                # הצג 5 אחרונים
                recent = reports[:5]
                
                for i, r in enumerate(recent, 1):
                    with st.expander(f"📅 {r['date']} - {r['meters_built']:.1f} מ' - {r.get('note', 'אין הערה')}"):
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.write(f"**כמות:** {r['meters_built']:.1f} מטרים")
                            if r.get('note'):
                                st.write(f"**הערה:** {r['note']}")
                        with col_b:
                            st.write(f"**דיווח #{i}**")
                            st.caption(f"ID: {r['id']}")
            else:
                st.info("אין דיווחים לפרויקט זה")
    
    # ==========================================
    # 💰 טאב 4: חשבונות חלקיים
    # ==========================================
    with tab4:
        st.markdown("## 💰 מחולל חשבונות חלקיים")
        st.caption("הפקת חשבונית לתשלום לקבלן על בסיס ביצוע בפועל")
        
        all_plans = get_all_plans()
        if not all_plans:
            st.info("אין פרויקטים במערכת")
        else:
            # בחירת פרויקט
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans]
            selected_plan_invoice = st.selectbox("בחר פרויקט:", plan_options, key="invoice_plan_select")
            plan_id = int(selected_plan_invoice.split("ID: ")[1].strip(")"))
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📅 בחר טווח תאריכים")
                
                # טווח מהיר
                quick_range = st.radio(
                    "בחירה מהירה:",
                    ["שבוע אחרון", "חודש אחרון", "טווח מותאם אישית"],
                    horizontal=True
                )
                
                from datetime import datetime, timedelta
                
                if quick_range == "שבוע אחרון":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                elif quick_range == "חודש אחרון":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                else:  # טווח מותאם אישית
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input(
                            "מתאריך:",
                            value=datetime.now() - timedelta(days=30),
                            key="start_date_picker"
                        )
                    with col_date2:
                        end_date = st.date_input(
                            "עד תאריך:",
                            value=datetime.now(),
                            key="end_date_picker"
                        )
                
                # המרה ל-string
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                st.info(f"📊 תקופת החשבון: {start_str} עד {end_str}")
                
                # === הגדרת מחירי יחידה ===
                st.markdown("### 💵 מחירי יחידה")
                
                # קבלת סוגי עבודות
                work_types = get_all_work_types_for_plan(plan_id)
                
                if not work_types:
                    st.warning("אין דיווחים לפרויקט זה עדיין")
                else:
                    st.caption("ערוך את המחירים לפי הצורך. המחירים המוצגים הם ברירות מחדל.")
                    
                    unit_prices = {}
                    
                    for work_type in work_types:
                        # מחיר ברירת מחדל
                        if 'ריצוף' in work_type.lower() or 'חיפוי' in work_type.lower():
                            default_price = 250
                            unit = 'מ"ר'
                        elif 'בטון' in work_type.lower():
                            default_price = 1200
                            unit = "מ'"
                        elif 'בלוק' in work_type.lower():
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
                                label_visibility="collapsed"
                            )
                            unit_prices[work_type] = price
            
            with col2:
                st.markdown("### 👷 פרטי קבלן")
                st.caption("שדות אלה יופיעו בחשבונית")
                
                contractor_name = st.text_input(
                    "שם הקבלן:",
                    value="",
                    placeholder="ישראל ישראלי",
                    key="contractor_name"
                )
                
                contractor_company = st.text_input(
                    "שם חברה:",
                    value="",
                    placeholder='בניית ישראל בע"מ',
                    key="contractor_company"
                )
                
                contractor_vat = st.text_input(
                    "ח.פ / ע.מ:",
                    value="",
                    placeholder="123456789",
                    key="contractor_vat"
                )
                
                contractor_address = st.text_area(
                    "כתובת:",
                    value="",
                    placeholder="רחוב הבניינים 1, תל אביב",
                    height=80,
                    key="contractor_address"
                )
                
                st.markdown("---")
                
                # כפתור יצירת חשבונית
                if st.button("🧾 צור חשבונית", type="primary", use_container_width=True):
                    # בדיקת שדות חובה
                    if not contractor_name or not contractor_vat:
                        st.error("❌ יש למלא שם קבלן ומספר עוסק")
                    else:
                        with st.spinner("מכין חשבונית..."):
                            try:
                                # קבלת נתוני חשבון
                                invoice_data = get_payment_invoice_data(
                                    plan_id,
                                    start_str,
                                    end_str,
                                    unit_prices
                                )
                                
                                if invoice_data.get('error'):
                                    st.error(f"❌ {invoice_data['error']}")
                                elif not invoice_data['items']:
                                    st.warning("⚠️ אין דיווחים בטווח התאריכים הזה")
                                else:
                                    # פרטי קבלן
                                    contractor_info = {
                                        'name': contractor_name,
                                        'company': contractor_company,
                                        'vat_id': contractor_vat,
                                        'address': contractor_address
                                    }
                                    
                                    # יצירת PDF
                                    pdf_buffer = generate_payment_invoice_pdf(
                                        invoice_data,
                                        contractor_info
                                    )
                                    
                                    # הצגת סיכום
                                    st.success("✅ החשבונית הוכנה בהצלחה!")
                                    
                                    st.markdown("### 📋 סיכום החשבונית")
                                    
                                    # טבלת פריטים
                                    import pandas as pd
                                    df_items = pd.DataFrame([
                                        {
                                            'סוג עבודה': item['work_type'],
                                            'כמות': f"{item['quantity']:.2f}",
                                            'יחידה': item['unit'],
                                            'מחיר יחידה': f"{item['unit_price']:,.0f} ₪",
                                            'סה"כ': f"{item['subtotal']:,.2f} ₪"
                                        }
                                        for item in invoice_data['items']
                                    ])
                                    
                                    st.dataframe(df_items, use_container_width=True, hide_index=True)
                                    
                                    # סיכום סופי
                                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                                    with col_sum1:
                                        st.metric("סכום ביניים", f"{invoice_data['total_amount']:,.2f} ₪")
                                    with col_sum2:
                                        st.metric('מע"מ (17%)', f"{invoice_data['vat']:,.2f} ₪")
                                    with col_sum3:
                                        st.metric("**סה\"כ לתשלום**", f"{invoice_data['total_with_vat']:,.2f} ₪")
                                    
                                    # כפתור הורדה
                                    st.download_button(
                                        label="📥 הורד חשבונית (PDF)",
                                        data=pdf_buffer,
                                        file_name=f"invoice_{invoice_data['plan']['plan_name']}_{start_str}_{end_str}.pdf",
                                        mime="application/pdf",
                                        type="primary",
                                        use_container_width=True
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ שגיאה ביצירת חשבונית: {str(e)}")
                                import traceback
                                with st.expander("פרטי שגיאה"):
                                    st.code(traceback.format_exc())
            
            # תצוגה מקדימה של דיווחים
            st.markdown("---")
            with st.expander("📊 דיווחים בטווח התאריכים"):
                summary = get_progress_summary_by_date_range(plan_id, start_str, end_str)
                if summary:
                    import pandas as pd
                    df_summary = pd.DataFrame([
                        {
                            'סוג עבודה': item['work_type'],
                            'כמות כוללת': f"{item['total_quantity']:.2f}",
                            'יחידה': item['unit'],
                            'מספר דיווחים': item['report_count']
                        }
                        for item in summary
                    ])
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("אין דיווחים בטווח זה")

# ==========================================
# 👷 מצב דיווח
# ==========================================
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    
    if not st.session_state.projects:
        st.warning("אין תוכניות זמינות")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True)
        
        # שימוש בגרסה המתוקנת
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
