import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from analyzer import FloorPlanAnalyzer
import tempfile
import os
import json
import io
from streamlit_drawable_canvas import st_canvas
from database import (
    init_database, save_plan, save_progress_report, 
    get_progress_reports, get_plan_by_filename, get_plan_by_id, get_all_plans,
    get_project_forecast, 
    calculate_material_estimates, get_project_financial_status, reset_all_data
)
from datetime import datetime
from reporter import generate_status_pdf

# --- הגדרות ראשוניות ---
Image.MAX_IMAGE_PIXELS = None
init_database()

# --- פונקציות עזר ---
def safe_process_metadata(raw_text):
    try:
        from brain import process_plan_metadata
        return process_plan_metadata(raw_text)
    except (ImportError, Exception):
        return {}

def safe_analyze_legend(image_bytes):
    try:
        from brain import analyze_legend_image
        return analyze_legend_image(image_bytes)
    except Exception as e:
        return f"Error: {str(e)}"

def load_stats_df():
    reports = get_progress_reports()
    if reports:
        df = pd.DataFrame(reports)
        return df.rename(columns={
            'date': 'תאריך', 'plan_name': 'שם תוכנית',
            'meters_built': 'כמות שבוצעה', 'note': 'הערה'
        })
    return pd.DataFrame()

# --- פונקציה משודרגת: יצירת תמונה צבעונית (בטון+בלוקים+ריצוף) ---
def create_colored_overlay(original, concrete_mask, blocks_mask, flooring_mask=None):
    # המרה ל-RGB
    img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)
    overlay = img_vis.copy()
    
    # צביעת בטון (כחול)
    overlay[concrete_mask > 0] = [30, 144, 255] 
    
    # צביעת בלוקים (כתום)
    overlay[blocks_mask > 0] = [255, 165, 0]
    
    # צביעת ריצוף (סגול בהיר) - אם נבחר להציג
    if flooring_mask is not None:
         overlay[flooring_mask > 0] = [200, 100, 255]
    
    # שילוב עם שקיפות
    cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, img_vis)
    return img_vis.astype(np.uint8)

st.set_page_config(page_title="ConTech Pro", layout="wide", page_icon="🏗️")

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Heebo', sans-serif; direction: rtl; }
    :root { --primary-blue: #0F62FE; --bg-gray: #F4F7F6; --card-border: #E0E0E0; --text-dark: #161616; --text-meta: #6F6F6F; }
    .stCard { background-color: white; padding: 24px; border-radius: 12px; border: 1px solid var(--card-border); box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 20px; }
    .kpi-container { display: flex; flex-direction: column; background: white; padding: 20px; border-radius: 12px; border: 1px solid #EAEAEA; box-shadow: 0 4px 12px rgba(0,0,0,0.03); height: 100%; }
    .mat-card { text-align: center; background: white; border: 1px solid #EEE; border-radius: 10px; padding: 15px; }
    .mat-val { font-size: 20px; font-weight: bold; color: var(--primary-blue); }
    .mat-lbl { font-size: 14px; color: #666; }
    .price-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-right: 4px solid #0F62FE; margin-bottom: 10px; }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label { text-align: right !important; width: 100%; direction: rtl; }
    .stButton button { border-radius: 8px; font-weight: 500; height: 45px; }
    section[data-testid="stSidebar"] { background-color: #FAFAFA; border-left: 1px solid #EEE; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'projects' not in st.session_state: st.session_state.projects = {}
if 'wall_height' not in st.session_state: st.session_state.wall_height = 2.5
if 'default_cost_per_meter' not in st.session_state: st.session_state.default_cost_per_meter = 0.0

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🏗️")
    st.markdown("### **ConTech Pro**")
    st.caption("מערכת ניהול ובקרה לקבלני שלד")
    st.markdown("---")
    mode = st.radio("בחר אזור עבודה:", ["🏢 מנהל פרויקט", "👷 דיווח שטח"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ הגדרות גלובליות", expanded=False):
        st.session_state.wall_height = st.number_input("גובה קירות (מ')", value=st.session_state.wall_height, step=0.1)
        st.session_state.default_cost_per_meter = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, step=10.0)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ איפוס מערכת מלא", help="מוחק את כל הנתונים והפרויקטים"):
        if reset_all_data():
            st.session_state.projects = {}
            st.success("המערכת אופסה")
            st.rerun()

# --- לוגיקה ראשית ---
if mode == "🏢 מנהל פרויקט":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("ניהול פרויקטים")
        st.caption("העלאת תוכניות, כיול ובקרת תקציב")
    
    tab1, tab2 = st.tabs(["📂 העלאת תוכניות", "📊 דשבורד מנהלים"])
    with tab1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        files = st.file_uploader("גרור לכאן קבצי PDF או לחץ לבחירה", type="pdf", accept_multiple_files=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if files:
            for f in files:
                if f.name not in st.session_state.projects:
                    with st.spinner(f"מפענח את {f.name}..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(f.getvalue())
                                path = tmp.name
                            
                            analyzer = FloorPlanAnalyzer()
                            # --- שדרוג: קבלת 8 ערכים כולל flooring_mask ---
                            pix, skel, thick, orig, meta, conc_mask, blok_mask, floor_mask = analyzer.process_file(path)
                            
                            if not meta.get("plan_name"): 
                                meta["plan_name"] = f.name.replace(".pdf", "").replace("-", " ").strip()
                            
                            raw_text = meta.get("raw_text", "")
                            llm_metadata = {}
                            if raw_text:
                                llm_metadata = safe_process_metadata(raw_text)
                                if llm_metadata.get("plan_name"): meta["plan_name"] = llm_metadata["plan_name"]
                                if llm_metadata.get("scale"): meta["scale"] = llm_metadata["scale"]
                                if llm_metadata.get("plan_type"): meta["plan_type"] = llm_metadata["plan_type"]
                            
                            # שמירה בזיכרון
                            st.session_state.projects[f.name] = {
                                "skeleton": skel, "thick_walls": thick, "original": orig,
                                "raw_pixels": pix, "scale": 200.0, "metadata": meta,
                                "concrete_mask": conc_mask, "blocks_mask": blok_mask,
                                "flooring_mask": floor_mask,  # שומרים את הריצוף
                                "total_length": pix / 200.0, "llm_suggestions": llm_metadata
                            }
                            os.unlink(path)
                            st.success(f"✅ {f.name} נטען בהצלחה")
                        except Exception as e:
                            st.error(f"❌ שגיאה בטעינת {f.name}: {str(e)}")

        if st.session_state.projects:
            st.markdown("---")
            selected = st.selectbox("בחר תוכנית לעריכה:", options=list(st.session_state.projects.keys()))
            proj = st.session_state.projects[selected]
            name_key = f"name_{selected}"
            scale_key = f"scale_{selected}"
            if name_key not in st.session_state: st.session_state[name_key] = proj["metadata"].get("plan_name", "")
            if scale_key not in st.session_state: st.session_state[scale_key] = proj["metadata"].get("scale", "")

            col_edit, col_preview = st.columns([1, 1.5])
            with col_edit:
                st.markdown("### הגדרות תוכנית")
                
                # --- סיווג תוכנית ---
                current_meta = proj.get("metadata", {})
                detected_type = current_meta.get("plan_type", "construction")
                type_map = {
                    "construction": "בנייה (ברירת מחדל)", "demolition": "הריסה 🔨",
                    "ceiling": "תקרה (לא למדידה) 💡", "electricity": "חשמל ⚡",
                    "plumbing": "אינסטלציה 💧", "other": "אחר"
                }
                index_val = list(type_map.keys()).index(detected_type) if detected_type in type_map else 0
                selected_type_key = st.selectbox("סוג תוכנית", options=list(type_map.keys()), format_func=lambda x: type_map[x], index=index_val, key=f"type_{selected}")
                
                if selected_type_key == "ceiling": st.warning("⚠️ שים לב: זו תוכנית תקרה.")
                elif selected_type_key == "demolition": st.error("🛑 זו תוכנית הריסה.")
                proj["metadata"]["plan_type"] = selected_type_key

                # --- שדות עריכה ---
                p_name = st.text_input("שם התוכנית", key=name_key)
                p_scale = st.text_input("קנה מידה", key=scale_key)
                
               # === לימוד מקרא ===
                with st.expander("📖 לימוד מקרא (AI Vision)", expanded=False):
                    st.info("סמן את המקרא בשרטוט כדי שהמערכת תלמד אותו.")
                    target_width = st.slider("🔍 זום (רוחב תצוגה)", 600, 1500, 800, step=50, key=f"zoom_{selected}")
                    img_for_legend = Image.fromarray(cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB))
                    w_percent = (target_width / float(img_for_legend.size[0]))
                    h_size = int((float(img_for_legend.size[1]) * float(w_percent)))
                    img_resized = img_for_legend.resize((target_width, h_size), Image.Resampling.NEAREST)
                    
                    canvas_legend = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#FFA500",
                        background_image=img_resized,
                        height=h_size,
                        width=target_width,
                        drawing_mode="rect",
                        key=f"legend_canv_{selected}_{target_width}",
                        display_toolbar=True
                    )
                    
                    if canvas_legend.json_data and canvas_legend.json_data["objects"]:
                        if st.button("👁️ פענח את הסימון", key=f"btn_leg_{selected}"):
                            obj = canvas_legend.json_data["objects"][-1]
                            left, top = int(obj["left"]), int(obj["top"])
                            width, height = int(obj["width"]), int(obj["height"])
                            img_arr = np.array(img_resized)
                            if width > 0 and height > 0:
                                cropped = img_arr[top:top+height, left:left+width]
                                if cropped.size > 0:
                                    pil_crop = Image.fromarray(cropped)
                                    buf = io.BytesIO()
                                    pil_crop.save(buf, format="PNG")
                                    byte_im = buf.getvalue()
                                    with st.spinner("ה-AI מנתח את המקרא..."):
                                        analysis = safe_analyze_legend(byte_im)
                                        st.success("פענוח הושלם!")
                                        st.text_area("תוצאת AI:", value=analysis, height=100)
                                        proj["metadata"]["legend_analysis"] = analysis
                            else:
                                st.warning("אנא סמן אזור תקין")

                # --- הגדרות תקציב וכיול ---
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    target_date_val = st.date_input("תאריך יעד", key=f"td_{selected}")
                    target_date_str = target_date_val.strftime("%Y-%m-%d") if target_date_val else None
                with col_d2: budget_limit_val = st.number_input("תקציב (₪)", step=1000.0, key=f"bl_{selected}")
                cost_per_meter_val = st.number_input("עלות למטר (₪)", value=st.session_state.default_cost_per_meter, key=f"cpm_{selected}")
                
                st.markdown("#### כיול")
                scale_val = st.slider("פיקסלים למטר", 10.0, 1000.0, float(proj["scale"]), key=f"sl_{selected}")
                proj["scale"] = scale_val
                
                # חישוב כמויות
                total_len = proj["raw_pixels"] / scale_val
                conc_len = proj["metadata"].get("pixels_concrete", 0) / scale_val
                block_len = proj["metadata"].get("pixels_blocks", 0) / scale_val
                
                # שטח ריצוף (מ"ר) = פיקסלים / (סקייל^2)
                floor_area_sqm = proj["metadata"].get("pixels_flooring_area", 0) / (scale_val * scale_val)
                proj["total_length"] = total_len
                
                st.info(f"📏 קירות: {total_len:.1f} מ' | 🔲 ריצוף: {floor_area_sqm:.1f} מ\"ר")

                # --- מחשבון הצעת מחיר (כולל ריצוף) ---
                with st.expander("💰 מחשבון הצעת מחיר", expanded=True):
                    st.markdown("""<div class="price-box">
                    <strong>מחירון בסיס:</strong><br>
                    בטון: 1200 | בלוקים: 600 | ריצוף: 250
                    </div>""", unsafe_allow_html=True)
                    
                    c_price = st.number_input("מחיר בטון (₪/מ')", value=1200.0, step=50.0)
                    b_price = st.number_input("מחיר בלוקים (₪/מ')", value=600.0, step=50.0)
                    f_price = st.number_input("מחיר ריצוף (₪/מ\"ר)", value=250.0, step=50.0)
                    
                    total_cost_calc = (conc_len * c_price) + (block_len * b_price) + (floor_area_sqm * f_price)
                    st.markdown(f"#### 💵 סה\"כ: {total_cost_calc:,.0f} ₪")
                    
                    # ייצוא לאקסל
                    quote_data = {
                        "פריט": ["קירות בטון", "קירות בלוקים", "ריצוף/חיפוי", "סה\"כ"],
                        "יחידה": ["מטר אורך", "מטר אורך", "מ\"ר", "-"],
                        "כמות": [f"{conc_len:.2f}", f"{block_len:.2f}", f"{floor_area_sqm:.2f}", "-"],
                        "מחיר יחידה (₪)": [c_price, b_price, f_price, "-"],
                        "סה\"כ (₪)": [f"{conc_len*c_price:.2f}", f"{block_len*b_price:.2f}", f"{floor_area_sqm*f_price:.2f}", f"{total_cost_calc:.2f}"]
                    }
                    df_quote = pd.DataFrame(quote_data)
                    csv = df_quote.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        "📥 הורד הצעת מחיר (Excel/CSV)",
                        data=csv,
                        file_name=f"quote_{p_name}.csv",
                        mime="text/csv",
                        type="primary"
                    )

                if st.button("💾 שמור נתונים ל-DB", type="primary", use_container_width=True):
                    proj["metadata"]["plan_name"] = p_name
                    proj["metadata"]["scale"] = p_scale
                    metadata_json = json.dumps(proj["metadata"], ensure_ascii=False)
                    materials = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    save_plan(selected, p_name, p_scale, scale_val, proj["raw_pixels"], metadata_json, target_date_str, budget_limit_val, cost_per_meter_val, json.dumps(materials, ensure_ascii=False))
                    st.success("נשמר!")

            with col_preview:
                st.markdown("### 👁️ ניתוח ויזואלי")
                
                # אפשרות להציג/להסתיר ריצוף
                show_floor = st.checkbox("הצג שכבת ריצוף (סגול)", value=True)
                f_mask_to_show = proj["flooring_mask"] if show_floor else None
                
                # תצוגה צבעונית
                if "concrete_mask" in proj and "blocks_mask" in proj:
                    colored_img = create_colored_overlay(proj["original"], proj["concrete_mask"], proj["blocks_mask"], f_mask_to_show)
                    st.image(colored_img, caption="🔵 כחול=בטון | 🟠 כתום=בלוקים | 🟣 סגול=ריצוף", use_column_width=True)
                else:
                    st.image(proj["skeleton"], caption="זיהוי קירות", use_column_width=True)
                
                # גרף חלוקה
                chart_data = pd.DataFrame(
                    [[conc_len, block_len, floor_area_sqm]], 
                    columns=["בטון", "בלוקים", "ריצוף"]
                )
                st.bar_chart(chart_data, color=["#1E90FF", "#FFA500", "#C864FF"])
                
                if proj["total_length"] > 0:
                    mats = calculate_material_estimates(proj["total_length"], st.session_state.wall_height)
                    st.markdown("###### הערכה מהירה")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['block_count']:,}</div><div class='mat-lbl'>בלוקים</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['cement_cubic_meters']:.1f}</div><div class='mat-lbl'>מ\"ק מלט</div></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='mat-card'><div class='mat-val'>{mats['wall_area_sqm']:.0f}</div><div class='mat-lbl'>מ\"ר קיר</div></div>", unsafe_allow_html=True)

    with tab2:
        # דשבורד מנהלים מלא
        all_plans_db = get_all_plans()
        
        if not all_plans_db:
            st.info("אין נתונים במסד הנתונים.")
        else:
            plan_options = [f"{p['plan_name']} (ID: {p['id']})" for p in all_plans_db]
            selected_display = st.selectbox("בחר פרויקט לצפייה בנתונים:", plan_options)
            
            selected_id = int(selected_display.split("(ID: ")[1].split(")")[0])
            forecast = get_project_forecast(selected_id)
            fin = get_project_financial_status(selected_id)
            
            days_val = forecast['days_to_finish']
            days_left_display = days_val if days_val > 0 else "-"

            st.markdown("#### 📊 סטטוס ביצוע")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1: st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">🏗️</div><div class="kpi-label">בוצע בפועל</div><div class="kpi-value">{forecast['cumulative_progress']:.1f} מ'</div><div class="kpi-sub">מתוך {forecast['total_planned']:.1f} מ'</div></div>""", unsafe_allow_html=True)
            with kpi2:
                pct = (forecast['cumulative_progress'] / forecast['total_planned'] * 100) if forecast['total_planned'] > 0 else 0
                st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">📈</div><div class="kpi-label">אחוז השלמה</div><div class="kpi-value">{pct:.1f}%</div><div class="kpi-sub">נותרו {forecast['remaining_work']:.1f} מ'</div></div>""", unsafe_allow_html=True)
            with kpi3: st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">📅</div><div class="kpi-label">ימים לסיום</div><div class="kpi-value">{days_left_display}</div><div class="kpi-sub">קצב: {forecast['average_velocity']:.1f} מ'/יום</div></div>""", unsafe_allow_html=True)
            with kpi4:
                cost_color = "#ef4444" if fin['budget_variance'] < 0 else "#10b981"
                st.markdown(f"""<div class="kpi-container"><div class="kpi-icon">💰</div><div class="kpi-label">עלות נוכחית</div><div class="kpi-value">{fin['current_cost']:,.0f} ₪</div><div class="kpi-sub" style="color: {cost_color}">תקציב: {fin['budget_limit']:,.0f} ₪</div></div>""", unsafe_allow_html=True)
            
            # === ייצוא PDF ===
            st.markdown("---")
            if st.button("📄 צור דוח PDF למנהל"):
                found_proj = None
                selected_name_clean = selected_display.split(" (ID")[0]
                for pname, pdata in st.session_state.projects.items():
                    if pdata["metadata"].get("plan_name") == selected_name_clean or pname.replace(".pdf","") == selected_name_clean:
                        found_proj = pdata
                        break
                if found_proj:
                    stats = {
                        "built": forecast['cumulative_progress'],
                        "total": forecast['total_planned'],
                        "percent": pct
                    }
                    try:
                        pdf_bytes = generate_status_pdf(found_proj["metadata"].get("plan_name", "Report"), found_proj["original"], stats)
                        st.download_button(label="📥 הורד קובץ PDF", data=pdf_bytes, file_name=f"report_{selected_id}.pdf", mime="application/pdf")
                    except Exception as e: st.error(f"שגיאה ביצירת PDF: {e}")
                else: st.warning("יש לטעון את הקובץ המקורי לזיכרון כדי לייצר PDF.")

            g_col, t_col = st.columns([2, 1])
            with g_col:
                st.markdown("##### קצב התקדמות")
                df = load_stats_df()
                if not df.empty: st.bar_chart(df, x="תאריך", y="כמות שבוצעה", use_container_width=True)
                else: st.info("אין נתונים להצגה")
            with t_col:
                st.markdown("##### דיווחים אחרונים")
                if not df.empty: st.dataframe(df[["תאריך", "כמות שבוצעה", "הערה"]].head(5), hide_index=True, use_container_width=True)
                else: st.caption("אין דיווחים אחרונים")

# --- דיווח שטח (משודרג וחכם) ---
elif mode == "👷 דיווח שטח":
    st.title("דיווח ביצוע")
    if not st.session_state.projects: 
        st.info("אין תוכניות זמינות. עבור למנהל פרויקט להעלאת תוכניות.")
    else:
        plan_name = st.selectbox("בחר תוכנית:", list(st.session_state.projects.keys()))
        proj = st.session_state.projects[plan_name]
        
        # --- פיצ'ר חדש: בחירת סוג דיווח ---
        st.markdown("### מה ביצעת היום?")
        report_type = st.radio("סוג עבודה:", ["🧱 בניית קירות", "🔲 ריצוף/חיפוי"], horizontal=True)
        
        orig_rgb = cv2.cvtColor(proj["original"], cv2.COLOR_BGR2RGB)
        h, w = orig_rgb.shape[:2]
        
        # הגדרת רקע וצבעים בהתאם לסוג הדיווח
        if report_type == "🧱 בניית קירות":
            # הדגשת קירות
            thick_walls = cv2.resize(proj["thick_walls"], (w, h), interpolation=cv2.INTER_NEAREST)
            kernel = np.ones((10, 10), np.uint8)
            highlight_mask = cv2.dilate((thick_walls > 0).astype(np.uint8) * 255, kernel, iterations=1)
            base_color = [0, 120, 255] # כחול
            draw_color = "#00FF00" # ירוק לסימון
            stroke_w = 8
            drawing_mode = "freedraw"
            msg = "סמן קירות שבוצעו (פס ירוק)"
            
        else: # ריצוף
            # הדגשת ריצוף
            floor_mask = cv2.resize(proj["flooring_mask"], (w, h), interpolation=cv2.INTER_NEAREST)
            highlight_mask = floor_mask
            base_color = [200, 100, 255] # סגול
            draw_color = "#FFFF00" # צהוב לסימון שטח
            stroke_w = 20 # מברשת עבה לשטח
            drawing_mode = "freedraw"
            msg = "צבע את האזור שרוצף (בצהוב)"
        
        # יצירת תמונה לרקע
        overlay = np.zeros_like(orig_rgb)
        overlay[highlight_mask > 0] = base_color
        combined = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0).astype(np.uint8)
        bg_image = Image.fromarray(combined)
        
        # התאמת גודל לקנבס
        max_canvas_width = 800
        if w > max_canvas_width:
            factor = max_canvas_width / w
            c_width = max_canvas_width
            c_height = int(h * factor)
        else:
            c_width = w
            c_height = h
            factor = 1.0
            
        bg_image_resized = bg_image.resize((c_width, c_height), Image.Resampling.LANCZOS)
        
        st.caption(msg)
        canvas_key = f"rep_{plan_name}_{report_type}"
        
        canvas = st_canvas(
            fill_color="rgba(255, 255, 0, 0.3)" if report_type == "🔲 ריצוף/חיפוי" else "rgba(0,0,0,0)",
            stroke_width=stroke_w,
            stroke_color=draw_color, 
            background_image=bg_image_resized,
            height=c_height,
            width=c_width,
            drawing_mode=drawing_mode,
            key=canvas_key, 
            update_streamlit=True
        )
        
        if canvas.json_data and canvas.json_data["objects"]:
            measured_value = 0
            unit = ""
            
            # --- חישוב לדיווח קירות (אורך) ---
            if report_type == "🧱 בניית קירות":
                try:
                    w_mask = np.zeros((c_height, c_width), dtype=np.uint8)
                    df_obj = pd.json_normalize(canvas.json_data["objects"])
                    for _, obj in df_obj.iterrows():
                        if 'path' in obj and isinstance(obj['path'], list):
                            points = []
                            for p in obj['path']:
                                if len(p) >= 3: points.append([int(p[1]), int(p[2])])
                            if len(points) > 1:
                                cv2.polylines(w_mask, [np.array(points, dtype=np.int32)], False, 255, 8)
                    
                    # חיתוך עם השלד המקורי
                    walls_res = cv2.resize(proj["thick_walls"], (c_width, c_height), interpolation=cv2.INTER_NEAREST)
                    # ניפוח קל כדי שהסימון יתפוס
                    walls_res = cv2.dilate(walls_res, np.ones((5,5), np.uint8))
                    
                    intersection = cv2.bitwise_and(w_mask, walls_res)
                    pixels = cv2.countNonZero(intersection)
                    
                    if proj["scale"] > 0:
                        measured_value = (pixels / factor) / proj["scale"]
                    unit = "מטר אורך"
                    
                except Exception as e:
                    st.error(f"שגיאה בחישוב: {e}")

            # --- חישוב לדיווח ריצוף (שטח) ---
            else:
                if canvas.image_data is not None:
                    # ספירת פיקסלים שהמשתמש צייר (ערוץ Alpha > 0)
                    user_drawn = canvas.image_data[:, :, 3] > 0
                    pixel_count = np.count_nonzero(user_drawn)
                    
                    # המרה למ"ר: פיקסלים חלקי (סקייל * פקטור)^2
                    real_scale_px_per_meter = proj["scale"] * factor
                    measured_value = pixel_count / (real_scale_px_per_meter ** 2)
                    unit = "מ\"ר"

            # הצגת תוצאה ושליחה
            if measured_value > 0:
                st.success(f"✅ כמות מחושבת: **{measured_value:.2f} {unit}**")
                note = st.text_input("הערה לדיווח", value=f"דיווח {report_type}")
                
                if st.button("🚀 שלח דיווח ליומן"):
                    # שמירת התוכנית ל-DB אם לא קיימת
                    rec = get_plan_by_filename(plan_name)
                    if rec: pid = rec['id']
                    else:
                        pid = save_plan(plan_name, proj["metadata"].get("plan_name", plan_name), "1:50", proj["scale"], proj["raw_pixels"], json.dumps(proj["metadata"], ensure_ascii=False))
                    
                    # שמירה (הערך נשמר בשדה meters_built, ההערה תפרט את הסוג)
                    full_note = f"{note} ({measured_value:.2f} {unit})"
                    save_progress_report(pid, measured_value, full_note)
                    st.balloons()
                    st.success("הדיווח נקלט בהצלחה!")
            else:
                st.info(f"נא לסמן על גבי השרטוט את ה{report_type} שבוצע.")