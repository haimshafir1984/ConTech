from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

def generate_status_pdf(plan_name, original_img_rgb, stats):
    """
    יוצר דוח PDF עם תמונת המצב הנוכחית ונתונים
    מקבל 3 פרמטרים בלבד (תואם ל-app.py החדש)
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)
    
    # 1. כותרות
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, f"Project Status: {plan_name}")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 75, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 2. נתונים
    built = stats.get('built', 0)
    total = stats.get('total', 1)
    percent = stats.get('percent', 0)
    
    c.drawString(50, height - 110, f"Completed: {built:.2f} m")
    c.drawString(200, height - 110, f"Total Scope: {total:.2f} m")
    c.drawString(400, height - 110, f"Progress: {percent:.1f}%")
    
    # 3. המרת התמונה ל-PDF
    try:
        # המרה ל-RGB בטוחה
        if len(original_img_rgb.shape) == 2:
            img_to_show = cv2.cvtColor(original_img_rgb, cv2.COLOR_GRAY2RGB)
        else:
            img_to_show = original_img_rgb.copy()
            
        # המרה ל-PIL
        img_pil = Image.fromarray(img_to_show)
        
        # שמירה לזיכרון כ-PNG
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_reader = ImageReader(img_byte_arr)
        
        # חישוב פרופורציות
        img_w, img_h = img_pil.size
        aspect = img_h / float(img_w)
        
        display_width = width - 100
        display_height = display_width * aspect
        
        # הגבלת גובה כדי לא לחרוג מהדף
        max_h = height - 150
        if display_height > max_h:
            display_height = max_h
            display_width = display_height / aspect
            
        c.drawImage(img_reader, 50, height - 140 - display_height, width=display_width, height=display_height)
        
    except Exception as e:
        c.drawString(50, height/2, f"Image Error: {str(e)}")

    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer