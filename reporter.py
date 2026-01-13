from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from datetime import datetime

def generate_status_pdf(plan_name, original_img_rgb, overlay_img, stats):
    """
    יוצר דוח PDF עם תמונת המצב הנוכחית ונתונים
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)
    
    # 1. כותרות (באנגלית כרגע למניעת בעיות פונט בענן)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, f"Project Status Report: {plan_name}")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 2. נתונים סטטיסטיים
    c.drawString(50, height - 100, f"Completed: {stats['built']:.2f} meters")
    c.drawString(250, height - 100, f"Total Planned: {stats['total']:.2f} meters")
    c.drawString(450, height - 100, f"Progress: {stats['percent']:.1f}%")
    
    # 3. שילוב התמונות (בניית תמונה ויזואלית ל-PDF)
    # אנו משתמשים ב-OpenCV כדי לחבר את המקור עם הסימון (כמו באפליקציה)
    import cv2
    import numpy as np
    from PIL import Image
    
    # יצירת שכבה משולבת
    # overlay_img מגיע כשכבה אדומה/ירוקה על רקע שחור
    # אנחנו צריכים להפוך את הרקע השחור לשקוף או לחבר בחוכמה
    
    # המרה ל-PIL לצורך שמירה ב-PDF
    # כאן אנו מניחים שמקבלים כבר תמונה סופית משולבת (Combined) מהאפליקציה
    # או שנבצע את החיבור כאן:
    try:
        # המרה ל-RGB אם צריך
        if len(original_img_rgb.shape) == 2:
            original_img_rgb = cv2.cvtColor(original_img_rgb, cv2.COLOR_GRAY2RGB)
            
        # שילוב פשוט (50% שקיפות)
        # overlay_img צריך להיות באותו גודל
        if original_img_rgb.shape[:2] != overlay_img.shape[:2]:
            overlay_img = cv2.resize(overlay_img, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
            
        combined = cv2.addWeighted(original_img_rgb, 0.7, overlay_img, 0.3, 0)
        
        # המרה לפורמט ש-ReportLab מבין
        img_pil = Image.fromarray(combined)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # ציור התמונה על ה-PDF (תוך שמירה על יחס גובה-רוחב)
        img_reader = ImageReader(img_byte_arr)
        img_w, img_h = img_pil.size
        aspect = img_h / float(img_w)
        
        display_width = width - 100
        display_height = display_width * aspect
        
        # אם זה גבוה מדי, נתאים לפי הגובה
        if display_height > (height - 150):
            display_height = height - 150
            display_width = display_height / aspect
            
        c.drawImage(img_reader, 50, height - 130 - display_height, width=display_width, height=display_height)
        
    except Exception as e:
        c.drawString(50, height/2, f"Error generating image: {str(e)}")

    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer