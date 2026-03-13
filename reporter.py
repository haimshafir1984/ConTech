from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import io
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

# ==========================================
# פונקציה 1: דוח סטטוס פרויקט (קיים)
# ==========================================

def generate_status_pdf(plan_name, original_img_rgb, stats):
    """
    יוצר דוח PDF מפורט עם תמונת סטטוס ונתונים
    
    Args:
        plan_name: שם התוכנית
        original_img_rgb: תמונה עם סימון ירוק של מה שבוצע
        stats: מילון עם: built, total, percent, remaining, cost, budget
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)
    
    # === כותרת ראשית ===
    c.setFont("Helvetica-Bold", 28)
    c.drawString(50, height - 50, "ConTech Pro - Project Status Report")
    
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 75, f"Project: {plan_name}")
    c.drawString(50, height - 95, f"Report Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # === קו הפרדה ===
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.setLineWidth(2)
    c.line(50, height - 105, width - 50, height - 105)
    
    # === סטטיסטיקות ===
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, height - 135, "Progress Summary")
    
    c.setFont("Helvetica", 12)
    
    # שורה 1: התקדמות
    y_pos = height - 160
    c.setFillColorRGB(0.2, 0.6, 0.2)  # ירוק
    c.drawString(50, y_pos, f"Completed: {stats.get('built', 0):.1f} m")
    
    c.setFillColorRGB(0, 0, 0)
    c.drawString(200, y_pos, f"Total Scope: {stats.get('total', 0):.1f} m")
    
    c.setFillColorRGB(0.8, 0.4, 0)  # כתום
    c.drawString(350, y_pos, f"Remaining: {stats.get('remaining', 0):.1f} m")
    
    # שורה 2: אחוז התקדמות
    y_pos -= 25
    percent = stats.get('percent', 0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, f"Progress: {percent:.1f}%")
    
    # פס התקדמות ויזואלי
    bar_x = 180
    bar_y = y_pos - 5
    bar_width = 300
    bar_height = 20
    
    # רקע (אפור)
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)
    
    # התקדמות (ירוק)
    progress_width = (percent / 100) * bar_width
    c.setFillColorRGB(0.2, 0.8, 0.2)
    c.rect(bar_x, bar_y, progress_width, bar_height, fill=1, stroke=0)
    
    # מסגרת
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.setLineWidth(1)
    c.rect(bar_x, bar_y, bar_width, bar_height, fill=0, stroke=1)
    
    # שורה 3: תקציב
    y_pos -= 40
    c.setFont("Helvetica", 12)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, y_pos, f"Current Cost: {stats.get('cost', 0):,.0f} ILS")
    c.drawString(250, y_pos, f"Budget: {stats.get('budget', 0):,.0f} ILS")
    
    budget_variance = stats.get('budget', 0) - stats.get('cost', 0)
    if budget_variance >= 0:
        c.setFillColorRGB(0.2, 0.6, 0.2)
        c.drawString(450, y_pos, f"Under Budget: {budget_variance:,.0f} ILS")
    else:
        c.setFillColorRGB(0.8, 0.2, 0.2)
        c.drawString(450, y_pos, f"Over Budget: {abs(budget_variance):,.0f} ILS")
    
    # === קו הפרדה נוסף ===
    y_pos -= 20
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.setLineWidth(1)
    c.line(50, y_pos, width - 50, y_pos)
    
    # === תמונה ===
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
        
        # מקסימום שטח לתמונה
        display_width = width - 100
        display_height = y_pos - 60  # השאר מקום למטה
        
        # התאמה לפרופורציות
        calc_height = display_width * aspect
        if calc_height > display_height:
            calc_height = display_height
            display_width = calc_height / aspect
        else:
            display_height = calc_height
        
        # ציור התמונה
        img_y = 50  # מהתחתית
        c.drawImage(img_reader, 50, img_y, width=display_width, height=display_height)
        
        # מקרא
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.2, 0.8, 0.2)
        c.drawString(50, img_y + display_height + 10, "Green = Completed Work")
        
    except Exception as e:
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.8, 0.2, 0.2)
        c.drawString(50, height/2, f"Image Error: {str(e)}")
    
    # === Footer ===
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 20, f"Generated by ConTech Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawRightString(width - 50, 20, "Page 1 of 1")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer


# ==========================================
# פונקציה 2: חשבון חלקי לקבלן (חדש)
# ==========================================

def generate_payment_invoice_pdf(invoice_data, contractor_info):
    """
    יוצר חשבון חלקי מפורט לקבלן
    
    Args:
        invoice_data: מילון מ-get_payment_invoice_data()
        contractor_info: מילון עם פרטי קבלן {name, company, vat_id, address}
    
    Returns:
        BytesIO buffer עם ה-PDF
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # === כותרת ראשית ===
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 50, "Payment Invoice")
    c.drawCentredString(width/2, height - 75, "חשבון חלקי")
    
    # === קו הפרדה ===
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(2)
    c.line(50, height - 90, width - 50, height - 90)
    
    # === פרטי חשבון ===
    y_pos = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, f"Invoice Date:")
    c.setFont("Helvetica", 12)
    c.drawString(150, y_pos, datetime.now().strftime("%d/%m/%Y"))
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(350, y_pos, f"Invoice #:")
    c.setFont("Helvetica", 12)
    invoice_number = f"INV-{datetime.now().strftime('%Y%m%d')}-{invoice_data['plan']['id']}"
    c.drawString(430, y_pos, invoice_number)
    
    # === פרטי פרויקט ===
    y_pos -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Project Details:")
    
    y_pos -= 20
    c.setFont("Helvetica", 11)
    c.drawString(60, y_pos, f"Project Name: {invoice_data['plan']['plan_name']}")
    
    y_pos -= 18
    c.drawString(60, y_pos, f"Period: {invoice_data['start_date']} → {invoice_data['end_date']}")
    
    # === פרטי קבלן ===
    y_pos -= 35
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Contractor Details:")
    
    y_pos -= 20
    c.setFont("Helvetica", 11)
    c.drawString(60, y_pos, f"Name: {contractor_info.get('name', 'N/A')}")
    
    y_pos -= 18
    c.drawString(60, y_pos, f"Company: {contractor_info.get('company', 'N/A')}")
    
    y_pos -= 18
    c.drawString(60, y_pos, f"VAT ID: {contractor_info.get('vat_id', 'N/A')}")
    
    if contractor_info.get('address'):
        y_pos -= 18
        c.drawString(60, y_pos, f"Address: {contractor_info['address']}")
    
    # === קו הפרדה ===
    y_pos -= 25
    c.setLineWidth(1)
    c.line(50, y_pos, width - 50, y_pos)
    
    # === טבלת פריטים ===
    y_pos -= 35
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Work Items:")
    
    y_pos -= 10
    
    # יצירת טבלה
    if invoice_data['items']:
        table_data = [
            ['#', 'Work Type', 'Quantity', 'Unit', 'Unit Price (ILS)', 'Subtotal (ILS)']
        ]
        
        for idx, item in enumerate(invoice_data['items'], 1):
            table_data.append([
                str(idx),
                item['work_type'],
                f"{item['quantity']:.2f}",
                item['unit'],
                f"{item['unit_price']:,.0f}",
                f"{item['subtotal']:,.2f}"
            ])
        
        # יצירת אובייקט Table
        table = Table(table_data, colWidths=[30, 140, 70, 50, 100, 100])
        
        # סגנון טבלה
        table.setStyle(TableStyle([
            # כותרות
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # תוכן
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),  # מספרים מיושרים ימינה
        ]))
        
        # ציור הטבלה
        table.wrapOn(c, width, height)
        table.drawOn(c, 50, y_pos - len(table_data)*25 - 40)
        
        y_pos = y_pos - len(table_data)*25 - 60
    
    # === סיכום תשלום ===
    y_pos -= 30
    c.setLineWidth(1)
    c.line(width - 250, y_pos, width - 50, y_pos)
    
    y_pos -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 150, y_pos, "Subtotal:")
    c.setFont("Helvetica", 12)
    c.drawRightString(width - 50, y_pos, f"{invoice_data['total_amount']:,.2f} ILS")
    
    y_pos -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 150, y_pos, "VAT (17%):")
    c.setFont("Helvetica", 12)
    c.drawRightString(width - 50, y_pos, f"{invoice_data['vat']:,.2f} ILS")
    
    y_pos -= 5
    c.setLineWidth(2)
    c.line(width - 250, y_pos, width - 50, y_pos)
    
    y_pos -= 25
    c.setFont("Helvetica-Bold", 14)
    c.drawRightString(width - 150, y_pos, "TOTAL:")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0, 0.5, 0)
    c.drawRightString(width - 50, y_pos, f"{invoice_data['total_with_vat']:,.2f} ILS")
    
    # === הערות ===
    c.setFillColorRGB(0, 0, 0)
    y_pos -= 50
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "Payment Terms:")
    y_pos -= 15
    c.setFont("Helvetica", 9)
    c.drawString(60, y_pos, "• Payment due within 30 days from invoice date")
    y_pos -= 12
    c.drawString(60, y_pos, "• Bank transfer to account specified in contract")
    y_pos -= 12
    c.drawString(60, y_pos, "• Please reference invoice number in payment description")
    
    # === חתימות ===
    y_pos -= 40
    c.setLineWidth(1)
    
    # חתימת קבלן
    c.line(70, y_pos, 220, y_pos)
    c.setFont("Helvetica", 9)
    c.drawString(70, y_pos - 15, "Contractor Signature")
    
    # חתימת מנהל
    c.line(width - 220, y_pos, width - 70, y_pos)
    c.drawString(width - 180, y_pos - 15, "Project Manager Signature")
    
    # === Footer ===
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 30, f"Generated by ConTech Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawRightString(width - 50, 30, f"Invoice #{invoice_number}")
    
    # תיבת אזהרה בתחתית
    c.setStrokeColorRGB(0.8, 0, 0)
    c.setLineWidth(1)
    c.rect(50, 50, width - 100, 30, stroke=1, fill=0)
    c.setFillColorRGB(0.8, 0, 0)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(width/2, 62, "This invoice is valid only with authorized signatures")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer
