import os
import base64
import json
try:
    import anthropic
except ImportError:
    anthropic = None
import streamlit as st

def get_anthropic_client():
    """יוצר חיבור ל-Claude בצורה מאובטחת עם עדיפות למשתני סביבה"""
    if anthropic is None:
        return None, "ספריית anthropic חסרה."
    
    # 1. ניסיון למשוך ממשתני סביבה (Render) - הכי חשוב
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # 2. אם אין, ניסיון למשוך מ-secrets (מקומי)
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass # התעלמות אם הקובץ לא קיים
    
    if not api_key:
        return None, "חסר מפתח API"
        
    return anthropic.Anthropic(api_key=api_key), None

def process_plan_metadata(raw_text):
    client, error = get_anthropic_client()
    if error: return {}

    try:
        prompt = f"""
        Analyze construction plan text.
        Input: '''{raw_text[:2000]}'''
        Return JSON with: plan_name, scale (e.g. 1:50), plan_type (construction/demolition/other).
        """
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
        if "{" in response_text: 
            response_text = "{" + response_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
        return json.loads(response_text)
    except:
        return {}

def analyze_legend_image(image_bytes):
    """
    מנתח תמונה של מקרא תוכנית בניה ומזהה סוג תוכנית וחומרים
    """
    client, error = get_anthropic_client()
    if error: return {"error": error}

    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded_image}},
                    {"type": "text", "text": """
נתח את המקרא בתמונה זו של תוכנית בניה.

זהה והחזר JSON בפורמט הבא:
{
    "plan_type": "קירות" או "תקרה" או "ריצוף" או "חשמל" או "מיזוג" או "אינסטלציה" או "הריסה" או "אחר",
    "confidence": 0-100 (רמת ביטחון),
    "materials_found": ["רשימה של חומרים שנמצאו, כמו: בטון, בלוקים, קרמיקה, גבס, וכו"],
    "symbols": [
        {"symbol": "סימן או קוד", "meaning": "המשמעות בעברית"},
        ...
    ],
    "notes": "הערות נוספות חשובות"
}

דוגמאות לסוגי תוכניות:
- "קירות" - תכנית קומה עם קירות, דלתות, חלונות
- "תקרה" - תכנית תקרה עם גבס, תאורה
- "ריצוף" - תכנית ריצוף/חיפוי
- "חשמל" - תכנית חשמל עם נקודות חשמל
- "מיזוג" - מיזוג אוויר
- "הריסה" - תוכנית הריסה

חשוב: החזר **רק** את ה-JSON, ללא טקסט נוסף.
"""}
                ]
            }]
        )
        
        response_text = message.content[0].text.strip()
        
        # ניקוי התשובה אם יש markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # פרסור JSON
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        return {"error": f"שגיאה בפענוח JSON: {str(e)}", "raw_response": response_text}
    except Exception as e:
        return {"error": f"שגיאה: {str(e)}"}