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
    """מעבד מטא-דאטה של תוכנית עם ניסיון מרובה מודלים"""
    client, error = get_anthropic_client()
    if error: return {}

    # רשימת מודלים לניסיון (מהחדש לישן)
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

    prompt = f"""
    Analyze construction plan text.
    Input: '''{raw_text[:2000]}'''
    Return JSON with: plan_name, scale (e.g. 1:50), plan_type (construction/demolition/other).
    """

    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text
            if "{" in response_text: 
                response_text = "{" + response_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            return json.loads(response_text)
        except Exception as e:
            # אם המודל לא זמין, נסה את הבא
            if "not_found_error" in str(e):
                continue
            else:
                # שגיאה אחרת - עצור
                return {}
    
    return {}

def analyze_legend_image(image_bytes):
    """
    מנתח תמונה של מקרא תוכנית בניה ומזהה סוג תוכנית וחומרים
    מנסה מספר מודלים עד שאחד עובד
    """
    client, error = get_anthropic_client()
    if error: return {"error": error}

    # רשימת מודלים לניסיון (מהחדש לישן)
    models = [
        "claude-3-5-sonnet-20241022",  # הכי חדש
        "claude-3-5-sonnet-20240620",  # גרסה קודמת
        "claude-3-opus-20240229",      # Opus (יקר יותר אבל טוב)
        "claude-3-sonnet-20240229",    # Sonnet ישן
        "claude-3-haiku-20240307"      # Haiku (זול וחלש)
    ]

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
נתח את המקרא (Legend) בתמונה זו של תוכנית בניה.

**שים לב במיוחד:**
1. אם כתוב "מקרא תקרה" → זו תכנית **תקרה**
2. אם כתוב "מקרא קירות" → זו תכנית **קירות**
3. אם כתוב "מקרא ריצוף" → זו תכנית **ריצוף**
4. חפש מילים כמו: "תקרה", "קיר", "ריצוף", "חשמל", "מיזוג"

**חשוב:** קרא את הכותרת של המקרא תחילה!

זהה והחזר JSON בפורמט הבא:
{
    "plan_type": "קירות" או "תקרה" או "ריצוף" או "חשמל" או "מיזוג" או "אינסטלציה" או "הריסה" או "אחר",
    "confidence": 0-100 (רמת ביטחון),
    "materials_found": ["רשימה של חומרים שנמצאו: גבס, מינרלים, בטון, בלוקים, קרמיקה, וכו"],
    "symbols": [
        {"symbol": "סימן או קוד מהמקרא", "meaning": "המשמעות בעברית"},
        ...
    ],
    "notes": "הערות נוספות חשובות"
}

**דוגמאות לזיהוי:**
- אם רואה "תקרה אקוסטית" / "לוחות מינרלים" → plan_type: "תקרה"
- אם רואה "קיר בלוקים" / "קיר בטון" → plan_type: "קירות"
- אם רואה "אריח קרמי" / "גרניט פורצלן" → plan_type: "ריצוף"

**חשוב מאוד:** 
- החזר **רק** JSON, ללא טקסט נוסף
- אם אתה רואה "מקרא תקרה" בכותרת, זו **בוודאות** תכנית תקרה!
- קרא את כל הטקסט בעברית בזהירות
"""

    last_error = None
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/png", 
                                "data": encoded_image
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            
            response_text = message.content[0].text.strip()
            
            # ניקוי התשובה אם יש markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # ניקוי נוסף - חילוץ רק ה-JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            # ניסיון ראשון לפרסור
            try:
                result = json.loads(response_text)
                result["_model_used"] = model
                return result
            except json.JSONDecodeError as json_err:
                # ניסיון לתקן שגיאות נפוצות
                fixed_text = response_text
                fixed_text = fixed_text.replace(",]", "]")  # פסיק מיותר לפני ]
                fixed_text = fixed_text.replace(",}", "}")  # פסיק מיותר לפני }
                
                try:
                    result = json.loads(fixed_text)
                    result["_model_used"] = model
                    result["_auto_fixed"] = True
                    return result
                except:
                    # נכשל - נשמור את השגיאה ונמשיך למודל הבא
                    last_error = f"JSON Error: {str(json_err)} | Response: {response_text[:200]}"
                    continue
            
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            
            # אם המודל לא נמצא (404), נסה את הבא
            if "not_found_error" in error_str or "404" in error_str:
                continue
            
            # שגיאה אחרת - נסה את המודל הבא
            continue
    
    # אם הגענו לכאן - כל המודלים נכשלו
    return {
        "error": f"כל המודלים נכשלו. שגיאה אחרונה: {last_error}",
        "tried_models": models
    }
