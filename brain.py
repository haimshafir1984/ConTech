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
    
    ✨ משופר: Few-shot learning + דוגמאות
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
    
    # ✨ שיפור: Few-shot learning עם דוגמאות
    prompt = """
אתה מומחה בניתוח תוכניות בניה ישראליות.
נתח את המקרא (Legend) בתמונה זו.

📚 **דוגמאות ללמידה:**

**דוגמה 1 - תקרה:**
```
תמונה: מקרא עם הכותרת "מקרא תקרה - קומה ב'"
תוכן: "E Advantage - תקרה חצי שקועה 60X60", "לוחות מינרלים", "H=2.80"

תשובה נכונה:
{
    "plan_type": "תקרה",
    "confidence": 98,
    "legend_title": "מקרא תקרה - קומה ב'",
    "materials_found": ["לוחות מינרלים", "גבס", "ארקליט"],
    "ceiling_types": [
        {
            "code": "E Advantage",
            "description": "תקרה חצי שקועה",
            "dimensions": "60X60"
        }
    ],
    "symbols": [
        {"symbol": "H=2.80", "meaning": "גובה תקרה 2.80 מטר"}
    ],
    "notes": "תכנית תקרה קומה ב'"
}
```

**דוגמה 2 - קירות:**
```
תמונה: מקרא עם "קיר בטון", "קיר בלוקים", "C11, C12, C13"
תוכן: "קיר בטון 20 ס\"מ", "קיר בלוקים 10 ס\"מ", "D14 - דלת"

תשובה נכונה:
{
    "plan_type": "קירות",
    "confidence": 95,
    "legend_title": "מקרא קירות",
    "materials_found": ["בטון", "בלוקים"],
    "symbols": [
        {"symbol": "C11", "meaning": "קורה סוג 11"},
        {"symbol": "D14", "meaning": "דלת 80 ס\"מ"}
    ],
    "notes": "תכנית קירות וחלוקה"
}
```

**דוגמה 3 - ריצוף:**
```
תמונה: מקרא עם "גרניט פורצלן 60X60", "מפלס גמר", "שיפוע"
תוכן: "אריח קרמי", "גרניט פורצלן", "F.F.L +0.00"

תשובה נכונה:
{
    "plan_type": "ריצוף",
    "confidence": 92,
    "legend_title": "מקרא ריצוף",
    "materials_found": ["גרניט פורצלן", "קרמיקה"],
    "symbols": [
        {"symbol": "F.F.L", "meaning": "Finished Floor Level"}
    ],
    "notes": "תכנית ריצוף וגמרים"
}
```

---

🎯 **עכשיו נתח את התמונה הזו:**

**צעדים לזיהוי:**

1️⃣ **קרא את הכותרת המרכזית במקרא**
   - חפש: "מקרא תקרה" / "מקרא קירות" / "מקרא ריצוף"
   - זו ההוכחה החזקה ביותר לסוג התוכנית!

2️⃣ **חפש מילות מפתח ספציפיות:**
   
   **תקרה →**
   - "תקרה אקוסטית" / "תקרת גבס" / "תקרה פריקה"
   - "לוחות מינרלים" / "ארקליט" 
   - מידות: "60X60" / "60X120" (אריחי תקרה)
   - "תליית תקרות" / "פרופילים נושאים"
   
   **קירות →**
   - "קיר בטון" / "קיר בלוקים" / "קיר קל משקל"
   - "עובי קיר" / "בידוד אקוסטי"
   - סימונים: C11, C12, C13 (קורות)
   
   **ריצוף →**
   - "אריח קרמי" / "גרניט פורצלן" / "פרקט"
   - "מפלס גמר" / "שיפוע"
   - מידות: "30X30" / "60X60" (אריחים)

3️⃣ **בדוק סמלים וקודים:**
   - C11/C12/C13 → קורות (תקרה)
   - D14/D17/D18 → דלתות (קירות)
   - H= → גובה (תקרה/קירות)

**פורמט תשובה - JSON בלבד:**
{
    "plan_type": "תקרה",
    "confidence": 95,
    "materials_found": ["לוחות מינרלים", "גבס", "ארקליט"],
    "ceiling_types": [
        {
            "code": "E Advantage",
            "description": "תקרה חצי שקועה",
            "dimensions": "60X60"
        }
    ],
    "symbols": [
        {"symbol": "C11", "meaning": "קורה סוג 11"},
        {"symbol": "H=2.80", "meaning": "גובה תקרה 2.80 מטר"}
    ],
    "notes": "תכנית תקרה קומה ב'",
    "legend_title": "מקרא תקרה"
}

**חשוב מאוד:**
- אם רואה "מקרא תקרה" → plan_type חייב להיות "תקרה" (ביטחון 98%)
- אם רואה "לוחות מינרלים" → זו בוודאות תקרה
- קרא את כל הטקסט בעברית בקפידה
- החזר **רק** JSON, אין טקסט נוסף
- אם לא בטוח ב-100%, כתב confidence נמוך (60-70)
- השתמש בדוגמאות למעלה כמדריך!

**דוגמאות:**
✅ נכון: {"plan_type": "תקרה", "confidence": 98, "legend_title": "מקרא תקרה"}
❌ שגוי: {"plan_type": "אחר", "confidence": 80}  ← אם יש "מקרא תקרה"!
"""

    last_error = None
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1000,  # ← הגדלתי ל-1000 (יותר מקום לדוגמאות)
                temperature=0.3,  # ← הורדתי temperature לדיוק
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
                result["_method"] = "few_shot_learning"  # ✨ סימון שזו גרסה משופרת
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
                    result["_method"] = "few_shot_learning"
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
        "tried_models": models,
        "_fallback_suggestion": "נסה לחתוך את המקרא ידנית ולנסות שוב"
    }
