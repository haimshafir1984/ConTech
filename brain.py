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
            pass
    
    if not api_key:
        return None, "חסר מפתח API"
        
    return anthropic.Anthropic(api_key=api_key), None


def process_plan_metadata(raw_text):
    """
    ✨ משולב: מחלץ מטא-דאטה מלאה עם פרומפט מקיף
    """
    client, error = get_anthropic_client()
    if error: 
        return {
            "status": "no_api_key",
            "error": error,
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": [error],
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
        }

    # רשימת מודלים לניסיון (מהחדש לישן)
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

    # ===== הפרומפט המלא =====
    prompt = f"""
אתה מומחה בחילוץ מידע מתוכניות בניה ישראליות.
המשימה: לחלץ **כל** המידע הזמין מהטקסט ולארגן אותו ב-JSON מובנה.

**חשוב מאוד:**
- החזר **רק** JSON תקין, ללא טקסט נוסף
- ודא שאין פסיקים מיותרים לפני ] או }}
- חלץ **כל** מידע זמין, במיוחד **מידות חדרים** ו**שטחים**
- אם יש טקסט חוזר או OCR לא מושלם - נסה להבין את הכוונה

**טקסט מהתוכנית:**
{raw_text[:3500]}

**מבנה JSON נדרש:**

{{
  "document": {{
    "plan_title": {{"value": "שם התוכנית", "confidence": 0-100, "evidence": ["ציטוט"]}},
    "plan_type": {{"value": "קירות/תקרה/ריצוף/חשמל", "confidence": 0-100, "evidence": []}},
    "scale": {{"value": "1:50", "confidence": 0-100, "evidence": []}},
    "date": {{"value": "2024-01-15", "confidence": 0-100, "evidence": []}},
    "floor_or_level": {{"value": "קומה א'", "confidence": 0-100, "evidence": []}},
    "project_name": {{"value": null, "confidence": 0, "evidence": []}},
    "project_address": {{"value": null, "confidence": 0, "evidence": []}},
    "architect_name": {{"value": null, "confidence": 0, "evidence": []}},
    "drawing_number": {{"value": null, "confidence": 0, "evidence": []}}
  }},
  "rooms": [
    {{
      "name": {{"value": "חדר שינה 1", "confidence": 95, "evidence": ["חדר שינה 1"]}},
      "area_m2": {{"value": 15.5, "confidence": 90, "evidence": ["15.5 מ\\"ר"]}},
      "ceiling_height_m": {{"value": 2.70, "confidence": 85, "evidence": ["H=2.70"]}},
      "flooring_notes": {{"value": "פרקט", "confidence": 80, "evidence": ["פרקט"]}},
      "ceiling_notes": {{"value": null, "confidence": 0, "evidence": []}},
      "other_notes": {{"value": null, "confidence": 0, "evidence": []}}
    }}
  ],
  "heights_and_levels": {{
    "default_ceiling_height_m": {{"value": 2.80, "confidence": 70, "evidence": ["H=2.80"]}},
    "default_floor_height_m": {{"value": null, "confidence": 0, "evidence": []}},
    "construction_level_m": {{"value": null, "confidence": 0, "evidence": []}}
  }},
  "execution_notes": {{
    "general_notes": {{"value": null, "confidence": 0, "evidence": []}},
    "structural_notes": {{"value": null, "confidence": 0, "evidence": []}},
    "hvac_notes": {{"value": null, "confidence": 0, "evidence": []}},
    "electrical_notes": {{"value": null, "confidence": 0, "evidence": []}},
    "plumbing_notes": {{"value": null, "confidence": 0, "evidence": []}}
  }},
  "limitations": ["רשום כאן בעיות/מגבלות אם יש"],
  "quantities_hint": {{
    "wall_types_mentioned": ["קיר בטון 20 ס\\"מ"],
    "material_hints": ["גרניט פורצלן"]
  }}
}}

**חיפוש חדרים:**
- שמות: "חדר שינה", "סלון", "מטבח", "שירותים"
- שטחים: "15 מ\\"ר", "15.5 m²", "15 sqm", או מספר ליד שם חדר
- גבהים: "H=2.80", "גובה 2.70", "ceiling height 2.80m"
- ריצוף: "קרמיקה", "פרקט", "שיש", "גרניט"
- תקרה: "גבס", "טרוול", "תקרה אקוסטית"

**התחל - החזר רק JSON:**
"""

    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=6000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # ניקוי
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # חילוץ JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            # פרסור
            try:
                result = json.loads(response_text)
                result["status"] = "success"
                result["_model_used"] = model
                return result
            except json.JSONDecodeError:
                # תיקון אוטומטי
                fixed = response_text.replace(",]", "]").replace(",}", "}")
                result = json.loads(fixed)
                result["status"] = "success"
                result["_model_used"] = model
                result["_auto_fixed"] = True
                return result
            
        except Exception as e:
            if "not_found_error" in str(e) or "404" in str(e):
                continue
            continue
    
    # כשלון בכל המודלים
    return {
        "status": "extraction_failed",
        "error": "כל המודלים נכשלו",
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": ["Failed to extract data with all models"],
        "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
    }


def analyze_legend_image(image_bytes):
    """
    ✨ משופר: מנתח תמונה של מקרא תוכנית בניה
    עם few-shot learning
    """
    client, error = get_anthropic_client()
    if error: return {"error": error}

    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
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
    "ceiling_types": [{"code": "E Advantage", "description": "תקרה חצי שקועה", "dimensions": "60X60"}],
    "symbols": [{"symbol": "H=2.80", "meaning": "גובה תקרה 2.80 מטר"}],
    "notes": "תכנית תקרה קומה ב'"
}
```

**דוגמה 2 - קירות:**
```
תמונה: מקרא עם "קיר בטון", "קיר בלוקים", "C11, C12"

תשובה נכונה:
{
    "plan_type": "קירות",
    "confidence": 95,
    "legend_title": "מקרא קירות",
    "materials_found": ["בטון", "בלוקים"],
    "symbols": [{"symbol": "C11", "meaning": "קורה סוג 11"}],
    "notes": "תכנית קירות וחלוקה"
}
```

**דוגמה 3 - ריצוף:**
```
תמונה: מקרא עם "גרניט פורצלן 60X60", "מפלס גמר"

תשובה נכונה:
{
    "plan_type": "ריצוף",
    "confidence": 92,
    "legend_title": "מקרא ריצוף",
    "materials_found": ["גרניט פורצלן", "קרמיקה"],
    "symbols": [{"symbol": "F.F.L", "meaning": "Finished Floor Level"}],
    "notes": "תכנית ריצוף וגמרים"
}
```

עכשיו נתח את התמונה הזו והחזר JSON בלבד.
"""

    last_error = None
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded_image}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            
            response_text = message.content[0].text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            try:
                result = json.loads(response_text)
                result["_model_used"] = model
                result["_method"] = "few_shot_learning"
                return result
            except json.JSONDecodeError:
                fixed_text = response_text.replace(",]", "]").replace(",}", "}")
                result = json.loads(fixed_text)
                result["_model_used"] = model
                result["_auto_fixed"] = True
                result["_method"] = "few_shot_learning"
                return result
            
        except Exception as e:
            last_error = str(e)
            if "not_found_error" in last_error or "404" in last_error:
                continue
            continue
    
    return {
        "error": f"כל המודלים נכשלו. שגיאה: {last_error}",
        "tried_models": models
    }
