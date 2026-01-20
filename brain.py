import os
import base64
import json
import re
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


def _sanitize_json(text: str) -> str:
    """
    מנקה JSON שבור - מסיר אינדקסים, מתקן NULL, ותווים לא חוקיים
    
    Args:
        text: טקסט JSON פגום
    
    Returns:
        JSON מנוקה
    """
    # שלב 0: תיקון newlines בתוך strings - החלפה ב-space או \\n
    # מוצא strings עם newline בפנים ומחליף ב-רווח
    lines = text.split('\n')
    fixed_lines = []
    in_string = False
    current_line = ""
    
    for line in lines:
        # ספירת מרכאות (זוגי = לא בתוך string, אי-זוגי = בתוך string)
        quote_count = line.count('"') - line.count('\\"')  # לא כולל מרכאות escaped
        
        if in_string:
            # אנחנו באמצע string שחוצה שורות
            current_line += " " + line.strip()
            if quote_count % 2 == 1:  # יצאנו מה-string
                in_string = False
                fixed_lines.append(current_line)
                current_line = ""
        else:
            if quote_count % 2 == 1:  # נכנסנו ל-string שלא נסגר
                in_string = True
                current_line = line
            else:
                fixed_lines.append(line)
    
    text = '\n'.join(fixed_lines)
    
    # שלב 1: הסרת אינדקסים במערכים: 0:"text" -> "text"
    text = re.sub(r'(?m)^\s*\d+\s*:\s*', '', text)
    
    # שלב 2: החלפת NULL ב-null
    text = re.sub(r'\bNULL\b', 'null', text)
    
    # שלב 3: הוספת פסיקים חסרים - גישה כוללת
    # 3.1: בין string לstring: "x"\n" -> "x",\n"
    text = re.sub(r'"(\s*\n\s*)"', r'",\1"', text)
    
    # 3.2: בין string למספר/bool/null: "x"\n5 -> "x",\n5
    text = re.sub(r'"(\s*\n\s*)(true|false|null|\d+)', r'",\1\2', text)
    
    # 3.3: בין מספר/bool/null לstring: 5\n" -> 5,\n"
    text = re.sub(r'(true|false|null|\d+)(\s*\n\s*)"', r'\1,\2"', text)
    
    # 3.4: בין ] לstring או למספר: ]\n" -> ],\n"
    text = re.sub(r'](\s*\n\s*)(["{0-9tfn])', r'],\1\2', text)
    
    # 3.5: בין } לstring או למספר: }\n" -> },\n"
    text = re.sub(r'}(\s*\n\s*)(["{0-9tfn])', r'},\1\2', text)
    
    # שלב 4: הסרת פסיקים מיותרים לפני } או ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text


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
Analyze construction plan text and extract metadata.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no explanations, no markdown, no comments
2. Use null (lowercase) for missing values, NEVER NULL
3. Do NOT add list indices like "0:" or "1:" inside arrays
4. Use double quotes for all keys and string values
5. No pretty-printing with numbered indices

Input text:
{raw_text[:2000]}

Required JSON format:
{{
    "plan_name": "string or null",
    "scale": "string like 1:50 or null",
    "plan_type": "construction/demolition/flooring/ceiling/electrical/other"
}}

Return ONLY the JSON object, nothing else.
"""

    last_error = None
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # ניקוי markdown אם יש
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # חילוץ JSON בלבד
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            # ניקוי JSON
            cleaned = _sanitize_json(response_text)
            
            # ניסיון לפרסר
            try:
                result = json.loads(cleaned)
                result["_model_used"] = model
                return result
            except json.JSONDecodeError as json_err:
                last_error = f"JSON Error with {model}: {str(json_err)[:100]}"
                continue
                
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            
            # אם המודל לא נמצא, נסה את הבא
            if "not_found_error" in error_str or "404" in error_str:
                continue
            else:
                continue
    
    # כל המודלים נכשלו
    return {"error": f"All models failed. Last: {last_error}"}


def analyze_legend_image(image_bytes):
    """
    מנתח תמונה של מקרא תוכנית בניה ומזהה סוג תוכנית וחומרים
    """
    client, error = get_anthropic_client()
    if error: return {"error": error}

    # רשימת מודלים לניסיון
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

**CRITICAL - JSON FORMAT RULES:**
1. Return ONLY valid JSON - no markdown, no explanations
2. Use null (lowercase) for missing values, NEVER NULL
3. Do NOT add indices like "0:" or "1:" in arrays
4. All keys and strings must use double quotes
5. No trailing commas before ] or }

**Required JSON structure:**
{
    "plan_type": "תקרה|קירות|ריצוף|אחר",
    "confidence": 0-100,
    "materials_found": ["material1", "material2"],
    "ceiling_types": [
        {
            "code": "string",
            "description": "string",
            "dimensions": "string"
        }
    ],
    "symbols": [
        {"symbol": "C11", "meaning": "description"}
    ],
    "notes": "string or null",
    "legend_title": "string or null"
}

**Examples of CORRECT output:**
✅ {"plan_type": "תקרה", "confidence": 98, "materials_found": ["גבס"], "ceiling_types": [], "symbols": [], "notes": null, "legend_title": "מקרא תקרה"}

**Examples of WRONG output:**
❌ {"plan_type": "אחר", "confidence": 80} ← Missing required fields!
❌ {0: {"symbol": "C11"}} ← Never use numeric indices!
❌ {"notes": NULL} ← Use null not NULL!

Return ONLY the JSON object, nothing else.
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
            
            # ניקוי markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # חילוץ JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            # ניקוי JSON
            cleaned = _sanitize_json(response_text)
            
            # פרסור
            try:
                result = json.loads(cleaned)
                
                # ולידציה בסיסית
                required_fields = ["plan_type", "confidence", "materials_found", "ceiling_types", "symbols"]
                if not all(field in result for field in required_fields):
                    # נסה למלא שדות חסרים
                    for field in required_fields:
                        if field not in result:
                            if field == "confidence":
                                result[field] = 50
                            elif field in ["materials_found", "ceiling_types", "symbols"]:
                                result[field] = []
                            else:
                                result[field] = "אחר"
                
                result["_model_used"] = model
                return result
                
            except json.JSONDecodeError as json_err:
                last_error = f"JSON Error with {model}: {str(json_err)[:100]} | Text: {cleaned[:200]}"
                continue
            
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            
            if "not_found_error" in error_str or "404" in error_str:
                continue
            else:
                continue
    
    return {
        "error": f"כל המודלים נכשלו. שגיאה אחרונה: {last_error}",
        "tried_models": models
    }
