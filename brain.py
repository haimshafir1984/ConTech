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
    """יוצר חיבור ל-Claude בצורה מאובטחת"""
    if anthropic is None:
        return None, "no_anthropic_library"
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    
    if not api_key:
        return None, "no_api_key"
        
    return anthropic.Anthropic(api_key=api_key), None


def _sanitize_json(text: str) -> str:
    """מנקה JSON שבור"""
    lines = text.split('\n')
    fixed_lines = []
    in_string = False
    current_line = ""
    
    for line in lines:
        quote_count = line.count('"') - line.count('\\"')
        
        if in_string:
            current_line += " " + line.strip()
            if quote_count % 2 == 1:
                in_string = False
                fixed_lines.append(current_line)
                current_line = ""
        else:
            if quote_count % 2 == 1:
                in_string = True
                current_line = line
            else:
                fixed_lines.append(line)
    
    text = '\n'.join(fixed_lines)
    
    # הסרת אינדקסים במערכים
    text = re.sub(r'(?m)^\s*\d+\s*:\s*', '', text)
    
    # החלפת NULL ב-null
    text = re.sub(r'\bNULL\b', 'null', text)
    
    # הוספת פסיקים חסרים
    text = re.sub(r'"(\s*\n\s*)"', r'",\1"', text)
    text = re.sub(r'"(\s*\n\s*)(true|false|null|\d+)', r'",\1\2', text)
    text = re.sub(r'(true|false|null|\d+)(\s*\n\s*)"', r'\1,\2"', text)
    text = re.sub(r'](\s*\n\s*)(["{0-9tfn])', r'],\1\2', text)
    text = re.sub(r'}(\s*\n\s*)(["{0-9tfn])', r'},\1\2', text)
    
    # הסרת פסיקים מיותרים
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text


def safe_process_metadata(raw_text):
    """
    מעבד מטא-דאטה מטקסט PDF ומחזיר סכמה מלאה
    
    Returns:
        dict עם: document, rooms, heights_and_levels, execution_notes, limitations, quantities_hint
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return {
            "status": "empty_text",
            "error": "אין מספיק טקסט לניתוח",
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": ["הטקסט שחולץ מה-PDF ריק או קצר מדי"],
            "quantities_hint": {}
        }
    
    client, error = get_anthropic_client()
    if error:
        return {
            "status": error,
            "error": "חסר מפתח API של Anthropic" if error == "no_api_key" else "ספריית anthropic לא מותקנת",
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": ["לא ניתן להפעיל AI ללא API key"],
            "quantities_hint": {}
        }

    # מודלים מעודכנים (ינואר 2025)
    models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514", 
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

    prompt = f"""
אתה מומחה בניתוח תוכניות בניה ישראליות. נתח את הטקסט הבא מ-PDF.

CRITICAL JSON REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no explanations
2. Use null (lowercase) for missing values, NEVER NULL
3. Do NOT add indices like "0:" or "1:" in arrays
4. Use double quotes for all keys and strings
5. No trailing commas before ] or }}

Input text (first 15000 chars):
{raw_text[:15000]}

Required JSON structure (MUST return exactly this schema):
{{
  "document": {{
    "plan_title": {{"value": "string or null", "confidence": 0-100, "evidence": ["source1"]}},
    "plan_type": {{"value": "תכנית קירות|תכנית תקרה|תכנית ריצוף|תכנית חשמל|other", "confidence": 0-100, "evidence": []}},
    "floor_or_level": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "project_name": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "date": {{"value": "DD/MM/YYYY or null", "confidence": 0-100, "evidence": []}},
    "scale": {{"value": "1:50 format or null", "confidence": 0-100, "evidence": []}},
    "sheet_numbers": {{"value": [] or null, "confidence": 0-100, "evidence": []}}
  }},
  "rooms": [
    {{
      "name": {{"value": "string", "confidence": 0-100, "evidence": []}},
      "area_m2": {{"value": 60 or null, "confidence": 0-100, "evidence": []}},
      "ceiling_height_m": {{"value": 2.8 or null, "confidence": 0-100, "evidence": []}},
      "ceiling_notes": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
      "flooring_notes": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
      "other_notes": {{"value": "string or null", "confidence": 0-100, "evidence": []}}
    }}
  ],
  "heights_and_levels": {{
    "floor_to_ceiling": {{"value": null, "confidence": 0, "evidence": []}},
    "reference_level": {{"value": null, "confidence": 0, "evidence": []}}
  }},
  "execution_notes": {{
    "materials_mentioned": [],
    "special_instructions": []
  }},
  "limitations": [
    "list of things that couldn't be extracted or were unclear"
  ],
  "quantities_hint": {{
    "walls_mentioned": false,
    "areas_mentioned": false
  }}
}}

IMPORTANT:
- If you find patterns like "חדר X מ\\"ר 60", extract as a room with area_m2
- Always return the FULL schema even if most fields are null
- Put extraction difficulties in "limitations" array
- Return ONLY the JSON, nothing else

JSON:"""

    errors_by_model = {}
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
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
            
            # ניקוי
            cleaned = _sanitize_json(response_text)
            
            # פרסור
            try:
                result = json.loads(cleaned)
                
                # ולידציה שיש את השדות החובה
                if "document" not in result:
                    result = {
                        "document": result if isinstance(result, dict) and "plan_title" in result else {},
                        "rooms": [],
                        "heights_and_levels": {},
                        "execution_notes": {},
                        "limitations": ["Schema mismatch - wrapped legacy format"],
                        "quantities_hint": {}
                    }
                
                # הוספת מידע על המודל
                result["_model_used"] = model
                result["status"] = "success"
                
                return result
                
            except json.JSONDecodeError as json_err:
                errors_by_model[model] = f"JSON parse error: {str(json_err)[:100]}"
                continue
                
        except Exception as e:
            error_str = str(e)
            
            if "not_found_error" in error_str or "404" in error_str:
                errors_by_model[model] = "404 - Model not available"
                continue
            elif "overloaded" in error_str.lower():
                errors_by_model[model] = "Model overloaded"
                continue
            else:
                errors_by_model[model] = error_str[:100]
                continue
    
    # כל המודלים נכשלו
    return {
        "status": "extraction_failed",
        "error": "כל המודלים נכשלו לחלץ מידע",
        "errors_by_model": errors_by_model,
        "tried_models": models,
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": [
            "כל המודלים שננסו נכשלו",
            "ייתכן שהטקסט לא מכיל מטא-דאטה ברורה",
            "או שיש בעיה ב-API"
        ],
        "quantities_hint": {}
    }


def analyze_legend_image(image_bytes):
    """מנתח תמונת מקרא"""
    client, error = get_anthropic_client()
    if error:
        return {"error": error}

    models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620"
    ]

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
נתח את המקרא בתמונה.

CRITICAL JSON:
- Return ONLY valid JSON
- Use null (lowercase), not NULL
- No indices like "0:"
- Double quotes only

Required structure:
{
  "plan_type": "תקרה|קירות|ריצוף|חשמל|אחר",
  "confidence": 0-100,
  "materials_found": ["material1"],
  "ceiling_types": [],
  "symbols": [{"symbol": "C11", "meaning": "description"}],
  "notes": "string or null",
  "legend_title": "string or null"
}

JSON:"""

    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=800,
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
            
            if "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            cleaned = _sanitize_json(response_text)
            
            try:
                result = json.loads(cleaned)
                
                # ולידציה
                required = ["plan_type", "confidence", "materials_found", "ceiling_types", "symbols"]
                for field in required:
                    if field not in result:
                        if field == "confidence":
                            result[field] = 50
                        elif field in ["materials_found", "ceiling_types", "symbols"]:
                            result[field] = []
                        else:
                            result[field] = "אחר"
                
                result["_model_used"] = model
                return result
                
            except json.JSONDecodeError:
                continue
            
        except Exception as e:
            if "not_found_error" in str(e) or "404" in str(e):
                continue
            continue
    
    return {"error": "כל המודלים נכשלו"}
