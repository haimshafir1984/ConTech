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


# ==========================================
# JSON SCHEMA DEFINITIONS
# ==========================================

# Schema for each field with confidence
FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ]
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["value", "confidence", "evidence"],
    "additionalProperties": False
}

# Full metadata schema
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "document": {
            "type": "object",
            "properties": {
                "plan_title": FIELD_SCHEMA,
                "plan_type": FIELD_SCHEMA,
                "scale": FIELD_SCHEMA,
                "date": FIELD_SCHEMA,
                "floor_or_level": FIELD_SCHEMA,
                "project_name": FIELD_SCHEMA,
                "project_address": FIELD_SCHEMA,
                "architect_name": FIELD_SCHEMA,
                "drawing_number": FIELD_SCHEMA
            },
            "required": ["plan_title", "plan_type", "scale", "date", "floor_or_level", 
                        "project_name", "project_address", "architect_name", "drawing_number"],
            "additionalProperties": False
        },
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": FIELD_SCHEMA,
                    "area_m2": FIELD_SCHEMA,
                    "ceiling_height_m": FIELD_SCHEMA,
                    "ceiling_notes": FIELD_SCHEMA,
                    "flooring_notes": FIELD_SCHEMA,
                    "other_notes": FIELD_SCHEMA
                },
                "required": ["name", "area_m2", "ceiling_height_m", "ceiling_notes", 
                            "flooring_notes", "other_notes"],
                "additionalProperties": False
            }
        },
        "heights_and_levels": {
            "type": "object",
            "properties": {
                "default_ceiling_height_m": FIELD_SCHEMA,
                "default_floor_height_m": FIELD_SCHEMA,
                "construction_level_m": FIELD_SCHEMA
            },
            "required": ["default_ceiling_height_m", "default_floor_height_m", "construction_level_m"],
            "additionalProperties": False
        },
        "execution_notes": {
            "type": "object",
            "properties": {
                "general_notes": FIELD_SCHEMA,
                "structural_notes": FIELD_SCHEMA,
                "hvac_notes": FIELD_SCHEMA,
                "electrical_notes": FIELD_SCHEMA,
                "plumbing_notes": FIELD_SCHEMA
            },
            "required": ["general_notes", "structural_notes", "hvac_notes", 
                        "electrical_notes", "plumbing_notes"],
            "additionalProperties": False
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "quantities_hint": {
            "type": "object",
            "properties": {
                "wall_types_mentioned": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "material_hints": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["wall_types_mentioned", "material_hints"],
            "additionalProperties": False
        }
    },
    "required": ["document", "rooms", "heights_and_levels", "execution_notes", 
                "limitations", "quantities_hint"],
    "additionalProperties": False
}


# ==========================================
# ACTIVE MODELS ONLY (No retired 3.x models)
# ==========================================

ACTIVE_MODELS = [
    "claude-sonnet-4-20250514",      # Sonnet 4.5 (latest)
    "claude-opus-4-20250514",        # Opus 4.5 (latest)
    "claude-sonnet-4-20250328",      # Sonnet 4 (stable)
    "claude-haiku-4-20250416"        # Haiku 4.5 (fastest)
]


def process_plan_metadata(raw_text):
    """
    מעבד מטא-דאטה של תוכנית עם Structured Outputs
    מבטיח JSON תקין תמיד
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

    if not raw_text or not raw_text.strip():
        return {
            "status": "empty_text",
            "error": "לא נמצא טקסט בקובץ",
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": ["לא נמצא טקסט בקובץ PDF"],
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
        }

    prompt = f"""
אתה מומחה מנוסה בניתוח תוכניות בניה ישראליות. המשימה שלך היא לחלץ את כל המידע הזמין מהטקסט שלפניך.

**קריטי - קרא בקפידה:**
- חלץ **כל** מידע זמין, במיוחד **מידות חדרים** ו**שטחים**
- אל תדלג על שום נתון מספרי (שטחים, גבהים, מידות)
- אם יש טקסט חוזר או OCR לא מושלם - נסה להבין את הכוונה
- confidence גבוה (80-100) רק כשאתה בטוח 100%
- confidence בינוני (50-79) כשיש ראיות חלקיות
- confidence נמוך (1-49) כשאתה משער
- evidence: צטט את המקור המדויק מהטקסט

**טקסט מהתוכנית (חולץ מ-PDF):**
{raw_text[:3500]}

**מבנה JSON נדרש:**

1. **document** - פרטי התוכנית הכללית:
   - plan_title: שם/כותרת התוכנית (חפש בראש המסמך)
   - plan_type: סוג התוכנית (קירות/תקרה/ריצוף/חשמל/אינסטלציה)
   - scale: קנה מידה (1:50, 1:100, 1:200 וכו')
   - date: תאריך התוכנית
   - floor_or_level: קומה/מפלס (קומה א', קרקע, מינוס 1)
   - project_name: שם הפרויקט
   - project_address: כתובת הפרויקט
   - architect_name: שם האדריכל/משרד
   - drawing_number: מספר שרטוט

2. **rooms** - מערך של חדרים (זה החשוב ביותר!):
   
   **חיפוש חכם של חדרים:**
   - חפש שמות חדרים: "חדר שינה", "סלון", "מטבח", "שירותים", "חדר רחצה", "מרפסת"
   - חפש מספרי חדרים: "חדר 1", "חדר 2", "101", "102"
   - **שטחים**: חפש מספרים עם יחידות:
     * "15 מ\"ר", "15.5 מ\"ר", "15 m²", "15 sqm"
     * מספר + "מ\"ר" או "מטר רבוע"
     * אפילו רק מספר ליד שם חדר (לדוגמה: "סלון 25" = 25 מ\"ר)
   - **גובה תקרה**: 
     * "H=2.80", "H=2.60", "גובה 2.70"
     * "תקרה 2.80 מ'", "ceiling height 2.80m"
   - **ריצוף**: "קרמיקה", "פרקט", "שיש", "גרניט פורצלן"
   - **תקרה**: "גבס", "טרוול", "תקרה אקוסטית"
   
   **דוגמה:**
   אם הטקסט: "חדר שינה 1 15.5 מ\"ר H=2.70 פרקט"
   אז:
   {{
     "name": {{"value": "חדר שינה 1", "confidence": 95, "evidence": ["חדר שינה 1"]}},
     "area_m2": {{"value": 15.5, "confidence": 95, "evidence": ["15.5 מ\"ר"]}},
     "ceiling_height_m": {{"value": 2.70, "confidence": 90, "evidence": ["H=2.70"]}},
     "flooring_notes": {{"value": "פרקט", "confidence": 90, "evidence": ["פרקט"]}},
     "ceiling_notes": {{"value": null, "confidence": 0, "evidence": []}},
     "other_notes": {{"value": null, "confidence": 0, "evidence": []}}
   }}

3. **heights_and_levels** - גבהים ומפלסים כלליים:
   - default_ceiling_height_m: גובה תקרה סטנדרטי בקומה
   - default_floor_height_m: גובה רצפה ממפלס 0
   - construction_level_m: מפלס בנייה

4. **execution_notes** - הערות ביצוע:
   - general_notes: הערות כלליות
   - structural_notes: הערות קונסטרוקציה
   - hvac_notes: הערות מיזוג אוויר
   - electrical_notes: הערות חשמל
   - plumbing_notes: הערות אינסטלציה

5. **limitations** - מערך של בעיות/מגבלות:
   - רשום כאן אם הטקסט חלקי, יש בעיות OCR, מידע חסר וכו'
   - דוגמה: ["Document appears to be partial", "Some room areas missing"]

6. **quantities_hint** - רמזים לכמויות:
   - wall_types_mentioned: ["קיר בטון 20 ס\"מ", "קיר בלוקים 10 ס\"מ"]
   - material_hints: ["גרניט פורצלן", "אריח קרמי 60x60"]

**פורמט שדה בסיסי:**
{{"value": <string/number/null>, "confidence": 0-100, "evidence": [ציטוטים]}}

**אסטרטגיית חילוץ:**
1. קרא את כל הטקסט מלמעלה למטה
2. זהה את סוג התוכנית קודם כל
3. חפש טבלאות של חדרים (לרוב בפורמט: שם | שטח | גובה)
4. חפש מספרים ליד שמות חדרים
5. אם יש טקסט חוזר - קח את הגרסה הברורה ביותר
6. רשום ב-limitations כל בעיה שמצאת

**סוגי תוכניות נפוצים:**
- "קירות/Walls" - קווי קירות, פתחים, דלתות
- "תקרה/Ceiling" - פרטי תקרה, גבהים, סוגי תקרה
- "ריצוף/Flooring" - סוגי ריצוף, שטחים
- "חשמל/Electrical" - נקודות חשמל, תאורה
- "אינסטלציה/Plumbing" - נקודות מים, ניקוז
- "אדריכלות/Architecture" - כללי, שילוב הכל

**התחל עכשיו - חלץ כל מה שאתה יכול!**
"""

    errors_by_model = {}
    last_error = None

    for model in ACTIVE_MODELS:
        try:
            # ===== STRUCTURED OUTPUTS - GUARANTEED JSON =====
            message = client.beta.messages.create(
                model=model,
                max_tokens=6000,  # ← הגדלנו מ-4000 ל-6000
                betas=["structured-outputs-2025-11-13"],
                messages=[{"role": "user", "content": prompt}],
                output_format={
                    "type": "json_schema",
                    "schema": METADATA_SCHEMA
                }
            )
            
            # Check if response was truncated
            if message.stop_reason == "max_tokens":
                # Retry with even higher max_tokens
                message = client.beta.messages.create(
                    model=model,
                    max_tokens=10000,  # ← הגדלנו מ-8000 ל-10000
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": prompt}],
                    output_format={
                        "type": "json_schema",
                        "schema": METADATA_SCHEMA
                    }
                )
            
            # Structured outputs guarantee valid JSON
            response_text = message.content[0].text
            result = json.loads(response_text)
            
            # Add metadata
            result["status"] = "success"
            result["model_used"] = model
            result["stop_reason"] = message.stop_reason
            
            return result
            
        except anthropic.NotFoundError as e:
            # Model not available (404)
            error_msg = f"Model not found: {model}"
            errors_by_model[model] = error_msg
            last_error = error_msg
            continue
            
        except anthropic.BadRequestError as e:
            # Structured outputs not supported or schema error
            error_msg = f"Bad request: {str(e)[:100]}"
            errors_by_model[model] = error_msg
            last_error = error_msg
            
            # Fallback to prompt-based JSON (legacy)
            try:
                result = _fallback_prompt_based_json(client, model, raw_text)
                if result:
                    result["model_used"] = model
                    result["fallback_used"] = True
                    return result
            except Exception as fallback_error:
                errors_by_model[model] += f" | Fallback failed: {str(fallback_error)[:50]}"
                continue
            
        except json.JSONDecodeError as e:
            # Should NEVER happen with structured outputs
            error_msg = f"JSON decode error (unexpected): {str(e)[:100]}"
            errors_by_model[model] = error_msg
            last_error = error_msg
            continue
            
        except Exception as e:
            error_msg = f"Error: {str(e)[:100]}"
            errors_by_model[model] = error_msg
            last_error = error_msg
            continue
    
    # All models failed
    return {
        "status": "extraction_failed",
        "error": f"כל המודלים נכשלו. שגיאה אחרונה: {last_error}",
        "errors_by_model": errors_by_model,
        "tried_models": ACTIVE_MODELS,
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": [f"חילוץ נכשל: {last_error}"],
        "quantities_hint": {"wall_types_mentioned": [], "material_hints": []}
    }


def _fallback_prompt_based_json(client, model, raw_text):
    """
    Fallback למצב legacy (ללא structured outputs)
    משתמש רק אם structured outputs לא זמין
    """
    prompt_legacy = f"""
Analyze this construction plan text and extract ALL available information, especially ROOM DIMENSIONS.

**CRITICAL INSTRUCTIONS:**
1. Extract EVERY piece of numerical data (areas, heights, dimensions)
2. Find ALL rooms with their exact measurements
3. Look for patterns: "room_name XX.X m²", "H=X.XX", "area: XX"
4. If text is repeated or has OCR errors - interpret and extract the data anyway
5. Return ONLY valid JSON - no markdown, no explanation

**Input text from PDF:**
{raw_text[:3000]}

**ROOM EXTRACTION PATTERNS:**
- "bedroom 15.5 m²" → area_m2: 15.5
- "H=2.70" or "height 2.70m" → ceiling_height_m: 2.70
- "living room 25" → likely 25 m² (use confidence: 70)
- "ceramic tiles" → flooring_notes
- Look for tables with: name | area | height

**Required JSON structure:**
{{
  "document": {{
    "plan_title": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "plan_type": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "scale": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "date": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "floor_or_level": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "project_name": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "project_address": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "architect_name": {{"value": "string or null", "confidence": 0-100, "evidence": []}},
    "drawing_number": {{"value": "string or null", "confidence": 0-100, "evidence": []}}
  }},
  "rooms": [
    {{
      "name": {{"value": "room name", "confidence": 90, "evidence": ["quote from text"]}},
      "area_m2": {{"value": 15.5, "confidence": 95, "evidence": ["15.5 m²"]}},
      "ceiling_height_m": {{"value": 2.70, "confidence": 90, "evidence": ["H=2.70"]}},
      "ceiling_notes": {{"value": "gypsum board", "confidence": 80, "evidence": ["gypsum"]}},
      "flooring_notes": {{"value": "ceramic tiles", "confidence": 85, "evidence": ["ceramic"]}},
      "other_notes": {{"value": null, "confidence": 0, "evidence": []}}
    }}
  ],
  "heights_and_levels": {{
    "default_ceiling_height_m": {{"value": null, "confidence": 0, "evidence": []}},
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
  "limitations": ["list any issues: partial text, OCR errors, missing data"],
  "quantities_hint": {{
    "wall_types_mentioned": ["concrete wall 20cm", "block wall 10cm"],
    "material_hints": ["ceramic tiles 60x60", "parquet flooring"]
  }}
}}

**START EXTRACTION NOW - Return JSON only:**
"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt_legacy}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Sanitize response
        response_text = _sanitize_json_response(response_text)
        
        # Parse
        result = json.loads(response_text)
        result["status"] = "success"
        return result
        
    except json.JSONDecodeError:
        # Try auto-fix
        try:
            fixed = _auto_fix_json(response_text)
            result = json.loads(fixed)
            result["status"] = "success"
            result["auto_fixed"] = True
            return result
        except Exception:
            return None
    except Exception:
        return None


def _sanitize_json_response(text):
    """ניקוי תשובת JSON מבעיות נפוצות"""
    # Remove markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Extract JSON object
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    
    # Fix common issues
    # Remove list indices like "0: {" → "{"
    import re
    text = re.sub(r'^\s*\d+:\s*', '', text, flags=re.MULTILINE)
    
    # Replace NULL with null
    text = text.replace("NULL", "null")
    text = text.replace("None", "null")
    
    return text


def _auto_fix_json(text):
    """ניסיון אוטומטי לתקן JSON שגוי"""
    # Remove trailing commas
    text = text.replace(",]", "]")
    text = text.replace(",}", "}")
    
    # Fix unescaped quotes (simple heuristic)
    # This is risky - only as last resort
    
    return text


# ==========================================
# LEGEND ANALYSIS (separate function)
# ==========================================

LEGEND_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_type": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "materials_found": {
            "type": "array",
            "items": {"type": "string"}
        },
        "ceiling_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"},
                    "dimensions": {"type": "string"}
                },
                "required": ["code", "description", "dimensions"],
                "additionalProperties": False
            }
        },
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"}
                },
                "required": ["symbol", "meaning"],
                "additionalProperties": False
            }
        },
        "notes": {"type": "string"},
        "legend_title": {"type": "string"}
    },
    "required": ["plan_type", "confidence", "materials_found", "ceiling_types", 
                "symbols", "notes", "legend_title"],
    "additionalProperties": False
}


def analyze_legend_image(image_bytes):
    """
    מנתח תמונה של מקרא תוכנית בניה ומזהה סוג תוכנית וחומרים
    משתמש ב-Structured Outputs למניעת שגיאות JSON
    """
    client, error = get_anthropic_client()
    if error:
        return {"error": error}

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
אתה מומחה בניתוח תוכניות בניה ישראליות.
נתח את המקרא (Legend) בתמונה זו.

**צעדים לזיהוי:**

1️⃣ **קרא את הכותרת המרכזית במקרא**
   - חפש: "מקרא תקרה" / "מקרא קירות" / "מקרא ריצוף"

2️⃣ **חפש מילות מפתח ספציפיות:**
   
   **תקרה →**
   - "תקרה אקוסטית" / "תקרת גבס" / "תקרה פריקה"
   - "לוחות מינרלים" / "ארקליט" 
   - מידות: "60X60" / "60X120"
   
   **קירות →**
   - "קיר בטון" / "קיר בלוקים"
   - "עובי קיר"
   - סימונים: C11, C12, C13 (קורות)
   
   **ריצוף →**
   - "אריח קרמי" / "גרניט פורצלן"
   - "מפלס גמר"
   - מידות: "30X30" / "60X60"

**חובה להחזיר:**
- plan_type: "תקרה" / "קירות" / "ריצוף" / "חשמל" / "אחר"
- confidence: 0-100
- materials_found: רשימת חומרים
- ceiling_types: רשימת סוגי תקרות (אם רלוונטי)
- symbols: רשימת סמלים
- notes: הערות
- legend_title: הכותרת שנמצאה
"""

    errors_by_model = {}
    
    for model in ACTIVE_MODELS:
        try:
            message = client.beta.messages.create(
                model=model,
                max_tokens=1500,
                betas=["structured-outputs-2025-11-13"],
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
                }],
                output_format={
                    "type": "json_schema",
                    "schema": LEGEND_SCHEMA
                }
            )
            
            # Guaranteed valid JSON
            response_text = message.content[0].text
            result = json.loads(response_text)
            result["_model_used"] = model
            
            return result
            
        except anthropic.NotFoundError:
            errors_by_model[model] = "Model not found (404)"
            continue
            
        except anthropic.BadRequestError as e:
            errors_by_model[model] = f"Bad request: {str(e)[:80]}"
            
            # Fallback to legacy
            try:
                result = _fallback_legend_prompt(client, model, encoded_image)
                if result:
                    result["_model_used"] = model
                    result["_fallback_used"] = True
                    return result
            except Exception:
                continue
            
        except Exception as e:
            errors_by_model[model] = str(e)[:100]
            continue
    
    # All models failed
    return {
        "error": f"כל המודלים נכשלו. ניסה: {', '.join(ACTIVE_MODELS)}",
        "errors_by_model": errors_by_model,
        "tried_models": ACTIVE_MODELS
    }


def _fallback_legend_prompt(client, model, encoded_image):
    """Fallback למקרא ללא structured outputs"""
    prompt_legacy = """
Analyze this construction plan legend.
Return ONLY valid JSON. No markdown, no explanation.

JSON structure:
{
  "plan_type": "string",
  "confidence": 0-100,
  "materials_found": [],
  "ceiling_types": [],
  "symbols": [],
  "notes": "string",
  "legend_title": "string"
}
"""
    
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
                    {"type": "text", "text": prompt_legacy}
                ]
            }]
        )
        
        response_text = message.content[0].text.strip()
        response_text = _sanitize_json_response(response_text)
        
        return json.loads(response_text)
    except Exception:
        return None
