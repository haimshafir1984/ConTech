import os
import base64
import json
from typing import Optional
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

def safe_process_metadata(raw_text=None, raw_text_full=None, normalized_text=None, raw_blocks=None, candidates=None):
    """
    Enhanced metadata extraction with deterministic pre-parsing and LLM validation.
    
    Args:
        raw_text: Legacy short text (3000 chars) - for backward compatibility
        raw_text_full: Full text extraction (up to 20000 chars)
        normalized_text: Block-sorted text for better reading order
        raw_blocks: Structured text blocks with bbox info
        candidates: Pre-extracted candidates from deterministic parser
        
    Returns:
        Structured JSON with evidence-based metadata
    """
    client, error = get_anthropic_client()
    if error:
        return {"error": error, "status": "no_api_client"}
    
    # Choose best available text source
    text_to_use = normalized_text or raw_text_full or raw_text or ""
    
    if len(text_to_use.strip()) < 10:
        return {"error": "Insufficient text", "status": "empty_text"}
    
    # If candidates not provided, extract them now
    if candidates is None:
        try:
            from extractor import ArchitecturalTextExtractor
            extractor = ArchitecturalTextExtractor()
            candidates = extractor.extract_candidates(text_to_use)
        except Exception as e:
            # Fallback: no candidates
            candidates = {}
    
    # Build strict LLM prompt
    prompt = _build_strict_prompt(text_to_use, candidates)
    
    # Try multiple models
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229"
    ]
    
    errors_by_model = {}  # Track what went wrong with each model
    
    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.1,  # Low temperature for factual extraction
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # Clean and extract JSON
            clean_json = _extract_json_from_response(response_text)
            
            # Parse and validate
            try:
                result = json.loads(clean_json)
                result["_model_used"] = model
                result["_extraction_method"] = "enhanced_v1"
                return result
                
            except json.JSONDecodeError as json_err:
                # Auto-fix attempt
                fixed_result = _auto_fix_json(client, model, clean_json, json_err)
                if fixed_result:
                    return fixed_result
                # Track JSON error and continue
                errors_by_model[model] = f"JSON parse error: {str(json_err)[:100]}"
                continue
                
        except Exception as e:
            # Track the actual error
            error_msg = str(e)
            errors_by_model[model] = error_msg[:200]  # First 200 chars
            
            # Model not available - try next
            if "not_found_error" in error_msg or "404" in error_msg:
                continue
            # Other error - try next model
            continue
    
    # All models failed - return detailed error info
    return {
        "error": "All models failed to extract metadata",
        "status": "extraction_failed",
        "tried_models": models,
        "errors_by_model": errors_by_model  # NEW: Show what went wrong
    }


def _build_strict_prompt(text: str, candidates: dict) -> str:
    """Build strict extraction prompt with candidates"""
    
    candidates_summary = _format_candidates(candidates)
    
    prompt = f"""אתה מומחה בחילוץ מטא-דאטה מתוכניות אדריכליות ישראליות.

**חוקים קשיחים (אסור להפר!):**
1. אסור להמציא מידע. אם אין evidence → null + confidence נמוך + reason
2. לכל שדה מלא → evidence חובה (מקור הטקסט המדויק)
3. תעדף את הנתונים מ-CANDIDATES (חולצו בצורה דטרמיניסטית)
4. אם אתה רוצה לשנות ערך מ-CANDIDATES → הבא evidence חזק יותר
5. החזר **רק JSON** - אסור טקסט נוסף

**CANDIDATES שחולצו אוטומטית:**
{candidates_summary}

**טקסט מלא התוכנית:**
```
{text[:8000]}
```

**פורמט פלט (חובה - JSON בלבד):**
{{
  "document": {{
    "plan_title": {{"value": null, "confidence": 0, "evidence": []}},
    "plan_type": {{"value": null, "confidence": 0, "evidence": []}},
    "floor_or_level": {{"value": null, "confidence": 0, "evidence": []}},
    "project_name": {{"value": null, "confidence": 0, "evidence": []}},
    "issue_or_revision": {{"value": null, "confidence": 0, "evidence": []}},
    "date": {{"value": null, "confidence": 0, "evidence": []}},
    "scale": {{"value": null, "confidence": 0, "evidence": []}},
    "sheet_numbers": {{"value": [], "confidence": 0, "evidence": []}}
  }},
  "rooms": [
    {{
      "name": {{"value": null, "confidence": 0, "evidence": []}},
      "area_m2": {{"value": null, "confidence": 0, "evidence": []}},
      "ceiling_height_m": {{"value": null, "confidence": 0, "evidence": []}},
      "ceiling_notes": {{"value": null, "confidence": 0, "evidence": []}},
      "flooring_notes": {{"value": null, "confidence": 0, "evidence": []}},
      "other_notes": {{"value": null, "confidence": 0, "evidence": []}}
    }}
  ],
  "heights_and_levels": {{
    "ceiling_levels_m": [{{"value": null, "confidence": 0, "evidence": []}}],
    "other_levels": [{{"value": null, "confidence": 0, "evidence": []}}]
  }},
  "execution_notes": {{
    "general_notes": [{{"value": null, "confidence": 0, "evidence": []}}],
    "standards": [{{"value": null, "confidence": 0, "evidence": []}}],
    "contractor_requirements": [{{"value": null, "confidence": 0, "evidence": []}}]
  }},
  "quantities_hint": {{
    "has_enough_data_for_wall_lengths": {{"value": false, "confidence": 0, "reason": null}},
    "has_enough_data_for_per_room_perimeters": {{"value": false, "confidence": 0, "reason": null}},
    "has_explicit_room_areas": {{"value": false, "confidence": 0, "reason": null}}
  }},
  "limitations": [
    {{"value": null, "confidence": 0, "evidence": []}}
  ]
}}

**דוגמה נכונה:**
{{
  "document": {{
    "plan_title": {{"value": "תכנית קומה ב'", "confidence": 90, "evidence": ["תכנית קומה ב' - בית ספר"]}},
    "scale": {{"value": "1:50", "confidence": 95, "evidence": ["קנ\\"מ 1:50"]}}
  }},
  "rooms": [
    {{
      "name": {{"value": "חדר מורים", "confidence": 85, "evidence": ["חדר מורים ר\\"מ 25.5"]}},
      "area_m2": {{"value": 25.5, "confidence": 90, "evidence": ["חדר מורים ר\\"מ 25.5"]}}
    }}
  ]
}}

**התחל עכשיו - החזר רק JSON:**"""
    
    return prompt


def _format_candidates(candidates: dict) -> str:
    """Format candidates for inclusion in prompt"""
    if not candidates:
        return "לא נמצאו candidates"
    
    parts = []
    
    # Rooms
    if candidates.get('rooms'):
        parts.append(f"חדרים ({len(candidates['rooms'])}):")
        for room in candidates['rooms'][:10]:  # Limit to first 10
            name = room.get('name', {}).get('value', '?')
            area = room.get('area_m2', {}).get('value', '?')
            parts.append(f"  - {name}: {area} מ\"ר")
    
    # Scale
    if candidates.get('scale'):
        scale = candidates['scale']
        parts.append(f"קנ\"מ: {scale.get('ratio', '?')} (value={scale.get('value', '?')})")
    
    # Levels
    if candidates.get('levels'):
        parts.append(f"מפלסים ({len(candidates['levels'])}):")
        for level in candidates['levels'][:5]:
            label = level.get('label', '?')
            value = level.get('value_m', '?')
            parts.append(f"  - {label}: {value}m")
    
    # Heights
    if candidates.get('heights'):
        parts.append(f"גבהים: {len(candidates['heights'])} נמצאו")
    
    # Document info
    doc_info = candidates.get('document_info', {})
    if doc_info.get('plan_title'):
        parts.append(f"כותרת: {doc_info['plan_title']['value']}")
    if doc_info.get('date'):
        parts.append(f"תאריך: {doc_info['date']['value']}")
    
    return "\n".join(parts) if parts else "אין candidates"


def _extract_json_from_response(response_text: str) -> str:
    """Extract clean JSON from LLM response"""
    # Remove markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    # Extract JSON object
    if "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        response_text = response_text[start:end]
    
    return response_text


def _auto_fix_json(client, model: str, broken_json: str, error) -> Optional[dict]:
    """
    Attempt to auto-fix broken JSON by asking LLM to repair it.
    Returns parsed dict or None if unfixable.
    """
    fix_prompt = f"""The following JSON has a syntax error. Fix it and return ONLY valid JSON (no explanations):

ERROR: {str(error)}

BROKEN JSON:
```
{broken_json[:2000]}
```

Return the corrected JSON:"""
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=0,
            messages=[{"role": "user", "content": fix_prompt}]
        )
        
        fixed_text = message.content[0].text.strip()
        clean_fixed = _extract_json_from_response(fixed_text)
        
        result = json.loads(clean_fixed)
        result["_auto_fixed"] = True
        return result
        
    except Exception:
        return None


def process_plan_metadata(raw_text):
    """Legacy function - redirects to safe_process_metadata for backward compatibility"""
    return safe_process_metadata(raw_text=raw_text)

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
    "notes": "תכנית תקרה קומה ב' - 8 כיתות",
    "legend_title": "מקרא תקרה"
}

**חשוב מאוד:**
- אם רואה "מקרא תקרה" → plan_type חייב להיות "תקרה" (ביטחון 98%)
- אם רואה "לוחות מינרלים" → זו בוודאות תקרה
- קרא את כל הטקסט בעברית בקפידה
- החזר **רק** JSON, אין טקסט נוסף
- אם לא בטוח ב-100%, כתב confidence נמוך (60-70)

**דוגמאות:**
✅ נכון: {"plan_type": "תקרה", "confidence": 98, "legend_title": "מקרא תקרה"}
❌ שגוי: {"plan_type": "אחר", "confidence": 80}  ← אם יש "מקרא תקרה"!
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
