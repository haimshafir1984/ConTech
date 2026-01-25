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
    ✨ עם Prompt Caching - חוסך 90% מהעלות!
    משתמש במודלים העדכניים של 2025
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
            "quantities_hint": {"wall_types_mentioned": [], "material_hints": []},
        }

    # ===== מודלים עדכניים 2025 =====
    models = [
        "claude-3-5-sonnet-20241022",  # הכי חדש
        "claude-3-7-sonnet-20250219",  # Claude 3.7 Sonnet (אם קיים)
        "claude-3-5-haiku-20241022",  # Haiku חדש
        "claude-3-opus-20240229",  # Opus (יקר)
        "claude-3-haiku-20240307",  # Haiku ישן (fallback)
    ]

    # ===== System Prompt עם Cache Breakpoint =====
    system_prompt = [
        {
            "type": "text",
            "text": """אתה מומחה בחילוץ מידע מתוכניות בניה ישראליות.
המשימה: לחלץ **כל** המידע הזמין מהטקסט ולארגן אותו ב-JSON מובנה.

**חשוב מאוד:**
- החזר **רק** JSON תקין, ללא טקסט נוסף
- ודא שאין פסיקים מיותרים לפני ] או }}
- חלץ **כל** מידע זמין, במיוחד **מידות חדרים** ו**שטחים**

**מבנה JSON נדרש:**

{
  "document": {
    "plan_title": {"value": "שם התוכנית", "confidence": 0-100, "evidence": ["ציטוט"]},
    "plan_type": {"value": "קירות/תקרה/ריצוף/חשמל", "confidence": 0-100, "evidence": []},
    "scale": {"value": "1:50", "confidence": 0-100, "evidence": []},
    "date": {"value": "2024-01-15", "confidence": 0-100, "evidence": []},
    "floor_or_level": {"value": "קומה א'", "confidence": 0-100, "evidence": []},
    "project_name": {"value": null, "confidence": 0, "evidence": []},
    "project_address": {"value": null, "confidence": 0, "evidence": []},
    "architect_name": {"value": null, "confidence": 0, "evidence": []},
    "drawing_number": {"value": null, "confidence": 0, "evidence": []}
  },
  "rooms": [
    {
      "name": {"value": "חדר שינה 1", "confidence": 95, "evidence": ["חדר שינה 1"]},
      "area_m2": {"value": 15.5, "confidence": 90, "evidence": ["15.5 מ\\"ר"]},
      "ceiling_height_m": {"value": 2.70, "confidence": 85, "evidence": ["H=2.70"]},
      "flooring_notes": {"value": "פרקט", "confidence": 80, "evidence": ["פרקט"]},
      "ceiling_notes": {"value": null, "confidence": 0, "evidence": []},
      "other_notes": {"value": null, "confidence": 0, "evidence": []}
    }
  ],
  "heights_and_levels": {
    "default_ceiling_height_m": {"value": 2.80, "confidence": 70, "evidence": ["H=2.80"]},
    "default_floor_height_m": {"value": null, "confidence": 0, "evidence": []},
    "construction_level_m": {"value": null, "confidence": 0, "evidence": []}
  },
  "execution_notes": {
    "general_notes": {"value": null, "confidence": 0, "evidence": []},
    "structural_notes": {"value": null, "confidence": 0, "evidence": []},
    "hvac_notes": {"value": null, "confidence": 0, "evidence": []},
    "electrical_notes": {"value": null, "confidence": 0, "evidence": []},
    "plumbing_notes": {"value": null, "confidence": 0, "evidence": []}
  },
  "limitations": ["רשום כאן בעיות/מגבלות אם יש"],
  "quantities_hint": {
    "wall_types_mentioned": ["קיר בטון 20 ס\\"מ"],
    "material_hints": ["גרניט פורצלן"]
  }
}

**חיפוש חדרים:**
- שמות: "חדר שינה", "סלון", "מטבח", "שירותים"
- שטחים: "15 מ\\"ר", "15.5 m²", "15 sqm", או מספר ליד שם חדר
- גבהים: "H=2.80", "גובה 2.70", "ceiling height 2.80m"
- ריצוף: "קרמיקה", "פרקט", "שיש", "גרניט"
- תקרה: "גבס", "טרוול", "תקרה אקוסטית"
""",
            "cache_control": {"type": "ephemeral"},  # ← Cache breakpoint!
        }
    ]

    # ===== User Message =====
    user_message = f"""**טקסט מהתוכנית (חולץ מ-PDF):**

{raw_text[:3500]}

**התחל - החזר רק JSON:**"""

    for model in models:
        try:
            # ===== שימוש ב-Prompt Caching =====
            message = client.messages.create(
                model=model,
                max_tokens=6000,
                system=system_prompt,  # ← System עם cache
                messages=[{"role": "user", "content": user_message}],
            )

            response_text = message.content[0].text.strip()

            # ניקוי
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
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
                result["_cache_stats"] = {
                    "cache_creation_input_tokens": getattr(
                        message.usage, "cache_creation_input_tokens", 0
                    ),
                    "cache_read_input_tokens": getattr(
                        message.usage, "cache_read_input_tokens", 0
                    ),
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
                return result
            except json.JSONDecodeError:
                # תיקון אוטומטי
                fixed = response_text.replace(",]", "]").replace(",}", "}")
                try:
                    result = json.loads(fixed)
                    result["status"] = "success"
                    result["_model_used"] = model
                    result["_auto_fixed"] = True
                    return result
                except:
                    # JSON לא תקין - נסה מודל הבא
                    continue

        except anthropic.NotFoundError:
            # Model לא קיים (404) - נסה הבא
            continue
        except anthropic.BadRequestError as e:
            # Bad request (אולי prompt ארוך מדי)
            error_str = str(e)
            if "prompt is too long" in error_str.lower():
                # נסה עם טקסט קצר יותר
                try:
                    short_message = (
                        f"**טקסט (חלקי):**\n{raw_text[:2000]}\n\n**החזר JSON:**"
                    )
                    message = client.messages.create(
                        model=model,
                        max_tokens=6000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": short_message}],
                    )
                    response_text = message.content[0].text.strip()

                    # ניקוי ופרסור (אותו קוד כמו למעלה)
                    if "```json" in response_text:
                        response_text = (
                            response_text.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in response_text:
                        response_text = (
                            response_text.split("```")[1].split("```")[0].strip()
                        )

                    if "{" in response_text:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        response_text = response_text[start:end]

                    result = json.loads(
                        response_text.replace(",]", "]").replace(",}", "}")
                    )
                    result["status"] = "success"
                    result["_model_used"] = model
                    result["_warning"] = "Used shorter text due to length limit"
                    return result
                except:
                    continue
            else:
                continue
        except anthropic.RateLimitError:
            # Rate limit - נסה מודל הבא
            continue
        except Exception as e:
            # שגיאה כללית - נסה הבא
            continue

    # כשלון בכל המודלים
    return {
        "status": "extraction_failed",
        "error": "כל המודלים נכשלו - ייתכן שה-API key לא תקין או שאין חיבור",
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": [
            "Failed to extract data with all models - check API key and network"
        ],
        "quantities_hint": {"wall_types_mentioned": [], "material_hints": []},
    }


def analyze_legend_image(image_bytes):
    """
    ✨ ניתוח מקרא עם Vision API
    """
    client, error = get_anthropic_client()
    if error:
        return {"error": error}

    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """נתח את המקרא בתמונה והחזר JSON:

{
    "plan_type": "תקרה/קירות/ריצוף",
    "confidence": 0-100,
    "legend_title": "כותרת המקרא",
    "materials_found": ["רשימת חומרים"],
    "symbols": [{"symbol": "C11", "meaning": "קורה"}],
    "notes": "הערות"
}

**דוגמה - תקרה:**
אם רואה "מקרא תקרה", "לוחות מינרלים", "60X60" → plan_type: "תקרה", confidence: 95

**דוגמה - קירות:**
אם רואה "קיר בטון", "C11" → plan_type: "קירות"

**דוגמה - ריצוף:**
אם רואה "גרניט פורצלן", "קרמיקה" → plan_type: "ריצוף"

החזר רק JSON, ללא טקסט נוסף."""

    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            response_text = message.content[0].text.strip()

            # ניקוי
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            if "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            try:
                result = json.loads(response_text)
                result["_model_used"] = model
                return result
            except json.JSONDecodeError:
                fixed = response_text.replace(",]", "]").replace(",}", "}")
                result = json.loads(fixed)
                result["_model_used"] = model
                result["_auto_fixed"] = True
                return result

        except anthropic.NotFoundError:
            continue
        except Exception:
            continue

    return {"error": "כל המודלים נכשלו בניתוח התמונה", "tried_models": models}
