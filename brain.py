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


def process_plan_metadata(raw_text, use_google_ocr=True, pdf_bytes=None):
    """
    ✨ מנוע היברידי: Google Vision OCR + Claude AI

    Args:
        raw_text: טקסט שחולץ מ-PDF (PyMuPDF fallback)
        use_google_ocr: האם להשתמש ב-Google OCR (ברירת מחדל: כן)
        pdf_bytes: bytes של ה-PDF (אם רוצים Google OCR)

    Returns:
        dict עם המידע המחולץ
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

    # ===== שלב 1: חילוץ טקסט (Google OCR או PyMuPDF) =====
    ocr_source = "pymupdf"
    text_to_analyze = raw_text

    if use_google_ocr and pdf_bytes:
        try:
            from ocr_google import ocr_pdf_google_vision

            ocr_result = ocr_pdf_google_vision(
                pdf_bytes, dpi=300, language_hints=["he", "en"]  # עברית + אנגלית
            )

            text_to_analyze = ocr_result["full_text"]
            ocr_source = "google_vision"

            # Debug info
            print(f"✅ Google Vision OCR: {len(text_to_analyze)} תווים")

        except Exception as e:
            print(f"⚠️ Google Vision נכשל, חוזר ל-PyMuPDF: {e}")
            # נשאר עם raw_text המקורי
            ocr_source = "pymupdf_fallback"

    # ===== שלב 2: ניתוח עם Claude =====
    # (שאר הקוד נשאר זהה מפה)

    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    system_prompt = [
        {
            "type": "text",
            "text": r"""אתה מומחה בחילוץ מידע מתוכניות בניה ישראליות.
המשימה: לחלץ **כל** המידע הזמין מהטקסט ולארגן אותו ב-JSON מובנה.

**חשוב מאוד:**
- החזר **רק** JSON תקין, ללא טקסט נוסף
- ודא שאין פסיקים מיותרים לפני ] או }}
- חלץ **כל** מידע זמין, במיוחד **מידות חדרים** ו**שטחים**

**מבנה JSON נדרש:**
"""

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
      "area_m2": {"value": 15.5, "confidence": 90, "evidence": ["15.5 מ\"ר"]},
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
    "wall_types_mentioned": ["קיר בטון 20 ס\"מ"],
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
            "cache_control": {"type": "ephemeral"},
        }
    ]

    user_message = f"""**טקסט מהתוכנית:**

{text_to_analyze[:3500]}

**התחל - החזר רק JSON:**"""

    for model in models:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=6000,
                system=system_prompt,
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

            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            try:
                result = json.loads(response_text)
                result["status"] = "success"
                result["_model_used"] = model
                result["_ocr_source"] = ocr_source  # ← מקור ה-OCR
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
                fixed = response_text.replace(",]", "]").replace(",}", "}")
                try:
                    result = json.loads(fixed)
                    result["status"] = "success"
                    result["_model_used"] = model
                    result["_ocr_source"] = ocr_source
                    result["_auto_fixed"] = True
                    return result
                except:
                    continue

        except anthropic.NotFoundError:
            continue
        except anthropic.BadRequestError as e:
            error_str = str(e)
            if "prompt is too long" in error_str.lower():
                try:
                    short_message = (
                        f"**טקסט (חלקי):**\n{text_to_analyze[:2000]}\n\n**החזר JSON:**"
                    )
                    message = client.messages.create(
                        model=model,
                        max_tokens=6000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": short_message}],
                    )
                    response_text = message.content[0].text.strip()

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
                    result["_ocr_source"] = ocr_source
                    result["_warning"] = "Used shorter text due to length limit"
                    return result
                except:
                    continue
            else:
                continue
        except anthropic.RateLimitError:
            continue
        except Exception:
            continue

    return {
        "status": "extraction_failed",
        "error": "כל המודלים נכשלו",
        "_ocr_source": ocr_source,
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": ["Failed to extract data with all models"],
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