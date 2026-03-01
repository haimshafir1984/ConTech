import os
import base64
import json
import logging
logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

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
    [UPGRADED] מנוע היברידי: Google Vision OCR + Claude AI
    משתמש ב-Prompt Caching - חוסך 90% מהעלות!
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
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            print("[WARNING] Google Cloud credentials not found - skipping Google OCR")
            ocr_source = "pymupdf_no_credentials"
        else:
            try:
                from ocr_google import ocr_pdf_google_vision

                ocr_result = ocr_pdf_google_vision(
                    pdf_bytes, dpi=150, language_hints=["he", "en"]
                )

                text_to_analyze = ocr_result["full_text"]
                ocr_source = "google_vision"

                print(f"[INFO] Google Vision OCR: {len(text_to_analyze)} chars")

            except Exception as e:
                print(f"[WARNING] Google Vision failed, fallback to PyMuPDF: {e}")
                ocr_source = "pymupdf_fallback"

    # ===== מודלים עדכניים 2025 =====
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    # ===== System Prompt עם Cache Breakpoint =====
    system_prompt = [
        {
            "type": "text",
            "text": r"""אתה מומחה בחילוץ מידע מתוכניות בניה ישראליות.
המשימה: לחלץ **כל** המידע הזמין מהטקסט ולארגן אותו ב-JSON מובנה.

**חשוב מאוד:**
- החזר **רק** JSON תקין, ללא טקסט נוסף
- ודא שאין פסיקים מיותרים לפני ] או }
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
    "drawing_number": {"value": null, "confidence": 0, "evidence": []},
    "sheet_number": {"value": null, "confidence": 0, "evidence": []},
    "sheet_name": {"value": null, "confidence": 0, "evidence": []},
    "status": {"value": null, "confidence": 0, "evidence": [], "note": "לאישור/למכרז/לביצוע/טיוטה"},
    "revision": {"value": null, "confidence": 0, "evidence": []},
    "drawn_by": {"value": null, "confidence": 0, "evidence": []},
    "designed_by": {"value": null, "confidence": 0, "evidence": []},
    "approved_by": {"value": null, "confidence": 0, "evidence": []}
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
- שטחים: "15 מ\"ר", "15.5 m^2", "15 sqm", או מספר ליד שם חדר
- גבהים: "H=2.80", "גובה 2.70", "ceiling height 2.80m"
- ריצוף: "קרמיקה", "פרקט", "שיש", "גרניט"
- תקרה: "גבס", "טרוול", "תקרה אקוסטית"

**חיפוש פרטי מסמך (בלוק כותרת):**
- status: "לאישור", "למכרז", "לביצוע", "Approved for Construction"
- revision: "Rev.A", "מהדורה 1", "גרסה B", "Rev.02"
- sheet_number: "A-01", "01/15", "גיליון מס' 3"
- sheet_name: "תוכנית קומה א'", "ריצוף קומה ראשונה"
- drawn_by: "שרטט:", "Drawn by:", "ש.ט."
- designed_by: "מתכנן:", "Designed by:", "מ.ת."
- approved_by: "מאשר:", "Approved by:", "נבדק:"
""",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # ===== User Message =====
    user_message = f"""**טקסט מהתוכנית:**

{text_to_analyze[:15000]}

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
                result["_ocr_source"] = ocr_source
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
                except Exception as _e:
                    logger.warning("brain: json-fix fallback model=%s err=%s", model, _e)
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
                except Exception as _e:
                    logger.warning("brain: short-text fallback model=%s err=%s", model, _e)
                    continue
            else:
                continue
        except anthropic.RateLimitError:
            continue
        except Exception as _e:
            logger.warning("brain: model=%s outer exception: %s", model, _e)
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
    [VISION API] ניתוח מקרא תוכנית בניה ומזהה סוג תוכנית וחומרים
    """
    client, error = get_anthropic_client()
    if error:
        return {"error": error}

    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    # ===== עיבוד תמונה לפני שליחה ל-Claude =====
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size

        # Resize אם מעל 6000px בכל ציר (שמירת יחס ממדים, LANCZOS)
        MAX_DIM = 6000
        if w > MAX_DIM or h > MAX_DIM:
            ratio = min(MAX_DIM / w, MAX_DIM / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # המרה ל-JPEG דחוס quality=85
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        processed_bytes = output.getvalue()

        # אם עדיין גדול מ-4.5MB → הורד quality
        if len(processed_bytes) > 4.5 * 1024 * 1024:
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=60, optimize=True)
            processed_bytes = output.getvalue()

        encoded_image = base64.b64encode(processed_bytes).decode("utf-8")
        media_type = "image/jpeg"

    except Exception as img_err:
        # fallback — שולח כמו שהוא
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/png"
        print(f"[WARNING] Image preprocessing failed, using original: {img_err}")

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
אם רואה "מקרא תקרה", "לוחות מינרלים", "60X60" => plan_type: "תקרה", confidence: 95

**דוגמה - קירות:**
אם רואה "קיר בטון", "C11" => plan_type: "קירות"

**דוגמה - ריצוף:**
אם רואה "גרניט פורצלן", "קרמיקה" => plan_type: "ריצוף"

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
                                    "media_type": media_type,
                                    "data": encoded_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            response_text = message.content[0].text.strip()

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

        except anthropic.BadRequestError as e:
            return {
                "error": f"תמונה לא תקינה עבור Claude: {e}",
                "tried_models": [model],
            }
        except anthropic.NotFoundError:
            continue
        except Exception as e:
            print(f"[ERROR] analyze_legend_image model={model}: {e}")
            continue

    return {"error": "כל המודלים נכשלו בניתוח התמונה", "tried_models": models}


# ─────────────────────────────────────────────────────────────────
# PDF ARCHITECTURAL EXTRACTOR — Vision + tool_use (JSON מובטח)
# ─────────────────────────────────────────────────────────────────

_ARCH_TOOL = {
    "name": "extract_floor_plan_data",
    "description": "חלץ את כל המידע מהתוכנית האדריכלית המוצגת בתמונה",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_title": {
                "type": "string",
                "description": "שם/כותרת התוכנית"
            },
            "plan_type": {
                "type": "string",
                "enum": ["floor_plan", "section", "elevation", "detail", "site", "other"],
                "description": "סוג התוכנית"
            },
            "scale": {
                "type": "string",
                "description": "קנה מידה, למשל 1:50"
            },
            "floor_level": {
                "type": "string",
                "description": "קומה או רמה, למשל קומה ראשונה"
            },
            "drawing_number": {
                "type": "string",
                "description": "מספר תוכנית"
            },
            "architect": {
                "type": "string",
                "description": "שם האדריכל אם מופיע"
            },
            "project_address": {
                "type": "string",
                "description": "כתובת הפרויקט אם מופיעה"
            },
            "rooms": {
                "type": "array",
                "description": "רשימת כל החדרים/מרחבים בתוכנית",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":               {"type": "string", "description": "שם החדר"},
                        "area_m2":            {"type": "number", "description": "שטח במ\"ר"},
                        "dimensions":         {"type": "string", "description": "מידות, למשל 3.50 x 4.20"},
                        "ceiling_height_m":   {"type": "number", "description": "גובה תקרה במטר"},
                        "flooring":           {"type": "string", "description": "סוג ריצוף"},
                        "notes":              {"type": "string", "description": "הערות נוספות"}
                    },
                    "required": ["name"]
                }
            },
            "dimensions_found": {
                "type": "array",
                "description": "כל קוטי המידה שנמצאו בתוכנית (מספרים על קווי מידה, חצים)",
                "items": {"type": "string"}
            },
            "default_ceiling_height_m": {
                "type": "number",
                "description": "גובה תקרה כללי לכל הקומה אם מצוין"
            },
            "materials": {
                "type": "array",
                "description": "חומרי בניה שמוזכרים",
                "items": {"type": "string"}
            },
            "execution_notes": {
                "type": "array",
                "description": "הערות ביצוע, דרישות מיוחדות",
                "items": {"type": "string"}
            },
            "warnings": {
                "type": "array",
                "description": "בעיות או מגבלות בקריאת התוכנית",
                "items": {"type": "string"}
            }
        },
        "required": ["rooms"]
    }
}

_ARCH_SYSTEM = """אתה מומחה בקריאת תוכניות אדריכליות ישראליות.
תפקידך לחלץ כמה שיותר מידע מדויק מהתמונה:

1. **חדרים ושטחים** - חפש שמות חדרים עם מספרים סמוך אליהם (שטח במ"ר)
2. **מידות** - מספרים על קווי מידה (בד"כ עם חצים משני הצדדים)
3. **גבהי תקרה** - H= או "ג.ת." עם מספר, או הערה כללית
4. **ריצוף וחומרים** - פרקט, קרמיקה, שיש, בטון, גרניט
5. **הערות ביצוע** - טקסט בשוליים, בטבלאות, בכותרות
6. **קנה מידה** - 1:50, 1:100 וכדומה
7. **פרטי מסמך** - שם תוכנית, קומה, מספר תוכנית, אדריכל

**מה לא לפספס:**
- מספרים קטנים בתוך חדרים = שטח
- מספרים על קווים = מידה (בד"כ בס"מ)
- ראשי תיבות: ח.ש. = חדר שינה, מ.ח. = מחסן, כ.ש. = כושר
- H= או ג.ת.= גובה תקרה"""


def extract_from_architectural_pdf(pdf_path: str) -> dict:
    """
    חילוץ מקסימלי מתוכנית אדריכלית - כל הדפים, Vision + tool_use.

    Args:
        pdf_path: נתיב לקובץ PDF

    Returns:
        dict עם pages (רשימה לפי דף) ו-merged (מיזוג כל הדפים)
    """
    if fitz is None:
        return {"error": "PyMuPDF לא מותקן. הרץ: pip install pymupdf", "pages": []}

    client, error = get_anthropic_client()
    if error:
        return {"error": error, "pages": []}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": f"לא ניתן לפתוח PDF: {e}", "pages": []}

    all_pages = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]

        # 300 DPI — חיוני לקריאת מידות קטנות בתוכניות
        matrix = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("jpeg", jpg_quality=92)
        img_b64 = base64.standard_b64encode(img_bytes).decode()

        # טקסט גולמי מה-PDF (אם הוא דיגיטלי ולא סרוק)
        page_text = page.get_text("text").strip()
        text_note = (
            f"טקסט שחולץ אוטומטית מהדף:\n{page_text[:3000]}"
            if page_text
            else "הדף נראה סרוק — סמוך על ניתוח ויזואלי בלבד."
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_ARCH_SYSTEM,
                tools=[_ARCH_TOOL],
                tool_choice={"type": "tool", "name": "extract_floor_plan_data"},
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                f"תוכנית אדריכלית ישראלית — דף {page_num + 1} מתוך {total_pages}.\n\n"
                                f"{text_note}\n\n"
                                "חלץ את כל המידע הנראה בתמונה."
                            )
                        }
                    ]
                }]
            )

            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            page_data = tool_block.input if tool_block else {}
            page_data["_tokens"] = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }

        except Exception as e:
            page_data = {"error": str(e), "rooms": [], "warnings": [str(e)]}

        all_pages.append({"page": page_num + 1, "data": page_data})

    doc.close()

    # ===== מיזוג תוצאות כל הדפים =====
    merged = _merge_pages(all_pages)

    return {
        "status": "success",
        "total_pages": total_pages,
        "pages": all_pages,
        "merged": merged,
    }


def _merge_pages(pages: list) -> dict:
    """מאחד תוצאות מכמה דפים לתוצאה אחת מאוחדת."""
    all_rooms = []
    all_dimensions = []
    all_materials = []
    all_notes = []
    all_warnings = []

    # מטה-דאטה מהדף הראשון שמחזיר ערך
    meta_fields = ["plan_title", "plan_type", "scale", "floor_level",
                   "drawing_number", "architect", "project_address",
                   "default_ceiling_height_m"]
    merged_meta = {f: None for f in meta_fields}

    for page in pages:
        d = page.get("data", {})

        for field in meta_fields:
            if merged_meta[field] is None and d.get(field):
                merged_meta[field] = d[field]

        for room in d.get("rooms", []):
            room["_from_page"] = page["page"]
            all_rooms.append(room)

        all_dimensions.extend(d.get("dimensions_found", []))
        all_materials.extend(d.get("materials", []))
        all_notes.extend(d.get("execution_notes", []))
        all_warnings.extend(d.get("warnings", []))

    return {
        **merged_meta,
        "rooms": all_rooms,
        "dimensions_found": list(dict.fromkeys(all_dimensions)),  # ייחודיים, שומר סדר
        "materials": list(dict.fromkeys(all_materials)),
        "execution_notes": list(dict.fromkeys(all_notes)),
        "warnings": all_warnings,
        "total_rooms": len(all_rooms),
        "total_area_m2": round(
            sum(r.get("area_m2", 0) or 0 for r in all_rooms), 2
        ),
    }
