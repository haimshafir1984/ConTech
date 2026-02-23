"""
Vision-based floor plan analyzer — standalone, no Streamlit dependency.

Calls Claude Vision (claude-sonnet-4-6) with tool_use to extract structured
data from an architectural PDF: rooms, scale, dimensions, materials.

Usage (from main.py):
    from .vision_analyzer import analyze_plan_with_vision
    vision_data = analyze_plan_with_vision(pdf_path)
"""

import os
import base64

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

try:
    import fitz as _fitz  # PyMuPDF
except ImportError:
    _fitz = None


# ─── Tool schema (identical to brain.py _ARCH_TOOL) ──────────────────────────

_ARCH_TOOL = {
    "name": "extract_floor_plan_data",
    "description": "חלץ את כל המידע מהתוכנית האדריכלית המוצגת בתמונה",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_title": {
                "type": "string",
                "description": "שם/כותרת התוכנית",
            },
            "plan_type": {
                "type": "string",
                "enum": ["floor_plan", "section", "elevation", "detail", "site", "other"],
                "description": "סוג התוכנית",
            },
            "scale": {
                "type": "string",
                "description": "קנה מידה, למשל 1:50",
            },
            "floor_level": {
                "type": "string",
                "description": "קומה או רמה, למשל קומה ראשונה",
            },
            "drawing_number": {
                "type": "string",
                "description": "מספר תוכנית",
            },
            "architect": {
                "type": "string",
                "description": "שם האדריכל אם מופיע",
            },
            "project_address": {
                "type": "string",
                "description": "כתובת הפרויקט אם מופיעה",
            },
            "rooms": {
                "type": "array",
                "description": "רשימת כל החדרים/מרחבים בתוכנית",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":             {"type": "string", "description": "שם החדר"},
                        "area_m2":          {"type": "number", "description": "שטח במ\"ר"},
                        "dimensions":       {"type": "string", "description": "מידות, למשל 3.50 x 4.20"},
                        "ceiling_height_m": {"type": "number", "description": "גובה תקרה במטר"},
                        "flooring":         {"type": "string", "description": "סוג ריצוף"},
                        "notes":            {"type": "string", "description": "הערות נוספות"},
                        "position_x_pct":   {"type": "number", "description": "מיקום אופקי של תווית החדר בתמונה, 0.0=שמאל עד 1.0=ימין"},
                        "position_y_pct":   {"type": "number", "description": "מיקום אנכי של תווית החדר בתמונה, 0.0=למעלה עד 1.0=למטה"},
                    },
                    "required": ["name"],
                },
            },
            "dimensions_found": {
                "type": "array",
                "description": "כל קוטי המידה שנמצאו בתוכנית (מספרים על קווי מידה, חצים)",
                "items": {"type": "string"},
            },
            "default_ceiling_height_m": {
                "type": "number",
                "description": "גובה תקרה כללי לכל הקומה אם מצוין",
            },
            "materials": {
                "type": "array",
                "description": "חומרי בניה שמוזכרים",
                "items": {"type": "string"},
            },
            "execution_notes": {
                "type": "array",
                "description": "הערות ביצוע, דרישות מיוחדות",
                "items": {"type": "string"},
            },
            "warnings": {
                "type": "array",
                "description": "בעיות או מגבלות בקריאת התוכנית",
                "items": {"type": "string"},
            },
        },
        "required": ["rooms"],
    },
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
- H= או ג.ת.= גובה תקרה

**לכל חדר:** הוסף position_x_pct ו-position_y_pct — המיקום המשוער של תווית שם החדר בתמונה כחלק עשרוני (0.0–1.0). לדוגמה, חדר בפינה שמאלית עליונה: x=0.1, y=0.15."""


def analyze_plan_with_vision(pdf_path: str) -> dict:
    """
    מנתח את עמוד 1 של PDF אדריכלי עם Claude Vision (300 DPI, tool_use).

    Returns dict with keys:
        rooms           - list of room dicts (name, area_m2, dimensions, ...)
        scale           - string like "1:50" or None
        dimensions_found - list of dimension strings
        materials        - list of material strings
        plan_title       - str or None
        total_area_m2    - float
        _model           - model name used
        _tokens          - {input, output}

    On failure, returns empty dict {}.
    """
    if _fitz is None:
        print("[vision_analyzer] PyMuPDF not installed — skipping vision analysis")
        return {}

    if _anthropic is None:
        print("[vision_analyzer] anthropic not installed — skipping vision analysis")
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[vision_analyzer] ANTHROPIC_API_KEY not set — skipping vision analysis")
        return {}

    client = _anthropic.Anthropic(api_key=api_key)

    try:
        doc = _fitz.open(pdf_path)
    except Exception as e:
        print(f"[vision_analyzer] Cannot open PDF: {e}")
        return {}

    try:
        page = doc[0]  # רק עמוד 1

        # 300 DPI — חיוני לקריאת מידות קטנות
        matrix = _fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("jpeg", jpg_quality=92)
        img_b64 = base64.standard_b64encode(img_bytes).decode()

        # טקסט גולמי מה-PDF (אם דיגיטלי)
        page_text = page.get_text("text").strip()
        text_note = (
            f"טקסט שחולץ אוטומטית מהדף:\n{page_text[:3000]}"
            if page_text
            else "הדף נראה סרוק — סמוך על ניתוח ויזואלי בלבד."
        )

        total_pages = len(doc)

    finally:
        doc.close()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_ARCH_SYSTEM,
            tools=[_ARCH_TOOL],
            tool_choice={"type": "tool", "name": "extract_floor_plan_data"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"תוכנית אדריכלית ישראלית — דף 1 מתוך {total_pages}.\n\n"
                                f"{text_note}\n\n"
                                "חלץ את כל המידע הנראה בתמונה."
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"[vision_analyzer] Claude API error: {e}")
        return {}

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if not tool_block:
        print("[vision_analyzer] No tool_use block in response")
        return {}

    data: dict = tool_block.input

    # מחשב שטח כולל
    rooms = data.get("rooms") or []
    total_area = round(sum(r.get("area_m2") or 0 for r in rooms), 2)

    result = {
        "rooms": rooms,
        "scale": data.get("scale"),
        "dimensions_found": data.get("dimensions_found") or [],
        "materials": data.get("materials") or [],
        "plan_title": data.get("plan_title"),
        "floor_level": data.get("floor_level"),
        "default_ceiling_height_m": data.get("default_ceiling_height_m"),
        "execution_notes": data.get("execution_notes") or [],
        "total_area_m2": total_area,
        "_model": "claude-sonnet-4-6",
        "_tokens": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }

    print(
        f"[vision_analyzer] OK — {len(rooms)} rooms, "
        f"scale={result['scale']}, "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
    )
    return result
