"""
Vision-based floor plan analyzer — standalone, no Streamlit dependency.

Calls Claude Vision (claude-sonnet-4-6) with tool_use to extract structured
data from an architectural PDF: rooms, scale, dimensions, materials, elements,
grid lines, systems and more.

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


# ─── Tool schema ─────────────────────────────────────────────────────────────

_ARCH_TOOL = {
    "name": "extract_floor_plan_data",
    "description": "חלץ את כל המידע מהתוכנית האדריכלית המוצגת בתמונה",
    "input_schema": {
        "type": "object",
        "properties": {
            # ── Document metadata ──────────────────────────────────────────
            "plan_title": {
                "type": "string",
                "description": "שם/כותרת התוכנית",
            },
            "project_name": {
                "type": "string",
                "description": "שם הפרויקט (נפרד מכותרת התוכנית, בד\"כ בבלוק הכותרת)",
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
                "description": "מספר תוכנית/גיליון",
            },
            "sheet_number": {
                "type": "string",
                "description": "מספר גיליון (sheet number), למשל A-01",
            },
            "sheet_name": {
                "type": "string",
                "description": "שם הגיליון, למשל 'קומה קרקע — תוכנית ריצוף'",
            },
            "status": {
                "type": "string",
                "enum": ["לאישור", "למכרז", "לביצוע", "טיוטה", "לא ידוע"],
                "description": "סטטוס התוכנית",
            },
            "revision": {
                "type": "string",
                "description": "גרסה/מהדורה (A, B, 01, Rev.2 וכו')",
            },
            "date": {
                "type": "string",
                "description": "תאריך התוכנית",
            },
            "architect": {
                "type": "string",
                "description": "שם האדריכל/משרד",
            },
            "drawn_by": {
                "type": "string",
                "description": "שרטט (drawn by)",
            },
            "designed_by": {
                "type": "string",
                "description": "מתכנן (designed by)",
            },
            "approved_by": {
                "type": "string",
                "description": "מאשר (approved by / checked by)",
            },
            "project_address": {
                "type": "string",
                "description": "כתובת הפרויקט אם מופיעה",
            },

            # ── Spaces / Rooms ─────────────────────────────────────────────
            "rooms": {
                "type": "array",
                "description": "רשימת כל החדרים/מרחבים בתוכנית",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":               {"type": "string",  "description": "שם החדר"},
                        "area_m2":            {"type": "number",  "description": "שטח במ\"ר"},
                        "dimensions":         {"type": "string",  "description": "מידות, למשל 3.50 x 4.20"},
                        "ceiling_height_m":   {"type": "number",  "description": "גובה תקרה במטר"},
                        "elevation_floor_m":  {"type": "number",  "description": "גובה ריצוף ±מטר (±0.00)"},
                        "elevation_slab_m":   {"type": "number",  "description": "גובה מצע/רצפה ±מטר"},
                        "flooring":           {"type": "string",  "description": "סוג ריצוף"},
                        "notes":              {"type": "string",  "description": "הערות נוספות"},
                        "position_x_pct":     {"type": "number",  "description": "מיקום אופקי של תווית החדר בתמונה, 0.0=שמאל עד 1.0=ימין"},
                        "position_y_pct":     {"type": "number",  "description": "מיקום אנכי של תווית החדר בתמונה, 0.0=למעלה עד 1.0=למטה"},
                    },
                    "required": ["name"],
                },
            },

            # ── Simple dimension list (legacy) ─────────────────────────────
            "dimensions_found": {
                "type": "array",
                "description": "כל קוטי המידה שנמצאו בתוכנית (מספרים על קווי מידה, חצים)",
                "items": {"type": "string"},
            },

            # ── Structured dimensions ──────────────────────────────────────
            "dimensions_structured": {
                "type": "array",
                "description": "קוטי מידה מפורטים עם מיקום וסוג",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw":      {"type": "string", "description": "מחרוזת המידה המקורית, למשל 3660"},
                        "unit":     {"type": "string", "enum": ["mm", "cm", "m"], "description": "יחידת מידה (ברירת מחדל: mm)"},
                        "location": {"type": "string", "description": "תיאור מיקום (ציר, חדר, קיר)"},
                        "type":     {"type": "string", "enum": ["overall", "partial", "height", "stair", "other"], "description": "סוג המידה"},
                    },
                    "required": ["raw"],
                },
            },

            # ── Elevations ─────────────────────────────────────────────────
            "elevations": {
                "type": "array",
                "description": "גבהים ורמות (±0.00 וכדומה)",
                "items": {
                    "type": "object",
                    "properties": {
                        "label":     {"type": "string", "description": "תווית הגובה (±0.00, +3.20)"},
                        "value":     {"type": "number", "description": "ערך מספרי"},
                        "reference": {"type": "string", "description": "מה מייצג הגובה (ריצוף, תקרה, מצע)"},
                    },
                    "required": ["label"],
                },
            },

            "default_ceiling_height_m": {
                "type": "number",
                "description": "גובה תקרה כללי לכל הקומה אם מצוין",
            },

            # ── Materials ──────────────────────────────────────────────────
            "materials": {
                "type": "array",
                "description": "חומרי בניה שמוזכרים (רשימה פשוטה)",
                "items": {"type": "string"},
            },
            "materials_legend": {
                "type": "array",
                "description": "מקרא חומרים מהתוכנית (symbol + description + fire_rating)",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol":      {"type": "string", "description": "סמל/קוד (C11, ///, ריבוע שחור)"},
                        "description": {"type": "string", "description": "תיאור החומר/אלמנט"},
                        "fire_rating": {"type": "string", "description": "עמידות אש אם מצוין (60 דקות, REI120)"},
                    },
                    "required": ["description"],
                },
            },

            # ── Elements ───────────────────────────────────────────────────
            "elements": {
                "type": "array",
                "description": "אלמנטים: דלתות, חלונות, מדרגות, מעלית, אינסטלציה, מכאניקה",
                "items": {
                    "type": "object",
                    "properties": {
                        "type":     {"type": "string", "description": "סוג: door | window | stair | elevator | sink | toilet | shower | boiler | other"},
                        "id":       {"type": "string", "description": "מזהה/מספר (ד1, D1, ח2, W2...)"},
                        "location": {"type": "string", "description": "מיקום בשרטוט (שם חדר, קיר, ציר)"},
                        "notes":    {"type": "string", "description": "הערות (גדלים: 90x200, חומרים, כיוון פתיחה)"},
                    },
                    "required": ["type"],
                },
            },

            # ── Grid lines ─────────────────────────────────────────────────
            "grid_lines": {
                "type": "object",
                "description": "קווי רשת/צירים בשרטוט",
                "properties": {
                    "horizontal": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "תוויות ציר אופקי (A, B, C... או 1, 2, 3...)",
                    },
                    "vertical": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "תוויות ציר אנכי (1, 2, 3... או A, B, C...)",
                    },
                },
            },

            # ── Special systems ────────────────────────────────────────────
            "systems": {
                "type": "object",
                "description": "מערכות מיוחדות המופיעות בתוכנית",
                "properties": {
                    "waterproofing":       {"type": "string", "description": "הערות איטום"},
                    "drainage_slopes_pct": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "שיפועי ניקוז (%, למשל '1.5%', '2%')",
                    },
                    "hvac_notes":          {"type": "string", "description": "הערות מיזוג/אוורור"},
                    "fire_suppression":    {"type": "string", "description": "מערכת כיבוי אש"},
                    "accessibility":       {"type": "string", "description": "נגישות (נכים, כבשים, מעקות)"},
                },
            },

            # ── Other ──────────────────────────────────────────────────────
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
2. **מידות** - מספרים על קווי מידה (בד"כ עם חצים משני הצדדים); חלץ גם dimensions_structured עם raw/unit/location/type
3. **גבהים** - H= או "ג.ת." עם מספר, ±0.00 לגבהי ריצוף/תקרה/מצע
4. **ריצוף וחומרים** - פרקט, קרמיקה, שיש, בטון, גרניט; חלץ גם materials_legend מהמקרא
5. **הערות ביצוע** - טקסט בשוליים, בטבלאות, בכותרות
6. **קנה מידה** - 1:50, 1:100 וכדומה
7. **פרטי מסמך** - שם תוכנית, קומה, מספר תוכנית, מספר גיליון, שם גיליון, אדריכל, שרטט, מתכנן, מאשר
8. **סטטוס ומהדורה** - לאישור/למכרז/לביצוע, גרסה (Rev.A, B, 01), תאריך
9. **אלמנטים** - דלתות (D1, ד1), חלונות (W1, ח1), מדרגות, מעלית, כיורים, אסלות, דודים
10. **קווי רשת/צירים** - אותיות (A,B,C) ומספרים (1,2,3) בשולי השרטוט → grid_lines
11. **מקרא חומרים** - טבלת מקרא עם סמל, תיאור, ועמידות אש אם מצוין
12. **מערכות מיוחדות** - איטום, שיפועי ניקוז (%), מיזוג, כיבוי אש, נגישות

**מה לא לפספס:**
- מספרים קטנים בתוך חדרים = שטח
- מספרים על קווים = מידה (בד"כ בס"מ או מ"מ)
- ראשי תיבות: ח.ש. = חדר שינה, מ.ח. = מחסן, כ.ש. = כושר
- H= או ג.ת.= גובה תקרה; ±0.00 = גובה ריצוף
- בלוק כותרת בפינה: project_name, sheet_number, status, revision, drawn_by, designed_by, approved_by

**לכל חדר:** הוסף position_x_pct ו-position_y_pct — המיקום המשוער של תווית שם החדר בתמונה כחלק עשרוני (0.0–1.0). לדוגמה, חדר בפינה שמאלית עליונה: x=0.1, y=0.15."""


def _render_page_to_b64(page) -> str:
    """Renders a PyMuPDF page as a base64 JPEG, staying under Claude's 5 MB limit."""
    import io
    from PIL import Image as _Image

    _MAX_BYTES = 5 * 1024 * 1024 - 1  # 5242879 bytes — strictly under the API limit

    matrix = _fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    quality = 70
    while quality >= 40:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg_bytes = buf.getvalue()
        if len(jpeg_bytes) <= _MAX_BYTES:
            return base64.standard_b64encode(jpeg_bytes).decode()
        quality -= 5

    # Still too large — halve dimensions and retry at quality=40
    img = img.resize((img.width // 2, img.height // 2), _Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_claude(client, img_b64: str, page_num: int, total_pages: int, page_text: str) -> dict:
    """Calls Claude Vision with tool_use for one page. Returns raw tool input dict or {}."""
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
                                f"תוכנית אדריכלית ישראלית — דף {page_num} מתוך {total_pages}.\n\n"
                                f"{text_note}\n\n"
                                "חלץ את כל המידע הנראה בתמונה."
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"[vision_analyzer] Claude API error (page {page_num}): {e}")
        return {}

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if not tool_block:
        print(f"[vision_analyzer] No tool_use block (page {page_num})")
        return {}

    data = dict(tool_block.input)
    data["_tokens"] = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
        "page": page_num,
    }
    return data


def _merge_page_results(pages: list[dict]) -> dict:
    """
    Merges results from multiple pages.
    Scalar metadata: first non-None value wins.
    Lists: accumulate and deduplicate where possible.
    """
    SCALAR_FIELDS = [
        "plan_title", "project_name", "plan_type", "scale", "floor_level",
        "drawing_number", "sheet_number", "sheet_name", "status", "revision",
        "date", "architect", "drawn_by", "designed_by", "approved_by",
        "project_address", "default_ceiling_height_m", "grid_lines", "systems",
    ]
    LIST_FIELDS = [
        "rooms", "dimensions_found", "dimensions_structured", "elevations",
        "materials", "materials_legend", "elements", "execution_notes", "warnings",
    ]

    merged: dict = {f: None for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        merged[f] = []

    total_input_tokens = 0
    total_output_tokens = 0

    for page_data in pages:
        page_num = page_data.get("_tokens", {}).get("page", "?")
        total_input_tokens += page_data.get("_tokens", {}).get("input", 0)
        total_output_tokens += page_data.get("_tokens", {}).get("output", 0)

        for field in SCALAR_FIELDS:
            if merged[field] is None and page_data.get(field):
                merged[field] = page_data[field]

        for field in LIST_FIELDS:
            items = page_data.get(field) or []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        if item not in merged[field]:
                            merged[field].append(item)
                    else:
                        merged[field].append(item)

    merged["_tokens"] = {"input": total_input_tokens, "output": total_output_tokens}
    return merged


def analyze_plan_with_vision(pdf_path: str, max_pages: int = 5) -> dict:
    """
    מנתח PDF אדריכלי עם Claude Vision (300 DPI, tool_use).
    מעבד עד max_pages דפים וממזג תוצאות.

    Returns dict with keys:
        rooms, scale, dimensions_found, dimensions_structured,
        elevations, materials, materials_legend, elements,
        grid_lines, systems, plan_title, project_name, sheet_number,
        sheet_name, status, revision, drawn_by, designed_by, approved_by,
        floor_level, default_ceiling_height_m, execution_notes,
        total_area_m2, _model, _tokens

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

    total_pages = len(doc)
    pages_to_process = min(total_pages, max(1, max_pages))

    all_page_data: list[dict] = []
    try:
        for page_idx in range(pages_to_process):
            page = doc[page_idx]
            img_b64 = _render_page_to_b64(page)
            page_text = page.get_text("text").strip()
            page_data = _call_claude(client, img_b64, page_idx + 1, total_pages, page_text)
            if page_data:
                all_page_data.append(page_data)
    finally:
        doc.close()

    if not all_page_data:
        return {}

    data = _merge_page_results(all_page_data)

    rooms = data.get("rooms") or []
    total_area = round(sum(r.get("area_m2") or 0 for r in rooms), 2)

    result = {
        # Metadata
        "plan_title":             data.get("plan_title"),
        "project_name":           data.get("project_name"),
        "plan_type":              data.get("plan_type"),
        "floor_level":            data.get("floor_level"),
        "drawing_number":         data.get("drawing_number"),
        "sheet_number":           data.get("sheet_number"),
        "sheet_name":             data.get("sheet_name"),
        "status":                 data.get("status"),
        "revision":               data.get("revision"),
        "date":                   data.get("date"),
        "architect":              data.get("architect"),
        "drawn_by":               data.get("drawn_by"),
        "designed_by":            data.get("designed_by"),
        "approved_by":            data.get("approved_by"),
        "project_address":        data.get("project_address"),
        # Spaces
        "rooms":                  rooms,
        "total_area_m2":          total_area,
        # Dimensions
        "scale":                  data.get("scale"),
        "default_ceiling_height_m": data.get("default_ceiling_height_m"),
        "dimensions_found":       data.get("dimensions_found") or [],
        "dimensions_structured":  data.get("dimensions_structured") or [],
        "elevations":             data.get("elevations") or [],
        # Materials
        "materials":              data.get("materials") or [],
        "materials_legend":       data.get("materials_legend") or [],
        # Elements
        "elements":               data.get("elements") or [],
        # Grid
        "grid_lines":             data.get("grid_lines") or {},
        # Systems
        "systems":                data.get("systems") or {},
        # Notes
        "execution_notes":        data.get("execution_notes") or [],
        # Internal
        "_model":  "claude-sonnet-4-6",
        "_tokens": data.get("_tokens", {}),
        "_pages_processed": pages_to_process,
    }

    print(
        f"[vision_analyzer] OK — {len(rooms)} rooms, scale={result['scale']}, "
        f"pages={pages_to_process}/{total_pages}, "
        f"elements={len(result['elements'])}, "
        f"tokens={result['_tokens']}"
    )
    return result
