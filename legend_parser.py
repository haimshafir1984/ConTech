"""
legend_parser.py — מוצא את המקרא בתוכנית הארכיטקטונית,
שולח ל-Claude Vision, מחזיר מיפוי: label → (type, subtype).
"""
import base64
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import fitz
import anthropic


LEGEND_KEYWORDS = ["מקרא", "legend", "סימון", "הסבר", "רשימה", "תוכן"]


def _find_legend_rect(page: fitz.Page) -> Optional[fitz.Rect]:
    """מחפש את אזור המקרא בדף לפי מילות מפתח."""
    blocks = page.get_text("blocks")
    for block in blocks:
        x0, y0, x1, y1, text = block[:5]
        if any(kw in text for kw in LEGEND_KEYWORDS):
            pw, ph = page.rect.width, page.rect.height
            rect = fitz.Rect(
                max(0, x0 - 20),
                max(0, y0 - 10),
                min(pw, x1 + 300),
                min(ph, y1 + 400),
            )
            return rect

    # Fallback: פינה שמאלית-תחתית
    pw, ph = page.rect.width, page.rect.height
    return fitz.Rect(0, ph * 0.65, pw * 0.35, ph)


def _crop_legend_to_image(page: fitz.Page, rect: fitz.Rect) -> bytes:
    """חותך את אזור המקרא לתמונת PNG ברזולוציה גבוהה."""
    mat  = fitz.Matrix(2.5, 2.5)
    clip = page.get_pixmap(matrix=mat, clip=rect, colorspace=fitz.csRGB)
    return clip.tobytes("png")


def parse_legend(pdf_bytes: bytes) -> Dict[str, Tuple[str, str]]:
    """
    קלט ראשי: bytes של PDF.
    מחזיר מיפוי: label_text → (type, subtype).
    דוגמה: {"קיר בטון": ("קירות", "בטון"), "כיור": ("כלים סניטריים", "כיור")}
    מחזיר {} אם אין מקרא.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
    except Exception as e:
        print(f"[legend_parser] fitz.open failed: {e}")
        return {}

    legend_rect = _find_legend_rect(page)
    if legend_rect is None:
        return {}

    try:
        img_bytes = _crop_legend_to_image(page, legend_rect)
    except Exception as e:
        print(f"[legend_parser] crop failed: {e}")
        return {}

    img_b64 = base64.standard_b64encode(img_bytes).decode()

    prompt = """זוהי תמונה של מקרא (legend) מתוך תוכנית ארכיטקטונית.
פענח את כל הרשומות במקרא.

החזר JSON בלבד, ללא טקסט נוסף, בפורמט זה:
{
  "has_legend": true,
  "entries": [
    {
      "label": "שם הקטגוריה כפי שמופיע במקרא",
      "type": "אחד מ: קירות | כלים סניטריים | ריצוף | תקרה | דלתות וחלונות | כלים סניטריים | חשמל ותאורה | מיזוג ואוורור | עמודים ושלד | פרטים",
      "subtype": "תת-קטגוריה ספציפית בתוך type"
    }
  ]
}

אם אין מקרא ברור בתמונה, החזר: {"has_legend": false, "entries": []}
אל תציג רשומות שאינן מופיעות בתמונה."""

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        raw  = response.content[0].text.strip()
        raw  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)

        if not data.get("has_legend"):
            return {}

        mapping: Dict[str, Tuple[str, str]] = {}
        for entry in data.get("entries", []):
            label    = entry.get("label", "").strip()
            etype    = entry.get("type",  "קירות").strip()
            esubtype = entry.get("subtype", "בלוקים").strip()
            if label:
                mapping[label] = (etype, esubtype)

        print(f"[legend_parser] Parsed {len(mapping)} legend entries")
        return mapping

    except Exception as e:
        print(f"[legend_parser] Claude Vision call failed: {e}")
        return {}


def apply_legend_to_segments(
    segments: List[dict],
    legend: Dict[str, Tuple[str, str]],
) -> List[dict]:
    """
    מחיל את המקרא על הסגמנטים.
    מחפש התאמה לפי מילות מפתח בין label המקרא לsubtype הנוכחי.
    """
    if not legend:
        return segments

    for seg in segments:
        current_subtype = seg.get("suggested_subtype", "") or ""
        best_match      = None
        best_score      = 0

        for label, (etype, esubtype) in legend.items():
            lwords = set(label.replace("/", " ").split())
            swords = set(current_subtype.replace("/", " ").split())
            score  = len(lwords & swords)
            if score > best_score:
                best_score = score
                best_match = (etype, esubtype)

        if best_match and best_score >= 1:
            seg["suggested_type"]    = best_match[0]
            seg["suggested_subtype"] = best_match[1]
            seg["confidence"]        = min(0.97, seg.get("confidence", 0.8) + 0.07)
            seg["_legend_matched"]   = True

    return segments
