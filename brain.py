import os
import base64
import json
import logging
from typing import Optional
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
import uuid as _uuid
from collections import defaultdict as _defaultdict


# ── Vector-based wall detection ───────────────────────────────────────────────

def _is_dark_color(color: tuple) -> bool:
    """Return True if the colour tuple (r,g,b) represents a dark stroke."""
    if not color or len(color) < 3:
        return True
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    return r < 0.3 and g < 0.3 and b < 0.3


def _walls_from_vectors(
    vector_cache: dict,
    scale_px_per_meter: float,
    image_shape: Optional[dict] = None,
) -> list:
    """Extract wall segments from PDF vector data via Union-Find line chaining.

    Extracts individual "l" (line) items from thick, dark path drawings,
    chains co-linear endpoint-adjacent segments using Union-Find, filters
    chains shorter than 0.5 m, and returns bbox segments in pixel space.

    Args:
        vector_cache:      dict from _build_vector_cache() — keys: drawings,
                           page_rect, words, text_dict.
        scale_px_per_meter: pixels per metre for length filtering.
        image_shape:       {"width": W, "height": H} of the raster image.

    Returns:
        List of segment dicts compatible with AutoAnalyzeResponse.segments.
        Returns [] when vector_cache is empty or no thick dark lines found.
    """
    if not vector_cache:
        return []

    drawings  = vector_cache.get("drawings") or []
    page_rect = vector_cache.get("page_rect") or {}

    pw = (page_rect.get("x1", 1000) - page_rect.get("x0", 0)) or 1000.0
    ph = (page_rect.get("y1", 1000) - page_rect.get("y0", 0)) or 1000.0

    img_w = (image_shape or {}).get("width",  1000)
    img_h = (image_shape or {}).get("height", 1000)

    # Scale factors: PDF points → pixels
    sx = img_w / pw
    sy = img_h / ph

    # ── Collect raw line items from thick dark paths ──────────────────────────
    raw_lines = []
    for d in drawings:
        if (d.get("width") or 0) < 0.3:
            continue
        if not _is_dark_color(d.get("color") or (0, 0, 0)):
            continue
        for item in (d.get("items") or []):
            if not item or item[0] != "l":
                continue
            # item = ("l", Point, Point)
            try:
                p1, p2 = item[1], item[2]
                raw_lines.append({
                    "x1": float(p1.x), "y1": float(p1.y),
                    "x2": float(p2.x), "y2": float(p2.y),
                    "width": d.get("width") or 1.0,
                })
            except Exception:
                continue

    if len(raw_lines) < 3:
        print(f"[_walls_from_vectors] Only {len(raw_lines)} raw lines — skipping")
        return []

    # ── Union-Find: chain segments whose endpoints are within 3 pt ───────────
    parent = list(range(len(raw_lines)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        parent[_find(a)] = _find(b)

    SNAP = 3.0  # PDF points snap tolerance
    for i in range(len(raw_lines)):
        li = raw_lines[i]
        for j in range(i + 1, len(raw_lines)):
            lj = raw_lines[j]
            for px, py in [(li["x1"], li["y1"]), (li["x2"], li["y2"])]:
                for qx, qy in [(lj["x1"], lj["y1"]), (lj["x2"], lj["y2"])]:
                    if abs(px - qx) < SNAP and abs(py - qy) < SNAP:
                        _union(i, j)
                        break

    # ── Group by component and build bounding boxes in pixel space ───────────
    groups: dict = _defaultdict(list)
    for i, line in enumerate(raw_lines):
        groups[_find(i)].append(line)

    min_px = 0.5 * scale_px_per_meter  # 0.5 m minimum wall length in pixels

    segments = []
    for grp_lines in groups.values():
        xs = [l["x1"] for l in grp_lines] + [l["x2"] for l in grp_lines]
        ys = [l["y1"] for l in grp_lines] + [l["y2"] for l in grp_lines]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        # Convert to pixel space
        bx = x0 * sx
        by = y0 * sy
        bw = max(1.0, (x1 - x0) * sx)
        bh = max(1.0, (y1 - y0) * sy)

        # Filter walls shorter than 0.5 m
        if max(bw, bh) < min_px:
            continue

        # Skip obvious page-frame lines (>80 % of image dimension)
        if bw > img_w * 0.80 or bh > img_h * 0.80:
            continue

        ratio = max(bw, bh) / max(min(bw, bh), 1)
        is_wall = ratio >= 3.0
        max_w = max(l["width"] for l in grp_lines)
        subtype = "בטון" if max_w >= 1.0 else "בלוקים"

        # Compute length_m and area_m2 (required by AutoAnalyzeSegment)
        length_px = max(bw, bh)
        length_m = round(length_px / max(scale_px_per_meter, 1.0), 3)
        area_m2 = round((bw * bh) / max(scale_px_per_meter ** 2, 1.0), 4)

        segments.append({
            "segment_id":        f"vw_{_uuid.uuid4().hex[:8]}",
            "element_class":     "wall" if is_wall else "fixture",
            "bbox":              [bx, by, bw, bh],
            "confidence":        0.85 if max_w >= 0.5 else 0.75,
            "suggested_type":    "קירות",
            "suggested_subtype": subtype,
            "wall_type":         subtype if is_wall else "interior",
            "label":             subtype,
            "length_m":          length_m,
            "area_m2":           area_m2,
            "material":          "לא_ידוע",
            "has_insulation":    False,
            "fire_resistance":   None,
            "room_name":         None,
            "area_label":        None,
            "category_color":    None,
        })

    print(f"[_walls_from_vectors] {len(segments)} wall segments from {len(raw_lines)} lines")
    return segments


# ── Spatial graph: text labels → room polygons ───────────────────────────────

def _assign_rooms_from_text(
    rooms_text: list,
    vector_cache: dict,
    wall_segs: list,
    scale_px_per_meter: float,
) -> list:
    """Match room labels from text_semantics to geometric areas.

    For each room in rooms_text:
    1. Find the word position in vector_cache["words"].
    2. Collect wall segment bboxes that surround that point within a radius.
    3. Compute geom_area_m2 from the convex hull of surrounding wall bboxes.
    4. Compute area_match_ratio = |geom - text| / text; flag if > 0.25.

    Returns enriched room list. Stores pixel_cx, pixel_cy, geom_area_m2,
    area_match_ratio, area_mismatch (bool) on each room dict.
    """
    try:
        import cv2 as _cv2
        import numpy as _np
    except ImportError:
        return rooms_text

    if not rooms_text or not vector_cache:
        return rooms_text

    words_raw  = vector_cache.get("words") or []
    page_rect  = vector_cache.get("page_rect") or {}
    pw = (page_rect.get("x1", 1000) - page_rect.get("x0", 0)) or 1000.0
    ph = (page_rect.get("y1", 1000) - page_rect.get("y0", 0)) or 1000.0

    # word positions in PDF-point space
    word_pts = []
    for w in words_raw:
        if len(w) > 4 and str(w[4]).strip():
            word_pts.append({
                "text": str(w[4]).strip(),
                "cx":   (float(w[0]) + float(w[2])) / 2,
                "cy":   (float(w[1]) + float(w[3])) / 2,
            })

    # Convert wall segment bboxes (pixel space) → PDF-point space
    px_to_pt_x = pw / max(float(scale_px_per_meter), 1)
    px_to_pt_y = ph / max(float(scale_px_per_meter), 1)

    def _bbox_of(seg):
        """Extract bbox list [x,y,w,h] from dict or object."""
        if isinstance(seg, dict):
            return seg.get("bbox")
        return getattr(seg, "bbox", None)

    enriched  = []
    mismatches = 0

    for room in rooms_text:
        room_copy  = dict(room)
        name       = room.get("name", "")
        text_area  = float(room.get("area_m2") or 0)

        # ── Find word position ────────────────────────────────────────────────
        tokens = [t for t in name.split() if len(t) > 2]
        best_pos = None
        for wp in word_pts:
            if any(tok in wp["text"] or wp["text"] in tok for tok in tokens):
                best_pos = wp
                break

        if best_pos:
            room_copy["pixel_cx"] = best_pos["cx"]
            room_copy["pixel_cy"] = best_pos["cy"]

        # ── Collect surrounding wall bboxes ───────────────────────────────────
        if best_pos and wall_segs:
            cx_pt = best_pos["cx"]
            cy_pt = best_pos["cy"]
            radius = pw * 0.35  # 35 % of page width as search radius

            near_bboxes = []
            for seg in wall_segs:
                bb = _bbox_of(seg)
                if not bb or len(bb) < 4:
                    continue
                bx_pt = bb[0] * px_to_pt_x
                by_pt = bb[1] * px_to_pt_y
                if abs(bx_pt - cx_pt) < radius and abs(by_pt - cy_pt) < radius:
                    near_bboxes.append(bb)

            if near_bboxes:
                all_pts = _np.array(
                    [[b[0], b[1]] for b in near_bboxes] +
                    [[b[0]+b[2], b[1]+b[3]] for b in near_bboxes],
                    dtype=_np.float32,
                )
                hull   = _cv2.convexHull(all_pts)
                area_px2 = float(_cv2.contourArea(hull))
                geom_m2  = round(area_px2 / max(scale_px_per_meter ** 2, 1), 2)
                room_copy["geom_area_m2"] = geom_m2

                if text_area > 0:
                    ratio = abs(geom_m2 - text_area) / text_area
                    room_copy["area_match_ratio"] = round(ratio, 3)
                    room_copy["area_mismatch"]    = ratio > 0.25
                    if ratio > 0.25:
                        mismatches += 1

        enriched.append(room_copy)

    avg = mismatches / max(len(enriched), 1)
    print(f"[_assign_rooms_from_text] {len(enriched)} rooms, {mismatches} mismatches "
          f"(avg_mismatch_rate={avg:.2f})")
    return enriched


# ── Confidence & Self-Validation Engine ──────────────────────────────────────

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _validate_segments(
    segments: list,
    rooms: list,
    walls: list,
    scale_px_per_meter: float,
) -> list:
    """Run validation rules on each segment and set confidence, flags,
    review_status.

    Rules:
    - area_match:     room geom_area vs text_area within 25 %
    - door_near_wall: fixture centre within 50 px of any wall centre
    - min_wall_length: wall length_m >= 0.3 m

    Confidence is multiplied by 0.95 when all applicable rules pass,
    reduced by 0.70 per failed rule (floor 0.10).
    review_status: "auto" >= 0.85, "medium" 0.60-0.85, "review" < 0.60 or flags.
    """
    def _cx(seg):
        bb = seg.get("bbox") if isinstance(seg, dict) else getattr(seg, "bbox", None)
        if bb and len(bb) >= 4:
            return bb[0] + bb[2] / 2
        return 0.0

    def _cy(seg):
        bb = seg.get("bbox") if isinstance(seg, dict) else getattr(seg, "bbox", None)
        if bb and len(bb) >= 4:
            return bb[1] + bb[3] / 2
        return 0.0

    def _get(seg, key, default=None):
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    def _set(seg, key, val):
        if isinstance(seg, dict):
            seg[key] = val
        else:
            setattr(seg, key, val)

    wall_centres = [(_cx(w), _cy(w)) for w in walls]

    out = []
    for seg in segments:
        conf    = float(_get(seg, "confidence", 0.75))
        flags   = []
        ec      = _get(seg, "element_class", "")
        lm      = float(_get(seg, "length_m",  0.0) or 0)

        # Rule: min_wall_length
        if ec == "wall" and lm < 0.3:
            flags.append("min_wall_length")
            conf *= 0.70

        # Rule: door_near_wall (fixture should be near a wall)
        if ec == "fixture" and wall_centres:
            fcx, fcy = _cx(seg), _cy(seg)
            if not any(_dist(fcx, fcy, wx, wy) < 50 for wx, wy in wall_centres):
                flags.append("door_near_wall")
                conf *= 0.70

        # Rule: area_match for room segments
        if ec == "room":
            geom  = float(_get(seg, "area_m2", 0) or 0)
            label = _get(seg, "area_label", "") or ""
            # Try to extract text_area from area_label "12.5 מ״ר"
            import re as _re
            m = _re.search(r'(\d+\.?\d*)', label)
            if m and geom > 0:
                text_a = float(m.group(1))
                if text_a > 0 and abs(geom - text_a) / text_a > 0.25:
                    flags.append("area_match")
                    conf *= 0.70

        if not flags:
            conf = min(conf * 1.05, 1.0)   # tiny boost when clean

        conf = max(0.10, min(conf, 1.0))

        if conf >= 0.85 and not flags:
            status = "auto"
        elif conf >= 0.60:
            status = "medium"
        else:
            status = "review"

        _set(seg, "confidence", round(conf, 3))
        _set(seg, "flags", flags)
        _set(seg, "review_status", status)
        out.append(seg)

    auto_n   = sum(1 for s in out if _get(s, "review_status") == "auto")
    medium_n = sum(1 for s in out if _get(s, "review_status") == "medium")
    review_n = sum(1 for s in out if _get(s, "review_status") == "review")
    print(f"[_validate_segments] Segments: {auto_n} auto / {medium_n} medium / {review_n} review")
    return out


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


def analyze_plan_with_vision(image_bytes: bytes, scale_text: str = "") -> dict:
    """
    שולח תמונת שרטוט ל-Claude Vision ומחלץ אלמנטים.
    מוחזר dict עם: walls, doors, windows, plumbing_fixtures, rooms,
                   materials_detected, execution_notes
    """
    client, error = get_anthropic_client()
    if error or not image_bytes:
        return {"status": "error", "error": error or "no image", "walls": [], "doors": [],
                "windows": [], "plumbing_fixtures": [], "rooms": [], "materials_detected": [], "execution_notes": []}

    img_b64 = base64.standard_b64encode(image_bytes).decode()

    prompt = f"""אתה מומחה בקריאת תוכניות בנייה ישראליות.
נתח תמונה זו של תוכנית קומה והחזר JSON בלבד.
קנה מידה: {scale_text or 'לא ידוע'}

מבנה נדרש (JSON בלבד, אין טקסט נוסף):
{{
  "walls": [{{"type": "concrete|block|gypsum|lightweight", "estimated_count": 0, "notes": ""}}],
  "doors": [{{"width_m": 0.9, "type": "single|double", "location_hint": ""}}],
  "windows": [{{"width_m": 1.2, "location_hint": ""}}],
  "plumbing_fixtures": [{{"type": "toilet|sink|kitchen_sink|bathtub|shower|bidet", "room": "", "count": 1}}],
  "rooms": [{{"name": "", "estimated_area_m2": 0, "floor_material": ""}}],
  "materials_detected": [],
  "execution_notes": [],
  "confidence": 0.0
}}"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = json.loads(text)
        result["status"] = "success"
        return result
    except Exception as e:
        logger.warning("vision analysis failed: %s", e)
        return {"status": "error", "error": str(e), "walls": [], "doors": [],
                "windows": [], "plumbing_fixtures": [], "rooms": [], "materials_detected": [], "execution_notes": []}


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
