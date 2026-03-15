"""
vector_extractor.py — מחלץ קירות ואביזרים ישירות מ-PDF וקטורי.
מחזיר רשימת segments תואמת ל-AutoAnalyzeResponse.segments.
"""
import uuid
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore
from typing import List, Dict, Optional, Tuple


# ── מיפוי צבע→קטגוריה (CAD conventions) ─────────────────────────────────────
LAYER_COLOR_MAP: Dict[Tuple, Tuple[str, str]] = {
    (0.0, 0.0, 0.0): ("קירות", "בטון"),
    (0.1, 0.1, 0.1): ("קירות", "בלוקים"),
    (0.0, 0.0, 1.0): ("חשמל ותאורה", "תאורה"),
    (0.0, 0.5, 1.0): ("חשמל ותאורה", "שקעים"),
    (1.0, 0.0, 0.0): ("כלים סניטריים", "כיור"),
    (0.0, 0.5, 0.0): ("כלים סניטריים", "כיור ירוק"),
    (0.0, 1.0, 0.0): ("כלים סניטריים", "כיור ירוק"),
    (0.0, 1.0, 1.0): ("מיזוג ואוורור", "מזגן"),
    (1.0, 0.0, 1.0): ("כלים סניטריים", "כיור"),
    (1.0, 1.0, 0.0): ("כלים סניטריים", "כיור"),
}

WALL_STROKE_THRESHOLD = 0.28
FIXTURE_AREA_SMALL   = 0.10
FIXTURE_AREA_SINK    = 0.40
FIXTURE_AREA_BATH    = 1.20

# ── שכבות OCG שיש לדלג עליהן (Revit/ArchiCAD layer names) ───────────────────
_EXCLUDED_LAYERS = {
    "dimensions", "dim", "grids", "grid", "annotations", "anno",
    "text", "tags", "notes", "titleblock", "title block",
    "a-anno-dims", "a-anno-grid", "a-anno-note",
}


def _round_color(c: Optional[tuple], decimals: int = 1) -> Tuple:
    if not c or len(c) < 3:
        return (0.0, 0.0, 0.0)
    return tuple(round(float(v), decimals) for v in c[:3])


def _classify_by_color(color: tuple, stroke_w: float, width_m: float, height_m: float) -> Tuple[str, str]:
    """מיפוי צבע+עובי → (type, subtype)."""
    r, g, b = _round_color(color)
    key = (round(r), round(g), round(b))
    if key in LAYER_COLOR_MAP:
        return LAYER_COLOR_MAP[key]

    area_m2 = width_m * height_m

    if b > 0.4 and r < 0.3 and g < 0.3:
        return "חשמל ותאורה", "תאורה"
    if r > 0.4 and g < 0.3 and b < 0.3:
        return "כלים סניטריים", "כיור"
    if g > 0.4 and r < 0.3 and b < 0.3:
        return "כלים סניטריים", "כיור"

    is_dark = r < 0.25 and g < 0.25 and b < 0.25
    if is_dark:
        if stroke_w >= 0.5:
            return "קירות", "בטון"
        elif stroke_w >= 0.35:
            return "קירות", "בלוקים"
        else:
            return "קירות", "גבס"

    if area_m2 < FIXTURE_AREA_SMALL:
        return "פרטים", "פרט קטן"
    if area_m2 < FIXTURE_AREA_SINK:
        return "כלים סניטריים", "כיור"
    if area_m2 < FIXTURE_AREA_BATH:
        return "כלים סניטריים", "אמבטיה"

    return "קירות", "בלוקים"


def _drawing_to_segment(
    drawing: dict,
    sx: float,
    sy: float,
    scale_px_per_meter: float,
    img_w: int,
    img_h: int,
    seg_idx: int,
) -> Optional[dict]:
    """ממיר drawing אחד מ-PyMuPDF לסגמנט."""
    # ── פילטר OCG שכבה ──
    layer_name = (drawing.get("layer") or "").lower().strip()
    if any(excl in layer_name for excl in _EXCLUDED_LAYERS):
        return None

    # ── פילטר Dashed lines (קווי מידה / ציר / מקוקו) ──
    if drawing.get("dashes"):
        return None

    stroke_w = drawing.get("width") or 0.0
    color    = drawing.get("color") or (0.0, 0.0, 0.0)
    rect     = drawing.get("rect")

    if rect is None:
        return None

    # ── פילטר: קירות בלבד — מסנן קווי מידות, האצ'ינג, גריד ──
    # קירות בלבד ≥ 1.0pt; מידות/האצ'ינג בדרך כלל ≤ 0.5pt
    if stroke_w < 1.0:
        return None

    bx = float(rect.x0) * sx
    by = float(rect.y0) * sy
    bw = max(1.0, float(rect.width)  * sx)
    bh = max(1.0, float(rect.height) * sy)

    if bw < 4 and bh < 4:
        return None
    if bx > img_w or by > img_h:
        return None

    width_m  = bw / max(scale_px_per_meter, 1)
    height_m = bh / max(scale_px_per_meter, 1)
    ratio    = max(bw, bh) / max(min(bw, bh), 1)
    is_wall  = ratio >= 3.0

    # ── פילטר אורך: קיר מינימלי 0.8 מטר ──
    length_m_check = max(width_m, height_m)
    if length_m_check < 0.8:
        return None

    # ── פילטר: shapes ריבועיות קטנות אינן קירות ──
    area_m2_check = width_m * height_m
    if ratio < 2.5 and area_m2_check < FIXTURE_AREA_SMALL:
        return None
    element_class = "wall" if is_wall else "fixture"

    stype, ssubtype = _classify_by_color(color, stroke_w, width_m, height_m)

    conf = 0.75
    if stroke_w >= 0.5:
        conf = 0.92
    elif stroke_w >= 0.35:
        conf = 0.85
    elif not is_wall:
        conf = 0.70

    length_m = round(max(width_m, height_m), 3)
    area_m2  = round(width_m * height_m, 4)

    return {
        "segment_id":        f"vec_{seg_idx:04d}_{uuid.uuid4().hex[:6]}",
        "element_class":     element_class,
        "bbox":              [bx, by, bw, bh],
        "confidence":        conf,
        "suggested_type":    stype,
        "suggested_subtype": ssubtype,
        "wall_type":         ssubtype if element_class == "wall" else "interior",
        "label":             ssubtype,
        "length_m":          length_m,
        "area_m2":           area_m2,
        "material":          "לא_ידוע",
        "has_insulation":    False,
        "fire_resistance":   None,
        "room_name":         None,
        "area_label":        None,
        "category_color":    None,
    }


def _build_dimension_zones(vector_cache: dict, sx: float, sy: float, margin: float = 15.0) -> list:
    """מזהה מספרי מידה בטקסט ומחזיר bbox zones לסינון קווי מידות סמוכים."""
    import re
    zones = []
    words = vector_cache.get("words") or []
    dim_pattern = re.compile(r"^\d{2,4}$")  # מספרים 2-4 ספרות (מידות בס"מ / ס"מ)
    for w in words:
        text = (w.get("text") or "").strip() if isinstance(w, dict) else (str(w[4]) if len(w) > 4 else "")
        if dim_pattern.match(text):
            bbox = w.get("bbox") if isinstance(w, dict) else (w[0], w[1], w[2], w[3])
            if bbox and len(bbox) >= 4:
                zones.append((
                    bbox[0] * sx - margin,
                    bbox[1] * sy - margin,
                    bbox[2] * sx + margin,
                    bbox[3] * sy + margin,
                ))
    return zones


def _point_in_zones(px: float, py: float, zones: list) -> bool:
    for (x0, y0, x1, y1) in zones:
        if x0 <= px <= x1 and y0 <= py <= y1:
            return True
    return False


def extract_from_pdf(
    pdf_bytes: bytes,
    scale_px_per_meter: float,
    image_shape: dict,
    filtered_indices: set = None,
    region_data: dict = None,
) -> List[dict]:
    """
    קלט: bytes של PDF + scale + מידות תמונה.
    פלט: רשימת segments, או [] אם הקובץ לא וקטורי.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"[vector_extractor] fitz.open failed: {e}")
        return []

    page      = doc[0]
    page_rect = page.rect

    if page_rect.width < 1 or page_rect.height < 1:
        return []

    img_w = image_shape.get("width",  1000)
    img_h = image_shape.get("height", 1000)
    sx    = img_w  / page_rect.width
    sy    = img_h  / page_rect.height

    drawings = page.get_drawings()
    thick    = [d for d in drawings if (d.get("width") or 0) >= WALL_STROKE_THRESHOLD]

    if len(thick) < 3:
        print(f"[vector_extractor] Only {len(thick)} thick paths — likely raster, fallback to OpenCV")
        return []

    # בנה zones של מספרי מידה לסינון קווי מידות
    _words_raw = page.get_text("words")  # list of (x0, y0, x1, y1, text, ...)
    _words_for_zones = [
        {"text": w[4], "bbox": (w[0], w[1], w[2], w[3])}
        for w in _words_raw if len(w) >= 5
    ]
    _vc_words = {"words": _words_for_zones}
    dim_zones = _build_dimension_zones(_vc_words, sx, sy, margin=15.0)

    # שמור original_idx → drawing mapping לפילטר filtered_indices
    _fi = set(filtered_indices) if filtered_indices else set()

    # Phase 2: region gate data
    _main_region: Optional[list] = None
    _excl_regions: list = []
    if region_data:
        _main_region = region_data.get("main_drawing_region")
        _excl_regions = region_data.get("excluded_regions", [])

    _region_gate_available = False
    _imd_fn = None
    _bie_fn = None
    if _main_region or _excl_regions:
        try:
            from engines import is_in_main_drawing as _imd_fn, bbox_in_excluded as _bie_fn
            _region_gate_available = True
        except Exception:
            pass

    segments = []
    for idx, d in enumerate(thick):
        # פילטר לפי annotation_filter אם סופק
        if idx in _fi:
            continue
        seg = _drawing_to_segment(d, sx, sy, scale_px_per_meter, img_w, img_h, idx)
        if seg is None:
            continue
        # סנן segments שפינותיהם חופפות אזור מספרי מידה
        bx, by, bw, bh = seg["bbox"]
        corners = [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]
        if any(_point_in_zones(cx, cy, dim_zones) for cx, cy in corners):
            continue
        # Phase 2: region gate
        if _region_gate_available:
            if _main_region and not _imd_fn(seg["bbox"], _main_region, threshold=0.25):
                continue
            if _excl_regions and _bie_fn(seg["bbox"], _excl_regions, threshold=0.5):
                continue
        segments.append(seg)

    segments = _deduplicate(segments, threshold_px=6.0)
    segments = _filter_page_frame(segments, img_w, img_h)

    print(f"[vector_extractor] Extracted {len(segments)} segments from {len(thick)} thick paths")
    return segments


def _deduplicate(segments: List[dict], threshold_px: float = 6.0) -> List[dict]:
    """מסיר כפילויות (bbox כמעט-זהה בטווח threshold_px)."""
    kept = []
    for seg in segments:
        bx, by, bw, bh = seg["bbox"]
        duplicate = False
        for k in kept:
            kx, ky, kw, kh = k["bbox"]
            if (abs(bx - kx) < threshold_px and abs(by - ky) < threshold_px and
                    abs(bw - kw) < threshold_px and abs(bh - kh) < threshold_px):
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
    return kept


def _filter_page_frame(segments: List[dict], img_w: int, img_h: int) -> List[dict]:
    """מסנן קווי מסגרת עמוד (>60% רוחב/גובה בשוליים 5%)."""
    filtered = []
    margin_x = img_w * 0.05
    margin_y = img_h * 0.05
    for seg in segments:
        bx, by, bw, bh = seg["bbox"]
        is_top    = by < margin_y            and bw > img_w * 0.60
        is_bottom = (by + bh) > img_h - margin_y and bw > img_w * 0.60
        is_left   = bx < margin_x            and bh > img_h * 0.60
        is_right  = (bx + bw) > img_w - margin_x and bh > img_h * 0.60
        if not (is_top or is_bottom or is_left or is_right):
            filtered.append(seg)
    return filtered
