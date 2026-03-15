"""
engines.py — מנועי pre-processing לניתוח תוכניות אדריכליות.

Phase 1: 5 מנועים שרצים לפני זיהוי הקירות:
  Engine 2 — Document Classification
  Engine 3 — Region Segmentation
  Engine 4 — Text Semantics
  Engine 5 — Annotation Filtering
"""
import re
import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ========================
# ENGINE 2 — DOCUMENT CLASSIFICATION
# ========================

_MEP_KEYWORDS_HE = {
    "אינסטלציה", "צנרת", "חשמל", "תאורה", "מיזוג", "אוורור",
    "גז", "ספרינקל", "מ.כ", "תקשורת", "ביוב", "מים וביוב"
}
_STRUCT_KEYWORDS_HE = {"קונסטרוקציה", "עמודים", "קורות", "יסודות", "פלדה", "בטון מזוין"}
_ARCH_KEYWORDS_HE = {"תוכנית קומה", "חלוקת", "משרד", "מגורים", "חדר", "כניסה"}
_MEP_KEYWORDS_EN = {"plumbing", "electrical", "hvac", "mechanical", "fire", "sprinkler"}
_STRUCT_KEYWORDS_EN = {"structural", "foundation", "column", "beam", "rebar"}


def classify_document(text_cache: Dict, vector_cache: Dict, page_rect: Dict) -> Dict:
    """
    מסווג את סוג הדף: architectural | mep | structural | site | unknown.

    מחזיר:
        sheet_type:       str
        sheet_confidence: float 0.0–1.0
        sheet_hints:      list[str]
    """
    words_raw = text_cache.get("words", [])
    all_words_lower = set()
    for w in words_raw:
        if len(w) > 4:
            all_words_lower.add(str(w[4]).lower())

    scores = {"architectural": 0.1, "mep": 0.0, "structural": 0.0, "site": 0.0}
    hints: List[str] = []

    for kw in _MEP_KEYWORDS_HE:
        if any(kw in w for w in all_words_lower):
            scores["mep"] += 0.3
            hints.append(f"mep_keyword:{kw}")
    for kw in _STRUCT_KEYWORDS_HE:
        if any(kw in w for w in all_words_lower):
            scores["structural"] += 0.3
            hints.append(f"struct_keyword:{kw}")
    for kw in _MEP_KEYWORDS_EN:
        if kw in all_words_lower:
            scores["mep"] += 0.2

    drawings = vector_cache.get("drawings", [])
    colored_count = 0
    black_count = 0
    for d in drawings[:200]:
        c = d.get("color") or d.get("stroke_color") or (0, 0, 0)
        if isinstance(c, (list, tuple)) and len(c) == 3:
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            if r > 0.3 or g > 0.3 or b > 0.3:
                if not (r > 0.8 and g > 0.8 and b > 0.8):
                    colored_count += 1
            else:
                black_count += 1

    if colored_count > 0:
        color_ratio = colored_count / max(black_count + colored_count, 1)
        if color_ratio > 0.4:
            scores["mep"] += 0.3
            hints.append(f"high_color_ratio:{color_ratio:.2f}")

    best = max(scores, key=scores.__getitem__)
    confidence = min(scores[best], 1.0)

    return {
        "sheet_type": best,
        "sheet_confidence": round(confidence, 2),
        "sheet_hints": hints,
    }


# ========================
# ENGINE 3 — REGION SEGMENTATION
# ========================

def segment_regions(text_cache: Dict, vector_cache: Dict, page_rect: Dict) -> Dict:
    """
    מזהה אזורים פונקציונליים של הדף:
        main_drawing_region: [x0, y0, x1, y1]
        excluded_regions:    list[[x0, y0, x1, y1]]
        legend_region:       [x0, y0, x1, y1] | None
        title_block_region:  [x0, y0, x1, y1] | None
    """
    pw = page_rect.get("w", 2480)
    ph = page_rect.get("h", 3508)

    words = text_cache.get("words", [])
    excluded: List = []
    legend_region = None
    title_block_region = None

    # Title Block Detection (RTL: bottom ~15%)
    bottom_y = ph * 0.85
    bottom_words = [w for w in words if len(w) > 1 and float(w[1]) >= bottom_y]

    if len(bottom_words) > 15:
        xs = [float(w[0]) for w in bottom_words] + [float(w[2]) for w in bottom_words]
        ys = [float(w[1]) for w in bottom_words] + [float(w[3]) for w in bottom_words]
        tb = [min(xs), min(ys), max(xs), max(ys)]
        title_block_region = tb
        excluded.append(tb)

    # Legend Detection
    legend_candidates = _find_legend_candidate(words, vector_cache, pw, ph)
    if legend_candidates:
        legend_region = legend_candidates[0]
        excluded.append(legend_region)

    # Revision Table (strip above title block)
    if title_block_region:
        rev_region = [
            title_block_region[0],
            max(0, title_block_region[1] - ph * 0.08),
            title_block_region[2],
            title_block_region[1],
        ]
        excluded.append(rev_region)

    # Main Drawing Region
    if excluded:
        all_excl_y0 = min(r[1] for r in excluded)
        main = [0, 0, pw, all_excl_y0]
    else:
        main = [0, 0, pw, ph]

    return {
        "main_drawing_region": main,
        "excluded_regions": excluded,
        "legend_region": legend_region,
        "title_block_region": title_block_region,
    }


def _find_legend_candidate(words, vector_cache, pw, ph):
    candidates = []
    zones = [
        [0, 0, pw * 0.25, ph * 0.35],
        [pw * 0.75, 0, pw, ph * 0.35],
        [0, ph * 0.65, pw * 0.25, ph * 0.85],
        [pw * 0.75, ph * 0.65, pw, ph * 0.85],
    ]
    for zone in zones:
        zx0, zy0, zx1, zy1 = zone
        zone_words = [w for w in words if len(w) > 1 and zx0 <= float(w[0]) <= zx1 and zy0 <= float(w[1]) <= zy1]
        if len(zone_words) < 5:
            continue
        avg_word_len = sum(len(str(w[4])) for w in zone_words if len(w) > 4) / len(zone_words)
        drawings = vector_cache.get("drawings", [])
        short_lines = 0
        for d in drawings:
            r = d.get("rect")
            if r:
                cx = (float(r[0]) + float(r[2])) / 2
                cy = (float(r[1]) + float(r[3])) / 2
                dw = abs(float(r[2]) - float(r[0]))
                dh = abs(float(r[3]) - float(r[1]))
                if zx0 <= cx <= zx1 and zy0 <= cy <= zy1:
                    if (dw < pw * 0.05 and dh < 5) or (dh < ph * 0.03 and dw < 5):
                        short_lines += 1
        if avg_word_len < 12 and short_lines >= 3:
            candidates.append(zone)
    return candidates


def point_in_excluded(px: float, py: float, excluded_regions: List) -> bool:
    for r in excluded_regions:
        if r[0] <= px <= r[2] and r[1] <= py <= r[3]:
            return True
    return False


def bbox_in_excluded(bbox: List, excluded_regions: List, threshold: float = 0.5) -> bool:
    if not excluded_regions:
        return False
    bx0, by0, bw, bh = bbox
    bx1, by1 = bx0 + bw, by0 + bh
    b_area = bw * bh
    if b_area == 0:
        return True
    for r in excluded_regions:
        ix0 = max(bx0, r[0])
        iy0 = max(by0, r[1])
        ix1 = min(bx1, r[2])
        iy1 = min(by1, r[3])
        if ix1 > ix0 and iy1 > iy0:
            inter = (ix1 - ix0) * (iy1 - iy0)
            if inter / b_area >= threshold:
                return True
    return False


# ========================
# ENGINE 4 — TEXT SEMANTICS
# ========================

_AREA_PATTERN = re.compile(r'(\d+\.?\d*)\s*מ[׳\'"]?\s*[²2]?')
_ROOM_TAGS = {
    "חדר שינה", "חד' שינה", "ח. שינה",
    "סלון", "מטבח", "פינת אוכל", "שירותים", "מ\"ק", "מבוא",
    "מסדרון", "מרפסת", "חדרפת", "פינת ישיבה",
    "חדר עבודה", "חד' עבודה"
}
_SCALE_PATTERN = re.compile(r'1\s*[:/]\s*(\d+)')
_DOOR_TAG_PATTERN = re.compile(r'^D[-_]?\d+$|^דלת\s*\d+$', re.IGNORECASE)
_WINDOW_TAG_PATTERN = re.compile(r'^W[-_]?\d+$|^חלון\s*\d+$', re.IGNORECASE)


def extract_text_semantics(text_cache: Dict, page_rect: Dict) -> Dict:
    """
    מחלץ סמנטיקה מהטקסט של ה-PDF.

    מחזיר:
        rooms:       [{name, bbox, area_m2?}]
        scale:       int | None
        door_tags:   [{tag, bbox}]
        window_tags: [{tag, bbox}]
        area_texts:  [{value, unit, bbox}]
        text_zones:  [{bbox, text}]
    """
    words = text_cache.get("words", [])
    rooms: List[Dict] = []
    scale = None
    door_tags: List[Dict] = []
    window_tags: List[Dict] = []
    area_texts: List[Dict] = []

    lines = _merge_words_to_lines(words)

    for line in lines:
        text = line["text"].strip()
        bbox = line["bbox"]
        if not text:
            continue

        scale_match = _SCALE_PATTERN.search(text)
        if scale_match and scale is None:
            try:
                scale = int(scale_match.group(1))
            except ValueError:
                pass

        text_lower = text.lower()
        for tag in _ROOM_TAGS:
            if tag in text or tag.lower() in text_lower:
                rooms.append({"name": text, "bbox": bbox, "raw_tag": tag})
                break

        if _DOOR_TAG_PATTERN.match(text.strip()):
            door_tags.append({"tag": text.strip(), "bbox": bbox})

        if _WINDOW_TAG_PATTERN.match(text.strip()):
            window_tags.append({"tag": text.strip(), "bbox": bbox})

        area_match = _AREA_PATTERN.search(text)
        if area_match:
            try:
                area_texts.append({"value": float(area_match.group(1)), "unit": "m2", "bbox": bbox})
            except ValueError:
                pass

    # שייך שטחים לחדרים קרובים
    for room in rooms:
        rb = room["bbox"]
        rx = (rb[0] + rb[2]) / 2
        ry = (rb[1] + rb[3]) / 2
        pw = page_rect.get("w", 2480)
        best_area, best_dist = None, float("inf")
        for at in area_texts:
            ab = at["bbox"]
            ax, ay = (ab[0] + ab[2]) / 2, (ab[1] + ab[3]) / 2
            dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)
            if dist < pw * 0.08 and dist < best_dist:
                best_dist = dist
                best_area = at["value"]
        if best_area is not None:
            room["area_m2"] = best_area

    text_zones = [{"bbox": ln["bbox"], "text": ln["text"]} for ln in lines]

    return {
        "rooms": rooms,
        "scale": scale,
        "door_tags": door_tags,
        "window_tags": window_tags,
        "area_texts": area_texts,
        "text_zones": text_zones,
    }


def _merge_words_to_lines(words: List) -> List[Dict]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (round(float(w[1]) / 4) * 4, float(w[0])))
    lines: List[List] = []
    current: List = []
    for w in sorted_words:
        if not current:
            current = [w]
            continue
        prev = current[-1]
        if abs(float(w[1]) - float(prev[1])) <= 6:
            if float(w[0]) - float(prev[2]) <= 15:
                current.append(w)
            else:
                lines.append(current)
                current = [w]
        else:
            lines.append(current)
            current = [w]
    if current:
        lines.append(current)
    return [_line_from_words(ln) for ln in lines]


def _line_from_words(words: List) -> Dict:
    text = " ".join(str(w[4]) for w in words if len(w) > 4)
    x0 = min(float(w[0]) for w in words)
    y0 = min(float(w[1]) for w in words)
    x1 = max(float(w[2]) for w in words)
    y1 = max(float(w[3]) for w in words)
    return {"text": text, "bbox": [x0, y0, x1, y1]}


# ========================
# ENGINE 5 — ANNOTATION FILTERING
# ========================

_EXCLUDED_LAYER_KEYWORDS = {
    "dim", "dimension", "anno", "annotation", "grid", "grids",
    "text", "tag", "note", "reference", "ref", "callout",
    "section", "elevation", "revision", "border", "titleblock",
    "title", "hatch", "pattern", "detail",
}


def build_annotation_filter(
    vector_cache: Dict,
    text_semantics: Dict,
    region_data: Dict,
    page_rect: Dict,
) -> Dict:
    """
    בונה מסנן annotations — מחזיר set של drawing indices שיש לדלג עליהם.

    קריטריונים:
    1. dashed line
    2. stroke_width < 0.8pt
    3. OCG layer name של annotation
    4. bbox נמצא ב-excluded region (legend/title_block)
    5. bbox קטן מאוד (< 5px) — dot/tick
    6. מרכז ה-path קרוב לטקסט מספרי (± zone מידה)
    """
    drawings = vector_cache.get("drawings", [])
    text_zones = text_semantics.get("text_zones", [])
    excluded_regions = region_data.get("excluded_regions", [])

    pw = page_rect.get("w", 2480)

    dim_zones = _build_dimension_zones_from_text(text_zones, pw)
    filtered_indices: set = set()

    for idx, d in enumerate(drawings):
        # 1. Dashed
        dashes = d.get("dashes") or d.get("dash_pattern") or []
        if dashes:
            filtered_indices.add(idx)
            continue

        # 2. Stroke width thin
        sw = d.get("width") or d.get("stroke_width") or 0
        if isinstance(sw, (int, float)) and 0 < float(sw) < 0.8:
            filtered_indices.add(idx)
            continue

        # 3. OCG layer
        layer = str(d.get("layer") or d.get("ocg") or "").lower()
        if layer and any(kw in layer for kw in _EXCLUDED_LAYER_KEYWORDS):
            filtered_indices.add(idx)
            continue

        r = d.get("rect")
        if r:
            bx0, by0 = float(r[0]), float(r[1])
            bx1, by1 = float(r[2]), float(r[3])
            bw = abs(bx1 - bx0)
            bh = abs(by1 - by0)

            # 4. bbox too small
            if max(bw, bh) < 5:
                filtered_indices.add(idx)
                continue

            # 5. In excluded region
            if bbox_in_excluded([bx0, by0, bw, bh], excluded_regions, threshold=0.6):
                filtered_indices.add(idx)
                continue

            # 6. Near dimension text
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2
            if _point_in_dim_zones(cx, cy, dim_zones):
                filtered_indices.add(idx)
                continue

    pass_count = len(drawings) - len(filtered_indices)
    logger.info(
        f"[annotation_filter] drawings={len(drawings)} "
        f"filtered={len(filtered_indices)} pass={pass_count}"
    )

    return {
        "filtered_indices": filtered_indices,
        "total_drawings": len(drawings),
        "filtered_count": len(filtered_indices),
        "pass_count": pass_count,
    }


def _build_dimension_zones_from_text(text_zones: List, pw: float) -> List:
    zones = []
    EXPAND = max(30, pw * 0.015)
    for tz in text_zones:
        text = tz.get("text", "")
        stripped = re.sub(r'[\s\.,\-\']', '', text)
        if stripped.isdigit() or re.match(r'^\d+[,\.]\d+$', text.strip()):
            b = tz["bbox"]
            zones.append([b[0] - EXPAND, b[1] - EXPAND, b[2] + EXPAND, b[3] + EXPAND])
    return zones


def _point_in_dim_zones(px: float, py: float, zones: List) -> bool:
    for z in zones:
        if z[0] <= px <= z[2] and z[1] <= py <= z[3]:
            return True
    return False


# ========================
# UTILITY
# ========================

def get_page_rect_wh(vector_cache: Dict) -> Dict:
    """ממיר page_rect מפורמט fitz ({x0,y0,x1,y1}) לפורמט engines ({w,h})."""
    pr = vector_cache.get("page_rect") or {}
    x0 = float(pr.get("x0", 0))
    y0 = float(pr.get("y0", 0))
    x1 = float(pr.get("x1", pr.get("w", 2480)))
    y1 = float(pr.get("y1", pr.get("h", 3508)))
    return {"w": x1 - x0, "h": y1 - y0}
