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

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NP_AVAILABLE = False

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
    # Cap to avoid O(N) startup cost on large PDFs
    MAX_DRAWINGS_FILTER = 2000
    if len(drawings) > MAX_DRAWINGS_FILTER:
        print(f"[build_annotation_filter] capping {len(drawings)} → {MAX_DRAWINGS_FILTER} drawings")
        drawings = drawings[:MAX_DRAWINGS_FILTER]
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


# ════════════════════════════════════════════════
# PHASE 2 — GEOMETRIC GATING + ANCHOR VALIDATION
# ════════════════════════════════════════════════

# ──────────────────────────────────────────────
# ENGINE P2-1 — MAIN DRAWING GATE ENGINE
# ──────────────────────────────────────────────

def refine_main_drawing_region(
    vector_cache: Dict,
    text_semantics: Dict,
    region_data: Dict,
    page_rect: Dict,
    preview_image=None,       # numpy BGR array, optional
) -> Dict:
    """
    מחשב את אזור השרטוט הראשי האמיתי לפי צפיפות פרימיטיבים.

    מחזיר:
      main_drawing_region: [x0, y0, x1, y1] מדויק יותר
      main_drawing_confidence: 0.0-1.0
      density_map: dict {cell_key: density}
    """
    pw = page_rect.get("w", 2480)
    ph = page_rect.get("h", 3508)
    drawings = vector_cache.get("drawings", [])
    excluded_regions = region_data.get("excluded_regions", [])

    CELL_SIZE = max(50, pw / 40)
    cols = max(1, int(pw / CELL_SIZE))
    rows = max(1, int(ph / CELL_SIZE))

    density_grid: Dict = {}
    for d in drawings:
        r = d.get("rect")
        if not r:
            continue
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        col = min(int(cx / CELL_SIZE), cols - 1)
        row = min(int(cy / CELL_SIZE), rows - 1)
        key = (row, col)
        density_grid[key] = density_grid.get(key, 0) + 1

    if not density_grid:
        return {
            "main_drawing_region": region_data.get("main_drawing_region", [0, 0, pw, ph]),
            "main_drawing_confidence": 0.3,
            "density_map": {}
        }

    max_density = max(density_grid.values())
    threshold = max(2, max_density * 0.15)
    active_cells = [(r, c) for (r, c), d in density_grid.items() if d >= threshold]

    if not active_cells:
        return {
            "main_drawing_region": region_data.get("main_drawing_region", [0, 0, pw, ph]),
            "main_drawing_confidence": 0.3,
            "density_map": density_grid
        }

    min_row = min(r for r, c in active_cells)
    max_row = max(r for r, c in active_cells)
    min_col = min(c for r, c in active_cells)
    max_col = max(c for r, c in active_cells)

    MARGIN = CELL_SIZE * 0.5
    x0 = max(0, min_col * CELL_SIZE - MARGIN)
    y0 = max(0, min_row * CELL_SIZE - MARGIN)
    x1 = min(pw, (max_col + 1) * CELL_SIZE + MARGIN)
    y1 = min(ph, (max_row + 1) * CELL_SIZE + MARGIN)

    raw_area = max(1, (x1 - x0) * (y1 - y0))
    total_excl_in_main = 0
    for er in excluded_regions:
        ix0 = max(x0, er[0]); iy0 = max(y0, er[1])
        ix1 = min(x1, er[2]); iy1 = min(y1, er[3])
        if ix1 > ix0 and iy1 > iy0:
            total_excl_in_main += (ix1 - ix0) * (iy1 - iy0)

    confidence = 1.0 - min(0.8, total_excl_in_main / raw_area)

    return {
        "main_drawing_region": [round(x0), round(y0), round(x1), round(y1)],
        "main_drawing_confidence": round(confidence, 2),
        "density_map": {f"{r},{c}": d for (r, c), d in density_grid.items()},
    }


def is_in_main_drawing(bbox: List, main_region: List, threshold: float = 0.3) -> bool:
    """בודק האם מרכז ה-bbox (או לפחות threshold% ממנו) נמצא ב-main_drawing_region."""
    if not main_region:
        return True

    bx0, by0, bw, bh = bbox
    bx1, by1 = bx0 + bw, by0 + bh
    b_area = max(1, bw * bh)

    cx = bx0 + bw / 2
    cy = by0 + bh / 2
    mr = main_region
    if mr[0] <= cx <= mr[2] and mr[1] <= cy <= mr[3]:
        return True

    ix0 = max(bx0, mr[0]); iy0 = max(by0, mr[1])
    ix1 = min(bx1, mr[2]); iy1 = min(by1, mr[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False

    overlap = (ix1 - ix0) * (iy1 - iy0)
    return (overlap / b_area) >= threshold


# ──────────────────────────────────────────────
# ENGINE P2-2 — EXCLUSION GATE ENGINE
# ──────────────────────────────────────────────

def compute_exclusion_gate(
    bbox: List,
    excluded_regions: List,
    reject_threshold: float = 0.40,
    penalty_threshold: float = 0.20,
) -> Dict:
    """
    מחשב ציון חפיפה עם אזורים מוחרגים ומחזיר החלטה.

    מחזיר:
      exclusion_overlap: float (0-1)
      exclusion_decision: "pass" | "penalty" | "reject"
      exclusion_reason: str
    """
    if not excluded_regions:
        return {"exclusion_overlap": 0.0, "exclusion_decision": "pass", "exclusion_reason": ""}

    bx0, by0, bw, bh = bbox
    bx1, by1 = bx0 + bw, by0 + bh
    b_area = max(1, bw * bh)
    cx = bx0 + bw / 2
    cy = by0 + bh / 2

    max_overlap_ratio = 0.0

    for r in excluded_regions:
        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
            return {
                "exclusion_overlap": 1.0,
                "exclusion_decision": "reject",
                "exclusion_reason": "center_in_excluded_region"
            }

        ix0 = max(bx0, r[0]); iy0 = max(by0, r[1])
        ix1 = min(bx1, r[2]); iy1 = min(by1, r[3])
        if ix1 > ix0 and iy1 > iy0:
            overlap_ratio = (ix1 - ix0) * (iy1 - iy0) / b_area
            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio

    if max_overlap_ratio >= reject_threshold:
        return {
            "exclusion_overlap": round(max_overlap_ratio, 3),
            "exclusion_decision": "reject",
            "exclusion_reason": f"bbox_overlap_{max_overlap_ratio:.0%}_with_excluded"
        }
    elif max_overlap_ratio >= penalty_threshold:
        return {
            "exclusion_overlap": round(max_overlap_ratio, 3),
            "exclusion_decision": "penalty",
            "exclusion_reason": f"partial_overlap_{max_overlap_ratio:.0%}"
        }
    else:
        return {
            "exclusion_overlap": round(max_overlap_ratio, 3),
            "exclusion_decision": "pass",
            "exclusion_reason": ""
        }


# ──────────────────────────────────────────────
# ENGINE P2-3 — PRIMITIVE FAMILY GATE ENGINE
# ──────────────────────────────────────────────

PRIM_WALL = "wall_candidate"
PRIM_OPENING = "opening_candidate"
PRIM_DIMENSION = "dimension_line"
PRIM_GRID = "grid_line"
PRIM_LEADER = "leader_line"
PRIM_SECTION = "section_marker_line"
PRIM_TABLE = "table_border"
PRIM_LEGEND = "legend_sample"
PRIM_SITE = "site_boundary"
PRIM_BUILDING = "building_line"
PRIM_HATCH = "hatch_pattern"
PRIM_UNKNOWN = "unknown"

FORBIDDEN_FAMILIES = {PRIM_DIMENSION, PRIM_GRID, PRIM_TABLE, PRIM_LEGEND, PRIM_SECTION, PRIM_HATCH}

_GRID_AXIS_PATTERN = re.compile(r'^[A-Z]$|^\d{1,2}$')
_UKOK_PATTERN = re.compile(r'^(UK|OK|U\.K\.|O\.K\.)$', re.IGNORECASE)
_SECTION_PATTERN = re.compile(r'^[A-Z]-?S\.?\d|^S-\d', re.IGNORECASE)


def classify_primitive_families(
    vector_cache: Dict,
    text_semantics: Dict,
    region_data: Dict,
    page_rect: Dict,
) -> Dict:
    """
    מסווג כל drawing primitive למשפחה.

    מחזיר:
      family_labels: dict {drawing_index: family_str}
      forbidden_indices: set של indices שלא צריכים להיכנס לזיהוי
      family_counts: dict {family_str: count}
    """
    drawings = vector_cache.get("drawings", [])
    # Cap at 1500 drawings to avoid O(N²) slowdown on large PDFs (Phase 2 guard)
    MAX_DRAWINGS = 1500
    if len(drawings) > MAX_DRAWINGS:
        print(f"[classify_primitive_families] capping {len(drawings)} → {MAX_DRAWINGS} drawings")
        drawings = drawings[:MAX_DRAWINGS]
    text_zones = text_semantics.get("text_zones", [])
    legend_region = region_data.get("legend_region")
    excluded_regions = region_data.get("excluded_regions", [])
    pw = page_rect.get("w", 2480)
    ph = page_rect.get("h", 3508)

    numeric_text_zones: List = []
    grid_text_zones: List = []
    section_zones: List = []

    for tz in text_zones:
        t = tz.get("text", "").strip()
        b = tz["bbox"]
        stripped = re.sub(r'[\s\.,\-\'\"]', '', t)

        if stripped.isdigit() or re.match(r'^\d+[,\.]\d+$', t.strip()):
            numeric_text_zones.append(b)
        if _GRID_AXIS_PATTERN.match(t.strip()):
            grid_text_zones.append(b)
        if _SECTION_PATTERN.match(t.strip()):
            section_zones.append(b)

    PROX = max(40, pw * 0.016)

    def _bbox_near_zones(r, zones, expand=PROX):
        if not r:
            return False
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        for z in zones:
            if (z[0] - expand) <= cx <= (z[2] + expand) and \
               (z[1] - expand) <= cy <= (z[3] + expand):
                return True
        return False

    family_labels: Dict = {}
    family_counts: Dict = {}
    forbidden_indices: set = set()

    for idx, d in enumerate(drawings):
        r = d.get("rect")
        dashes = d.get("dashes") or []
        sw = d.get("width") or d.get("stroke_width") or 0
        layer = str(d.get("layer") or d.get("ocg") or "").lower()

        family = PRIM_UNKNOWN

        if r:
            bx0, by0, bx1, by1 = r[0], r[1], r[2], r[3]
            bw = abs(bx1 - bx0)
            bh = abs(by1 - by0)
            length = max(bw, bh)
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2

            if length < pw * 0.03 and bw > 0 and bh > 0:
                aspect = max(bw, bh) / max(min(bw, bh), 0.1)
                if aspect > 4 and length < pw * 0.02:
                    family = PRIM_HATCH

            if family == PRIM_UNKNOWN and excluded_regions:
                if bbox_in_excluded([bx0, by0, bw, bh], excluded_regions, threshold=0.5):
                    family = PRIM_TABLE

            if family == PRIM_UNKNOWN and legend_region:
                lr = legend_region
                if lr[0] <= cx <= lr[2] and lr[1] <= cy <= lr[3]:
                    family = PRIM_LEGEND

            if family == PRIM_UNKNOWN:
                is_very_long = length > pw * 0.6 or length > ph * 0.6
                if is_very_long and _bbox_near_zones(r, grid_text_zones):
                    family = PRIM_GRID
                elif is_very_long and not dashes:
                    family = PRIM_GRID

            if family == PRIM_UNKNOWN and _bbox_near_zones(r, section_zones):
                family = PRIM_SECTION

            if family == PRIM_UNKNOWN:
                if dashes:
                    family = PRIM_DIMENSION
                elif sw > 0 and sw < 0.6 and _bbox_near_zones(r, numeric_text_zones):
                    family = PRIM_DIMENSION
                elif sw == 0 and _bbox_near_zones(r, numeric_text_zones, expand=PROX * 0.7):
                    family = PRIM_DIMENSION

            if family == PRIM_UNKNOWN:
                if dashes and length > pw * 0.15:
                    family = PRIM_SITE

            if family == PRIM_UNKNOWN:
                if sw >= 1.0 and length > pw * 0.02:
                    family = PRIM_WALL
                elif sw >= 0.5 and length > pw * 0.04:
                    family = PRIM_WALL

        family_labels[idx] = family
        family_counts[family] = family_counts.get(family, 0) + 1
        if family in FORBIDDEN_FAMILIES:
            forbidden_indices.add(idx)

    return {
        "family_labels": family_labels,
        "forbidden_indices": forbidden_indices,
        "family_counts": family_counts,
    }


# ──────────────────────────────────────────────
# ENGINE P2-4 — INK / GEOMETRY OVERLAP ENGINE
# ──────────────────────────────────────────────

def compute_ink_overlap(
    bbox: List,
    vector_cache: Dict,
    preview_image=None,
    scale_factor: float = 1.0,
) -> Dict:
    """
    בודק האם ה-bbox מכיל תוכן ציור אמיתי.

    מחזיר:
      ink_overlap_score: 0.0-1.0
      vector_overlap_count: int
      pixel_density: float או None
    """
    bx0, by0, bw, bh = bbox
    bx1, by1 = bx0 + bw, by0 + bh

    drawings = vector_cache.get("drawings", [])
    vector_count = 0
    for d in drawings:
        r = d.get("rect")
        if not r:
            continue
        if r[0] < bx1 and r[2] > bx0 and r[1] < by1 and r[3] > by0:
            vector_count += 1

    vector_score = min(1.0, vector_count / 5.0)

    pixel_density = None
    pixel_score = None

    if preview_image is not None and _NP_AVAILABLE:
        try:
            h_img, w_img = preview_image.shape[:2]
            sf = scale_factor
            px0 = max(0, int(bx0 * sf))
            py0 = max(0, int(by0 * sf))
            px1 = min(w_img, int(bx1 * sf))
            py1 = min(h_img, int(by1 * sf))

            if px1 > px0 and py1 > py0:
                crop = preview_image[py0:py1, px0:px1]
                if len(crop.shape) == 3:
                    gray = crop.mean(axis=2)
                else:
                    gray = crop
                dark_pixels = (gray < 180).sum()
                total_pixels = max(1, (px1 - px0) * (py1 - py0))
                pixel_density = dark_pixels / total_pixels
                pixel_score = min(1.0, pixel_density * 10)
        except Exception:
            pass

    if pixel_score is not None:
        ink_score = pixel_score * 0.6 + vector_score * 0.4
    else:
        ink_score = vector_score

    return {
        "ink_overlap_score": round(ink_score, 3),
        "vector_overlap_count": vector_count,
        "pixel_density": round(pixel_density, 4) if pixel_density is not None else None,
    }


# ──────────────────────────────────────────────
# ENGINE P2-5 — ANCHOR SUPPORT ENGINE
# ──────────────────────────────────────────────

_EQUIPMENT_ANCHOR_TEXTS = {
    "ארון כיבוי", "פנל כבאים", "כיבוי אש",
    "פיר חשמל", "פיר תקשורת", "פיר שירות",
    "מעלית", "עלייה במעלית", "ירידה במעלית",
    "שער ברזל", "שער", "פחי אשפה", "חדר מזגנים",
    "משאבה", "גנרטור", "ט' כיבוי", "ת.כ.",
}

_ROOM_ANCHOR_MIN_DIST_RATIO = 0.12


def compute_anchor_support(
    bbox: List,
    text_semantics: Dict,
    page_rect: Dict,
    wall_segments: List = None,
) -> Dict:
    """
    בודק האם ה-bbox מקבל תמיכה מ-"עוגן" הנדסי/סמנטי קרוב.

    מחזיר:
      anchor_support_score: 0.0-1.0
      anchor_types: list[str]
      anchor_boost: bool
    """
    pw = page_rect.get("w", 2480)

    bx0, by0, bw, bh = bbox
    cx = bx0 + bw / 2
    cy = by0 + bh / 2

    PROX_ROOM = pw * _ROOM_ANCHOR_MIN_DIST_RATIO
    PROX_EQUIP = pw * 0.08
    PROX_WALL = max(bw, bh) * 2.0

    anchor_types: List[str] = []
    anchor_boost = False
    score = 0.0

    rooms = text_semantics.get("rooms", [])
    door_tags = text_semantics.get("door_tags", [])
    area_texts = text_semantics.get("area_texts", [])
    text_zones = text_semantics.get("text_zones", [])

    for tz in text_zones:
        t = tz.get("text", "")
        for eq in _EQUIPMENT_ANCHOR_TEXTS:
            if eq in t:
                b = tz["bbox"]
                tc = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
                dist = math.sqrt((cx - tc[0])**2 + (cy - tc[1])**2)
                if dist < PROX_EQUIP:
                    anchor_types.append("equipment_text")
                    anchor_boost = True
                    score += 0.6
                    break

    for room in rooms:
        rb = room.get("bbox", [])
        if len(rb) >= 4:
            rc = [(rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2]
            dist = math.sqrt((cx - rc[0])**2 + (cy - rc[1])**2)
            if dist < PROX_ROOM:
                anchor_types.append("room_label")
                score += 0.3
                break

    for at in area_texts:
        ab = at.get("bbox", [])
        if len(ab) >= 4:
            ac = [(ab[0] + ab[2]) / 2, (ab[1] + ab[3]) / 2]
            dist = math.sqrt((cx - ac[0])**2 + (cy - ac[1])**2)
            if dist < PROX_ROOM * 0.6:
                anchor_types.append("area_text")
                score += 0.2
                break

    for dt in door_tags:
        db = dt.get("bbox", [])
        if len(db) >= 4:
            dc = [(db[0] + db[2]) / 2, (db[1] + db[3]) / 2]
            dist = math.sqrt((cx - dc[0])**2 + (cy - dc[1])**2)
            if dist < PROX_EQUIP * 1.5:
                anchor_types.append("door_tag")
                score += 0.25
                break

    if wall_segments:
        for seg in wall_segments:
            sb = seg.get("bbox") or seg.get("bounding_box") or []
            if len(sb) >= 4:
                sx0, sy0, sw2, sh2 = sb[0], sb[1], sb[2], sb[3]
                sx1, sy1 = sx0 + sw2, sy0 + sh2
                expanded = [sx0 - PROX_WALL, sy0 - PROX_WALL,
                            sx1 + PROX_WALL, sy1 + PROX_WALL]
                if expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]:
                    anchor_types.append("wall_nearby")
                    score += 0.2
                    break

    return {
        "anchor_support_score": round(min(1.0, score), 3),
        "anchor_types": list(set(anchor_types)),
        "anchor_boost": anchor_boost,
    }


# ──────────────────────────────────────────────
# ENGINE P2-6 — CANDIDATE LEGALITY SCORING ENGINE
# ──────────────────────────────────────────────

LEGALITY_PASS_THRESHOLD = 0.55
LEGALITY_SOFT_PASS_THRESHOLD = 0.35
LEGALITY_REVIEW_THRESHOLD = 0.20


def compute_legality_score(
    main_region_in: bool,
    exclusion_result: Dict,
    primitive_family: str,
    ink_result: Dict,
    anchor_result: Dict,
) -> Dict:
    """
    מאחד את כל ציוני החוקיות לציון אחד ומחזיר gate_decision.

    מחזיר:
      legality_score: 0.0-1.0
      legality_breakdown: dict
      gate_decision: "pass" | "soft_pass" | "review" | "reject"
      rejection_reason: str או None
    """
    breakdown: Dict = {}

    if not main_region_in:
        return {
            "legality_score": 0.0,
            "legality_breakdown": {"main_region": False},
            "gate_decision": "reject",
            "rejection_reason": "outside_main_drawing_region",
        }

    if exclusion_result.get("exclusion_decision") == "reject":
        return {
            "legality_score": 0.0,
            "legality_breakdown": {"exclusion": exclusion_result},
            "gate_decision": "reject",
            "rejection_reason": exclusion_result.get("exclusion_reason", "excluded_region_overlap"),
        }

    if primitive_family in FORBIDDEN_FAMILIES:
        return {
            "legality_score": 0.0,
            "legality_breakdown": {"family": primitive_family},
            "gate_decision": "reject",
            "rejection_reason": f"forbidden_primitive_family:{primitive_family}",
        }

    score = 0.30
    breakdown["main_region"] = True

    if exclusion_result.get("exclusion_decision") == "penalty":
        score -= 0.15
        breakdown["exclusion_penalty"] = exclusion_result.get("exclusion_overlap", 0)
    else:
        score += 0.10
        breakdown["exclusion_clear"] = True

    if primitive_family == PRIM_WALL:
        score += 0.20
        breakdown["family_wall"] = True
    elif primitive_family == PRIM_OPENING:
        score += 0.15
        breakdown["family_opening"] = True
    elif primitive_family == PRIM_UNKNOWN:
        score += 0.05
    breakdown["primitive_family"] = primitive_family

    ink_score = ink_result.get("ink_overlap_score", 0.5)
    score += ink_score * 0.20
    breakdown["ink_score"] = round(ink_score, 3)

    if ink_score < 0.15:
        score -= 0.25
        breakdown["low_ink_penalty"] = True

    anchor_score = anchor_result.get("anchor_support_score", 0.0)
    anchor_boost = anchor_result.get("anchor_boost", False)
    score += anchor_score * 0.20
    breakdown["anchor_score"] = round(anchor_score, 3)

    if anchor_boost:
        score += 0.15
        breakdown["anchor_boost"] = True

    score = round(max(0.0, min(1.0, score)), 3)
    breakdown["final_score"] = score

    if score >= LEGALITY_PASS_THRESHOLD:
        decision = "pass"
    elif score >= LEGALITY_SOFT_PASS_THRESHOLD:
        decision = "soft_pass"
    elif score >= LEGALITY_REVIEW_THRESHOLD:
        decision = "review"
    else:
        decision = "reject"

    return {
        "legality_score": score,
        "legality_breakdown": breakdown,
        "gate_decision": decision,
        "rejection_reason": None if decision != "reject" else "low_legality_score",
    }


# ──────────────────────────────────────────────
# ENGINE P2-7 — POST-GATE REJECTION ENGINE
# ──────────────────────────────────────────────

def post_gate_filter(
    segments: List[Dict],
    legality_map: Dict,
    min_gate: str = "review",
) -> Dict:
    """
    מסנן רשימת segments לפי legality map.

    min_gate קובע מה הרמה המינימלית שעוברת:
    - "pass"       → רק pass עובר
    - "soft_pass"  → pass + soft_pass עוברים
    - "review"     → pass + soft_pass + review עוברים
    - "reject"     → הכל עובר

    מחזיר:
      filtered_segments, rejected_segments, rejection_log, pass_count, reject_count
    """
    ORDER = ["pass", "soft_pass", "review", "reject"]
    min_idx = ORDER.index(min_gate)

    filtered: List[Dict] = []
    rejected: List[Dict] = []
    log: Dict = {}

    for seg in segments:
        seg_id = seg.get("segment_id") or seg.get("id") or ""
        legality = legality_map.get(seg_id)

        if not legality:
            filtered.append(seg)
            continue

        decision = legality.get("gate_decision", "review")
        decision_idx = ORDER.index(decision) if decision in ORDER else 2

        if decision_idx <= min_idx:
            filtered.append({
                **seg,
                "legality_score": legality.get("legality_score"),
                "gate_decision": decision,
            })
        else:
            reason = legality.get("rejection_reason") or decision
            rejected.append({**seg, "rejection_reason": reason})
            log[seg_id] = reason

    return {
        "filtered_segments": filtered,
        "rejected_segments": rejected,
        "rejection_log": log,
        "pass_count": len(filtered),
        "reject_count": len(rejected),
    }
