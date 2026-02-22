"""
drawing_logic.py
================
שינויים בגרסה זו (מינימום נדרש):
- תיקון sample_line_hits() כך ש-path (freedraw) נדגם לאורך כל הפוליליין (לא רק start/end).
- תיקון detect_opening_gaps_on_wall_line():
  - midpoint מבוסס נקודת דגימה אמיתית (ולא אינטרפולציה start->end)
  - דחיית זיהוי אם יש יציאה משמעותית מהמסכה (oob_ratio) כדי לא לייצר פתחים מזויפים.
- שאר הפונקציות נשארו זהות.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Dict, Optional, List
import uuid


# ==========================================
# Wall Confidence Maps (ללא שינוי)
# ==========================================


def build_wall_confidence_masks(proj: dict) -> dict:
    """
    יוצר מסכות ביטחון לזיהוי קירות

    Returns:
        {
            'strict': מסכת קירות קפדנית (גרסה מוקשחת של thick_walls),
            'relaxed': מסכת קירות מורחבת (לתפיסת קירות דקים/רעש),
            'uncertain': אזור ביניים,
            'bbox_relaxed': bbox מחושב לפי relaxed
        }
    """
    base = proj.get("thick_walls", None)

    if base is None:
        base = np.zeros((100, 100), dtype=np.uint8)

    # ודא binary 0/255
    base = (base > 0).astype(np.uint8) * 255

    # strict = erosion (מקשה) כדי שקטעים דקים/פתחים יפלו ל-0
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    strict = cv2.erode(base, kernel_erode, iterations=1)

    # relaxed = dilation (מרחיב) כדי לתפוס גם קירות דקים/רעש קל
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    relaxed = cv2.dilate(base, kernel_dilate, iterations=2)

    # אזור uncertain
    uncertain = cv2.subtract(relaxed, strict)

    # חישוב bbox לפי relaxed
    coords = cv2.findNonZero(relaxed)
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        bbox_relaxed = (x, y, w, h)
    else:
        # fallback אם אין קירות
        h, w = base.shape
        bbox_relaxed = (0, 0, w, h)

    return {
        "strict": strict,
        "relaxed": relaxed,
        "uncertain": uncertain,
        "bbox_relaxed": bbox_relaxed,
    }


def object_to_mask(
    obj: dict, canvas_w: int, canvas_h: int, fill_for_area: bool = False
) -> np.ndarray:
    """
    ממיר אובייקט JSON (מ-st_canvas) למסכת numpy
    """
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    obj_type = obj.get("type", "")

    if obj_type == "rect":
        x = int(obj.get("left", 0))
        y = int(obj.get("top", 0))
        w = int(obj.get("width", 0))
        h = int(obj.get("height", 0))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    elif obj_type == "line":
        x1 = int(obj.get("x1", 0))
        y1 = int(obj.get("y1", 0))
        x2 = int(obj.get("x2", 0))
        y2 = int(obj.get("y2", 0))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)

    elif obj_type == "path":
        pts = []
        for seg in obj.get("path", []) or []:
            if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                try:
                    pts.append((int(seg[1]), int(seg[2])))
                except Exception:
                    continue
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                cv2.line(mask, pts[i - 1], pts[i], 255, thickness=3)

    return mask


def calc_hit_rate(drawn_mask: np.ndarray, wall_mask: np.ndarray) -> float:
    inter = cv2.bitwise_and(drawn_mask, wall_mask)
    a = int(np.count_nonzero(drawn_mask))
    if a == 0:
        return 0.0
    b = int(np.count_nonzero(inter))
    return (b / a) * 100.0


def classify_against_walls(
    canvas_obj: dict,
    strict: np.ndarray,
    relaxed: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    scale_factor: float,
) -> dict:
    drawn_mask = object_to_mask(canvas_obj, canvas_w, canvas_h)

    strict_hit = calc_hit_rate(drawn_mask, strict)
    relaxed_hit = calc_hit_rate(drawn_mask, relaxed)

    return {
        "strict_hit": strict_hit,
        "relaxed_hit": relaxed_hit,
        "is_wall_like": strict_hit > 40 or relaxed_hit > 60,
        "scale_factor": scale_factor,
    }


def generate_opening_uid() -> str:
    return f"op_{uuid.uuid4().hex[:10]}"


def _extract_polyline_points_canvas(obj: dict) -> List[Tuple[float, float]]:
    obj_type = obj.get("type", "")
    if obj_type == "line":
        return [
            (float(obj.get("x1", 0.0)), float(obj.get("y1", 0.0))),
            (float(obj.get("x2", 0.0)), float(obj.get("y2", 0.0))),
        ]

    if obj_type == "path":
        pts: List[Tuple[float, float]] = []
        for seg in obj.get("path", []) or []:
            if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                try:
                    x = float(seg[1])
                    y = float(seg[2])
                    pts.append((x, y))
                except Exception:
                    continue
        cleaned: List[Tuple[float, float]] = []
        for p in pts:
            if not cleaned or (
                abs(p[0] - cleaned[-1][0]) + abs(p[1] - cleaned[-1][1]) > 0.01
            ):
                cleaned.append(p)
        return cleaned

    return []


def _resample_polyline(
    points: List[Tuple[float, float]], step_px: float
) -> List[Tuple[int, int]]:
    if not points or len(points) < 2:
        return []

    cum = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        total += seg
        cum.append(total)

    if total < 1.0:
        x, y = points[0]
        return [(int(round(x)), int(round(y)))]

    step_px = max(1.0, float(step_px))
    n = int(total / step_px)

    targets = [0.0] + [min(total, k * step_px) for k in range(1, n + 1)]
    if targets[-1] < total:
        targets.append(total)

    res: List[Tuple[int, int]] = []
    j = 0
    for t in targets:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = cum[j], cum[j + 1]
        if t1 <= t0 + 1e-9:
            x, y = points[j]
        else:
            alpha = (t - t0) / (t1 - t0)
            x0, y0 = points[j]
            x1, y1 = points[j + 1]
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
        res.append((int(round(x)), int(round(y))))

    out: List[Tuple[int, int]] = []
    for p in res:
        if not out or p != out[-1]:
            out.append(p)
    return out


def _sample_hits_and_points(
    obj_line: dict, walls_mask: np.ndarray, scale_factor: float, step_px: int = 3
) -> Tuple[List[float], List[Tuple[int, int]], int]:
    poly = _extract_polyline_points_canvas(obj_line)
    if len(poly) < 2:
        return [], [], 0

    sampled = _resample_polyline(poly, step_px)
    if len(sampled) < 2:
        return [], [], 0

    hits: List[float] = []
    oob = 0
    mask_h, mask_w = walls_mask.shape[:2]

    for x_canvas, y_canvas in sampled:
        x_orig = int(round(x_canvas / scale_factor))
        y_orig = int(round(y_canvas / scale_factor))
        if 0 <= x_orig < mask_w and 0 <= y_orig < mask_h:
            hits.append(1.0 if walls_mask[y_orig, x_orig] > 0 else 0.0)
        else:
            oob += 1
            hits.append(0.0)

    return hits, sampled, oob


def sample_line_hits(
    obj_line: dict, walls_mask: np.ndarray, scale_factor: float, step_px: int = 3
) -> List[float]:
    hits, _, _ = _sample_hits_and_points(obj_line, walls_mask, scale_factor, step_px)
    return hits


def detect_opening_gaps_on_wall_line(
    obj_line: dict,
    strict_mask: np.ndarray,
    relaxed_mask: np.ndarray,
    scale_factor: float,
    scale_px_per_m: float,
    params: Optional[dict] = None,
) -> List[dict]:
    if params is None:
        params = {}

    min_gap_length_m = params.get("min_gap_length_m", 0.3)
    max_gap_length_m = params.get("max_gap_length_m", 2.0)
    min_strict_hit_overall = params.get("min_strict_hit_overall", 65.0)
    step_px = params.get("step_px", 3)
    min_gap_samples = params.get("min_gap_samples", 3)
    max_oob_ratio = params.get("max_oob_ratio", 0.15)

    strict_hits, sampled_pts, strict_oob = _sample_hits_and_points(
        obj_line, strict_mask, scale_factor, step_px
    )
    relaxed_hits, _, _ = _sample_hits_and_points(
        obj_line, relaxed_mask, scale_factor, step_px
    )

    if len(strict_hits) < 5 or len(strict_hits) != len(relaxed_hits):
        return []

    oob_ratio = (strict_oob / len(strict_hits)) if strict_hits else 1.0
    if oob_ratio > max_oob_ratio:
        return []

    overall_strict_hit = (sum(strict_hits) / len(strict_hits)) * 100.0
    if overall_strict_hit < min_strict_hit_overall:
        return []

    gaps: List[dict] = []
    in_gap = False
    gap_start_idx: Optional[int] = None

    def _emit_gap(gap_start: int, gap_end: int):
        gap_len_samples = gap_end - gap_start + 1
        if gap_len_samples < min_gap_samples:
            return

        gap_length_canvas = gap_len_samples * float(step_px)
        gap_length_m = gap_length_canvas / (float(scale_factor) * float(scale_px_per_m))
        if not (min_gap_length_m <= gap_length_m <= max_gap_length_m):
            return

        mid_idx = int(round((gap_start + gap_end) / 2))
        mid_idx = max(0, min(mid_idx, len(sampled_pts) - 1))
        mid_x, mid_y = sampled_pts[mid_idx]

        gaps.append(
            {
                "gap_id": generate_opening_uid(),
                "start_t": gap_start / (len(strict_hits) - 1),
                "end_t": gap_end / (len(strict_hits) - 1),
                "length_m": float(gap_length_m),
                "midpoint_canvas": [int(mid_x), int(mid_y)],
                "debug": {
                    "gap_samples": int(gap_len_samples),
                    "gap_start_idx": int(gap_start),
                    "gap_end_idx": int(gap_end),
                    "overall_strict_hit": float(overall_strict_hit),
                    "oob_ratio": float(oob_ratio),
                },
            }
        )

    for idx, strict_hit in enumerate(strict_hits):
        relaxed_hit = relaxed_hits[idx]
        is_gap_point = (strict_hit == 0.0) and (relaxed_hit > 0.0)

        if is_gap_point and not in_gap:
            in_gap = True
            gap_start_idx = idx
        elif (not is_gap_point) and in_gap:
            _emit_gap(gap_start_idx, idx - 1)  # type: ignore[arg-type]
            in_gap = False
            gap_start_idx = None

    if in_gap and gap_start_idx is not None:
        _emit_gap(gap_start_idx, len(strict_hits) - 1)

    return gaps
