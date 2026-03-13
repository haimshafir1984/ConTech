import math


def compute_line_length_px(line_obj: dict) -> float:
    x1 = float(line_obj.get("x1", 0))
    y1 = float(line_obj.get("y1", 0))
    x2 = float(line_obj.get("x2", 0))
    y2 = float(line_obj.get("y2", 0))
    return math.hypot(x2 - x1, y2 - y1)


def compute_rect_area_px(rect_obj: dict) -> float:
    w = float(rect_obj.get("width", 0))
    h = float(rect_obj.get("height", 0))
    return abs(w * h)


def px_to_m(px: float, *args) -> float:
    """
    תואם לאחור:
    - px_to_m(px, scale_px_per_m)
    - px_to_m(px, unit_scale, scale_px_per_m)
    """
    if len(args) == 1:
        scale_px_per_m = args[0]
    elif len(args) >= 2:
        scale_px_per_m = args[1]
    else:
        scale_px_per_m = 0

    if not scale_px_per_m:
        return 0.0
    return float(px) / float(scale_px_per_m)


def px2_to_m2(px2: float, *args) -> float:
    """
    תואם לאחור:
    - px2_to_m2(px2, scale_px_per_m)
    - px2_to_m2(px2, unit_scale, scale_px_per_m)
    """
    if len(args) == 1:
        scale_px_per_m = args[0]
    elif len(args) >= 2:
        scale_px_per_m = args[1]
    else:
        scale_px_per_m = 0

    if not scale_px_per_m:
        return 0.0
    s = float(scale_px_per_m)
    return float(px2) / (s * s)


def get_scale_with_fallback(proj: dict, default_scale: float = 200.0) -> float:
    s = proj.get("scale") or 0
    return float(s) if s and s > 0 else float(default_scale)
