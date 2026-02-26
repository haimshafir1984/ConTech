import os
import sys
import tempfile
import uuid
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .analyzer import FloorPlanAnalyzer, parse_scale
from .floor_extractor import analyze_floor_and_rooms
from .database import (
    get_all_plans,
    get_all_work_types_for_plan,
    get_payment_invoice_data,
    get_plan_by_id,
    get_plan_by_filename,
    get_progress_reports,
    get_progress_summary_by_date_range,
    get_project_financial_status,
    get_project_forecast,
    init_database,
    save_plan,
    save_progress_report,
    run_query,
    update_plan_metadata,
    save_plan_images,
    load_plan_images,
    reset_all_data,
)
from .models import (
    AnalysisResult,
    CorrectionStroke,
    DashboardRecentReport,
    DashboardBoqProgressRow,
    DashboardResponse,
    DashboardTimelinePoint,
    DrawingDataScaleRequest,
    DrawingDataScaleResult,
    DrawingDataSummary,
    FloorAnalysisResponse,
    FloorAnalysisRunRequest,
    FloorAnalysisTotals,
    FloorRoomRow,
    HealthResponse,
    InvoiceCalculateRequest,
    InvoiceCalculationResponse,
    InvoiceSummaryRow,
    InvoiceWorkItem,
    ManualCorrectionApplyRequest,
    ManualCorrectionsSummary,
    MeasurementInfo,
    PaperSizeInfo,
    WorkerMeasuredItem,
    WorkerMeasuredItemRequest,
    WorkerReport,
    WorkerReportCreateRequest,
    PlanningAddItemRequest,
    PlanningCalibrateRequest,
    PlanningCategoryUpsertRequest,
    PlanningItem,
    PlanningResolveOpeningRequest,
    PlanningResolveWallRequest,
    PlanningState,
    PlanningTextItemRequest,
    PlanningZoneRequest,
    AutoAnalyzeResponse,
    AutoAnalyzeSegment,
    AutoAnalyzeVisionData,
    ConfirmAutoSegmentRequest,
    WorkSection,
    WorkSectionCreateRequest,
    PlanDetail,
    PlanListResponse,
    PlanSummary,
    PlanReadinessResponse,
    WorkshopScaleUpdateRequest,
    WallMaterialsSummary,
)
from .utils import (
    calculate_area_m2,
    clean_metadata_for_json,
    create_colored_overlay,
    extract_segments_from_mask,
    load_stats_df,
    refine_flooring_mask_with_rooms,
    safe_process_metadata,
)
from drawing_logic import build_wall_confidence_masks, detect_opening_gaps_on_wall_line
from pages.measure_utils import (
    compute_line_length_px,
    compute_rect_area_px,
    get_scale_with_fallback,
)
from plan_store import persist_project_assets


# ── Load .streamlit/secrets.toml into env vars (local dev) ───────────────────
def _load_streamlit_secrets_into_env() -> None:
    """If API keys are stored in .streamlit/secrets.toml (Streamlit dev setup),
    copy them into os.environ so FastAPI/uvicorn code can use them too."""
    import pathlib, re as _re
    _secrets = pathlib.Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if not _secrets.exists():
        return
    try:
        text = _secrets.read_text(encoding="utf-8")
        for _m in _re.finditer(r'^([A-Z_][A-Z0-9_]*)\s*=\s*"([^"]*)"', text, _re.MULTILINE):
            key, val = _m.group(1), _m.group(2)
            if key not in os.environ and val:
                os.environ[key] = val
    except Exception:
        pass

_load_streamlit_secrets_into_env()


# ─────────────────────────────────────────────────────────────────
# JSON helpers — handle numpy types, Pydantic models, bytes, etc.
# ─────────────────────────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder שמטפל ב-numpy types ובכל object שאינו serializable."""
    def default(self, obj):  # noqa: ANN001
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "model_dump"):          # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):                # Pydantic v1
            return obj.dict()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _safe_json_dumps(obj, **kwargs) -> str:
    """json.dumps עם NumpyEncoder + סינון bytes/bytearray."""
    def _strip(o):
        if isinstance(o, (bytes, bytearray)):
            return None
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items() if not isinstance(v, (bytes, bytearray))}
        if isinstance(o, list):
            return [_strip(i) for i in o]
        return o
    return json.dumps(_strip(obj), cls=_NumpyEncoder, ensure_ascii=False, **kwargs)


import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="ConTech Analyzer API", version="1.0.0")

# Thread pool for CPU-bound / blocking-IO work (keeps event loop free for /health)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="contech-worker")

# In-memory store, מחקה את st.session_state.projects עבור צד מנהל
PROJECTS: Dict[str, Dict] = {}
WORKER_REPORTS: list[Dict] = []
MANUAL_CORRECTIONS: Dict[str, Dict[str, np.ndarray]] = {}
DEFAULT_UNIT_PRICES: Dict[str, float] = {
    "קירות": 800.0,
    "ריצוף/חיפוי": 250.0,
}

init_database()

# CORS — מאפשר לפרונטאנד לתקשר עם הבאקאנד
# ניתן להגדיר ALLOWED_ORIGINS כמשתנה סביבה ב-Render (מופרד בפסיקים)
_cors_env = os.environ.get("ALLOWED_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

# ברירת מחדל: localhost לפיתוח + כל דומיין של onrender.com לפרודקשן
if not _cors_origins:
    _cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.onrender\.com",  # כל subdomain של Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _compute_materials(
    thick_walls,
    metadata: dict,
) -> WallMaterialsSummary:
    """
    Approximate concrete/blocks split and flooring area,
    following the same logic used in the Streamlit manager tab.
    Returns a WallMaterialsSummary with all-None fields when thick_walls is unavailable.
    """
    if thick_walls is None or not isinstance(thick_walls, np.ndarray) or thick_walls.size == 0:
        print("[_compute_materials] thick_walls unavailable — returning empty summary")
        return WallMaterialsSummary(
            total_wall_length_m=None,
            concrete_length_m=None,
            blocks_length_m=None,
            flooring_area_m2=None,
        )
    if not isinstance(metadata, dict):
        metadata = {}
    h, w = thick_walls.shape[:2]
    kernel = np.ones((6, 6), np.uint8)

    # Concrete mask (dilated / eroded walls)
    concrete = cv2.dilate(
        cv2.erode(thick_walls, kernel, iterations=1),
        kernel,
        iterations=2,
    )
    # Blocks mask is "what's left" after concrete
    blocks = cv2.subtract(thick_walls, concrete)

    # Use existing utility to convert masks into geometric segments
    meters_per_pixel = metadata.get("meters_per_pixel")
    scale = None
    if meters_per_pixel and meters_per_pixel > 0:
        # pixels_per_meter = 1 / meters_per_pixel
        scale = 1.0 / meters_per_pixel

    total_len_m: Optional[float] = None
    conc_len_m: Optional[float] = None
    block_len_m: Optional[float] = None

    if scale:
        # Use contour-based segments extraction when possible
        try:
            segments_all = extract_segments_from_mask(thick_walls, scale)
            total_len_m = sum(seg.get("length_m", 0.0) for seg in segments_all)

            conc_segments = extract_segments_from_mask(concrete, scale)
            block_segments = extract_segments_from_mask(blocks, scale)

            conc_len_m = sum(seg.get("length_m", 0.0) for seg in conc_segments)
            block_len_m = sum(seg.get("length_m", 0.0) for seg in block_segments)
        except Exception:
            # Fall back to simple pixel counting if something goes wrong
            total_pixels = int(cv2.countNonZero(thick_walls))
            concrete_pixels = int(cv2.countNonZero(concrete))
            blocks_pixels = int(cv2.countNonZero(blocks))

            total_len_m = total_pixels / scale
            conc_len_m = concrete_pixels / scale
            block_len_m = blocks_pixels / scale

    # Flooring area – uses pixels + scale and reuses existing guardrails
    pixels_flooring = metadata.get("pixels_flooring_area_refined") or metadata.get(
        "pixels_flooring_area"
    )
    pixels_per_meter: Optional[float] = None
    if meters_per_pixel and meters_per_pixel > 0:
        pixels_per_meter = 1.0 / meters_per_pixel

    flooring_area_m2: Optional[float] = None
    if pixels_flooring:
        flooring_area_m2 = calculate_area_m2(
            area_px=pixels_flooring,
            meters_per_pixel=meters_per_pixel,
            meters_per_pixel_x=metadata.get("meters_per_pixel_x"),
            meters_per_pixel_y=metadata.get("meters_per_pixel_y"),
            pixels_per_meter=pixels_per_meter,
        )

    return WallMaterialsSummary(
        total_wall_length_m=total_len_m,
        concrete_length_m=conc_len_m,
        blocks_length_m=block_len_m,
        flooring_area_m2=flooring_area_m2,
    )


def _build_plan_detail(
    *,
    plan_id: str,
    filename: str,
    meta_clean: dict,
    materials: WallMaterialsSummary,
) -> PlanDetail:
    """
    יוצר אובייקטי PlanSummary + PlanDetail מתוך מטא-דאטה וחומרי קירות.
    """
    summary = PlanSummary(
        id=plan_id,
        filename=filename,
        plan_name=meta_clean.get("plan_name") or filename,
        scale_px_per_meter=meta_clean.get("scale_px_per_meter")
        or meta_clean.get("pixels_per_meter")
        or None,
        total_wall_length_m=materials.total_wall_length_m,
        concrete_length_m=materials.concrete_length_m,
        blocks_length_m=materials.blocks_length_m,
        flooring_area_m2=materials.flooring_area_m2,
    )

    return PlanDetail(summary=summary, meta=meta_clean)


def _get_project_or_404(plan_id: str) -> Dict:
    proj = PROJECTS.get(plan_id)
    if proj:
        return proj
    # Fallback: try to reload from database (e.g. after server restart)
    try:
        row = get_plan_by_filename(plan_id)
        if not row:
            # Try by plan_id matching plan_name or filename
            all_rows = get_all_plans() or []
            for r in all_rows:
                if str(r.get("id")) == str(plan_id) or r.get("filename") == plan_id or r.get("plan_name") == plan_id:
                    row = r
                    break
        if row:
            import json as _json
            meta = {}
            try:
                meta = _json.loads(row.get("metadata") or "{}")
            except Exception:
                pass

            # Try to load numpy arrays from persisted asset paths
            def _try_load_color(path):
                if not path or not os.path.exists(path): return None
                try:
                    data = np.fromfile(path, dtype=np.uint8)
                    return cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception: return None
            def _try_load_gray(path):
                if not path or not os.path.exists(path): return None
                try:
                    data = np.fromfile(path, dtype=np.uint8)
                    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                except Exception: return None

            original_img = _try_load_color(meta.get("_asset_original_path", ""))
            thick_walls   = _try_load_gray(meta.get("_asset_thick_walls_path", ""))
            skeleton       = _try_load_gray(meta.get("_asset_skeleton_path", ""))
            concrete_mask  = _try_load_gray(meta.get("_asset_concrete_mask_path", ""))
            blocks_mask    = _try_load_gray(meta.get("_asset_blocks_mask_path", ""))
            flooring_mask  = _try_load_gray(meta.get("_asset_flooring_mask_path", ""))

            # Rebuild minimal project skeleton from DB
            # DB row may use 'scale_value' (old schema) or 'confirmed_scale' (new schema)
            # meta may store 'scale_px_per_meter' or 'pixels_per_meter'
            scale_val = float(
                row.get("scale_value") or
                row.get("confirmed_scale") or
                meta.get("scale_px_per_meter") or
                meta.get("pixels_per_meter") or
                200.0
            )
            proj = {
                "skeleton": skeleton,
                "thick_walls": thick_walls,
                "original": original_img,
                "raw_pixels": int(row.get("pixels") or 0),
                "scale": scale_val,
                "metadata": meta,
                "concrete_mask": concrete_mask,
                "blocks_mask": blocks_mask,
                "flooring_mask": flooring_mask,
                "total_length": float(row.get("total_wall_length_m") or 0.0),
                "llm_data": meta.get("llm_data") or {},
                "llm_suggestions": meta.get("llm_suggestions") or {},
                "assets": {},
                "planning": meta.get("planning") or {
                    "categories": {},
                    "items": [],
                    "boq": {},
                    "totals": {"total_length_m": 0.0, "total_area_m2": 0.0},
                },
                "db_plan_id": row.get("id"),
            }
            PROJECTS[plan_id] = proj
            # ── טעינת arrays מ-DB BLOB אם הנתיבים הדיסקיים לא קיימים (Render restart) ──
            _ensure_arrays_loaded(proj)
            return proj
    except Exception:
        pass
    raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found. Please re-upload.")



def _decode_color(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_gray(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _ensure_arrays_loaded(proj: Dict) -> None:
    """
    אם proj נטען מה-DB (אחרי restart), מנסה לטעון מחדש את ה-numpy arrays.
    סדר עדיפויות:
      1. דיסק (נתיב שנשמר ב-metadata) — מהיר, עובד בסביבה מקומית
      2. DB BLOB (img_original / img_thick_walls) — שרידות בין restarts ב-Render
    מעדכן את proj in-place.
    """
    meta = proj.get("metadata", {})

    def _load_color_disk(key: str):
        path = meta.get(key, "")
        if not path or not os.path.exists(path):
            return None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _load_gray_disk(key: str):
        path = meta.get(key, "")
        if not path or not os.path.exists(path):
            return None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

    # ── שלב 1: נסה דיסק ──
    if proj.get("original") is None:
        img = _load_color_disk("_asset_original_path")
        if img is not None:
            proj["original"] = img
    if proj.get("thick_walls") is None:
        arr = _load_gray_disk("_asset_thick_walls_path")
        if arr is not None:
            proj["thick_walls"] = arr
    if proj.get("skeleton") is None:
        arr = _load_gray_disk("_asset_skeleton_path")
        if arr is not None:
            proj["skeleton"] = arr
    if proj.get("concrete_mask") is None:
        arr = _load_gray_disk("_asset_concrete_mask_path")
        if arr is not None:
            proj["concrete_mask"] = arr
    if proj.get("blocks_mask") is None:
        arr = _load_gray_disk("_asset_blocks_mask_path")
        if arr is not None:
            proj["blocks_mask"] = arr
    if proj.get("flooring_mask") is None:
        arr = _load_gray_disk("_asset_flooring_mask_path")
        if arr is not None:
            proj["flooring_mask"] = arr

    # ── שלב 2: טען כל masks חסרים מ-DB BLOB ──
    _needs_blob = (
        proj.get("original") is None or
        proj.get("thick_walls") is None or
        proj.get("flooring_mask") is None or
        proj.get("skeleton") is None or
        proj.get("concrete_mask") is None or
        proj.get("blocks_mask") is None
    )
    if _needs_blob:
        filename = meta.get("filename") or meta.get("plan_id", "")
        if filename:
            try:
                (orig_bytes, walls_bytes, flooring_bytes,
                 skeleton_bytes, concrete_bytes, blocks_bytes) = load_plan_images(filename)
                _loaded = []
                if proj.get("original") is None and orig_bytes:
                    img = _decode_color(orig_bytes)
                    if img is not None:
                        proj["original"] = img; _loaded.append("original")
                if proj.get("thick_walls") is None and walls_bytes:
                    arr = _decode_gray(walls_bytes)
                    if arr is not None:
                        proj["thick_walls"] = arr; _loaded.append("thick_walls")
                if proj.get("flooring_mask") is None and flooring_bytes:
                    arr = _decode_gray(flooring_bytes)
                    if arr is not None:
                        proj["flooring_mask"] = arr; _loaded.append("flooring_mask")
                if proj.get("skeleton") is None and skeleton_bytes:
                    arr = _decode_gray(skeleton_bytes)
                    if arr is not None:
                        proj["skeleton"] = arr; _loaded.append("skeleton")
                if proj.get("concrete_mask") is None and concrete_bytes:
                    arr = _decode_gray(concrete_bytes)
                    if arr is not None:
                        proj["concrete_mask"] = arr; _loaded.append("concrete_mask")
                if proj.get("blocks_mask") is None and blocks_bytes:
                    arr = _decode_gray(blocks_bytes)
                    if arr is not None:
                        proj["blocks_mask"] = arr; _loaded.append("blocks_mask")
                if _loaded:
                    print(f"[DB-BLOB] loaded from DB for {filename}: {_loaded}")
            except Exception as e:
                print(f"[DB-BLOB] load_plan_images failed for {filename}: {e}")

def _get_manual_corrections(plan_id: str) -> Dict[str, np.ndarray]:
    return MANUAL_CORRECTIONS.setdefault(plan_id, {})


def _build_mask_from_strokes(
    *,
    strokes: list[CorrectionStroke],
    display_width: int,
    display_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    if display_width <= 0 or display_height <= 0:
        return mask

    sx = float(target_width) / float(display_width)
    sy = float(target_height) / float(display_height)
    for stroke in strokes:
        points = stroke.points or []
        if len(points) < 2:
            continue
        stroke_width = max(1, int(round(float(stroke.width) * ((sx + sy) / 2.0))))
        for idx in range(1, len(points)):
            p1 = points[idx - 1]
            p2 = points[idx]
            if len(p1) < 2 or len(p2) < 2:
                continue
            x1 = int(round(float(p1[0]) * sx))
            y1 = int(round(float(p1[1]) * sy))
            x2 = int(round(float(p2[0]) * sx))
            y2 = int(round(float(p2[1]) * sy))
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=stroke_width)
    return mask


def _get_corrected_walls(plan_id: str, proj: Dict) -> Optional[np.ndarray]:
    walls = proj.get("thick_walls")
    if walls is None:
        return None
    corrected = walls.copy()
    corr = MANUAL_CORRECTIONS.get(plan_id, {})
    added = corr.get("added_walls")
    removed = corr.get("removed_walls")
    if isinstance(added, np.ndarray):
        corrected = cv2.bitwise_or(corrected, added)
    if isinstance(removed, np.ndarray):
        corrected = cv2.subtract(corrected, removed)
    return corrected


def _build_corrections_summary(plan_id: str, proj: Dict) -> ManualCorrectionsSummary:
    walls = proj.get("thick_walls")
    if walls is None:
        # Try reload from disk
        meta_tmp = proj.get("metadata", {})
        wpath = meta_tmp.get("_asset_thick_walls_path", "")
        if wpath and os.path.exists(wpath):
            try:
                data = np.fromfile(wpath, dtype=np.uint8)
                walls = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if walls is not None:
                    proj["thick_walls"] = walls
            except Exception:
                pass
    if walls is None:
        raise HTTPException(status_code=400, detail="Plan walls mask not available. Please re-upload the plan PDF.")
    corrected = _get_corrected_walls(plan_id, proj)
    if corrected is None:
        raise HTTPException(status_code=404, detail="Plan walls mask not found")
    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))
    if scale_px_per_meter <= 0:
        scale_px_per_meter = 200.0
    auto_pixels = int(np.count_nonzero(walls))
    corrected_pixels = int(np.count_nonzero(corrected))
    auto_length = auto_pixels / scale_px_per_meter
    corrected_length = corrected_pixels / scale_px_per_meter
    corr = MANUAL_CORRECTIONS.get(plan_id, {})
    has_corrections = bool(corr.get("added_walls") is not None or corr.get("removed_walls") is not None)
    return ManualCorrectionsSummary(
        plan_id=plan_id,
        has_corrections=has_corrections,
        auto_wall_length_m=round(float(auto_length), 4),
        corrected_wall_length_m=round(float(corrected_length), 4),
        delta_wall_length_m=round(float(corrected_length - auto_length), 4),
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_llm_metadata(meta_clean: dict) -> dict:
    """
    Keep same metadata extraction path as Streamlit:
    utils.safe_process_metadata -> brain.process_plan_metadata.
    """
    raw_text = meta_clean.get("raw_text")
    if not raw_text:
        return {}
    try:
        llm_data = safe_process_metadata(meta=meta_clean)
        if isinstance(llm_data, dict) and llm_data:
            return llm_data
    except Exception:
        pass

    # Fallback for non-Streamlit runtime.
    try:
        from brain import process_plan_metadata

        llm_data = process_plan_metadata(raw_text)
        return llm_data if isinstance(llm_data, dict) else {}
    except Exception:
        return {}


def _update_scale_fields_from_scale_text(meta: dict, image_width: int, image_height: int) -> None:
    """
    Keep scale parsing consistent with original analyzer logic.
    """
    scale_text = str(meta.get("scale") or "").strip()
    if not scale_text:
        meta["scale_denominator"] = None
        return

    denom = parse_scale(scale_text)
    meta["scale_denominator"] = denom
    if not denom:
        return

    mm_per_pixel_x = meta.get("mm_per_pixel_x")
    mm_per_pixel_y = meta.get("mm_per_pixel_y")
    mm_per_pixel = meta.get("mm_per_pixel")

    if (mm_per_pixel_x is None or mm_per_pixel_y is None or mm_per_pixel is None) and isinstance(
        meta.get("paper_mm"), dict
    ):
        paper = meta.get("paper_mm") or {}
        paper_w = _safe_float(paper.get("width"), 0.0)
        paper_h = _safe_float(paper.get("height"), 0.0)
        if paper_w > 0 and paper_h > 0 and image_width > 0 and image_height > 0:
            mm_per_pixel_x = paper_w / image_width
            mm_per_pixel_y = paper_h / image_height
            mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2.0
            meta["mm_per_pixel_x"] = mm_per_pixel_x
            meta["mm_per_pixel_y"] = mm_per_pixel_y
            meta["mm_per_pixel"] = mm_per_pixel

    if mm_per_pixel and mm_per_pixel_x and mm_per_pixel_y:
        meters_per_pixel = (float(mm_per_pixel) * denom) / 1000.0
        meters_per_pixel_x = (float(mm_per_pixel_x) * denom) / 1000.0
        meters_per_pixel_y = (float(mm_per_pixel_y) * denom) / 1000.0
        meta["meters_per_pixel"] = meters_per_pixel
        meta["meters_per_pixel_x"] = meters_per_pixel_x
        meta["meters_per_pixel_y"] = meters_per_pixel_y
        if meters_per_pixel > 0:
            px_per_meter = 1.0 / meters_per_pixel
            meta["pixels_per_meter"] = px_per_meter
            meta["scale_px_per_meter"] = px_per_meter


def _ensure_scale_fields(meta: dict, fallback_px_per_meter: float = 200.0) -> None:
    """
    Guarantee presence of scale fields even when paper/mm metadata is missing.
    """
    px_per_meter = (
        _safe_float(meta.get("pixels_per_meter"), 0.0)
        or _safe_float(meta.get("scale_px_per_meter"), 0.0)
        or _safe_float(fallback_px_per_meter, 0.0)
    )
    if px_per_meter <= 0:
        px_per_meter = 200.0
    meta["pixels_per_meter"] = float(px_per_meter)
    meta["scale_px_per_meter"] = float(px_per_meter)
    meters_per_pixel = 1.0 / float(px_per_meter)
    if _safe_float(meta.get("meters_per_pixel"), 0.0) <= 0:
        meta["meters_per_pixel"] = meters_per_pixel
    if _safe_float(meta.get("meters_per_pixel_x"), 0.0) <= 0:
        meta["meters_per_pixel_x"] = meta["meters_per_pixel"]
    if _safe_float(meta.get("meters_per_pixel_y"), 0.0) <= 0:
        meta["meters_per_pixel_y"] = meta["meters_per_pixel"]


def _get_db_plan_id(proj: Dict) -> Optional[int]:
    db_plan_id = proj.get("db_plan_id")
    if db_plan_id is None:
        return None
    try:
        return int(db_plan_id)
    except Exception:
        return None


def _resolve_db_plan_id(plan_id: str) -> Optional[int]:
    """
    Resolve db plan id from:
    1) in-memory PROJECTS mapping
    2) numeric path id
    3) filename in plans table
    """
    proj = PROJECTS.get(plan_id)
    if proj:
        existing = _get_db_plan_id(proj)
        if existing is not None:
            return existing
        persisted = _persist_plan_to_database(plan_id, proj)
        if persisted is not None:
            return persisted

    if str(plan_id).isdigit():
        db_row = get_plan_by_id(int(plan_id))
        if db_row:
            return int(db_row["id"])

    by_filename = get_plan_by_filename(plan_id)
    if by_filename:
        return int(by_filename["id"])
    return None


def _persist_plan_to_database(plan_id: str, proj: Dict) -> Optional[int]:
    meta = proj.get("metadata", {})
    filename = meta.get("filename") or plan_id
    plan_name = meta.get("plan_name") or filename
    scale_text = meta.get("scale") or ""
    scale_val = _safe_float(proj.get("scale"), 0.0)
    raw_pixels = int(_safe_float(proj.get("raw_pixels"), 0.0))
    # Use _safe_json_dumps to handle numpy types and Pydantic models inside planning/items
    metadata_json = _safe_json_dumps(meta)
    materials_json = _safe_json_dumps(
        {
            "total_length": _safe_float(proj.get("total_length"), 0.0),
            "concrete_length_m": _safe_float(meta.get("concrete_length_m"), 0.0),
            "blocks_length_m": _safe_float(meta.get("blocks_length_m"), 0.0),
            "flooring_area_m2": _safe_float(meta.get("flooring_area_m2"), 0.0),
        },
    )

    # ── Strategy 1: update directly by known db_plan_id (fastest, most reliable) ──
    existing_db_id = _get_db_plan_id(proj)
    if existing_db_id is not None:
        update_plan_metadata(existing_db_id, metadata_json)
        PROJECTS[plan_id] = proj
        return existing_db_id

    # ── Strategy 2: fall back to save_plan (upsert by filename) ──
    db_plan_id = save_plan(
        filename=filename,
        plan_name=plan_name,
        scale_text=scale_text,
        scale_val=scale_val,
        pixels=raw_pixels,
        metadata=metadata_json,
        materials=materials_json,
    )
    if db_plan_id:
        proj["db_plan_id"] = int(db_plan_id)
        PROJECTS[plan_id] = proj
        return int(db_plan_id)

    # ── Strategy 3: resolve from DB by filename and force update metadata ──
    row = get_plan_by_filename(filename)
    if row:
        rid = int(row["id"])
        update_plan_metadata(rid, metadata_json)
        proj["db_plan_id"] = rid
        PROJECTS[plan_id] = proj
        return rid

    return None


def _build_drawing_data_summary(plan_id: str, proj: Dict) -> DrawingDataSummary:
    img = proj.get("original")
    if img is None:
        _ensure_arrays_loaded(proj)
        img = proj.get("original")
    if img is None:
        raise HTTPException(status_code=400, detail="Plan image not available. Please re-upload the plan PDF.")

    meta = proj.get("metadata", {})
    image_width_px = int(meta.get("image_width_px") or img.shape[1])
    image_height_px = int(meta.get("image_height_px") or img.shape[0])
    scale = _safe_float(proj.get("scale"), 0.0)
    if scale <= 0:
        scale = _safe_float(meta.get("pixels_per_meter"), 200.0)

    # Streamlit-like calculations from render_plan_data_tab.
    walls = proj.get("thick_walls")
    if walls is None:
        # Try reload from disk
        meta_tmp = proj.get("metadata", {})
        wpath = meta_tmp.get("_asset_thick_walls_path", "")
        if wpath and os.path.exists(wpath):
            try:
                data = np.fromfile(wpath, dtype=np.uint8)
                walls = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if walls is not None:
                    proj["thick_walls"] = walls
            except Exception:
                pass
    if walls is None:
        raise HTTPException(status_code=400, detail="Plan walls mask not available. Please re-upload the plan PDF.")
    kernel = np.ones((6, 6), np.uint8)
    concrete_mask = cv2.dilate(cv2.erode(walls, kernel, iterations=1), kernel, iterations=2)
    blocks_mask = cv2.subtract(walls, concrete_mask)
    concrete = _safe_float(np.count_nonzero(concrete_mask), 0.0) / max(scale, 1e-9)
    blocks = _safe_float(np.count_nonzero(blocks_mask), 0.0) / max(scale, 1e-9)
    total_wall = concrete + blocks
    flooring_pixels = _safe_float(
        meta.get("pixels_flooring_area_refined") or meta.get("pixels_flooring_area"), 0.0
    )
    flooring = flooring_pixels / max(scale * scale, 1e-9)

    return DrawingDataSummary(
        plan_id=plan_id,
        plan_name=meta.get("plan_name") or plan_id,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        image_size={"width_px": image_width_px, "height_px": image_height_px},
        scale_px_per_meter=scale,
        scale_text=meta.get("scale"),
        scale={"px_per_meter": round(float(scale), 4), "scale_text": meta.get("scale")},
        total_wall_length_m=round(total_wall, 4),
        concrete_length_m=round(concrete, 4),
        blocks_length_m=round(blocks, 4),
        flooring_area_m2=round(flooring, 4),
        materials={
            "concrete_m": round(concrete, 4),
            "blocks_m": round(blocks, 4),
            "total_m": round(total_wall, 4),
        },
        metadata=meta,
    )


def _build_plan_readiness(plan_id: str, proj: Dict) -> PlanReadinessResponse:
    meta = proj.get("metadata") or {}
    img = proj.get("original")
    walls = proj.get("thick_walls")
    flooring = proj.get("flooring_mask")

    has_original = img is not None
    has_thick_walls = walls is not None
    has_flooring_mask = flooring is not None
    wall_pixels = int(np.count_nonzero(walls)) if has_thick_walls else 0
    flooring_pixels = int(np.count_nonzero(flooring)) if has_flooring_mask else 0
    has_scale_px_per_meter = _safe_float(meta.get("scale_px_per_meter"), 0.0) > 0
    has_meters_per_pixel = _safe_float(meta.get("meters_per_pixel"), 0.0) > 0
    llm_data = proj.get("llm_data") or proj.get("llm_suggestions") or {}
    llm_rooms = llm_data.get("rooms") if isinstance(llm_data, dict) else None
    has_llm_rooms = isinstance(llm_rooms, list) and len(llm_rooms) > 0

    issues: list[str] = []
    if not has_original:
        issues.append("חסרה תמונת מקור לתוכנית (original).")
    if not has_thick_walls:
        issues.append("חסרה מסכת קירות מעובים (thick_walls).")
    elif wall_pixels == 0:
        issues.append("מסכת קירות קיימת אבל ריקה (0 פיקסלים).")
    if not has_flooring_mask:
        issues.append("חסרה מסכת ריצוף (flooring_mask).")
    if not has_scale_px_per_meter:
        issues.append("חסר scale_px_per_meter תקין לחישובי מטרים.")
    if not has_meters_per_pixel and not has_scale_px_per_meter:
        issues.append("חסר scale/metros-per-pixel תקין לניתוח שטחים.")
    # llm_rooms is optional: area analysis should still run without it.

    return PlanReadinessResponse(
        plan_id=plan_id,
        plan_name=str(meta.get("plan_name") or plan_id),
        has_original=has_original,
        has_thick_walls=has_thick_walls,
        has_flooring_mask=has_flooring_mask,
        wall_pixels=wall_pixels,
        flooring_pixels=flooring_pixels,
        has_scale_px_per_meter=has_scale_px_per_meter,
        has_meters_per_pixel=has_meters_per_pixel,
        has_llm_rooms=has_llm_rooms,
        issues=issues,
    )


def _parse_iso_date_or_400(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}") from exc


def _get_plan_reports(plan_id: str) -> list[Dict]:
    reports = [r for r in WORKER_REPORTS if r.get("plan_id") == plan_id]
    reports.sort(key=lambda r: r.get("date", ""))
    return reports


def _get_report_work_type(report: Dict) -> str:
    return "ריצוף/חיפוי" if report.get("report_type") == "floor" else "קירות"


def _get_report_unit(report: Dict) -> str:
    return "מ\"ר" if report.get("report_type") == "floor" else "מ'"


def _get_report_quantity(report: Dict) -> float:
    if report.get("report_type") == "floor":
        return _safe_float(report.get("total_area_m2"), 0.0)
    return _safe_float(report.get("total_length_m"), 0.0)


def _is_floor_work_type(text: object) -> bool:
    t = str(text or "")
    return ("ריצוף" in t) or ("חיפוי" in t) or ("floor" in t.lower())


def _extract_planning_from_sources(proj: Optional[Dict], plan_row: Dict) -> Dict:
    if proj and isinstance(proj.get("planning"), dict):
        return proj.get("planning") or {}
    if proj and isinstance(proj.get("metadata"), dict):
        planning = (proj.get("metadata") or {}).get("planning")
        if isinstance(planning, dict):
            return planning
    metadata_raw = plan_row.get("metadata")
    if isinstance(metadata_raw, dict):
        planning = metadata_raw.get("planning")
        if isinstance(planning, dict):
            return planning
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        try:
            meta_obj = json.loads(metadata_raw)
            if isinstance(meta_obj, dict):
                planning = meta_obj.get("planning")
                if isinstance(planning, dict):
                    return planning
        except Exception:
            pass
    return {}


def _compute_planned_vs_built_from_planning(
    *,
    planning: Dict,
    reports: list[Dict],
) -> tuple[float, float, list[DashboardBoqProgressRow]]:
    boq = planning.get("boq", {}) if isinstance(planning, dict) else {}
    planned_walls = 0.0
    planned_floor = 0.0
    if isinstance(boq, dict):
        for _, raw in boq.items():
            if not isinstance(raw, dict):
                continue
            cat_type = str(raw.get("type", ""))
            length_m = _safe_float(raw.get("total_length_m"), 0.0)
            area_m2 = _safe_float(raw.get("total_area_m2"), 0.0)
            if _is_floor_work_type(cat_type):
                planned_floor += area_m2
            else:
                planned_walls += length_m

    built_walls = 0.0
    built_floor = 0.0
    for report in reports:
        qty = _safe_float(report.get("meters_built"), 0.0)
        if _is_floor_work_type(report.get("note", "")):
            built_floor += qty
        else:
            built_walls += qty

    walls_progress = (built_walls / planned_walls * 100.0) if planned_walls > 0 else 0.0
    floor_progress = (built_floor / planned_floor * 100.0) if planned_floor > 0 else 0.0
    rows = [
        DashboardBoqProgressRow(
            label="קירות",
            planned_qty=round(planned_walls, 4),
            built_qty=round(built_walls, 4),
            remaining_qty=round(max(0.0, planned_walls - built_walls), 4),
            unit="מ'",
            progress_percent=round(min(100.0, max(0.0, walls_progress)), 2),
        ),
        DashboardBoqProgressRow(
            label="ריצוף/חיפוי",
            planned_qty=round(planned_floor, 4),
            built_qty=round(built_floor, 4),
            remaining_qty=round(max(0.0, planned_floor - built_floor), 4),
            unit='מ"ר',
            progress_percent=round(min(100.0, max(0.0, floor_progress)), 2),
        ),
    ]
    return planned_walls, planned_floor, rows


def _clean_floor_rooms(result: Dict) -> list[FloorRoomRow]:
    rooms: list[FloorRoomRow] = []
    for room in result.get("rooms", []):
        rooms.append(
            FloorRoomRow(
                room_id=int(room.get("room_id", 0)),
                matched_name=room.get("matched_name"),
                area_px=_safe_float(room.get("area_px"), 0.0),
                area_m2=(
                    _safe_float(room.get("area_m2"), 0.0)
                    if room.get("area_m2") is not None
                    else None
                ),
                area_text_m2=(
                    _safe_float(room.get("area_text_m2"), 0.0)
                    if room.get("area_text_m2") is not None
                    else None
                ),
                diff_m2=(
                    _safe_float(room.get("diff_m2"), 0.0)
                    if room.get("diff_m2") is not None
                    else None
                ),
                perimeter_px=(
                    _safe_float(room.get("perimeter_px"), 0.0)
                    if room.get("perimeter_px") is not None
                    else None
                ),
                perimeter_m=(
                    _safe_float(room.get("perimeter_m"), 0.0)
                    if room.get("perimeter_m") is not None
                    else None
                ),
                baseboard_m=(
                    _safe_float(room.get("baseboard_m"), 0.0)
                    if room.get("baseboard_m") is not None
                    else None
                ),
                match_confidence=(
                    _safe_float(room.get("match_confidence"), 0.0)
                    if room.get("match_confidence") is not None
                    else None
                ),
                center=[int(v) for v in room.get("center", [])]
                if isinstance(room.get("center"), (list, tuple))
                else None,
                bbox=[int(v) for v in room.get("bbox", [])]
                if isinstance(room.get("bbox"), (list, tuple))
                else None,
            )
        )
    return rooms


def _compute_line_length_px(raw_object: Dict) -> float:
    return float(compute_line_length_px(raw_object))


def _compute_rect_area_px(raw_object: Dict) -> float:
    return float(compute_rect_area_px(raw_object))


def _compute_path_length_px(raw_object: Dict) -> float:
    points = raw_object.get("points", [])
    if not isinstance(points, list) or len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        if not isinstance(p1, (list, tuple)) or not isinstance(p2, (list, tuple)):
            continue
        if len(p1) < 2 or len(p2) < 2:
            continue
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        total += float(np.sqrt(dx * dx + dy * dy))
    return total


def _extract_canvas_points_for_opening(obj: Dict, object_type: str) -> list[tuple[float, float]]:
    if object_type == "line":
        return [
            (float(obj.get("x1", 0.0)), float(obj.get("y1", 0.0))),
            (float(obj.get("x2", 0.0)), float(obj.get("y2", 0.0))),
        ]
    if object_type == "path":
        points = obj.get("points", [])
        if isinstance(points, list):
            out: list[tuple[float, float]] = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    out.append((float(p[0]), float(p[1])))
            return out
    return []


def _resample_polyline_points(
    points: list[tuple[float, float]], step_px: float = 2.0
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return []
    step_px = max(1.0, float(step_px))
    sampled: list[tuple[float, float]] = [points[0]]
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        if seg_len < 1e-6:
            continue
        n = max(1, int(seg_len // step_px))
        for k in range(1, n + 1):
            t = min(1.0, (k * step_px) / seg_len)
            sampled.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _get_drawing_bbox_original(proj: Dict) -> tuple[int, int, int, int]:
    """
    Returns drawing footprint bbox in ORIGINAL image coordinates.
    Prefer thick_walls footprint; fallback to full image size from metadata or image array.
    """
    walls = proj.get("thick_walls")
    if isinstance(walls, np.ndarray) and walls.size > 0:
        nz = cv2.findNonZero((walls > 0).astype(np.uint8))
        if nz is not None and len(nz) > 0:
            x, y, w, h = cv2.boundingRect(nz)
            if w > 0 and h > 0:
                return int(x), int(y), int(w), int(h)
    img = proj.get("original")
    if isinstance(img, np.ndarray) and img.size > 0:
        h, w = img.shape[:2]
        return 0, 0, int(w), int(h)
    # Fallback: use stored metadata image dimensions (available even after server restart)
    meta = proj.get("metadata", {})
    mw = int(meta.get("image_width_px") or 0)
    mh = int(meta.get("image_height_px") or 0)
    if mw > 0 and mh > 0:
        return 0, 0, mw, mh
    return 0, 0, 1, 1


def _point_in_bbox(px: float, py: float, bbox: tuple[int, int, int, int], margin: int = 0) -> bool:
    x, y, w, h = bbox
    return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)


def _object_inside_drawing_bbox(
    *,
    object_type: str,
    obj_original: Dict,
    bbox: tuple[int, int, int, int],
) -> bool:
    """
    Generic guardrail: reject marks that are mostly outside drawing footprint.
    """
    if object_type == "line":
        x1 = float(obj_original.get("x1", 0.0))
        y1 = float(obj_original.get("y1", 0.0))
        x2 = float(obj_original.get("x2", 0.0))
        y2 = float(obj_original.get("y2", 0.0))
        pts = _resample_polyline_points([(x1, y1), (x2, y2)], step_px=8.0)
        if len(pts) < 2:
            return False
        inside = sum(1 for x, y in pts if _point_in_bbox(x, y, bbox, margin=2))
        return (inside / max(1, len(pts))) >= 0.25

    if object_type == "path":
        points = obj_original.get("points", [])
        pts: list[tuple[float, float]] = []
        if isinstance(points, list):
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        if len(pts) < 2:
            return False
        sampled = _resample_polyline_points(pts, step_px=8.0)
        inside = sum(1 for x, y in sampled if _point_in_bbox(x, y, bbox, margin=2))
        return (inside / max(1, len(sampled))) >= 0.25

    if object_type == "rect":
        rx = float(obj_original.get("x", 0.0))
        ry = float(obj_original.get("y", 0.0))
        rw = abs(float(obj_original.get("width", 0.0)))
        rh = abs(float(obj_original.get("height", 0.0)))
        if rw <= 0 or rh <= 0:
            return False
        bx, by, bw, bh = bbox
        ix1 = max(rx, bx)
        iy1 = max(ry, by)
        ix2 = min(rx + rw, bx + bw)
        iy2 = min(ry + rh, by + bh)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih
        rect_area = rw * rh
        return (inter_area / max(1e-9, rect_area)) >= 0.15

    return True


def _is_wall_category_type(category_type: object) -> bool:
    text = str(category_type or "").strip().lower()
    if not text:
        return False
    return ("קיר" in text) or ("wall" in text)


def _get_wall_overlap_thresholds(proj: Dict) -> tuple[float, float]:
    """
    Returns (reject_threshold, confirm_threshold).
    If project was confirmed to contain thin walls, use more permissive thresholds.
    """
    meta = proj.get("metadata", {}) if isinstance(proj.get("metadata"), dict) else {}
    thin_mode = bool(meta.get("thin_wall_mode"))
    if thin_mode:
        return 0.03, 0.16
    return 0.08, 0.28


def _fallback_detect_openings_on_walls(
    *,
    obj: Dict,
    object_type: str,
    walls_mask: np.ndarray,
    display_scale: float,
    scale_px_per_meter: float,
) -> list[Dict]:
    """
    Generic opening detection:
    sample along drawn wall line/path on thick_walls and find internal non-wall runs
    that are surrounded by wall pixels on both sides.
    """
    pts = _extract_canvas_points_for_opening(obj, object_type)
    sampled = _resample_polyline_points(pts, step_px=2.0)
    if len(sampled) < 12:
        return []

    h, w = walls_mask.shape[:2]
    ds = max(float(display_scale), 1e-9)
    hits: list[int] = []
    valid_idx: list[int] = []
    for i, (x, y) in enumerate(sampled):
        xo = int(round(x / ds))
        yo = int(round(y / ds))
        if 0 <= xo < w and 0 <= yo < h:
            hits.append(1 if walls_mask[yo, xo] > 0 else 0)
            valid_idx.append(i)
        else:
            hits.append(0)

    # If the line barely overlaps walls, skip.
    wall_ratio = float(sum(hits)) / float(max(1, len(hits)))
    if wall_ratio < 0.35:
        return []

    openings: list[Dict] = []
    i = 0
    n = len(hits)
    while i < n:
        if hits[i] == 0:
            s = i
            while i + 1 < n and hits[i + 1] == 0:
                i += 1
            e = i
            # interior gaps only (not near the ends)
            if s > int(0.08 * n) and e < int(0.92 * n):
                left = hits[max(0, s - 6):s]
                right = hits[e + 1:min(n, e + 7)]
                left_ok = len(left) >= 3 and (sum(left) / len(left)) >= 0.6
                right_ok = len(right) >= 3 and (sum(right) / len(right)) >= 0.6
                if left_ok and right_ok:
                    run_len_px_canvas = float(max(1, e - s + 1) * 2.0)
                    run_len_m = run_len_px_canvas / max(ds * float(scale_px_per_meter), 1e-9)
                    if 0.2 <= run_len_m <= 3.5:
                        mid = (s + e) // 2
                        mx, my = sampled[mid]
                        openings.append(
                            {
                                "gap_id": f"fb_{uuid.uuid4().hex[:8]}",
                                "length_m": round(float(run_len_m), 4),
                                "midpoint_canvas": [int(round(mx)), int(round(my))],
                                "debug": {
                                    "method": "fallback_line_sampling",
                                    "wall_ratio": round(wall_ratio, 4),
                                    "range": [s, e],
                                },
                            }
                        )
        i += 1
    return openings


def _estimate_wall_overlap_ratio(
    *,
    obj: Dict,
    object_type: str,
    walls_mask: np.ndarray,
    display_scale: float,
) -> float:
    pts = _extract_canvas_points_for_opening(obj, object_type)
    sampled = _resample_polyline_points(pts, step_px=2.0)
    if len(sampled) < 8:
        return 0.0
    h, w = walls_mask.shape[:2]
    ds = max(float(display_scale), 1e-9)
    hits = 0
    total = 0
    for x, y in sampled:
        xo = int(round(x / ds))
        yo = int(round(y / ds))
        if 0 <= xo < w and 0 <= yo < h:
            total += 1
            if walls_mask[yo, xo] > 0:
                hits += 1
    if total == 0:
        return 0.0
    return float(hits) / float(total)


def _estimate_nonwhite_ink_ratio(
    *,
    obj: Dict,
    object_type: str,
    original_img: np.ndarray,
    display_scale: float,
) -> float:
    """
    Estimates whether user-marked geometry passes through non-white drawing content.
    Useful for thin CAD lines that may be missed by wall masks.
    """
    pts = _extract_canvas_points_for_opening(obj, object_type)
    sampled = _resample_polyline_points(pts, step_px=2.0)
    if len(sampled) < 8:
        return 0.0

    if len(original_img.shape) == 2:
        gray = original_img
    else:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    ds = max(float(display_scale), 1e-9)
    nonwhite = 0
    total = 0
    # small neighborhood around each sampled point to catch thin lines
    for x, y in sampled:
        xo = int(round(x / ds))
        yo = int(round(y / ds))
        if not (0 <= xo < w and 0 <= yo < h):
            continue
        x1 = max(0, xo - 1)
        x2 = min(w, xo + 2)
        y1 = max(0, yo - 1)
        y2 = min(h, yo + 2)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        total += int(patch.size)
        # "ink" = not white background
        nonwhite += int(np.count_nonzero(patch < 245))

    if total == 0:
        return 0.0
    return float(nonwhite) / float(total)


def _detect_openings_for_planning_item(
    *,
    proj: Dict,
    object_type: str,
    raw_object_display: Dict,
    scale_px_per_meter: float,
    display_scale: float,
    line_length_m: float,
) -> Dict:
    if object_type not in ("line", "path"):
        return {"openings": [], "is_wall_like": None}
    walls = proj.get("thick_walls")
    if walls is None:
        return {"openings": [], "is_wall_like": None}

    try:
        conf = build_wall_confidence_masks(proj)
        strict = conf.get("strict")
        relaxed = conf.get("relaxed")
        if strict is None or relaxed is None:
            return {"openings": [], "is_wall_like": None}

        if object_type == "line":
            canvas_obj = {
                "type": "line",
                "x1": float(raw_object_display.get("x1", 0.0)),
                "y1": float(raw_object_display.get("y1", 0.0)),
                "x2": float(raw_object_display.get("x2", 0.0)),
                "y2": float(raw_object_display.get("y2", 0.0)),
            }
        else:
            points = raw_object_display.get("points", [])
            path = []
            if isinstance(points, list) and points:
                for idx, p in enumerate(points):
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    cmd = "M" if idx == 0 else "L"
                    path.append([cmd, float(p[0]), float(p[1])])
            canvas_obj = {"type": "path", "path": path}

        openings_primary = detect_opening_gaps_on_wall_line(
            canvas_obj,
            strict_mask=strict,
            relaxed_mask=relaxed,
            scale_factor=max(float(display_scale), 1e-9),
            scale_px_per_m=max(float(scale_px_per_meter), 1e-9),
            params={
                "min_strict_hit_overall": 35.0,
                "min_gap_length_m": 0.2,
                "max_gap_length_m": 3.2,
                "max_oob_ratio": 0.35,
                "step_px": 2,
            },
        )
        openings_fallback = _fallback_detect_openings_on_walls(
            obj=raw_object_display,
            object_type=object_type,
            walls_mask=walls,
            display_scale=float(display_scale),
            scale_px_per_meter=float(scale_px_per_meter),
        )
        openings: list[Dict] = []
        if isinstance(openings_primary, list):
            openings.extend(openings_primary)
        if isinstance(openings_fallback, list):
            openings.extend(openings_fallback)

        # de-duplicate by midpoint closeness
        dedup: list[Dict] = []
        for op in openings:
            mp = op.get("midpoint_canvas") if isinstance(op, dict) else None
            if not isinstance(mp, (list, tuple)) or len(mp) < 2:
                continue
            x, y = float(mp[0]), float(mp[1])
            keep = True
            for ex in dedup:
                emp = ex.get("midpoint_canvas", [0, 0])
                dx = x - float(emp[0])
                dy = y - float(emp[1])
                if float(np.hypot(dx, dy)) < 12.0:
                    keep = False
                    break
            if keep:
                dedup.append(op)

        overlap_ratio = _estimate_wall_overlap_ratio(
            obj=raw_object_display,
            object_type=object_type,
            walls_mask=walls,
            display_scale=float(display_scale),
        )
        opening_total = float(
            sum(float(g.get("length_m", 0.0) or 0.0) for g in dedup)
        ) if dedup else 0.0
        is_wall_like = overlap_ratio >= 0.35
        prompt_opening_question = bool(dedup) or (
            is_wall_like and line_length_m >= 1.0 and 0.45 <= overlap_ratio <= 0.9
        )
        estimated_opening_length_m = (
            round(float(opening_total / len(dedup)), 4) if dedup else 0.9
        )
        return {
            "openings": dedup,
            "is_wall_like": is_wall_like,
            "wall_overlap_ratio": round(float(overlap_ratio), 4),
            "opening_total_m": opening_total,
            "prompt_opening_question": prompt_opening_question,
            "estimated_opening_length_m": estimated_opening_length_m,
            "resolved_opening_type": None,
            "resolved_gap_id": None,
        }
    except Exception:
        return {
            "openings": [],
            "is_wall_like": None,
            "prompt_opening_question": False,
            "estimated_opening_length_m": 0.9,
        }


def _init_planning_if_missing(proj: Dict) -> None:
    if "planning" not in proj or not isinstance(proj["planning"], dict):
        proj["planning"] = {
            "categories": {},
            "items": [],
            "boq": {},
            "totals": {"total_length_m": 0.0, "total_area_m2": 0.0},
        }


def _recompute_boq(proj: Dict) -> None:
    _init_planning_if_missing(proj)
    planning = proj["planning"]
    categories = planning.get("categories", {})
    items = planning.get("items", [])

    boq: Dict[str, Dict] = {}
    total_length = 0.0
    total_area = 0.0

    for item in items:
        cat_key = item.get("category", "unknown")
        if cat_key not in boq:
            cat = categories.get(cat_key, {})
            boq[cat_key] = {
                "type": cat.get("type", "unknown"),
                "subtype": cat.get("subtype", ""),
                "total_length_m": 0.0,
                "total_area_m2": 0.0,
                "count": 0,
                "params": cat.get("params", {}),
            }
        length_m = float(item.get("length_m_effective", item.get("length_m", 0.0)) or 0.0)
        area_m2 = float(item.get("area_m2", 0.0) or 0.0)
        boq[cat_key]["total_length_m"] += length_m
        boq[cat_key]["total_area_m2"] += area_m2
        boq[cat_key]["count"] += 1
        total_length += length_m
        total_area += area_m2

    planning["boq"] = boq
    planning["totals"] = {
        "total_length_m": round(total_length, 3),
        "total_area_m2": round(total_area, 3),
    }


def _build_planning_state(plan_id: str, proj: Dict) -> PlanningState:
    _init_planning_if_missing(proj)
    planning = proj["planning"]
    meta = proj.get("metadata", {})
    img = proj.get("original")
    h, w = img.shape[:2] if img is not None else (0, 0)
    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))

    categories_raw = planning.get("categories", {})
    categories = {}
    for key, data in categories_raw.items():
        categories[key] = {
            "key": key,
            "type": data.get("type", ""),
            "subtype": data.get("subtype", ""),
            "params": data.get("params", {}),
        }

    items = planning.get("items", [])
    _recompute_boq(proj)
    planning = proj["planning"]

    raw_sections = planning.get("sections", [])
    sections = [WorkSection(**s) if isinstance(s, dict) else s for s in raw_sections]

    return PlanningState(
        plan_id=plan_id,
        plan_name=meta.get("plan_name") or plan_id,
        scale_px_per_meter=scale_px_per_meter,
        image_width=int(w),
        image_height=int(h),
        categories=categories,
        items=[PlanningItem(**item) for item in items],
        boq=planning.get("boq", {}),
        totals=planning.get("totals", {"total_length_m": 0.0, "total_area_m2": 0.0}),
        sections=sections,
    )


def _measure_worker_item(
    *,
    plan_id: str,
    object_type: str,
    raw_object: Dict,
    display_scale: float,
    report_type: str,
) -> WorkerMeasuredItem:
    proj = _get_project_or_404(plan_id)
    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))

    display_scale = display_scale if display_scale > 0 else 1.0
    length_px = 0.0
    area_px = 0.0

    if object_type == "line":
        length_px = _compute_line_length_px(raw_object)
        obj_original = {
            "x1": float(raw_object.get("x1", 0.0)) / display_scale,
            "y1": float(raw_object.get("y1", 0.0)) / display_scale,
            "x2": float(raw_object.get("x2", 0.0)) / display_scale,
            "y2": float(raw_object.get("y2", 0.0)) / display_scale,
        }
    elif object_type == "rect":
        area_px = _compute_rect_area_px(raw_object)
        obj_original = {
            "x": float(raw_object.get("x", 0.0)) / display_scale,
            "y": float(raw_object.get("y", 0.0)) / display_scale,
            "width": float(raw_object.get("width", 0.0)) / display_scale,
            "height": float(raw_object.get("height", 0.0)) / display_scale,
        }
    elif object_type == "path":
        length_px = _compute_path_length_px(raw_object)
        points = raw_object.get("points", [])
        obj_original = {
            "points": [
                [float(p[0]) / display_scale, float(p[1]) / display_scale]
                for p in points
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if isinstance(points, list)
            else []
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported object_type")

    # Worker endpoint: no wall-overlap validation — field workers measure completed
    # work regardless of whether the algorithm detected that exact wall segment.

    length_px_original = length_px / display_scale
    area_px_original = area_px / (display_scale * display_scale)

    if report_type == "walls":
        measurement = length_px_original / scale_px_per_meter if scale_px_per_meter > 0 else 0.0
        unit = "m"
    else:
        measurement = (
            area_px_original / (scale_px_per_meter * scale_px_per_meter)
            if scale_px_per_meter > 0
            else 0.0
        )
        unit = "m2"

    return WorkerMeasuredItem(
        uid=str(uuid.uuid4())[:8],
        type=object_type,
        measurement=round(float(measurement), 4),
        unit=unit,
        raw_object=raw_object,
    )


@app.post("/analyze/pdf", response_model=AnalysisResult)
async def analyze_pdf(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Endpoint בסיסי לשימוש כללי (המסך הפשוט ב-React).
    ממשיך להשתמש ישירות באנלייזר ומחזיר AnalysisResult.
    """
    # Persist uploaded file to a temporary PDF so the existing analyzer can read it
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        analyzer = FloorPlanAnalyzer()
        (
            pix,
            skel,
            final_walls,
            image_proc,
            meta,
            concrete,
            blocks_mask,
            flooring,
            debug_img,
        ) = analyzer.process_file(tmp_path, save_debug=False, crop_bbox=None)

        meta_clean = clean_metadata_for_json(meta)

        paper_info = meta_clean.get("paper_mm")
        paper_detected = meta_clean.get("paper_size_detected")

        paper = PaperSizeInfo(
            detected_size=paper_detected,
            width_mm=paper_info.get("width") if paper_info else None,
            height_mm=paper_info.get("height") if paper_info else None,
            error_mm=meta_clean.get("paper_detection_error_mm"),
            confidence=meta_clean.get("paper_detection_confidence"),
        )

        measurements = MeasurementInfo(
            meters_per_pixel=meta_clean.get("meters_per_pixel"),
            meters_per_pixel_x=meta_clean.get("meters_per_pixel_x"),
            meters_per_pixel_y=meta_clean.get("meters_per_pixel_y"),
            scale_denominator=meta_clean.get("scale_denominator"),
            measurement_confidence=meta_clean.get("measurement_confidence"),
        )

        materials = _compute_materials(final_walls, meta_clean)

        plan_name = meta_clean.get("plan_name") or os.path.basename(file.filename or "")

        return AnalysisResult(
            plan_name=plan_name,
            meta=meta_clean,
            paper=paper,
            measurements=measurements,
            materials=materials,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/manager/workshop/upload", response_model=PlanDetail)
async def manager_upload_plan(file: UploadFile = File(...)) -> PlanDetail:
    """
    סדנת עבודה - העלאת תוכנית:
    - מריץ את FloorPlanAnalyzer
    - שומר את התוצאה ב-PROJECTS (in-memory)
    - מחזיר PlanDetail לתצוגה בצד React
    """
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    filename = file.filename or os.path.basename(tmp_path)

    try:
        loop = asyncio.get_event_loop()

        # ── CPU-bound: PDF render + OpenCV analysis (runs in thread, frees event loop) ──
        analyzer = FloorPlanAnalyzer()
        (
            pix,
            skel,
            final_walls,
            image_proc,
            meta,
            concrete,
            blocks_mask,
            flooring,
            debug_img,
        ) = await loop.run_in_executor(
            _executor,
            lambda: analyzer.process_file(tmp_path, save_debug=False, crop_bbox=None)
        )

        meta_clean = clean_metadata_for_json(meta)
        meta_clean["filename"] = filename
        meta_clean["image_width_px"] = int(image_proc.shape[1])
        meta_clean["image_height_px"] = int(image_proc.shape[0])
        _update_scale_fields_from_scale_text(meta_clean, int(image_proc.shape[1]), int(image_proc.shape[0]))
        _ensure_scale_fields(meta_clean, fallback_px_per_meter=_safe_float(meta_clean.get("pixels_per_meter"), 200.0))

        materials = _compute_materials(final_walls, meta_clean)
        meta_clean["concrete_length_m"] = materials.concrete_length_m
        meta_clean["blocks_length_m"] = materials.blocks_length_m
        meta_clean["flooring_area_m2"] = materials.flooring_area_m2

        llm_data = _extract_llm_metadata(meta_clean)
        if llm_data:
            for key, value in llm_data.items():
                if key not in meta_clean:
                    meta_clean[key] = value

        # ── Vision analysis — blocking HTTP to Anthropic (runs in thread) ────────────
        try:
            from .vision_analyzer import analyze_plan_with_vision
            vision_data = await loop.run_in_executor(
                _executor, analyze_plan_with_vision, tmp_path
            )
            if vision_data:
                # חדרים מהויזן → llm_rooms (מקור עיקרי ל-floor_extractor)
                if vision_data.get("rooms"):
                    meta_clean["llm_rooms"] = vision_data["rooms"]
                # קנה מידה — רק אם עדיין לא זוהה
                if vision_data.get("scale") and not meta_clean.get("scale_text"):
                    meta_clean["scale_text"] = vision_data["scale"]
                    _update_scale_fields_from_scale_text(
                        meta_clean,
                        meta_clean.get("image_width_px", 0),
                        meta_clean.get("image_height_px", 0),
                    )
                # מידות, חומרים, הערות
                if vision_data.get("dimensions_found"):
                    meta_clean["vision_dimensions"] = vision_data["dimensions_found"]
                if vision_data.get("materials"):
                    meta_clean["vision_materials"] = vision_data["materials"]
                if vision_data.get("total_area_m2"):
                    meta_clean["vision_total_area_m2"] = vision_data["total_area_m2"]
                if vision_data.get("plan_title") and not meta_clean.get("plan_title"):
                    meta_clean["plan_title"] = vision_data["plan_title"]
                # ── שדות חדשים ──────────────────────────────────────────
                if vision_data.get("dimensions_structured"):
                    meta_clean["vision_dimensions_structured"] = vision_data["dimensions_structured"]
                if vision_data.get("elevations"):
                    meta_clean["vision_elevations"] = vision_data["elevations"]
                if vision_data.get("materials_legend"):
                    meta_clean["vision_materials_legend"] = vision_data["materials_legend"]
                if vision_data.get("elements"):
                    meta_clean["vision_elements"] = vision_data["elements"]
                if vision_data.get("grid_lines"):
                    meta_clean["vision_grid_lines"] = vision_data["grid_lines"]
                if vision_data.get("systems"):
                    meta_clean["vision_systems"] = vision_data["systems"]
                # מטה-דאטה מבלוק הכותרת
                for _vf in ("project_name", "sheet_number", "sheet_name", "status",
                            "revision", "date", "drawn_by", "designed_by", "approved_by",
                            "architect", "project_address", "floor_level",
                            "drawing_number", "default_ceiling_height_m",
                            "execution_notes"):
                    if vision_data.get(_vf) and not meta_clean.get(_vf):
                        meta_clean[_vf] = vision_data[_vf]
                if vision_data.get("_pages_processed"):
                    meta_clean["vision_pages_processed"] = vision_data["_pages_processed"]
        except Exception as _ve:
            print(f"[WARNING] Vision analysis skipped: {_ve}")
        # ── Auto-generate plan_name from title-block if still using raw filename ──────
        if not meta_clean.get("plan_name") or meta_clean.get("plan_name") == filename:
            _nm_parts: list[str] = []
            if meta_clean.get("project_name"):
                _nm_parts.append(str(meta_clean["project_name"]))
            _sheet_id = (
                meta_clean.get("plan_title")
                or meta_clean.get("sheet_name")
                or meta_clean.get("sheet_number")
            )
            if _sheet_id:
                _nm_parts.append(str(_sheet_id))
            if _nm_parts:
                meta_clean["plan_name"] = " — ".join(_nm_parts)
        # ─────────────────────────────────────────────────────────────────────────────

        plan_id = meta_clean.get("plan_id") or filename
        assets = {}
        _orig_jpg_bytes_for_blob = None
        _walls_png_bytes_for_blob = None
        try:
            assets = persist_project_assets(
                plan_key=str(plan_id),
                original_bgr=image_proc,
                thick_walls=final_walls,
                skeleton=skel,
                concrete_mask=concrete,
                blocks_mask=blocks_mask,
                flooring_mask=flooring,
                keep_debug=False,
            )
            # שמירת נתיבי assets ב-metadata לצורך reload אחרי restart
            meta_clean["_asset_original_path"] = assets.get("original_path", "")
            meta_clean["_asset_thick_walls_path"] = assets.get("thick_walls_path", "")
            meta_clean["_asset_skeleton_path"] = assets.get("skeleton_path", "")
            meta_clean["_asset_concrete_mask_path"] = assets.get("concrete_mask_path", "")
            meta_clean["_asset_blocks_mask_path"] = assets.get("blocks_mask_path", "")
            meta_clean["_asset_flooring_mask_path"] = assets.get("flooring_mask_path", "")

            # שמירת bytes לשימוש מאוחר יותר (אחרי INSERT ל-DB) — כל ה-masks
            try:
                def _read_asset(key: str) -> bytes:
                    p = assets.get(key, "")
                    if p and os.path.exists(p):
                        with open(p, "rb") as _f:
                            return _f.read()
                    return b""
                _orig_jpg_bytes_for_blob = _read_asset("original_path")
                _walls_png_bytes_for_blob = _read_asset("thick_walls_path")
                _flooring_bytes_for_blob = _read_asset("flooring_mask_path")
                _skeleton_bytes_for_blob = _read_asset("skeleton_path")
                _concrete_bytes_for_blob = _read_asset("concrete_mask_path")
                _blocks_bytes_for_blob = _read_asset("blocks_mask_path")
            except Exception as _read_err:
                print(f"[DB-BLOB] read asset files failed: {_read_err}")
                _flooring_bytes_for_blob = b""
                _skeleton_bytes_for_blob = b""
                _concrete_bytes_for_blob = b""
                _blocks_bytes_for_blob = b""
        except Exception:
            assets = {}

        # שמירה ב-PROJECTS, מחקה את st.session_state.projects
        PROJECTS[plan_id] = {
            "skeleton": skel,
            "thick_walls": final_walls,
            "original": image_proc,
            "raw_pixels": pix,
            "scale": _safe_float(meta_clean.get("pixels_per_meter"), 200.0),
            "metadata": meta_clean,
            "concrete_mask": concrete,
            "blocks_mask": blocks_mask,
            "flooring_mask": flooring,
            "total_length": materials.total_wall_length_m,
            "llm_data": llm_data,
            "llm_suggestions": llm_data,
            "assets": assets,
            "planning": {
                "categories": {},
                "items": [],
                "boq": {},
                "totals": {"total_length_m": 0.0, "total_area_m2": 0.0},
            },
        }
        _persist_plan_to_database(plan_id, PROJECTS[plan_id])

        # ── שמירת כל ה-masks כ-BLOB ב-DB (אחרי INSERT, שרידות בין restarts ו-workers) ──
        try:
            if _orig_jpg_bytes_for_blob or _walls_png_bytes_for_blob:
                save_plan_images(
                    filename,
                    _orig_jpg_bytes_for_blob or b"",
                    _walls_png_bytes_for_blob or b"",
                    flooring_mask_png=_flooring_bytes_for_blob or b"",
                    skeleton_png=_skeleton_bytes_for_blob or b"",
                    concrete_mask_png=_concrete_bytes_for_blob or b"",
                    blocks_mask_png=_blocks_bytes_for_blob or b"",
                )
        except Exception as _blob_err:
            print(f"[DB-BLOB] save_plan_images failed: {_blob_err}")

        detail = _build_plan_detail(
            plan_id=plan_id,
            filename=filename,
            meta_clean=meta_clean,
            materials=materials,
        )
        return detail

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/manager/workshop/plans", response_model=PlanListResponse)
async def manager_list_plans() -> PlanListResponse:
    """
    מחזיר את כל התוכניות — קודם PROJECTS (in-memory), ואם ריק אחרי restart
    משלים מה-DB כך שהרשימה תמיד מוצגת.
    """
    plans = []

    # ── 1. תוכניות שכבר בזיכרון ──
    seen_ids: set[str] = set()
    for plan_id, proj in PROJECTS.items():
        try:
            meta_clean = proj.get("metadata") or {}
            # _compute_materials handles thick_walls=None gracefully
            materials = _compute_materials(proj.get("thick_walls"), meta_clean)
            summary = _build_plan_detail(
                plan_id=plan_id,
                filename=meta_clean.get("filename") or plan_id,
                meta_clean=meta_clean,
                materials=materials,
            ).summary
            plans.append(summary)
        except Exception as _e:
            print(f"[list_plans] skipping plan {plan_id!r} due to error: {_e}")
        seen_ids.add(plan_id)
        seen_ids.add((proj.get("metadata") or {}).get("filename") or plan_id)

    # ── 2. השלם מה-DB (אחרי restart כש-PROJECTS ריק) ──
    try:
        for row in get_all_plans() or []:
            filename = str(row.get("filename") or row.get("id") or "")
            if not filename or filename in seen_ids:
                continue
            import json as _json
            meta: dict = {}
            try:
                meta = _json.loads(row.get("metadata") or "{}")
            except Exception:
                pass
            scale_value = _safe_float(row.get("scale_value"), 0.0)
            raw_pixels = _safe_float(row.get("raw_pixels"), 0.0)
            total_wall = raw_pixels / scale_value if scale_value > 0 else 0.0
            plan_name = str(row.get("plan_name") or meta.get("plan_name") or filename)
            plans.append(
                PlanSummary(
                    id=filename,
                    filename=filename,
                    plan_name=plan_name,
                    scale_px_per_meter=scale_value if scale_value > 0 else None,
                    total_wall_length_m=round(total_wall, 4),
                    concrete_length_m=None,
                    blocks_length_m=None,
                    flooring_area_m2=None,
                )
            )
            seen_ids.add(filename)
    except Exception as _db_err:
        print(f"[list_plans] DB fallback error: {_db_err}")

    return PlanListResponse(plans=plans)


@app.get("/manager/database/plans", response_model=PlanListResponse)
async def manager_list_database_plans() -> PlanListResponse:
    plans: list[PlanSummary] = []
    for row in get_all_plans() or []:
        scale_value = _safe_float(row.get("scale_value"), 0.0)
        raw_pixels = _safe_float(row.get("raw_pixels"), 0.0)
        total_wall = raw_pixels / scale_value if scale_value > 0 else 0.0
        plans.append(
            PlanSummary(
                id=str(row.get("filename") or row.get("id")),
                filename=str(row.get("filename") or row.get("id")),
                plan_name=str(row.get("plan_name") or row.get("filename") or row.get("id")),
                scale_px_per_meter=scale_value if scale_value > 0 else None,
                total_wall_length_m=round(total_wall, 4),
                concrete_length_m=None,
                blocks_length_m=None,
                flooring_area_m2=None,
            )
        )
    return PlanListResponse(plans=plans)


@app.delete("/manager/workshop/plans", status_code=200)
async def manager_clear_all_plans() -> dict:
    """
    מוחק את כל התוכניות מה-DB ומה-זיכרון.
    """
    PROJECTS.clear()
    MANUAL_CORRECTIONS.clear()
    try:
        reset_all_data()
    except Exception as e:
        print(f"[clear_plans] DB reset error: {e}")
        raise HTTPException(status_code=500, detail=f"שגיאה במחיקת נתונים: {e}")
    return {"ok": True, "message": "כל התוכניות נמחקו"}


@app.get("/manager/workshop/plans/{plan_id}", response_model=PlanDetail)
async def manager_get_plan(plan_id: str) -> PlanDetail:
    """
    מחזיר פרטי תוכנית בודדת עבור סדנת עבודה.
    """
    proj = _get_project_or_404(plan_id)

    meta_clean = proj.get("metadata") or {}
    # _compute_materials handles thick_walls=None gracefully
    materials = _compute_materials(proj.get("thick_walls"), meta_clean)
    return _build_plan_detail(
        plan_id=plan_id,
        filename=meta_clean.get("filename") or plan_id,
        meta_clean=meta_clean,
        materials=materials,
    )


@app.patch("/manager/workshop/plans/{plan_id}/scale", response_model=PlanDetail)
async def manager_update_plan_scale_text(
    plan_id: str, request: WorkshopScaleUpdateRequest
) -> PlanDetail:
    proj = _get_project_or_404(plan_id)
    _ensure_arrays_loaded(proj)  # טעינה מדיסק/DB אם חסר
    meta = proj.get("metadata", {})
    image = proj.get("original")
    if image is None:
        raise HTTPException(status_code=409, detail="PLAN_RESTART_LOST: נתוני התוכנית לא זמינים בשרת (ייתכן שהשרת עלה מחדש). אנא העלה את קובץ ה-PDF שוב.")

    scale_text = str(request.scale_text or "").strip()
    if not scale_text:
        raise HTTPException(status_code=400, detail="scale_text is required")

    meta["scale"] = scale_text
    if request.plan_name is not None and str(request.plan_name).strip():
        meta["plan_name"] = str(request.plan_name).strip()
    _update_scale_fields_from_scale_text(meta, int(image.shape[1]), int(image.shape[0]))
    _ensure_scale_fields(meta, fallback_px_per_meter=_safe_float(proj.get("scale"), 200.0))

    proj["metadata"] = meta
    proj["scale"] = _safe_float(meta.get("pixels_per_meter"), _safe_float(proj.get("scale"), 200.0))

    _persist_plan_to_database(plan_id, proj)
    materials = _compute_materials(proj.get("thick_walls"), meta)
    return _build_plan_detail(
        plan_id=plan_id,
        filename=meta.get("filename") or plan_id,
        meta_clean=meta,
        materials=materials,
    )


@app.get("/manager/workshop/plans/{plan_id}/image")
async def manager_get_plan_image(plan_id: str) -> Response:
    """
    מחזיר תמונת PNG של התוכנית לציור ב-Frontend.
    """
    proj = _get_project_or_404(plan_id)
    _ensure_arrays_loaded(proj)  # טעינה מדיסק/DB אם חסר
    image = proj.get("original")
    if image is None:
        raise HTTPException(status_code=409, detail="PLAN_RESTART_LOST: נתוני התוכנית לא זמינים בשרת (ייתכן שהשרת עלה מחדש). אנא העלה את קובץ ה-PDF שוב.")

    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image

    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/manager/workshop/plans/{plan_id}/overlay")
async def manager_get_plan_overlay(
    plan_id: str,
    show_flooring: bool = True,
    show_room_numbers: bool = True,
    highlight_walls: bool = False,
) -> Response:
    """
    Returns a real analysis overlay (walls/flooring) for visual feedback.
    """
    proj = _get_project_or_404(plan_id)
    _ensure_arrays_loaded(proj)  # טעינה מדיסק/DB אם חסר (כולל original)
    image = proj.get("original")
    walls = proj.get("thick_walls")
    flooring = proj.get("flooring_mask")
    if image is None:
        raise HTTPException(status_code=409, detail="PLAN_RESTART_LOST: נתוני התוכנית לא זמינים בשרת (ייתכן שהשרת עלה מחדש). אנא העלה את קובץ ה-PDF שוב.")
    if walls is None:
        raise HTTPException(status_code=400, detail="Plan walls mask not available. Please re-upload the plan PDF.")

    kernel = np.ones((6, 6), np.uint8)
    concrete = cv2.dilate(cv2.erode(walls, kernel, iterations=1), kernel, iterations=2)
    blocks = cv2.subtract(walls, concrete)

    flooring_mask = flooring if show_flooring else None
    overlay_rgb = create_colored_overlay(
        image,
        concrete,
        blocks,
        flooring_mask=flooring_mask,
        alpha=0.42,
    )
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    if highlight_walls:
        contours, _ = cv2.findContours(walls.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 1)

    if show_room_numbers:
        llm_data = proj.get("llm_data") or proj.get("llm_suggestions") or {}
        rooms = llm_data.get("rooms") if isinstance(llm_data, dict) else []
        if isinstance(rooms, list):
            for idx, room in enumerate(rooms[:40], start=1):
                if not isinstance(room, dict):
                    continue
                center = room.get("center")
                if (
                    isinstance(center, (list, tuple))
                    and len(center) >= 2
                    and isinstance(center[0], (int, float))
                    and isinstance(center[1], (int, float))
                ):
                    x, y = int(center[0]), int(center[1])
                    cv2.putText(
                        overlay_bgr,
                        str(idx),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (20, 20, 20),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay_bgr,
                        str(idx),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

    ok, encoded = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode overlay")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/manager/corrections/{plan_id}/summary", response_model=ManualCorrectionsSummary)
async def manager_get_corrections_summary(plan_id: str) -> ManualCorrectionsSummary:
    proj = _get_project_or_404(plan_id)
    return _build_corrections_summary(plan_id, proj)


@app.post("/manager/corrections/{plan_id}/apply", response_model=ManualCorrectionsSummary)
async def manager_apply_correction(
    plan_id: str, request: ManualCorrectionApplyRequest
) -> ManualCorrectionsSummary:
    proj = _get_project_or_404(plan_id)
    walls = proj.get("thick_walls")
    if walls is None:
        # Try reload from disk
        meta_tmp = proj.get("metadata", {})
        wpath = meta_tmp.get("_asset_thick_walls_path", "")
        if wpath and os.path.exists(wpath):
            try:
                data = np.fromfile(wpath, dtype=np.uint8)
                walls = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if walls is not None:
                    proj["thick_walls"] = walls
            except Exception:
                pass
    if walls is None:
        raise HTTPException(status_code=400, detail="Plan walls mask not available. Please re-upload the plan PDF.")
    if request.mode not in {"add", "remove"}:
        raise HTTPException(status_code=400, detail="mode must be add/remove")
    if request.display_width <= 0 or request.display_height <= 0:
        raise HTTPException(status_code=400, detail="display dimensions must be positive")

    h, w = walls.shape[:2]
    stroke_mask = _build_mask_from_strokes(
        strokes=request.strokes,
        display_width=request.display_width,
        display_height=request.display_height,
        target_width=w,
        target_height=h,
    )
    corr = _get_manual_corrections(plan_id)
    key = "added_walls" if request.mode == "add" else "removed_walls"
    existing = corr.get(key)
    if isinstance(existing, np.ndarray):
        corr[key] = cv2.bitwise_or(existing, stroke_mask)
    else:
        corr[key] = stroke_mask
    MANUAL_CORRECTIONS[plan_id] = corr
    return _build_corrections_summary(plan_id, proj)


@app.post("/manager/corrections/{plan_id}/reset", response_model=ManualCorrectionsSummary)
async def manager_reset_corrections(plan_id: str) -> ManualCorrectionsSummary:
    proj = _get_project_or_404(plan_id)
    MANUAL_CORRECTIONS.pop(plan_id, None)
    return _build_corrections_summary(plan_id, proj)


@app.post("/manager/corrections/{plan_id}/save", response_model=ManualCorrectionsSummary)
async def manager_save_corrections(plan_id: str) -> ManualCorrectionsSummary:
    proj = _get_project_or_404(plan_id)
    corrected = _get_corrected_walls(plan_id, proj)
    if corrected is None:
        raise HTTPException(status_code=404, detail="Plan walls mask not found")
    proj["thick_walls"] = corrected
    proj["raw_pixels"] = int(np.count_nonzero(corrected))
    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))
    if scale_px_per_meter <= 0:
        scale_px_per_meter = 200.0
    proj["total_length"] = float(proj["raw_pixels"]) / scale_px_per_meter
    MANUAL_CORRECTIONS.pop(plan_id, None)
    _persist_plan_to_database(plan_id, proj)
    return _build_corrections_summary(plan_id, proj)


@app.get("/manager/corrections/{plan_id}/overlay")
async def manager_get_corrections_overlay(
    plan_id: str,
    variant: str = "auto",
) -> Response:
    proj = _get_project_or_404(plan_id)
    image = proj.get("original")
    walls = proj.get("thick_walls")
    if image is None or walls is None:
        raise HTTPException(status_code=404, detail="Plan image or walls not found")

    if variant == "corrected":
        walls_for_overlay = _get_corrected_walls(plan_id, proj)
        if walls_for_overlay is None:
            raise HTTPException(status_code=404, detail="Corrected walls not found")
    else:
        walls_for_overlay = walls

    overlay_bgr = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay_bgr[walls_for_overlay > 0] = (0, 165, 255)
    ok, encoded = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode overlay")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/manager/plans/{plan_id}/readiness", response_model=PlanReadinessResponse)
async def manager_get_plan_readiness(plan_id: str) -> PlanReadinessResponse:
    proj = _get_project_or_404(plan_id)
    return _build_plan_readiness(plan_id, proj)


@app.get("/manager/planning/{plan_id}", response_model=PlanningState)
async def manager_get_planning_state(plan_id: str) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    return _build_planning_state(plan_id, proj)


@app.post("/manager/planning/{plan_id}/calibrate", response_model=PlanningState)
async def manager_calibrate_scale(
    plan_id: str,
    request: PlanningCalibrateRequest,
) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    display_scale = request.display_scale if request.display_scale > 0 else 1.0
    dx = (request.x2 - request.x1) / display_scale
    dy = (request.y2 - request.y1) / display_scale
    line_len_px_original = float(np.sqrt(dx * dx + dy * dy))
    if request.real_length_m <= 0:
        raise HTTPException(status_code=400, detail="real_length_m must be positive")
    new_scale = line_len_px_original / request.real_length_m
    proj["scale"] = new_scale
    meta = proj.get("metadata", {})
    meta["scale_px_per_meter"] = new_scale
    proj["metadata"] = meta
    return _build_planning_state(plan_id, proj)


@app.put("/manager/planning/{plan_id}/categories", response_model=PlanningState)
async def manager_upsert_categories(
    plan_id: str,
    request: PlanningCategoryUpsertRequest,
) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    categories = {}
    for key, cat in request.categories.items():
        categories[key] = {
            "type": cat.type,
            "subtype": cat.subtype,
            "params": cat.params or {},
        }
    proj["planning"]["categories"] = categories
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


@app.post("/manager/planning/{plan_id}/items", response_model=PlanningState)
async def manager_add_planning_item(
    plan_id: str,
    request: PlanningAddItemRequest,
) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    categories = proj["planning"].get("categories", {})
    if request.category_key not in categories:
        raise HTTPException(status_code=400, detail="Invalid category_key")

    display_scale = request.display_scale if request.display_scale > 0 else 1.0
    obj = request.raw_object
    obj_original = dict(obj)
    length_px = 0.0
    area_px = 0.0

    if request.object_type == "line":
        length_px = _compute_line_length_px(obj)
        obj_original["x1"] = float(obj.get("x1", 0.0)) / display_scale
        obj_original["y1"] = float(obj.get("y1", 0.0)) / display_scale
        obj_original["x2"] = float(obj.get("x2", 0.0)) / display_scale
        obj_original["y2"] = float(obj.get("y2", 0.0)) / display_scale
    elif request.object_type == "rect":
        area_px = _compute_rect_area_px(obj)
        obj_original["x"] = float(obj.get("x", 0.0)) / display_scale
        obj_original["y"] = float(obj.get("y", 0.0)) / display_scale
        obj_original["width"] = float(obj.get("width", 0.0)) / display_scale
        obj_original["height"] = float(obj.get("height", 0.0)) / display_scale
    elif request.object_type == "path":
        length_px = _compute_path_length_px(obj)
        points = obj.get("points", [])
        if isinstance(points, list):
            obj_original["points"] = [
                [float(p[0]) / display_scale, float(p[1]) / display_scale]
                for p in points
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
    else:
        raise HTTPException(status_code=400, detail="Unsupported object_type")

    drawing_bbox = _get_drawing_bbox_original(proj)
    if not _object_inside_drawing_bbox(
        object_type=request.object_type,
        obj_original=obj_original,
        bbox=drawing_bbox,
    ):
        raise HTTPException(
            status_code=400,
            detail="הסימון מחוץ לתחום השרטוט ולכן לא נספר.",
        )

    category_meta = categories.get(request.category_key, {})
    is_wall_category = _is_wall_category_type(category_meta.get("type"))
    wall_overlap_ratio: Optional[float] = None
    ink_ratio: Optional[float] = None
    requires_wall_confirmation = False
    if is_wall_category and request.object_type in {"line", "path"}:
        walls = proj.get("thick_walls")
        original_img = proj.get("original")
        if isinstance(walls, np.ndarray) and walls.size > 0:
            wall_overlap_ratio = _estimate_wall_overlap_ratio(
                obj=obj,
                object_type=request.object_type,
                walls_mask=walls,
                display_scale=display_scale,
            )
            ink_ratio = (
                _estimate_nonwhite_ink_ratio(
                    obj=obj,
                    object_type=request.object_type,
                    original_img=original_img,
                    display_scale=display_scale,
                )
                if isinstance(original_img, np.ndarray) and original_img.size > 0
                else 0.0
            )
            reject_th, confirm_th = _get_wall_overlap_thresholds(proj)
            # Reject only when both signals say "nothing here".
            if wall_overlap_ratio < reject_th and ink_ratio < 0.015:
                raise HTTPException(
                    status_code=400,
                    detail="הסימון לא יושב על קיר מזוהה ולכן לא נספר כקיר.",
                )
            # Ask only in truly ambiguous cases (not enough mask and low ink evidence).
            if wall_overlap_ratio < confirm_th and ink_ratio < 0.03:
                requires_wall_confirmation = True

    # conversion: canvas px -> original px -> meters
    length_px_original = length_px / display_scale
    area_px_original = area_px / (display_scale * display_scale)

    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))

    length_m = length_px_original / scale_px_per_meter if length_px_original > 0 else 0.0
    area_m2 = (
        area_px_original / (scale_px_per_meter * scale_px_per_meter)
        if area_px_original > 0
        else 0.0
    )
    analysis = _detect_openings_for_planning_item(
        proj=proj,
        object_type=request.object_type,
        raw_object_display=obj,
        scale_px_per_meter=scale_px_per_meter,
        display_scale=display_scale,
        line_length_m=float(length_m),
    )
    if wall_overlap_ratio is not None:
        analysis["wall_overlap_ratio"] = round(float(wall_overlap_ratio), 4)
    if ink_ratio is not None:
        analysis["ink_ratio"] = round(float(ink_ratio), 4)
    analysis["requires_wall_confirmation"] = bool(requires_wall_confirmation)
    analysis["wall_confirmed"] = None if requires_wall_confirmation else True

    item = {
        "uid": str(uuid.uuid4())[:8],
        "type": request.object_type,
        "category": request.category_key,
        "length_m": round(float(length_m), 4),
        "length_m_effective": round(float(0.0 if requires_wall_confirmation else length_m), 4),
        "area_m2": round(float(area_m2), 4),
        "raw_object": obj_original,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
    }
    proj["planning"]["items"].append(item)
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


@app.delete("/manager/planning/{plan_id}/items/{item_uid}", response_model=PlanningState)
async def manager_delete_planning_item(plan_id: str, item_uid: str) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    items = proj["planning"].get("items", [])
    proj["planning"]["items"] = [it for it in items if it.get("uid") != item_uid]
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


# ── Text item endpoint ─────────────────────────────────────────────────────
@app.post("/manager/planning/{plan_id}/text-items", response_model=PlanningState)
async def manager_add_text_item(
    plan_id: str,
    request: PlanningTextItemRequest,
) -> PlanningState:
    """Add a free-text BOQ row (no drawing required)."""
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    categories = proj["planning"].get("categories", {})

    # Allow a special "__manual__" category key for ad-hoc items
    if request.category_key != "__manual__" and request.category_key not in categories:
        raise HTTPException(status_code=400, detail="Invalid category_key")

    item = {
        "uid": str(uuid.uuid4())[:8],
        "type": "text",
        "item_subtype": "text",
        "category": request.category_key,
        "description": request.description.strip(),
        "quantity": round(float(request.quantity), 4),
        "unit": request.unit.strip() or "יח׳",
        "note": request.note.strip(),
        # Map to standard fields so BOQ computation works
        "length_m": round(float(request.quantity), 4) if request.unit in ("מ׳", "מטר") else 0.0,
        "length_m_effective": round(float(request.quantity), 4) if request.unit in ("מ׳", "מטר") else 0.0,
        "area_m2": round(float(request.quantity), 4) if request.unit in ('מ"ר', "מטר רבוע") else 0.0,
        "raw_object": {"description": request.description, "quantity": request.quantity, "unit": request.unit},
        "analysis": {},
        "timestamp": datetime.now().isoformat(),
    }
    proj["planning"]["items"].append(item)
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


# ── Import vision elements as free text items ──────────────────────────────
_VIS_ELEM_LABELS: dict[str, tuple[str, str]] = {
    "door":     ("דלת",              "יח׳"),
    "window":   ("חלון",             "יח׳"),
    "stair":    ("מדרגות",           "מ'"),
    "elevator": ("מעלית",            "יח׳"),
    "sink":     ("כיור",             "יח׳"),
    "toilet":   ("אסלה",             "יח׳"),
    "shower":   ("מקלחת / אמבטיה",   "יח׳"),
    "boiler":   ("דוד מים",          "יח׳"),
    "other":    ("פריט מיוחד",       "יח׳"),
}


@app.post("/manager/planning/{plan_id}/import-vision-items", response_model=PlanningState)
async def import_vision_items(plan_id: str) -> PlanningState:
    """
    Read vision_elements from the plan's title-block extraction and bulk-insert
    them as free text items (category __manual__).
    Elements are grouped by type; a count + location notes are stored per group.
    """
    from collections import Counter as _Counter
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    meta = proj.get("metadata") or {}
    elements: list[dict] = meta.get("vision_elements") or []

    if not elements:
        return _build_planning_state(plan_id, proj)

    type_counts: _Counter[str] = _Counter()
    type_notes: dict[str, list[str]] = {}
    for elem in elements:
        etype = str(elem.get("type", "other")).lower().strip()
        type_counts[etype] += 1
        loc = str(elem.get("location") or elem.get("id") or "").strip()
        if loc:
            type_notes.setdefault(etype, []).append(loc)

    now = datetime.now().isoformat()
    for etype, count in type_counts.most_common():
        label, unit = _VIS_ELEM_LABELS.get(etype, (etype, "יח׳"))
        locs = type_notes.get(etype, [])
        note = ", ".join(locs[:5])
        if len(locs) > 5:
            note += f" ועוד {len(locs) - 5}"
        if not note:
            note = "מיובא מתוכנית"
        item = {
            "uid": str(uuid.uuid4())[:8],
            "type": "text",
            "item_subtype": "text",
            "category": "__manual__",
            "description": label,
            "quantity": float(count),
            "unit": unit,
            "note": note,
            "length_m": 0.0,
            "length_m_effective": 0.0,
            "area_m2": 0.0,
            "raw_object": {"description": label, "quantity": float(count), "unit": unit},
            "analysis": {},
            "timestamp": now,
        }
        proj["planning"]["items"].append(item)
    _recompute_boq(proj)
    _persist_plan_to_database(plan_id, proj)
    return _build_planning_state(plan_id, proj)


# ── Zone item endpoint ─────────────────────────────────────────────────────
@app.post("/manager/planning/{plan_id}/zone-items", response_model=PlanningState)
async def manager_add_zone_item(
    plan_id: str,
    request: PlanningZoneRequest,
) -> PlanningState:
    """
    Paint a rectangular zone on the plan.
    Counts thick_walls pixels inside the rect and converts to meters.
    Falls back to perimeter/2 if no wall mask available.
    """
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    categories = proj["planning"].get("categories", {})
    if request.category_key not in categories:
        raise HTTPException(status_code=400, detail="Invalid category_key")

    _ensure_arrays_loaded(proj)
    scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))
    category_meta = categories.get(request.category_key, {})
    cat_type = category_meta.get("type", "")

    x = max(0, int(request.x))
    y = max(0, int(request.y))
    w = max(1, int(request.width))
    h = max(1, int(request.height))

    length_m = 0.0
    area_m2 = 0.0
    method = "perimeter"

    walls = proj.get("thick_walls")
    original_img = proj.get("original")

    # Clamp to image bounds
    img_h, img_w = (0, 0)
    if isinstance(original_img, np.ndarray) and original_img.size > 0:
        img_h, img_w = original_img.shape[:2]
    elif isinstance(walls, np.ndarray) and walls.size > 0:
        img_h, img_w = walls.shape[:2]

    if img_w > 0 and img_h > 0:
        x = min(x, img_w - 1)
        y = min(y, img_h - 1)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

    is_wall = _is_wall_category_type(cat_type)
    is_floor = cat_type in ("ריצוף", "ריצוף/תקרה")

    if is_floor:
        # Area = rect area in m²
        area_m2 = (w * h) / (scale_px_per_meter ** 2)
        method = "rect_area"
    elif is_wall and isinstance(walls, np.ndarray) and walls.size > 0:
        # Count wall pixels inside rect → skeleton length → meters
        roi = walls[y:y + h, x:x + w]
        wall_px = int(np.count_nonzero(roi))
        if wall_px > 0:
            # Skeletonize to get centerline length estimate
            try:
                from skimage.morphology import skeletonize
                roi_bin = (roi > 0).astype(np.uint8)
                skel = skeletonize(roi_bin)
                length_px = int(np.count_nonzero(skel))
            except Exception:
                # Fallback: wall pixels / avg wall thickness (estimate 8px)
                avg_thickness_px = max(1, int(scale_px_per_meter * 0.15))
                length_px = wall_px // avg_thickness_px
            length_m = length_px / scale_px_per_meter
            method = "wall_mask"
        else:
            # No walls detected in zone → use half-perimeter as fallback
            length_m = (w + h) / scale_px_per_meter
            method = "perimeter_fallback"
    else:
        # Generic fallback: half-perimeter
        length_m = (w + h) / scale_px_per_meter
        method = "perimeter"

    item = {
        "uid": str(uuid.uuid4())[:8],
        "type": "zone",
        "item_subtype": "zone",
        "category": request.category_key,
        "length_m": round(float(length_m), 4),
        "length_m_effective": round(float(length_m), 4),
        "area_m2": round(float(area_m2), 4),
        "raw_object": {"x": x, "y": y, "width": w, "height": h},
        "analysis": {"zone_method": method},
        "timestamp": datetime.now().isoformat(),
    }
    proj["planning"]["items"].append(item)
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


# ── Auto-analyze endpoint ──────────────────────────────────────────────────
@app.post("/manager/planning/{plan_id}/auto-analyze", response_model=AutoAnalyzeResponse)
async def manager_auto_analyze(plan_id: str) -> AutoAnalyzeResponse:
    """
    Segment thick_walls into individual wall segments using skeleton-based
    junction splitting (branch-point removal).
    Falls back to a safe downsampled computation when the stored skeleton
    is unavailable; never runs full-resolution skeletonize in-request to
    avoid OOM on large plans.
    """
    try:
        proj = _get_project_or_404(plan_id)
        _ensure_arrays_loaded(proj)
        scale_px_per_meter = float(get_scale_with_fallback(proj, default_scale=200.0))

        walls = proj.get("thick_walls")

        if not (isinstance(walls, np.ndarray) and walls.size > 0):
            meta_dbg = proj.get("metadata", {})
            filename_dbg = meta_dbg.get("filename") or meta_dbg.get("plan_id", "?")
            walls_nonzero = int(np.count_nonzero(walls)) if isinstance(walls, np.ndarray) else -1
            print(f"[auto-analyze] EMPTY: plan={filename_dbg} thick_walls={type(walls).__name__} nonzero={walls_nonzero} "
                  f"has_skel={proj.get('skeleton') is not None} has_orig={proj.get('original') is not None}")
            return AutoAnalyzeResponse(segments=[])

        binary = (walls > 0).astype(np.uint8)
        img_area = binary.shape[0] * binary.shape[1]
        img_h, img_w = binary.shape

        # ── Step 1: Get skeleton (prefer pre-computed; fallback at 1/4 res) ─────
        skeleton: np.ndarray | None = None
        stored_skel = proj.get("skeleton")
        if isinstance(stored_skel, np.ndarray) and stored_skel.shape[:2] == (img_h, img_w):
            skeleton = (stored_skel > 0).astype(np.uint8)

        if skeleton is None or not np.any(skeleton):
            # Pre-computed skeleton unavailable → compute at ¼ resolution to cap memory
            try:
                from skimage.morphology import skeletonize as _skeletonize
                sf = 4                                        # downsample factor
                sw, sh = max(1, img_w // sf), max(1, img_h // sf)
                small = cv2.resize(binary, (sw, sh), interpolation=cv2.INTER_NEAREST)
                small_skel = _skeletonize(small > 0).astype(np.uint8)
                skeleton = cv2.resize(small_skel, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                skeleton = (skeleton > 0).astype(np.uint8)
            except Exception as _se:
                print(f"[auto-analyze] skeletonize fallback failed: {_se}")
                return AutoAnalyzeResponse(segments=[])

        # ── Step 2: Detect branch points (junction pixels with ≥3 neighbours) ─
        neighbor_sum = cv2.filter2D(
            skeleton.astype(np.float32), -1,
            np.ones((3, 3), np.float32)
        )
        branch_mask = (skeleton > 0) & (neighbor_sum >= 4)

        # ── Step 3: Cut skeleton at branch points ─────────────────────────────
        cut_skeleton = skeleton.copy()
        cut_skeleton[branch_mask] = 0

        # ── Step 4: Label segments (connectedComponentsWithStats for quick area)
        num_skel_labels, skel_labels, skel_stats, _ = cv2.connectedComponentsWithStats(
            cut_skeleton, connectivity=8
        )

        # Dilation kernel: expands each skeleton segment back to wall thickness
        half_thick = max(2, int(scale_px_per_meter * 0.10))
        dil_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (half_thick * 2 + 1, half_thick * 2 + 1)
        )
        pad = half_thick + 2      # ROI padding for dilation spillover

        wall_counter = 0
        fixture_counter = 0
        concrete_arr = proj.get("concrete_mask")
        blocks_arr   = proj.get("blocks_mask")
        has_concrete = isinstance(concrete_arr, np.ndarray) and concrete_arr.size > 0

        segments: list[AutoAnalyzeSegment] = []

        for label_id in range(1, num_skel_labels):
            # Quick pre-filter: skeleton pixel count from stats (cheap)
            skel_area = int(skel_stats[label_id, cv2.CC_STAT_AREA])
            if skel_area < 5:       # orphan / single pixel – skip
                continue

            # ── Step 5: Recover thick-wall pixels using ROI (fast) ───────────
            sx = int(skel_stats[label_id, cv2.CC_STAT_LEFT])
            sy = int(skel_stats[label_id, cv2.CC_STAT_TOP])
            sw = int(skel_stats[label_id, cv2.CC_STAT_WIDTH])
            sh = int(skel_stats[label_id, cv2.CC_STAT_HEIGHT])

            x1 = max(0, sx - pad);  y1 = max(0, sy - pad)
            x2 = min(img_w, sx + sw + pad);  y2 = min(img_h, sy + sh + pad)

            skel_roi    = (skel_labels[y1:y2, x1:x2] == label_id).astype(np.uint8)
            dilated_roi = cv2.dilate(skel_roi, dil_k)
            region_roi  = binary[y1:y2, x1:x2] & dilated_roi

            area_px = int(np.count_nonzero(region_roi))
            # Pre-filter: skip absolute orphans (< 0.01% of image)
            if area_px < img_area * 0.0001:
                continue

            ys_r, xs_r = np.where(region_roi > 0)
            if len(ys_r) == 0:
                continue
            g_bx = int(xs_r.min()) + x1
            g_by = int(ys_r.min()) + y1
            g_bw = int(xs_r.max()) - int(xs_r.min()) + 1
            g_bh = int(ys_r.max()) - int(ys_r.min()) + 1

            aspect = g_bw / max(1, g_bh)
            is_elongated = aspect > 3.0 or aspect < 0.33

            length_px = skel_area
            length_m  = round(length_px / scale_px_per_meter, 2)
            area_m2   = round(area_px / (scale_px_per_meter ** 2), 2)

            length_area_ratio = length_m / max(0.001, area_m2)
            is_fixture = (
                not is_elongated
                and length_area_ratio < 4.0
                and area_m2 < 1.5
                and area_px >= img_area * 0.00008
            )

            # Keep walls with ≥ 0.05% area (was 0.5% — too aggressive for short walls)
            # Keep fixtures with ≥ 0.008% area
            if is_fixture:
                if area_px < img_area * 0.00008:
                    continue
            else:
                if area_px < img_area * 0.0005:
                    continue

            if is_fixture:
                fixture_counter += 1
                if area_m2 < 0.12:
                    subtype = "פרט קטן"
                elif 0.5 <= aspect <= 2.0 and area_m2 < 0.45:
                    subtype = "כיור / אסלה"
                elif area_m2 < 1.2 and (aspect > 1.6 or aspect < 0.62):
                    subtype = "אמבטיה / מקלחת"
                else:
                    subtype = "ריהוט / מכשיר"
                segments.append(AutoAnalyzeSegment(
                    segment_id=f"fix_{label_id}",
                    label=f"{subtype} {fixture_counter}",
                    suggested_type="אביזר",
                    suggested_subtype=subtype,
                    confidence=0.55,
                    length_m=float(length_m),
                    area_m2=float(area_m2),
                    bbox=[float(g_bx), float(g_by), float(g_bw), float(g_bh)],
                    element_class="fixture",
                ))
            else:
                wall_counter += 1
                if is_elongated:
                    suggested_subtype = "בטון"
                    confidence = 0.82
                else:
                    suggested_subtype = "בלוקים"
                    confidence = 0.65

                if has_concrete:
                    roi_mask     = region_roi.astype(bool)
                    concrete_roi = concrete_arr[y1:y2, x1:x2]
                    blocks_roi   = (
                        blocks_arr[y1:y2, x1:x2]
                        if isinstance(blocks_arr, np.ndarray) and blocks_arr.size > 0
                        else None
                    )
                    c_px = int(np.count_nonzero(concrete_roi[roi_mask]))
                    b_px = int(np.count_nonzero(blocks_roi[roi_mask])) if blocks_roi is not None else 0
                    if c_px > b_px and c_px > area_px * 0.3:
                        suggested_subtype = "בטון"
                        confidence = min(0.95, 0.75 + c_px / area_px * 0.3)
                    elif b_px > c_px and b_px > area_px * 0.3:
                        suggested_subtype = "בלוקים"
                        confidence = min(0.92, 0.72 + b_px / area_px * 0.3)

                segments.append(AutoAnalyzeSegment(
                    segment_id=f"seg_{label_id}",
                    label=f"קיר {wall_counter}",
                    suggested_type="קירות",
                    suggested_subtype=suggested_subtype,
                    confidence=round(float(confidence), 2),
                    length_m=float(length_m),
                    area_m2=float(area_m2),
                    bbox=[float(g_bx), float(g_by), float(g_bw), float(g_bh)],
                    element_class="wall",
                ))

        walls_out    = sorted([s for s in segments if s.element_class == "wall"],    key=lambda s: s.length_m, reverse=True)
        fixtures_out = sorted([s for s in segments if s.element_class == "fixture"], key=lambda s: s.area_m2,  reverse=True)
        total_skel_labels = num_skel_labels - 1  # exclude background
        print(f"[auto-analyze] plan={plan_id} skel_labels={total_skel_labels} "
              f"walls={len(walls_out)} fixtures={len(fixtures_out)} img_area={img_area}")

        # ── Attach Vision data (extracted by Claude Vision during upload) ──────
        meta = proj.get("metadata") or {}
        vision_data = AutoAnalyzeVisionData(
            rooms=meta.get("llm_rooms"),
            dimensions=meta.get("vision_dimensions"),
            dimensions_structured=meta.get("vision_dimensions_structured"),
            materials=meta.get("vision_materials"),
            materials_legend=meta.get("vision_materials_legend"),
            elements=meta.get("vision_elements"),
            elevations=meta.get("vision_elevations"),
            grid_lines=meta.get("vision_grid_lines"),
            systems=meta.get("vision_systems"),
            total_area_m2=meta.get("vision_total_area_m2"),
            plan_title=meta.get("plan_title"),
            project_name=meta.get("project_name"),
            sheet_number=meta.get("sheet_number"),
            sheet_name=meta.get("sheet_name"),
            status=meta.get("status"),
            architect=meta.get("architect"),
            date=meta.get("date"),
            scale=meta.get("scale_text"),
            execution_notes=meta.get("execution_notes"),
        )
        # Only include vision_data if at least one field has content
        has_vision = any([
            vision_data.rooms, vision_data.dimensions, vision_data.materials,
            vision_data.elements, vision_data.rooms, vision_data.total_area_m2,
        ])

        return AutoAnalyzeResponse(
            segments=walls_out + fixtures_out,
            vision_data=vision_data if has_vision else None,
        )

    except HTTPException:
        raise
    except Exception as _exc:
        import traceback as _tb
        print(f"[auto-analyze ERROR] {_exc}\n{_tb.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ניתוח אוטומטי נכשל: {_exc}")


@app.post("/manager/planning/{plan_id}/confirm-auto-segment", response_model=PlanningState)
async def manager_confirm_auto_segment(
    plan_id: str,
    request: ConfirmAutoSegmentRequest,
) -> PlanningState:
    """Convert a confirmed auto-segment into a zone item."""
    bx, by, bw, bh = (request.bbox + [0.0, 0.0, 0.0, 0.0])[:4]
    zone_req = PlanningZoneRequest(
        category_key=request.category_key,
        x=float(bx), y=float(by),
        width=float(bw), height=float(bh),
    )
    return await manager_add_zone_item(plan_id, zone_req)


@app.post("/manager/planning/{plan_id}/items/{item_uid}/resolve-opening", response_model=PlanningState)
async def manager_resolve_planning_opening(
    plan_id: str, item_uid: str, request: PlanningResolveOpeningRequest
) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    items = proj["planning"].get("items", [])
    target = None
    for item in items:
        if item.get("uid") == item_uid:
            target = item
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Planning item not found")

    opening_type = str(request.opening_type or "").strip().lower()
    if opening_type not in {"door", "window", "none"}:
        raise HTTPException(status_code=400, detail="opening_type must be door/window/none")

    analysis = target.get("analysis", {}) if isinstance(target.get("analysis"), dict) else {}
    openings = analysis.get("openings", []) if isinstance(analysis.get("openings"), list) else []
    selected_gap = None
    if openings:
        if request.gap_id:
            for gap in openings:
                if str(gap.get("gap_id")) == str(request.gap_id):
                    selected_gap = gap
                    break
        if selected_gap is None and request.gap_id:
            selected_gap = openings[0]

    original_len = float(target.get("length_m", 0.0) or 0.0)
    if opening_type in {"door", "window"}:
        if selected_gap is not None:
            gap_len = float((selected_gap or {}).get("length_m", 0.0) or 0.0)
        else:
            # If user accepted opening type without choosing a specific gap,
            # deduct all detected internal gaps for a more faithful BOQ length.
            gap_len = float(
                sum(
                    _safe_float(g.get("length_m"), 0.0)
                    for g in openings
                    if isinstance(g, dict)
                )
            )
            # Guardrail: never deduct unrealistic amount from one wall.
            gap_len = min(gap_len, max(0.0, original_len * 0.65))
        if gap_len <= 0:
            # Generic defaults when automatic gap localization misses exact segment.
            gap_len = 0.9 if opening_type == "door" else 1.2
    else:
        gap_len = 0.0
    if opening_type in {"door", "window"}:
        target["length_m_effective"] = round(max(0.0, original_len - gap_len), 4)
    else:
        target["length_m_effective"] = round(original_len, 4)

    analysis["resolved_opening_type"] = opening_type
    analysis["resolved_gap_id"] = selected_gap.get("gap_id") if isinstance(selected_gap, dict) else None
    analysis["deducted_length_m"] = round(gap_len if opening_type in {"door", "window"} else 0.0, 4)
    target["analysis"] = analysis

    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


@app.post("/manager/planning/{plan_id}/items/{item_uid}/resolve-wall", response_model=PlanningState)
async def manager_resolve_planning_wall(
    plan_id: str, item_uid: str, request: PlanningResolveWallRequest
) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    items = proj["planning"].get("items", [])
    target = None
    for item in items:
        if item.get("uid") == item_uid:
            target = item
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Planning item not found")

    analysis = target.get("analysis", {}) if isinstance(target.get("analysis"), dict) else {}
    original_len = float(target.get("length_m", 0.0) or 0.0)
    if request.is_wall:
        target["length_m_effective"] = round(original_len, 4)
        analysis["wall_confirmed"] = True
        # Learn that this drawing likely has thin wall lines.
        meta = proj.get("metadata", {})
        if isinstance(meta, dict):
            meta["thin_wall_mode"] = True
            proj["metadata"] = meta
    else:
        target["length_m_effective"] = 0.0
        analysis["wall_confirmed"] = False

    analysis["requires_wall_confirmation"] = False
    target["analysis"] = analysis
    _recompute_boq(proj)
    return _build_planning_state(plan_id, proj)


@app.post("/manager/planning/{plan_id}/finalize", response_model=PlanningState)
async def manager_finalize_planning(plan_id: str) -> PlanningState:
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    _recompute_boq(proj)

    meta = proj.get("metadata", {})

    # Serialize items to plain dicts so _safe_json_dumps can handle them
    raw_items = proj["planning"].get("items", [])
    serializable_items = []
    for item in raw_items:
        if hasattr(item, "model_dump"):
            serializable_items.append(item.model_dump())
        elif hasattr(item, "dict"):
            serializable_items.append(item.dict())
        elif isinstance(item, dict):
            serializable_items.append(item)
        else:
            serializable_items.append(str(item))

    meta["planning"] = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "scale": float(proj.get("scale") or 0.0),
        "categories": proj["planning"].get("categories", {}),
        "planned_items": serializable_items,
        "planned_items_count": len(serializable_items),
        "boq": proj["planning"].get("boq", {}),
        "totals": proj["planning"].get("totals", {"total_length_m": 0.0, "total_area_m2": 0.0}),
        "sections": proj["planning"].get("sections", []),
    }
    proj["metadata"] = meta

    try:
        db_id = _persist_plan_to_database(plan_id, proj)
        if db_id is None:
            raise HTTPException(
                status_code=500,
                detail="שמירה למסד הנתונים נכשלה — ייתכן שהתוכנית אינה קיימת ב-DB. נסה להעלות מחדש."
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"שגיאה בשמירה למסד הנתונים: {exc}"
        ) from exc

    return _build_planning_state(plan_id, proj)


# ── Work Sections (גזרות עבודה) ─────────────────────────────────────────────

@app.get("/manager/planning/{plan_id}/sections", response_model=list[WorkSection])
async def manager_get_sections(plan_id: str) -> list[WorkSection]:
    """מחזיר את רשימת הגזרות של תוכנית."""
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    raw = proj["planning"].get("sections", [])
    return [WorkSection(**s) if isinstance(s, dict) else s for s in raw]


@app.post("/manager/planning/{plan_id}/sections", response_model=PlanningState)
async def manager_add_section(plan_id: str, req: WorkSectionCreateRequest) -> PlanningState:
    """מוסיף גזרת עבודה חדשה לתוכנית."""
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)

    uid = str(uuid.uuid4())[:8]
    name = req.name.strip() or f"גזרה {len(proj['planning'].get('sections', [])) + 1}"
    section = WorkSection(
        uid=uid,
        name=name,
        contractor=req.contractor.strip(),
        worker=req.worker.strip(),
        color=req.color,
        x=req.x,
        y=req.y,
        width=req.width,
        height=req.height,
    )
    sections = proj["planning"].setdefault("sections", [])
    sections.append(section.model_dump())
    return _build_planning_state(plan_id, proj)


@app.delete("/manager/planning/{plan_id}/sections/{section_uid}", response_model=PlanningState)
async def manager_delete_section(plan_id: str, section_uid: str) -> PlanningState:
    """מוחק גזרת עבודה."""
    proj = _get_project_or_404(plan_id)
    _init_planning_if_missing(proj)
    sections = proj["planning"].get("sections", [])
    proj["planning"]["sections"] = [s for s in sections if (s.get("uid") if isinstance(s, dict) else s.uid) != section_uid]
    return _build_planning_state(plan_id, proj)


@app.post("/worker/measure-item", response_model=WorkerMeasuredItem)
async def worker_measure_item(request: WorkerMeasuredItemRequest) -> WorkerMeasuredItem:
    """
    מחשב מדידה לפריט מצויר בצד העובד.
    """
    return _measure_worker_item(
        plan_id=request.plan_id,
        object_type=request.object_type,
        raw_object=request.raw_object,
        display_scale=request.display_scale,
        report_type=request.report_type,
    )


@app.post("/worker/reports", response_model=WorkerReport)
async def worker_create_report(request: WorkerReportCreateRequest) -> WorkerReport:
    """
    שומר דיווח עובד מלא.
    """
    proj = _get_project_or_404(request.plan_id)
    meta = proj.get("metadata", {})
    plan_name = meta.get("plan_name") or request.plan_id

    total_length = (
        sum(item.measurement for item in request.items if item.unit == "m")
        if request.report_type == "walls"
        else 0.0
    )
    total_area = (
        sum(item.measurement for item in request.items if item.unit in ("m2", "m²"))
        if request.report_type != "walls"
        else 0.0
    )

    report = WorkerReport(
        id=str(uuid.uuid4())[:8],
        plan_id=request.plan_id,
        plan_name=plan_name,
        date=request.date,
        shift=request.shift,
        report_type=request.report_type,
        draw_mode=request.draw_mode,
        items=request.items,
        total_length_m=round(float(total_length), 4),
        total_area_m2=round(float(total_area), 4),
        note=request.note or "",
    )
    WORKER_REPORTS.append(report.model_dump())

    db_plan_id = _get_db_plan_id(proj)
    if db_plan_id is not None:
        report_quantity = (
            report.total_area_m2 if request.report_type != "walls" else report.total_length_m
        )
        default_work_type = "ריצוף/חיפוי" if request.report_type != "walls" else "קירות"
        save_progress_report(db_plan_id, float(report_quantity), default_work_type)

    return report


@app.get("/worker/reports/{plan_id}", response_model=list[WorkerReport])
async def worker_list_reports(plan_id: str) -> list[WorkerReport]:
    """
    מחזיר את כל הדיווחים של תוכנית.
    """
    reports = [r for r in WORKER_REPORTS if r.get("plan_id") == plan_id]
    return [WorkerReport(**r) for r in reports]


@app.get("/manager/drawing-data/{plan_id}", response_model=DrawingDataSummary)
async def manager_get_drawing_data(plan_id: str) -> DrawingDataSummary:
    proj = _get_project_or_404(plan_id)
    return _build_drawing_data_summary(plan_id, proj)


@app.post("/manager/drawing-data/{plan_id}/scale", response_model=DrawingDataScaleResult)
async def manager_calculate_drawing_scale(
    plan_id: str, request: DrawingDataScaleRequest
) -> DrawingDataScaleResult:
    proj = _get_project_or_404(plan_id)
    summary = _build_drawing_data_summary(plan_id, proj)
    if request.paper_width_mm <= 0 or request.paper_height_mm <= 0:
        raise HTTPException(status_code=400, detail="paper dimensions must be positive")

    paper_width_m = request.paper_width_mm / 1000.0
    paper_height_m = request.paper_height_mm / 1000.0
    px_per_meter_w = summary.image_width_px / max(paper_width_m, 1e-9)
    px_per_meter_h = summary.image_height_px / max(paper_height_m, 1e-9)
    calculated_scale = (px_per_meter_w + px_per_meter_h) / 2.0
    current_scale = summary.scale_px_per_meter
    scale_diff = calculated_scale - current_scale
    error_percent = abs(scale_diff / calculated_scale * 100.0) if calculated_scale > 0 else 0.0

    walls = proj.get("thick_walls")
    wall_pixels = _safe_float(np.count_nonzero(walls), 0.0)
    length_current = wall_pixels / max(current_scale, 1e-9)
    length_calculated = wall_pixels / max(calculated_scale, 1e-9)
    applied = False
    updated_summary: Optional[DrawingDataSummary] = None

    if request.apply_to_plan:
        proj["scale"] = calculated_scale
        meta = proj.get("metadata", {})
        meta["scale_px_per_meter"] = calculated_scale
        meta["pixels_per_meter"] = calculated_scale
        proj["metadata"] = meta
        _persist_plan_to_database(plan_id, proj)
        applied = True
        updated_summary = _build_drawing_data_summary(plan_id, proj)

    return DrawingDataScaleResult(
        calculated_scale_px_per_meter=round(float(calculated_scale), 4),
        current_scale_px_per_meter=round(float(current_scale), 4),
        scale_diff=round(float(scale_diff), 4),
        error_percent=round(float(error_percent), 4),
        length_with_current_scale_m=round(float(length_current), 4),
        length_with_calculated_scale_m=round(float(length_calculated), 4),
        length_diff_m=round(float(length_calculated - length_current), 4),
        applied=applied,
        updated_summary=updated_summary,
    )


@app.get("/manager/drawing-data/{plan_id}/export/csv")
async def manager_export_drawing_data_csv(plan_id: str) -> Response:
    summary = _build_drawing_data_summary(plan_id, _get_project_or_404(plan_id))
    csv_rows = [
        "סוג,כמות,יחידה",
        f"קירות בטון,{summary.concrete_length_m:.2f},מטר",
        f"קירות בלוקים,{summary.blocks_length_m:.2f},מטר",
        f"סה\"כ קירות,{summary.total_wall_length_m:.2f},מטר",
        f"ריצוף,{summary.flooring_area_m2:.2f},מ\"ר",
    ]
    return Response("\n".join(csv_rows), media_type="text/csv; charset=utf-8")


@app.get("/manager/drawing-data/{plan_id}/export/json")
async def manager_export_drawing_data_json(plan_id: str) -> Dict:
    summary = _build_drawing_data_summary(plan_id, _get_project_or_404(plan_id))
    return summary.model_dump()


@app.post("/manager/area-analysis/{plan_id}/run", response_model=FloorAnalysisResponse)
async def manager_run_area_analysis(
    plan_id: str, request: FloorAnalysisRunRequest
) -> FloorAnalysisResponse:
    proj = _get_project_or_404(plan_id)
    _ensure_arrays_loaded(proj)
    walls_mask = proj.get("thick_walls")
    original_img = proj.get("original")
    if walls_mask is None or original_img is None:
        # Try to reload from persisted asset paths (if plan was loaded from DB after server restart)
        meta_tmp = proj.get("metadata", {})
        def _try_load_color_local(path):
            if not path or not os.path.exists(path): return None
            try:
                data = np.fromfile(path, dtype=np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception: return None
        def _try_load_gray_local(path):
            if not path or not os.path.exists(path): return None
            try:
                data = np.fromfile(path, dtype=np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            except Exception: return None
        if original_img is None:
            original_img = _try_load_color_local(meta_tmp.get("_asset_original_path", ""))
            if original_img is not None:
                proj["original"] = original_img
        if walls_mask is None:
            walls_mask = _try_load_gray_local(meta_tmp.get("_asset_thick_walls_path", ""))
            if walls_mask is not None:
                proj["thick_walls"] = walls_mask
        if walls_mask is None or original_img is None:
            raise HTTPException(
                status_code=409,
                detail="PLAN_RESTART_LOST: נתוני התוכנית לא זמינים בשרת (ייתכן שהשרת עלה מחדש). אנא העלה את קובץ ה-PDF שוב."
            )

    meta = proj.get("metadata", {})
    llm_data = proj.get("llm_data") or proj.get("llm_suggestions")
    llm_rooms = llm_data.get("rooms", []) if isinstance(llm_data, dict) else []
    meters_per_pixel = _safe_float(meta.get("meters_per_pixel"), 0.0) or None
    meters_per_pixel_x = _safe_float(meta.get("meters_per_pixel_x"), 0.0) or None
    meters_per_pixel_y = _safe_float(meta.get("meters_per_pixel_y"), 0.0) or None
    if meters_per_pixel is None:
        px_per_m = _safe_float(proj.get("scale"), 0.0) or _safe_float(meta.get("pixels_per_meter"), 0.0)
        if px_per_m > 0:
            meters_per_pixel = 1.0 / px_per_m
            meters_per_pixel_x = meters_per_pixel_x or meters_per_pixel
            meters_per_pixel_y = meters_per_pixel_y or meters_per_pixel
            meta["meters_per_pixel"] = meters_per_pixel
            meta["meters_per_pixel_x"] = meters_per_pixel_x
            meta["meters_per_pixel_y"] = meters_per_pixel_y
            proj["metadata"] = meta

    try:
        result = analyze_floor_and_rooms(
            walls_mask=walls_mask,
            original_image=original_img,
            meters_per_pixel=meters_per_pixel,
            meters_per_pixel_x=meters_per_pixel_x,
            meters_per_pixel_y=meters_per_pixel_y,
            llm_rooms=llm_rooms,
            segmentation_method=request.segmentation_method,
            min_room_area_px=0 if request.auto_min_area else int(request.min_area_px),
        )
    except Exception as exc:
        result = {
            "success": False,
            "rooms": [],
            "totals": {
                "num_rooms": 0,
                "total_area_m2": None,
                "total_perimeter_m": None,
                "total_baseboard_m": None,
            },
            "limitations": [f"שגיאת backend בהרצת ניתוח שטחים: {str(exc)}"],
            "visualizations": {"overlay": None, "masks": {}},
            "debug_json_safe": {"exception": str(exc)},
        }

    try:
        refined_flooring = refine_flooring_mask_with_rooms(
            proj.get("flooring_mask"),
            result.get("visualizations", {}).get("masks"),
        )
        if refined_flooring is not None:
            proj["flooring_mask_refined"] = refined_flooring
            meta["pixels_flooring_area_refined"] = int(np.count_nonzero(refined_flooring))
            proj["metadata"] = meta
    except Exception:
        pass

    proj["floor_analysis_result"] = result
    _persist_plan_to_database(plan_id, proj)

    response = FloorAnalysisResponse(
        success=bool(result.get("success")),
        totals=FloorAnalysisTotals(**(result.get("totals") or {})),
        rooms=_clean_floor_rooms(result),
        limitations=[str(v) for v in result.get("limitations", [])],
        has_overlay=result.get("visualizations", {}).get("overlay") is not None,
        overlay_image_url=f"/manager/area-analysis/{plan_id}/overlay",
        segmentation_method=request.segmentation_method,
        min_area_px=0 if request.auto_min_area else int(request.min_area_px),
        debug_json_safe=result.get("debug_json_safe") or {},
    )
    return response


@app.get("/manager/area-analysis/{plan_id}", response_model=FloorAnalysisResponse)
async def manager_get_area_analysis(plan_id: str) -> FloorAnalysisResponse:
    proj = _get_project_or_404(plan_id)
    result = proj.get("floor_analysis_result")
    if not result:
        return FloorAnalysisResponse(
            success=False,
            totals=FloorAnalysisTotals(),
            rooms=[],
            limitations=["עדיין לא בוצע ניתוח שטחים לתוכנית זו."],
            has_overlay=False,
        )

    return FloorAnalysisResponse(
        success=bool(result.get("success")),
        totals=FloorAnalysisTotals(**(result.get("totals") or {})),
        rooms=_clean_floor_rooms(result),
        limitations=[str(v) for v in result.get("limitations", [])],
        has_overlay=result.get("visualizations", {}).get("overlay") is not None,
        overlay_image_url=f"/manager/area-analysis/{plan_id}/overlay",
        debug_json_safe=result.get("debug_json_safe") or {},
    )


@app.get("/manager/area-analysis/{plan_id}/overlay")
async def manager_get_area_overlay(plan_id: str) -> Response:
    proj = _get_project_or_404(plan_id)
    result = proj.get("floor_analysis_result")
    if not result:
        raise HTTPException(status_code=404, detail="Area analysis not found")
    overlay = result.get("visualizations", {}).get("overlay")
    if overlay is None:
        raise HTTPException(status_code=404, detail="Overlay image not found")

    if len(overlay.shape) == 3 and overlay.shape[2] == 3:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    else:
        overlay_bgr = overlay
    ok, encoded = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode overlay")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/manager/dashboard/{plan_id}", response_model=DashboardResponse)
async def manager_get_dashboard(plan_id: str) -> DashboardResponse:
    proj = PROJECTS.get(plan_id)
    db_plan_id = _resolve_db_plan_id(plan_id)
    if db_plan_id is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found in database")

    forecast = get_project_forecast(db_plan_id) or {}
    financial = get_project_financial_status(db_plan_id) or {}
    plan_row = get_plan_by_id(db_plan_id) or {}

    total_planned = _safe_float(forecast.get("total_planned"), 0.0)
    built_m = _safe_float(forecast.get("cumulative_progress"), 0.0)
    remaining = _safe_float(forecast.get("remaining_work"), 0.0)
    percent = (built_m / total_planned * 100.0) if total_planned > 0 else 0.0
    days_to_finish_raw = forecast.get("days_to_finish")
    days_to_finish = (
        _safe_float(days_to_finish_raw, 0.0) if days_to_finish_raw is not None else None
    )

    stats_df = load_stats_df()
    timeline: list[DashboardTimelinePoint] = []
    if not stats_df.empty and "שם תוכנית" in stats_df.columns:
        plan_name_for_chart = plan_row.get("plan_name") or (
            proj.get("metadata", {}).get("plan_name") if proj else None
        )
        df_current = stats_df[stats_df["שם תוכנית"] == plan_name_for_chart]
        for _, row in df_current.sort_values("תאריך").iterrows():
            timeline.append(
                DashboardTimelinePoint(
                    date=str(row.get("תאריך", ""))[:10],
                    quantity_m=round(_safe_float(row.get("כמות שבוצעה"), 0.0), 4),
                )
            )

    quantities = [item.quantity_m for item in timeline]
    avg_daily = float(np.mean(quantities)) if quantities else 0.0
    max_daily = float(np.max(quantities)) if quantities else 0.0

    reports = get_progress_reports(db_plan_id) or []
    planning_data = _extract_planning_from_sources(proj, plan_row)
    planned_walls_m, planned_floor_m2, boq_progress = _compute_planned_vs_built_from_planning(
        planning=planning_data,
        reports=reports,
    )
    built_walls_m = _safe_float(boq_progress[0].built_qty if boq_progress else 0.0, 0.0)
    built_floor_m2 = _safe_float(boq_progress[1].built_qty if len(boq_progress) > 1 else 0.0, 0.0)

    recent_reports: list[DashboardRecentReport] = []
    for r in reports[:5]:
        note = str(r.get("note", "") or "")
        is_floor = "ריצוף" in note or "חיפוי" in note
        recent_reports.append(
            DashboardRecentReport(
                id=str(r.get("id", "")),
                date=str(r.get("date", ""))[:10],
                shift="-",
                report_type="floor" if is_floor else "walls",
                total_length_m=0.0 if is_floor else _safe_float(r.get("meters_built"), 0.0),
                total_area_m2=_safe_float(r.get("meters_built"), 0.0) if is_floor else 0.0,
                note=note,
            )
        )

    return DashboardResponse(
        plan_id=plan_id,
        plan_name=str(
            plan_row.get("plan_name")
            or (proj.get("metadata", {}).get("plan_name") if proj else None)
            or plan_id
        ),
        total_planned_m=round(total_planned, 4),
        built_m=round(built_m, 4),
        percent_complete=round(percent, 2),
        remaining_m=round(remaining, 4),
        days_to_finish=round(days_to_finish, 1) if days_to_finish is not None else None,
        budget_limit_ils=round(_safe_float(financial.get("budget_limit"), 0.0), 2),
        current_cost_ils=round(_safe_float(financial.get("current_cost"), 0.0), 2),
        budget_variance_ils=round(_safe_float(financial.get("budget_variance"), 0.0), 2),
        reports_count=len(reports),
        average_daily_m=round(avg_daily, 4),
        max_daily_m=round(max_daily, 4),
        planned_walls_m=round(planned_walls_m, 4),
        built_walls_m=round(built_walls_m, 4),
        planned_floor_m2=round(planned_floor_m2, 4),
        built_floor_m2=round(built_floor_m2, 4),
        boq_progress=boq_progress,
        timeline=timeline,
        recent_reports=recent_reports,
    )


@app.get("/manager/invoices/{plan_id}/work-types", response_model=list[str])
async def manager_get_invoice_work_types(plan_id: str) -> list[str]:
    db_plan_id = _resolve_db_plan_id(plan_id)
    if db_plan_id is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found in database")
    types = get_all_work_types_for_plan(db_plan_id) or []
    return types if types else ["קירות", "ריצוף/חיפוי"]


@app.post("/manager/invoices/{plan_id}/calculate", response_model=InvoiceCalculationResponse)
async def manager_calculate_invoice(
    plan_id: str, request: InvoiceCalculateRequest
) -> InvoiceCalculationResponse:
    proj = PROJECTS.get(plan_id)
    start_dt = _parse_iso_date_or_400(request.start_date)
    end_dt = _parse_iso_date_or_400(request.end_date)
    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")

    db_plan_id = _resolve_db_plan_id(plan_id)
    if db_plan_id is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found in database")

    invoice_data = get_payment_invoice_data(
        db_plan_id,
        request.start_date,
        request.end_date,
        unit_prices=request.unit_prices,
    )
    if not invoice_data:
        raise HTTPException(status_code=404, detail="Invoice data unavailable")

    items = [
        InvoiceWorkItem(
            work_type=str(item.get("work_type", "")),
            quantity=round(_safe_float(item.get("quantity"), 0.0), 4),
            unit=str(item.get("unit", "")),
            unit_price=round(_safe_float(item.get("unit_price"), 0.0), 2),
            subtotal=round(_safe_float(item.get("subtotal"), 0.0), 2),
        )
        for item in (invoice_data.get("items") or [])
    ]

    summary_rows = get_progress_summary_by_date_range(
        db_plan_id, request.start_date, request.end_date
    ) or []
    summary = [
        InvoiceSummaryRow(
            work_type=str(item.get("work_type", "")),
            total_quantity=round(_safe_float(item.get("total_quantity"), 0.0), 4),
            unit=str(item.get("unit", "")),
            report_count=int(item.get("report_count", 0) or 0),
        )
        for item in summary_rows
    ]

    plan_obj = invoice_data.get("plan") or {}
    return InvoiceCalculationResponse(
        plan_id=plan_id,
        plan_name=str(
            plan_obj.get("plan_name")
            or (proj.get("metadata", {}).get("plan_name") if proj else None)
            or plan_id
        ),
        start_date=request.start_date,
        end_date=request.end_date,
        items=items,
        total_amount=round(_safe_float(invoice_data.get("total_amount"), 0.0), 2),
        vat=round(_safe_float(invoice_data.get("vat"), 0.0), 2),
        total_with_vat=round(_safe_float(invoice_data.get("total_with_vat"), 0.0), 2),
        summary=summary,
        contractor=request.contractor,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

# ── Worker report snapshot (plan image + worker drawings overlay) ─────────────
@app.get("/worker/reports/{plan_id}/snapshot")
async def worker_report_snapshot(
    plan_id: str,
    report_id: Optional[int] = None,
    width: int = 1200,
) -> Response:
    """
    Returns a PNG: the plan image with all worker SVG drawings rendered on top.
    If report_id is given, only renders that report's items.
    """
    proj = PROJECTS.get(plan_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found in memory")

    original = proj.get("original")
    if original is None:
        raise HTTPException(status_code=409, detail="PLAN_RESTART_LOST: נתוני התוכנית לא זמינים בשרת (ייתכן שהשרת עלה מחדש). אנא העלה את קובץ ה-PDF שוב.")

    # Get reports
    if report_id is not None:
        reports = [r for r in WORKER_REPORTS if r.get("plan_id") == plan_id and r.get("id") == report_id]
    else:
        reports = [r for r in WORKER_REPORTS if r.get("plan_id") == plan_id]

    # Start with original image
    img = original.copy()
    h, w = img.shape[:2]

    scale = proj.get("scale") or 200.0
    if scale <= 0:
        scale = 200.0

    # Draw each report's items onto the image
    color_map = {
        "walls": (34, 211, 238),   # cyan for walls
        "floor": (251, 146, 60),   # orange for floor
    }

    for report in reports:
        report_type = report.get("report_type", "walls")
        color = color_map.get(report_type, (34, 211, 238))
        items = report.get("items", [])

        for item in items:
            obj = item.get("raw_object", {})
            item_type = item.get("type", "line")

            # Items are stored in display-scale coords, convert back
            ds = float(item.get("display_scale", 1.0))
            if ds <= 0:
                ds = 1.0

            try:
                if item_type == "line":
                    x1 = int(float(obj.get("x1", 0)) / ds * (w / 1000))
                    y1 = int(float(obj.get("y1", 0)) / ds * (h / 1000))
                    x2 = int(float(obj.get("x2", 0)) / ds * (w / 1000))
                    y2 = int(float(obj.get("y2", 0)) / ds * (h / 1000))
                    # Clamp to image bounds
                    x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
                    x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness=4)
                    # Label measurement
                    measurement = item.get("measurement", 0)
                    unit = item.get("unit", "m")
                    label = f"{measurement:.2f}{unit}"
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.putText(img, label, (mx, max(20, my - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                elif item_type == "rect":
                    rx = int(float(obj.get("x", 0)) / ds * (w / 1000))
                    ry = int(float(obj.get("y", 0)) / ds * (h / 1000))
                    rw = int(float(obj.get("width", 0)) / ds * (w / 1000))
                    rh = int(float(obj.get("height", 0)) / ds * (h / 1000))
                    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, thickness=3)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
                    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
                    measurement = item.get("measurement", 0)
                    unit = item.get("unit", "m2")
                    label = f"{measurement:.2f}{unit}"
                    cv2.putText(img, label, (rx + rw//2 - 20, ry + rh//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                elif item_type == "path":
                    points_raw = obj.get("points", [])
                    pts = []
                    for p in points_raw:
                        px = int(float(p[0]) / ds * (w / 1000))
                        py = int(float(p[1]) / ds * (h / 1000))
                        pts.append((px, py))
                    for i in range(1, len(pts)):
                        cv2.line(img, pts[i-1], pts[i], color, thickness=3)
            except Exception:
                continue

    # Resize to requested width
    if w != width:
        aspect = h / w
        new_h = int(width * aspect)
        img = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)

    # Encode to PNG
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/worker/reports/{plan_id}/summary")
async def worker_reports_summary(plan_id: str) -> dict:
    """Returns aggregated summary of all worker reports for a plan."""
    reports = [r for r in WORKER_REPORTS if r.get("plan_id") == plan_id]
    proj = PROJECTS.get(plan_id)
    meta = (proj or {}).get("metadata", {})

    total_walls_m = sum(r.get("total_length_m", 0) for r in reports if r.get("report_type") == "walls")
    total_floor_m2 = sum(r.get("total_area_m2", 0) for r in reports if r.get("report_type") == "floor")

    # BOQ from planning
    planning_state = {}
    boq_items = []
    try:
        from pages.measure_utils import get_scale_with_fallback
        if proj:
            # Attempt to get planning BOQ
            boq_items = proj.get("planning_boq", [])
    except Exception:
        pass

    return {
        "plan_id": plan_id,
        "plan_name": meta.get("plan_name", plan_id),
        "total_reports": len(reports),
        "total_walls_m": round(total_walls_m, 3),
        "total_floor_m2": round(total_floor_m2, 3),
        "reports": [
            {
                "id": r.get("id"),
                "date": r.get("date"),
                "shift": r.get("shift"),
                "report_type": r.get("report_type"),
                "total_length_m": r.get("total_length_m", 0),
                "total_area_m2": r.get("total_area_m2", 0),
                "note": r.get("note", ""),
                "items_count": len(r.get("items", [])),
            }
            for r in sorted(reports, key=lambda r: r.get("date", ""))
        ],
        "boq": boq_items,
    }


# ─────────────────────────────────────────────────────────────────
# ARCHITECTURAL PDF EXTRACTOR — Vision + tool_use
# ─────────────────────────────────────────────────────────────────

@app.post("/api/extract-pdf")
async def extract_pdf_endpoint(file: UploadFile = File(...)):
    """
    חילוץ מידע מתוכנית אדריכלית PDF.
    שולח כל דף ל-Claude Vision עם tool_use לJSON מובטח.
    מחזיר תוצאות לפי דף + מיזוג אחוד.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="יש להעלות קובץ PDF בלבד")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="קובץ גדול מדי (מקסימום 50MB)")

    try:
        from brain import extract_from_architectural_pdf
    except ImportError:
        raise HTTPException(status_code=500, detail="מודול brain לא זמין")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = extract_from_architectural_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה בחילוץ: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return result

