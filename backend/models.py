from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class PaperSizeInfo(BaseModel):
    detected_size: Optional[str] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    error_mm: Optional[float] = None
    confidence: Optional[float] = None


class MeasurementInfo(BaseModel):
    meters_per_pixel: Optional[float] = None
    meters_per_pixel_x: Optional[float] = None
    meters_per_pixel_y: Optional[float] = None
    scale_denominator: Optional[int] = None
    measurement_confidence: Optional[float] = None


class WallMaterialsSummary(BaseModel):
    total_wall_length_m: Optional[float] = None
    concrete_length_m: Optional[float] = None
    blocks_length_m: Optional[float] = None
    flooring_area_m2: Optional[float] = None


class AnalysisResult(BaseModel):
    plan_name: str
    meta: Dict[str, Any]
    paper: PaperSizeInfo
    measurements: MeasurementInfo
    materials: WallMaterialsSummary


class HealthResponse(BaseModel):
    status: str


class PlanSummary(BaseModel):
    """
    תקציר תוכנית עבור מסך 'סדנת עבודה'
    """

    id: str
    filename: str
    plan_name: str
    scale_px_per_meter: Optional[float] = None
    total_wall_length_m: Optional[float] = None
    concrete_length_m: Optional[float] = None
    blocks_length_m: Optional[float] = None
    flooring_area_m2: Optional[float] = None


class PlanDetail(BaseModel):
    """
    פרטי תוכנית מלאה (מטא-דאטה + תקציר)
    """

    summary: PlanSummary
    meta: Dict[str, Any]


class PlanListResponse(BaseModel):
    plans: list[PlanSummary]


class WorkshopScaleUpdateRequest(BaseModel):
    scale_text: str
    plan_name: Optional[str] = None


class PlanningCategory(BaseModel):
    key: str
    type: str
    subtype: str
    params: Dict[str, Any] = {}


class PlanningItem(BaseModel):
    uid: str
    type: str
    category: str
    length_m: float = 0.0
    length_m_effective: Optional[float] = None
    area_m2: float = 0.0
    raw_object: Dict[str, Any]
    analysis: Dict[str, Any] = {}
    timestamp: str


class WorkSection(BaseModel):
    """גזרת עבודה — תחום גיאוגרפי עם קבלן ועובד אחראי."""
    uid: str
    name: str                      # שם הגזרה (אופציונלי, נוצר אוטומטית)
    contractor: str                # שם קבלן מבצע
    worker: str                    # שם עובד אחראי
    color: str = "#6366f1"        # צבע (hex)
    # rect in natural image coords
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class WorkSectionCreateRequest(BaseModel):
    name: str = ""
    contractor: str
    worker: str
    color: str = "#6366f1"
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


# ── Auto-analyze response ──────────────────────────────────────────────────
class AutoAnalyzeSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segment_id: str
    label: str            # "קיר צפוני", "כיור", …
    suggested_type: str   # "קירות" / "אביזר" / "לא ידוע"
    suggested_subtype: str
    confidence: float     # 0.0 – 1.0
    length_m: float
    area_m2: float
    bbox: List[float]     # [x, y, w, h] natural coords
    element_class: str = "wall"  # "wall" | "fixture" | "room"
    # שדות עשירים
    wall_type: str = "interior"
    material: str = "לא_ידוע"
    has_insulation: bool = False
    fire_resistance: Optional[str] = None
    room_name: Optional[str] = None
    area_label: Optional[str] = None
    category_color: Optional[str] = None
    # Phase 2: confidence engine fields
    review_status: Optional[str] = None   # "auto" | "medium" | "review"
    flags: Optional[List[str]] = None
    drawing_source_index: Optional[int] = None
    legality_score: Optional[float] = None
    gate_decision: Optional[str] = None
    rejection_reason: Optional[str] = None


class PlanningState(BaseModel):
    plan_id: str
    plan_name: str
    scale_px_per_meter: float
    image_width: int
    image_height: int
    categories: Dict[str, PlanningCategory]
    items: list[PlanningItem]
    boq: Dict[str, Any]
    totals: Dict[str, float]
    sections: list[WorkSection] = []
    auto_segments: list[AutoAnalyzeSegment] = []


class PlanningCalibrateRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    display_scale: float = 1.0
    real_length_m: float


class PlanningCategoryUpsertRequest(BaseModel):
    categories: Dict[str, PlanningCategory]


class PlanningAddItemRequest(BaseModel):
    category_key: str
    object_type: str
    raw_object: Dict[str, Any]
    display_scale: float = 1.0


# ── Text item (free-text BOQ row, no drawing) ──────────────────────────────
class PlanningTextItemRequest(BaseModel):
    category_key: str          # existing or "__manual__"
    description: str           # תיאור חופשי
    quantity: float            # כמות
    unit: str = "מ׳"           # מ׳ / מ"ר / יח׳ / ק"ג / וכו׳
    note: str = ""             # הערה אופציונלית


# ── Zone item (paint a rect, measure thick_walls inside) ───────────────────
class PlanningZoneRequest(BaseModel):
    category_key: str
    # rect in NATURAL image coords (display_scale already applied by client)
    x: float
    y: float
    width: float
    height: float


class AutoAnalyzeVisionData(BaseModel):
    """Vision extraction data from Claude Vision (extracted during PDF upload)."""
    rooms: Optional[list[dict]] = None
    dimensions: Optional[list[str]] = None
    dimensions_structured: Optional[list[dict]] = None
    materials: Optional[list[str]] = None
    materials_legend: Optional[list[dict]] = None
    elements: Optional[list[dict]] = None
    elevations: Optional[list[dict]] = None
    grid_lines: Optional[dict] = None
    systems: Optional[dict] = None
    total_area_m2: Optional[float] = None
    # Title-block metadata
    plan_title: Optional[str] = None
    project_name: Optional[str] = None
    sheet_number: Optional[str] = None
    sheet_name: Optional[str] = None
    status: Optional[str] = None
    architect: Optional[str] = None
    date: Optional[str] = None
    scale: Optional[str] = None
    execution_notes: Optional[list[str]] = None
    walls: Optional[list[dict]] = None
    openings: Optional[list[dict]] = None
    stairs: Optional[list[dict]] = None
    total_wall_length_m: Optional[float] = None


class AutoAnalyzeResponse(BaseModel):
    segments: List[AutoAnalyzeSegment]
    vision_data: Optional[AutoAnalyzeVisionData] = None
    legend_items: Optional[List[dict]] = None


class ConfirmAutoSegmentRequest(BaseModel):
    segment_id: str
    category_key: str
    bbox: list[float]  # [x, y, w, h] natural coords


class PlanningResolveOpeningRequest(BaseModel):
    opening_type: str  # door | window | none
    gap_id: Optional[str] = None


class PlanningResolveWallRequest(BaseModel):
    is_wall: bool


class WorkerMeasuredItemRequest(BaseModel):
    plan_id: str
    object_type: str
    raw_object: Dict[str, Any]
    display_scale: float = 1.0
    report_type: str = "walls"


class WorkerMeasuredItem(BaseModel):
    uid: str
    type: str
    measurement: float
    unit: str
    raw_object: Dict[str, Any]


class WorkerReportCreateRequest(BaseModel):
    plan_id: str
    date: str
    shift: str
    report_type: str
    draw_mode: str
    items: list[WorkerMeasuredItem]
    note: Optional[str] = ""


class WorkerReport(BaseModel):
    id: str
    plan_id: str
    plan_name: str
    date: str
    shift: str
    report_type: str
    draw_mode: str
    items: list[WorkerMeasuredItem]
    total_length_m: float
    total_area_m2: float
    note: Optional[str] = ""


class DrawingDataSummary(BaseModel):
    plan_id: str
    plan_name: str
    image_width_px: int
    image_height_px: int
    image_size: Dict[str, int] = {}
    scale_px_per_meter: float
    scale_text: Optional[str] = None
    scale: Dict[str, Any] = {}
    total_wall_length_m: float
    concrete_length_m: float
    blocks_length_m: float
    flooring_area_m2: float
    materials: Dict[str, float] = {}
    metadata: Dict[str, Any]


class DrawingDataScaleRequest(BaseModel):
    paper_width_mm: float
    paper_height_mm: float
    apply_to_plan: bool = False


class DrawingDataScaleResult(BaseModel):
    calculated_scale_px_per_meter: float
    current_scale_px_per_meter: float
    scale_diff: float
    error_percent: float
    length_with_current_scale_m: float
    length_with_calculated_scale_m: float
    length_diff_m: float
    applied: bool = False
    updated_summary: Optional[DrawingDataSummary] = None


class FloorAnalysisRunRequest(BaseModel):
    segmentation_method: str = "watershed"
    auto_min_area: bool = True
    min_area_px: int = 500


class FloorRoomRow(BaseModel):
    room_id: int
    matched_name: Optional[str] = None
    area_px: Optional[float] = None
    area_m2: Optional[float] = None
    area_text_m2: Optional[float] = None
    diff_m2: Optional[float] = None
    perimeter_px: Optional[float] = None
    perimeter_m: Optional[float] = None
    baseboard_m: Optional[float] = None
    match_confidence: Optional[float] = None
    center: Optional[list[int]] = None
    bbox: Optional[list[int]] = None


class FloorAnalysisTotals(BaseModel):
    num_rooms: int = 0
    total_area_m2: Optional[float] = None
    total_perimeter_m: Optional[float] = None
    total_baseboard_m: Optional[float] = None


class FloorAnalysisResponse(BaseModel):
    success: bool
    totals: FloorAnalysisTotals
    rooms: list[FloorRoomRow]
    limitations: list[str]
    has_overlay: bool = False
    overlay_image_url: Optional[str] = None
    segmentation_method: Optional[str] = None
    min_area_px: Optional[int] = None
    debug_json_safe: Dict[str, Any] = {}


class PlanReadinessResponse(BaseModel):
    plan_id: str
    plan_name: str
    has_original: bool
    has_thick_walls: bool
    has_flooring_mask: bool
    wall_pixels: int
    flooring_pixels: int
    has_scale_px_per_meter: bool
    has_meters_per_pixel: bool
    has_llm_rooms: bool
    issues: list[str]


class CorrectionStroke(BaseModel):
    points: list[list[float]]
    width: float = 8.0


class ManualCorrectionApplyRequest(BaseModel):
    mode: str  # add | remove
    display_width: int
    display_height: int
    strokes: list[CorrectionStroke]


class ManualCorrectionsSummary(BaseModel):
    plan_id: str
    has_corrections: bool
    auto_wall_length_m: float
    corrected_wall_length_m: float
    delta_wall_length_m: float


class DashboardTimelinePoint(BaseModel):
    date: str
    quantity_m: float


class DashboardRecentReport(BaseModel):
    id: str
    date: str
    shift: str
    report_type: str
    total_length_m: float
    total_area_m2: float
    note: Optional[str] = ""


class DashboardBoqProgressRow(BaseModel):
    label: str
    planned_qty: float
    built_qty: float
    remaining_qty: float
    unit: str
    progress_percent: float


class DashboardResponse(BaseModel):
    plan_id: str
    plan_name: str
    total_planned_m: float
    built_m: float
    percent_complete: float
    remaining_m: float
    days_to_finish: Optional[float] = None
    budget_limit_ils: float
    current_cost_ils: float
    budget_variance_ils: float
    reports_count: int
    average_daily_m: float
    max_daily_m: float
    planned_walls_m: float
    built_walls_m: float
    planned_floor_m2: float
    built_floor_m2: float
    boq_progress: list[DashboardBoqProgressRow]
    timeline: list[DashboardTimelinePoint]
    recent_reports: list[DashboardRecentReport]


class InvoiceSummaryRow(BaseModel):
    work_type: str
    total_quantity: float
    unit: str
    report_count: int


class InvoiceWorkItem(BaseModel):
    work_type: str
    quantity: float
    unit: str
    unit_price: float
    subtotal: float


class InvoiceContractorInfo(BaseModel):
    name: str
    company: Optional[str] = ""
    vat_id: str
    address: Optional[str] = ""


class InvoiceCalculateRequest(BaseModel):
    start_date: str
    end_date: str
    unit_prices: Dict[str, float]
    contractor: Optional[InvoiceContractorInfo] = None


class InvoiceCalculationResponse(BaseModel):
    plan_id: str
    plan_name: str
    start_date: str
    end_date: str
    items: list[InvoiceWorkItem]
    total_amount: float
    vat: float
    total_with_vat: float
    summary: list[InvoiceSummaryRow]
    contractor: Optional[InvoiceContractorInfo] = None

