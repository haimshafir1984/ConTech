import { apiClient } from "./client";

export interface DrawingDataSummary {
  plan_id: string;
  plan_name: string;
  image_width_px: number;
  image_height_px: number;
  image_size?: { width_px: number; height_px: number };
  scale_px_per_meter: number;
  scale_text?: string | null;
  scale?: { px_per_meter: number; scale_text?: string | null };
  total_wall_length_m: number;
  concrete_length_m: number;
  blocks_length_m: number;
  flooring_area_m2: number;
  materials?: { concrete_m: number; blocks_m: number; total_m: number };
  metadata: Record<string, unknown>;
}

export interface DrawingDataScaleResult {
  calculated_scale_px_per_meter: number;
  current_scale_px_per_meter: number;
  scale_diff: number;
  error_percent: number;
  length_with_current_scale_m: number;
  length_with_calculated_scale_m: number;
  length_diff_m: number;
  applied: boolean;
  updated_summary?: DrawingDataSummary | null;
}

export interface FloorRoomRow {
  room_id: number;
  matched_name?: string | null;
  area_px?: number | null;
  area_m2?: number | null;
  area_text_m2?: number | null;
  diff_m2?: number | null;
  perimeter_px?: number | null;
  perimeter_m?: number | null;
  baseboard_m?: number | null;
  match_confidence?: number | null;
}

export interface FloorAnalysisResponse {
  success: boolean;
  totals: {
    num_rooms: number;
    total_area_m2?: number | null;
    total_perimeter_m?: number | null;
    total_baseboard_m?: number | null;
  };
  rooms: FloorRoomRow[];
  limitations: string[];
  has_overlay: boolean;
  overlay_image_url?: string | null;
  segmentation_method?: string | null;
  min_area_px?: number | null;
  debug_json_safe?: Record<string, unknown>;
}

export interface PlanReadinessResponse {
  plan_id: string;
  plan_name: string;
  has_original: boolean;
  has_thick_walls: boolean;
  has_flooring_mask: boolean;
  wall_pixels: number;
  flooring_pixels: number;
  has_scale_px_per_meter: boolean;
  has_meters_per_pixel: boolean;
  has_llm_rooms: boolean;
  issues: string[];
}

export interface DashboardRecentReport {
  id: string;
  date: string;
  shift: string;
  report_type: string;
  total_length_m: number;
  total_area_m2: number;
  note?: string;
}

export interface DashboardResponse {
  plan_id: string;
  plan_name: string;
  total_planned_m: number;
  built_m: number;
  percent_complete: number;
  remaining_m: number;
  days_to_finish?: number | null;
  budget_limit_ils: number;
  current_cost_ils: number;
  budget_variance_ils: number;
  reports_count: number;
  average_daily_m: number;
  max_daily_m: number;
  planned_walls_m: number;
  built_walls_m: number;
  planned_floor_m2: number;
  built_floor_m2: number;
  boq_progress: Array<{
    label: string;
    planned_qty: number;
    built_qty: number;
    remaining_qty: number;
    unit: string;
    progress_percent: number;
  }>;
  timeline: Array<{ date: string; quantity_m: number }>;
  recent_reports: DashboardRecentReport[];
}

export interface InvoiceSummaryRow {
  work_type: string;
  total_quantity: number;
  unit: string;
  report_count: number;
}

export interface InvoiceCalculationResponse {
  plan_id: string;
  plan_name: string;
  start_date: string;
  end_date: string;
  items: Array<{
    work_type: string;
    quantity: number;
    unit: string;
    unit_price: number;
    subtotal: number;
  }>;
  total_amount: number;
  vat: number;
  total_with_vat: number;
  summary: InvoiceSummaryRow[];
  contractor?: {
    name: string;
    company?: string;
    vat_id: string;
    address?: string;
  } | null;
}

export async function getDrawingData(planId: string): Promise<DrawingDataSummary> {
  const { data } = await apiClient.get<DrawingDataSummary>(
    `/manager/drawing-data/${encodeURIComponent(planId)}`
  );
  return data;
}

export async function calculateDrawingScale(
  planId: string,
  payload: { paper_width_mm: number; paper_height_mm: number; apply_to_plan?: boolean }
): Promise<DrawingDataScaleResult> {
  const { data } = await apiClient.post<DrawingDataScaleResult>(
    `/manager/drawing-data/${encodeURIComponent(planId)}/scale`,
    payload
  );
  return data;
}

export function getDrawingCsvUrl(planId: string): string {
  return `${apiClient.defaults.baseURL}/manager/drawing-data/${encodeURIComponent(planId)}/export/csv`;
}

export function getDrawingJsonUrl(planId: string): string {
  return `${apiClient.defaults.baseURL}/manager/drawing-data/${encodeURIComponent(planId)}/export/json`;
}

export async function runAreaAnalysis(
  planId: string,
  payload: { segmentation_method: "watershed" | "cc"; auto_min_area: boolean; min_area_px: number }
): Promise<FloorAnalysisResponse> {
  const { data } = await apiClient.post<FloorAnalysisResponse>(
    `/manager/area-analysis/${encodeURIComponent(planId)}/run`,
    payload,
    { timeout: 300000 }
  );
  return data;
}

export async function getAreaAnalysis(planId: string): Promise<FloorAnalysisResponse> {
  const { data } = await apiClient.get<FloorAnalysisResponse>(
    `/manager/area-analysis/${encodeURIComponent(planId)}`
  );
  return data;
}

export function getAreaOverlayUrl(planId: string): string {
  return `${apiClient.defaults.baseURL}/manager/area-analysis/${encodeURIComponent(planId)}/overlay`;
}

export async function getPlanReadiness(planId: string): Promise<PlanReadinessResponse> {
  const { data } = await apiClient.get<PlanReadinessResponse>(
    `/manager/plans/${encodeURIComponent(planId)}/readiness`
  );
  return data;
}

export async function getDashboard(planId: string): Promise<DashboardResponse> {
  const { data } = await apiClient.get<DashboardResponse>(
    `/manager/dashboard/${encodeURIComponent(planId)}`
  );
  return data;
}

export async function getInvoiceWorkTypes(planId: string): Promise<string[]> {
  const { data } = await apiClient.get<string[]>(
    `/manager/invoices/${encodeURIComponent(planId)}/work-types`
  );
  return data;
}

export async function calculateInvoice(
  planId: string,
  payload: {
    start_date: string;
    end_date: string;
    unit_prices: Record<string, number>;
    contractor?: { name: string; company?: string; vat_id: string; address?: string };
  }
): Promise<InvoiceCalculationResponse> {
  const { data } = await apiClient.post<InvoiceCalculationResponse>(
    `/manager/invoices/${encodeURIComponent(planId)}/calculate`,
    payload
  );
  return data;
}


// ── Worker report snapshot + summary ─────────────────────────────────────────
export function getWorkerReportSnapshotUrl(planId: string, reportId?: number): string {
  const base = apiClient.defaults.baseURL ?? "";
  const params = reportId != null ? `?report_id=${reportId}` : "";
  return `${base}/worker/reports/${encodeURIComponent(planId)}/snapshot${params}`;
}

export interface WorkerReportSummaryItem {
  id: number;
  date: string;
  shift: string;
  report_type: "walls" | "floor";
  total_length_m: number;
  total_area_m2: number;
  note: string;
  items_count: number;
}

export interface WorkerReportsSummary {
  plan_id: string;
  plan_name: string;
  total_reports: number;
  total_walls_m: number;
  total_floor_m2: number;
  reports: WorkerReportSummaryItem[];
  boq: unknown[];
}

export async function getWorkerReportsSummary(planId: string): Promise<WorkerReportsSummary> {
  const res = await apiClient.get<WorkerReportsSummary>(
    `/worker/reports/${encodeURIComponent(planId)}/summary`
  );
  return res.data;
}
