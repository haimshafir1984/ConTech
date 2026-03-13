import { apiClient } from "./client";

export interface PlanningCategory {
  key: string;
  type: string;
  subtype: string;
  params: Record<string, unknown>;
}

export interface PlanningItem {
  uid: string;
  type: string;
  category: string;
  length_m: number;
  length_m_effective?: number | null;
  area_m2: number;
  raw_object: Record<string, unknown>;
  analysis?: {
    openings?: Array<{
      gap_id: string;
      length_m: number;
      midpoint_canvas?: [number, number];
    }>;
    is_wall_like?: boolean | null;
    wall_overlap_ratio?: number;
    prompt_opening_question?: boolean;
    estimated_opening_length_m?: number;
    requires_wall_confirmation?: boolean;
    wall_confirmed?: boolean | null;
    resolved_opening_type?: "door" | "window" | "none" | null;
    deducted_length_m?: number;
  };
  timestamp: string;
}

export async function resolvePlanningOpening(
  planId: string,
  itemUid: string,
  payload: { opening_type: "door" | "window" | "none"; gap_id?: string }
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/items/${encodeURIComponent(itemUid)}/resolve-opening`,
    payload
  );
  return data;
}

export async function resolvePlanningWall(
  planId: string,
  itemUid: string,
  payload: { is_wall: boolean }
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/items/${encodeURIComponent(itemUid)}/resolve-wall`,
    payload
  );
  return data;
}

export interface WorkSection {
  uid: string;
  name: string;
  contractor: string;
  worker: string;
  color: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WorkSectionPayload {
  name?: string;
  contractor: string;
  worker: string;
  color?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export interface PlanningState {
  plan_id: string;
  plan_name: string;
  scale_px_per_meter: number;
  image_width: number;
  image_height: number;
  categories: Record<string, PlanningCategory>;
  items: PlanningItem[];
  boq: Record<string, unknown>;
  totals: {
    total_length_m: number;
    total_area_m2: number;
  };
  sections: WorkSection[];
  auto_segments?: AutoSegment[];
}

export async function getPlanningState(planId: string): Promise<PlanningState> {
  const { data } = await apiClient.get<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}`
  );
  return data;
}

export async function calibratePlanningScale(
  planId: string,
  payload: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    display_scale: number;
    real_length_m: number;
  }
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/calibrate`,
    payload
  );
  return data;
}

export async function upsertPlanningCategories(
  planId: string,
  categories: Record<string, PlanningCategory>
): Promise<PlanningState> {
  const { data } = await apiClient.put<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/categories`,
    { categories }
  );
  return data;
}

export async function addPlanningItem(
  planId: string,
  payload: {
    category_key: string;
    object_type: "line" | "rect" | "path";
    raw_object: Record<string, unknown>;
    display_scale: number;
  }
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/items`,
    payload
  );
  return data;
}

export async function deletePlanningItem(
  planId: string,
  itemUid: string
): Promise<PlanningState> {
  const { data } = await apiClient.delete<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/items/${encodeURIComponent(itemUid)}`
  );
  return data;
}

export async function finalizePlanning(planId: string): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/finalize`
  );
  return data;
}

// ── Text item ──────────────────────────────────────────────────────────────
export interface TextItemPayload {
  category_key: string;   // existing key or "__manual__"
  description: string;
  quantity: number;
  unit: string;           // מ׳ / מ"ר / יח׳ / ק"ג …
  note?: string;
}

export async function addTextItem(
  planId: string,
  payload: TextItemPayload
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/text-items`,
    payload
  );
  return data;
}

export async function deleteTextItem(
  planId: string,
  itemUid: string
): Promise<PlanningState> {
  // reuses the same delete endpoint as drawn items
  return deletePlanningItem(planId, itemUid);
}

// ── Zone item ──────────────────────────────────────────────────────────────
export interface ZoneItemPayload {
  category_key: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export async function addZoneItem(
  planId: string,
  payload: ZoneItemPayload
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/zone-items`,
    payload
  );
  return data;
}

// ── Auto-analyze ───────────────────────────────────────────────────────────
export interface AutoSegment {
  segment_id: string;
  label: string;
  suggested_type: string;
  suggested_subtype: string;
  confidence: number;       // 0–1
  length_m: number;
  area_m2: number;
  bbox: [number, number, number, number]; // x, y, w, h  natural coords
  element_class?: string;   // "wall" | "fixture" | "room"
  // enriched fields
  wall_type?: string;       // "exterior" | "interior" | "partition" | "column"
  material?: string;        // "בלוקים" | "בטון" | "גבס" | ...
  has_insulation?: boolean;
  fire_resistance?: string;
  room_name?: string;
  area_label?: string;
  category_color?: string;  // hex color
}

export interface VisionRoom {
  name: string;
  area_m2?: number;
  dimensions?: string;
  ceiling_height_m?: number;
  flooring?: string;
  notes?: string;
}

export interface VisionElement {
  type: string;
  id?: string;
  location?: string;
  notes?: string;
}

export interface AutoAnalyzeVisionData {
  rooms?: VisionRoom[];
  dimensions?: string[];
  dimensions_structured?: Array<{ raw: string; unit?: string; location?: string; type?: string }>;
  materials?: string[];
  materials_legend?: Array<{ symbol?: string; description?: string }>;
  elements?: VisionElement[];
  elevations?: Array<{ label?: string; value?: number; reference?: string }>;
  grid_lines?: { horizontal?: string[]; vertical?: string[] };
  systems?: Record<string, string>;
  total_area_m2?: number;
  // Title-block metadata
  plan_title?: string;
  project_name?: string;
  sheet_number?: string;
  sheet_name?: string;
  status?: string;
  architect?: string;
  date?: string;
  scale?: string;
  execution_notes?: string[];
}

export interface AutoAnalyzeResult {
  segments: AutoSegment[];
  vision_data?: AutoAnalyzeVisionData;
}

export async function autoAnalyzePlan(planId: string): Promise<AutoAnalyzeResult> {
  const { data } = await apiClient.post<AutoAnalyzeResult>(
    `/manager/planning/${encodeURIComponent(planId)}/auto-analyze`
  );
  return data;
}

// Confirm a list of auto-segments as planning items
export async function confirmAutoSegments(
  planId: string,
  segments: Array<{ segment_id: string; category_key: string }>
): Promise<PlanningState> {
  // Each confirmed segment becomes a zone-item using its bbox
  // We batch them sequentially and return the final state
  let state!: PlanningState;
  for (const seg of segments) {
    // The segment bbox comes from the caller who holds the AutoAnalyzeResult
    const { data } = await apiClient.post<PlanningState>(
      `/manager/planning/${encodeURIComponent(planId)}/confirm-auto-segment`,
      seg
    );
    state = data;
  }
  return state;
}

// ── Import vision elements ─────────────────────────────────────────────────
export async function importVisionItems(planId: string): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/import-vision-items`
  );
  return data;
}

// ── Work Sections (גזרות עבודה) ────────────────────────────────────────────
export async function addWorkSection(
  planId: string,
  payload: WorkSectionPayload
): Promise<PlanningState> {
  const { data } = await apiClient.post<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/sections`,
    payload
  );
  return data;
}

export async function deleteWorkSection(
  planId: string,
  sectionUid: string
): Promise<PlanningState> {
  const { data } = await apiClient.delete<PlanningState>(
    `/manager/planning/${encodeURIComponent(planId)}/sections/${encodeURIComponent(sectionUid)}`
  );
  return data;
}

export async function deleteAutoSegment(
  planId: string,
  segmentId: string
): Promise<void> {
  await apiClient.delete(
    `/manager/planning/${encodeURIComponent(planId)}/auto-segments/${encodeURIComponent(segmentId)}`
  );
}

export interface BoqRoom {
  name: string;
  area_m2: number;
}

export interface BoqWallRow {
  wall_type: string;
  wall_type_key?: string;
  color?: string;
  count: number;
  total_length_m: number;
  wall_area_m2: number;
}

export interface BoqSummary {
  rooms: BoqRoom[];
  walls: BoqWallRow[];
  door_count: number;
  window_count: number;
  stair_count: number;
  fixture_counts: Record<string, number>;
  total_rooms: number;
  total_area_m2: number;
  total_wall_length_m: number;
  total_wall_area_m2: number;
  plan_title?: string;
  scale?: string;
  floor_height_m?: number;
}

export async function fetchBoqSummary(planId: string): Promise<BoqSummary> {
  const { data } = await apiClient.get<BoqSummary>(
    `/manager/planning/${encodeURIComponent(planId)}/boq-summary`
  );
  return data;
}

