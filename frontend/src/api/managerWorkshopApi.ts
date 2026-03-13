import { apiClient } from "./client";
import type { MaterialsSummary } from "./analysisApi";

// ── Structured blueprint extraction types ─────────────────────────────────────

export interface BlueprintRoom {
  name: string;
  area_m2?: number | null;
  dimensions?: string | null;
  ceiling_height_m?: number | null;
  elevation_floor_m?: number | null;
  elevation_slab_m?: number | null;
  flooring?: string | null;
  notes?: string | null;
  position_x_pct?: number | null;
  position_y_pct?: number | null;
}

export interface BlueprintDimension {
  raw: string;
  unit?: "mm" | "cm" | "m";
  location?: string | null;
  type?: "overall" | "partial" | "height" | "stair" | "other";
}

export interface BlueprintElevation {
  label: string;
  value?: number | null;
  reference?: string | null;
}

export interface BlueprintMaterialLegend {
  symbol?: string | null;
  description: string;
  fire_rating?: string | null;
}

export interface BlueprintElement {
  type: string;          // door | window | stair | elevator | sink | toilet | shower | boiler | other
  id?: string | null;
  location?: string | null;
  notes?: string | null;
}

export interface BlueprintGridLines {
  horizontal?: string[];
  vertical?: string[];
}

export interface BlueprintSystems {
  waterproofing?: string | null;
  drainage_slopes_pct?: string[];
  hvac_notes?: string | null;
  fire_suppression?: string | null;
  accessibility?: string | null;
}

/** Typed view over the free-form meta dict returned from the server. */
export interface PlanMeta extends Record<string, unknown> {
  // Document metadata
  plan_title?: string | null;
  project_name?: string | null;
  plan_type?: string | null;
  floor_level?: string | null;
  drawing_number?: string | null;
  sheet_number?: string | null;
  sheet_name?: string | null;
  status?: "לאישור" | "למכרז" | "לביצוע" | "טיוטה" | "לא ידוע" | null;
  revision?: string | null;
  date?: string | null;
  architect?: string | null;
  drawn_by?: string | null;
  designed_by?: string | null;
  approved_by?: string | null;
  project_address?: string | null;
  scale_text?: string | null;
  scale_denominator?: number | null;
  // Vision extractions
  llm_rooms?: BlueprintRoom[];
  vision_dimensions?: string[];
  vision_dimensions_structured?: BlueprintDimension[];
  vision_elevations?: BlueprintElevation[];
  vision_materials?: string[];
  vision_materials_legend?: BlueprintMaterialLegend[];
  vision_elements?: BlueprintElement[];
  vision_grid_lines?: BlueprintGridLines;
  vision_systems?: BlueprintSystems;
  vision_total_area_m2?: number | null;
  vision_pages_processed?: number;
}

export interface PlanSummary {
  id: string;
  filename: string;
  plan_name: string;
  scale_px_per_meter?: number | null;
  total_wall_length_m?: number | null;
  concrete_length_m?: number | null;
  blocks_length_m?: number | null;
  flooring_area_m2?: number | null;
}

export interface PlanDetail {
  summary: PlanSummary;
  meta: PlanMeta;
}

export interface PlanListResponse {
  plans: PlanSummary[];
}

export async function uploadWorkshopPlan(file: File): Promise<PlanDetail> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await apiClient.post<PlanDetail>(
    "/manager/workshop/upload",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 300000
    }
  );
  return data;
}

export async function listWorkshopPlans(): Promise<PlanSummary[]> {
  const { data } = await apiClient.get<PlanListResponse>("/manager/workshop/plans");
  return data.plans;
}

export async function getWorkshopPlan(planId: string): Promise<PlanDetail> {
  const { data } = await apiClient.get<PlanDetail>(
    `/manager/workshop/plans/${encodeURIComponent(planId)}`
  );
  return data;
}

export async function updateWorkshopPlanScale(payload: {
  plan_id: string;
  scale_text: string;
  plan_name?: string;
}): Promise<PlanDetail> {
  const { data } = await apiClient.patch<PlanDetail>(
    `/manager/workshop/plans/${encodeURIComponent(payload.plan_id)}/scale`,
    {
      scale_text: payload.scale_text,
      plan_name: payload.plan_name
    }
  );
  return data;
}

export async function listDatabasePlans(): Promise<PlanSummary[]> {
  const { data } = await apiClient.get<PlanListResponse>("/manager/database/plans");
  return data.plans;
}

export async function clearAllWorkshopPlans(): Promise<void> {
  await apiClient.delete("/manager/workshop/plans");
}

export function getWorkshopOverlayUrl(
  planId: string,
  options?: { show_flooring?: boolean; show_room_numbers?: boolean; highlight_walls?: boolean; version?: number }
): string {
  const params = new URLSearchParams();
  if (options?.show_flooring !== undefined) params.set("show_flooring", String(options.show_flooring));
  if (options?.show_room_numbers !== undefined) params.set("show_room_numbers", String(options.show_room_numbers));
  if (options?.highlight_walls !== undefined) params.set("highlight_walls", String(options.highlight_walls));
  if (options?.version !== undefined) params.set("v", String(options.version));
  const query = params.toString();
  const base = `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(planId)}/overlay`;
  return query ? `${base}?${query}` : base;
}

