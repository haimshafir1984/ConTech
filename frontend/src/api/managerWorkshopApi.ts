import { apiClient } from "./client";
import type { MaterialsSummary } from "./analysisApi";

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
  meta: Record<string, unknown>;
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

