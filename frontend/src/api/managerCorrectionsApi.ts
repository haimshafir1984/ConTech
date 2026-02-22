import { apiClient } from "./client";

export interface CorrectionStrokePayload {
  points: Array<[number, number]>;
  width: number;
}

export interface ManualCorrectionsSummary {
  plan_id: string;
  has_corrections: boolean;
  auto_wall_length_m: number;
  corrected_wall_length_m: number;
  delta_wall_length_m: number;
}

export async function getCorrectionsSummary(planId: string): Promise<ManualCorrectionsSummary> {
  const { data } = await apiClient.get<ManualCorrectionsSummary>(
    `/manager/corrections/${encodeURIComponent(planId)}/summary`
  );
  return data;
}

export async function applyCorrection(
  planId: string,
  payload: {
    mode: "add" | "remove";
    display_width: number;
    display_height: number;
    strokes: CorrectionStrokePayload[];
  }
): Promise<ManualCorrectionsSummary> {
  const { data } = await apiClient.post<ManualCorrectionsSummary>(
    `/manager/corrections/${encodeURIComponent(planId)}/apply`,
    payload
  );
  return data;
}

export async function resetCorrections(planId: string): Promise<ManualCorrectionsSummary> {
  const { data } = await apiClient.post<ManualCorrectionsSummary>(
    `/manager/corrections/${encodeURIComponent(planId)}/reset`
  );
  return data;
}

export async function saveCorrections(planId: string): Promise<ManualCorrectionsSummary> {
  const { data } = await apiClient.post<ManualCorrectionsSummary>(
    `/manager/corrections/${encodeURIComponent(planId)}/save`
  );
  return data;
}

export function getCorrectionsOverlayUrl(
  planId: string,
  variant: "auto" | "corrected",
  version?: number
): string {
  const params = new URLSearchParams({ variant });
  if (typeof version === "number") params.set("v", String(version));
  return `${apiClient.defaults.baseURL}/manager/corrections/${encodeURIComponent(planId)}/overlay?${params.toString()}`;
}

