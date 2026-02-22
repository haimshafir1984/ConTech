import { apiClient } from "./client";

export interface WorkerMeasuredItem {
  uid: string;
  type: string;
  measurement: number;
  unit: string;
  raw_object: Record<string, unknown>;
}

export interface WorkerReport {
  id: string;
  plan_id: string;
  plan_name: string;
  date: string;
  shift: string;
  report_type: string;
  draw_mode: string;
  items: WorkerMeasuredItem[];
  total_length_m: number;
  total_area_m2: number;
  note?: string;
}

export async function measureWorkerItem(payload: {
  plan_id: string;
  object_type: "line" | "rect" | "path";
  raw_object: Record<string, unknown>;
  display_scale: number;
  report_type: "walls" | "floor";
}): Promise<WorkerMeasuredItem> {
  const { data } = await apiClient.post<WorkerMeasuredItem>("/worker/measure-item", payload);
  return data;
}

export async function createWorkerReport(payload: {
  plan_id: string;
  date: string;
  shift: string;
  report_type: "walls" | "floor";
  draw_mode: "line" | "rect" | "path";
  items: WorkerMeasuredItem[];
  note?: string;
}): Promise<WorkerReport> {
  const { data } = await apiClient.post<WorkerReport>("/worker/reports", payload);
  return data;
}

export async function listWorkerReports(planId: string): Promise<WorkerReport[]> {
  const { data } = await apiClient.get<WorkerReport[]>(`/worker/reports/${encodeURIComponent(planId)}`);
  return data;
}

