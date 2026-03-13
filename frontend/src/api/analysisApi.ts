import { apiClient } from "./client";

export interface PaperInfo {
  detected_size?: string | null;
  width_mm?: number | null;
  height_mm?: number | null;
  error_mm?: number | null;
  confidence?: number | null;
}

export interface MeasurementInfo {
  meters_per_pixel?: number | null;
  meters_per_pixel_x?: number | null;
  meters_per_pixel_y?: number | null;
  scale_denominator?: number | null;
  measurement_confidence?: number | null;
}

export interface MaterialsSummary {
  total_wall_length_m?: number | null;
  concrete_length_m?: number | null;
  blocks_length_m?: number | null;
  flooring_area_m2?: number | null;
}

export interface AnalysisResult {
  plan_name: string;
  meta: Record<string, unknown>;
  paper: PaperInfo;
  measurements: MeasurementInfo;
  materials: MaterialsSummary;
}

export async function analyzePdf(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<AnalysisResult>("/analyze/pdf", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });

  return response.data;
}

