import React from "react";
import type { AnalysisResult } from "../api/analysisApi";

interface Props {
  result: AnalysisResult;
}

export const ResultsTable: React.FC<Props> = ({ result }) => {
  const rows: { label: string; value: string }[] = [];

  const m = result.measurements;
  const p = result.paper;
  const mat = result.materials;

  if (m.scale_denominator) {
    rows.push({ label: "קנה מידה מזוהה", value: `1:${m.scale_denominator}` });
  }

  if (p.detected_size) {
    rows.push({
      label: "גודל נייר",
      value: `${p.detected_size} (${p.width_mm?.toFixed(0)}×${p.height_mm?.toFixed(
        0
      )} מ״מ)`
    });
  }

  if (m.meters_per_pixel) {
    rows.push({
      label: "מטר לפיקסל (ממוצע)",
      value: m.meters_per_pixel.toExponential(3)
    });
  }

  if (mat.flooring_area_m2) {
    rows.push({
      label: "שטח ריצוף משוער",
      value: `${mat.flooring_area_m2.toFixed(1)} מ״ר`
    });
  }

  if (!rows.length) {
    return null;
  }

  return (
    <div className="mt-6 bg-slate-800/70 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700">
        <p className="font-semibold">פרטי ניתוח</p>
      </div>
      <div className="max-h-72 overflow-y-auto">
        <table className="min-w-full text-sm">
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-b border-slate-800">
                <td className="px-4 py-2 text-slate-300 w-1/2">{row.label}</td>
                <td className="px-4 py-2 text-slate-100">{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

