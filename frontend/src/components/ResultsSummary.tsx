import React from "react";
import type { AnalysisResult } from "../api/analysisApi";

interface Props {
  result: AnalysisResult;
}

export const ResultsSummary: React.FC<Props> = ({ result }) => {
  const m = result.measurements;
  const mat = result.materials;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
      <div className="bg-slate-800/70 rounded-xl p-4">
        <p className="text-xs text-slate-400 mb-1">שם תוכנית</p>
        <p className="font-semibold truncate">{result.plan_name || "ללא שם"}</p>
      </div>
      <div className="bg-slate-800/70 rounded-xl p-4">
        <p className="text-xs text-slate-400 mb-1">אורך קירות כולל</p>
        <p className="font-semibold">
          {mat.total_wall_length_m ? mat.total_wall_length_m.toFixed(1) : "—"} מ&apos;
        </p>
      </div>
      <div className="bg-slate-800/70 rounded-xl p-4">
        <p className="text-xs text-slate-400 mb-1">בטון / בלוקים</p>
        <p className="font-semibold space-x-2 space-x-reverse">
          <span>
            בטון:{" "}
            {mat.concrete_length_m ? mat.concrete_length_m.toFixed(1) : "—"} מ&apos;
          </span>
          <span className="text-slate-300">
            | בלוקים:{" "}
            {mat.blocks_length_m ? mat.blocks_length_m.toFixed(1) : "—"} מ&apos;
          </span>
        </p>
      </div>
      <div className="bg-slate-800/70 rounded-xl p-4">
        <p className="text-xs text-slate-400 mb-1">אמינות מדידה</p>
        <p className="font-semibold">
          {m.measurement_confidence != null
            ? `${Math.round(m.measurement_confidence * 100)}%`
            : "—"}
        </p>
      </div>
    </div>
  );
};

