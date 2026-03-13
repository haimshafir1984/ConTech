import React from "react";
import { useToast } from "../components/Toast";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  calculateDrawingScale,
  getDrawingCsvUrl,
  getDrawingJsonUrl,
  getDrawingData,
  getPlanReadiness,
  type PlanReadinessResponse,
  type DrawingDataScaleResult,
  type DrawingDataSummary
} from "../api/managerInsightsApi";

const PAPER_SIZES: Record<string, [number, number]> = {
  A4: [210, 297],
  A3: [297, 420],
  A2: [420, 594],
  A1: [594, 841],
  A0: [841, 1189]
};

export const DrawingDataPage: React.FC = () => {
  const toast = useToast();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [summary, setSummary] = React.useState<DrawingDataSummary | null>(null);
  const [scaleResult, setScaleResult] = React.useState<DrawingDataScaleResult | null>(null);
  const [paperType, setPaperType] = React.useState<"A4" | "A3" | "A2" | "A1" | "A0" | "custom">("A3");
  const [orientation, setOrientation] = React.useState<"portrait" | "landscape">("portrait");
  const [paperWidth, setPaperWidth] = React.useState(297);
  const [paperHeight, setPaperHeight] = React.useState(420);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [readiness, setReadiness] = React.useState<PlanReadinessResponse | null>(null);
  const [progressText, setProgressText] = React.useState<string | null>(null);

  React.useEffect(() => {
    void (async () => {
      try {
        const data = await listWorkshopPlans();
        setPlans(data);
        if (data.length > 0) {
          setSelectedPlanId(data[0].id);
        }
      } catch (e) {
        console.error(e);
        setError("שגיאה בטעינת תוכניות.");
      }
    })();
  }, []);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    void (async () => {
      try {
        setLoading(true);
        setProgressText("טוען נתוני תוכנית...");
        setError("");
        const [data, readinessData] = await Promise.all([
          getDrawingData(selectedPlanId),
          getPlanReadiness(selectedPlanId)
        ]);
        setSummary(data);
        setReadiness(readinessData);
      } catch (e) {
        console.error(e);
        setError("שגיאה בטעינת נתוני שרטוט.");
      } finally {
        setLoading(false);
        setProgressText(null);
      }
    })();
  }, [selectedPlanId]);

  React.useEffect(() => {
    if (paperType === "custom") return;
    const [w, h] = PAPER_SIZES[paperType];
    if (orientation === "portrait") {
      setPaperWidth(Math.min(w, h));
      setPaperHeight(Math.max(w, h));
    } else {
      setPaperWidth(Math.max(w, h));
      setPaperHeight(Math.min(w, h));
    }
  }, [paperType, orientation]);

  const calculateScale = async (applyToPlan: boolean) => {
    if (!selectedPlanId) return;
    try {
      setLoading(true);
      setProgressText("בודק מידות דף...");
      window.setTimeout(() => setProgressText("מחשב יחס פיקסל/מטר..."), 350);
      window.setTimeout(() => setProgressText("מעדכן חישובי חומרים..."), 750);
      const result = await calculateDrawingScale(selectedPlanId, {
        paper_width_mm: paperWidth,
        paper_height_mm: paperHeight,
        apply_to_plan: applyToPlan
      });
      setScaleResult(result);
      if (result.updated_summary) {
        setSummary(result.updated_summary);
      }
      const readinessData = await getPlanReadiness(selectedPlanId);
      setReadiness(readinessData);
      toast(applyToPlan ? "הסקייל עודכן בתוכנית" : "הסקייל חושב בהצלחה");
    } catch (e) {
      console.error(e);
      setError("שגיאה בחישוב הסקייל.");
    } finally {
      setLoading(false);
      setProgressText(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-[#31333F]">📄 נתונים מהשרטוט</h2>
        <p className="text-xs text-slate-500 mt-1">
          חישוב סקייל מדויק לפי גודל דף, בדיקת חומרים, וייצוא נתונים.
        </p>
      </div>

      {error && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">{error}</div>}
      {progressText && (
        <div className="text-sm text-blue-700 bg-blue-50 border border-blue-200 rounded-lg p-3">
          {progressText}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[320px,1fr] gap-5">
        <aside className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm space-y-4">
          <label className="text-xs block">
            תוכנית
            <select
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={selectedPlanId}
              onChange={(e) => setSelectedPlanId(e.target.value)}
            >
              {plans.length === 0 && <option value="">אין תוכניות</option>}
              {plans.map((plan) => (
                <option key={plan.id} value={plan.id}>
                  {plan.plan_name}
                </option>
              ))}
            </select>
          </label>

          <div className="space-y-2 text-xs">
            <label className="block">
              גודל דף
              <select
                className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
                value={paperType}
                onChange={(e) => setPaperType(e.target.value as typeof paperType)}
              >
                <option value="A4">A4</option>
                <option value="A3">A3</option>
                <option value="A2">A2</option>
                <option value="A1">A1</option>
                <option value="A0">A0</option>
                <option value="custom">מותאם אישית</option>
              </select>
            </label>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setOrientation("portrait")}
                className={`rounded border px-2 py-2 ${orientation === "portrait" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
              >
                לאורך
              </button>
              <button
                type="button"
                onClick={() => setOrientation("landscape")}
                className={`rounded border px-2 py-2 ${orientation === "landscape" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
              >
                לרוחב
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                רוחב (מ"מ)
                <input
                  type="number"
                  min={100}
                  value={paperWidth}
                  onChange={(e) => setPaperWidth(Number(e.target.value))}
                  className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
                />
              </label>
              <label className="block">
                גובה (מ"מ)
                <input
                  type="number"
                  min={100}
                  value={paperHeight}
                  onChange={(e) => setPaperHeight(Number(e.target.value))}
                  className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
                />
              </label>
            </div>
          </div>

          <button
            type="button"
            disabled={!selectedPlanId || loading}
            onClick={() => void calculateScale(false)}
            className="w-full bg-[#FF4B4B] text-white rounded py-2 text-sm font-semibold disabled:opacity-40"
          >
            🧮 חשב סקייל אמיתי
          </button>
          <button
            type="button"
            disabled={!scaleResult || loading}
            onClick={() => void calculateScale(true)}
            className="w-full bg-white border border-[#FF4B4B] text-[#FF4B4B] rounded py-2 text-sm font-semibold disabled:opacity-40"
          >
            ✅ עדכן סקייל לתוכנית
          </button>

          {selectedPlanId && (
            <div className="grid grid-cols-2 gap-2">
              <a
                href={getDrawingCsvUrl(selectedPlanId)}
                target="_blank"
                rel="noreferrer"
                className="block text-center w-full bg-white border border-slate-300 rounded py-2 text-sm"
              >
                📥 CSV
              </a>
              <a
                href={getDrawingJsonUrl(selectedPlanId)}
                target="_blank"
                rel="noreferrer"
                className="block text-center w-full bg-white border border-slate-300 rounded py-2 text-sm"
              >
                📥 JSON
              </a>
            </div>
          )}
        </aside>

        <main className="space-y-4">
          <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
            <h3 className="text-sm font-semibold mb-3">מידע בסיסי</h3>
            {!summary ? (
              <p className="text-sm text-slate-500">{loading ? "טוען..." : "בחר תוכנית."}</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">גודל תמונה</div>
                  <div className="font-semibold">
                    {summary.image_width_px} × {summary.image_height_px} px
                  </div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סקייל</div>
                  <div className="font-semibold">{summary.scale_px_per_meter.toFixed(2)} px/מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">קנה מידה טקסטואלי</div>
                  <div className="font-semibold">{summary.scale_text || "לא זוהה"}</div>
                </div>
              </div>
            )}
          </section>

          {readiness && readiness.issues.length > 0 && (
            <section className="bg-white border border-amber-200 rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-2 text-amber-800">בדיקת מוכנות חישוב</h3>
              <ul className="list-disc list-inside text-sm text-amber-700 space-y-1">
                {readiness.issues.map((issue, idx) => (
                  <li key={`${issue}-${idx}`}>{issue}</li>
                ))}
              </ul>
            </section>
          )}

          {scaleResult && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">תוצאת חישוב סקייל</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סקייל מחושב</div>
                  <div className="font-semibold">{scaleResult.calculated_scale_px_per_meter.toFixed(2)} px/מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סקייל נוכחי</div>
                  <div className="font-semibold">{scaleResult.current_scale_px_per_meter.toFixed(2)} px/מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סטייה</div>
                  <div className="font-semibold">{scaleResult.error_percent.toFixed(2)}%</div>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3">
                  אורך קירות עם סקייל נוכחי: <b>{scaleResult.length_with_current_scale_m.toFixed(2)} מ'</b>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  אורך קירות עם סקייל מחושב: <b>{scaleResult.length_with_calculated_scale_m.toFixed(2)} מ'</b>
                </div>
              </div>
            </section>
          )}

          {summary && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">פירוט חומרים</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">בטון</div>
                  <div className="font-semibold">{summary.concrete_length_m.toFixed(2)} מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">בלוקים</div>
                  <div className="font-semibold">{summary.blocks_length_m.toFixed(2)} מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סה"כ קירות</div>
                  <div className="font-semibold">{summary.total_wall_length_m.toFixed(2)} מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">ריצוף</div>
                  <div className="font-semibold">{summary.flooring_area_m2.toFixed(2)} מ"ר</div>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
};
