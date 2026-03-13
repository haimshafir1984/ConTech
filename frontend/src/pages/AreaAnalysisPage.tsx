import React from "react";
import { useToast } from "../components/Toast";
import axios from "axios";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  apiClient
} from "../api/client";
import {
  getAreaAnalysis,
  getAreaOverlayUrl,
  getPlanReadiness,
  runAreaAnalysis,
  type FloorAnalysisResponse,
  type PlanReadinessResponse
} from "../api/managerInsightsApi";

export const AreaAnalysisPage: React.FC = () => {
  const toast = useToast();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [result, setResult] = React.useState<FloorAnalysisResponse | null>(null);
  const [segmentationMethod, setSegmentationMethod] = React.useState<"watershed" | "cc">("watershed");
  const [autoMinArea, setAutoMinArea] = React.useState(true);
  const [minAreaPx, setMinAreaPx] = React.useState(500);
  const [overlayVersion, setOverlayVersion] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
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
        const [existing, readinessData] = await Promise.all([
          getAreaAnalysis(selectedPlanId),
          getPlanReadiness(selectedPlanId)
        ]);
        setResult(existing);
        setReadiness(readinessData);
      } catch (e) {
        console.error(e);
      }
    })();
  }, [selectedPlanId]);

  const runAnalysis = async () => {
    if (!selectedPlanId) return;
    try {
      setLoading(true);
      setError("");
      setProgressText("מאתר קירות וחדרים...");
      window.setTimeout(() => setProgressText("מריץ סגמנטציה של רצפה..."), 350);
      window.setTimeout(() => setProgressText("מחשב שטחים/היקפים..."), 750);
      const data = await runAreaAnalysis(selectedPlanId, {
        segmentation_method: segmentationMethod,
        auto_min_area: autoMinArea,
        min_area_px: minAreaPx
      });
      setResult(data);
      if (!data.success && data.limitations.length > 0) {
        setError(data.limitations[0]);
      }
      const readinessData = await getPlanReadiness(selectedPlanId);
      setReadiness(readinessData);
      setOverlayVersion((v) => v + 1);
      toast("הניתוח הושלם בהצלחה");
    } catch (e) {
      console.error(e);
      const detail = axios.isAxiosError(e)
        ? (e.response?.data?.detail as string | undefined) || e.message
        : e instanceof Error
          ? e.message
          : "שגיאה לא ידועה";
      setError(`שגיאה בהרצת ניתוח שטחים: ${detail}`);
    } finally {
      setLoading(false);
      setProgressText(null);
    }
  };

  const overlayUrl = React.useMemo(() => {
    if (!selectedPlanId) return "";
    const serverRelative = result?.overlay_image_url;
    const base = serverRelative
      ? `${apiClient.defaults.baseURL}${serverRelative}`
      : getAreaOverlayUrl(selectedPlanId);
    return `${base}?v=${overlayVersion}`;
  }, [selectedPlanId, result?.overlay_image_url, overlayVersion]);

  return (
    <div className="space-y-4">
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-[#31333F]">📐 ניתוח שטחי רצפה והיקפים</h2>
        <p className="text-xs text-slate-500 mt-1">
          חישוב אוטומטי של חדרים, שטח, היקף ופאנלים לפי מסכת קירות.
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

          <label className="text-xs block">
            שיטת סגמנטציה
            <select
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={segmentationMethod}
              onChange={(e) => setSegmentationMethod(e.target.value as "watershed" | "cc")}
            >
              <option value="watershed">watershed (מומלץ)</option>
              <option value="cc">connected components</option>
            </select>
          </label>

          <label className="text-xs inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoMinArea}
              onChange={(e) => setAutoMinArea(e.target.checked)}
            />
            סף מינימום אוטומטי
          </label>

          <label className="text-xs block">
            סף מינימום חדר (px)
            <input
              type="number"
              min={100}
              step={100}
              disabled={autoMinArea}
              value={minAreaPx}
              onChange={(e) => setMinAreaPx(Number(e.target.value))}
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2 disabled:bg-slate-100"
            />
          </label>

          <button
            type="button"
            onClick={() => void runAnalysis()}
            disabled={!selectedPlanId || loading}
            className="w-full bg-[#FF4B4B] text-white rounded py-2 text-sm font-semibold disabled:opacity-40"
          >
            {loading ? "מנתח..." : "🔍 חשב שטחים והיקפים"}
          </button>
        </aside>

        <main className="space-y-4">
          <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
            <h3 className="text-sm font-semibold mb-3">תקציר</h3>
            {!result ? (
              <p className="text-sm text-slate-500">אין תוצאות עדיין.</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">מספר חדרים</div>
                  <div className="font-semibold">{result.totals.num_rooms}</div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סה"כ שטח רצפה</div>
                  <div className="font-semibold">
                    {result.totals.total_area_m2 != null ? `${result.totals.total_area_m2.toFixed(2)} מ"ר` : "N/A"}
                  </div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סה"כ היקף</div>
                  <div className="font-semibold">
                    {result.totals.total_perimeter_m != null ? `${result.totals.total_perimeter_m.toFixed(2)} מ'` : "N/A"}
                  </div>
                </div>
                <div className="bg-slate-50 rounded p-3">
                  <div className="text-slate-500 text-xs">סה"כ פאנלים</div>
                  <div className="font-semibold">
                    {result.totals.total_baseboard_m != null ? `${result.totals.total_baseboard_m.toFixed(2)} מ'` : "N/A"}
                  </div>
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

          <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
            <h3 className="text-sm font-semibold mb-3">פירוט חדרים</h3>
            {!result || result.rooms.length === 0 ? (
              <p className="text-sm text-slate-500">לא נמצאו חדרים כרגע.</p>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-sm min-w-[680px]">
                  <thead>
                    <tr className="text-slate-500 border-b">
                      <th className="text-right p-2">#</th>
                      <th className="text-right p-2">שם חדר</th>
                      <th className="text-right p-2">שטח (מ"ר)</th>
                      <th className="text-right p-2">היקף (מ')</th>
                      <th className="text-right p-2">פאנלים (מ')</th>
                      <th className="text-right p-2">התאמה</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.rooms.map((room) => (
                      <tr key={room.room_id} className="border-b last:border-b-0">
                        <td className="p-2">#{room.room_id}</td>
                        <td className="p-2">{room.matched_name || "-"}</td>
                        <td className="p-2">{room.area_m2 != null ? room.area_m2.toFixed(2) : "-"}</td>
                        <td className="p-2">{room.perimeter_m != null ? room.perimeter_m.toFixed(2) : "-"}</td>
                        <td className="p-2">{room.baseboard_m != null ? room.baseboard_m.toFixed(2) : "-"}</td>
                        <td className="p-2">
                          {room.match_confidence != null && room.match_confidence > 0
                            ? `${Math.round(room.match_confidence * 100)}%`
                            : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {result?.has_overlay && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">ויזואליזציה</h3>
              <img src={overlayUrl} alt="overlay" className="max-w-full rounded border border-slate-200" />
            </section>
          )}

          {result && result.limitations.length > 0 && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">מגבלות וזיהוי בעיות</h3>
              <ul className="list-disc list-inside text-sm text-amber-700 space-y-1">
                {result.limitations.map((limitation, idx) => (
                  <li key={`${limitation}-${idx}`}>{limitation}</li>
                ))}
              </ul>
            </section>
          )}

          {result?.debug_json_safe && Object.keys(result.debug_json_safe).length > 0 && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">דיבאג ניתוח (שרת)</h3>
              <pre className="text-xs bg-slate-50 border border-slate-200 rounded p-3 overflow-auto">
                {JSON.stringify(result.debug_json_safe, null, 2)}
              </pre>
            </section>
          )}
        </main>
      </div>
    </div>
  );
};
