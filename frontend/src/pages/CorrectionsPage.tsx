import React from "react";
import { useToast } from "../components/Toast";
import { useConfirm } from "../components/ConfirmDialog";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  applyCorrection,
  getCorrectionsOverlayUrl,
  getCorrectionsSummary,
  resetCorrections,
  saveCorrections,
  type ManualCorrectionsSummary
} from "../api/managerCorrectionsApi";

type DrawMode = "add" | "remove" | "compare";
type Point = { x: number; y: number };

const BRUSH_WIDTH = 10;

export const CorrectionsPage: React.FC = () => {
  const toast = useToast();
  const confirm = useConfirm();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [mode, setMode] = React.useState<DrawMode>("add");
  const [summary, setSummary] = React.useState<ManualCorrectionsSummary | null>(null);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [overlayVersion, setOverlayVersion] = React.useState(0);

  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const imageRef = React.useRef<HTMLImageElement | null>(null);
  const [displaySize, setDisplaySize] = React.useState({ width: 860, height: 560 });
  const [drawing, setDrawing] = React.useState(false);
  const [draftStroke, setDraftStroke] = React.useState<Point[]>([]);
  const [strokes, setStrokes] = React.useState<Point[][]>([]);

  React.useEffect(() => {
    void (async () => {
      try {
        const data = await listWorkshopPlans();
        setPlans(data);
        if (data.length > 0) setSelectedPlanId(data[0].id);
      } catch (e) {
        console.error(e);
        setError("שגיאה בטעינת רשימת תוכניות.");
      }
    })();
  }, []);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    void (async () => {
      try {
        const data = await getCorrectionsSummary(selectedPlanId);
        setSummary(data);
        setError("");
      } catch (e) {
        console.error(e);
        setError("שגיאה בטעינת נתוני תיקונים.");
      }
    })();
  }, [selectedPlanId, overlayVersion]);

  const toLocalPoint = (clientX: number, clientY: number): Point => {
    const rect = wrapperRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.max(0, Math.min(displaySize.width, clientX - rect.left)),
      y: Math.max(0, Math.min(displaySize.height, clientY - rect.top))
    };
  };

  const handleImageLoad = () => {
    const img = imageRef.current;
    if (!img) return;
    const maxW = 980;
    const scale = Math.min(1, maxW / Math.max(1, img.naturalWidth));
    setDisplaySize({
      width: Math.max(280, Math.round(img.naturalWidth * scale)),
      height: Math.max(180, Math.round(img.naturalHeight * scale))
    });
  };

  const onMouseDown: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (mode === "compare") return;
    const p = toLocalPoint(e.clientX, e.clientY);
    setDrawing(true);
    setDraftStroke([p]);
  };

  const onMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (!drawing || mode === "compare") return;
    const p = toLocalPoint(e.clientX, e.clientY);
    setDraftStroke((prev) => [...prev, p]);
  };

  const onMouseUp: React.MouseEventHandler<HTMLDivElement> = () => {
    if (!drawing || mode === "compare") return;
    setDrawing(false);
    if (draftStroke.length >= 2) {
      setStrokes((prev) => [...prev, draftStroke]);
    }
    setDraftStroke([]);
  };

  const submitCorrection = async () => {
    if (!selectedPlanId || mode === "compare" || strokes.length === 0) return;
    try {
      setLoading(true);
      const data = await applyCorrection(selectedPlanId, {
        mode,
        display_width: displaySize.width,
        display_height: displaySize.height,
        strokes: strokes.map((stroke) => ({
          points: stroke.map((p) => [p.x, p.y] as [number, number]),
          width: BRUSH_WIDTH
        }))
      });
      setSummary(data);
      setStrokes([]);
      setDraftStroke([]);
      setOverlayVersion((v) => v + 1);
      toast("התיקון הוחל בהצלחה");
    } catch (e) {
      console.error(e);
      setError("שגיאה בשמירת התיקון.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!selectedPlanId) return;
    const ok = await confirm({ title: "איפוס תיקונים", message: "האם לאפס את כל התיקונים הידניים? פעולה זו בלתי הפיכה.", confirmText: "אפס", danger: true });
    if (!ok) return;
    try {
      setLoading(true);
      const data = await resetCorrections(selectedPlanId);
      setSummary(data);
      setStrokes([]);
      setDraftStroke([]);
      setOverlayVersion((v) => v + 1);
      toast("התיקונים אופסו");
    } catch (e) {
      console.error(e);
      setError("שגיאה באיפוס תיקונים.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCorrected = async () => {
    if (!selectedPlanId) return;
    try {
      setLoading(true);
      const data = await saveCorrections(selectedPlanId);
      setSummary(data);
      setStrokes([]);
      setDraftStroke([]);
      setOverlayVersion((v) => v + 1);
      toast("הגרסה המתוקנת נשמרה");
    } catch (e) {
      console.error(e);
      setError("שגיאה בשמירת גרסה מתוקנת.");
    } finally {
      setLoading(false);
    }
  };

  const mainOverlayUrl = selectedPlanId
    ? getCorrectionsOverlayUrl(selectedPlanId, mode === "compare" ? "corrected" : "auto", overlayVersion)
    : "";
  const autoUrl = selectedPlanId ? getCorrectionsOverlayUrl(selectedPlanId, "auto", overlayVersion) : "";
  const correctedUrl = selectedPlanId
    ? getCorrectionsOverlayUrl(selectedPlanId, "corrected", overlayVersion)
    : "";

  return (
    <div className="space-y-4">
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-[#31333F]">🎨 תיקונים ידניים</h2>
        <p className="text-xs text-slate-500 mt-1">הוספה/הסרה ידנית של קירות והשוואת לפני/אחרי.</p>
      </div>

      {error && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">{error}</div>}

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

          <div className="grid grid-cols-1 gap-2 text-xs">
            <button
              type="button"
              onClick={() => setMode("add")}
              className={`rounded border px-2 py-2 ${mode === "add" ? "border-green-500 text-green-700" : "border-slate-300"}`}
            >
              ➕ הוסף קירות
            </button>
            <button
              type="button"
              onClick={() => setMode("remove")}
              className={`rounded border px-2 py-2 ${mode === "remove" ? "border-red-500 text-red-700" : "border-slate-300"}`}
            >
              ➖ הסר קירות
            </button>
            <button
              type="button"
              onClick={() => setMode("compare")}
              className={`rounded border px-2 py-2 ${mode === "compare" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
            >
              👁️ השוואה
            </button>
          </div>

          {summary && (
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs space-y-1">
              <div>אורך אוטומטי: <b>{summary.auto_wall_length_m.toFixed(2)} מ&apos;</b></div>
              <div>אורך מתוקן: <b>{summary.corrected_wall_length_m.toFixed(2)} מ&apos;</b></div>
              <div>שינוי: <b>{summary.delta_wall_length_m >= 0 ? "+" : ""}{summary.delta_wall_length_m.toFixed(2)} מ&apos;</b></div>
            </div>
          )}

          {mode !== "compare" && (
            <>
              <button
                type="button"
                disabled={strokes.length === 0 || loading}
                onClick={() => void submitCorrection()}
                className="w-full bg-[#FF4B4B] text-white rounded py-2 text-sm font-semibold disabled:opacity-40"
              >
                {mode === "add" ? "✅ אשר הוספה" : "✅ אשר הסרה"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setStrokes([]);
                  setDraftStroke([]);
                }}
                className="w-full bg-white border border-slate-300 rounded py-2 text-sm"
              >
                נקה ציור נוכחי
              </button>
            </>
          )}

          <button
            type="button"
            disabled={loading}
            onClick={() => void handleSaveCorrected()}
            className="w-full bg-white border border-[#FF4B4B] text-[#FF4B4B] rounded py-2 text-sm font-semibold disabled:opacity-40"
          >
            💾 שמור גרסה מתוקנת
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={() => void handleReset()}
            className="w-full bg-white border border-slate-300 rounded py-2 text-sm disabled:opacity-40"
          >
            🔄 אפס תיקונים
          </button>
        </aside>

        <main className="space-y-4">
          {mode === "compare" ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <section className="bg-white border border-[#E6E6EA] rounded-lg p-3 shadow-sm">
                <div className="text-sm font-semibold mb-2">🤖 זיהוי אוטומטי</div>
                {selectedPlanId ? (
                  <img src={autoUrl} alt="auto-overlay" className="max-w-full rounded border border-slate-200" />
                ) : (
                  <div className="text-sm text-slate-500">בחר תוכנית להצגה.</div>
                )}
              </section>
              <section className="bg-white border border-[#E6E6EA] rounded-lg p-3 shadow-sm">
                <div className="text-sm font-semibold mb-2">✅ אחרי תיקון</div>
                {selectedPlanId ? (
                  <img src={correctedUrl} alt="corrected-overlay" className="max-w-full rounded border border-slate-200" />
                ) : (
                  <div className="text-sm text-slate-500">בחר תוכנית להצגה.</div>
                )}
              </section>
            </div>
          ) : (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-3 shadow-sm">
              <div className="text-sm font-semibold mb-2">
                {mode === "add" ? "🖌️ צייר על קירות חסרים" : "🖌️ סמן קירות להסרה"}
              </div>
              {selectedPlanId ? (
                <div
                  ref={wrapperRef}
                  className="relative border border-slate-200 rounded overflow-hidden w-fit cursor-crosshair"
                  onMouseDown={onMouseDown}
                  onMouseMove={onMouseMove}
                  onMouseUp={onMouseUp}
                  onMouseLeave={() => setDrawing(false)}
                >
                  <img
                    ref={imageRef}
                    src={mainOverlayUrl}
                    alt="corrections-base"
                    className="block"
                    style={{ width: displaySize.width, height: displaySize.height }}
                    onLoad={handleImageLoad}
                    draggable={false}
                  />
                  <svg width={displaySize.width} height={displaySize.height} className="absolute inset-0 pointer-events-none">
                    {strokes.map((stroke, idx) => (
                      <polyline
                        key={`stroke-${idx}`}
                        points={stroke.map((p) => `${p.x},${p.y}`).join(" ")}
                        fill="none"
                        stroke={mode === "add" ? "#16a34a" : "#dc2626"}
                        strokeWidth={BRUSH_WIDTH}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity={0.85}
                      />
                    ))}
                    {draftStroke.length > 1 && (
                      <polyline
                        points={draftStroke.map((p) => `${p.x},${p.y}`).join(" ")}
                        fill="none"
                        stroke={mode === "add" ? "#16a34a" : "#dc2626"}
                        strokeWidth={BRUSH_WIDTH}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity={0.85}
                      />
                    )}
                  </svg>
                </div>
              ) : (
                <div className="text-sm text-slate-500">בחר תוכנית כדי להתחיל תיקון.</div>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
};

