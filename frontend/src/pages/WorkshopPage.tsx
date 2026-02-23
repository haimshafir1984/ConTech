import React from "react";
import { useToast } from "../components/Toast";
import axios from "axios";
import { apiClient } from "../api/client";
import {
  getWorkshopPlan,
  getWorkshopOverlayUrl,
  listWorkshopPlans,
  type PlanDetail,
  type PlanSummary,
  updateWorkshopPlanScale,
  uploadWorkshopPlan
} from "../api/managerWorkshopApi";
import { getAreaAnalysis, getPlanReadiness, runAreaAnalysis, type PlanReadinessResponse } from "../api/managerInsightsApi";

type WorkshopTab = "overview" | "rooms" | "cost" | "diagnostics";

// ── Zoom-able canvas wrapper ────────────────────────────────────────────────
interface ZoomCanvasProps {
  imageUrl: string;
  overlayUrl: string;
  onImageLoad?: (w: number, h: number) => void;
  overlayLoading?: boolean;
}

const ZoomCanvas: React.FC<ZoomCanvasProps> = ({ imageUrl, overlayUrl, onImageLoad, overlayLoading }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = React.useState(false);
  const lastMouse = React.useRef({ x: 0, y: 0 });

  const clampZoom = (z: number) => Math.min(8, Math.max(0.5, z));

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => clampZoom(z * delta));
  };

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
  };

  const onMouseUp = () => setIsPanning(false);

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  return (
    <div className="relative bg-slate-100 rounded-lg overflow-hidden border border-slate-200" style={{ minHeight: 480 }}>
      <div
        ref={containerRef}
        className="w-full h-full overflow-hidden"
        style={{ cursor: isPanning ? "grabbing" : "grab", minHeight: 480 }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <div
          style={{
            transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`,
            transformOrigin: "top right",
            display: "inline-block",
            position: "relative",
            transition: isPanning ? "none" : "transform 0.05s"
          }}
        >
          <img
            src={imageUrl}
            alt="plan"
            draggable={false}
            style={{ display: "block", maxWidth: "100%", userSelect: "none" }}
            onLoad={(e) => {
              const img = e.currentTarget;
              onImageLoad?.(img.naturalWidth, img.naturalHeight);
            }}
          />
          <img
            src={overlayUrl}
            alt="overlay"
            draggable={false}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", userSelect: "none" }}
          />
          {overlayLoading && (
            <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.25)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 14 }}>
              מעדכן שכבות...
            </div>
          )}
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-3 left-3 flex flex-col gap-1 z-10">
        <button
          type="button"
          onClick={() => setZoom((z) => clampZoom(z * 1.25))}
          className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50"
        >+</button>
        <button
          type="button"
          onClick={resetView}
          className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-[10px] hover:bg-slate-50"
        >↺</button>
        <button
          type="button"
          onClick={() => setZoom((z) => clampZoom(z * 0.8))}
          className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50"
        >−</button>
      </div>
      <div className="absolute bottom-3 right-3 bg-black/40 text-white text-[11px] px-2 py-0.5 rounded">
        {Math.round(zoom * 100)}%
      </div>
    </div>
  );
};

// ── UploadZone ──────────────────────────────────────────────────────────────
interface UploadZoneProps {
  onFile: (f: File) => void;
  isLoading: boolean;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onFile, isLoading }) => {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [drag, setDrag] = React.useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${drag ? "border-[#FF4B4B] bg-red-50" : "border-slate-300 bg-slate-50 hover:bg-slate-100"}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
    >
      <div className="text-3xl mb-2">📂</div>
      <p className="font-semibold text-sm text-slate-700">{isLoading ? "מעלה..." : "גרור PDF לכאן או לחץ לבחירה"}</p>
      <p className="text-xs text-slate-400 mt-1">קבצי PDF של תוכניות בניה</p>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); if (inputRef.current) inputRef.current.value = ""; }}
      />
    </div>
  );
};

// ── Main WorkshopPage ────────────────────────────────────────────────────────
export const WorkshopPage: React.FC = () => {
  const toast = useToast();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = React.useState<PlanDetail | null>(null);
  const [readiness, setReadiness] = React.useState<PlanReadinessResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [uploadProgress, setUploadProgress] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [showFlooring, setShowFlooring] = React.useState(true);
  const [showRoomNumbers, setShowRoomNumbers] = React.useState(true);
  const [highlightWalls, setHighlightWalls] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState<WorkshopTab>("overview");
  const [concretePrice, setConcretePrice] = React.useState(1200);
  const [blocksPrice, setBlocksPrice] = React.useState(600);
  const [floorPrice, setFloorPrice] = React.useState(250);
  const [planDisplayName, setPlanDisplayName] = React.useState("");
  const [scaleText, setScaleText] = React.useState("1:50");
  const [overlayVersion, setOverlayVersion] = React.useState(0);
  const [overlayLoading, setOverlayLoading] = React.useState(false);
  const [analysisStatus, setAnalysisStatus] = React.useState<string | null>(null);

  const loadPlans = React.useCallback(async () => {
    try {
      const data = await listWorkshopPlans();
      setPlans(data);
      if (!selectedPlanId && data.length > 0) setSelectedPlanId(data[0].id);
    } catch (e) {
      console.error(e);
      setError("שגיאה בטעינת רשימת התוכניות מהשרת.");
    }
  }, [selectedPlanId]);

  const loadSelected = React.useCallback(async (planId: string) => {
    try {
      const [detail, ready] = await Promise.all([
        getWorkshopPlan(planId),
        getPlanReadiness(planId).catch(() => null)
      ]);
      setSelectedDetail(detail);
      setReadiness(ready);
    } catch (e) {
      console.error(e);
      setError("שגיאה בטעינת נתוני סדנת עבודה.");
    }
  }, []);

  React.useEffect(() => { void loadPlans(); }, [loadPlans]);
  React.useEffect(() => { if (selectedPlanId) void loadSelected(selectedPlanId); }, [selectedPlanId, loadSelected]);

  React.useEffect(() => {
    if (!selectedDetail) return;
    setPlanDisplayName(selectedDetail.summary.plan_name ?? "");
    const metaScale = typeof selectedDetail.meta?.scale === "string" ? selectedDetail.meta.scale : "1:50";
    setScaleText(metaScale || "1:50");
  }, [selectedDetail]);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    setOverlayLoading(true);
    setOverlayVersion((v) => v + 1);
  }, [selectedPlanId, showFlooring, showRoomNumbers, highlightWalls]);

  const savePlanSettings = async () => {
    if (!selectedPlanId) return;
    try {
      setIsLoading(true);
      const detail = await updateWorkshopPlanScale({ plan_id: selectedPlanId, scale_text: scaleText, plan_name: planDisplayName });
      setSelectedDetail(detail);
      await loadPlans();
      const ready = await getPlanReadiness(selectedPlanId).catch(() => null);
      setReadiness(ready);
      setError(null);
      toast('ההגדרות נשמרו בהצלחה');
    } catch (e) {
      console.error(e);
      const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה";
      setError(`שגיאה בשמירת קנ"מ: ${msg}`);
    } finally { setIsLoading(false); }
  };

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setUploadProgress(10);
    setAnalysisStatus("מעלה קובץ...");
    setError(null);
    try {
      setUploadProgress(30);
      setAnalysisStatus("מנתח תוכנית...");
      const detail = await uploadWorkshopPlan(file);
      setUploadProgress(90);
      setSelectedPlanId(detail.summary.id);
      setSelectedDetail(detail);
      await loadPlans();
      await loadSelected(detail.summary.id);
      const ready = await getPlanReadiness(detail.summary.id).catch(() => null);
      setReadiness(ready);
      setOverlayVersion((v) => v + 1);
      setUploadProgress(100);
      setAnalysisStatus("✅ הושלם: זוהו קירות, חומרים וריצוף.");
    } catch (e) {
      console.error(e);
      const isTimeout = axios.isAxiosError(e) && (e.code === "ECONNABORTED" || String(e.message).includes("timeout"));
      if (isTimeout) {
        try {
          const data = await listWorkshopPlans();
          setPlans(data);
          const latest = data.length > 0 ? data[data.length - 1] : null;
          if (latest) {
            setSelectedPlanId(latest.id);
            await loadSelected(latest.id);
            setOverlayVersion((v) => v + 1);
            setAnalysisStatus("הקובץ נותח (תגובה איטית). נטען בהצלחה.");
            setError(null);
            return;
          }
        } catch { /* fall through */ }
      }
      const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה לא ידועה";
      setError(`שגיאה בהעלאה: ${msg}`);
      setAnalysisStatus(null);
    } finally {
      setIsLoading(false);
      setUploadProgress(null);
      window.setTimeout(() => setAnalysisStatus(null), 3000);
    }
  };

  const runAnalysisNow = async () => {
    if (!selectedPlanId) return;
    try {
      setIsLoading(true);
      setAnalysisStatus("מאתר חדרים וקירות...");
      await runAreaAnalysis(selectedPlanId, { segmentation_method: "watershed", auto_min_area: true, min_area_px: 500 });
      setAnalysisStatus("✅ ניתוח הסתיים.");
      setOverlayLoading(true);
      setOverlayVersion((v) => v + 1);
      const ready = await getPlanReadiness(selectedPlanId).catch(() => null);
      setReadiness(ready);
    } catch (e) {
      const isTimeout = axios.isAxiosError(e) && (e.code === "ECONNABORTED" || String(e.message).includes("timeout"));
      if (isTimeout && selectedPlanId) {
        try {
          const existing = await getAreaAnalysis(selectedPlanId);
          if (existing && (existing.success || existing.rooms.length > 0)) {
            setAnalysisStatus("✅ ניתוח הושלם בשרת.");
            setOverlayLoading(true);
            setOverlayVersion((v) => v + 1);
            return;
          }
        } catch { /* fall through */ }
      }
      const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה";
      setError(`שגיאה בניתוח: ${msg}`);
    } finally {
      setIsLoading(false);
      window.setTimeout(() => setAnalysisStatus(null), 2000);
    }
  };

  const selectedSummary = React.useMemo(
    () => plans.find((p) => p.id === selectedPlanId) ?? selectedDetail?.summary ?? null,
    [plans, selectedPlanId, selectedDetail]
  );

  const selectedScale = selectedSummary?.scale_px_per_meter ?? 200;

  const imageUrl = selectedPlanId
    ? `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(selectedPlanId)}/image`
    : "";
  const overlayUrl = selectedPlanId
    ? getWorkshopOverlayUrl(selectedPlanId, { show_flooring: showFlooring, show_room_numbers: showRoomNumbers, highlight_walls: highlightWalls, version: overlayVersion })
    : "";

  const roomRows = React.useMemo(() => {
    const meta = selectedDetail?.meta as Record<string, unknown> | undefined;
    if (!meta) return [];
    const rooms = Array.isArray(meta.rooms) ? meta.rooms : [];
    const safeN = (v: unknown): number | null => {
      if (typeof v === "number" && isFinite(v)) return v;
      if (typeof v === "string") { const n = Number(v); return isFinite(n) ? n : null; }
      if (v && typeof v === "object" && "value" in v) { const n = Number((v as { value?: unknown }).value); return isFinite(n) ? n : null; }
      return null;
    };
    return rooms.slice(0, 50).map((room, idx) => {
      const r = (room ?? {}) as Record<string, unknown>;
      return {
        id: idx + 1,
        name: (typeof r.room_name === "string" && r.room_name) || (typeof r.name === "string" && r.name) || `חדר ${idx + 1}`,
        area: safeN(r.area_m2),
        perimeter: safeN(r.perimeter_m ?? r.perimeter)
      };
    });
  }, [selectedDetail]);

  const totalQuote = React.useMemo(() => {
    if (!selectedSummary) return 0;
    return (selectedSummary.concrete_length_m ?? 0) * concretePrice
      + (selectedSummary.blocks_length_m ?? 0) * blocksPrice
      + (selectedSummary.flooring_area_m2 ?? 0) * floorPrice;
  }, [selectedSummary, concretePrice, blocksPrice, floorPrice]);

  const hasPlan = Boolean(selectedPlanId);

  return (
    <div className="space-y-4">
      {/* Header */}
      <section className="bg-white rounded-lg shadow-sm border border-[#E6E6EA] p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="m-0 text-lg font-semibold text-[#31333F]">🛠️ סדנת עבודה</h2>
            <p className="m-0 text-xs text-slate-500">העלאה, בקרה, ניתוח ותמחור תוכנית</p>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <select
              className="px-2 py-2 rounded border border-slate-300 min-w-[200px] bg-white"
              value={selectedPlanId ?? ""}
              onChange={(e) => setSelectedPlanId(e.target.value)}
            >
              {plans.length === 0 && <option value="">אין תוכניות</option>}
              {plans.map((p) => <option key={p.id} value={p.id}>{p.plan_name}</option>)}
            </select>
          </div>
        </div>

        {/* Upload progress bar */}
        {uploadProgress !== null && (
          <div className="space-y-1">
            <div className="h-2 bg-slate-200 rounded overflow-hidden">
              <div
                className="h-full bg-[#FF4B4B] transition-all duration-500"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">{analysisStatus}</p>
          </div>
        )}
        {analysisStatus && uploadProgress === null && (
          <div className="rounded border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-700">{analysisStatus}</div>
        )}
      </section>

      {error && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</div>}

      {/* KPI cards */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: "קנ\"מ", value: `${selectedScale.toFixed(0)} px/מ'` },
          { label: "קירות", value: `${selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'` },
          { label: "ריצוף", value: `${selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר` },
          { label: "עלות משוערת", value: `${totalQuote.toLocaleString()} ₪` },
        ].map((c) => (
          <div key={c.label} className="bg-white border border-[#E6E6EA] rounded-lg p-3">
            <div className="text-xs text-slate-500">{c.label}</div>
            <div className="font-semibold text-[#31333F]">{c.value}</div>
          </div>
        ))}
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-[300px,1fr] gap-5">
        {/* Sidebar */}
        <aside className="bg-white rounded-lg shadow-sm border border-[#E6E6EA] p-4 space-y-4">
          {/* Upload zone */}
          <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} />

          <div className="space-y-2 pt-2 border-t border-[#E6E6EA]">
            <div className="text-sm font-semibold text-[#31333F]">הגדרות תוכנית</div>
            <input
              className="w-full px-2 py-2 border border-slate-300 rounded"
              value={planDisplayName}
              onChange={(e) => setPlanDisplayName(e.target.value)}
              placeholder="שם תוכנית"
            />
            <select
              className="w-full px-2 py-2 border border-slate-300 rounded bg-white"
              value={scaleText}
              onChange={(e) => setScaleText(e.target.value)}
            >
              {["1:20","1:25","1:50","1:75","1:100","1:200"].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <button
              type="button"
              onClick={() => void savePlanSettings()}
              disabled={!hasPlan || isLoading}
              className="w-full bg-white border border-[#FF4B4B] text-[#FF4B4B] rounded px-3 py-2 font-semibold disabled:opacity-50 text-sm"
            >
              🧭 שמור קנ"מ ושם
            </button>
          </div>

          <div className="space-y-2 pt-2 border-t border-[#E6E6EA]">
            <div className="text-sm font-semibold text-[#31333F]">ניתוח ותצוגה</div>
            <button
              type="button"
              onClick={() => void runAnalysisNow()}
              className="w-full bg-[#FF4B4B] text-white rounded px-3 py-2 font-semibold disabled:opacity-50 text-sm"
              disabled={!hasPlan || isLoading}
            >
              🔍 הרץ ניתוח חדרים
            </button>
            {[
              { label: "הצג ריצוף", val: showFlooring, set: setShowFlooring },
              { label: "הצג מספרי חדרים", val: showRoomNumbers, set: setShowRoomNumbers },
              { label: "הדגש קירות", val: highlightWalls, set: setHighlightWalls },
            ].map(({ label, val, set }) => (
              <label key={label} className="flex items-center gap-2 text-xs cursor-pointer">
                <input type="checkbox" checked={val} onChange={(e) => set(e.target.checked)} />
                {label}
              </label>
            ))}
          </div>

          <div className="space-y-2 pt-2 border-t border-[#E6E6EA]">
            <div className="text-sm font-semibold text-[#31333F]">תמחור מהיר</div>
            {[
              { label: "בטון (₪/מ')", val: concretePrice, set: setConcretePrice },
              { label: "בלוקים (₪/מ')", val: blocksPrice, set: setBlocksPrice },
              { label: 'ריצוף (₪/מ"ר)', val: floorPrice, set: setFloorPrice },
            ].map(({ label, val, set }) => (
              <label key={label} className="text-xs flex items-center justify-between gap-2">
                <span>{label}</span>
                <input type="number" value={val} onChange={(e) => set(Number(e.target.value))} className="w-24 px-2 py-1 border border-slate-300 rounded" />
              </label>
            ))}
          </div>
        </aside>

        {/* Main content */}
        <main className="space-y-4">
          {/* Canvas */}
          {hasPlan ? (
            <ZoomCanvas
              imageUrl={imageUrl}
              overlayUrl={overlayUrl}
              overlayLoading={overlayLoading}
            />
          ) : (
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm min-h-[480px] flex items-center justify-center text-slate-400 text-sm">
              העלה תוכנית PDF כדי להתחיל
            </div>
          )}

          {/* Tabs */}
          <section className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm overflow-hidden">
            <div className="flex border-b border-[#E6E6EA] bg-slate-50 text-sm overflow-x-auto">
              {[
                { id: "overview", label: "📋 סקירה" },
                { id: "rooms", label: "📐 חדרים" },
                { id: "cost", label: "💰 עלויות" },
                { id: "diagnostics", label: "🩺 בדיקות" },
              ].map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setActiveTab(t.id as WorkshopTab)}
                  className={`px-4 py-3 border-b-[3px] whitespace-nowrap ${activeTab === t.id ? "border-[#FF4B4B] text-[#FF4B4B] bg-white" : "border-transparent text-slate-600"}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="p-4">
              {activeTab === "overview" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  <div className="bg-slate-50 rounded p-3">
                    <div className="font-semibold mb-1">פרטי תוכנית</div>
                    <div>שם: {planDisplayName || "ללא שם"}</div>
                    <div>קנ"מ: {scaleText}</div>
                    <div>מחושב: {selectedScale.toFixed(2)} px/מ'</div>
                  </div>
                  <div className="bg-slate-50 rounded p-3">
                    <div className="font-semibold mb-1">תוצרים</div>
                    <div>קירות: {selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בטון: {selectedSummary?.concrete_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בלוקים: {selectedSummary?.blocks_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>ריצוף: {selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר</div>
                  </div>
                </div>
              )}
              {activeTab === "rooms" && (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-slate-500 border-b">
                      <th className="text-right p-2">#</th>
                      <th className="text-right p-2">שם חדר</th>
                      <th className="text-right p-2">שטח (מ"ר)</th>
                      <th className="text-right p-2">היקף (מ')</th>
                    </tr>
                  </thead>
                  <tbody>
                    {roomRows.length === 0 && <tr><td className="p-3 text-slate-500" colSpan={4}>אין נתוני חדרים. הרץ ניתוח חדרים קודם.</td></tr>}
                    {roomRows.map((row) => (
                      <tr key={row.id} className="border-b last:border-b-0">
                        <td className="p-2">{row.id}</td>
                        <td className="p-2">{row.name}</td>
                        <td className="p-2">{row.area?.toFixed(2) ?? "—"}</td>
                        <td className="p-2">{row.perimeter?.toFixed(2) ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {activeTab === "cost" && (
                <div className="space-y-3 text-sm">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <div className="bg-slate-50 rounded p-3">
                      <div className="text-xs text-slate-500">בטון</div>
                      <div className="font-semibold">{(selectedSummary?.concrete_length_m ?? 0).toFixed(2)} מ' × {concretePrice}₪</div>
                      <div className="text-[#FF4B4B] font-bold">{((selectedSummary?.concrete_length_m ?? 0) * concretePrice).toLocaleString()} ₪</div>
                    </div>
                    <div className="bg-slate-50 rounded p-3">
                      <div className="text-xs text-slate-500">בלוקים</div>
                      <div className="font-semibold">{(selectedSummary?.blocks_length_m ?? 0).toFixed(2)} מ' × {blocksPrice}₪</div>
                      <div className="text-[#FF4B4B] font-bold">{((selectedSummary?.blocks_length_m ?? 0) * blocksPrice).toLocaleString()} ₪</div>
                    </div>
                    <div className="bg-slate-50 rounded p-3">
                      <div className="text-xs text-slate-500">ריצוף</div>
                      <div className="font-semibold">{(selectedSummary?.flooring_area_m2 ?? 0).toFixed(2)} מ"ר × {floorPrice}₪</div>
                      <div className="text-[#FF4B4B] font-bold">{((selectedSummary?.flooring_area_m2 ?? 0) * floorPrice).toLocaleString()} ₪</div>
                    </div>
                  </div>
                  <div className="border-t pt-2 text-base font-bold text-[#31333F]">סה"כ משוער: {totalQuote.toLocaleString()} ₪</div>
                </div>
              )}
              {activeTab === "diagnostics" && (
                <div className="space-y-2 text-sm">
                  {!readiness ? (
                    <div className="text-slate-500">אין נתוני בדיקה.</div>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                        {[
                          ["original", readiness.has_original],
                          ["thick_walls", readiness.has_thick_walls],
                          ["flooring_mask", readiness.has_flooring_mask],
                          ["scale", readiness.has_scale_px_per_meter],
                          ["meters/px", readiness.has_meters_per_pixel],
                          ["llm_rooms", readiness.has_llm_rooms],
                        ].map(([k, v]) => (
                          <div key={String(k)} className="bg-slate-50 rounded p-2">{String(k)}: {v ? "✅" : k === "llm_rooms" ? "⚠️" : "❌"}</div>
                        ))}
                      </div>
                      {readiness.issues.length > 0 ? (
                        <ul className="list-disc list-inside text-amber-700">
                          {readiness.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                        </ul>
                      ) : (
                        <div className="text-green-700">✅ כל הבדיקות עברו.</div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};
