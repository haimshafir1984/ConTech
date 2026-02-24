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
    <div className="relative bg-slate-100 rounded-xl overflow-hidden border border-slate-200" style={{ minHeight: 480 }}>
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
        <button type="button" onClick={() => setZoom((z) => clampZoom(z * 1.25))} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50">+</button>
        <button type="button" onClick={resetView} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-[10px] hover:bg-slate-50">↺</button>
        <button type="button" onClick={() => setZoom((z) => clampZoom(z * 0.8))} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50">−</button>
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
  compact?: boolean;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onFile, isLoading, compact }) => {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [drag, setDrag] = React.useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  if (compact) {
    return (
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={isLoading}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          border: "1.5px dashed #94A3B8", borderRadius: 10,
          background: "#F8FAFC", padding: "8px 18px",
          cursor: "pointer", color: "#475569", fontSize: 13, fontWeight: 600,
        }}
      >
        <span style={{ fontSize: 18 }}>📂</span>
        {isLoading ? "מעלה..." : "העלה תוכנית נוספת"}
        <input ref={inputRef} type="file" accept="application/pdf" style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); if (inputRef.current) inputRef.current.value = ""; }} />
      </button>
    );
  }

  return (
    <div
      style={{
        border: `2px dashed ${drag ? "#1B3A6B" : "#CBD5E1"}`,
        borderRadius: 16,
        padding: "40px 24px",
        textAlign: "center",
        cursor: "pointer",
        background: drag ? "#EFF6FF" : "#F8FAFC",
        transition: "border-color 0.15s, background 0.15s",
      }}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
    >
      <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
      <p style={{ fontWeight: 700, fontSize: 15, color: "#1B3A6B", margin: 0 }}>
        {isLoading ? "מעלה ומנתח..." : "גרור קובץ PDF לכאן"}
      </p>
      <p style={{ fontSize: 12, color: "#94A3B8", marginTop: 6 }}>או לחץ לבחירת קובץ · תוכניות בניה PDF</p>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        style={{ display: "none" }}
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
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* ── Upload zone ── */}
      {plans.length === 0 ? (
        <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} />
      ) : (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} compact />
          {analysisStatus && uploadProgress === null && (
            <div style={{ fontSize: 13, color: "#1d4ed8", background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8, padding: "6px 14px" }}>{analysisStatus}</div>
          )}
        </div>
      )}

      {/* Upload progress bar */}
      {uploadProgress !== null && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ height: 6, background: "#E2E8F0", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ height: "100%", background: "#1B3A6B", borderRadius: 99, width: `${uploadProgress}%`, transition: "width 0.5s" }} />
          </div>
          <div style={{ fontSize: 12, color: "#64748b" }}>{analysisStatus}</div>
        </div>
      )}

      {error && (
        <div style={{ fontSize: 13, color: "#DC2626", background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 10, padding: "10px 14px" }}>{error}</div>
      )}

      {/* ── Plan cards grid ── */}
      {plans.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 14 }}>
          {plans.map((p) => {
            const active = p.id === selectedPlanId;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => setSelectedPlanId(p.id)}
                style={{
                  textAlign: "right",
                  background: "#fff",
                  border: active ? "2.5px solid #1B3A6B" : "1.5px solid #E2E8F0",
                  borderRadius: 14,
                  overflow: "hidden",
                  cursor: "pointer",
                  boxShadow: active ? "0 4px 16px rgba(27,58,107,0.15)" : "0 1px 4px rgba(0,0,0,0.06)",
                  transition: "box-shadow 0.15s, border-color 0.15s",
                  padding: 0,
                }}
              >
                {/* Thumbnail */}
                <div style={{ height: 120, background: "#F1F5F9", overflow: "hidden" }}>
                  <img
                    src={`${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(p.id)}/image`}
                    alt={p.plan_name}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                  />
                </div>
                {/* Info */}
                <div style={{ padding: "10px 12px" }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: "#1B3A6B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.plan_name}</div>
                  <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 2 }}>
                    {p.total_wall_length_m != null ? `${p.total_wall_length_m.toFixed(1)} מ' קירות` : "—"}
                  </div>
                  {active && (
                    <div style={{ marginTop: 6, fontSize: 10, fontWeight: 700, color: "#fff", background: "#1B3A6B", borderRadius: 6, padding: "2px 8px", display: "inline-block" }}>✓ פעיל</div>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* ── Empty state ── */}
      {plans.length === 0 && !isLoading && (
        <div style={{ textAlign: "center", padding: "48px 0", color: "#94A3B8", fontSize: 14 }}>
          העלה תוכנית PDF כדי להתחיל
        </div>
      )}

      {/* ── Selected plan detail ── */}
      {hasPlan && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Settings toolbar */}
          <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 14, padding: "14px 18px", display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end", boxShadow: "0 1px 4px rgba(0,0,0,0.05)" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 160, flex: 1 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>שם תוכנית</span>
              <input
                style={{ padding: "6px 10px", border: "1px solid #CBD5E1", borderRadius: 8, fontSize: 13 }}
                value={planDisplayName}
                onChange={(e) => setPlanDisplayName(e.target.value)}
                placeholder="שם תוכנית"
              />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>קנ"מ</span>
              <select
                style={{ padding: "6px 10px", border: "1px solid #CBD5E1", borderRadius: 8, fontSize: 13, background: "#fff" }}
                value={scaleText}
                onChange={(e) => setScaleText(e.target.value)}
              >
                {["1:20","1:25","1:50","1:75","1:100","1:200"].map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <button
              type="button"
              onClick={() => void savePlanSettings()}
              disabled={isLoading}
              style={{ padding: "7px 16px", border: "1.5px solid #1B3A6B", color: "#1B3A6B", borderRadius: 8, fontSize: 13, fontWeight: 600, background: "#fff", cursor: "pointer", opacity: isLoading ? 0.5 : 1 }}
            >
              שמור
            </button>
            <button
              type="button"
              onClick={() => void runAnalysisNow()}
              disabled={isLoading}
              style={{ padding: "7px 16px", background: "#1B3A6B", color: "#fff", borderRadius: 8, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer", opacity: isLoading ? 0.5 : 1 }}
            >
              🔍 ניתוח
            </button>
            <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
              {[
                { label: "ריצוף", val: showFlooring, set: setShowFlooring },
                { label: "מספרים", val: showRoomNumbers, set: setShowRoomNumbers },
                { label: "קירות", val: highlightWalls, set: setHighlightWalls },
              ].map(({ label, val, set }) => (
                <label key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "#475569", cursor: "pointer" }}>
                  <input type="checkbox" checked={val} onChange={(e) => set(e.target.checked)} />
                  {label}
                </label>
              ))}
            </div>
          </div>

          {/* KPI row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 12 }}>
            {[
              { label: "קנ\"מ", value: `${selectedScale.toFixed(0)} px/מ'` },
              { label: "קירות", value: `${selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'` },
              { label: "ריצוף", value: `${selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר` },
              { label: "עלות משוערת", value: `${totalQuote.toLocaleString()} ₪` },
            ].map((c) => (
              <div key={c.label} style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 12, padding: "12px 14px" }}>
                <div style={{ fontSize: 11, color: "#94A3B8" }}>{c.label}</div>
                <div style={{ fontWeight: 700, fontSize: 15, color: "#1B3A6B", marginTop: 2 }}>{c.value}</div>
              </div>
            ))}
          </div>

          {/* Canvas */}
          <ZoomCanvas
            imageUrl={imageUrl}
            overlayUrl={overlayUrl}
            overlayLoading={overlayLoading}
          />

          {/* Tabs */}
          <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 14, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.05)" }}>
            <div style={{ display: "flex", borderBottom: "1px solid #E2E8F0", background: "#F8FAFC", overflowX: "auto" }}>
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
                  style={{
                    padding: "12px 20px",
                    border: "none",
                    borderBottom: activeTab === t.id ? "3px solid #1B3A6B" : "3px solid transparent",
                    background: activeTab === t.id ? "#fff" : "transparent",
                    color: activeTab === t.id ? "#1B3A6B" : "#64748b",
                    fontWeight: activeTab === t.id ? 700 : 400,
                    fontSize: 13,
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div style={{ padding: 18 }}>
              {activeTab === "overview" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 13 }}>
                  <div style={{ background: "#F8FAFC", borderRadius: 10, padding: 14 }}>
                    <div style={{ fontWeight: 700, marginBottom: 8, color: "#1B3A6B" }}>פרטי תוכנית</div>
                    <div>שם: {planDisplayName || "ללא שם"}</div>
                    <div>קנ"מ: {scaleText}</div>
                    <div>מחושב: {selectedScale.toFixed(2)} px/מ'</div>
                  </div>
                  <div style={{ background: "#F8FAFC", borderRadius: 10, padding: 14 }}>
                    <div style={{ fontWeight: 700, marginBottom: 8, color: "#1B3A6B" }}>תוצרים</div>
                    <div>קירות: {selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בטון: {selectedSummary?.concrete_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בלוקים: {selectedSummary?.blocks_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>ריצוף: {selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר</div>
                  </div>
                </div>
              )}
              {activeTab === "rooms" && (
                <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ color: "#94A3B8", borderBottom: "1px solid #E2E8F0" }}>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>#</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>שם חדר</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>שטח (מ"ר)</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>היקף (מ')</th>
                    </tr>
                  </thead>
                  <tbody>
                    {roomRows.length === 0 && <tr><td style={{ padding: 12, color: "#94A3B8" }} colSpan={4}>אין נתוני חדרים. הרץ ניתוח חדרים קודם.</td></tr>}
                    {roomRows.map((row) => (
                      <tr key={row.id} style={{ borderBottom: "1px solid #F1F5F9" }}>
                        <td style={{ padding: "6px 8px" }}>{row.id}</td>
                        <td style={{ padding: "6px 8px" }}>{row.name}</td>
                        <td style={{ padding: "6px 8px" }}>{row.area?.toFixed(2) ?? "—"}</td>
                        <td style={{ padding: "6px 8px" }}>{row.perimeter?.toFixed(2) ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {activeTab === "cost" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14, fontSize: 13 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12 }}>
                    {[
                      { label: "בטון", length: selectedSummary?.concrete_length_m ?? 0, price: concretePrice, set: setConcretePrice, unit: "מ'" },
                      { label: "בלוקים", length: selectedSummary?.blocks_length_m ?? 0, price: blocksPrice, set: setBlocksPrice, unit: "מ'" },
                      { label: 'ריצוף', length: selectedSummary?.flooring_area_m2 ?? 0, price: floorPrice, set: setFloorPrice, unit: 'מ"ר' },
                    ].map(({ label, length, price, set, unit }) => (
                      <div key={label} style={{ background: "#F8FAFC", borderRadius: 10, padding: 14 }}>
                        <div style={{ fontSize: 11, color: "#94A3B8" }}>{label}</div>
                        <div style={{ fontWeight: 600, marginTop: 4 }}>{length.toFixed(2)} {unit}</div>
                        <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8, fontSize: 12, color: "#475569" }}>
                          ₪/יחידה
                          <input type="number" value={price} onChange={(e) => set(Number(e.target.value))}
                            style={{ width: 72, padding: "3px 6px", border: "1px solid #CBD5E1", borderRadius: 6, fontSize: 12 }} />
                        </label>
                        <div style={{ fontWeight: 700, color: "#1B3A6B", marginTop: 6 }}>{(length * price).toLocaleString()} ₪</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ borderTop: "1px solid #E2E8F0", paddingTop: 12, fontWeight: 700, fontSize: 15, color: "#1B3A6B" }}>
                    סה"כ משוער: {totalQuote.toLocaleString()} ₪
                  </div>
                </div>
              )}
              {activeTab === "diagnostics" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 12, fontSize: 13 }}>
                  {!readiness ? (
                    <div style={{ color: "#94A3B8" }}>אין נתוני בדיקה.</div>
                  ) : (
                    <>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8, fontSize: 12 }}>
                        {[
                          ["original", readiness.has_original],
                          ["thick_walls", readiness.has_thick_walls],
                          ["flooring_mask", readiness.has_flooring_mask],
                          ["scale", readiness.has_scale_px_per_meter],
                          ["meters/px", readiness.has_meters_per_pixel],
                          ["llm_rooms", readiness.has_llm_rooms],
                        ].map(([k, v]) => (
                          <div key={String(k)} style={{ background: "#F8FAFC", borderRadius: 8, padding: "8px 10px" }}>
                            {String(k)}: {v ? "✅" : k === "llm_rooms" ? "⚠️" : "❌"}
                          </div>
                        ))}
                      </div>
                      {readiness.issues.length > 0 ? (
                        <ul style={{ paddingRight: 20, color: "#B45309" }}>
                          {readiness.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                        </ul>
                      ) : (
                        <div style={{ color: "#15803D" }}>✅ כל הבדיקות עברו.</div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
