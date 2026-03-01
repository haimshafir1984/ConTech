import React from "react";
import { useToast } from "../components/Toast";
import { ErrorAlert, SkeletonGrid } from "../components/UiHelpers";
import axios from "axios";
import { apiClient } from "../api/client";
import {
  clearAllWorkshopPlans,
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
  onOverlayLoad?: () => void;
}

const ZoomCanvas: React.FC<ZoomCanvasProps> = ({ imageUrl, overlayUrl, onImageLoad, overlayLoading, onOverlayLoad }) => {
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

  // ── Touch support (pan + pinch-to-zoom) ──
  const lastPinchDistRef = React.useRef<number | null>(null);
  const lastTouchPosRef = React.useRef<{ x: number; y: number } | null>(null);

  const onTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    if (e.touches.length === 2) {
      lastPinchDistRef.current = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      lastTouchPosRef.current = null;
    } else if (e.touches.length === 1) {
      lastPinchDistRef.current = null;
      lastTouchPosRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      setIsPanning(true);
    }
  };

  const onTouchMove = (e: React.TouchEvent) => {
    e.preventDefault();
    if (e.touches.length === 2 && lastPinchDistRef.current !== null) {
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const ratio = dist / lastPinchDistRef.current;
      setZoom((z) => clampZoom(z * ratio));
      lastPinchDistRef.current = dist;
    } else if (e.touches.length === 1 && lastTouchPosRef.current) {
      const t = e.touches[0];
      const dx = t.clientX - lastTouchPosRef.current.x;
      const dy = t.clientY - lastTouchPosRef.current.y;
      lastTouchPosRef.current = { x: t.clientX, y: t.clientY };
      setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
    }
  };

  const onTouchEnd = (e: React.TouchEvent) => {
    e.preventDefault();
    lastPinchDistRef.current = null;
    lastTouchPosRef.current = null;
    setIsPanning(false);
  };

  return (
    <div className="relative bg-slate-100 rounded-xl overflow-hidden border border-slate-200" style={{ minHeight: 480 }}>
      <div
        ref={containerRef}
        className="w-full h-full overflow-hidden"
        style={{ cursor: isPanning ? "grabbing" : "grab", minHeight: 480, touchAction: "none" }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
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
            onLoad={() => onOverlayLoad?.()}
            onError={() => onOverlayLoad?.()}
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
        <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        {isLoading ? "מעלה..." : "העלה תוכנית נוספת"}
        <input ref={inputRef} type="file" accept="application/pdf" style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); if (inputRef.current) inputRef.current.value = ""; }} />
      </button>
    );
  }

  return (
    <div
      className="upload-zone"
      style={drag ? { borderColor: "var(--navy)", background: "#e0eaf5" } : {}}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
    >
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 12, color: drag ? "var(--navy)" : "var(--text-3)" }}>
        <svg width={44} height={44} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <h3>{isLoading ? "מעלה ומנתח..." : "גרור קובץ PDF לכאן"}</h3>
      <p>או לחץ לבחירת קובץ מהמחשב</p>
      <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>PDF בלבד · מקסימום 50MB</p>
      <div style={{ marginTop: 14 }}>
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
        >
          <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          בחר קובץ
        </button>
      </div>
      <input
        ref={inputRef}
        id="workshop-upload-input"
        type="file"
        accept="application/pdf"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); if (inputRef.current) inputRef.current.value = ""; }}
      />
    </div>
  );
};

// ── Main WorkshopPage ────────────────────────────────────────────────────────
export const WorkshopPage: React.FC<{ onNavigatePlanning?: () => void }> = ({ onNavigatePlanning }) => {
  const toast = useToast();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = React.useState<PlanDetail | null>(null);
  const [readiness, setReadiness] = React.useState<PlanReadinessResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [plansLoading, setPlansLoading] = React.useState(true);
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
  const analysisStatusTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadPlans = React.useCallback(async () => {
    try {
      setPlansLoading(true);
      const data = await listWorkshopPlans();
      setPlans(data);
      setSelectedPlanId((prev) => (!prev && data.length > 0 ? data[0].id : prev));
    } catch (e) {
      console.error(e);
      setError("שגיאה בטעינת רשימת התוכניות מהשרת.");
    } finally {
      setPlansLoading(false);
    }
  }, []);

  const _isRestartLost = (e: unknown) => {
    if (!axios.isAxiosError(e)) return false;
    const detail: string = (e.response?.data as { detail?: string })?.detail ?? "";
    return e.response?.status === 409 || detail.includes("PLAN_RESTART_LOST");
  };

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
      if (_isRestartLost(e)) {
        setError("נתוני התוכנית אינם זמינים — השרת עלה מחדש. אנא העלה את קובץ ה-PDF שוב.");
      } else {
        setError("שגיאה בטעינת נתוני סדנת עבודה.");
      }
    }
  }, []);

  // cleanup analysisStatus timer on unmount
  React.useEffect(() => {
    return () => {
      if (analysisStatusTimerRef.current !== null) clearTimeout(analysisStatusTimerRef.current);
    };
  }, []);

  const scheduleStatusClear = React.useCallback((ms: number) => {
    if (analysisStatusTimerRef.current !== null) clearTimeout(analysisStatusTimerRef.current);
    analysisStatusTimerRef.current = setTimeout(() => {
      setAnalysisStatus(null);
      analysisStatusTimerRef.current = null;
    }, ms);
  }, []);

  React.useEffect(() => { void loadPlans(); }, [loadPlans]);
  React.useEffect(() => { if (selectedPlanId) void loadSelected(selectedPlanId); }, [selectedPlanId, loadSelected]);

  React.useEffect(() => {
    if (!selectedDetail) return;
    // Prefer a descriptive name auto-built from title-block; fall back to stored plan_name
    const meta = selectedDetail.meta as Record<string, unknown> | undefined;
    const stored = selectedDetail.summary.plan_name ?? "";
    let bestName = stored;
    if (!bestName || bestName.endsWith(".pdf") || bestName === selectedDetail.summary.filename) {
      const parts: string[] = [];
      const proj = typeof meta?.project_name === "string" ? meta.project_name : "";
      const sheet = (typeof meta?.plan_title === "string" && meta.plan_title)
        || (typeof meta?.sheet_name === "string" && meta.sheet_name)
        || (typeof meta?.sheet_number === "string" && meta.sheet_number)
        || "";
      if (proj) parts.push(proj);
      if (sheet) parts.push(sheet);
      if (parts.length) bestName = parts.join(" — ");
    }
    setPlanDisplayName(bestName || stored);
    const metaScale = typeof selectedDetail.meta?.scale === "string" ? selectedDetail.meta.scale : "1:50";
    setScaleText(metaScale || "1:50");
  }, [selectedDetail]);

  const overlayDebounceRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  React.useEffect(() => {
    if (!selectedPlanId) return;
    // Immediate on plan change, debounced on checkbox change
    const trigger = () => {
      setOverlayLoading(true);
      setOverlayVersion((v) => v + 1);
    };
    if (overlayDebounceRef.current !== null) clearTimeout(overlayDebounceRef.current);
    overlayDebounceRef.current = setTimeout(trigger, 300);
    return () => {
      if (overlayDebounceRef.current !== null) clearTimeout(overlayDebounceRef.current);
    };
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
      if (_isRestartLost(e)) {
        setError('נתוני התוכנית אינם זמינים — השרת עלה מחדש. אנא העלה את קובץ ה-PDF שוב.');
      } else {
        const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה";
        setError(`שגיאה בשמירת קנ"מ: ${msg}`);
      }
    } finally { setIsLoading(false); }
  };

  const handleClearAll = async () => {
    if (!window.confirm("האם אתה בטוח שברצונך למחוק את כל התוכניות? פעולה זו אינה הפיכה.")) return;
    try {
      setIsLoading(true);
      setError(null);
      await clearAllWorkshopPlans();
      setPlans([]);
      setSelectedPlanId(null);
      setSelectedDetail(null);
      setReadiness(null);
      toast("כל התוכניות נמחקו בהצלחה");
    } catch (e) {
      console.error(e);
      setError("שגיאה במחיקת התוכניות.");
    } finally {
      setIsLoading(false);
    }
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
      setAnalysisStatus("הושלם: זוהו קירות, חומרים וריצוף.");
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
      scheduleStatusClear(3000);
    }
  };

  const runAnalysisNow = async () => {
    if (!selectedPlanId) return;
    try {
      setIsLoading(true);
      setAnalysisStatus("מאתר חדרים וקירות...");
      await runAreaAnalysis(selectedPlanId, { segmentation_method: "watershed", auto_min_area: true, min_area_px: 500 });
      setAnalysisStatus("ניתוח הסתיים.");
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
            setAnalysisStatus("ניתוח הושלם בשרת.");
            setOverlayLoading(true);
            setOverlayVersion((v) => v + 1);
            return;
          }
        } catch { /* fall through */ }
      }
      if (_isRestartLost(e)) {
        setError("נתוני התוכנית אינם זמינים — השרת עלה מחדש. אנא העלה את קובץ ה-PDF שוב.");
      } else {
        const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה";
        setError(`שגיאה בניתוח: ${msg}`);
      }
    } finally {
      setIsLoading(false);
      scheduleStatusClear(2000);
    }
  };

  const selectedSummary = React.useMemo(
    () => plans.find((p) => p.id === selectedPlanId) ?? selectedDetail?.summary ?? null,
    [plans, selectedPlanId, selectedDetail]
  );

  const selectedScale = selectedSummary?.scale_px_per_meter ?? 200;

  const imageUrl = React.useMemo(
    () => selectedPlanId
      ? `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(selectedPlanId)}/image`
      : "",
    [selectedPlanId]
  );
  const overlayUrl = React.useMemo(
    () => selectedPlanId
      ? getWorkshopOverlayUrl(selectedPlanId, { show_flooring: showFlooring, show_room_numbers: showRoomNumbers, highlight_walls: highlightWalls, version: overlayVersion })
      : "",
    [selectedPlanId, showFlooring, showRoomNumbers, highlightWalls, overlayVersion]
  );

  const roomRows = React.useMemo(() => {
    const meta = selectedDetail?.meta as Record<string, unknown> | undefined;
    if (!meta) return [];
    const safeN = (v: unknown): number | null => {
      if (typeof v === "number" && isFinite(v)) return v;
      if (typeof v === "string") { const n = Number(v); return isFinite(n) ? n : null; }
      if (v && typeof v === "object" && "value" in v) { const n = Number((v as { value?: unknown }).value); return isFinite(n) ? n : null; }
      return null;
    };
    // Prefer LLM-extracted rooms (vision, structured); fall back to CV-detected rooms
    const llmRooms = Array.isArray(meta.llm_rooms) ? meta.llm_rooms : [];
    if (llmRooms.length > 0) {
      return llmRooms.slice(0, 100).map((room, idx) => {
        const r = (room ?? {}) as Record<string, unknown>;
        return {
          id: idx + 1,
          name: (typeof r.name === "string" && r.name) || `חדר ${idx + 1}`,
          area: safeN(r.area_m2),
          ceiling: safeN(r.ceiling_height_m),
          floor_elev: safeN(r.elevation_floor_m),
          flooring: typeof r.flooring === "string" ? r.flooring : null,
          notes: typeof r.notes === "string" ? r.notes : null,
          isLlm: true,
        };
      });
    }
    // Fallback: CV-detected rooms
    const rooms = Array.isArray(meta.rooms) ? meta.rooms : [];
    return rooms.slice(0, 50).map((room, idx) => {
      const r = (room ?? {}) as Record<string, unknown>;
      return {
        id: idx + 1,
        name: (typeof r.room_name === "string" && r.room_name) || (typeof r.name === "string" && r.name) || `חדר ${idx + 1}`,
        area: safeN(r.area_m2),
        ceiling: null,
        floor_elev: null,
        flooring: null,
        notes: null,
        isLlm: false,
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
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} compact />
            <button
              type="button"
              onClick={() => void handleClearAll()}
              disabled={isLoading}
              title="מחק את כל התוכניות"
              style={{
                display: "flex", alignItems: "center", gap: 6,
                border: "1.5px solid #FECACA", borderRadius: 10,
                background: "#FFF5F5", padding: "8px 14px",
                cursor: "pointer", color: "#DC2626", fontSize: 13, fontWeight: 600,
                opacity: isLoading ? 0.5 : 1,
              }}
            >
              <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
              נקה מערכת
            </button>
          </div>
          {analysisStatus && uploadProgress === null && (
            <div style={{ fontSize: 13, color: "#1d4ed8", background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8, padding: "6px 14px" }}>{analysisStatus}</div>
          )}
        </div>
      )}

      {/* Upload progress bar */}
      {uploadProgress !== null && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ height: 6, background: "#E2E8F0", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ height: "100%", background: "var(--navy)", borderRadius: 99, width: `${uploadProgress}%`, transition: "width 0.5s" }} />
          </div>
          <div style={{ fontSize: 12, color: "#64748b" }}>{analysisStatus}</div>
        </div>
      )}

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {/* ── Plan list / skeleton ── */}
      {plansLoading && plans.length === 0 && <SkeletonGrid count={3} />}

      {plans.length > 0 && (
        <>
          <div className="section-divider">תוכניות קומה — {plans.length} קבצים</div>
          {plans.map((p) => {
            const active = p.id === selectedPlanId;
            const analyzed = p.total_wall_length_m != null && p.total_wall_length_m > 0;
            const badgeCls = active ? "badge-navy" : analyzed ? "badge-green" : "badge-amber";
            const badgeLabel = active ? "פעיל" : analyzed ? "נותח" : "ממתין";
            return (
              <div
                key={p.id}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedPlanId(p.id)}
                onKeyDown={(e) => { if (e.key === "Enter") setSelectedPlanId(p.id); }}
                className={`plan-item${active ? " selected" : ""}`}
              >
                {/* Thumb */}
                <div className="plan-thumb">
                  <img
                    src={`${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(p.id)}/image`}
                    alt=""
                    style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: 8 }}
                    onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                  />
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                </div>
                {/* Meta */}
                <div className="plan-meta">
                  <div className="plan-name">{p.plan_name}</div>
                  <div className="plan-sub">
                    {p.total_wall_length_m != null ? `${p.total_wall_length_m.toFixed(1)} מ' קירות` : "טרם נותח"}
                  </div>
                </div>
                {/* Actions */}
                <div className="plan-actions">
                  <span className={`badge ${badgeCls}`}>{badgeLabel}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedPlanId(p.id);
                      if (analyzed && onNavigatePlanning) onNavigatePlanning();
                      else if (!analyzed) void runAnalysisNow();
                    }}
                    className={`btn btn-sm ${analyzed ? "btn-orange" : "btn-ghost"}`}
                    style={{ marginTop: 4 }}
                  >
                    {analyzed ? "פתח ←" : "נתח"}
                  </button>
                </div>
              </div>
            );
          })}
        </>
      )}

      {/* ── Empty state ── */}
      {plans.length === 0 && !plansLoading && (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
          padding: "56px 24px", border: "2px dashed #E2E8F0", borderRadius: 16,
          background: "#FAFBFC", textAlign: "center",
        }}>
          <div style={{ color: "var(--text-3)" }}>
            <svg width={56} height={56} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
          </div>
          <div style={{ fontWeight: 700, fontSize: 16, color: "var(--navy)" }}>אין תוכניות בניה</div>
          <div style={{ fontSize: 13, color: "#94A3B8", maxWidth: 300 }}>
            העלה קובץ PDF של תוכנית בניה כדי להתחיל — המערכת תנתח קירות, ריצוף וחדרים אוטומטית.
          </div>
          <button
            type="button"
            onClick={() => document.getElementById("workshop-upload-input")?.click()}
            style={{
              marginTop: 4, padding: "10px 28px",
              background: "var(--navy)", color: "#fff",
              border: "none", borderRadius: 10,
              fontSize: 14, fontWeight: 700, cursor: "pointer",
            }}
          >
            העלה תוכנית PDF
          </button>
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
              style={{ padding: "7px 16px", border: "1.5px solid var(--navy)", color: "var(--navy)", borderRadius: 8, fontSize: 13, fontWeight: 600, background: "#fff", cursor: "pointer", opacity: isLoading ? 0.5 : 1 }}
            >
              שמור
            </button>
            <button
              type="button"
              onClick={() => void runAnalysisNow()}
              disabled={isLoading}
              style={{ padding: "7px 16px", background: "var(--navy)", color: "#fff", borderRadius: 8, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer", opacity: isLoading ? 0.5 : 1 }}
            >
              ניתוח
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
            {([
              { label: "קנ\"מ", value: `${selectedScale.toFixed(0)} px/מ'`, accent: "var(--navy)" },
              { label: "קירות", value: `${selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'`, accent: "var(--orange)" },
              { label: "ריצוף", value: `${selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר`, accent: "var(--green)" },
              { label: "עלות משוערת", value: `${totalQuote.toLocaleString()} ₪`, accent: "var(--amber)" },
            ] as { label: string; value: string; accent: string }[]).map((c) => (
              <div key={c.label} style={{ background: "#fff", border: "1px solid #E2E8F0", borderTop: `3px solid ${c.accent}`, borderRadius: 12, padding: "14px 14px", boxShadow: "var(--sh1)" }}>
                <div style={{ fontSize: 10, color: "var(--s400)", textTransform: "uppercase", letterSpacing: ".5px", marginBottom: 6 }}>{c.label}</div>
                <div style={{ fontWeight: 800, fontSize: "1.4rem", color: "var(--s900)", lineHeight: 1 }}>{c.value}</div>
              </div>
            ))}
          </div>

          {/* Canvas */}
          <ZoomCanvas
            imageUrl={imageUrl}
            overlayUrl={overlayUrl}
            overlayLoading={overlayLoading}
            onOverlayLoad={() => setOverlayLoading(false)}
          />

          {/* Tabs */}
          <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 14, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.05)" }}>
            <div style={{ display: "flex", borderBottom: "1px solid #E2E8F0", background: "#F8FAFC", overflowX: "auto" }}>
              {[
                { id: "overview", label: "סקירה" },
                { id: "rooms", label: "חדרים" },
                { id: "cost", label: "עלויות" },
                { id: "diagnostics", label: "בדיקות" },
              ].map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setActiveTab(t.id as WorkshopTab)}
                  style={{
                    padding: "12px 20px",
                    border: "none",
                    height: 44,
                    borderBottom: activeTab === t.id ? "3px solid var(--orange)" : "3px solid transparent",
                    background: activeTab === t.id ? "#fff" : "transparent",
                    color: activeTab === t.id ? "var(--navy)" : "#64748b",
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
                    <div style={{ fontWeight: 700, marginBottom: 8, color: "var(--navy)" }}>פרטי תוכנית</div>
                    <div>שם: {planDisplayName || "ללא שם"}</div>
                    <div>קנ"מ: {scaleText}</div>
                    <div>מחושב: {selectedScale.toFixed(2)} px/מ'</div>
                  </div>
                  <div style={{ background: "#F8FAFC", borderRadius: 10, padding: 14 }}>
                    <div style={{ fontWeight: 700, marginBottom: 8, color: "var(--navy)" }}>תוצרים</div>
                    <div>קירות: {selectedSummary?.total_wall_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בטון: {selectedSummary?.concrete_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>בלוקים: {selectedSummary?.blocks_length_m?.toFixed(2) ?? "—"} מ'</div>
                    <div>ריצוף: {selectedSummary?.flooring_area_m2?.toFixed(2) ?? "—"} מ"ר</div>
                  </div>
                </div>
              )}
              {activeTab === "rooms" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {roomRows.length > 0 && roomRows[0].isLlm && (
                    <div style={{ fontSize: 11, color: "#059669", background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: 8, padding: "5px 10px", display: "inline-flex", alignItems: "center", gap: 5 }}>
                      נתונים מתוכנית AI — {roomRows.length} חדרים · סה"כ {roomRows.reduce((s, r) => s + (r.area ?? 0), 0).toFixed(1)} מ"ר
                    </div>
                  )}
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse", minWidth: 520 }}>
                      <thead>
                        <tr style={{ color: "#94A3B8", borderBottom: "2px solid #E2E8F0", background: "#F8FAFC" }}>
                          <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>#</th>
                          <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>שם חדר</th>
                          <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>שטח (מ"ר)</th>
                          {roomRows[0]?.isLlm && <>
                            <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>תקרה (מ')</th>
                            <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>ריצוף</th>
                            <th style={{ textAlign: "right", padding: "7px 8px", fontWeight: 600 }}>הערות</th>
                          </>}
                        </tr>
                      </thead>
                      <tbody>
                        {roomRows.length === 0 && (
                          <tr><td style={{ padding: 12, color: "#94A3B8" }} colSpan={6}>אין נתוני חדרים. הרץ ניתוח קודם.</td></tr>
                        )}
                        {roomRows.map((row) => (
                          <tr key={row.id} style={{ borderBottom: "1px solid #F1F5F9" }}>
                            <td style={{ padding: "6px 8px", color: "#94A3B8" }}>{row.id}</td>
                            <td style={{ padding: "6px 8px", fontWeight: 600, color: "var(--navy)" }}>{row.name}</td>
                            <td style={{ padding: "6px 8px" }}>{row.area != null ? row.area.toFixed(1) : "—"}</td>
                            {row.isLlm && <>
                              <td style={{ padding: "6px 8px", color: "#64748b" }}>{row.ceiling != null ? row.ceiling.toFixed(2) : "—"}</td>
                              <td style={{ padding: "6px 8px", color: "#64748b", fontSize: 12 }}>{row.flooring ?? "—"}</td>
                              <td style={{ padding: "6px 8px", color: "#94A3B8", fontSize: 11, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.notes ?? ""}</td>
                            </>}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
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
                        <div style={{ fontWeight: 700, color: "var(--navy)", marginTop: 6 }}>{(length * price).toLocaleString()} ₪</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ borderTop: "1px solid #E2E8F0", paddingTop: 12, fontWeight: 700, fontSize: 15, color: "var(--navy)" }}>
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
                            {String(k)}: <span style={{ fontWeight: 700, color: v ? "var(--green)" : k === "llm_rooms" ? "var(--amber)" : "var(--red)" }}>{v ? "עבר" : k === "llm_rooms" ? "חסר" : "נכשל"}</span>
                          </div>
                        ))}
                      </div>
                      {readiness.issues.length > 0 ? (
                        <ul style={{ paddingRight: 20, color: "#B45309" }}>
                          {readiness.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                        </ul>
                      ) : (
                        <div style={{ color: "#15803D", fontWeight: 600 }}>כל הבדיקות עברו.</div>
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
