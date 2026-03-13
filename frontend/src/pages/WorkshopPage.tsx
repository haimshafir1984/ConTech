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

// ג”€ג”€ Zoom-able canvas wrapper ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
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

  // ג”€ג”€ Touch support (pan + pinch-to-zoom) ג”€ג”€
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
        <button type="button" onClick={resetView} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-[10px] hover:bg-slate-50">אפס</button>
        <button type="button" onClick={() => setZoom((z) => clampZoom(z * 0.8))} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50">גˆ’</button>
      </div>
      <div className="absolute bottom-3 right-3 bg-black/40 text-white text-[11px] px-2 py-0.5 rounded">
        {Math.round(zoom * 100)}%
      </div>
    </div>
  );
};

// ג”€ג”€ UploadZone ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
interface UploadZoneProps {
  onFile: (f: File) => void;
  isLoading: boolean;
  compact?: boolean;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onFile, isLoading, compact }) => {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
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
        <input ref={inputRef} type="file" accept="application/pdf" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); if (inputRef.current) inputRef.current.value = ""; }} />
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
      setError("לא ניתן לטעון את רשימת התוכניות. ודא שה-backend המקומי רץ על http://localhost:8000.");
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
        setError("נתוני התוכנית אינם זמינים כרגע. ייתכן שהשרת עלה מחדש, ולכן צריך להעלות את קובץ ה-PDF שוב.");
      } else {
        setError("לא ניתן לטעון את נתוני סדנת העבודה עבור התוכנית שנבחרה.");
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
      if (parts.length) bestName = parts.join(" - ");
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
        setError('נתוני התוכנית אינם זמינים כרגע. ייתכן שהשרת עלה מחדש, ולכן צריך להעלות את קובץ ה-PDF שוב.');
      } else {
        const msg = axios.isAxiosError(e) ? ((e.response?.data as { detail?: string })?.detail || e.message) : e instanceof Error ? e.message : "שגיאה";
        setError(`שגיאה בשמירת קנ"מ: ${msg}`);
      }
    } finally { setIsLoading(false); }
  };

  const handleClearAll = async () => {
    if (!window.confirm("האם למחוק את כל התוכניות? הפעולה הזו אינה הפיכה.")) return;
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
      setError("לא ניתן למחוק את התוכניות כרגע.");
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
            setAnalysisStatus("הקובץ נותח בשרת ונטען בהצלחה.");
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
      setAnalysisStatus("הניתוח הסתיים.");
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
            setAnalysisStatus("הניתוח הושלם בשרת.");
            setOverlayLoading(true);
            setOverlayVersion((v) => v + 1);
            return;
          }
        } catch { /* fall through */ }
      }
      if (_isRestartLost(e)) {
        setError("נתוני התוכנית אינם זמינים כרגע. ייתכן שהשרת עלה מחדש, ולכן צריך להעלות את קובץ ה-PDF שוב.");
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
    <div style={{ display: "grid", gridTemplateColumns: hasPlan ? "minmax(0,1fr) 250px" : "1fr", gap: 14, alignItems: "start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
        <section className="pro-panel" style={{ padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", flexWrap: "wrap", marginBottom: 14 }}>
            <div>
              <div className="pro-section-title">Workshop</div>
              <h1 style={{ margin: "4px 0 0", fontSize: 20, lineHeight: 1.1, fontWeight: 800, color: "#0F172A" }}>ספריית תוכניות</h1>
              <p style={{ margin: "6px 0 0", fontSize: 12, color: "#64748B" }}>קליטה, ניתוח וסקירת מוכנות לפני מעבר להגדרת תכולה.</p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              {plans.length > 0 && (
                <>
                  <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} compact />
                  <button
                    type="button"
                    onClick={() => void handleClearAll()}
                    disabled={isLoading}
                    className="pro-btn pro-btn-soft"
                    style={{ color: "#B91C1C", borderColor: "#FECACA", background: "#FEF2F2", opacity: isLoading ? 0.5 : 1 }}
                  >
                    נקה מערכת
                  </button>
                </>
              )}
            </div>
          </div>

          {plans.length === 0 ? (
            <UploadZone onFile={(f) => void handleUpload(f)} isLoading={isLoading} />
          ) : (
            <>
              {analysisStatus && uploadProgress === null && (
                <div style={{ marginBottom: 10, border: "1px solid #BFDBFE", background: "#EFF6FF", color: "#1D4ED8", borderRadius: 10, padding: "8px 10px", fontSize: 12, fontWeight: 600 }}>
                  {analysisStatus}
                </div>
              )}
              {uploadProgress !== null && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
                  <div style={{ height: 6, background: "#E2E8F0", borderRadius: 999, overflow: "hidden" }}>
                    <div style={{ width: `${uploadProgress}%`, height: "100%", background: "#111827", borderRadius: 999, transition: "width 0.35s" }} />
                  </div>
                  <div style={{ fontSize: 11, color: "#64748B" }}>{analysisStatus}</div>
                </div>
              )}
              {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
              {plansLoading && plans.length === 0 && <SkeletonGrid count={3} />}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
                {plans.map((p) => {
                  const active = p.id === selectedPlanId;
                  const analyzed = p.total_wall_length_m != null && p.total_wall_length_m > 0;
                  const statusColor = active ? "#2563EB" : analyzed ? "#16A34A" : "#F59E0B";
                  const statusLabel = active ? "נבחר" : analyzed ? "מוכן" : "ממתין";
                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setSelectedPlanId(p.id)}
                      style={{
                        textAlign: "right",
                        border: `1px solid ${active ? "#94A3B8" : "#D7DEE8"}`,
                        background: active ? "#F8FAFC" : "#FFFFFF",
                        borderRadius: 14,
                        overflow: "hidden",
                        cursor: "pointer",
                        padding: 0,
                        boxShadow: active ? "0 10px 28px rgba(37,99,235,0.08)" : "0 6px 20px rgba(15,23,42,0.04)",
                      }}
                    >
                      <div style={{ height: 132, background: "linear-gradient(135deg, #CBD5E1, #F8FAFC)", position: "relative", borderBottom: "1px solid #E2E8F0" }}>
                        <img
                          src={`${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(p.id)}/image`}
                          alt=""
                          style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                        />
                        <div style={{ position: "absolute", top: 10, right: 10, display: "inline-flex", alignItems: "center", gap: 6, background: "rgba(255,255,255,0.92)", border: "1px solid #E2E8F0", borderRadius: 999, padding: "4px 9px", fontSize: 11, fontWeight: 700, color: "#334155" }}>
                          <span className="pro-status-dot" style={{ background: statusColor }} />
                          {statusLabel}
                        </div>
                      </div>
                      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#0F172A", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.plan_name}</div>
                          <div style={{ marginTop: 4, fontSize: 11, color: "#64748B" }}>{p.total_wall_length_m != null ? `${p.total_wall_length_m.toFixed(1)} מ' קירות` : "טרם נותח"}</div>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 11, color: "#475569" }}>
                          <div style={{ border: "1px solid #E2E8F0", borderRadius: 10, padding: "8px 9px", background: "#F8FAFC" }}>
                            <div style={{ color: "#94A3B8" }}>בטון</div>
                            <div style={{ marginTop: 3, fontWeight: 700, color: "#0F172A" }}>{p.concrete_length_m?.toFixed(1) ?? "-"}</div>
                          </div>
                          <div style={{ border: "1px solid #E2E8F0", borderRadius: 10, padding: "8px 9px", background: "#F8FAFC" }}>
                            <div style={{ color: "#94A3B8" }}>ריצוף</div>
                            <div style={{ marginTop: 3, fontWeight: 700, color: "#0F172A" }}>{p.flooring_area_m2?.toFixed(1) ?? "-"}</div>
                          </div>
                        </div>
                        <div style={{ display: "flex", gap: 8 }}>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedPlanId(p.id);
                              if (analyzed && onNavigatePlanning) onNavigatePlanning();
                              else if (!analyzed) void runAnalysisNow();
                            }}
                            className="pro-btn"
                            style={{ flex: 1, background: analyzed ? "#111827" : "#F8FAFC", color: analyzed ? "#FFFFFF" : "#0F172A", border: analyzed ? "1px solid #111827" : "1px solid #CBD5E1" }}
                          >
                            {analyzed ? "פתח לתכולה" : "נתח עכשיו"}
                          </button>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </section>

        {hasPlan && (
          <section className="pro-panel" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "end", justifyContent: "space-between" }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", flex: 1, minWidth: 0 }}>
                <label style={{ minWidth: 180, flex: 1 }}>
                  <div className="pro-section-title" style={{ marginBottom: 6 }}>שם תוכנית</div>
                  <input className="pro-field" value={planDisplayName} onChange={(e) => setPlanDisplayName(e.target.value)} placeholder="שם תוכנית" />
                </label>
                <label style={{ width: 110 }}>
                  <div className="pro-section-title" style={{ marginBottom: 6 }}>קנ"מ</div>
                  <select className="pro-field" value={scaleText} onChange={(e) => setScaleText(e.target.value)}>
                    {["1:20", "1:25", "1:50", "1:75", "1:100", "1:200"].map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button type="button" onClick={() => void savePlanSettings()} disabled={isLoading} className="pro-btn pro-btn-soft" style={{ opacity: isLoading ? 0.5 : 1 }}>שמור</button>
                <button type="button" onClick={() => void runAnalysisNow()} disabled={isLoading} className="pro-btn pro-btn-dark" style={{ opacity: isLoading ? 0.5 : 1 }}>נתח</button>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(132px, 1fr))", gap: 8 }}>
              {[
                { label: "קנ\"מ", value: `${selectedScale.toFixed(0)} px/m` },
                { label: "קירות", value: `${selectedSummary?.total_wall_length_m?.toFixed(2) ?? "-"} מ'` },
                { label: "ריצוף", value: `${selectedSummary?.flooring_area_m2?.toFixed(2) ?? "-"} מ"ר` },
                { label: "עלות", value: `${totalQuote.toLocaleString()} ₪` },
              ].map((item) => (
                <div key={item.label} style={{ border: "1px solid #E2E8F0", borderRadius: 12, background: "#F8FAFC", padding: "10px 12px" }}>
                  <div style={{ fontSize: 10, color: "#94A3B8", marginBottom: 4 }}>{item.label}</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: "#0F172A" }}>{item.value}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
              {[
                { label: "ריצוף", val: showFlooring, set: setShowFlooring },
                { label: "מספרי חדרים", val: showRoomNumbers, set: setShowRoomNumbers },
                { label: "הדגש קירות", val: highlightWalls, set: setHighlightWalls },
              ].map(({ label, val, set }) => (
                <label key={label} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "#475569", cursor: "pointer" }}>
                  <input type="checkbox" checked={val} onChange={(e) => set(e.target.checked)} />
                  {label}
                </label>
              ))}
            </div>

            <ZoomCanvas imageUrl={imageUrl} overlayUrl={overlayUrl} overlayLoading={overlayLoading} onOverlayLoad={() => setOverlayLoading(false)} />

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
              <div style={{ border: "1px solid #E2E8F0", borderRadius: 12, background: "#FFFFFF", padding: 12 }}>
                <div className="pro-section-title" style={{ marginBottom: 8 }}>חדרים</div>
                <div style={{ maxHeight: 180, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
                  {roomRows.length === 0 ? (
                    <div style={{ fontSize: 12, color: "#94A3B8" }}>אין נתוני חדרים עדיין.</div>
                  ) : roomRows.slice(0, 10).map((room) => (
                    <div key={room.id} style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 12, paddingBottom: 6, borderBottom: "1px solid #F1F5F9" }}>
                      <span style={{ color: "#0F172A", fontWeight: 600 }}>{room.name}</span>
                      <span style={{ color: "#64748B" }}>{room.area != null ? `${room.area.toFixed(1)} מ"ר` : "-"}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ border: "1px solid #E2E8F0", borderRadius: 12, background: "#FFFFFF", padding: 12 }}>
                <div className="pro-section-title" style={{ marginBottom: 8 }}>אומדן מהיר</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { label: "בטון", length: selectedSummary?.concrete_length_m ?? 0, price: concretePrice, set: setConcretePrice, unit: "מ'" },
                    { label: "בלוקים", length: selectedSummary?.blocks_length_m ?? 0, price: blocksPrice, set: setBlocksPrice, unit: "מ'" },
                    { label: "ריצוף", length: selectedSummary?.flooring_area_m2 ?? 0, price: floorPrice, set: setFloorPrice, unit: 'מ"ר' },
                  ].map(({ label, length, price, set, unit }) => (
                    <label key={label} style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 8, alignItems: "center", fontSize: 12 }}>
                      <span style={{ fontWeight: 600, color: "#0F172A" }}>{label}</span>
                      <span style={{ color: "#64748B", whiteSpace: "nowrap" }}>{length.toFixed(2)} {unit}</span>
                      <input type="number" value={price} onChange={(e) => set(Number(e.target.value))} style={{ width: 84, height: 30, border: "1px solid #CBD5E1", borderRadius: 8, padding: "0 8px", fontSize: 12 }} />
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}
      </div>

      {hasPlan && (
        <aside className="pro-panel" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14, position: "sticky", top: 62 }}>
          <div>
            <div className="pro-section-title">Selected Plan</div>
            <div style={{ marginTop: 4, fontSize: 16, fontWeight: 800, color: "#0F172A", lineHeight: 1.2 }}>{(selectedSummary?.plan_name ?? planDisplayName) || "תוכנית"}</div>
            <div style={{ marginTop: 4, fontSize: 11, color: "#64748B" }}>{selectedDetail?.summary.filename ?? ""}</div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[
              ["סטטוס", selectedSummary?.total_wall_length_m ? "מנותח" : "ממתין"],
              ["קנ\"מ", scaleText],
              ["קירות", `${selectedSummary?.total_wall_length_m?.toFixed(2) ?? "-"} מ'`],
              ["בטון", `${selectedSummary?.concrete_length_m?.toFixed(2) ?? "-"} מ'`],
              ["בלוקים", `${selectedSummary?.blocks_length_m?.toFixed(2) ?? "-"} מ'`],
              ["ריצוף", `${selectedSummary?.flooring_area_m2?.toFixed(2) ?? "-"} מ"ר`],
            ].map(([label, value]) => (
              <div key={String(label)} style={{ display: "flex", justifyContent: "space-between", gap: 8, paddingBottom: 8, borderBottom: "1px solid #F1F5F9", fontSize: 12 }}>
                <span style={{ color: "#64748B" }}>{label}</span>
                <span style={{ color: "#0F172A", fontWeight: 700, textAlign: "left" }}>{value}</span>
              </div>
            ))}
          </div>

          <div>
            <div className="pro-section-title" style={{ marginBottom: 10 }}>Diagnostics</div>
            {!readiness ? (
              <div style={{ fontSize: 12, color: "#94A3B8" }}>אין נתוני בדיקה.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  ["Original", readiness.has_original],
                  ["Thick Walls", readiness.has_thick_walls],
                  ["Floor Mask", readiness.has_flooring_mask],
                  ["Scale", readiness.has_scale_px_per_meter],
                  ["Meters / Px", readiness.has_meters_per_pixel],
                  ["LLM Rooms", readiness.has_llm_rooms],
                ].map(([label, value]) => (
                  <div key={String(label)} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontSize: 12 }}>
                    <span style={{ color: "#334155" }}>{label}</span>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: value ? "#15803D" : "#B45309", fontWeight: 700 }}>
                      <span className="pro-status-dot" style={{ background: value ? "#16A34A" : "#F59E0B" }} />
                      {value ? "OK" : "Missing"}
                    </span>
                  </div>
                ))}
                {readiness.issues.length > 0 && (
                  <div style={{ marginTop: 6, borderTop: "1px solid #F1F5F9", paddingTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                    {readiness.issues.map((issue, i) => (
                      <div key={i} style={{ fontSize: 11, color: "#92400E", lineHeight: 1.45 }}>{issue}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          <button type="button" onClick={() => onNavigatePlanning?.()} disabled={!selectedSummary?.total_wall_length_m} className="pro-btn pro-btn-dark" style={{ width: "100%", opacity: selectedSummary?.total_wall_length_m ? 1 : 0.45, cursor: selectedSummary?.total_wall_length_m ? "pointer" : "not-allowed" }}>
            עבור להגדרת תכולה
          </button>
        </aside>
      )}
    </div>
  );
};


