import React from "react";
import { useToast } from "../components/Toast";
import { useConfirm } from "../components/ConfirmDialog";
import { apiClient } from "../api/client";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  createWorkerReport,
  listWorkerReports,
  measureWorkerItem,
  type WorkerMeasuredItem,
  type WorkerReport
} from "../api/workerApi";
import { getPlanningState, type WorkSection } from "../api/planningApi";

type DrawMode = "line" | "rect" | "path";
type Point = { x: number; y: number };
type WorkerTab = "map" | "notes" | "history";

// ─── Printable progress report ───────────────────────────────────────────────
function printProgressReport(reports: WorkerReport[], planName: string) {
  const rows = reports.map((r) => `
    <tr>
      <td>${r.date}</td>
      <td>${r.shift}</td>
      <td>${r.report_type === "walls" ? "קירות" : "ריצוף/חיפוי"}</td>
      <td>${r.report_type === "walls" ? r.total_length_m.toFixed(2) + " מ'" : r.total_area_m2.toFixed(2) + " מ\"ר"}</td>
      <td>${r.note || "—"}</td>
    </tr>`).join("");

  const totalWalls = reports.filter(r => r.report_type === "walls").reduce((s, r) => s + r.total_length_m, 0);
  const totalFloor = reports.filter(r => r.report_type === "floor").reduce((s, r) => s + r.total_area_m2, 0);

  const html = `<!doctype html><html dir="rtl" lang="he"><head>
    <meta charset="UTF-8"><title>דוח התקדמות - ${planName}</title>
    <style>
      body { font-family: Arial, sans-serif; padding: 24px; direction: rtl; }
      h1 { color: #FF4B4B; } table { border-collapse: collapse; width: 100%; margin-top: 16px; }
      th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
      th { background: #f5f5f5; } .totals { margin-top: 16px; font-weight: bold; }
    </style></head><body>
    <h1>📊 דוח התקדמות עובד</h1>
    <p>פרויקט: <b>${planName}</b> | הופק: ${new Date().toLocaleDateString("he-IL")}</p>
    <table><thead><tr><th>תאריך</th><th>משמרת</th><th>סוג</th><th>כמות</th><th>הערה</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="totals">
      <div>סה"כ קירות: ${totalWalls.toFixed(2)} מ'</div>
      <div>סה"כ ריצוף: ${totalFloor.toFixed(2)} מ"ר</div>
    </div>
  </body></html>`;

  const w = window.open("", "_blank");
  if (w) { w.document.write(html); w.document.close(); w.print(); }
}

// ─── Zoom canvas for worker drawing ──────────────────────────────────────────
interface WorkerCanvasProps {
  imageUrl: string;
  items: WorkerMeasuredItem[];
  drawMode: DrawMode;
  sections?: WorkSection[];
  activeSectionUid?: string;
  onDrawComplete: (raw: { object_type: DrawMode; raw_object: Record<string, unknown> }) => void;
}

const WorkerCanvas: React.FC<WorkerCanvasProps> = ({ imageUrl, items, drawMode, sections = [], activeSectionUid = "", onDrawComplete }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const imgRef = React.useRef<HTMLImageElement>(null);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const isPanningRef = React.useRef(false);
  const isDrawingRef = React.useRef(false);
  const lastMouse = React.useRef({ x: 0, y: 0 });
  const panRef = React.useRef({ x: 0, y: 0 });
  const zoomRef = React.useRef(1);
  const [startPt, setStartPt] = React.useState<Point | null>(null);
  const [tempPt, setTempPt] = React.useState<Point | null>(null);
  const [pathPts, setPathPts] = React.useState<Point[]>([]);
  const [imgSize, setImgSize] = React.useState({ w: 0, h: 0 });
  const [renderZoom, setRenderZoom] = React.useState(1);
  const [renderPan, setRenderPan] = React.useState({ x: 0, y: 0 });
  const [isDrawingState, setIsDrawingState] = React.useState(false);
  const [isPanningState, setIsPanningState] = React.useState(false);

  const clampZ = (z: number) => Math.min(8, Math.max(0.25, z));

  const setZoomSync = (z: number) => { zoomRef.current = z; setZoom(z); setRenderZoom(z); };
  const setPanSync = (p: { x: number; y: number }) => { panRef.current = p; setPan(p); setRenderPan(p); };

  const toCanvasPoint = React.useCallback((clientX: number, clientY: number): Point => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    const z = zoomRef.current;
    const p = panRef.current;
    return {
      x: Math.max(0, (clientX - rect.left - p.x) / z),
      y: Math.max(0, (clientY - rect.top - p.y) / z),
    };
  }, []);

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      if (isDrawingRef.current) return;
      const rect = el.getBoundingClientRect();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const oldZ = zoomRef.current;
      const newZ = clampZ(oldZ * factor);
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const newP = {
        x: mx - (mx - panRef.current.x) * (newZ / oldZ),
        y: my - (my - panRef.current.y) * (newZ / oldZ),
      };
      setZoomSync(newZ);
      setPanSync(newP);
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    if (!imageUrl) return;
    if (e.altKey || e.metaKey) {
      isPanningRef.current = true;
      setIsPanningState(true);
      lastMouse.current = { x: e.clientX, y: e.clientY };
      return;
    }
    isDrawingRef.current = true;
    setIsDrawingState(true);
    const p = toCanvasPoint(e.clientX, e.clientY);
    setStartPt(p);
    setTempPt(p);
    if (drawMode === "path") setPathPts([p]);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (isPanningRef.current) {
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      lastMouse.current = { x: e.clientX, y: e.clientY };
      setPanSync({ x: panRef.current.x + dx, y: panRef.current.y + dy });
      return;
    }
    if (!isDrawingRef.current) return;
    const p = toCanvasPoint(e.clientX, e.clientY);
    setTempPt(p);
    if (drawMode === "path") setPathPts((prev) => [...prev, p]);
  };

  const finishDraw = React.useCallback((currentStartPt: Point | null, currentTempPt: Point | null, currentPathPts: Point[]) => {
    if (!isDrawingRef.current || !currentStartPt || !currentTempPt) {
      isDrawingRef.current = false;
      setIsDrawingState(false);
      return;
    }
    isDrawingRef.current = false;
    setIsDrawingState(false);
    const dx = currentTempPt.x - currentStartPt.x;
    const dy = currentTempPt.y - currentStartPt.y;
    if (drawMode !== "path" && Math.sqrt(dx * dx + dy * dy) < 5) return;

    let raw_object: Record<string, unknown> = {};
    if (drawMode === "line") {
      raw_object = { x1: currentStartPt.x, y1: currentStartPt.y, x2: currentTempPt.x, y2: currentTempPt.y };
    } else if (drawMode === "rect") {
      raw_object = {
        x: Math.min(currentStartPt.x, currentTempPt.x), y: Math.min(currentStartPt.y, currentTempPt.y),
        width: Math.abs(currentTempPt.x - currentStartPt.x), height: Math.abs(currentTempPt.y - currentStartPt.y),
      };
    } else {
      raw_object = { points: currentPathPts.map((p) => [p.x, p.y]) };
    }
    onDrawComplete({ object_type: drawMode, raw_object });
    setStartPt(null); setTempPt(null); setPathPts([]);
  }, [drawMode, onDrawComplete]);

  const onMouseUp = React.useCallback((e?: React.MouseEvent) => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      setIsPanningState(false);
      return;
    }
    setStartPt(sp => {
      setTempPt(tp => {
        setPathPts(pp => {
          finishDraw(sp, tp, pp);
          return [];
        });
        return null;
      });
      return null;
    });
    void e;
  }, [finishDraw]);

  return (
    <div style={{ position: "relative", background: "#F1F5F9", borderRadius: 12, overflow: "hidden", border: "1px solid #E2E8F0", minHeight: 420 }}>
      <div
        ref={containerRef}
        style={{ width: "100%", overflow: "hidden", userSelect: "none", cursor: isDrawingState ? "crosshair" : isPanningState ? "grabbing" : "crosshair", minHeight: 420 }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          if (isPanningRef.current) { isPanningRef.current = false; setIsPanningState(false); }
        }}
      >
        <div style={{
          transform: `translate(${renderPan.x}px,${renderPan.y}px) scale(${renderZoom})`,
          transformOrigin: "top left",
          display: "inline-block",
          position: "relative"
        }}>
          {imageUrl ? (
            <img
              ref={imgRef}
              src={imageUrl}
              alt="plan"
              draggable={false}
              style={{ display: "block", userSelect: "none" }}
              onLoad={(e) => {
                const img = e.currentTarget;
                setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
              }}
            />
          ) : (
            <div style={{ width: 760, height: 420, display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8", fontSize: 14 }}>
              בחר תוכנית כדי להתחיל סימון
            </div>
          )}

          {imgSize.w > 0 && (
            <svg width={imgSize.w} height={imgSize.h} style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
              {sections.map((sec) => {
                if (sec.width < 2 || sec.height < 2) return null;
                const isActive = sec.uid === activeSectionUid;
                return (
                  <g key={sec.uid}>
                    <rect x={sec.x} y={sec.y} width={sec.width} height={sec.height}
                      fill={isActive ? `${sec.color}30` : `${sec.color}10`}
                      stroke={sec.color}
                      strokeWidth={isActive ? 3 / zoom : 1.5 / zoom}
                      strokeDasharray={isActive ? undefined : "6 4"}
                      rx={2}
                    />
                    <rect x={sec.x} y={sec.y} width={Math.min(sec.width, 200)} height={18 / zoom} fill={sec.color} rx={2} />
                    <text x={sec.x + 4} y={sec.y + 13 / zoom} fill="white" fontSize={11 / zoom} fontWeight="bold">
                      {sec.name || "גזרה"} | {sec.worker || sec.contractor || ""}
                    </text>
                  </g>
                );
              })}

              {items.map((item) => {
                const obj = item.raw_object;
                const label = item.unit === "m" ? `${item.measurement.toFixed(2)} מ'` : `${item.measurement.toFixed(2)} מ"ר`;
                if (item.type === "line") {
                  return (
                    <g key={item.uid}>
                      <line x1={Number(obj.x1)} y1={Number(obj.y1)} x2={Number(obj.x2)} y2={Number(obj.y2)} stroke="#22d3ee" strokeWidth={2/zoom} />
                      <text x={(Number(obj.x1)+Number(obj.x2))/2} y={(Number(obj.y1)+Number(obj.y2))/2 - 4/zoom} fill="#0891b2" fontSize={12/zoom} textAnchor="middle">{label}</text>
                    </g>
                  );
                }
                if (item.type === "rect") {
                  return (
                    <g key={item.uid}>
                      <rect x={Number(obj.x)} y={Number(obj.y)} width={Number(obj.width)} height={Number(obj.height)} fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth={2/zoom} />
                      <text x={Number(obj.x)+Number(obj.width)/2} y={Number(obj.y)+Number(obj.height)/2} fill="#1d4ed8" fontSize={12/zoom} textAnchor="middle">{label}</text>
                    </g>
                  );
                }
                const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map((p) => `${p[0]},${p[1]}`).join(" ") : "";
                return <polyline key={item.uid} points={pts} fill="none" stroke="#f59e0b" strokeWidth={2/zoom} />;
              })}

              {isDrawingState && startPt && tempPt && drawMode === "line" && (
                <line x1={startPt.x} y1={startPt.y} x2={tempPt.x} y2={tempPt.y} stroke="#22c55e" strokeWidth={2/renderZoom} strokeDasharray="4" />
              )}
              {isDrawingState && startPt && tempPt && drawMode === "rect" && (
                <rect
                  x={Math.min(startPt.x, tempPt.x)} y={Math.min(startPt.y, tempPt.y)}
                  width={Math.abs(tempPt.x - startPt.x)} height={Math.abs(tempPt.y - startPt.y)}
                  fill="rgba(34,197,94,0.15)" stroke="#22c55e" strokeWidth={2/renderZoom} strokeDasharray="4"
                />
              )}
              {isDrawingState && drawMode === "path" && pathPts.length > 1 && (
                <polyline points={pathPts.map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#22c55e" strokeWidth={2/renderZoom} />
              )}
            </svg>
          )}
        </div>
      </div>

      {/* Zoom controls */}
      <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", flexDirection: "column", gap: 4, zIndex: 10 }}>
        <button type="button" onClick={() => setZoomSync(clampZ(zoomRef.current * 1.25))} style={zoomBtnStyle}>+</button>
        <button type="button" onClick={() => { setZoomSync(1); setPanSync({ x: 0, y: 0 }); }} style={zoomBtnStyle}>↺</button>
        <button type="button" onClick={() => setZoomSync(clampZ(zoomRef.current * 0.8))} style={zoomBtnStyle}>−</button>
      </div>
      <div style={{ position: "absolute", bottom: 12, right: 12, background: "rgba(0,0,0,0.4)", color: "#fff", fontSize: 11, padding: "2px 8px", borderRadius: 6 }}>{Math.round(renderZoom * 100)}%</div>
      <div style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.4)", color: "#fff", fontSize: 10, padding: "2px 8px", borderRadius: 6 }}>גלגלת=זום | Alt+גרור=הזזה</div>
    </div>
  );
};

const zoomBtnStyle: React.CSSProperties = {
  width: 32, height: 32, background: "#fff", border: "1px solid #CBD5E1",
  borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.12)", fontSize: 16,
  fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
};

// ─── Main WorkerPage ──────────────────────────────────────────────────────────
export const WorkerPage: React.FC = () => {
  const toast = useToast();
  const confirm = useConfirm();
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [reportType, setReportType] = React.useState<"walls" | "floor">("walls");
  const [drawMode, setDrawMode] = React.useState<DrawMode>("line");
  const [shift, setShift] = React.useState("בוקר");
  const [reportDate, setReportDate] = React.useState(() => new Date().toISOString().slice(0, 10));
  const [note, setNote] = React.useState("");
  const [items, setItems] = React.useState<WorkerMeasuredItem[]>([]);
  const [reports, setReports] = React.useState<WorkerReport[]>([]);
  const [error, setError] = React.useState("");
  const [saving, setSaving] = React.useState(false);
  const [measuring, setMeasuring] = React.useState(false);
  const [sections, setSections] = React.useState<WorkSection[]>([]);
  const [selectedSectionUid, setSelectedSectionUid] = React.useState<string>("");
  const [activeTab, setActiveTab] = React.useState<WorkerTab>("map");

  const selectedPlan = plans.find((p) => p.id === selectedPlanId) ?? null;
  const imageUrl = selectedPlanId
    ? `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(selectedPlanId)}/image`
    : "";

  const loadPlans = React.useCallback(async () => {
    const list = await listWorkshopPlans();
    setPlans(list);
    if (!selectedPlanId && list.length > 0) setSelectedPlanId(list[0].id);
  }, [selectedPlanId]);

  const loadReports = React.useCallback(async () => {
    if (!selectedPlanId) return;
    const data = await listWorkerReports(selectedPlanId);
    setReports(data);
  }, [selectedPlanId]);

  React.useEffect(() => { void loadPlans().catch(console.error); }, [loadPlans]);
  React.useEffect(() => { void loadReports().catch(console.error); }, [loadReports]);

  React.useEffect(() => {
    if (!selectedPlanId) { setSections([]); setSelectedSectionUid(""); return; }
    getPlanningState(selectedPlanId)
      .then(state => {
        setSections(state.sections ?? []);
        setSelectedSectionUid("");
      })
      .catch(() => setSections([]));
  }, [selectedPlanId]);

  const handleDrawComplete = async (payload: { object_type: DrawMode; raw_object: Record<string, unknown> }) => {
    if (!selectedPlanId) return;
    try {
      setMeasuring(true);
      setError("");
      const measured = await measureWorkerItem({
        plan_id: selectedPlanId,
        object_type: payload.object_type,
        raw_object: payload.raw_object,
        display_scale: 1,
        report_type: reportType
      });
      setItems((prev) => [...prev, measured]);
    } catch (e) {
      console.error(e);
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as { message?: string })?.message
        || "שגיאה במדידת פריט";
      setError(msg);
    } finally {
      setMeasuring(false);
    }
  };

  const totalLength = React.useMemo(() => items.filter((i) => i.unit === "m").reduce((s, i) => s + i.measurement, 0), [items]);
  const totalArea = React.useMemo(() => items.filter((i) => i.unit !== "m").reduce((s, i) => s + i.measurement, 0), [items]);

  const saveReport = async () => {
    if (!selectedPlanId || items.length === 0) return;
    try {
      setSaving(true);
      await createWorkerReport({ plan_id: selectedPlanId, date: reportDate, shift, report_type: reportType, draw_mode: drawMode, items, note });
      setItems([]);
      setNote("");
      await loadReports();
      toast("הדיווח נשמר בהצלחה");
    } catch (e) {
      console.error(e);
      setError("שגיאה בשמירת הדיווח.");
    } finally { setSaving(false); }
  };

  const totalReportWalls = reports.filter(r => r.report_type === "walls").reduce((s, r) => s + r.total_length_m, 0);
  const totalReportFloor = reports.filter(r => r.report_type === "floor").reduce((s, r) => s + r.total_area_m2, 0);

  const TAB_ITEMS: { id: WorkerTab; icon: string; label: string }[] = [
    { id: "map", icon: "🗺️", label: "מפה" },
    { id: "notes", icon: "📝", label: "הערות" },
    { id: "history", icon: "📋", label: "היסטוריה" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0, minHeight: "100%" }}>

      {/* ── Top bar ── */}
      <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 14, padding: "14px 18px", marginBottom: 16, display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", boxShadow: "0 1px 4px rgba(0,0,0,0.05)" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3, flex: 1, minWidth: 160 }}>
          <span style={{ fontSize: 11, color: "#64748b" }}>פרויקט</span>
          <select
            style={{ padding: "7px 10px", border: "1px solid #CBD5E1", borderRadius: 8, fontSize: 13, background: "#fff" }}
            value={selectedPlanId}
            onChange={(e) => setSelectedPlanId(e.target.value)}
          >
            {plans.length === 0 && <option value="">אין תוכניות</option>}
            {plans.map((p) => <option key={p.id} value={p.id}>{p.plan_name}</option>)}
          </select>
        </div>

        {sections.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3, flex: 1, minWidth: 160 }}>
            <span style={{ fontSize: 11, color: "#64748b" }}>גזרת עבודה</span>
            <select
              style={{ padding: "7px 10px", border: "1px solid #CBD5E1", borderRadius: 8, fontSize: 13, background: "#fff" }}
              value={selectedSectionUid}
              onChange={(e) => setSelectedSectionUid(e.target.value)}
            >
              <option value="">— כל הפרויקט —</option>
              {sections.map(sec => (
                <option key={sec.uid} value={sec.uid}>
                  {sec.name || "גזרה"} — {sec.contractor || sec.worker || ""}
                </option>
              ))}
            </select>
          </div>
        )}

        {reports.length > 0 && (
          <button
            type="button"
            onClick={() => printProgressReport(reports, selectedPlan?.plan_name ?? "פרויקט")}
            style={{ padding: "7px 14px", border: "1.5px solid #1B3A6B", color: "#1B3A6B", borderRadius: 8, fontSize: 13, fontWeight: 600, background: "#fff", cursor: "pointer" }}
          >
            🖨️ הדפס דוח
          </button>
        )}
      </div>

      {/* ── Amber alert bar (active session) ── */}
      {(items.length > 0 || measuring) && (
        <div style={{
          background: "#FFFBEB", border: "1px solid #FCD34D", borderRadius: 12,
          padding: "10px 18px", marginBottom: 16,
          display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap",
        }}>
          <span style={{ fontSize: 14 }}>⚡</span>
          <span style={{ fontWeight: 700, fontSize: 13, color: "#92400E" }}>סשן פעיל</span>
          <span style={{ fontSize: 13, color: "#78350F" }}>{items.length} פריטים</span>
          {totalLength > 0 && <span style={{ fontSize: 13, color: "#78350F" }}>{totalLength.toFixed(2)} מ' קירות</span>}
          {totalArea > 0 && <span style={{ fontSize: 13, color: "#78350F" }}>{totalArea.toFixed(2)} מ"ר ריצוף</span>}
          {measuring && <span style={{ fontSize: 12, color: "#1d4ed8" }}>מודד...</span>}
        </div>
      )}

      {error && (
        <div style={{ fontSize: 13, color: "#DC2626", background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 10, padding: "10px 14px", marginBottom: 12 }}>{error}</div>
      )}

      {/* ── Tab bar ── */}
      <div style={{ display: "flex", borderBottom: "2px solid #E2E8F0", marginBottom: 16, gap: 4 }}>
        {TAB_ITEMS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveTab(t.id)}
            style={{
              padding: "10px 20px",
              border: "none",
              borderBottom: activeTab === t.id ? "2px solid #1B3A6B" : "2px solid transparent",
              marginBottom: -2,
              background: "none",
              color: activeTab === t.id ? "#1B3A6B" : "#64748b",
              fontWeight: activeTab === t.id ? 700 : 400,
              fontSize: 14,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
            {t.id === "notes" && items.length > 0 && (
              <span style={{ background: "#EF4444", color: "#fff", borderRadius: 99, fontSize: 10, fontWeight: 700, padding: "1px 6px" }}>{items.length}</span>
            )}
            {t.id === "history" && reports.length > 0 && (
              <span style={{ background: "#E2E8F0", color: "#475569", borderRadius: 99, fontSize: 10, fontWeight: 700, padding: "1px 6px" }}>{reports.length}</span>
            )}
          </button>
        ))}
      </div>

      {/* ── Map tab ── */}
      {activeTab === "map" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Draw mode toolbar */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
            <div style={{ display: "flex", gap: 6 }}>
              {([
                { val: "line", label: "—  קו", icon: "📏" },
                { val: "rect", label: "□  מלבן", icon: "⬛" },
                { val: "path", label: "〰  חופשי", icon: "✏️" },
              ] as { val: DrawMode; label: string; icon: string }[]).map((m) => (
                <button
                  key={m.val}
                  type="button"
                  onClick={() => setDrawMode(m.val)}
                  style={{
                    padding: "7px 14px",
                    border: drawMode === m.val ? "2px solid #1B3A6B" : "1.5px solid #CBD5E1",
                    borderRadius: 8,
                    background: drawMode === m.val ? "#EFF6FF" : "#fff",
                    color: drawMode === m.val ? "#1B3A6B" : "#475569",
                    fontWeight: drawMode === m.val ? 700 : 400,
                    fontSize: 12,
                    cursor: "pointer",
                  }}
                >
                  {m.icon} {m.label}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <label style={{ fontSize: 12, color: "#64748b", display: "flex", alignItems: "center", gap: 4 }}>
                סוג:
                <select
                  style={{ padding: "5px 8px", border: "1px solid #CBD5E1", borderRadius: 7, fontSize: 12, background: "#fff" }}
                  value={reportType}
                  onChange={(e) => setReportType(e.target.value as "walls" | "floor")}
                >
                  <option value="walls">קירות</option>
                  <option value="floor">ריצוף/חיפוי</option>
                </select>
              </label>
            </div>
          </div>

          <WorkerCanvas
            imageUrl={imageUrl}
            items={items}
            drawMode={drawMode}
            sections={sections}
            activeSectionUid={selectedSectionUid}
            onDrawComplete={(p) => void handleDrawComplete(p)}
          />
        </div>
      )}

      {/* ── Notes tab ── */}
      {activeTab === "notes" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 600 }}>
          {/* Date / shift / report type */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>תאריך</span>
              <input type="date" style={fieldStyle} value={reportDate} onChange={(e) => setReportDate(e.target.value)} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>משמרת</span>
              <select style={fieldStyle} value={shift} onChange={(e) => setShift(e.target.value)}>
                <option>בוקר</option>
                <option>צהריים</option>
                <option>לילה</option>
              </select>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>סוג דיווח</span>
              <select style={fieldStyle} value={reportType} onChange={(e) => setReportType(e.target.value as "walls" | "floor")}>
                <option value="walls">קירות</option>
                <option value="floor">ריצוף/חיפוי</option>
              </select>
            </div>
          </div>

          {/* Items list */}
          {items.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#64748b", fontWeight: 600 }}>פריטים מסומנים ({items.length})</div>
              {items.map((item, idx) => (
                <div key={item.uid} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "8px 12px" }}>
                  <span style={{ fontSize: 13 }}>#{idx + 1} {item.type}</span>
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    <span style={{ fontWeight: 700, fontSize: 13, color: "#1B3A6B" }}>{item.measurement.toFixed(2)} {item.unit}</span>
                    <button type="button" onClick={() => setItems((prev) => prev.filter((_, i) => i !== idx))} style={{ color: "#EF4444", background: "none", border: "none", cursor: "pointer", fontSize: 14, lineHeight: 1 }}>✕</button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: 13, color: "#94A3B8", background: "#F8FAFC", border: "1px dashed #CBD5E1", borderRadius: 10, padding: "24px", textAlign: "center" }}>
              עבור לטאב מפה וסמן פריטים על השרטוט
            </div>
          )}

          {/* Note */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 11, color: "#64748b" }}>הערה</span>
            <textarea
              style={{ ...fieldStyle, minHeight: 80, resize: "vertical" }}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="הערות לדיווח..."
            />
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 10 }}>
            <button
              type="button"
              onClick={async () => {
                const ok = await confirm({ title: "נקה סימון", message: "האם למחוק את כל הפריטים המסומנים?", confirmText: "נקה", danger: true });
                if (ok) setItems([]);
              }}
              style={{ flex: 1, padding: "10px", border: "1.5px solid #CBD5E1", borderRadius: 9, fontSize: 13, background: "#fff", cursor: "pointer", fontWeight: 600, color: "#475569" }}
            >
              🗑️ נקה
            </button>
            <button
              type="button"
              onClick={() => void saveReport()}
              disabled={items.length === 0 || saving}
              style={{ flex: 2, padding: "10px", background: items.length === 0 ? "#94A3B8" : "#1B3A6B", color: "#fff", borderRadius: 9, fontSize: 13, fontWeight: 700, border: "none", cursor: items.length === 0 ? "not-allowed" : "pointer" }}
            >
              {saving ? "שומר..." : "💾 שמור דיווח"}
            </button>
          </div>
        </div>
      )}

      {/* ── History tab ── */}
      {activeTab === "history" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Totals */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 12, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "#94A3B8" }}>סה"כ קירות</div>
              <div style={{ fontWeight: 700, fontSize: 20, color: "#1B3A6B", marginTop: 4 }}>{totalReportWalls.toFixed(2)} מ'</div>
            </div>
            <div style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 12, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "#94A3B8" }}>סה"כ ריצוף</div>
              <div style={{ fontWeight: 700, fontSize: 20, color: "#1B3A6B", marginTop: 4 }}>{totalReportFloor.toFixed(2)} מ"ר</div>
            </div>
          </div>

          {reports.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "#94A3B8", fontSize: 14 }}>אין דיווחים עדיין.</div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse", minWidth: 480 }}>
                <thead>
                  <tr style={{ color: "#94A3B8", borderBottom: "1px solid #E2E8F0" }}>
                    <th style={{ textAlign: "right", padding: "8px 10px" }}>תאריך</th>
                    <th style={{ textAlign: "right", padding: "8px 10px" }}>משמרת</th>
                    <th style={{ textAlign: "right", padding: "8px 10px" }}>סוג</th>
                    <th style={{ textAlign: "right", padding: "8px 10px" }}>כמות</th>
                    <th style={{ textAlign: "right", padding: "8px 10px" }}>הערה</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.slice().reverse().map((r) => (
                    <tr key={r.id} style={{ borderBottom: "1px solid #F1F5F9" }}>
                      <td style={{ padding: "8px 10px" }}>{r.date}</td>
                      <td style={{ padding: "8px 10px" }}>{r.shift}</td>
                      <td style={{ padding: "8px 10px" }}>{r.report_type === "walls" ? "קירות" : "ריצוף"}</td>
                      <td style={{ padding: "8px 10px", fontWeight: 700, color: "#1B3A6B" }}>
                        {r.report_type === "walls" ? `${r.total_length_m.toFixed(2)} מ'` : `${r.total_area_m2.toFixed(2)} מ"ר`}
                      </td>
                      <td style={{ padding: "8px 10px", color: "#64748b" }}>{r.note || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const fieldStyle: React.CSSProperties = {
  padding: "7px 10px",
  border: "1px solid #CBD5E1",
  borderRadius: 8,
  fontSize: 13,
  background: "#fff",
  width: "100%",
  boxSizing: "border-box",
};
