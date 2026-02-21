import React from "react";
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
  // Use refs for pan/draw so handlers always have current value without stale closures
  const isPanningRef = React.useRef(false);
  const isDrawingRef = React.useRef(false);
  const lastMouse = React.useRef({ x: 0, y: 0 });
  const panRef = React.useRef({ x: 0, y: 0 });
  const zoomRef = React.useRef(1);
  const [startPt, setStartPt] = React.useState<Point | null>(null);
  const [tempPt, setTempPt] = React.useState<Point | null>(null);
  const [pathPts, setPathPts] = React.useState<Point[]>([]);
  const [imgSize, setImgSize] = React.useState({ w: 0, h: 0 });
  // Keep render-triggering copies in sync
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

  // Passive-false wheel via ref
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
    // Use functional updater to read latest state without stale closure
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
    <div className="relative bg-slate-100 rounded-lg overflow-hidden border border-slate-200" style={{ minHeight: 480 }}>
      <div
        ref={containerRef}
        className="w-full overflow-hidden select-none"
        style={{ cursor: isDrawingState ? "crosshair" : isPanningState ? "grabbing" : "crosshair", minHeight: 480 }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          // Only cancel panning on leave, not drawing (user may return)
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

          {/* SVG overlay – coords in natural image pixels */}
          {imgSize.w > 0 && (
            <svg
              width={imgSize.w}
              height={imgSize.h}
              style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
            >
              {/* Section overlays */}
              {sections.map((sec) => {
                if (sec.width < 2 || sec.height < 2) return null;
                const isActive = sec.uid === activeSectionUid;
                return (
                  <g key={sec.uid}>
                    <rect
                      x={sec.x} y={sec.y} width={sec.width} height={sec.height}
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

              {/* Saved items */}
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

              {/* Live drawing preview */}
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
      <div className="absolute bottom-3 left-3 flex flex-col gap-1 z-10">
        <button type="button" onClick={() => setZoomSync(clampZ(zoomRef.current * 1.25))} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50">+</button>
        <button type="button" onClick={() => { setZoomSync(1); setPanSync({ x: 0, y: 0 }); }} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-[10px] hover:bg-slate-50">↺</button>
        <button type="button" onClick={() => setZoomSync(clampZ(zoomRef.current * 0.8))} className="w-8 h-8 bg-white border border-slate-300 rounded shadow text-base font-bold hover:bg-slate-50">−</button>
      </div>
      <div className="absolute bottom-3 right-3 bg-black/40 text-white text-[11px] px-2 py-0.5 rounded">{Math.round(renderZoom * 100)}%</div>
      <div className="absolute top-2 right-2 bg-black/40 text-white text-[10px] px-2 py-0.5 rounded">גלגלת=זום | Alt+גרור=הזזה</div>
    </div>
  );
};

// ─── Main WorkerPage ──────────────────────────────────────────────────────────
export const WorkerPage: React.FC = () => {
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

  // Load sections when plan changes
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
        display_scale: 1, // canvas uses natural pixel coords
        report_type: reportType
      });
      setItems((prev) => [...prev, measured]);
    } catch (e) {
      console.error(e);
      setError("שגיאה במדידת פריט.");
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
    } catch (e) {
      console.error(e);
      setError("שגיאה בשמירת הדיווח.");
    } finally { setSaving(false); }
  };

  const totalReportWalls = reports.filter(r => r.report_type === "walls").reduce((s, r) => s + r.total_length_m, 0);
  const totalReportFloor = reports.filter(r => r.report_type === "floor").reduce((s, r) => s + r.total_area_m2, 0);

  return (
    <div className="space-y-4">
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-[#31333F]">👷 צד עובד – דיווח שטח</h2>
          <p className="text-xs text-slate-500 mt-1">בחר תוכנית, סמן ביצוע על השרטוט, ושמור דיווח יומי.</p>
        </div>
        {reports.length > 0 && (
          <button
            type="button"
            onClick={() => printProgressReport(reports, selectedPlan?.plan_name ?? "פרויקט")}
            className="bg-white border border-[#FF4B4B] text-[#FF4B4B] px-3 py-2 rounded text-sm font-semibold hover:bg-red-50"
          >
            🖨️ הדפס דוח התקדמות
          </button>
        )}
      </div>

      {error && <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-2">{error}</p>}

      <div className="grid grid-cols-1 lg:grid-cols-[300px,1fr] gap-5">
        {/* Sidebar */}
        <aside className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm space-y-4">
          <label className="text-xs block">
            תוכנית
            <select className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2" value={selectedPlanId} onChange={(e) => setSelectedPlanId(e.target.value)}>
              {plans.length === 0 && <option value="">אין תוכניות</option>}
              {plans.map((p) => <option key={p.id} value={p.id}>{p.plan_name}</option>)}
            </select>
          </label>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <label className="block">
              סוג דיווח
              <select className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2" value={reportType} onChange={(e) => setReportType(e.target.value as "walls" | "floor")}>
                <option value="walls">קירות</option>
                <option value="floor">ריצוף/חיפוי</option>
              </select>
            </label>
            <label className="block">
              כלי ציור
              <select className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2" value={drawMode} onChange={(e) => setDrawMode(e.target.value as DrawMode)}>
                <option value="line">קו (אורך)</option>
                <option value="rect">מלבן (שטח)</option>
                <option value="path">חופשי</option>
              </select>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <label className="block">
              תאריך
              <input type="date" className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2" value={reportDate} onChange={(e) => setReportDate(e.target.value)} />
            </label>
            <label className="block">
              משמרת
              <select className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2" value={shift} onChange={(e) => setShift(e.target.value)}>
                <option>בוקר</option>
                <option>צהריים</option>
                <option>לילה</option>
              </select>
            </label>
          </div>

          {sections.length > 0 && (
            <label className="text-xs block">
              גזרת עבודה
              <select
                className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
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
              {selectedSectionUid && (() => {
                const sec = sections.find(s => s.uid === selectedSectionUid);
                return sec ? (
                  <div className="mt-1 text-[10px] text-slate-500 bg-slate-50 rounded px-2 py-1" style={{ borderRight: `3px solid ${sec.color}` }}>
                    🏗 {sec.contractor || "—"} | 👷 {sec.worker || "—"}
                  </div>
                ) : null;
              })()}
            </label>
          )}

          <label className="text-xs block">
            הערה
            <textarea className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2 min-h-[60px]" value={note} onChange={(e) => setNote(e.target.value)} />
          </label>

          {/* Session totals */}
          <div className="bg-slate-50 rounded-lg p-3 space-y-1 text-sm">
            <div className="text-xs text-slate-500 font-semibold">סשן נוכחי</div>
            <div className="flex justify-between">
              <span>אורך קירות:</span>
              <span className="font-bold text-[#FF4B4B]">{totalLength.toFixed(2)} מ'</span>
            </div>
            <div className="flex justify-between">
              <span>שטח ריצוף:</span>
              <span className="font-bold text-[#FF4B4B]">{totalArea.toFixed(2)} מ"ר</span>
            </div>
            {measuring && <div className="text-xs text-blue-600">מודד...</div>}
          </div>

          {/* Items list */}
          {items.length > 0 && (
            <div className="text-xs space-y-1 max-h-36 overflow-y-auto">
              {items.map((item, idx) => (
                <div key={item.uid} className="flex justify-between bg-slate-50 rounded px-2 py-1">
                  <span>#{idx + 1} {item.type}</span>
                  <div className="flex gap-2 items-center">
                    <span className="font-semibold">{item.measurement.toFixed(2)} {item.unit}</span>
                    <button type="button" onClick={() => setItems((prev) => prev.filter((_, i) => i !== idx))} className="text-red-400 hover:text-red-600">✕</button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="space-y-2">
            <button type="button" onClick={() => setItems([])} className="w-full bg-white border border-slate-300 rounded py-2 text-sm">
              🗑️ נקה סימון
            </button>
            <button
              type="button"
              onClick={() => void saveReport()}
              disabled={items.length === 0 || saving}
              className="w-full bg-[#FF4B4B] text-white rounded py-2 text-sm font-semibold disabled:opacity-40"
            >
              {saving ? "שומר..." : "💾 שמור דיווח"}
            </button>
          </div>
        </aside>

        {/* Canvas */}
        <main className="space-y-4">
          <WorkerCanvas
            imageUrl={imageUrl}
            items={items}
            drawMode={drawMode}
            sections={sections}
            activeSectionUid={selectedSectionUid}
            onDrawComplete={(p) => void handleDrawComplete(p)}
          />

          {/* Reports history */}
          <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold">📋 דיווחים קודמים</h3>
              <div className="text-xs text-slate-500">
                סה"כ: {totalReportWalls.toFixed(2)} מ' קירות | {totalReportFloor.toFixed(2)} מ"ר ריצוף
              </div>
            </div>
            {reports.length === 0 ? (
              <p className="text-xs text-slate-500">אין דיווחים עדיין.</p>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-xs min-w-[500px]">
                  <thead>
                    <tr className="text-slate-500 border-b">
                      <th className="text-right p-2">תאריך</th>
                      <th className="text-right p-2">משמרת</th>
                      <th className="text-right p-2">סוג</th>
                      <th className="text-right p-2">כמות</th>
                      <th className="text-right p-2">הערה</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reports.slice().reverse().map((r) => (
                      <tr key={r.id} className="border-b last:border-b-0 hover:bg-slate-50">
                        <td className="p-2">{r.date}</td>
                        <td className="p-2">{r.shift}</td>
                        <td className="p-2">{r.report_type === "walls" ? "קירות" : "ריצוף"}</td>
                        <td className="p-2 font-semibold">
                          {r.report_type === "walls" ? `${r.total_length_m.toFixed(2)} מ'` : `${r.total_area_m2.toFixed(2)} מ"ר`}
                        </td>
                        <td className="p-2 text-slate-500">{r.note || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
};
