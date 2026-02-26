import React from "react";
import axios from "axios";
import { ErrorAlert } from "../components/UiHelpers";
import { apiClient } from "../api/client";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  addPlanningItem,
  addTextItem,
  addWorkSection,
  addZoneItem,
  autoAnalyzePlan,
  calibratePlanningScale,
  deletePlanningItem,
  deleteWorkSection,
  finalizePlanning,
  getPlanningState,
  importVisionItems,
  resolvePlanningOpening,
  resolvePlanningWall,
  type AutoSegment,
  type AutoAnalyzeVisionData,
  type PlanningCategory,
  type PlanningState,
  type TextItemPayload,
  type WorkSection,
  upsertPlanningCategories,
} from "../api/planningApi";

type WizardStep = 1 | 2 | 3 | 4 | 5;
type DrawMode = "line" | "rect" | "path";
type Step3Tab = "auto" | "zone" | "manual" | "text";

type Point = { x: number; y: number };

// A shape drawn but not yet assigned to a category
interface PendingShape {
  id: string; // local temp id
  object_type: DrawMode;
  raw_object: Record<string, unknown>;
  display_scale: number;
}

const CATEGORY_SUBTYPES: Record<string, string[]> = {
  "קירות": ["בטון", "בלוקים", "גבס", "מחיצה קלה"],
  "ריצוף": ["קרמיקה", "גרניט פורצלן", "פרקט", "בטון מוחלק"],
  "תקרה": ["גבס", "אקוסטית", "חשופה", "צבועה"]
};

const CATEGORY_COLORS: Record<string, string> = {
  "קירות:בטון": "#0ea5e9",
  "קירות:בלוקים": "#2563eb",
  "קירות:גבס": "#6366f1",
  "קירות:מחיצה קלה": "#8b5cf6",
  "ריצוף:קרמיקה": "#f97316",
  "ריצוף:גרניט פורצלן": "#ea580c",
  "ריצוף:פרקט": "#a16207",
  "ריצוף:בטון מוחלק": "#b45309",
  "תקרה:גבס": "#14b8a6",
  "תקרה:אקוסטית": "#0d9488",
  "תקרה:חשופה": "#059669",
  "תקרה:צבועה": "#10b981"
};

const DEFAULT_CATEGORY_COLOR = "#334155";
const PENDING_COLOR = "#F59E0B"; // amber for unassigned

function getCategoryColor(type?: string, subtype?: string): string {
  if (!type || !subtype) return DEFAULT_CATEGORY_COLOR;
  return CATEGORY_COLORS[`${type}:${subtype}`] ?? DEFAULT_CATEGORY_COLOR;
}

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const full = normalized.length === 3
    ? normalized.split("").map((c) => `${c}${c}`).join("")
    : normalized;
  const intVal = Number.parseInt(full, 16);
  const r = (intVal >> 16) & 255;
  const g = (intVal >> 8) & 255;
  const b = intVal & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function generateTempId(): string {
  return `pending_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

// ──────────────────────────────────────────────────────────────────────────────
// CategoryPickerModal — choose or create a category for pending shapes
// ──────────────────────────────────────────────────────────────────────────────
interface CategoryPickerProps {
  categories: Record<string, PlanningCategory>;
  pendingCount: number;
  onPick: (categoryKey: string) => void;
  onCreateAndPick: (type: string, subtype: string, paramValue: number, paramNote: string) => void;
  onCancel: () => void;
}

const CategoryPickerModal: React.FC<CategoryPickerProps> = ({
  categories,
  pendingCount,
  onPick,
  onCreateAndPick,
  onCancel,
}) => {
  const [newType, setNewType] = React.useState("קירות");
  const [newSubtype, setNewSubtype] = React.useState("בטון");
  const [newParamValue, setNewParamValue] = React.useState(2.6);
  const [newParamNote, setNewParamNote] = React.useState("");
  const [tab, setTab] = React.useState<"existing" | "new">(
    Object.keys(categories).length > 0 ? "existing" : "new"
  );

  const subtypeOptions = CATEGORY_SUBTYPES[newType] ?? ["כללי"];

  React.useEffect(() => {
    if (!subtypeOptions.includes(newSubtype)) setNewSubtype(subtypeOptions[0]);
  }, [newType, newSubtype, subtypeOptions]);

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 2000,
        background: "rgba(0,0,0,0.55)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div style={{
        background: "#fff", borderRadius: 16, padding: 28, width: 420, maxWidth: "95vw",
        boxShadow: "0 8px 40px rgba(0,0,0,0.22)",
        direction: "rtl",
      }}>
        <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4, color: "#1B3A6B" }}>
          📂 שיוך {pendingCount} {pendingCount === 1 ? "פריט" : "פריטים"} לקטגוריה
        </div>
        <p style={{ fontSize: 12, color: "#64748b", marginBottom: 16 }}>
          כל הפריטים שסומנו ישוייכו לקטגוריה שתבחר.
        </p>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <button
            type="button"
            onClick={() => setTab("existing")}
            disabled={Object.keys(categories).length === 0}
            style={{
              flex: 1, padding: "6px 0", borderRadius: 8, border: "none", cursor: "pointer",
              background: tab === "existing" ? "#1B3A6B" : "#F1F5F9",
              color: tab === "existing" ? "#fff" : "#334155",
              fontWeight: 600, fontSize: 13,
              opacity: Object.keys(categories).length === 0 ? 0.4 : 1,
            }}
          >
            קטגוריות קיימות ({Object.keys(categories).length})
          </button>
          <button
            type="button"
            onClick={() => setTab("new")}
            style={{
              flex: 1, padding: "6px 0", borderRadius: 8, border: "none", cursor: "pointer",
              background: tab === "new" ? "#1B3A6B" : "#F1F5F9",
              color: tab === "new" ? "#fff" : "#334155",
              fontWeight: 600, fontSize: 13,
            }}
          >
            + קטגוריה חדשה
          </button>
        </div>

        {tab === "existing" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 260, overflowY: "auto" }}>
            {Object.values(categories).map((cat) => {
              const color = getCategoryColor(cat.type, cat.subtype);
              return (
                <button
                  key={cat.key}
                  type="button"
                  onClick={() => onPick(cat.key)}
                  style={{
                    display: "flex", alignItems: "center", gap: 10,
                    padding: "10px 14px", borderRadius: 10,
                    border: `1.5px solid ${hexToRgba(color, 0.4)}`,
                    background: hexToRgba(color, 0.07),
                    cursor: "pointer", textAlign: "right", width: "100%",
                  }}
                >
                  <span style={{ width: 12, height: 12, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ fontWeight: 600, fontSize: 14, color: "#1e293b" }}>{cat.type} — {cat.subtype}</span>
                  <span style={{ fontSize: 11, color: "#94a3b8", marginRight: "auto" }}>{cat.key}</span>
                </button>
              );
            })}
          </div>
        )}

        {tab === "new" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <label style={{ fontSize: 13 }}>
              סוג
              <select
                style={{ display: "block", width: "100%", marginTop: 4, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                value={newType}
                onChange={(e) => setNewType(e.target.value)}
              >
                <option>קירות</option>
                <option>ריצוף</option>
                <option>תקרה</option>
              </select>
            </label>
            <label style={{ fontSize: 13 }}>
              תת-סוג
              <select
                style={{ display: "block", width: "100%", marginTop: 4, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                value={newSubtype}
                onChange={(e) => setNewSubtype(e.target.value)}
              >
                {subtypeOptions.map((sub) => <option key={sub}>{sub}</option>)}
              </select>
            </label>
            <label style={{ fontSize: 13 }}>
              פרמטר (גובה/עובי)
              <input
                type="number"
                style={{ display: "block", width: "100%", marginTop: 4, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                value={newParamValue}
                onChange={(e) => setNewParamValue(Number(e.target.value))}
              />
            </label>
            <label style={{ fontSize: 13 }}>
              הערה
              <input
                style={{ display: "block", width: "100%", marginTop: 4, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                value={newParamNote}
                onChange={(e) => setNewParamNote(e.target.value)}
                placeholder="אופציונלי"
              />
            </label>
            <button
              type="button"
              onClick={() => onCreateAndPick(newType, newSubtype, newParamValue, newParamNote)}
              style={{ padding: "10px 0", borderRadius: 10, background: "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer" }}
            >
              צור קטגוריה ושייך
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={onCancel}
          style={{ marginTop: 14, width: "100%", padding: "8px 0", borderRadius: 10, border: "1px solid #e2e8f0", background: "#fff", color: "#64748b", fontSize: 13, cursor: "pointer" }}
        >
          ביטול — המשך לצייר
        </button>
      </div>
    </div>
  );
};

// ──────────────────────────────────────────────────────────────────────────────
// ZoomModal — fullscreen plan viewer with scroll-zoom + drag-pan + drawing
// Now supports pending shapes (unassigned, shown in amber)
// ──────────────────────────────────────────────────────────────────────────────
interface ZoomModalProps {
  imageUrl: string;
  planningState: PlanningState;
  pendingShapes: PendingShape[];
  displayScale: number;
  onClose: () => void;
  onDrawComplete: (shape: PendingShape) => void;
  onAssignCategory: () => void;
  onDeletePending: (id: string) => void;
  onDeleteItem: (uid: string) => Promise<void>;
}

const ZoomModal: React.FC<ZoomModalProps> = ({
  imageUrl,
  planningState,
  pendingShapes,
  onClose,
  onDrawComplete,
  onAssignCategory,
  onDeletePending,
  onDeleteItem,
}) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const imgRef = React.useRef<HTMLImageElement | null>(null);

  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const [imgNatural, setImgNatural] = React.useState({ w: 1, h: 1 });
  const isPanning = React.useRef(false);
  const lastPan = React.useRef({ x: 0, y: 0 });

  const [modalDrawMode, setModalDrawMode] = React.useState<DrawMode>("line");
  const [drawing, setDrawing] = React.useState(false);
  const [startPt, setStartPt] = React.useState<Point | null>(null);
  const [tempPt, setTempPt] = React.useState<Point | null>(null);
  const [pathPts, setPathPts] = React.useState<Point[]>([]);

  const toNatural = React.useCallback(
    (clientX: number, clientY: number): Point => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      const sx = (clientX - rect.left - pan.x) / zoom;
      const sy = (clientY - rect.top - pan.y) / zoom;
      return { x: Math.max(0, sx), y: Math.max(0, sy) };
    },
    [zoom, pan]
  );

  const handleWheel = React.useCallback((e: WheelEvent) => {
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const delta = e.deltaY > 0 ? 0.85 : 1.18;
    setZoom((prev) => {
      const next = Math.max(0.25, Math.min(10, prev * delta));
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setPan((p) => ({
        x: mx - (mx - p.x) * (next / prev),
        y: my - (my - p.y) * (next / prev),
      }));
      return next;
    });
  }, []);

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.altKey) {
      e.preventDefault();
      isPanning.current = true;
      lastPan.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
      return;
    }
    if (e.button === 0) {
      const p = toNatural(e.clientX, e.clientY);
      setDrawing(true);
      setStartPt(p);
      setTempPt(p);
      if (modalDrawMode === "path") setPathPts([p]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning.current) {
      setPan({ x: e.clientX - lastPan.current.x, y: e.clientY - lastPan.current.y });
      return;
    }
    if (!drawing) return;
    const p = toNatural(e.clientX, e.clientY);
    setTempPt(p);
    if (modalDrawMode === "path") setPathPts((prev) => [...prev, p]);
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (isPanning.current) { isPanning.current = false; return; }
    if (!drawing || !startPt || !tempPt) { setDrawing(false); return; }
    setDrawing(false);
    const dx = tempPt.x - startPt.x, dy = tempPt.y - startPt.y;
    if (modalDrawMode !== "path" && Math.sqrt(dx * dx + dy * dy) < 4) return;

    let raw_object: Record<string, unknown>;
    if (modalDrawMode === "line") {
      raw_object = { x1: startPt.x, y1: startPt.y, x2: tempPt.x, y2: tempPt.y };
    } else if (modalDrawMode === "rect") {
      raw_object = { x: Math.min(startPt.x, tempPt.x), y: Math.min(startPt.y, tempPt.y), width: Math.abs(tempPt.x - startPt.x), height: Math.abs(tempPt.y - startPt.y) };
    } else {
      raw_object = { points: pathPts.map((p) => [p.x, p.y]) };
    }
    onDrawComplete({ id: generateTempId(), object_type: modalDrawMode, raw_object, display_scale: 1 });
    setStartPt(null); setTempPt(null); setPathPts([]);
    void e; // suppress unused warning
  };

  const imgW = imgNatural.w;
  const imgH = imgNatural.h;

  // Render a pending shape onto the SVG
  const renderPendingShape = (s: PendingShape) => {
    const obj = s.raw_object;
    if (s.object_type === "line") {
      return <line key={s.id} x1={Number(obj.x1)} y1={Number(obj.y1)} x2={Number(obj.x2)} y2={Number(obj.y2)} stroke={PENDING_COLOR} strokeWidth={3} strokeLinecap="round" strokeDasharray="8 4" />;
    }
    if (s.object_type === "rect") {
      return <rect key={s.id} x={Number(obj.x)} y={Number(obj.y)} width={Number(obj.width)} height={Number(obj.height)} fill={hexToRgba(PENDING_COLOR, 0.15)} stroke={PENDING_COLOR} strokeWidth={3} strokeDasharray="8 4" />;
    }
    const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map((p) => `${p[0]},${p[1]}`).join(" ") : "";
    return <polyline key={s.id} points={pts} fill="none" stroke={PENDING_COLOR} strokeWidth={3} strokeDasharray="8 4" />;
  };

  return (
    <div
      style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(0,0,0,0.85)", display: "flex", flexDirection: "column" }}
      onMouseLeave={() => { isPanning.current = false; setDrawing(false); }}
    >
      {/* Top toolbar */}
      <div style={{ background: "#1B3A6B", padding: "8px 16px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0, flexWrap: "wrap" }}>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 14 }}>🔍 תצוגה מוגדלת — שלב 3</span>
        <span style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>Scroll לזום • Alt+גרור לזזה • לחץ לסימון</span>

        {/* draw mode */}
        <div style={{ display: "flex", gap: 6, marginRight: "auto" }}>
          {(["line", "rect", "path"] as DrawMode[]).map((m) => (
            <button key={m} type="button" onClick={() => setModalDrawMode(m)} style={{ padding: "4px 10px", borderRadius: 6, fontSize: 12, border: "none", cursor: "pointer", background: modalDrawMode === m ? "#10B981" : "rgba(255,255,255,0.15)", color: "#fff" }}>
              {m === "line" ? "קו" : m === "rect" ? "מלבן" : "חופשי"}
            </button>
          ))}
        </div>

        {/* pending count + assign button */}
        {pendingShapes.length > 0 && (
          <button
            type="button"
            onClick={onAssignCategory}
            style={{ padding: "5px 14px", borderRadius: 8, background: PENDING_COLOR, border: "none", color: "#fff", fontWeight: 700, cursor: "pointer", fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}
          >
            📂 שייך {pendingShapes.length} {pendingShapes.length === 1 ? "פריט" : "פריטים"} לקטגוריה
          </button>
        )}
        {pendingShapes.length === 0 && (
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", padding: "4px 8px" }}>ציור → שייך לקטגוריה</span>
        )}

        {/* zoom controls */}
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button type="button" onClick={() => setZoom(z => Math.max(0.25, z * 0.8))} style={{ width: 28, height: 28, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 16 }}>−</button>
          <span style={{ color: "#fff", fontSize: 12, minWidth: 42, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => setZoom(z => Math.min(10, z * 1.25))} style={{ width: 28, height: 28, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 16 }}>+</button>
          <button type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} style={{ padding: "4px 8px", borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 11 }}>איפוס</button>
        </div>

        <button type="button" onClick={onClose} style={{ padding: "5px 14px", borderRadius: 6, background: "#EF4444", border: "none", color: "#fff", fontWeight: 700, cursor: "pointer", fontSize: 13 }}>✕ סגור</button>
      </div>

      {/* Canvas area */}
      <div
        ref={containerRef}
        style={{ flex: 1, overflow: "hidden", position: "relative", cursor: drawing ? "crosshair" : "grab" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <div style={{ position: "absolute", transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "0 0", userSelect: "none" }}>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="plan"
            style={{ display: "block", maxWidth: "none" }}
            draggable={false}
            onLoad={() => {
              if (imgRef.current) {
                setImgNatural({ w: imgRef.current.naturalWidth, h: imgRef.current.naturalHeight });
                const vw = containerRef.current?.clientWidth ?? 1200;
                const vh = containerRef.current?.clientHeight ?? 700;
                const fz = Math.min(vw / imgRef.current.naturalWidth, vh / imgRef.current.naturalHeight, 1) * 0.92;
                setZoom(fz);
                setPan({ x: (vw - imgRef.current.naturalWidth * fz) / 2, y: (vh - imgRef.current.naturalHeight * fz) / 2 });
              }
            }}
          />
          <svg width={imgW} height={imgH} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
            {/* saved+assigned items */}
            {planningState.items.map((item) => {
              const obj = item.raw_object;
              const cat = planningState.categories[item.category];
              const color = cat ? (CATEGORY_COLORS[`${cat.type}:${cat.subtype}`] ?? DEFAULT_CATEGORY_COLOR) : DEFAULT_CATEGORY_COLOR;
              if (item.type === "line") return <line key={item.uid} x1={Number(obj.x1)} y1={Number(obj.y1)} x2={Number(obj.x2)} y2={Number(obj.y2)} stroke={color} strokeWidth={3} strokeLinecap="round" />;
              if (item.type === "rect") return <rect key={item.uid} x={Number(obj.x)} y={Number(obj.y)} width={Number(obj.width)} height={Number(obj.height)} fill={hexToRgba(color, 0.15)} stroke={color} strokeWidth={3} />;
              const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map((p) => `${p[0]},${p[1]}`).join(" ") : "";
              return <polyline key={item.uid} points={pts} fill="none" stroke={color} strokeWidth={3} strokeLinejoin="round" />;
            })}
            {/* pending shapes (amber dashed) */}
            {pendingShapes.map(renderPendingShape)}
            {/* live drawing preview */}
            {drawing && startPt && tempPt && modalDrawMode === "line" && <line x1={startPt.x} y1={startPt.y} x2={tempPt.x} y2={tempPt.y} stroke={PENDING_COLOR} strokeWidth={3} strokeDasharray="6 3" opacity={0.7} />}
            {drawing && startPt && tempPt && modalDrawMode === "rect" && <rect x={Math.min(startPt.x, tempPt.x)} y={Math.min(startPt.y, tempPt.y)} width={Math.abs(tempPt.x - startPt.x)} height={Math.abs(tempPt.y - startPt.y)} fill={hexToRgba(PENDING_COLOR, 0.12)} stroke={PENDING_COLOR} strokeWidth={3} strokeDasharray="6 3" opacity={0.7} />}
            {drawing && modalDrawMode === "path" && pathPts.length > 1 && <polyline points={pathPts.map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke={PENDING_COLOR} strokeWidth={3} strokeDasharray="6 3" opacity={0.7} />}
          </svg>
        </div>
      </div>

      {/* Bottom bar */}
      <div style={{ background: "#1B3A6B", padding: "6px 16px", maxHeight: 120, overflowY: "auto", flexShrink: 0 }}>
        {/* Pending row */}
        {pendingShapes.length > 0 && (
          <div style={{ marginBottom: 6 }}>
            <div style={{ color: PENDING_COLOR, fontSize: 11, fontWeight: 600, marginBottom: 4 }}>
              ⏳ ממתין לשיוך ({pendingShapes.length}) — לחץ כפתור בסרגל למעלה לבחירת קטגוריה
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {pendingShapes.map((s) => (
                <div key={s.id} style={{ background: hexToRgba(PENDING_COLOR, 0.15), border: `1px solid ${hexToRgba(PENDING_COLOR, 0.4)}`, borderRadius: 6, padding: "2px 8px", fontSize: 11, color: "#fef3c7", display: "flex", gap: 5, alignItems: "center" }}>
                  <span>{s.object_type === "line" ? "קו" : s.object_type === "rect" ? "מלבן" : "חופשי"}</span>
                  <button type="button" onClick={() => onDeletePending(s.id)} style={{ background: "none", border: "none", color: "#F87171", cursor: "pointer", fontSize: 11, padding: 0 }}>✕</button>
                </div>
              ))}
            </div>
          </div>
        )}
        {/* Assigned items row */}
        <div style={{ color: "rgba(255,255,255,0.6)", fontSize: 11, marginBottom: 4 }}>
          פריטים משויכים ({planningState.items.length})
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
          {planningState.items.map((item) => {
            const cat = planningState.categories[item.category];
            const color = cat ? (CATEGORY_COLORS[`${cat.type}:${cat.subtype}`] ?? DEFAULT_CATEGORY_COLOR) : DEFAULT_CATEGORY_COLOR;
            return (
              <div key={item.uid} style={{ background: "rgba(255,255,255,0.1)", borderRadius: 6, padding: "2px 8px", fontSize: 11, color: "#fff", display: "flex", gap: 5, alignItems: "center" }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0, display: "inline-block" }} />
                <span>{cat?.subtype ?? item.category} | {item.type} | {(item.length_m_effective ?? item.length_m).toFixed(2)}מ׳</span>
                <button type="button" onClick={() => void onDeleteItem(item.uid)} style={{ background: "none", border: "none", color: "#F87171", cursor: "pointer", fontSize: 11, padding: 0 }}>✕</button>
              </div>
            );
          })}
          {planningState.items.length === 0 && <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>אין פריטים משויכים עדיין.</span>}
        </div>
      </div>
    </div>
  );
};

// ──────────────────────────────────────────────────────────────────────────────
// PlanningPage — main wizard
// ──────────────────────────────────────────────────────────────────────────────
export const PlanningPage: React.FC = () => {
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState<string>("");
  const [planningState, setPlanningState] = React.useState<PlanningState | null>(null);
  const [step, setStep] = React.useState<WizardStep>(1);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string>("");
  const [finalizeNotice, setFinalizeNotice] = React.useState<string>("");
  const [zoomModalOpen, setZoomModalOpen] = React.useState(false);
  const [step3Tab, setStep3Tab] = React.useState<Step3Tab>("auto");

  // ── Pending shapes (drawn but not yet assigned to a category) ──
  const [pendingShapes, setPendingShapes] = React.useState<PendingShape[]>([]);
  const [categoryPickerOpen, setCategoryPickerOpen] = React.useState(false);

  // ── Category management state (used in step 3 side panel) ──
  const [categoriesDraft, setCategoriesDraft] = React.useState<Record<string, PlanningCategory>>({});
  const [newType, setNewType] = React.useState("קירות");
  const [newSubtype, setNewSubtype] = React.useState("בטון");
  const [newParamValue, setNewParamValue] = React.useState<number>(2.6);
  const [newParamNote, setNewParamNote] = React.useState("");

  // ── Auto-analyze state ──
  const [autoSegments, setAutoSegments] = React.useState<AutoSegment[] | null>(null);
  const [autoVisionData, setAutoVisionData] = React.useState<AutoAnalyzeVisionData | null>(null);
  const [autoLoading, setAutoLoading] = React.useState(false);
  const [autoSelected, setAutoSelected] = React.useState<Set<string>>(new Set());
  const [autoConfirmedKeys, setAutoConfirmedKeys] = React.useState<Record<string, string>>({}); // segId→catKey

  // ── Zone state ──
  const [zoneDrawing, setZoneDrawing] = React.useState(false);
  const [zoneStart, setZoneStart] = React.useState<Point | null>(null);
  const [zoneEnd, setZoneEnd] = React.useState<Point | null>(null);
  const [zoneTemp, setZoneTemp] = React.useState<Point | null>(null);
  const [zoneCatKey, setZoneCatKey] = React.useState<string>("");
  const zoneCanvasRef = React.useRef<SVGSVGElement | null>(null);

  // ── Text items state ──
  const [textRows, setTextRows] = React.useState<TextItemPayload[]>([
    { category_key: "__manual__", description: "", quantity: 1, unit: "יח׳", note: "" }
  ]);

  // ── Step 5 — Sections (גזרות עבודה) ──
  const [secContractor, setSecContractor] = React.useState("");
  const [secWorker, setSecWorker] = React.useState("");
  const [secName, setSecName] = React.useState("");
  const [secColor, setSecColor] = React.useState("#6366f1");
  // Section canvas drawing (draw rect on plan to define section boundary)
  const [secDrawing, setSecDrawing] = React.useState(false);
  const [secStart, setSecStart] = React.useState<Point | null>(null);
  const [secEnd, setSecEnd] = React.useState<Point | null>(null);
  const [secTemp, setSecTemp] = React.useState<Point | null>(null);
  const secCanvasRef = React.useRef<SVGSVGElement | null>(null);
  const secImageRef = React.useRef<HTMLImageElement | null>(null);

  // ── Drawing state (main canvas) ──
  const [drawMode, setDrawMode] = React.useState<DrawMode>("line");
  const [drawing, setDrawing] = React.useState(false);
  const [startPoint, setStartPoint] = React.useState<Point | null>(null);
  const [tempPoint, setTempPoint] = React.useState<Point | null>(null);
  const [pathPoints, setPathPoints] = React.useState<Point[]>([]);

  // ── Opening / wall confirmation prompts ──
  const [openingPrompt, setOpeningPrompt] = React.useState<{ itemUid: string; gapId?: string; gapLengthM?: number } | null>(null);
  const [wallPrompt, setWallPrompt] = React.useState<{ itemUid: string; overlapRatio?: number } | null>(null);

  // ── Calibration state ──
  const [calStart, setCalStart] = React.useState<Point | null>(null);
  const [calEnd, setCalEnd] = React.useState<Point | null>(null);
  const [calDrawing, setCalDrawing] = React.useState(false);
  const [calTemp, setCalTemp] = React.useState<Point | null>(null);
  const [calibrationLengthM, setCalibrationLengthM] = React.useState<number>(1);

  const calibrationImageRef = React.useRef<HTMLImageElement | null>(null);
  const drawingImageRef = React.useRef<HTMLImageElement | null>(null);
  const calibrationSurfaceRef = React.useRef<SVGSVGElement | null>(null);
  const drawingSurfaceRef = React.useRef<SVGSVGElement | null>(null);
  const lastPathPointRef = React.useRef<number>(0); // timestamp for throttle
  const [baseDisplaySize, setBaseDisplaySize] = React.useState({ width: 800, height: 600 });
  const [zoomPercent, setZoomPercent] = React.useState(100);

  const displaySize = React.useMemo(() => {
    const factor = Math.max(0.5, Math.min(3.0, zoomPercent / 100));
    return {
      width: Math.max(1, Math.round(baseDisplaySize.width * factor)),
      height: Math.max(1, Math.round(baseDisplaySize.height * factor))
    };
  }, [baseDisplaySize, zoomPercent]);

  const imageUrl = selectedPlanId
    ? `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(selectedPlanId)}/image`
    : "";

  const displayScale = React.useMemo(() => {
    const naturalW = drawingImageRef.current?.naturalWidth || calibrationImageRef.current?.naturalWidth;
    if (naturalW && naturalW > 0) return displaySize.width / naturalW;
    if (!planningState || planningState.image_width <= 0) return 1;
    return displaySize.width / planningState.image_width;
  }, [planningState, displaySize.width, selectedPlanId]);

  const subtypeOptions = CATEGORY_SUBTYPES[newType] ?? ["כללי"];

  React.useEffect(() => {
    if (!subtypeOptions.includes(newSubtype)) setNewSubtype(subtypeOptions[0]);
  }, [newType, newSubtype, subtypeOptions]);

  // ── Load plans ──
  const loadPlans = React.useCallback(async () => {
    const data = await listWorkshopPlans();
    setPlans(data);
    if (!selectedPlanId && data.length > 0) setSelectedPlanId(data[0].id);
  }, [selectedPlanId]);

  const loadPlanningState = React.useCallback(async (planId: string) => {
    const state = await getPlanningState(planId);
    setPlanningState(state);
    setCategoriesDraft(state.categories);
  }, []);

  React.useEffect(() => {
    void loadPlans().catch(() => setError("שגיאה בטעינת תוכניות."));
  }, [loadPlans]);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    setZoomPercent(100);
    setLoading(true);
    void loadPlanningState(selectedPlanId)
      .catch(() => setError("שגיאה בטעינת נתוני הגדרת תכולה."))
      .finally(() => setLoading(false));
  }, [selectedPlanId, loadPlanningState]);

  const selectedPlan = plans.find((p) => p.id === selectedPlanId) ?? null;

  const updateDisplaySizeFromImage = (img: HTMLImageElement | null) => {
    if (!img) return;
    const maxW = 920;
    const naturalW = img.naturalWidth || planningState?.image_width || 1;
    const naturalH = img.naturalHeight || planningState?.image_height || 1;
    const scale = Math.min(1, maxW / naturalW);
    setBaseDisplaySize({ width: Math.max(1, Math.round(naturalW * scale)), height: Math.max(1, Math.round(naturalH * scale)) });
  };

  const toLocalPoint = (targetRef: React.RefObject<HTMLElement | null>, clientX: number, clientY: number): Point => {
    const rect = targetRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: Math.max(0, Math.min(displaySize.width, clientX - rect.left)), y: Math.max(0, Math.min(displaySize.height, clientY - rect.top)) };
  };

  // ── Category helpers ──
  const handleAddCategory = () => {
    const key = `${newType}_${newSubtype}_${Object.keys(categoriesDraft).length + 1}`;
    setCategoriesDraft((prev) => ({
      ...prev,
      [key]: { key, type: newType, subtype: newSubtype, params: { height_or_thickness: newParamValue, note: newParamNote } }
    }));
    return key;
  };

  const handleSaveCategories = async () => {
    if (!selectedPlanId || loading) return;
    setLoading(true);
    try {
      const state = await upsertPlanningCategories(selectedPlanId, categoriesDraft);
      setPlanningState(state);
      setError("");
    } catch {
      setError("שגיאה בשמירת קטגוריות.");
    } finally { setLoading(false); }
  };

  // ── Assign pending shapes to a category ──
  const handleAssignCategory = async (categoryKey: string) => {
    if (!selectedPlanId || pendingShapes.length === 0 || loading) return;
    setCategoryPickerOpen(false);
    setLoading(true);
    try {
      let lastState = planningState!;
      for (const shape of pendingShapes) {
        lastState = await addPlanningItem(selectedPlanId, {
          category_key: categoryKey,
          object_type: shape.object_type,
          raw_object: shape.raw_object,
          display_scale: shape.display_scale,
        });
        // Handle prompts for last item
        const latest = lastState.items[lastState.items.length - 1];
        if (latest?.analysis?.requires_wall_confirmation) {
          setWallPrompt({ itemUid: latest.uid, overlapRatio: latest.analysis.wall_overlap_ratio });
          setOpeningPrompt(null);
        } else {
          setWallPrompt(null);
          const opening = latest?.analysis?.openings?.[0];
          if (latest?.uid && latest?.analysis?.prompt_opening_question) {
            setOpeningPrompt({ itemUid: latest.uid, gapId: opening?.gap_id, gapLengthM: typeof opening?.length_m === "number" ? opening.length_m : latest?.analysis?.estimated_opening_length_m });
          } else {
            setOpeningPrompt(null);
          }
        }
      }
      setPlanningState(lastState);
      setPendingShapes([]); // clear after assigning
      setError("");
    } catch (e) {
      console.error(e);
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`שגיאה בשיוך פריטים: ${detail}`);
    } finally { setLoading(false); }
  };

  // ── Create new category then assign ──
  const handleCreateAndAssign = async (type: string, subtype: string, paramValue: number, paramNote: string) => {
    const key = `${type}_${subtype}_${Object.keys(categoriesDraft).length + 1}`;
    const newCats = {
      ...categoriesDraft,
      [key]: { key, type, subtype, params: { height_or_thickness: paramValue, note: paramNote } }
    };
    setCategoriesDraft(newCats);
    if (selectedPlanId) {
      try {
        await upsertPlanningCategories(selectedPlanId, newCats);
      } catch { /* ignore, will still try to assign */ }
    }
    await handleAssignCategory(key);
  };

  // ── Canvas: main page drawing (adds to pending) ──
  const handleCanvasMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (step !== 3) return;
    const p = toLocalPoint(drawingSurfaceRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setDrawing(true);
    setStartPoint(p);
    setTempPoint(p);
    if (drawMode === "path") setPathPoints([p]);
  };

  const handleCanvasMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!drawing || step !== 3) return;
    const p = toLocalPoint(drawingSurfaceRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setTempPoint(p);
    if (drawMode === "path") {
      // Throttle path point collection to ~15fps to avoid 60fps re-render thrashing
      const now = Date.now();
      if (now - lastPathPointRef.current >= 66) {
        lastPathPointRef.current = now;
        setPathPoints((prev) => [...prev, p]);
      }
    }
  };

  const handleCanvasMouseUp: React.MouseEventHandler<SVGSVGElement> = () => {
    if (!drawing || !startPoint || !tempPoint) { setDrawing(false); return; }
    setDrawing(false);
    if (drawMode !== "path") {
      const dx = tempPoint.x - startPoint.x, dy = tempPoint.y - startPoint.y;
      if (Math.sqrt(dx * dx + dy * dy) < 6) return;
    }
    let raw_object: Record<string, unknown>;
    if (drawMode === "line") {
      raw_object = { x1: startPoint.x, y1: startPoint.y, x2: tempPoint.x, y2: tempPoint.y };
    } else if (drawMode === "rect") {
      raw_object = { x: Math.min(startPoint.x, tempPoint.x), y: Math.min(startPoint.y, tempPoint.y), width: Math.abs(tempPoint.x - startPoint.x), height: Math.abs(tempPoint.y - startPoint.y) };
    } else {
      raw_object = { points: pathPoints.map((p) => [p.x, p.y]) };
    }
    // Store as pending with display_scale applied (raw canvas coords → natural coords)
    const naturalRaw = convertToNaturalCoords(raw_object, drawMode, displayScale);
    setPendingShapes((prev) => [...prev, { id: generateTempId(), object_type: drawMode, raw_object: naturalRaw, display_scale: 1 }]);
    setStartPoint(null); setTempPoint(null); setPathPoints([]);
  };

  // Convert canvas coords to natural image coords
  const convertToNaturalCoords = (raw: Record<string, unknown>, mode: DrawMode, ds: number): Record<string, unknown> => {
    if (ds <= 0 || ds === 1) return raw;
    if (mode === "line") return { x1: Number(raw.x1) / ds, y1: Number(raw.y1) / ds, x2: Number(raw.x2) / ds, y2: Number(raw.y2) / ds };
    if (mode === "rect") return { x: Number(raw.x) / ds, y: Number(raw.y) / ds, width: Number(raw.width) / ds, height: Number(raw.height) / ds };
    const pts = Array.isArray(raw.points) ? (raw.points as number[][]).map(([px, py]) => [px / ds, py / ds]) : [];
    return { points: pts };
  };

  // Render pending shape on main canvas (scaled)
  const renderPendingOnCanvas = (s: PendingShape) => {
    const obj = s.raw_object;
    const ds = displayScale;
    if (s.object_type === "line") {
      return <line key={s.id} x1={Number(obj.x1) * ds} y1={Number(obj.y1) * ds} x2={Number(obj.x2) * ds} y2={Number(obj.y2) * ds} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="8 4" strokeLinecap="round" />;
    }
    if (s.object_type === "rect") {
      return <rect key={s.id} x={Number(obj.x) * ds} y={Number(obj.y) * ds} width={Number(obj.width) * ds} height={Number(obj.height) * ds} fill={hexToRgba(PENDING_COLOR, 0.15)} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="8 4" />;
    }
    const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map(([px, py]) => `${px * ds},${py * ds}`).join(" ") : "";
    return <polyline key={s.id} points={pts} fill="none" stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="8 4" />;
  };

  // ── Delete handlers ──
  const handleDeleteItem = async (uid: string) => {
    if (!selectedPlanId) return;
    try {
      const state = await deletePlanningItem(selectedPlanId, uid);
      setPlanningState(state);
    } catch { setError("שגיאה במחיקת פריט."); }
  };

  // ── Calibration ──
  const handleCalibrate = async () => {
    if (!selectedPlanId || !calStart || !calEnd || calibrationLengthM <= 0) return;
    setLoading(true);
    try {
      const state = await calibratePlanningScale(selectedPlanId, { x1: calStart.x, y1: calStart.y, x2: calEnd.x, y2: calEnd.y, display_scale: displayScale, real_length_m: calibrationLengthM });
      setPlanningState(state);
      setError("");
    } catch { setError("שגיאה בכיול סקייל."); } finally { setLoading(false); }
  };

  const handleCalMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const p = toLocalPoint(calibrationSurfaceRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setCalStart(p); setCalEnd(null); setCalTemp(p); setCalDrawing(true);
  };
  const handleCalMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!calDrawing) return;
    setCalTemp(toLocalPoint(calibrationSurfaceRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY));
  };
  const handleCalMouseUp: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!calDrawing) return;
    const p = toLocalPoint(calibrationSurfaceRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setCalEnd(p); setCalTemp(p); setCalDrawing(false);
  };

  // ── Opening / Wall prompts ──
  const handleResolveOpening = async (openingType: "door" | "window" | "none") => {
    if (!selectedPlanId || !openingPrompt) return;
    try {
      const state = await resolvePlanningOpening(selectedPlanId, openingPrompt.itemUid, { opening_type: openingType, gap_id: openingPrompt.gapId });
      setPlanningState(state); setOpeningPrompt(null);
    } catch { setError("שגיאה בשיוך פתח."); }
  };

  const handleResolveWall = async (isWall: boolean) => {
    if (!selectedPlanId || !wallPrompt) return;
    try {
      const state = await resolvePlanningWall(selectedPlanId, wallPrompt.itemUid, { is_wall: isWall });
      setPlanningState(state); setWallPrompt(null);
    } catch { setError("שגיאה באישור קיר."); }
  };

  // ── Finalize ──
  const handleFinalize = async () => {
    if (!selectedPlanId) return;
    setLoading(true);
    try {
      const state = await finalizePlanning(selectedPlanId);
      setPlanningState(state); setError("");
      setFinalizeNotice(`נשמר בהצלחה: ${new Date().toLocaleTimeString("he-IL")}`);
    } catch { setError("שגיאה בשמירה סופית של התכולה."); } finally { setLoading(false); }
  };

  // ── Section handlers ──
  const handleAddSection = async () => {
    if (!selectedPlanId || !planningState) return;
    if (!secContractor.trim() && !secWorker.trim()) {
      setError("יש למלא לפחות שם קבלן או שם עובד.");
      return;
    }
    setLoading(true);
    try {
      // Compute section rect in natural coords
      let x = 0, y = 0, width = 0, height = 0;
      if (secStart && secEnd) {
        const secImgW = secImageRef.current?.naturalWidth || planningState.image_width || 1;
        const secImgDisplayW = secImageRef.current?.clientWidth || displaySize.width || 1;
        const sf = secImgW / secImgDisplayW;
        x = Math.min(secStart.x, secEnd.x) * sf;
        y = Math.min(secStart.y, secEnd.y) * sf;
        width = Math.abs(secEnd.x - secStart.x) * sf;
        height = Math.abs(secEnd.y - secStart.y) * sf;
      }
      const state = await addWorkSection(selectedPlanId, {
        name: secName.trim(),
        contractor: secContractor.trim(),
        worker: secWorker.trim(),
        color: secColor,
        x, y, width, height,
      });
      setPlanningState(state);
      // Reset form
      setSecContractor(""); setSecWorker(""); setSecName("");
      setSecColor("#6366f1"); setSecStart(null); setSecEnd(null); setSecTemp(null);
      setError("");
    } catch { setError("שגיאה בהוספת גזרה."); } finally { setLoading(false); }
  };

  const handleDeleteSection = async (uid: string) => {
    if (!selectedPlanId) return;
    try {
      const state = await deleteWorkSection(selectedPlanId, uid);
      setPlanningState(state);
    } catch { setError("שגיאה במחיקת גזרה."); }
  };

  // ── Section canvas mouse handlers ──
  const handleSecMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const p = toLocalPoint(secCanvasRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setSecStart(p); setSecEnd(null); setSecTemp(p); setSecDrawing(true);
  };
  const handleSecMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!secDrawing) return;
    setSecTemp(toLocalPoint(secCanvasRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY));
  };
  const handleSecMouseUp: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!secDrawing) return;
    const p = toLocalPoint(secCanvasRef as unknown as React.RefObject<HTMLElement | null>, e.clientX, e.clientY);
    setSecEnd(p); setSecTemp(p); setSecDrawing(false);
  };

  // ── Auto-analyze handlers ──
  const handleAutoAnalyze = async () => {
    if (!selectedPlanId) return;
    setAutoLoading(true);
    try {
      const result = await autoAnalyzePlan(selectedPlanId);
      setAutoSegments(result.segments);
      setAutoVisionData(result.vision_data ?? null);
      // Pre-select all
      setAutoSelected(new Set(result.segments.map(s => s.segment_id)));
      // Pre-fill category keys: find best match for walls; leave blank for fixtures
      const keys: Record<string, string> = {};
      if (planningState) {
        for (const seg of result.segments) {
          if (seg.element_class === "fixture") {
            keys[seg.segment_id] = "";
          } else {
            const match = Object.values(planningState.categories).find(
              c => c.type === seg.suggested_type && c.subtype === seg.suggested_subtype
            );
            keys[seg.segment_id] = match?.key ?? "";
          }
        }
      }
      setAutoConfirmedKeys(keys);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e)
        ? ((e.response?.data as { detail?: string })?.detail || e.message)
        : e instanceof Error ? e.message : String(e);
      setError(`שגיאה בניתוח אוטומטי: ${detail}`);
    }
    finally { setAutoLoading(false); }
  };

  const handleConfirmAutoSegments = async (selectedOnly: boolean) => {
    if (!selectedPlanId || !autoSegments || !planningState) return;
    const toConfirm = autoSegments.filter(s =>
      (!selectedOnly || autoSelected.has(s.segment_id)) &&
      autoConfirmedKeys[s.segment_id]
    );
    if (toConfirm.length === 0) { setError("בחר לפחות אזור אחד עם קטגוריה."); return; }
    setLoading(true);
    try {
      let lastState = planningState;
      for (const seg of toConfirm) {
        const catKey = autoConfirmedKeys[seg.segment_id];
        const { data } = await apiClient.post<PlanningState>(
          `/manager/planning/${encodeURIComponent(selectedPlanId)}/confirm-auto-segment`,
          { segment_id: seg.segment_id, category_key: catKey, bbox: seg.bbox }
        );
        lastState = data;
      }
      setPlanningState(lastState);
      setAutoSegments(null);
      setAutoVisionData(null);
      setAutoSelected(new Set());
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`שגיאה באישור אזורים: ${detail}`);
    } finally { setLoading(false); }
  };

  // ── Zone handlers ──
  const handleZoneMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const rect = zoneCanvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const p = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setZoneDrawing(true); setZoneStart(p); setZoneEnd(null); setZoneTemp(p);
  };
  const handleZoneMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!zoneDrawing) return;
    const rect = zoneCanvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    setZoneTemp({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };
  const handleZoneMouseUp: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (!zoneDrawing || !zoneStart) { setZoneDrawing(false); return; }
    const rect = zoneCanvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const p = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setZoneEnd(p); setZoneDrawing(false);
  };

  const handleAddZone = async () => {
    if (!selectedPlanId || !zoneStart || !zoneEnd || !zoneCatKey) {
      setError("בחר קטגוריה וצייר מלבן."); return;
    }
    const x = Math.min(zoneStart.x, zoneEnd.x) / displayScale;
    const y = Math.min(zoneStart.y, zoneEnd.y) / displayScale;
    const w = Math.abs(zoneEnd.x - zoneStart.x) / displayScale;
    const h = Math.abs(zoneEnd.y - zoneStart.y) / displayScale;
    setLoading(true);
    try {
      const state = await addZoneItem(selectedPlanId, { category_key: zoneCatKey, x, y, width: w, height: h });
      setPlanningState(state);
      setZoneStart(null); setZoneEnd(null); setZoneTemp(null);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`שגיאה בהוספת אזור: ${detail}`);
    } finally { setLoading(false); }
  };

  // ── Text item handlers ──
  const handleAddTextRow = () => {
    setTextRows(prev => [...prev, { category_key: "__manual__", description: "", quantity: 1, unit: "יח׳", note: "" }]);
  };
  const handleTextRowChange = (idx: number, field: keyof TextItemPayload, value: string | number) => {
    setTextRows(prev => prev.map((r, i) => i === idx ? { ...r, [field]: value } : r));
  };
  const handleRemoveTextRow = (idx: number) => {
    setTextRows(prev => prev.filter((_, i) => i !== idx));
  };
  const handleSaveTextRows = async () => {
    if (!selectedPlanId) return;
    const valid = textRows.filter(r => r.description.trim() && r.quantity > 0);
    if (valid.length === 0) { setError("הזן לפחות פריט אחד עם תיאור וכמות."); return; }
    setLoading(true);
    try {
      let lastState = planningState!;
      for (const row of valid) {
        lastState = await addTextItem(selectedPlanId, row);
      }
      setPlanningState(lastState);
      setTextRows([{ category_key: "__manual__", description: "", quantity: 1, unit: "יח׳", note: "" }]);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`שגיאה בשמירת פריטי טקסט: ${detail}`);
    } finally { setLoading(false); }
  };

  const canStep2 = selectedPlanId.length > 0;
  const canStep3 = planningState != null;
  const canStep4 = planningState != null && planningState.items.length > 0;

  const stepTitle = ["", "שלב 1: בחירת תוכנית", "שלב 2: כיול סקייל", "שלב 3: סימון תכולה", "שלב 4: כתב כמויות", "שלב 5: גזרות עבודה"][step];

  const openingsSummary = React.useMemo(() => {
    if (!planningState) return { doorCount: 0, windowCount: 0, deductedLengthM: 0 };
    let doorCount = 0, windowCount = 0, deductedLengthM = 0;
    for (const item of planningState.items) {
      const kind = item.analysis?.resolved_opening_type;
      if (kind === "door") doorCount++;
      if (kind === "window") windowCount++;
      deductedLengthM += Number(item.analysis?.deducted_length_m ?? 0);
    }
    return { doorCount, windowCount, deductedLengthM };
  }, [planningState]);

  const activeColor = "#10B981";

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
        <h2 className="text-lg font-semibold text-[#31333F] mb-1">🧱 הגדרת תכולה</h2>
        <p className="text-xs text-slate-500">זרימת עבודה: בחירת תוכנית → כיול סקייל → סימון תכולה → BOQ ושמירה.</p>
        {error && <div className="mt-3"><ErrorAlert message={error} onDismiss={() => setError("")} /></div>}
      </div>

      {/* Opening prompt */}
      {openingPrompt && (
        <div className="rounded-lg p-4" style={{ background: "#FFFBEB", border: "1px solid #FCD34D", borderRight: "5px solid #F59E0B", boxShadow: "0 2px 8px rgba(245,158,11,0.12)" }}>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 18 }}>🚪</span>
            <span className="text-sm font-bold" style={{ color: "#92400E" }}>זוהה פתח באמצע הקיר</span>
          </div>
          <div className="text-sm mb-3" style={{ color: "#B45309" }}>
            פתח באורך משוער <strong>{openingPrompt.gapLengthM?.toFixed(2)} מ׳</strong> — האם מדובר בדלת, חלון, או לא פתח?
          </div>
          <div className="flex gap-2 flex-wrap">
            <button type="button" onClick={() => void handleResolveOpening("door")} style={{ padding: "8px 18px", borderRadius: 9, background: "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(255,75,75,0.3)" }}>🚪 דלת</button>
            <button type="button" onClick={() => void handleResolveOpening("window")} style={{ padding: "8px 18px", borderRadius: 9, background: "#F97316", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(249,115,22,0.3)" }}>🪟 חלון</button>
            <button type="button" onClick={() => void handleResolveOpening("none")} style={{ padding: "8px 16px", borderRadius: 9, background: "#fff", color: "#64748b", border: "1px solid #CBD5E1", fontWeight: 600, fontSize: 13, cursor: "pointer" }}>לא פתח</button>
          </div>
        </div>
      )}

      {/* Wall confirmation prompt */}
      {wallPrompt && (
        <div className="rounded-lg p-4" style={{ background: "#EFF6FF", border: "1px solid #93C5FD", borderRight: "5px solid #3B82F6", boxShadow: "0 2px 8px rgba(59,130,246,0.1)" }}>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 18 }}>🧱</span>
            <span className="text-sm font-bold" style={{ color: "#1E3A8A" }}>סימון גבולי על קיר</span>
          </div>
          <div className="text-sm mb-3" style={{ color: "#1D4ED8" }}>
            המערכת לא בטוחה שזה קיר (חפיפה: <strong>{wallPrompt.overlapRatio != null ? `${Math.round(wallPrompt.overlapRatio * 100)}%` : "לא ידוע"}</strong>). האם לשמור כסימון קיר?
          </div>
          <div className="flex gap-2 flex-wrap">
            <button type="button" onClick={() => void handleResolveWall(true)} style={{ padding: "8px 18px", borderRadius: 9, background: "#1B3A6B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(27,58,107,0.3)" }}>✓ כן, זה קיר</button>
            <button type="button" onClick={() => void handleResolveWall(false)} style={{ padding: "8px 16px", borderRadius: 9, background: "#fff", color: "#64748b", border: "1px solid #CBD5E1", fontWeight: 600, fontSize: 13, cursor: "pointer" }}>✕ להתעלם</button>
          </div>
        </div>
      )}

      {/* Loading overlay for plan load */}
      {loading && !planningState && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", background: "#F0F9FF", border: "1px solid #BAE6FD", borderRadius: 10, fontSize: 13, color: "#0369A1" }}>
          <span style={{ animation: "spin 1s linear infinite", display: "inline-block", fontSize: 16 }}>⏳</span>
          <span>טוען נתוני תכנון...</span>
        </div>
      )}

      {/* Step nav — horizontal stepper */}
      <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm px-5 py-4">
        <div style={{ display: "flex", alignItems: "center", gap: 0, overflowX: "auto" }}>
          {(
            [
              { s: 1 as WizardStep, label: "בחירת תוכנית",  icon: "📁", canGo: true },
              { s: 2 as WizardStep, label: "כיול סקייל",    icon: "📏", canGo: canStep2 },
              { s: 3 as WizardStep, label: "סימון תכולה",   icon: "✏️", canGo: canStep3 },
              { s: 4 as WizardStep, label: "כתב כמויות",    icon: "📋", canGo: canStep4 },
              { s: 5 as WizardStep, label: "גזרות עבודה",   icon: "🗺️", canGo: canStep4 },
            ]
          ).map(({ s, label, icon, canGo }, idx) => {
            const isActive = step === s;
            const isDone   = step > s;
            const isLocked = !canGo;
            return (
              <React.Fragment key={s}>
                <button
                  type="button"
                  onClick={() => {
                    if (s === 1) setStep(1);
                    if (s === 2 && canStep2) setStep(2);
                    if (s === 3 && canStep3) setStep(3);
                    if (s === 4 && canStep4) setStep(4);
                    if (s === 5 && canStep4) setStep(5);
                  }}
                  disabled={isLocked}
                  style={{ flexShrink: 0, textAlign: "center", background: "none", border: "none", padding: "0 4px", cursor: isLocked ? "not-allowed" : "pointer", opacity: isLocked ? 0.45 : 1 }}
                >
                  <div style={{
                    width: 36, height: 36, borderRadius: "50%", margin: "0 auto 5px",
                    background: isActive ? "#FF4B4B" : isDone ? "#10B981" : "#F1F5F9",
                    color: isActive || isDone ? "#fff" : "#64748b",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: isActive || isDone ? 15 : 14, fontWeight: 700,
                    boxShadow: isActive ? "0 0 0 4px rgba(255,75,75,0.18)" : isDone ? "0 0 0 3px rgba(16,185,129,0.14)" : "none",
                    transition: "all 0.2s",
                  }}>
                    {isDone ? "✓" : <span style={{ fontSize: 11, fontWeight: 700 }}>{s}</span>}
                  </div>
                  <div style={{ fontSize: 10.5, color: isActive ? "#FF4B4B" : isDone ? "#10B981" : "#94a3b8", fontWeight: isActive ? 700 : 500, whiteSpace: "nowrap", lineHeight: 1.2 }}>
                    {icon} {label}
                  </div>
                </button>
                {idx < 4 && (
                  <div style={{ flex: 1, height: 2, minWidth: 16, background: step > idx + 1 ? "#10B981" : "#e2e8f0", marginBottom: 20, transition: "background 0.3s" }} />
                )}
              </React.Fragment>
            );
          })}
        </div>
        {loading && (
          <div style={{ textAlign: "center", marginTop: 6, fontSize: 11, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center", gap: 5 }}>
            <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: "50%", border: "2px solid #CBD5E1", borderTopColor: "#94a3b8", animation: "spin 0.7s linear infinite" }} />
            טוען...
            <style>{`@keyframes spin { to { transform: rotate(360deg); } } @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.7; } }`}</style>
          </div>
        )}
      </div>

      {/* ── STEP 1: Pick plan ── */}
      {step === 1 && (
        <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-5 space-y-4">
          <div>
            <p className="text-base font-bold text-[#1B3A6B] mb-1">📁 בחר תוכנית לעבודה</p>
            <p className="text-xs text-slate-400">בחר מהרשימה תוכנית שהועלתה בסדנת עבודה.</p>
          </div>
          {plans.length === 0 ? (
            <div className="rounded-lg p-4 text-sm text-amber-800" style={{ background: "#FFFBEB", border: "1px solid #FCD34D" }}>
              ⚠️ אין תוכניות זמינות. העלה קודם תוכנית ב&quot;סדנת עבודה&quot;.
            </div>
          ) : (
            <select
              className="w-full bg-white border-2 border-slate-300 rounded-lg px-3 py-2.5 text-sm font-medium"
              style={{ borderColor: selectedPlanId ? "#1B3A6B" : "#CBD5E1", outline: "none" }}
              value={selectedPlanId}
              onChange={(e) => setSelectedPlanId(e.target.value)}
            >
              {plans.map((p) => <option key={p.id} value={p.id}>{p.plan_name}</option>)}
            </select>
          )}
          {selectedPlan && (
            <div className="rounded-lg p-3 flex items-center gap-3" style={{ background: "#EFF6FF", border: "1px solid #BFDBFE" }}>
              <span style={{ fontSize: 28 }}>📐</span>
              <div className="text-sm">
                <div className="font-bold text-[#1E3A8A]">{selectedPlan.plan_name}</div>
                <div className="text-xs text-blue-600 mt-0.5">
                  {selectedPlan.total_wall_length_m != null && <span>אורך קירות: <strong>{selectedPlan.total_wall_length_m.toFixed(1)} מ׳</strong></span>}
                  {selectedPlan.concrete_length_m != null && <span className="mr-3">בטון: {selectedPlan.concrete_length_m.toFixed(1)} מ׳</span>}
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-end">
            <button type="button" onClick={() => setStep(2)} disabled={!selectedPlanId}
              style={{ padding: "10px 28px", borderRadius: 10, background: selectedPlanId ? "#FF4B4B" : "#CBD5E1", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: selectedPlanId ? "pointer" : "not-allowed", boxShadow: selectedPlanId ? "0 3px 10px rgba(255,75,75,0.3)" : "none", transition: "all 0.15s" }}>
              המשך לשלב 2 ←
            </button>
          </div>
        </div>
      )}

      {/* ── STEP 2: Calibration only ── */}
      {step === 2 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr,300px] gap-4">
          {/* Calibration canvas */}
          <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
            <p className="text-sm font-semibold text-[#31333F] mb-1">שלב 2: כיול סקייל</p>
            <p className="text-xs text-slate-500 mb-3">גרור קו על אורך ידוע בתוכנית, הזן את האורך האמיתי ולחץ &quot;עדכן סקייל&quot;.</p>
            <div className="relative border border-slate-300 rounded-lg overflow-hidden w-fit cursor-crosshair bg-slate-50">
              <img
                ref={calibrationImageRef}
                src={imageUrl}
                alt="plan"
                className="block"
                style={{ width: displaySize.width, height: displaySize.height }}
                onLoad={() => updateDisplaySizeFromImage(calibrationImageRef.current)}
                draggable={false}
              />
              <svg
                ref={calibrationSurfaceRef}
                width={displaySize.width}
                height={displaySize.height}
                className="absolute inset-0"
                onMouseDown={handleCalMouseDown}
                onMouseMove={handleCalMouseMove}
                onMouseUp={handleCalMouseUp}
                onMouseLeave={() => setCalDrawing(false)}
              >
                {calStart && (calEnd || calTemp) && (
                  <line x1={calStart.x} y1={calStart.y} x2={(calEnd ?? calTemp)?.x ?? calStart.x} y2={(calEnd ?? calTemp)?.y ?? calStart.y} stroke="#FF4B4B" strokeWidth={3} />
                )}
                {calStart && <circle cx={calStart.x} cy={calStart.y} r={5} fill="#FF4B4B" />}
                {calEnd && <circle cx={calEnd.x} cy={calEnd.y} r={5} fill="#FF4B4B" />}
              </svg>
            </div>
          </div>

          {/* Calibration controls */}
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
              <p className="text-sm font-semibold text-[#31333F]">בקרת כיול</p>
              <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-700 space-y-1">
                <p>סקייל נוכחי: <span className="font-semibold text-[#1B3A6B]">{planningState.scale_px_per_meter.toFixed(1)} px/m</span></p>
                {calStart && calEnd && (
                  <p className="text-slate-500">
                    קו שנגרר: {Math.round(Math.hypot(calEnd.x - calStart.x, calEnd.y - calStart.y))} px
                  </p>
                )}
              </div>
              <label className="text-xs block">
                אורך אמיתי (מטר)
                <input type="number" className="mt-1 w-full bg-white border border-slate-300 rounded-lg px-2 py-1.5 text-sm" min={0.1} step={0.1} value={calibrationLengthM} onChange={(e) => setCalibrationLengthM(Number(e.target.value))} />
              </label>
              <button type="button" onClick={handleCalibrate} disabled={!calStart || !calEnd}
                style={{ width: "100%", padding: "10px 0", borderRadius: 9, background: (!calStart || !calEnd) ? "#CBD5E1" : "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: (!calStart || !calEnd) ? "not-allowed" : "pointer", boxShadow: (!calStart || !calEnd) ? "none" : "0 2px 8px rgba(255,75,75,0.28)", transition: "all 0.15s" }}>
                📏 עדכן סקייל
              </button>
              <button type="button" onClick={() => { setCalStart(null); setCalEnd(null); setCalTemp(null); setCalDrawing(false); }}
                style={{ width: "100%", padding: "8px 0", borderRadius: 9, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                נקה קו
              </button>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800 space-y-1">
              <p className="font-semibold">💡 טיפ</p>
              <p>גרור קו על קיר שאורכו ידוע (למשל 5 מטר). הכיול יחושב אוטומטית.</p>
              <p>אחרי כיול מדויק, תוצאות המדידה יהיו מדויקות יותר.</p>
            </div>
          </div>

          <div className="xl:col-span-2 flex justify-between">
            <button type="button" onClick={() => setStep(1)}
              style={{ padding: "9px 18px", borderRadius: 9, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
              ← חזור לשלב 1
            </button>
            <button type="button" onClick={() => setStep(3)}
              style={{ padding: "10px 24px", borderRadius: 10, background: "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 3px 10px rgba(255,75,75,0.3)" }}>
              המשך לשלב 3 ←
            </button>
          </div>
        </div>
      )}

      {/* ── ZoomModal ── */}
      {zoomModalOpen && planningState && (
        <ZoomModal
          imageUrl={imageUrl}
          planningState={planningState}
          pendingShapes={pendingShapes}
          displayScale={displayScale}
          onClose={() => setZoomModalOpen(false)}
          onDrawComplete={(shape) => setPendingShapes((prev) => [...prev, shape])}
          onAssignCategory={() => {
            if (pendingShapes.length > 0) setCategoryPickerOpen(true);
          }}
          onDeletePending={(id) => setPendingShapes((prev) => prev.filter((s) => s.id !== id))}
          onDeleteItem={handleDeleteItem}
        />
      )}

      {/* ── Category Picker Modal ── */}
      {categoryPickerOpen && planningState && (
        <CategoryPickerModal
          categories={planningState.categories}
          pendingCount={pendingShapes.length}
          onPick={(key) => { void handleAssignCategory(key); }}
          onCreateAndPick={(type, subtype, param, note) => { void handleCreateAndAssign(type, subtype, param, note); }}
          onCancel={() => setCategoryPickerOpen(false)}
        />
      )}

      {/* ── STEP 3: 4 tabs ── */}
      {step === 3 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr,300px] gap-4">
          {/* Main area */}
          <div className="space-y-3">
            {/* Tab bar */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-2 flex gap-2 flex-wrap items-center">
              {(["auto","zone","manual","text"] as Step3Tab[]).map(tab => {
                const labels: Record<Step3Tab, string> = { auto: "🤖 ניתוח אוטומטי", zone: "🎨 אזורים", manual: "✏️ ציור ידני", text: "📋 פריטים חופשיים" };
                const active = step3Tab === tab;
                return (
                  <button key={tab} type="button" onClick={() => setStep3Tab(tab)}
                    style={{
                      padding: "8px 14px", borderRadius: 9, fontSize: 12.5, fontWeight: active ? 700 : 500,
                      border: active ? "none" : "1.5px solid #E2E8F0",
                      background: active ? "#1B3A6B" : "#F8FAFC",
                      color: active ? "#fff" : "#475569",
                      cursor: "pointer",
                      boxShadow: active ? "0 2px 8px rgba(27,58,107,0.22)" : "none",
                      transition: "all 0.15s",
                    }}>
                    {labels[tab]}
                  </button>
                );
              })}
              <div style={{ marginRight: "auto", display: "flex", alignItems: "center", gap: 6 }}>
                <button type="button" onClick={() => setZoomModalOpen(true)}
                  style={{ background: "#F1F5F9", color: "#1B3A6B", border: "1.5px solid #CBD5E1", borderRadius: 8, padding: "7px 12px", fontSize: 12, cursor: "pointer", fontWeight: 600 }}>
                  🔍 הגדלה
                </button>
                {pendingShapes.length > 0 && (
                  <button type="button" onClick={() => setCategoryPickerOpen(true)}
                    style={{ background: PENDING_COLOR, color: "#fff", border: "none", borderRadius: 9, padding: "7px 14px", fontSize: 12, cursor: "pointer", fontWeight: 700, boxShadow: "0 2px 8px rgba(245,158,11,0.35)", animation: "pulse 1.8s infinite" }}>
                    📂 שייך {pendingShapes.length}
                  </button>
                )}
              </div>
            </div>

            {/* ── TAB: AUTO ── */}
            {step3Tab === "auto" && (
              <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
                {/* Header */}
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <p className="text-sm font-semibold text-[#31333F]">ניתוח אוטומטי</p>
                    <p className="text-xs text-slate-500 mt-0.5">המערכת מזהה אזורי קירות ומציעה קטגוריות. אשר הכל או בחר חלק.</p>
                  </div>
                  <button type="button" onClick={() => void handleAutoAnalyze()} disabled={autoLoading}
                    style={{ padding: "9px 20px", borderRadius: 10, background: autoLoading ? "#475569" : "#1B3A6B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: autoLoading ? "not-allowed" : "pointer", boxShadow: autoLoading ? "none" : "0 2px 10px rgba(27,58,107,0.28)", transition: "all 0.15s", display: "flex", alignItems: "center", gap: 6 }}>
                    {autoLoading ? <><span style={{ display: "inline-block", width: 14, height: 14, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.4)", borderTopColor: "#fff", animation: "spin 0.7s linear infinite" }} />מנתח...</> : "🤖 נתח אוטומטית"}
                  </button>
                </div>

                {/* Before analysis */}
                {autoSegments === null && (
                  <div className="overflow-auto border border-slate-200 rounded-lg bg-slate-50">
                    <div className="relative w-fit">
                      <img src={imageUrl} alt="plan" className="block"
                        style={{ width: displaySize.width, height: displaySize.height }} />
                      <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
                        <div className="bg-white/90 rounded-xl px-6 py-4 text-center shadow-lg">
                          <p className="text-sm font-semibold text-[#1B3A6B] mb-2">לחץ &quot;נתח אוטומטית&quot; להתחיל</p>
                          <p className="text-xs text-slate-500">המערכת תסרוק את השרטוט ותסמן אזורים</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* No segments found */}
                {autoSegments !== null && autoSegments.length === 0 && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
                    לא זוהו אזורים. ייתכן שהתוכנית לא עובדה עדיין — נסה להעלות ולנתח בסדנת עבודה.
                  </div>
                )}

                {/* Segments found — image + table side by side */}
                {autoSegments !== null && autoSegments.length > 0 && (
                  <>
                    {/* Image with segment overlays */}
                    <div className="overflow-auto border border-slate-200 rounded-lg bg-slate-50">
                      <div className="relative w-fit select-none">
                        <img ref={drawingImageRef} src={imageUrl} alt="plan" className="block"
                          style={{ width: displaySize.width, height: displaySize.height }}
                          onLoad={() => updateDisplaySizeFromImage(drawingImageRef.current)} />
                        <svg width={displaySize.width} height={displaySize.height}
                          className="absolute inset-0 pointer-events-none">
                          {autoSegments.map((seg, idx) => {
                            const [bx, by, bw, bh] = seg.bbox.map(v => v * displayScale);
                            const checked = autoSelected.has(seg.segment_id);
                            const isFixture = seg.element_class === "fixture";
                            const color = isFixture
                              ? (checked ? "#7C3AED" : "#A78BFA")
                              : seg.confidence >= 0.8 ? "#10B981" : seg.confidence >= 0.6 ? "#F59E0B" : "#EF4444";
                            const opacity = checked ? 0.35 : 0.1;
                            return (
                              <g key={seg.segment_id}>
                                <rect x={bx} y={by} width={bw} height={bh}
                                  fill={color} fillOpacity={opacity}
                                  stroke={color} strokeWidth={checked ? 2.5 : 1}
                                  strokeDasharray={isFixture ? "5 3" : (checked ? "none" : "6 3")} />
                                <text x={bx + 4} y={by + 14} fill={color}
                                  fontSize={Math.max(9, Math.min(13, bw / 8))}
                                  fontWeight="700" style={{ pointerEvents: "none" }}>
                                  {isFixture ? "🔧" : idx + 1}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                    </div>

                    {/* Controls */}
                    {(() => {
                      const wallSegs = autoSegments.filter(s => s.element_class !== "fixture");
                      const fixSegs  = autoSegments.filter(s => s.element_class === "fixture");
                      return (
                        <>
                          <div className="flex gap-2 flex-wrap items-center">
                            <span className="text-xs text-slate-500">
                              {wallSegs.length} קירות · {fixSegs.length} אביזרים
                            </span>
                            <button type="button" onClick={() => setAutoSelected(new Set(autoSegments.map(s => s.segment_id)))}
                              className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">בחר הכל</button>
                            <button type="button" onClick={() => setAutoSelected(new Set())}
                              className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">בטל הכל</button>
                            <button type="button" onClick={() => setAutoSelected(new Set(autoSegments.filter(s => s.element_class !== "fixture" && s.confidence >= 0.8).map(s => s.segment_id)))}
                              className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">קירות ביטחון {">"}80%</button>
                          </div>

                          {/* ── Walls table ── */}
                          {wallSegs.length > 0 && (
                            <div className="overflow-x-auto">
                              <div className="text-xs font-semibold text-slate-500 mb-1 flex items-center gap-1">🧱 קירות שזוהו</div>
                              <table className="w-full text-xs border-collapse">
                                <thead>
                                  <tr className="bg-slate-50 text-slate-500">
                                    <th className="p-2 text-right font-medium border-b border-slate-200 w-6">#</th>
                                    <th className="p-2 text-right font-medium border-b border-slate-200 w-8">✓</th>
                                    <th className="p-2 text-right font-medium border-b border-slate-200">הצעה</th>
                                    <th className="p-2 text-right font-medium border-b border-slate-200">אורך</th>
                                    <th className="p-2 text-right font-medium border-b border-slate-200">ביטחון</th>
                                    <th className="p-2 text-right font-medium border-b border-slate-200">קטגוריה</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {wallSegs.map((seg, idx) => {
                                    const checked = autoSelected.has(seg.segment_id);
                                    const catKey = autoConfirmedKeys[seg.segment_id] ?? "";
                                    const conf = seg.confidence;
                                    const confColor = conf >= 0.8 ? "#10B981" : conf >= 0.6 ? "#F59E0B" : "#EF4444";
                                    return (
                                      <tr key={seg.segment_id}
                                        className={`border-b border-slate-100 cursor-pointer ${checked ? "bg-blue-50" : "hover:bg-slate-50"}`}
                                        onClick={() => setAutoSelected(prev => { const n = new Set(prev); checked ? n.delete(seg.segment_id) : n.add(seg.segment_id); return n; })}>
                                        <td className="p-2 text-slate-400 font-mono">{idx + 1}</td>
                                        <td className="p-2"><input type="checkbox" checked={checked} readOnly /></td>
                                        <td className="p-2 text-slate-600">{seg.suggested_type} / {seg.suggested_subtype}</td>
                                        <td className="p-2 font-medium">{seg.length_m.toFixed(1)}מ׳</td>
                                        <td className="p-2"><span style={{ color: confColor, fontWeight: 600 }}>{Math.round(conf * 100)}%</span></td>
                                        <td className="p-2" onClick={e => e.stopPropagation()}>
                                          <select value={catKey}
                                            onChange={e => setAutoConfirmedKeys(prev => ({ ...prev, [seg.segment_id]: e.target.value }))}
                                            className="border border-slate-300 rounded px-1 py-0.5 text-xs w-full" style={{ minWidth: 110 }}>
                                            <option value="">-- בחר --</option>
                                            {Object.values(planningState.categories).map(c => (
                                              <option key={c.key} value={c.key}>{c.type} / {c.subtype}</option>
                                            ))}
                                          </select>
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          )}

                          {/* ── Fixtures table ── */}
                          {fixSegs.length > 0 && (
                            <div className="overflow-x-auto">
                              <div className="text-xs font-semibold mb-1 flex items-center gap-1" style={{ color: "#7C3AED" }}>🔧 אביזרים ואלמנטים שזוהו</div>
                              <table className="w-full text-xs border-collapse">
                                <thead>
                                  <tr className="text-slate-500" style={{ background: "#F5F3FF" }}>
                                    <th className="p-2 text-right font-medium border-b border-purple-100 w-6">#</th>
                                    <th className="p-2 text-right font-medium border-b border-purple-100 w-8">✓</th>
                                    <th className="p-2 text-right font-medium border-b border-purple-100">זיהוי</th>
                                    <th className="p-2 text-right font-medium border-b border-purple-100">שטח</th>
                                    <th className="p-2 text-right font-medium border-b border-purple-100">קטגוריה (אופציונלי)</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {fixSegs.map((seg, idx) => {
                                    const checked = autoSelected.has(seg.segment_id);
                                    const catKey = autoConfirmedKeys[seg.segment_id] ?? "";
                                    return (
                                      <tr key={seg.segment_id}
                                        className="border-b border-purple-50 cursor-pointer"
                                        style={{ background: checked ? "#EDE9FE" : undefined }}
                                        onClick={() => setAutoSelected(prev => { const n = new Set(prev); checked ? n.delete(seg.segment_id) : n.add(seg.segment_id); return n; })}>
                                        <td className="p-2 text-slate-400 font-mono">{idx + 1}</td>
                                        <td className="p-2"><input type="checkbox" checked={checked} readOnly /></td>
                                        <td className="p-2 font-medium" style={{ color: "#7C3AED" }}>{seg.label}</td>
                                        <td className="p-2">{seg.area_m2.toFixed(2)} מ"ר</td>
                                        <td className="p-2" onClick={e => e.stopPropagation()}>
                                          <select value={catKey}
                                            onChange={e => setAutoConfirmedKeys(prev => ({ ...prev, [seg.segment_id]: e.target.value }))}
                                            className="border border-purple-200 rounded px-1 py-0.5 text-xs w-full" style={{ minWidth: 110 }}>
                                            <option value="">-- ללא קטגוריה --</option>
                                            {Object.values(planningState.categories).map(c => (
                                              <option key={c.key} value={c.key}>{c.type} / {c.subtype}</option>
                                            ))}
                                          </select>
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </>
                      );
                    })()}

                    <div style={{
                      position: "sticky",
                      bottom: 0,
                      background: "#fff",
                      borderTop: "2px solid #F1F5F9",
                      padding: "12px 0 6px",
                      marginTop: 10,
                      zIndex: 5,
                    }}>
                      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                        <button type="button" disabled={loading}
                          onClick={() => void handleConfirmAutoSegments(false)}
                          style={{ padding: "10px 22px", borderRadius: 10, background: loading ? "#94a3b8" : "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer", boxShadow: loading ? "none" : "0 3px 10px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
                          {loading ? "שומר..." : "✓ אשר הכל"}
                        </button>
                        <button type="button" disabled={loading || autoSelected.size === 0}
                          onClick={() => void handleConfirmAutoSegments(true)}
                          style={{ padding: "10px 22px", borderRadius: 10, background: (loading || autoSelected.size === 0) ? "#94a3b8" : "#1B3A6B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: (loading || autoSelected.size === 0) ? "not-allowed" : "pointer", boxShadow: (loading || autoSelected.size === 0) ? "none" : "0 3px 10px rgba(27,58,107,0.25)", transition: "all 0.15s" }}>
                          {loading ? "שומר..." : `✓ אשר נבחרים (${autoSelected.size})`}
                        </button>
                        <button type="button" onClick={() => { setAutoSegments(null); setAutoVisionData(null); }}
                          style={{ padding: "10px 16px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                          נקה
                        </button>
                      </div>
                    </div>
                  </>
                )}

                {/* Vision Data Panel — shown whenever vision_data is available */}
                {autoVisionData && (() => {
                  const vd = autoVisionData;
                  const hasRooms = vd.rooms && vd.rooms.length > 0;
                  const hasElements = vd.elements && vd.elements.length > 0;
                  const hasMaterials = vd.materials && vd.materials.length > 0;
                  const hasDims = vd.dimensions && vd.dimensions.length > 0;
                  const hasSystems = vd.systems && Object.keys(vd.systems).length > 0;
                  const hasNotes = vd.execution_notes && vd.execution_notes.length > 0;
                  if (!hasRooms && !hasElements && !hasMaterials && !hasDims && !hasSystems && !hasNotes) return null;
                  return (
                    <div style={{ marginTop: 20, borderTop: "2px solid #E2E8F0", paddingTop: 16 }}>
                      <div style={{ fontWeight: 700, fontSize: 14, color: "#1B3A6B", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                        🔍 נתוני Vision — מה המערכת מצאה בתוכנית
                      </div>

                      {/* Title-block info */}
                      {(vd.plan_title || vd.project_name || vd.scale || vd.architect || vd.date || vd.sheet_number) && (
                        <div style={{ background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "10px 14px", marginBottom: 12, fontSize: 13 }}>
                          <div style={{ fontWeight: 600, color: "#475569", marginBottom: 6 }}>פרטי תוכנית</div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
                            {vd.plan_title && <div><span style={{ color: "#94A3B8" }}>כותרת: </span>{vd.plan_title}</div>}
                            {vd.project_name && <div><span style={{ color: "#94A3B8" }}>פרויקט: </span>{vd.project_name}</div>}
                            {vd.scale && <div><span style={{ color: "#94A3B8" }}>קנ"מ: </span>{vd.scale}</div>}
                            {vd.architect && <div><span style={{ color: "#94A3B8" }}>אדריכל: </span>{vd.architect}</div>}
                            {vd.date && <div><span style={{ color: "#94A3B8" }}>תאריך: </span>{vd.date}</div>}
                            {vd.sheet_number && <div><span style={{ color: "#94A3B8" }}>מספר דף: </span>{vd.sheet_number}</div>}
                            {vd.total_area_m2 && <div><span style={{ color: "#94A3B8" }}>שטח כולל: </span>{vd.total_area_m2} מ"ר</div>}
                            {vd.status && <div><span style={{ color: "#94A3B8" }}>סטטוס: </span>{vd.status}</div>}
                          </div>
                        </div>
                      )}

                      {/* Rooms */}
                      {hasRooms && (
                        <div style={{ marginBottom: 14 }}>
                          <div style={{ fontWeight: 600, fontSize: 13, color: "#334155", marginBottom: 6 }}>חדרים / מרחבים ({vd.rooms!.length})</div>
                          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                            <thead>
                              <tr style={{ background: "#F1F5F9" }}>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>שם</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>שטח</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>מידות</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>ריצוף</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>הגובה</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>הערות</th>
                              </tr>
                            </thead>
                            <tbody>
                              {vd.rooms!.map((r, i) => (
                                <tr key={i} style={{ borderBottom: "1px solid #F1F5F9" }}>
                                  <td style={{ padding: "5px 8px", fontWeight: 600 }}>{r.name}</td>
                                  <td style={{ padding: "5px 8px" }}>{r.area_m2 ? `${r.area_m2} מ"ר` : "—"}</td>
                                  <td style={{ padding: "5px 8px" }}>{r.dimensions ?? "—"}</td>
                                  <td style={{ padding: "5px 8px" }}>{r.flooring ?? "—"}</td>
                                  <td style={{ padding: "5px 8px" }}>{r.ceiling_height_m ? `${r.ceiling_height_m} מ'` : "—"}</td>
                                  <td style={{ padding: "5px 8px", color: "#64748B" }}>{r.notes ?? "—"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Elements (doors, windows, etc.) */}
                      {hasElements && (
                        <div style={{ marginBottom: 14 }}>
                          <div style={{ fontWeight: 600, fontSize: 13, color: "#334155", marginBottom: 6 }}>אלמנטים ({vd.elements!.length})</div>
                          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                            <thead>
                              <tr style={{ background: "#F1F5F9" }}>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>סוג</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>מזהה</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>מיקום</th>
                                <th style={{ textAlign: "right", padding: "5px 8px", borderBottom: "1px solid #E2E8F0" }}>הערות</th>
                              </tr>
                            </thead>
                            <tbody>
                              {vd.elements!.map((el, i) => (
                                <tr key={i} style={{ borderBottom: "1px solid #F1F5F9" }}>
                                  <td style={{ padding: "5px 8px", fontWeight: 600 }}>{el.type}</td>
                                  <td style={{ padding: "5px 8px" }}>{el.id ?? "—"}</td>
                                  <td style={{ padding: "5px 8px" }}>{el.location ?? "—"}</td>
                                  <td style={{ padding: "5px 8px", color: "#64748B" }}>{el.notes ?? "—"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Materials + Dimensions + Systems in 3 columns */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                        {hasMaterials && (
                          <div style={{ background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "10px 12px" }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#475569", marginBottom: 6 }}>חומרים</div>
                            {vd.materials!.map((m, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#334155", padding: "2px 0" }}>• {m}</div>
                            ))}
                          </div>
                        )}
                        {hasDims && (
                          <div style={{ background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "10px 12px" }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#475569", marginBottom: 6 }}>מידות שנמצאו</div>
                            {vd.dimensions!.slice(0, 10).map((d, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#334155", padding: "2px 0" }}>• {d}</div>
                            ))}
                            {vd.dimensions!.length > 10 && (
                              <div style={{ fontSize: 11, color: "#94A3B8" }}>ועוד {vd.dimensions!.length - 10}...</div>
                            )}
                          </div>
                        )}
                        {hasSystems && (
                          <div style={{ background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "10px 12px" }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#475569", marginBottom: 6 }}>מערכות</div>
                            {Object.entries(vd.systems!).map(([k, v]) => v ? (
                              <div key={k} style={{ fontSize: 12, color: "#334155", padding: "2px 0" }}>
                                <span style={{ color: "#94A3B8" }}>{k}: </span>{String(v)}
                              </div>
                            ) : null)}
                          </div>
                        )}
                      </div>

                      {/* Execution notes */}
                      {hasNotes && (
                        <div style={{ marginTop: 12, background: "#FFFBEB", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px" }}>
                          <div style={{ fontWeight: 600, fontSize: 12, color: "#92400E", marginBottom: 4 }}>הערות ביצוע</div>
                          {vd.execution_notes!.map((n, i) => (
                            <div key={i} style={{ fontSize: 12, color: "#78350F", padding: "2px 0" }}>• {n}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}

            {/* ── TAB: ZONE ── */}
            {step3Tab === "zone" && (
              <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
                <p className="text-sm font-semibold text-[#31333F]">צביעת אזור</p>
                <p className="text-xs text-slate-500">גרור מלבן סביב חדר/אזור — המערכת תחשב את הקירות בתוכו אוטומטית.</p>

                <div className="flex gap-2 items-center flex-wrap">
                  <label className="text-xs">קטגוריה לאזור:
                    <select value={zoneCatKey} onChange={e => setZoneCatKey(e.target.value)}
                      className="mr-2 border border-slate-300 rounded px-2 py-1 text-xs">
                      <option value="">-- בחר --</option>
                      {Object.values(planningState.categories).map(c => (
                        <option key={c.key} value={c.key}>{c.type} / {c.subtype}</option>
                      ))}
                    </select>
                  </label>
                  {zoneStart && zoneEnd && (
                    <button type="button" onClick={() => void handleAddZone()} disabled={loading || !zoneCatKey}
                      className="px-3 py-1.5 rounded-lg bg-[#FF4B4B] text-white text-xs font-semibold disabled:opacity-40">
                      {loading ? "מחשב..." : "הוסף אזור"}
                    </button>
                  )}
                  {(zoneStart || zoneEnd) && (
                    <button type="button" onClick={() => { setZoneStart(null); setZoneEnd(null); setZoneTemp(null); }}
                      className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs">נקה</button>
                  )}
                </div>

                <div className="overflow-auto max-h-[72vh] border border-slate-200 rounded-lg bg-slate-50 p-1">
                  <div className="relative border border-slate-300 rounded-lg overflow-hidden w-fit select-none">
                    <img ref={drawingImageRef} src={imageUrl} alt="plan" className="block"
                      style={{ width: displaySize.width, height: displaySize.height }}
                      onLoad={() => updateDisplaySizeFromImage(drawingImageRef.current)} draggable={false} />
                    <svg ref={zoneCanvasRef} width={displaySize.width} height={displaySize.height}
                      className="absolute inset-0 cursor-crosshair"
                      onMouseDown={handleZoneMouseDown} onMouseMove={handleZoneMouseMove}
                      onMouseUp={handleZoneMouseUp} onMouseLeave={() => setZoneDrawing(false)}>
                      {/* Existing items overlay */}
                      {planningState.items.filter(it => it.type === "zone" || it.type === "rect").map(item => {
                        const obj = item.raw_object;
                        const cat = planningState.categories[item.category];
                        const color = getCategoryColor(cat?.type, cat?.subtype);
                        return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale}
                          width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale}
                          fill={hexToRgba(color, 0.2)} stroke={color} strokeWidth={2} />;
                      })}
                      {/* Live zone preview */}
                      {zoneStart && (zoneEnd ?? zoneTemp) && (() => {
                        const end = zoneEnd ?? zoneTemp!;
                        return <rect x={Math.min(zoneStart.x, end.x)} y={Math.min(zoneStart.y, end.y)}
                          width={Math.abs(end.x - zoneStart.x)} height={Math.abs(end.y - zoneStart.y)}
                          fill={hexToRgba("#1B3A6B", 0.15)} stroke="#1B3A6B" strokeWidth={2} strokeDasharray="8 4" />;
                      })()}
                    </svg>
                  </div>
                </div>
              </div>
            )}

            {/* ── TAB: MANUAL ── */}
            {step3Tab === "manual" && (
              <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
                <p className="text-sm font-semibold text-[#31333F]">ציור ידני</p>
                <div className="flex flex-wrap gap-2 mb-1 items-center">
                  <select value={drawMode} onChange={(e) => setDrawMode(e.target.value as DrawMode)} className="bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs">
                    <option value="line">קו</option>
                    <option value="rect">מלבן</option>
                    <option value="path">חופשי</option>
                  </select>
                  <div className="inline-flex items-center gap-1 bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs">
                    <button type="button" onClick={() => setZoomPercent(z => Math.max(70, z - 10))} className="px-2 py-0.5 border border-slate-300 rounded">-</button>
                    <span>{zoomPercent}%</span>
                    <button type="button" onClick={() => setZoomPercent(z => Math.min(220, z + 10))} className="px-2 py-0.5 border border-slate-300 rounded">+</button>
                    <button type="button" onClick={() => setZoomPercent(100)} className="px-2 py-0.5 border border-slate-300 rounded text-[10px]">איפוס</button>
                  </div>
                  {pendingShapes.length > 0 && (
                    <span style={{ background: hexToRgba(PENDING_COLOR, 0.15), border: `1px solid ${hexToRgba(PENDING_COLOR, 0.5)}`, color: "#92400e", borderRadius: 20, padding: "3px 10px", fontSize: 11, fontWeight: 600 }}>
                      ⏳ {pendingShapes.length} ממתין לשיוך
                    </span>
                  )}
                </div>
                <div className="overflow-auto max-h-[72vh] border border-slate-200 rounded-lg bg-slate-50 p-1">
                  <div className="relative border border-slate-300 rounded-lg overflow-hidden w-fit select-none bg-slate-50">
                    <img ref={drawingImageRef} src={imageUrl} alt="plan" className="block"
                      style={{ width: displaySize.width, height: displaySize.height }}
                      onLoad={() => updateDisplaySizeFromImage(drawingImageRef.current)} draggable={false} />
                    <svg ref={drawingSurfaceRef} width={displaySize.width} height={displaySize.height}
                      className="absolute inset-0 cursor-crosshair"
                      onMouseDown={handleCanvasMouseDown} onMouseMove={handleCanvasMouseMove}
                      onMouseUp={handleCanvasMouseUp} onMouseLeave={() => setDrawing(false)}>
                      {planningState.items.map((item) => {
                        const obj = item.raw_object;
                        const cat = planningState.categories[item.category];
                        const color = getCategoryColor(cat?.type, cat?.subtype);
                        if (item.type === "line") return <line key={item.uid} x1={Number(obj.x1) * displayScale} y1={Number(obj.y1) * displayScale} x2={Number(obj.x2) * displayScale} y2={Number(obj.y2) * displayScale} stroke={color} strokeWidth={2} strokeLinecap="round" />;
                        if (item.type === "rect" || item.type === "zone") return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale} width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale} fill={hexToRgba(color, 0.15)} stroke={color} strokeWidth={2} />;
                        const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map(([px, py]) => `${px * displayScale},${py * displayScale}`).join(" ") : "";
                        return <polyline key={item.uid} points={pts} fill="none" stroke={color} strokeWidth={2} />;
                      })}
                      {pendingShapes.map(renderPendingOnCanvas)}
                      {drawing && startPoint && tempPoint && drawMode === "line" && <line x1={startPoint.x} y1={startPoint.y} x2={tempPoint.x} y2={tempPoint.y} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                      {drawing && startPoint && tempPoint && drawMode === "rect" && <rect x={Math.min(startPoint.x, tempPoint.x)} y={Math.min(startPoint.y, tempPoint.y)} width={Math.abs(tempPoint.x - startPoint.x)} height={Math.abs(tempPoint.y - startPoint.y)} fill={hexToRgba(PENDING_COLOR, 0.15)} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                      {drawing && drawMode === "path" && pathPoints.length > 1 && <polyline points={pathPoints.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                    </svg>
                  </div>
                </div>
              </div>
            )}

            {/* ── TAB: TEXT ── */}
            {step3Tab === "text" && (
              <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
                <div className="flex items-start justify-between gap-3 flex-wrap">
                  <div>
                    <p className="text-sm font-semibold text-[#31333F]">פריטים בטקסט חופשי</p>
                    <p className="text-xs text-slate-500">פריטים שלא ניתן לצייר (ספקלינג, פינות, תוספות אחוזיות וכו׳).</p>
                  </div>
                  <button
                    type="button"
                    disabled={loading}
                    onClick={async () => {
                      if (!selectedPlanId) return;
                      setLoading(true);
                      try {
                        const state = await importVisionItems(selectedPlanId);
                        setPlanningState(state);
                        setError("");
                      } catch (e) {
                        const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
                        setError(`שגיאה בייבוא: ${detail}`);
                      } finally { setLoading(false); }
                    }}
                    style={{
                      display: "flex", alignItems: "center", gap: 6,
                      padding: "7px 14px", borderRadius: 9,
                      background: "#1B3A6B", color: "#fff",
                      border: "none", fontWeight: 700, fontSize: 13,
                      cursor: "pointer", whiteSpace: "nowrap",
                      opacity: loading ? 0.5 : 1,
                      boxShadow: "0 2px 8px rgba(27,58,107,0.25)",
                    }}
                  >
                    📥 ייבא מתוכנית
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse min-w-[600px]">
                    <thead>
                      <tr className="bg-slate-50 text-slate-500">
                        <th className="p-2 text-right font-medium border-b border-slate-200 w-[30%]">תיאור *</th>
                        <th className="p-2 text-right font-medium border-b border-slate-200 w-[20%]">קטגוריה</th>
                        <th className="p-2 text-right font-medium border-b border-slate-200 w-[10%]">כמות *</th>
                        <th className="p-2 text-right font-medium border-b border-slate-200 w-[12%]">יחידה</th>
                        <th className="p-2 text-right font-medium border-b border-slate-200 w-[20%]">הערה</th>
                        <th className="p-2 border-b border-slate-200 w-[8%]"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {textRows.map((row, idx) => (
                        <tr key={idx} className="border-b border-slate-100">
                          <td className="p-1">
                            <input value={row.description} onChange={e => handleTextRowChange(idx, "description", e.target.value)}
                              placeholder="תיאור הפריט..." className="w-full border border-slate-300 rounded px-2 py-1 text-xs" />
                          </td>
                          <td className="p-1">
                            <select value={row.category_key} onChange={e => handleTextRowChange(idx, "category_key", e.target.value)}
                              className="w-full border border-slate-300 rounded px-1 py-1 text-xs">
                              <option value="__manual__">ידני</option>
                              {Object.values(planningState.categories).map(c => (
                                <option key={c.key} value={c.key}>{c.subtype}</option>
                              ))}
                            </select>
                          </td>
                          <td className="p-1">
                            <input type="number" min={0} step={0.01} value={row.quantity}
                              onChange={e => handleTextRowChange(idx, "quantity", Number(e.target.value))}
                              className="w-full border border-slate-300 rounded px-2 py-1 text-xs" />
                          </td>
                          <td className="p-1">
                            <select value={row.unit} onChange={e => handleTextRowChange(idx, "unit", e.target.value)}
                              className="w-full border border-slate-300 rounded px-1 py-1 text-xs">
                              {["מ׳", 'מ"ר', "יח׳", "ק״ג", "ליטר", "מ״ק"].map(u => <option key={u}>{u}</option>)}
                            </select>
                          </td>
                          <td className="p-1">
                            <input value={row.note ?? ""} onChange={e => handleTextRowChange(idx, "note", e.target.value)}
                              placeholder="הערה..." className="w-full border border-slate-300 rounded px-2 py-1 text-xs" />
                          </td>
                          <td className="p-1 text-center">
                            <button type="button" onClick={() => handleRemoveTextRow(idx)}
                              className="text-red-400 hover:text-red-600 text-sm">🗑</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="flex gap-2 flex-wrap">
                  <button type="button" onClick={handleAddTextRow}
                    className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs hover:bg-slate-50">
                    + שורה חדשה
                  </button>
                  <button type="button" onClick={() => void handleSaveTextRows()} disabled={loading}
                    className="px-4 py-2 rounded-lg bg-[#FF4B4B] text-white text-sm font-semibold disabled:opacity-40">
                    {loading ? "שומר..." : "💾 שמור פריטים"}
                  </button>
                </div>

                {/* Existing text items */}
                {planningState.items.filter(it => it.type === "text").length > 0 && (
                  <div className="mt-3 border-t border-slate-100 pt-3">
                    <p className="text-xs font-semibold text-slate-500 mb-2">פריטי טקסט שנשמרו:</p>
                    <div className="space-y-1">
                      {planningState.items.filter(it => it.type === "text").map(item => (
                        <div key={item.uid} className="flex items-center justify-between text-xs bg-slate-50 border border-slate-200 rounded px-2 py-1">
                          <span className="font-medium">{String(item.raw_object.description ?? "")}</span>
                          <span className="text-slate-500">{String(item.raw_object.quantity ?? "")} {String(item.raw_object.unit ?? "")}</span>
                          <button type="button" onClick={() => void handleDeleteItem(item.uid)} className="text-red-400 hover:text-red-600">✕</button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right panel */}
          <div className="space-y-4">
            {/* Category manager */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
              <p className="text-sm font-semibold text-[#31333F]">קטגוריות</p>
              <div className="space-y-1 max-h-[160px] overflow-y-auto">
                {Object.values(planningState.categories).map((cat) => {
                  const color = getCategoryColor(cat.type, cat.subtype);
                  return (
                    <div key={cat.key} className="flex items-center gap-2 rounded-lg px-2 py-1 text-xs" style={{ background: hexToRgba(color, 0.08), border: `1px solid ${hexToRgba(color, 0.35)}` }}>
                      <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
                      {cat.type} - {cat.subtype}
                    </div>
                  );
                })}
                {Object.keys(planningState.categories).length === 0 && <p className="text-xs text-slate-400">אין קטגוריות עדיין.</p>}
              </div>
              <details className="text-xs">
                <summary className="cursor-pointer text-slate-600 hover:text-slate-900 select-none py-1">+ הוסף קטגוריה</summary>
                <div className="mt-2 space-y-2">
                  <label>סוג<select className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newType} onChange={e => setNewType(e.target.value)}><option>קירות</option><option>ריצוף</option><option>תקרה</option></select></label>
                  <label>תת-סוג<select className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newSubtype} onChange={e => setNewSubtype(e.target.value)}>{subtypeOptions.map(s => <option key={s}>{s}</option>)}</select></label>
                  <label>פרמטר<input type="number" className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newParamValue} onChange={e => setNewParamValue(Number(e.target.value))} /></label>
                  <label>הערה<input className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newParamNote} onChange={e => setNewParamNote(e.target.value)} /></label>
                  <button type="button" onClick={() => { handleAddCategory(); void handleSaveCategories(); }} className="w-full px-2 py-1.5 rounded bg-[#1B3A6B] text-white text-xs font-semibold">הוסף ושמור</button>
                </div>
              </details>
            </div>

            {/* Items list */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
              <p className="text-sm font-semibold mb-2 text-[#31333F]">כל הפריטים ({planningState.items.length})</p>
              {planningState.items.length === 0
                ? <p className="text-xs text-slate-400">הוסף פריטים דרך אחד הטאבים.</p>
                : <div className="space-y-1 max-h-[320px] overflow-y-auto">
                    {planningState.items.map((item) => {
                      const cat = planningState.categories[item.category];
                      const color = getCategoryColor(cat?.type, cat?.subtype);
                      const label = item.type === "text"
                        ? `📋 ${String(item.raw_object.description ?? "").slice(0, 22)}`
                        : item.type === "zone"
                          ? `🎨 אזור | ${(item.length_m_effective ?? item.length_m).toFixed(1)}מ׳`
                          : `✏️ ${item.type} | ${(item.length_m_effective ?? item.length_m).toFixed(2)}מ׳`;
                      return (
                        <div key={item.uid} className="flex items-center justify-between rounded-lg px-2 py-1 text-xs" style={{ background: hexToRgba(color, 0.07), border: `1px solid ${hexToRgba(color, 0.3)}` }}>
                          <span className="flex items-center gap-1.5 min-w-0">
                            <span className="inline-block w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                            <span className="truncate">{cat?.subtype ?? item.category} | {label}</span>
                          </span>
                          <button type="button" className="text-red-500 hover:text-red-700 flex-shrink-0 mr-1" onClick={() => void handleDeleteItem(item.uid)}>✕</button>
                        </div>
                      );
                    })}
                  </div>
              }
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button type="button" onClick={() => setStep(4)} disabled={planningState.items.length === 0}
                style={{ padding: "11px 20px", borderRadius: 10, background: planningState.items.length === 0 ? "#CBD5E1" : "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: planningState.items.length === 0 ? "not-allowed" : "pointer", boxShadow: planningState.items.length === 0 ? "none" : "0 3px 10px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
                המשך לשלב 4 ←
              </button>
              <button type="button" onClick={() => setStep(2)}
                style={{ padding: "9px 16px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                ← חזור לשלב 2
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── STEP 4: BOQ + Save ── */}
      {step === 4 && planningState && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
            <p className="text-sm font-semibold mb-3 text-[#31333F]">שלב 4: כתב כמויות (BOQ) ושמירה</p>
            {finalizeNotice && (
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-green-800 bg-green-50 rounded-lg px-4 py-3" style={{ border: "1px solid #86EFAC", borderRight: "5px solid #22C55E" }}>
                <span style={{ fontSize: 20 }}>✅</span>
                <span className="flex-1">{finalizeNotice}</span>
                <button type="button" onClick={() => setFinalizeNotice("")} style={{ color: "#16A34A", background: "none", border: "none", cursor: "pointer", fontSize: 15 }}>✕</button>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סה״כ פריטים</p>
                <p className="font-semibold text-[#31333F]">{planningState.items.length}</p>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סה״כ אורך</p>
                <p className="font-semibold text-[#31333F]">{planningState.totals.total_length_m.toFixed(2)} מ&apos;</p>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סה״כ שטח</p>
                <p className="font-semibold text-[#31333F]">{planningState.totals.total_area_m2.toFixed(2)} מ&quot;ר</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-2 text-xs">
                <p className="text-amber-700">דלתות שסומנו</p>
                <p className="font-semibold text-amber-900">{openingsSummary.doorCount}</p>
              </div>
              <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-2 text-xs">
                <p className="text-cyan-700">חלונות שסומנו</p>
                <p className="font-semibold text-cyan-900">{openingsSummary.windowCount}</p>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">אורך שהופחת בגלל פתחים</p>
                <p className="font-semibold text-[#31333F]">{openingsSummary.deductedLengthM.toFixed(2)} מ&apos;</p>
              </div>
            </div>

            {/* BOQ table */}
            <div className="space-y-1 mb-4">
              {Object.keys(planningState.boq).length === 0 && <p className="text-xs text-slate-500">אין נתוני BOQ להצגה.</p>}
              {Object.entries(planningState.boq).map(([key, value]) => {
                const row = value as { type?: string; subtype?: string; count?: number; total_length_m?: number; total_area_m2?: number };
                const color = getCategoryColor(row.type, row.subtype);
                return (
                  <div key={key} className="rounded-lg px-2 py-2 text-xs grid grid-cols-5 gap-2" style={{ background: hexToRgba(color, 0.07), border: `1px solid ${hexToRgba(color, 0.25)}` }}>
                    <span className="font-semibold" style={{ color }}>{row.type ?? "-"}</span>
                    <span>{row.subtype ?? "-"}</span>
                    <span>{row.count ?? 0} פריטים</span>
                    <span>{(row.total_length_m ?? 0).toFixed(2)} מ&apos;</span>
                    <span>{(row.total_area_m2 ?? 0).toFixed(2)} מ&quot;ר</span>
                  </div>
                );
              })}
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <button type="button" onClick={() => void handleFinalize()} disabled={loading}
                style={{ padding: "11px 24px", borderRadius: 10, background: loading ? "#94a3b8" : "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer", boxShadow: loading ? "none" : "0 3px 12px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
                {loading ? "שומר..." : "💾 שמירה סופית"}
              </button>
              <button type="button" onClick={() => setStep(5)}
                style={{ padding: "11px 20px", borderRadius: 10, background: "#1B3A6B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 8px rgba(27,58,107,0.22)" }}>
                גזרות עבודה ▶
              </button>
              <button type="button" onClick={() => setStep(3)}
                style={{ padding: "10px 16px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                ← חזור לשלב 3
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── STEP 5: Work Sections (גזרות עבודה) ── */}
      {step === 5 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr,340px] gap-4">
          {/* Left: canvas + section overlays */}
          <div className="space-y-3">
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
              <p className="text-sm font-semibold mb-2 text-[#31333F]">שלב 5: גזרות עבודה</p>
              <p className="text-xs text-slate-500 mb-3">
                סמן אזור על הגרמושה לכל גזרה ומלא שם קבלן ועובד אחראי. גזרות אינן חובה — ניתן לדלג.
              </p>
              <div className="relative border border-slate-200 rounded-lg overflow-hidden w-fit">
                <img
                  ref={secImageRef}
                  src={imageUrl}
                  alt="plan"
                  className="block"
                  style={{ maxWidth: "100%", width: displaySize.width, height: "auto" }}
                  draggable={false}
                />
                <svg
                  ref={secCanvasRef}
                  width={secImageRef.current?.clientWidth || displaySize.width}
                  height={secImageRef.current?.clientHeight || displaySize.height}
                  className="absolute inset-0 cursor-crosshair"
                  onMouseDown={handleSecMouseDown}
                  onMouseMove={handleSecMouseMove}
                  onMouseUp={handleSecMouseUp}
                  onMouseLeave={() => setSecDrawing(false)}
                >
                  {/* Existing sections */}
                  {planningState.sections.map((sec) => {
                    const imgW = secImageRef.current?.naturalWidth || planningState.image_width || 1;
                    const dispW = secImageRef.current?.clientWidth || displaySize.width || 1;
                    const sf = dispW / imgW;
                    const rx = sec.x * sf;
                    const ry = sec.y * sf;
                    const rw = sec.width * sf;
                    const rh = sec.height * sf;
                    if (rw < 2 || rh < 2) return null;
                    return (
                      <g key={sec.uid}>
                        <rect x={rx} y={ry} width={rw} height={rh} fill={`${sec.color}33`} stroke={sec.color} strokeWidth={2} rx={3} />
                        <rect x={rx} y={ry} width={Math.min(rw, 180)} height={20} fill={sec.color} rx={2} />
                        <text x={rx + 4} y={ry + 14} fill="white" fontSize={11} fontWeight="bold">
                          {sec.name} | {sec.contractor}
                        </text>
                      </g>
                    );
                  })}

                  {/* Current drawing preview */}
                  {secDrawing && secStart && secTemp && (
                    <rect
                      x={Math.min(secStart.x, secTemp.x)} y={Math.min(secStart.y, secTemp.y)}
                      width={Math.abs(secTemp.x - secStart.x)} height={Math.abs(secTemp.y - secStart.y)}
                      fill={`${secColor}22`} stroke={secColor} strokeWidth={2} strokeDasharray="6 3"
                    />
                  )}
                  {/* Finished but unsaved rect */}
                  {!secDrawing && secStart && secEnd && (
                    <rect
                      x={Math.min(secStart.x, secEnd.x)} y={Math.min(secStart.y, secEnd.y)}
                      width={Math.abs(secEnd.x - secStart.x)} height={Math.abs(secEnd.y - secStart.y)}
                      fill={`${secColor}22`} stroke={secColor} strokeWidth={2} strokeDasharray="6 3"
                    />
                  )}
                </svg>
              </div>
              <p className="text-[10px] text-slate-400 mt-1">גרור על הגרמושה לסמן תחום גזרה (אופציונלי)</p>
            </div>
          </div>

          {/* Right: form + list */}
          <div className="space-y-4">
            {/* Add section form */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
              <p className="text-sm font-semibold text-[#31333F]">הוסף גזרה</p>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">שם הגזרה (אופציונלי)</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="גזרה צפונית / קומה א׳..."
                    value={secName}
                    onChange={e => setSecName(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">שם קבלן מבצע</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="שם הקבלן..."
                    value={secContractor}
                    onChange={e => setSecContractor(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">שם עובד אחראי</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="שם העובד..."
                    value={secWorker}
                    onChange={e => setSecWorker(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">צבע</label>
                  <div className="flex gap-1 flex-wrap">
                    {["#6366f1","#0ea5e9","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6"].map(c => (
                      <button
                        key={c}
                        type="button"
                        onClick={() => setSecColor(c)}
                        className="w-6 h-6 rounded-full border-2 transition-transform"
                        style={{ background: c, borderColor: secColor === c ? "#1e293b" : "transparent", transform: secColor === c ? "scale(1.25)" : "scale(1)" }}
                      />
                    ))}
                  </div>
                </div>
                {secStart && secEnd && (
                  <div className="text-xs text-slate-500 bg-slate-50 rounded px-2 py-1">
                    אזור סומן ✓ | {Math.round(Math.abs(secEnd.x - secStart.x))}×{Math.round(Math.abs(secEnd.y - secStart.y))} px
                    <button type="button" className="text-red-400 hover:text-red-600 mr-2" onClick={() => { setSecStart(null); setSecEnd(null); }}>✕ נקה</button>
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={() => void handleAddSection()}
                disabled={loading || (!secContractor.trim() && !secWorker.trim())}
                className="w-full px-3 py-2 rounded-lg bg-[#1B3A6B] text-white text-xs font-semibold disabled:opacity-40 hover:bg-[#162d56]"
              >
                {loading ? "שומר..." : "+ הוסף גזרה"}
              </button>
            </div>

            {/* Existing sections list */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
              <p className="text-sm font-semibold mb-2 text-[#31333F]">גזרות קיימות ({planningState.sections.length})</p>
              {planningState.sections.length === 0 ? (
                <p className="text-xs text-slate-400">אין גזרות עדיין. ניתן לדלג על שלב זה.</p>
              ) : (
                <div className="space-y-2">
                  {planningState.sections.map((sec) => (
                    <div key={sec.uid} className="rounded-lg px-3 py-2 text-xs flex items-start justify-between gap-2"
                      style={{ background: `${sec.color}12`, border: `1px solid ${sec.color}55` }}>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: sec.color, display: "inline-block" }} />
                          <span className="font-semibold text-slate-700">{sec.name || "ללא שם"}</span>
                        </div>
                        <div className="text-slate-500">🏗 קבלן: <span className="text-slate-700 font-medium">{sec.contractor || "—"}</span></div>
                        <div className="text-slate-500">👷 עובד: <span className="text-slate-700 font-medium">{sec.worker || "—"}</span></div>
                        {sec.width > 0 && (
                          <div className="text-slate-400 mt-0.5">
                            אזור: {Math.round(sec.width)}×{Math.round(sec.height)} px
                          </div>
                        )}
                      </div>
                      <button type="button" onClick={() => void handleDeleteSection(sec.uid)}
                        className="text-red-400 hover:text-red-600 flex-shrink-0 text-sm">✕</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button type="button" onClick={() => void handleFinalize()} disabled={loading}
                style={{ padding: "11px 20px", borderRadius: 10, background: loading ? "#94a3b8" : "#FF4B4B", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer", boxShadow: loading ? "none" : "0 3px 12px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
                {loading ? "שומר..." : "💾 שמירה סופית"}
              </button>
              <button type="button" onClick={() => setStep(4)}
                style={{ padding: "9px 16px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                ← חזור לשלב 4
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Suppress unused variable warning */}
      {activeColor && null}
    </div>
  );
};
