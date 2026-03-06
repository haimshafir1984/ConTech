import React from "react";
import axios from "axios";
import { ErrorAlert, PlanningCanvasErrorBoundary } from "../components/UiHelpers";
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
  "קירות":          ["בטון", "בלוקים", "גבס", "מחיצה קלה"],
  "ריצוף":          ["קרמיקה", "גרניט פורצלן", "פרקט", "בטון מוחלק"],
  "תקרה":           ["גבס", "אקוסטית", "חשופה", "צבועה"],
  "דלתות וחלונות": ["דלת פנים", "דלת כניסה", "חלון אלומיניום", "חלון עץ", "ויטרינה"],
  "אינסטלציה":      ["מים קרים", "מים חמים", "ביוב", "גז"],
  "חשמל":           ["תאורה", "שקעים", "לוח חשמל", "גנרטור"],
  "טיח וצבע":       ["טיח פנים", "טיח חוץ", "צבע פנים", "צבע חוץ"],
  "עמודים":         ["עמוד בטון", "עמוד מתכת", "קורה"],
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
  "תקרה:צבועה": "#10b981",
  "דלתות וחלונות:דלת פנים": "#84cc16",
  "דלתות וחלונות:דלת כניסה": "#65a30d",
  "דלתות וחלונות:חלון אלומיניום": "#4ade80",
  "דלתות וחלונות:חלון עץ": "#22c55e",
  "דלתות וחלונות:ויטרינה": "#16a34a",
  "אינסטלציה:מים קרים": "#38bdf8",
  "אינסטלציה:מים חמים": "#fb923c",
  "אינסטלציה:ביוב": "#a78bfa",
  "אינסטלציה:גז": "#fde047",
  "חשמל:תאורה": "#facc15",
  "חשמל:שקעים": "#eab308",
  "חשמל:לוח חשמל": "#ca8a04",
  "חשמל:גנרטור": "#a16207",
  "טיח וצבע:טיח פנים": "#f9a8d4",
  "טיח וצבע:טיח חוץ": "#f472b6",
  "טיח וצבע:צבע פנים": "#e879f9",
  "טיח וצבע:צבע חוץ": "#d946ef",
  "עמודים:עמוד בטון": "#94a3b8",
  "עמודים:עמוד מתכת": "#64748b",
  "עמודים:קורה": "#475569",
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

// Maps suggested_subtype (from backend) → Hebrew label + icon
function getFixGroupLabel(subtype: string): { label: string; icon: string } {
  const t = subtype.toLowerCase();
  // Backend exact values (Hebrew)
  if (t.includes("כיור") || t.includes("אסלה")) return { label: "כיורים ואסלות", icon: "🚰" };
  if (t.includes("אמבטיה") || t.includes("מקלחת")) return { label: "אמבטיות ומקלחות", icon: "🚿" };
  if (t.includes("ריהוט") || t.includes("מכשיר")) return { label: "ריהוט ומכשירים", icon: "🛋️" };
  // English fallbacks
  if (t.includes("door") || t.includes("דלת")) return { label: "דלתות", icon: "🚪" };
  if (t.includes("window") || t.includes("חלון")) return { label: "חלונות", icon: "🪟" };
  if (t.includes("column") || t.includes("pillar") || t.includes("עמוד")) return { label: "עמודים", icon: "🏛️" };
  if (t.includes("stair") || t.includes("מדרגה") || t.includes("מדרגות")) return { label: "מדרגות", icon: "🪜" };
  if (t.includes("elevator") || t.includes("lift") || t.includes("מעלית")) return { label: "מעליות", icon: "🛗" };
  if (t.includes("beam") || t.includes("קורה")) return { label: "קורות", icon: "🔩" };
  return { label: subtype || "אחר", icon: "📌" };
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
        <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4, color: "var(--navy)" }}>
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
              background: tab === "existing" ? "var(--navy)" : "#F1F5F9",
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
              background: tab === "new" ? "var(--navy)" : "#F1F5F9",
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
                {Object.keys(CATEGORY_SUBTYPES).map(t => <option key={t}>{t}</option>)}
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
              style={{ padding: "10px 0", borderRadius: 10, background: "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer" }}
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
    // `el` is captured in the closure, so cleanup removes from the same element
    // even if containerRef.current later changes. This is the correct pattern.
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

  // ── Touch support for ZoomModal ──
  const lastPinchDist = React.useRef<number | null>(null);
  const lastTouchPan = React.useRef<{ x: number; y: number } | null>(null);

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    if (e.touches.length === 2) {
      lastPinchDist.current = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      lastTouchPan.current = null;
    } else if (e.touches.length === 1) {
      const t = e.touches[0];
      lastTouchPan.current = { x: t.clientX - pan.x, y: t.clientY - pan.y };
      lastPinchDist.current = null;
      // Also start drawing
      const p = toNatural(t.clientX, t.clientY);
      setDrawing(true); setStartPt(p); setTempPt(p);
      if (modalDrawMode === "path") setPathPts([p]);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    e.preventDefault();
    if (e.touches.length === 2 && lastPinchDist.current !== null) {
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const ratio = dist / lastPinchDist.current;
      const rect = containerRef.current?.getBoundingClientRect();
      const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - (rect?.left ?? 0);
      const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - (rect?.top ?? 0);
      setZoom((prev) => {
        const next = Math.max(0.25, Math.min(10, prev * ratio));
        setPan((p) => ({ x: cx - (cx - p.x) * (next / prev), y: cy - (cy - p.y) * (next / prev) }));
        return next;
      });
      lastPinchDist.current = dist;
    } else if (e.touches.length === 1) {
      const t = e.touches[0];
      if (lastTouchPan.current && !drawing) {
        setPan({ x: t.clientX - lastTouchPan.current.x, y: t.clientY - lastTouchPan.current.y });
      } else if (drawing) {
        const p = toNatural(t.clientX, t.clientY);
        setTempPt(p);
        if (modalDrawMode === "path") setPathPts((prev) => [...prev, p]);
      }
    }
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    e.preventDefault();
    lastPinchDist.current = null;
    lastTouchPan.current = null;
    if (drawing && startPt && tempPt) {
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
    }
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
      <div style={{ background: "var(--navy)", padding: "8px 16px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0, flexWrap: "wrap" }}>
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
        style={{ flex: 1, overflow: "hidden", position: "relative", cursor: drawing ? "crosshair" : "grab", touchAction: "none" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
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
      <div style={{ background: "var(--navy)", padding: "6px 16px", maxHeight: 120, overflowY: "auto", flexShrink: 0 }}>
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
  const [visionCatSuggestions, setVisionCatSuggestions] = React.useState<{ type: string; subtype: string; paramValue: number }[]>([]);
  const [visionActiveCard, setVisionActiveCard] = React.useState<string | null>(null);
  const [autoLoading, setAutoLoading] = React.useState(false);
  const [autoSelected, setAutoSelected] = React.useState<Set<string>>(new Set());
  const [autoConfirmedKeys, setAutoConfirmedKeys] = React.useState<Record<string, string>>({}); // segId→catKey
  const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(new Set(["walls"]));
  const [bulkOpen, setBulkOpen] = React.useState(false);
  const [bulkCatKeys, setBulkCatKeys] = React.useState<Record<string, string>>({}); // "type/subtype"→catKey
  const [lastAddedUid, setLastAddedUid] = React.useState<string | null>(null);
  const [focusedUid, setFocusedUid] = React.useState<string | null>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement | null>(null);

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
    // שחזר תוצאות ניתוח אוטומטי שנשמרו בדאטהבייס
    if (state.auto_segments && state.auto_segments.length > 0 && autoSegments === null) {
      setAutoSegments(state.auto_segments);
      setAutoSelected(new Set(
        state.auto_segments
          .filter(s => s.suggested_subtype !== "פרט קטן")
          .map(s => s.segment_id)
      ));
    }
  }, [autoSegments]);

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

  // ── Touch support: wraps a mouse handler to accept touch events ──
  const makeTouchHandler = <T extends SVGSVGElement>(
    handler: React.MouseEventHandler<T>
  ): React.TouchEventHandler<T> =>
    (e: React.TouchEvent<T>) => {
      e.preventDefault();
      const touch = e.touches[0] ?? e.changedTouches[0];
      if (!touch) return;
      handler({ clientX: touch.clientX, clientY: touch.clientY, currentTarget: e.currentTarget, preventDefault: () => {} } as unknown as React.MouseEvent<T>);
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
    let lastState = planningState!;
    const failures: string[] = [];
    try {
      for (const shape of pendingShapes) {
        try {
          lastState = await addPlanningItem(selectedPlanId, {
            category_key: categoryKey,
            object_type: shape.object_type,
            raw_object: shape.raw_object,
            display_scale: shape.display_scale,
          });
          // Handle prompts for last saved item
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
        } catch (itemErr) {
          console.error("[handleAssignCategory] item failed:", itemErr);
          const detail = axios.isAxiosError(itemErr)
            ? (itemErr.response?.data?.detail as string | undefined) || itemErr.message
            : String(itemErr);
          failures.push(detail);
        }
      }
      // Always save whatever succeeded
      setPlanningState(lastState);
      setLastAddedUid(lastState.items.at(-1)?.uid ?? null);
      const saved = pendingShapes.length - failures.length;
      if (failures.length === 0) {
        setPendingShapes([]);
        setError("");
      } else if (saved > 0) {
        // Partial success: clear only the saved shapes
        setPendingShapes(prev => prev.slice(saved));
        setError(`נשמרו ${saved} מתוך ${pendingShapes.length} פריטים. ${failures.length} נכשלו — נסה שוב.`);
      } else {
        setError(`שגיאה בשיוך פריטים: ${failures[0]}`);
      }
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
    setTempPoint(drawMode === "line" ? snapLinePoint(p, startPoint) : p);
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
    const finalPoint = drawMode === "line" ? snapLinePoint(tempPoint, startPoint) : tempPoint;
    if (drawMode !== "path") {
      const dx = finalPoint.x - startPoint.x, dy = finalPoint.y - startPoint.y;
      if (Math.sqrt(dx * dx + dy * dy) < 6) return;
    }
    let raw_object: Record<string, unknown>;
    if (drawMode === "line") {
      raw_object = { x1: startPoint.x, y1: startPoint.y, x2: finalPoint.x, y2: finalPoint.y };
    } else if (drawMode === "rect") {
      raw_object = { x: Math.min(startPoint.x, finalPoint.x), y: Math.min(startPoint.y, finalPoint.y), width: Math.abs(finalPoint.x - startPoint.x), height: Math.abs(finalPoint.y - startPoint.y) };
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

  // ── Vision material → category matcher ──
  const matchMaterialToCategory = (material: string): { type: string; subtype: string } | null => {
    const m = material.toLowerCase();
    if (m.includes("בטון מוחלק")) return { type: "ריצוף", subtype: "בטון מוחלק" };
    if (m.includes("בטון")) return { type: "קירות", subtype: "בטון" };
    if (m.includes("בלוק")) return { type: "קירות", subtype: "בלוקים" };
    if (m.includes("מחיצה קלה")) return { type: "קירות", subtype: "מחיצה קלה" };
    if (m.includes("גרניט") || m.includes("פורצלן")) return { type: "ריצוף", subtype: "גרניט פורצלן" };
    if (m.includes("קרמיקה")) return { type: "ריצוף", subtype: "קרמיקה" };
    if (m.includes("פרקט")) return { type: "ריצוף", subtype: "פרקט" };
    if (m.includes("תקרה") && m.includes("גבס")) return { type: "תקרה", subtype: "גבס" };
    if (m.includes("גבס") && (m.includes("מחיצה") || m.includes("קיר"))) return { type: "קירות", subtype: "גבס" };
    if (m.includes("גבס")) return { type: "קירות", subtype: "גבס" };
    if (m.includes("טיח") && m.includes("חוץ")) return { type: "טיח וצבע", subtype: "טיח חוץ" };
    if (m.includes("טיח") && m.includes("פנים")) return { type: "טיח וצבע", subtype: "טיח פנים" };
    if (m.includes("צבע") && m.includes("חוץ")) return { type: "טיח וצבע", subtype: "צבע חוץ" };
    if (m.includes("צבע") && m.includes("פנים")) return { type: "טיח וצבע", subtype: "צבע פנים" };
    if (m.includes("דלת") && m.includes("כניסה")) return { type: "דלתות וחלונות", subtype: "דלת כניסה" };
    if (m.includes("דלת")) return { type: "דלתות וחלונות", subtype: "דלת פנים" };
    if (m.includes("חלון") && m.includes("עץ")) return { type: "דלתות וחלונות", subtype: "חלון עץ" };
    if (m.includes("חלון")) return { type: "דלתות וחלונות", subtype: "חלון אלומיניום" };
    if (m.includes("ויטרינה")) return { type: "דלתות וחלונות", subtype: "ויטרינה" };
    if (m.includes("עמוד") && m.includes("מתכת")) return { type: "עמודים", subtype: "עמוד מתכת" };
    if (m.includes("עמוד")) return { type: "עמודים", subtype: "עמוד בטון" };
    if (m.includes("קורה")) return { type: "עמודים", subtype: "קורה" };
    if (m.includes("אקוסטי")) return { type: "תקרה", subtype: "אקוסטית" };
    return null;
  };

  // ── Auto-create categories from Vision materials ──
  const handleAutoCreateCategoriesFromVision = async () => {
    if (!selectedPlanId || visionCatSuggestions.length === 0) return;
    setLoading(true);
    try {
      const newCats: Record<string, PlanningCategory> = { ...categoriesDraft };
      for (const sug of visionCatSuggestions) {
        if (!Object.values(newCats).find(c => c.type === sug.type && c.subtype === sug.subtype)) {
          const key = `${sug.type}_${sug.subtype}_${Object.keys(newCats).length + 1}`;
          newCats[key] = { key, type: sug.type, subtype: sug.subtype, params: { height_or_thickness: sug.paramValue, note: "" } };
        }
      }
      const state = await upsertPlanningCategories(selectedPlanId, newCats);
      setPlanningState(state);
      setCategoriesDraft(newCats);
      // Auto-assign segments to newly created categories
      if (autoSegments) {
        const newConfirmedKeys: Record<string, string> = { ...autoConfirmedKeys };
        for (const seg of autoSegments) {
          if (!newConfirmedKeys[seg.segment_id]) {
            const match = Object.values(newCats).find(
              c => c.type === seg.suggested_type && c.subtype === seg.suggested_subtype
            );
            if (match) newConfirmedKeys[seg.segment_id] = match.key;
          }
        }
        setAutoConfirmedKeys(newConfirmedKeys);
      }
      setVisionCatSuggestions([]);
      setError("");
    } catch { setError("שגיאה ביצירת קטגוריות אוטומטית."); }
    finally { setLoading(false); }
  };

  // ── Auto-analyze handlers ──
  const handleAutoAnalyze = async () => {
    if (!selectedPlanId) return;
    setAutoLoading(true);
    try {
      const result = await autoAnalyzePlan(selectedPlanId);
      setAutoSegments(result.segments);
      setAutoVisionData(result.vision_data ?? null);
      setVisionActiveCard(null);
      // Pre-select all except unidentified small fixtures ("פרט קטן")
      setAutoSelected(new Set(
        result.segments.filter(s => s.suggested_subtype !== "פרט קטן").map(s => s.segment_id)
      ));
      // Build category suggestions — primary: from detected segments; secondary: Vision materials
      {
        const seen = new Set<string>();
        const suggestions: { type: string; subtype: string; paramValue: number }[] = [];

        // Primary: unique (type, subtype) pairs from wall segments — guaranteed valid
        for (const seg of result.segments) {
          if (seg.element_class === "fixture") continue;
          const k = `${seg.suggested_type}/${seg.suggested_subtype}`;
          if (!seen.has(k) && CATEGORY_SUBTYPES[seg.suggested_type]?.includes(seg.suggested_subtype)) {
            seen.add(k);
            suggestions.push({ type: seg.suggested_type, subtype: seg.suggested_subtype, paramValue: 2.6 });
          }
        }

        // Secondary: Vision materials / legend OCR
        if (result.vision_data) {
          const vd = result.vision_data;
          const allSources = [
            ...(vd.materials ?? []),
            ...(vd.elements?.map((e: { type?: string }) => e.type ?? "") ?? []),
          ];
          for (const mat of allSources) {
            const match = matchMaterialToCategory(mat);
            if (match) {
              const k = `${match.type}/${match.subtype}`;
              if (!seen.has(k)) {
                seen.add(k);
                suggestions.push({ ...match, paramValue: match.type === "ריצוף" ? 0.012 : 2.6 });
              }
            }
          }
        }

        // Filter out categories that already exist
        const existingKeys = new Set(
          Object.values(planningState?.categories ?? {}).map(c => `${c.type}/${c.subtype}`)
        );
        setVisionCatSuggestions(suggestions.filter(s => !existingKeys.has(`${s.type}/${s.subtype}`)));
      }
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
      setLastAddedUid(lastState.items.at(-1)?.uid ?? null);
      setAutoSegments(null);
      setAutoVisionData(null);
      setVisionActiveCard(null);
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

  // ── Focus / zoom to an item on canvas ──
  const focusOnItem = React.useCallback((uid: string) => {
    setFocusedUid(uid);
    if (!planningState || !canvasContainerRef.current) return;
    const item = planningState.items.find(i => i.uid === uid);
    if (!item) return;
    const obj = item.raw_object;
    let cx = 0, cy = 0;
    if (item.type === "line") {
      cx = ((Number(obj.x1) + Number(obj.x2)) / 2) * displayScale;
      cy = ((Number(obj.y1) + Number(obj.y2)) / 2) * displayScale;
    } else if (item.type === "rect" || item.type === "zone") {
      cx = (Number(obj.x) + Number(obj.width) / 2) * displayScale;
      cy = (Number(obj.y) + Number(obj.height) / 2) * displayScale;
    }
    const container = canvasContainerRef.current;
    const scrollLeft = cx - container.clientWidth / 2;
    const scrollTop = cy - container.clientHeight / 2;
    container.scrollTo({ left: Math.max(0, scrollLeft), top: Math.max(0, scrollTop), behavior: "smooth" });
    setZoomPercent(z => Math.max(z, 160));
    // Clear focus highlight after 2s
    setTimeout(() => setFocusedUid(f => f === uid ? null : f), 2000);
  }, [planningState, displayScale]);

  // ── Real-time BOQ preview (per category totals) ──
  const liveBoq = React.useMemo(() => {
    if (!planningState) return [];
    const map = new Map<string, { cat: PlanningCategory; lengthM: number; areaM2: number; count: number }>();
    for (const item of planningState.items) {
      const cat = planningState.categories[item.category];
      if (!cat) continue;
      const key = item.category;
      const entry = map.get(key) ?? { cat, lengthM: 0, areaM2: 0, count: 0 };
      entry.lengthM += Number(item.length_m_effective ?? item.length_m ?? 0);
      entry.areaM2 += Number(item.area_m2 ?? 0);
      entry.count += 1;
      map.set(key, entry);
    }
    return Array.from(map.values()).sort((a, b) => a.cat.type.localeCompare(b.cat.type));
  }, [planningState]);

  const activeColor = "#10B981";

  // ── Line Snap constants ──
  const SNAP_ANGLE_DEG = 8;   // degrees — snap to nearest 45° if within this tolerance
  const SNAP_ENDPOINT_PX = 14; // pixels — snap to existing endpoint if within this distance

  // Snap a candidate end-point toward 45°/90° angles or existing endpoints
  const snapLinePoint = React.useCallback((p: Point, anchor: Point | null): Point => {
    if (!anchor) return p;
    // Endpoint snap: attract to existing line endpoints (canvas coords)
    if (planningState) {
      for (const item of planningState.items) {
        if (item.type === "line") {
          const obj = item.raw_object;
          const candidates: Point[] = [
            { x: Number(obj.x1) * displayScale, y: Number(obj.y1) * displayScale },
            { x: Number(obj.x2) * displayScale, y: Number(obj.y2) * displayScale },
          ];
          for (const ep of candidates) {
            const dx = p.x - ep.x, dy = p.y - ep.y;
            if (Math.sqrt(dx * dx + dy * dy) < SNAP_ENDPOINT_PX) return ep;
          }
        }
      }
    }
    // Angle snap: constrain to multiples of 45° from anchor
    const dx = p.x - anchor.x, dy = p.y - anchor.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 6) return p;
    const angleDeg = Math.atan2(dy, dx) * 180 / Math.PI;
    const snappedAngle = Math.round(angleDeg / 45) * 45;
    if (Math.abs(((angleDeg - snappedAngle) + 180) % 360 - 180) <= SNAP_ANGLE_DEG) {
      const snapRad = snappedAngle * Math.PI / 180;
      return { x: anchor.x + dist * Math.cos(snapRad), y: anchor.y + dist * Math.sin(snapRad) };
    }
    return p;
  }, [planningState, displayScale]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: "100%" }}>
      {error && <ErrorAlert message={error} onDismiss={() => setError("")} />}

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
            <button type="button" onClick={() => void handleResolveOpening("door")} style={{ padding: "8px 18px", borderRadius: 9, background: "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(255,75,75,0.3)" }}>🚪 דלת</button>
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
            <button type="button" onClick={() => void handleResolveWall(true)} style={{ padding: "8px 18px", borderRadius: 9, background: "var(--navy)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(27,58,107,0.3)" }}>✓ כן, זה קיר</button>
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

      {/* Step nav — horizontal stepper strip */}
      <div style={{ background: "#fff", borderBottom: "1px solid var(--s200)", padding: "16px 20px", flexShrink: 0, margin: "0 -24px", paddingLeft: 28, paddingRight: 28 }}>
        <div className="wizard-steps">
          {(
            [
              { s: 1 as WizardStep, label: "בחירת\nתוכנית",  canGo: true },
              { s: 2 as WizardStep, label: "כיול\nסקייל",    canGo: canStep2 },
              { s: 3 as WizardStep, label: "סימון\nתכולה",   canGo: canStep3 },
              { s: 4 as WizardStep, label: "כתב\nכמויות",    canGo: canStep4 },
              { s: 5 as WizardStep, label: "גזרות\nעבודה",   canGo: canStep4 },
            ]
          ).map(({ s, label, canGo }) => {
            const isActive = step === s;
            const isDone   = step > s;
            const isLocked = !canGo;
            return (
              <div
                key={s}
                className={`wizard-step${isDone ? " done" : ""}${isActive ? " active" : ""}`}
                style={{ opacity: isLocked ? 0.45 : 1, cursor: isLocked ? "not-allowed" : "pointer" }}
                onClick={() => {
                  if (!isLocked) {
                    if (s === 1) setStep(1);
                    if (s === 2 && canStep2) setStep(2);
                    if (s === 3 && canStep3) setStep(3);
                    if (s === 4 && canStep4) setStep(4);
                    if (s === 5 && canStep4) setStep(5);
                  }
                }}
              >
                <div className="step-circle">{isDone ? "✓" : s}</div>
                <div className="step-label" style={{ whiteSpace: "pre-line" }}>{label}</div>
              </div>
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
            <p className="text-base font-bold text-[var(--navy)] mb-1">📁 בחר תוכנית לעבודה</p>
            <p className="text-xs text-slate-400">בחר מהרשימה תוכנית שהועלתה בסדנת עבודה.</p>
          </div>
          {plans.length === 0 ? (
            <div className="rounded-lg p-4 text-sm text-amber-800" style={{ background: "#FFFBEB", border: "1px solid #FCD34D" }}>
              ⚠️ אין תוכניות זמינות. העלה קודם תוכנית ב&quot;סדנת עבודה&quot;.
            </div>
          ) : (
            <select
              className="w-full bg-white border-2 border-slate-300 rounded-lg px-3 py-2.5 text-sm font-medium"
              style={{ borderColor: selectedPlanId ? "var(--navy)" : "#CBD5E1", outline: "none" }}
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
              style={{ height: 48, padding: "0 28px", borderRadius: 10, background: selectedPlanId ? "var(--navy)" : "var(--s300)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: selectedPlanId ? "pointer" : "not-allowed", transition: "all 0.15s" }}>
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
            {planningState.scale_px_per_meter > 0 && planningState.scale_px_per_meter !== 200 && (
              <div style={{ background: "#F0FDF4", border: "1px solid #86EFAC", borderRight: "4px solid #22C55E", borderRadius: 10, padding: "10px 14px", marginBottom: 12, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>✅</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: "#15803D" }}>כיול קיים</div>
                  <div style={{ fontSize: 11, color: "#166534" }}>סקייל: {planningState.scale_px_per_meter.toFixed(1)} px/m — ניתן לדלג לשלב 3</div>
                </div>
                <button type="button" onClick={() => setStep(3)}
                  style={{ padding: "7px 16px", borderRadius: 9, background: "#15803D", color: "#fff", border: "none", fontWeight: 700, fontSize: 12, cursor: "pointer" }}>
                  דלג לשלב 3 →
                </button>
              </div>
            )}
            <p className="text-xs text-slate-500 mb-3">גרור קו על אורך ידוע בתוכנית, הזן את האורך האמיתי ולחץ &quot;עדכן סקייל&quot;.</p>
            <div style={{ background: "#1A2744", borderRadius: 12, padding: 8, display: "inline-block" }}>
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
                style={{ touchAction: "none" }}
                onMouseDown={handleCalMouseDown}
                onMouseMove={handleCalMouseMove}
                onMouseUp={handleCalMouseUp}
                onMouseLeave={() => setCalDrawing(false)}
                onTouchStart={makeTouchHandler(handleCalMouseDown)}
                onTouchMove={makeTouchHandler(handleCalMouseMove)}
                onTouchEnd={makeTouchHandler(handleCalMouseUp)}
              >
                {calStart && (calEnd || calTemp) && (
                  <line x1={calStart.x} y1={calStart.y} x2={(calEnd ?? calTemp)?.x ?? calStart.x} y2={(calEnd ?? calTemp)?.y ?? calStart.y} stroke="var(--orange)" strokeWidth={3} />
                )}
                {calStart && <circle cx={calStart.x} cy={calStart.y} r={5} fill="var(--orange)" />}
                {calEnd && <circle cx={calEnd.x} cy={calEnd.y} r={5} fill="var(--orange)" />}
              </svg>
            </div>
            </div>
          </div>

          {/* Calibration controls */}
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
              <p className="text-sm font-semibold text-[#31333F]">בקרת כיול</p>
              <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-700 space-y-1">
                <p>סקייל נוכחי: <span className="font-semibold text-[var(--navy)]">{planningState.scale_px_per_meter.toFixed(1)} px/m</span></p>
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
                className="btn btn-primary btn-full"
                style={{ cursor: (!calStart || !calEnd) ? "not-allowed" : "pointer", opacity: (!calStart || !calEnd) ? 0.5 : 1 }}>
                📏 עדכן סקייל
              </button>
              <button type="button" onClick={() => { setCalStart(null); setCalEnd(null); setCalTemp(null); setCalDrawing(false); }}
                className="btn btn-ghost btn-full" style={{ marginTop: 6 }}>
                נקה קו
              </button>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800 space-y-1">
              <p className="font-semibold">💡 טיפ</p>
              <p>גרור קו על קיר שאורכו ידוע (למשל 5 מטר). הכיול יחושב אוטומטית.</p>
              <p>אחרי כיול מדויק, תוצאות המדידה יהיו מדויקות יותר.</p>
            </div>
          </div>

          <div className="xl:col-span-2" style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
            <button type="button" onClick={() => setStep(1)} className="btn btn-ghost">← חזור לשלב 1</button>
            <button type="button" onClick={() => setStep(3)} className="btn btn-orange">המשך לשלב 3 ←</button>
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

      {/* ── STEP 3: canvas-left + panel-right ── */}
      {step === 3 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px]" style={{ border: "1px solid var(--s200)", borderRadius: "var(--r)", overflow: "hidden", boxShadow: "var(--sh1)", minHeight: 560 }}>

          {/* ── LEFT: Dark canvas area ── */}
          <div ref={canvasContainerRef} style={{ background: "#1A2744", position: "relative", overflow: "auto", display: "flex", alignItems: "flex-start", justifyContent: "center", padding: 16, minHeight: 400 }}>
            <PlanningCanvasErrorBoundary>
            <div className="relative select-none" style={{ flexShrink: 0 }}>
              <img
                ref={drawingImageRef}
                src={imageUrl}
                alt="plan"
                className="block"
                style={{ width: displaySize.width, height: displaySize.height }}
                onLoad={() => updateDisplaySizeFromImage(drawingImageRef.current)}
                draggable={false}
              />

              {/* AUTO: segment overlays (clickable) */}
              {step3Tab === "auto" && autoSegments !== null && autoSegments.length > 0 && (
                <svg width={displaySize.width} height={displaySize.height} className="absolute inset-0">
                  {autoSegments
                    .filter(seg => seg.suggested_subtype !== "פרט קטן")
                    .map((seg, idx) => {
                    const [bx, by, bw, bh] = seg.bbox.map(v => v * displayScale);
                    const checked = autoSelected.has(seg.segment_id);
                    const isFixture = seg.element_class === "fixture";
                    const color = isFixture ? (checked ? "#7C3AED" : "#A78BFA") : seg.confidence >= 0.8 ? "#10B981" : seg.confidence >= 0.6 ? "#F59E0B" : "#EF4444";
                    const opacity = checked ? 0.35 : 0.1;
                    return (
                      <g key={seg.segment_id} style={{ cursor: "pointer" }}
                        onClick={() => setAutoSelected(prev => { const n = new Set(prev); checked ? n.delete(seg.segment_id) : n.add(seg.segment_id); return n; })}>
                        <rect x={bx} y={by} width={bw} height={bh} fill={color} fillOpacity={opacity}
                          stroke={color} strokeWidth={checked ? 2.5 : 1} strokeDasharray={isFixture ? "5 3" : (checked ? "none" : "6 3")} />
                        <text x={bx + 4} y={by + 14} fill={color} fontSize={Math.max(9, Math.min(13, bw / 8))} fontWeight="700" style={{ pointerEvents: "none" }}>
                          {isFixture ? "🔧" : idx + 1}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}

              {/* AUTO: no analysis yet — overlay prompt */}
              {step3Tab === "auto" && autoSegments === null && (
                <div className="absolute inset-0 flex items-center justify-center" style={{ background: "rgba(0,0,0,.25)", borderRadius: 8 }}>
                  <div style={{ background: "rgba(255,255,255,.92)", borderRadius: 12, padding: "16px 24px", textAlign: "center" }}>
                    <p style={{ fontSize: 14, fontWeight: 700, color: "var(--navy)", marginBottom: 4 }}>לחץ "נתח" בפאנל</p>
                    <p style={{ fontSize: 12, color: "#64748B" }}>המערכת תסרוק ותסמן אזורים</p>
                  </div>
                </div>
              )}

              {/* ZONE: interactive zone SVG */}
              {step3Tab === "zone" && (
                <svg ref={zoneCanvasRef} width={displaySize.width} height={displaySize.height}
                  className="absolute inset-0 cursor-crosshair" style={{ touchAction: "none" }}
                  onMouseDown={handleZoneMouseDown} onMouseMove={handleZoneMouseMove}
                  onMouseUp={handleZoneMouseUp} onMouseLeave={() => setZoneDrawing(false)}
                  onTouchStart={makeTouchHandler(handleZoneMouseDown)}
                  onTouchMove={makeTouchHandler(handleZoneMouseMove)}
                  onTouchEnd={makeTouchHandler(handleZoneMouseUp)}>
                  {planningState.items.filter(it => it.type === "zone" || it.type === "rect").map(item => {
                    const obj = item.raw_object;
                    const cat = planningState.categories[item.category];
                    const color = getCategoryColor(cat?.type, cat?.subtype);
                    return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale}
                      width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale}
                      fill={hexToRgba(color, 0.2)} stroke={color} strokeWidth={2} />;
                  })}
                  {zoneStart && (zoneEnd ?? zoneTemp) && (() => {
                    const end = zoneEnd ?? zoneTemp!;
                    return <rect x={Math.min(zoneStart.x, end.x)} y={Math.min(zoneStart.y, end.y)}
                      width={Math.abs(end.x - zoneStart.x)} height={Math.abs(end.y - zoneStart.y)}
                      fill={hexToRgba("var(--navy)", 0.15)} stroke="var(--navy)" strokeWidth={2} strokeDasharray="8 4" />;
                  })()}
                </svg>
              )}

              {/* MANUAL: interactive drawing SVG */}
              {step3Tab === "manual" && (
                <svg ref={drawingSurfaceRef} width={displaySize.width} height={displaySize.height}
                  className="absolute inset-0 cursor-crosshair" style={{ touchAction: "none" }}
                  onMouseDown={handleCanvasMouseDown} onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp} onMouseLeave={() => setDrawing(false)}
                  onTouchStart={makeTouchHandler(handleCanvasMouseDown)}
                  onTouchMove={makeTouchHandler(handleCanvasMouseMove)}
                  onTouchEnd={makeTouchHandler(handleCanvasMouseUp)}>
                  {planningState.items.map((item) => {
                    const obj = item.raw_object;
                    const cat = planningState.categories[item.category];
                    const color = getCategoryColor(cat?.type, cat?.subtype);
                    if (item.type === "line") return <line key={item.uid} x1={Number(obj.x1) * displayScale} y1={Number(obj.y1) * displayScale} x2={Number(obj.x2) * displayScale} y2={Number(obj.y2) * displayScale} stroke={color} strokeWidth={2} strokeLinecap="round" />;
                    if (item.type === "rect" || item.type === "zone") return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale} width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale} fill={hexToRgba(color, 0.15)} stroke={color} strokeWidth={2} />;
                    const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map(([px, py]) => `${px * displayScale},${py * displayScale}`).join(" ") : "";
                    return <polyline key={item.uid} points={pts} fill="none" stroke={color} strokeWidth={2} />;
                  })}
                  {/* Focus highlight ring */}
                  {focusedUid && (() => {
                    const fi = planningState.items.find(i => i.uid === focusedUid);
                    if (!fi) return null;
                    const fo = fi.raw_object;
                    const ds = displayScale;
                    if (fi.type === "line") {
                      const mx = ((Number(fo.x1) + Number(fo.x2)) / 2) * ds;
                      const my = ((Number(fo.y1) + Number(fo.y2)) / 2) * ds;
                      return <circle cx={mx} cy={my} r={18} fill="none" stroke="#FBBF24" strokeWidth={3} strokeDasharray="5 3" style={{ pointerEvents: "none", animation: "pulse 1s infinite" }} />;
                    }
                    if (fi.type === "rect" || fi.type === "zone") {
                      return <rect x={Number(fo.x) * ds - 4} y={Number(fo.y) * ds - 4} width={Number(fo.width) * ds + 8} height={Number(fo.height) * ds + 8} fill="none" stroke="#FBBF24" strokeWidth={3} strokeDasharray="5 3" rx={4} style={{ pointerEvents: "none", animation: "pulse 1s infinite" }} />;
                    }
                    return null;
                  })()}
                  {pendingShapes.map(renderPendingOnCanvas)}
                  {drawing && startPoint && tempPoint && drawMode === "line" && <line x1={startPoint.x} y1={startPoint.y} x2={tempPoint.x} y2={tempPoint.y} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                  {drawing && startPoint && tempPoint && drawMode === "line" && <circle cx={tempPoint.x} cy={tempPoint.y} r={4} fill="#FFF" stroke={PENDING_COLOR} strokeWidth={1.5} style={{ pointerEvents: "none" }} />}
                  {drawing && startPoint && tempPoint && drawMode === "rect" && <rect x={Math.min(startPoint.x, tempPoint.x)} y={Math.min(startPoint.y, tempPoint.y)} width={Math.abs(tempPoint.x - startPoint.x)} height={Math.abs(tempPoint.y - startPoint.y)} fill={hexToRgba(PENDING_COLOR, 0.15)} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                  {drawing && drawMode === "path" && pathPoints.length > 1 && <polyline points={pathPoints.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                </svg>
              )}
            </div>
            </PlanningCanvasErrorBoundary>

            {/* Zoom indicator — top-right */}
            <div style={{ position: "absolute", top: 12, right: 12, background: "rgba(0,0,0,.35)", color: "rgba(255,255,255,.6)", fontSize: 11, padding: "3px 10px", borderRadius: 20, pointerEvents: "none" }}>
              {zoomPercent}%
            </div>

            {/* Floating tools pill — bottom-center */}
            <div style={{ position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 4, background: "rgba(10,20,40,.65)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,.08)", borderRadius: 30, padding: "5px 8px" }}>
              <button type="button" title="הקטן" onClick={() => setZoomPercent(z => Math.max(70, z - 10))} style={{ width: 34, height: 34, borderRadius: "50%", border: "none", background: "transparent", color: "rgba(255,255,255,.7)", fontSize: 17, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>−</button>
              <button type="button" title="הגדל" onClick={() => setZoomPercent(z => Math.min(220, z + 10))} style={{ width: 34, height: 34, borderRadius: "50%", border: "none", background: "transparent", color: "rgba(255,255,255,.7)", fontSize: 17, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>+</button>
              <button type="button" title="איפוס" onClick={() => setZoomPercent(100)} style={{ width: 34, height: 34, borderRadius: "50%", border: "none", background: "transparent", color: "rgba(255,255,255,.45)", fontSize: 11, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>⊙</button>
              <div style={{ width: 1, background: "rgba(255,255,255,.15)", margin: "4px 2px" }} />
              <button type="button" title="הגדלה" onClick={() => setZoomModalOpen(true)} style={{ width: 34, height: 34, borderRadius: "50%", border: "none", background: "transparent", color: "rgba(255,255,255,.7)", fontSize: 15, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>🔍</button>
              {lastAddedUid && (
                <button type="button" title="בטל אחרון" onClick={() => { void handleDeleteItem(lastAddedUid); setLastAddedUid(null); }} style={{ width: 34, height: 34, borderRadius: "50%", border: "none", background: "transparent", color: "#FCA5A5", fontSize: 17, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>↩</button>
              )}
              {pendingShapes.length > 0 && (
                <button type="button" onClick={() => setCategoryPickerOpen(true)} style={{ height: 34, borderRadius: 17, border: "none", background: PENDING_COLOR, color: "#fff", fontSize: 11, fontWeight: 700, cursor: "pointer", padding: "0 12px", animation: "pulse 1.8s infinite", whiteSpace: "nowrap" }}>
                  📂 {pendingShapes.length}
                </button>
              )}
            </div>
          </div>

          {/* ── RIGHT: Panel ── */}
          <div style={{ background: "#fff", borderRight: "1px solid var(--s200)", display: "flex", flexDirection: "column", overflow: "hidden" }}>

            {/* Panel tabs */}
            <div className="tabs" style={{ flexShrink: 0, borderRadius: 0, marginBottom: 0 }}>
              {(["auto", "zone", "manual", "text"] as Step3Tab[]).map(tab => {
                const labels: Record<Step3Tab, string> = { auto: "🤖 אוטו", zone: "🎨 אזור", manual: "✏️ ציור", text: "📋 טקסט" };
                return (
                  <button key={tab} type="button" onClick={() => setStep3Tab(tab)}
                    className={`tab-btn${step3Tab === tab ? " active" : ""}`}
                    style={{ flex: 1, textAlign: "center", fontSize: 11.5 }}>
                    {labels[tab]}
                  </button>
                );
              })}
            </div>

            {/* ── Sticky: Add Category (always visible, between tab bar and body) ── */}
            <details className="text-xs" style={{ flexShrink: 0, borderBottom: "1px solid var(--s200)", background: "#FAFBFC" }}>
              <summary style={{
                cursor: "pointer", userSelect: "none",
                display: "flex", alignItems: "center", gap: 6, listStyle: "none",
                padding: "8px 16px",
                background: "var(--navy)", color: "#fff",
                fontWeight: 700, fontSize: 12,
              }}>
                <span style={{ fontSize: 15, lineHeight: 1 }}>＋</span> הוסף קטגוריה
                <span style={{ marginRight: "auto", fontSize: 10, opacity: 0.7 }}>▼</span>
              </summary>
              <div style={{ padding: "10px 16px 12px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <label style={{ fontSize: 12 }}>סוג
                  <select className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newType} onChange={e => setNewType(e.target.value)}>
                    {Object.keys(CATEGORY_SUBTYPES).map(t => <option key={t}>{t}</option>)}
                  </select>
                </label>
                <label style={{ fontSize: 12 }}>תת-סוג
                  <select className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newSubtype} onChange={e => setNewSubtype(e.target.value)}>{subtypeOptions.map(s => <option key={s}>{s}</option>)}</select>
                </label>
                <label style={{ fontSize: 12 }}>גובה / עובי (מ&apos;)
                  <input type="number" className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newParamValue} onChange={e => setNewParamValue(Number(e.target.value))} step={0.1} min={0} />
                </label>
                <label style={{ fontSize: 12 }}>הערה
                  <input className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs" value={newParamNote} onChange={e => setNewParamNote(e.target.value)} placeholder="אופציונלי" />
                </label>
                <button type="button" onClick={() => { handleAddCategory(); void handleSaveCategories(); }}
                  style={{ gridColumn: "1 / -1", padding: "7px 0", borderRadius: 8, background: "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 12, cursor: "pointer" }}>
                  ✅ הוסף ושמור
                </button>
              </div>
            </details>

            {/* Panel body — scrollable */}
            <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px" }}>

              {/* ── TAB: AUTO ── */}
              {step3Tab === "auto" && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                      <p className="text-sm font-semibold text-[#31333F]">ניתוח אוטומטי</p>
                      <p className="text-xs text-slate-500 mt-0.5">לחץ לניתוח — המערכת מסמנת קירות ומציעה קטגוריות.</p>
                    </div>
                    <button type="button" onClick={() => void handleAutoAnalyze()} disabled={autoLoading}
                      style={{ padding: "9px 16px", borderRadius: 10, background: autoLoading ? "#475569" : "var(--navy)", color: "#fff", border: "none", fontWeight: 700, fontSize: 13, cursor: autoLoading ? "not-allowed" : "pointer", boxShadow: autoLoading ? "none" : "0 2px 10px rgba(27,58,107,0.28)", transition: "all 0.15s", display: "flex", alignItems: "center", gap: 6 }}>
                      {autoLoading ? <><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.4)", borderTopColor: "#fff", animation: "spin 0.7s linear infinite" }} />מנתח...</> : "🤖 נתח"}
                    </button>
                  </div>

                  {/* ── Banner: Auto-create categories from Vision ── */}
                  {visionCatSuggestions.length > 0 && (
                    <div style={{
                      background: "linear-gradient(135deg, #F0FDF4, #EFF6FF)",
                      border: "1.5px solid #86EFAC",
                      borderRight: "4px solid #22C55E",
                      borderRadius: 12,
                      padding: "12px 14px",
                    }}>
                      <div style={{ fontWeight: 700, fontSize: 13, color: "#15803D", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                        ✨ זוהו {visionCatSuggestions.length} קטגוריות מהמקרא
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 10 }}>
                        {visionCatSuggestions.map(s => {
                          const color = getCategoryColor(s.type, s.subtype);
                          return (
                            <span key={`${s.type}/${s.subtype}`} style={{ background: hexToRgba(color, 0.12), border: `1px solid ${hexToRgba(color, 0.4)}`, borderRadius: 6, padding: "2px 8px", fontSize: 11, color: "var(--text-1)", fontWeight: 600 }}>
                              {s.subtype}
                            </span>
                          );
                        })}
                      </div>
                      <button
                        type="button"
                        onClick={() => void handleAutoCreateCategoriesFromVision()}
                        disabled={loading}
                        style={{ width: "100%", padding: "9px 0", borderRadius: 10, background: loading ? "#94a3b8" : "#15803D", color: "#fff", border: "none", fontWeight: 700, fontSize: 13, cursor: loading ? "not-allowed" : "pointer", boxShadow: "0 2px 8px rgba(21,128,61,0.25)" }}
                      >
                        {loading ? "יוצר קטגוריות..." : "✅ צור קטגוריות ושייך אוטומטית"}
                      </button>
                    </div>
                  )}

                  {autoSegments !== null && autoSegments.length === 0 && (
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-800">
                      <div className="font-semibold mb-1">לא זוהו קירות/אזורים.</div>
                      <div className="text-xs">נסה להעלות ולנתח בסדנת עבודה.</div>
                    </div>
                  )}

                  {autoSegments !== null && autoSegments.length > 0 && (() => {
                    const wallSegs = autoSegments.filter(s => s.element_class !== "fixture");
                    const fixSegs  = autoSegments.filter(s => s.element_class === "fixture");
                    return (
                      <>
                        <div className="flex gap-2 flex-wrap items-center">
                          <span className="text-xs text-slate-500">{wallSegs.length} קירות · {fixSegs.length} אביזרים</span>
                          <button type="button" onClick={() => setAutoSelected(new Set(autoSegments.map(s => s.segment_id)))} className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">בחר הכל</button>
                          <button type="button" onClick={() => setAutoSelected(new Set())} className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">בטל הכל</button>
                          <button type="button" onClick={() => setAutoSelected(new Set(autoSegments.filter(s => s.element_class !== "fixture" && s.confidence >= 0.8).map(s => s.segment_id)))} className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50">ביטחון {">"}80%</button>
                        </div>

                        {/* ── Bulk-assign panel ── */}
                        {(() => {
                          const groups = autoSegments.reduce<Record<string, { type: string; subtype: string; count: number }>>((acc, seg) => {
                            const k = `${seg.suggested_type}/${seg.suggested_subtype}`;
                            if (!acc[k]) acc[k] = { type: seg.suggested_type, subtype: seg.suggested_subtype, count: 0 };
                            acc[k].count++;
                            return acc;
                          }, {});
                          const groupList = Object.values(groups);
                          return (
                            <div style={{ marginTop: 2, marginBottom: 2 }}>
                              <div
                                onClick={() => setBulkOpen(p => !p)}
                                style={{
                                  display: "flex", alignItems: "center", gap: 7, cursor: "pointer",
                                  padding: "6px 10px", borderRadius: "var(--r-sm)",
                                  background: bulkOpen ? "var(--blue-50)" : "var(--s100)",
                                  border: `1px solid ${bulkOpen ? "#93C5FD" : "var(--s200)"}`,
                                  userSelect: "none",
                                }}
                              >
                                <span style={{ fontSize: 13 }}>⚡</span>
                                <span style={{ fontSize: 11, fontWeight: 700, flex: 1, color: "var(--navy)" }}>שיוך מהיר לפי סוג</span>
                                <span style={{ fontSize: 10, background: "var(--s200)", color: "var(--s700)", borderRadius: 10, padding: "1px 7px", fontWeight: 600 }}>{groupList.length}</span>
                                <span style={{ fontSize: 11, color: "var(--s400)", marginRight: 2 }}>{bulkOpen ? "▲" : "▼"}</span>
                              </div>
                              {bulkOpen && (
                                <div style={{ marginTop: 5, display: "flex", flexDirection: "column", gap: 4 }}>
                                  {groupList.map(({ type, subtype, count }) => {
                                    const gKey = `${type}/${subtype}`;
                                    const existingCatKey = (() => {
                                      const tally: Record<string, number> = {};
                                      autoSegments
                                        .filter(s => s.suggested_type === type && s.suggested_subtype === subtype)
                                        .forEach(s => { const ck = autoConfirmedKeys[s.segment_id]; if (ck) tally[ck] = (tally[ck] || 0) + 1; });
                                      return Object.entries(tally).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
                                    })();
                                    const selectedKey = bulkCatKeys[gKey] ?? existingCatKey;
                                    return (
                                      <div key={gKey} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 10px", borderRadius: "var(--r-sm)", background: "var(--s50)", border: "1px solid var(--s200)" }}>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--s700)" }}>{type}/{subtype}</span>
                                          <span style={{ fontSize: 10, color: "var(--s400)", marginRight: 4 }}>— {count} פריטים</span>
                                        </div>
                                        <select
                                          value={selectedKey}
                                          onChange={e => setBulkCatKeys(p => ({ ...p, [gKey]: e.target.value }))}
                                          style={{ fontSize: 11, border: "1px solid var(--s300)", borderRadius: 5, padding: "3px 6px", background: "#fff", color: "var(--s700)", flexShrink: 0, maxWidth: 110 }}
                                        >
                                          <option value="">-- --</option>
                                          {Object.values(planningState.categories).map(c => (
                                            <option key={c.key} value={c.key}>{c.type}/{c.subtype}</option>
                                          ))}
                                        </select>
                                        <button
                                          type="button"
                                          disabled={!selectedKey}
                                          onClick={() => {
                                            if (!selectedKey) return;
                                            const ids = autoSegments
                                              .filter(s => s.suggested_type === type && s.suggested_subtype === subtype)
                                              .map(s => s.segment_id);
                                            setAutoConfirmedKeys(prev => { const next = { ...prev }; ids.forEach(id => { next[id] = selectedKey; }); return next; });
                                            setAutoSelected(prev => { const next = new Set(prev); ids.forEach(id => next.add(id)); return next; });
                                          }}
                                          style={{ fontSize: 11, fontWeight: 700, padding: "4px 10px", borderRadius: "var(--r-sm)", border: "none", cursor: selectedKey ? "pointer" : "not-allowed", background: selectedKey ? "var(--navy)" : "var(--s300)", color: selectedKey ? "#fff" : "var(--s500)", flexShrink: 0, opacity: selectedKey ? 1 : 0.6 }}
                                        >
                                          שייך
                                        </button>
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          );
                        })()}

                        {wallSegs.length > 0 && (
                          <div>
                            {/* ── Collapsible walls group header ── */}
                            <div
                              onClick={() => setExpandedGroups(prev => { const n = new Set(prev); n.has("walls") ? n.delete("walls") : n.add("walls"); return n; })}
                              style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", padding: "7px 10px", borderRadius: "var(--r-sm)", background: "var(--s100)", marginBottom: 6, userSelect: "none" }}
                            >
                              <span style={{ fontSize: 14 }}>🧱</span>
                              <span style={{ fontSize: 11, fontWeight: 700, flex: 1, color: "var(--s700)" }}>קירות וקטעים</span>
                              <span style={{ fontSize: 10, background: "var(--s200)", color: "var(--s500)", borderRadius: 10, padding: "1px 7px", fontWeight: 600 }}>{wallSegs.length}</span>
                              <span style={{ fontSize: 11, color: "var(--s400)", marginRight: 2 }}>{expandedGroups.has("walls") ? "▲" : "▼"}</span>
                            </div>
                            {expandedGroups.has("walls") && wallSegs.map((seg) => {
                              const checked = autoSelected.has(seg.segment_id);
                              const catKey = autoConfirmedKeys[seg.segment_id] ?? "";
                              const conf = seg.confidence;
                              return (
                                <div
                                  key={seg.segment_id}
                                  onClick={() => setAutoSelected(prev => { const n = new Set(prev); checked ? n.delete(seg.segment_id) : n.add(seg.segment_id); return n; })}
                                  style={{
                                    background: checked ? "var(--blue-50)" : "var(--s50)",
                                    borderRadius: "var(--r-sm)",
                                    border: `1px solid ${checked ? "#93C5FD" : "var(--s200)"}`,
                                    padding: "9px 11px", marginBottom: 5,
                                    display: "flex", alignItems: "center", gap: 9,
                                    cursor: "pointer",
                                  }}
                                >
                                  <div style={{ width: 17, height: 17, borderRadius: 4, background: checked ? "var(--blue)" : "var(--s300)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 10, flexShrink: 0, fontWeight: 700 }}>
                                    {checked ? "✓" : "—"}
                                  </div>
                                  <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: 12, fontWeight: 600, color: "var(--s900)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{seg.suggested_type}/{seg.suggested_subtype}</div>
                                    <div style={{ fontSize: 11, color: "var(--s400)" }}>{seg.length_m.toFixed(1)} מ׳ · {Math.round(conf * 100)}%</div>
                                  </div>
                                  <select
                                    value={catKey}
                                    onChange={e => setAutoConfirmedKeys(prev => ({ ...prev, [seg.segment_id]: e.target.value }))}
                                    onClick={e => e.stopPropagation()}
                                    style={{ fontSize: 11, border: "1px solid var(--s300)", borderRadius: 5, padding: "3px 6px", color: "var(--s700)", background: "#fff", flexShrink: 0, maxWidth: 110 }}
                                  >
                                    <option value="">-- --</option>
                                    {Object.values(planningState.categories).map(c => (
                                      <option key={c.key} value={c.key}>{c.type}/{c.subtype}</option>
                                    ))}
                                  </select>
                                </div>
                              );
                            })}
                          </div>
                        )}

                        {fixSegs.length > 0 && (() => {
                          // Split: known (plumbing/furniture) vs unidentified (פרט קטן)
                          const knownSegs    = fixSegs.filter(s => s.suggested_subtype !== "פרט קטן");
                          const unknownSegs  = fixSegs.filter(s => s.suggested_subtype === "פרט קטן");
                          // Group known fixtures by suggested_subtype
                          const knownGroups  = knownSegs.reduce<Record<string, typeof fixSegs>>((acc, seg) => {
                            const key = seg.suggested_subtype || "אחר";
                            if (!acc[key]) acc[key] = [];
                            acc[key].push(seg);
                            return acc;
                          }, {});
                          // Reusable segment row
                          const FixRow = ({ seg, indented }: { seg: typeof fixSegs[0]; indented?: boolean }) => {
                            const checked = autoSelected.has(seg.segment_id);
                            const catKey  = autoConfirmedKeys[seg.segment_id] ?? "";
                            return (
                              <div
                                key={seg.segment_id}
                                onClick={() => setAutoSelected(prev => { const n = new Set(prev); checked ? n.delete(seg.segment_id) : n.add(seg.segment_id); return n; })}
                                style={{ background: checked ? "#EDE9FE" : "var(--s50)", borderRadius: "var(--r-sm)", border: `1px solid ${checked ? "#C4B5FD" : "var(--s200)"}`, padding: "9px 11px", marginBottom: 5, marginRight: indented ? 12 : 0, display: "flex", alignItems: "center", gap: 9, cursor: "pointer" }}
                              >
                                <div style={{ width: 17, height: 17, borderRadius: 4, background: checked ? "#7C3AED" : "var(--s300)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 10, flexShrink: 0, fontWeight: 700 }}>
                                  {checked ? "✓" : "—"}
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                  <div style={{ fontSize: 12, fontWeight: 600, color: "#7C3AED", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{seg.label}</div>
                                  <div style={{ fontSize: 11, color: "var(--s400)" }}>{seg.area_m2.toFixed(2)} מ"ר</div>
                                </div>
                                <select value={catKey} onChange={e => setAutoConfirmedKeys(prev => ({ ...prev, [seg.segment_id]: e.target.value }))} onClick={e => e.stopPropagation()} style={{ fontSize: 11, border: "1px solid #E9D5FF", borderRadius: 5, padding: "3px 6px", color: "var(--s700)", background: "#fff", flexShrink: 0, maxWidth: 110 }}>
                                  <option value="">-- ללא --</option>
                                  {Object.values(planningState.categories).map(c => (
                                    <option key={c.key} value={c.key}>{c.type}/{c.subtype}</option>
                                  ))}
                                </select>
                              </div>
                            );
                          };
                          return (
                            <div>
                              {/* ── Parent: אביזרים ── */}
                              <div onClick={() => setExpandedGroups(prev => { const n = new Set(prev); n.has("fixtures") ? n.delete("fixtures") : n.add("fixtures"); return n; })}
                                style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", padding: "7px 10px", borderRadius: "var(--r-sm)", background: "#F5F3FF", marginBottom: 6, userSelect: "none" }}>
                                <span style={{ fontSize: 14 }}>🔧</span>
                                <span style={{ fontSize: 11, fontWeight: 700, flex: 1, color: "#5B21B6" }}>אביזרים</span>
                                <span style={{ fontSize: 10, background: "#EDE9FE", color: "#7C3AED", borderRadius: 10, padding: "1px 7px", fontWeight: 600 }}>{fixSegs.length}</span>
                                <span style={{ fontSize: 11, color: "#A78BFA", marginRight: 2 }}>{expandedGroups.has("fixtures") ? "▲" : "▼"}</span>
                              </div>

                              {expandedGroups.has("fixtures") && (
                                <>
                                  {/* ── Section A: אביזרים מזוהים ── */}
                                  {knownSegs.length > 0 && (
                                    <div style={{ marginBottom: 8 }}>
                                      <div style={{ fontSize: 10, fontWeight: 700, color: "var(--s500)", textTransform: "uppercase", letterSpacing: "0.6px", padding: "0 4px", marginBottom: 5 }}>
                                        🏠 אביזרים מזוהים ({knownSegs.length})
                                      </div>
                                      {Object.entries(knownGroups).map(([subtypeKey, segs]) => {
                                        const { label, icon } = getFixGroupLabel(subtypeKey);
                                        const subKey = `fix_k_${subtypeKey}`;
                                        const subOpen = expandedGroups.has(subKey);
                                        const anyChecked = segs.some(s => autoSelected.has(s.segment_id));
                                        return (
                                          <div key={subtypeKey} style={{ marginBottom: 5 }}>
                                            <div onClick={() => setExpandedGroups(prev => { const n = new Set(prev); n.has(subKey) ? n.delete(subKey) : n.add(subKey); return n; })}
                                              style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", padding: "5px 10px 5px 18px", borderRadius: "var(--r-sm)", background: anyChecked ? "#EDE9FE" : "var(--s50)", border: `1px solid ${anyChecked ? "#C4B5FD" : "var(--s200)"}`, marginBottom: 4, userSelect: "none" }}>
                                              <span style={{ fontSize: 13 }}>{icon}</span>
                                              <span style={{ fontSize: 11, fontWeight: 600, flex: 1, color: "var(--s700)" }}>{label}</span>
                                              <span style={{ fontSize: 10, background: "#EDE9FE", color: "#7C3AED", borderRadius: 10, padding: "1px 6px", fontWeight: 600 }}>{segs.length}</span>
                                              <span style={{ fontSize: 10, color: "var(--s400)" }}>{subOpen ? "▲" : "▼"}</span>
                                            </div>
                                            {subOpen && segs.map(seg => <FixRow key={seg.segment_id} seg={seg} indented />)}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}

                                  {/* ── Section B: פרטים לא מזוהים ── closed by default, grayed */}
                                  {unknownSegs.length > 0 && (
                                    <div>
                                      <div onClick={() => setExpandedGroups(prev => { const n = new Set(prev); n.has("fix_unknown") ? n.delete("fix_unknown") : n.add("fix_unknown"); return n; })}
                                        style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", padding: "5px 10px", borderRadius: "var(--r-sm)", background: "var(--s100)", border: "1px dashed var(--s300)", marginBottom: 4, userSelect: "none", opacity: 0.8 }}>
                                        <span style={{ fontSize: 13 }}>📌</span>
                                        <span style={{ fontSize: 11, fontWeight: 600, flex: 1, color: "var(--s500)" }}>פרטים לא מזוהים</span>
                                        <span style={{ fontSize: 10, background: "var(--s200)", color: "var(--s500)", borderRadius: 10, padding: "1px 6px", fontWeight: 600 }}>{unknownSegs.length}</span>
                                        <span style={{ fontSize: 10, color: "var(--s400)" }}>{expandedGroups.has("fix_unknown") ? "▲" : "▼"}</span>
                                      </div>
                                      {expandedGroups.has("fix_unknown") && (
                                        <div style={{ opacity: 0.75 }}>
                                          {unknownSegs.map(seg => <FixRow key={seg.segment_id} seg={seg} />)}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                          );
                        })()}

                        <div className="flex gap-2 flex-wrap">
                          <button type="button" disabled={loading} onClick={() => void handleConfirmAutoSegments(false)}
                            style={{ padding: "9px 16px", borderRadius: 10, background: loading ? "#94a3b8" : "var(--navy)", color: "#fff", border: "none", fontWeight: 700, fontSize: 13, cursor: loading ? "not-allowed" : "pointer", transition: "all 0.15s" }}>
                            {loading ? "שומר..." : "✓ אשר הכל"}
                          </button>
                          <button type="button" disabled={loading || autoSelected.size === 0} onClick={() => void handleConfirmAutoSegments(true)}
                            style={{ padding: "9px 16px", borderRadius: 10, background: (loading || autoSelected.size === 0) ? "#94a3b8" : "var(--blue)", color: "#fff", border: "none", fontWeight: 700, fontSize: 13, cursor: (loading || autoSelected.size === 0) ? "not-allowed" : "pointer", transition: "all 0.15s" }}>
                            {loading ? "שומר..." : `✓ נבחרים (${autoSelected.size})`}
                          </button>
                          <button type="button" onClick={() => { setAutoSegments(null); setAutoVisionData(null); }}
                            style={{ padding: "9px 12px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 12, cursor: "pointer" }}>נקה</button>
                        </div>
                      </>
                    );
                  })()}

                  {/* Vision Data Panel */}
                  {autoVisionData && (() => {
                    const vd = autoVisionData;
                    const hasRooms = !!(vd.rooms && vd.rooms.length > 0);
                    const hasElements = !!(vd.elements && vd.elements.length > 0);
                    const hasMaterials = !!(vd.materials && vd.materials.length > 0);
                    const hasDims = !!(vd.dimensions && vd.dimensions.length > 0);
                    const hasSystems = !!(vd.systems && Object.keys(vd.systems).length > 0);
                    const hasNotes = !!(vd.execution_notes && vd.execution_notes.length > 0);
                    if (!hasRooms && !hasElements && !hasMaterials && !hasDims && !hasSystems && !hasNotes) return null;

                    const elementGroups: Record<string, Array<{ type: string; id?: string; location?: string; notes?: string }>> = {};
                    if (vd.elements) {
                      for (const el of vd.elements) {
                        const key = el.type || "אחר";
                        if (!elementGroups[key]) elementGroups[key] = [];
                        elementGroups[key].push(el);
                      }
                    }
                    const elementIcon = (type: string) => {
                      const t = type.toLowerCase();
                      if (t.includes("דלת")) return "🚪";
                      if (t.includes("חלון")) return "🪟";
                      if (t.includes("עמוד")) return "🏛️";
                      if (t.includes("מדרגות") || t.includes("מדרגה")) return "🪜";
                      if (t.includes("מעלית")) return "🛗";
                      if (t.includes("שירותים")) return "🚽";
                      if (t.includes("מקלחת") || t.includes("אמבט")) return "🚿";
                      return "📌";
                    };
                    const cards = [
                      hasRooms     ? { id: "rooms",      icon: "🏠", label: "חדרים",      count: vd.rooms!.length,                  color: "#EFF6FF", border: "#BFDBFE", text: "#1E40AF" } : null,
                      hasElements  ? { id: "elements",   icon: "🔧", label: "אלמנטים",    count: vd.elements!.length,               color: "#F0FDF4", border: "#BBF7D0", text: "#166534" } : null,
                      hasMaterials ? { id: "materials",  icon: "🧱", label: "חומרים",      count: vd.materials!.length,              color: "#FFF7ED", border: "#FED7AA", text: "#9A3412" } : null,
                      hasDims      ? { id: "dimensions", icon: "📏", label: "מידות",       count: vd.dimensions!.length,             color: "#FAF5FF", border: "#E9D5FF", text: "#6B21A8" } : null,
                      hasSystems   ? { id: "systems",    icon: "⚙️", label: "מערכות",      count: Object.keys(vd.systems!).length,   color: "#F8FAFC", border: "#CBD5E1", text: "#334155" } : null,
                      hasNotes     ? { id: "notes",      icon: "📝", label: "הערות",       count: vd.execution_notes!.length,        color: "#FFFBEB", border: "#FDE68A", text: "#92400E" } : null,
                    ].filter(Boolean) as Array<{ id: string; icon: string; label: string; count: number; color: string; border: string; text: string }>;
                    const active = visionActiveCard ?? (cards[0]?.id ?? null);

                    return (
                      <div style={{ marginTop: 16, borderTop: "2px solid #E2E8F0", paddingTop: 14 }}>
                        <div style={{ fontWeight: 700, fontSize: 13, color: "var(--navy)", marginBottom: 10 }}>🔍 נתוני Vision</div>
                        {(vd.plan_title || vd.project_name || vd.scale) && (
                          <div style={{ background: "#F8FAFC", border: "1px solid #E2E8F0", borderRadius: 8, padding: "8px 12px", marginBottom: 10, fontSize: 12 }}>
                            {vd.plan_title && <div><span style={{ color: "#94A3B8" }}>כותרת: </span>{vd.plan_title}</div>}
                            {vd.project_name && <div><span style={{ color: "#94A3B8" }}>פרויקט: </span>{vd.project_name}</div>}
                            {vd.scale && <div><span style={{ color: "#94A3B8" }}>קנ"מ: </span>{vd.scale}</div>}
                          </div>
                        )}
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 12 }}>
                          {cards.map(card => (
                            <button key={card.id} type="button" onClick={() => setVisionActiveCard(card.id)}
                              style={{ background: active === card.id ? card.color : "#fff", border: `2px solid ${active === card.id ? card.border : "#E2E8F0"}`, borderRadius: 9, padding: "8px 12px", cursor: "pointer", textAlign: "center", minWidth: 66, transition: "all 0.15s" }}>
                              <div style={{ fontSize: 18, marginBottom: 1 }}>{card.icon}</div>
                              <div style={{ fontWeight: 700, fontSize: 15, color: active === card.id ? card.text : "#1E293B" }}>{card.count}</div>
                              <div style={{ fontSize: 10, color: active === card.id ? card.text : "#64748B", fontWeight: 600 }}>{card.label}</div>
                            </button>
                          ))}
                        </div>
                        {active === "rooms" && hasRooms && (
                          <div style={{ background: "#EFF6FF", border: "1px solid #BFDBFE", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#1E40AF", marginBottom: 8 }}>🏠 חדרים ({vd.rooms!.length})</div>
                            <div style={{ overflowX: "auto" }}>
                              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                                <thead><tr style={{ background: "#DBEAFE" }}>
                                  <th style={{ textAlign: "right", padding: "5px 7px", borderBottom: "1px solid #BFDBFE" }}>שם</th>
                                  <th style={{ textAlign: "right", padding: "5px 7px", borderBottom: "1px solid #BFDBFE" }}>שטח</th>
                                  <th style={{ textAlign: "right", padding: "5px 7px", borderBottom: "1px solid #BFDBFE" }}>מידות</th>
                                  <th style={{ textAlign: "right", padding: "5px 7px", borderBottom: "1px solid #BFDBFE" }}>הערות</th>
                                </tr></thead>
                                <tbody>{vd.rooms!.map((r, i) => (
                                  <tr key={i} style={{ borderBottom: "1px solid #DBEAFE", background: i % 2 === 0 ? "#fff" : "#EFF6FF" }}>
                                    <td style={{ padding: "5px 7px", fontWeight: 600, color: "#1E40AF" }}>{r.name || "—"}</td>
                                    <td style={{ padding: "5px 7px" }}>{r.area_m2 ? `${r.area_m2}מ"ר` : "—"}</td>
                                    <td style={{ padding: "5px 7px" }}>{r.dimensions ?? "—"}</td>
                                    <td style={{ padding: "5px 7px", color: "#64748B" }}>{r.notes ?? "—"}</td>
                                  </tr>
                                ))}</tbody>
                              </table>
                            </div>
                          </div>
                        )}
                        {active === "elements" && hasElements && (
                          <div style={{ background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#166534", marginBottom: 8 }}>🔧 אלמנטים ({vd.elements!.length})</div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 8 }}>
                              {Object.entries(elementGroups).map(([type, items]) => (
                                <div key={type} style={{ background: "#fff", border: "1px solid #BBF7D0", borderRadius: 8, padding: "8px 10px" }}>
                                  <div style={{ fontWeight: 700, fontSize: 12, color: "#15803D", marginBottom: 4 }}>{elementIcon(type)} {type} ×{items.length}</div>
                                  {items.map((el, i) => (
                                    <div key={i} style={{ fontSize: 10, color: "#334155", padding: "2px 0", borderTop: i > 0 ? "1px solid #F0FDF4" : "none" }}>
                                      {el.id && <span style={{ color: "#64748B", marginLeft: 4, fontWeight: 600 }}>{el.id}</span>}
                                      {el.location && <span>{el.location}</span>}
                                      {!el.id && !el.location && <span style={{ color: "#94A3B8" }}>ללא פרטים</span>}
                                    </div>
                                  ))}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {active === "materials" && hasMaterials && (
                          <div style={{ background: "#FFF7ED", border: "1px solid #FED7AA", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#9A3412", marginBottom: 8 }}>🧱 חומרים ({vd.materials!.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                              {vd.materials!.map((m, i) => <div key={i} style={{ background: "#fff", border: "1px solid #FED7AA", borderRadius: 6, padding: "4px 8px", fontSize: 11, color: "#431407" }}>{m}</div>)}
                            </div>
                          </div>
                        )}
                        {active === "dimensions" && hasDims && (
                          <div style={{ background: "#FAF5FF", border: "1px solid #E9D5FF", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#6B21A8", marginBottom: 8 }}>📏 מידות ({vd.dimensions!.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                              {vd.dimensions!.map((d, i) => <div key={i} style={{ background: "#fff", border: "1px solid #E9D5FF", borderRadius: 6, padding: "4px 8px", fontSize: 11, color: "#4C1D95", fontFamily: "monospace" }}>{d}</div>)}
                            </div>
                          </div>
                        )}
                        {active === "systems" && hasSystems && (
                          <div style={{ background: "#F8FAFC", border: "1px solid #CBD5E1", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#334155", marginBottom: 8 }}>⚙️ מערכות</div>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                              {Object.entries(vd.systems!).filter(([, v]) => v).map(([k, v]) => (
                                <div key={k} style={{ background: "#fff", border: "1px solid #E2E8F0", borderRadius: 6, padding: "6px 8px", fontSize: 11 }}>
                                  <span style={{ color: "#64748B", fontWeight: 600 }}>{k}: </span><span>{String(v)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {active === "notes" && hasNotes && (
                          <div style={{ background: "#FFFBEB", border: "1px solid #FDE68A", borderRadius: 10, padding: 12 }}>
                            <div style={{ fontWeight: 600, fontSize: 12, color: "#92400E", marginBottom: 8 }}>📝 הערות ({vd.execution_notes!.length})</div>
                            {vd.execution_notes!.map((n, i) => (
                              <div key={i} style={{ fontSize: 11, color: "#78350F", padding: "4px 0", borderBottom: i < vd.execution_notes!.length - 1 ? "1px solid #FDE68A" : "none" }}>{i + 1}. {n}</div>
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
                <div className="space-y-3">
                  <p className="text-sm font-semibold text-[#31333F]">צביעת אזור</p>
                  <p className="text-xs text-slate-500">גרור מלבן סביב חדר/אזור — המערכת תחשב קירות בתוכו.</p>
                  <div className="flex gap-2 items-center flex-wrap">
                    <label className="text-xs w-full">קטגוריה לאזור:
                      <select value={zoneCatKey} onChange={e => setZoneCatKey(e.target.value)}
                        className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-xs">
                        <option value="">-- בחר --</option>
                        {Object.values(planningState.categories).map(c => (
                          <option key={c.key} value={c.key}>{c.type} / {c.subtype}</option>
                        ))}
                      </select>
                    </label>
                    {zoneStart && zoneEnd && (
                      <button type="button" onClick={() => void handleAddZone()} disabled={loading || !zoneCatKey}
                        className="btn btn-primary btn-full"
                        style={{ cursor: (loading || !zoneCatKey) ? "not-allowed" : "pointer", opacity: (loading || !zoneCatKey) ? 0.5 : 1 }}>
                        {loading ? "מחשב..." : "הוסף אזור"}
                      </button>
                    )}
                    {(zoneStart || zoneEnd) && (
                      <button type="button" onClick={() => { setZoneStart(null); setZoneEnd(null); setZoneTemp(null); }}
                        className="btn btn-ghost btn-full" style={{ marginTop: 4 }}>נקה</button>
                    )}
                  </div>
                </div>
              )}

              {/* ── TAB: MANUAL ── */}
              {step3Tab === "manual" && (
                <div className="space-y-3">
                  <p className="text-sm font-semibold text-[#31333F]">ציור ידני</p>
                  <div className="flex flex-wrap gap-2">
                    <select value={drawMode} onChange={(e) => setDrawMode(e.target.value as DrawMode)} className="bg-white border border-slate-300 rounded-lg px-2 py-1.5 text-xs flex-1">
                      <option value="line">קו</option>
                      <option value="rect">מלבן</option>
                      <option value="path">חופשי</option>
                    </select>
                  </div>
                  {pendingShapes.length > 0 && (
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: hexToRgba(PENDING_COLOR, 0.12), border: `1px solid ${hexToRgba(PENDING_COLOR, 0.4)}`, borderRadius: 8, padding: "8px 12px" }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "#92400e" }}>⏳ {pendingShapes.length} ממתין לשיוך</span>
                      <button type="button" onClick={() => setCategoryPickerOpen(true)} style={{ background: PENDING_COLOR, color: "#fff", border: "none", borderRadius: 7, padding: "5px 12px", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>שייך</button>
                    </div>
                  )}
                  <p className="text-xs text-slate-400">גרור על הקנבס לציור. לחץ "שייך" לשיוך לקטגוריה.</p>
                </div>
              )}

              {/* ── TAB: TEXT ── */}
              {step3Tab === "text" && (
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2 flex-wrap">
                    <div>
                      <p className="text-sm font-semibold text-[#31333F]">פריטים חופשיים</p>
                      <p className="text-xs text-slate-500">פריטים שלא ניתן לצייר (ספקלינג, פינות, תוספות וכו׳).</p>
                    </div>
                    <button type="button" disabled={loading}
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
                      style={{ padding: "7px 12px", borderRadius: 9, background: "var(--navy)", color: "#fff", border: "none", fontWeight: 700, fontSize: 12, cursor: "pointer", opacity: loading ? 0.5 : 1 }}>
                      📥 ייבא מתוכנית
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse" style={{ minWidth: 320 }}>
                      <thead>
                        <tr className="bg-slate-50 text-slate-500">
                          <th className="p-1.5 text-right font-medium border-b border-slate-200">תיאור *</th>
                          <th className="p-1.5 text-right font-medium border-b border-slate-200">קטגוריה</th>
                          <th className="p-1.5 text-right font-medium border-b border-slate-200 w-14">כמות</th>
                          <th className="p-1.5 text-right font-medium border-b border-slate-200 w-14">יחידה</th>
                          <th className="p-1.5 border-b border-slate-200 w-8"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {textRows.map((row, idx) => (
                          <tr key={idx} className="border-b border-slate-100">
                            <td className="p-0.5"><input value={row.description} onChange={e => handleTextRowChange(idx, "description", e.target.value)} placeholder="תיאור..." className="w-full border border-slate-300 rounded px-1.5 py-1 text-xs" /></td>
                            <td className="p-0.5">
                              <select value={row.category_key} onChange={e => handleTextRowChange(idx, "category_key", e.target.value)} className="w-full border border-slate-300 rounded px-1 py-1 text-xs">
                                <option value="__manual__">ידני</option>
                                {Object.values(planningState.categories).map(c => <option key={c.key} value={c.key}>{c.subtype}</option>)}
                              </select>
                            </td>
                            <td className="p-0.5"><input type="number" min={0} step={0.01} value={row.quantity} onChange={e => handleTextRowChange(idx, "quantity", Number(e.target.value))} className="w-full border border-slate-300 rounded px-1.5 py-1 text-xs" /></td>
                            <td className="p-0.5">
                              <select value={row.unit} onChange={e => handleTextRowChange(idx, "unit", e.target.value)} className="w-full border border-slate-300 rounded px-1 py-1 text-xs">
                                {["מ׳", 'מ"ר', "יח׳", "ק״ג", "ליטר", "מ״ק"].map(u => <option key={u}>{u}</option>)}
                              </select>
                            </td>
                            <td className="p-0.5 text-center"><button type="button" onClick={() => handleRemoveTextRow(idx)} className="text-red-400 hover:text-red-600">🗑</button></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex gap-2">
                    <button type="button" onClick={handleAddTextRow} className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs hover:bg-slate-50">+ שורה</button>
                    <button type="button" onClick={() => void handleSaveTextRows()} disabled={loading} className="px-4 py-1.5 rounded-lg bg-[var(--navy)] text-white text-xs font-semibold disabled:opacity-40">{loading ? "שומר..." : "💾 שמור"}</button>
                  </div>
                  {planningState.items.filter(it => it.type === "text").length > 0 && (
                    <div className="mt-2 border-t border-slate-100 pt-2">
                      <p className="text-xs font-semibold text-slate-500 mb-1.5">פריטי טקסט שנשמרו:</p>
                      <div className="space-y-1">
                        {planningState.items.filter(it => it.type === "text").map(item => (
                          <div key={item.uid} className="flex items-center justify-between text-xs bg-slate-50 border border-slate-200 rounded px-2 py-1">
                            <span className="font-medium truncate">{String(item.raw_object.description ?? "")}</span>
                            <span className="text-slate-500 mx-1 flex-shrink-0">{String(item.raw_object.quantity ?? "")} {String(item.raw_object.unit ?? "")}</span>
                            <button type="button" onClick={() => void handleDeleteItem(item.uid)} className="text-red-400 hover:text-red-600 flex-shrink-0">✕</button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── Shared: Categories + Items ── */}
              <div style={{ marginTop: 20, borderTop: "1px solid var(--s200)", paddingTop: 14 }}>
                {/* Categories */}
                <p className="text-xs font-semibold text-slate-500 mb-2">קטגוריות</p>
                <div className="space-y-1 max-h-[180px] overflow-y-auto mb-2">
                  {Object.values(planningState.categories).map((cat) => {
                    const color = getCategoryColor(cat.type, cat.subtype);
                    return (
                      <div key={cat.key} className="flex items-center gap-2 rounded-lg px-2 py-1 text-xs" style={{ background: hexToRgba(color, 0.08), border: `1px solid ${hexToRgba(color, 0.35)}` }}>
                        <span className="inline-block w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                        {cat.type} - {cat.subtype}
                      </div>
                    );
                  })}
                  {Object.keys(planningState.categories).length === 0 && <p className="text-xs text-slate-400">אין קטגוריות עדיין.</p>}
                </div>

                {/* Items list */}
                <p className="text-xs font-semibold text-slate-500 mb-1.5">פריטים ({planningState.items.length})</p>
                {planningState.items.length === 0
                  ? <p className="text-xs text-slate-400">הוסף פריטים דרך הטאבים.</p>
                  : <div className="space-y-1 max-h-[200px] overflow-y-auto">
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
            </div>

            {/* Real-time BOQ strip */}
            {liveBoq.length > 0 && (
              <div style={{ borderTop: "1px solid var(--s200)", background: "var(--s50)", padding: "6px 12px", flexShrink: 0, maxHeight: 120, overflowY: "auto" }}>
                <p style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 4, fontWeight: 600 }}>סיכום חי</p>
                {liveBoq.map(({ cat, lengthM, areaM2, count }) => {
                  const color = getCategoryColor(cat.type, cat.subtype);
                  const firstItemUid = planningState?.items.find(i => i.category === cat.key)?.uid;
                  return (
                    <div key={cat.key} className="flex items-center justify-between text-xs" style={{ gap: 6, marginBottom: 2 }}>
                      <span className="flex items-center gap-1 min-w-0">
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0, display: "inline-block" }} />
                        <span className="truncate" style={{ color: "var(--text-1)", fontSize: 10 }}>{cat.subtype}</span>
                      </span>
                      <span className="flex items-center gap-1" style={{ flexShrink: 0 }}>
                        <span style={{ color: "var(--text-2)", fontSize: 10, whiteSpace: "nowrap" }}>
                          {count} פריטים
                          {lengthM > 0 && ` · ${lengthM.toFixed(1)}מ׳`}
                          {areaM2 > 0 && ` · ${areaM2.toFixed(1)}מ"ר`}
                        </span>
                        {firstItemUid && (
                          <button type="button" title="מרכז על הפריט" onClick={() => focusOnItem(firstItemUid)}
                            style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, padding: "0 2px", color: "var(--text-3)", lineHeight: 1 }}>🎯</button>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Panel footer — navigation */}
            <div style={{ padding: "10px 14px", borderTop: "1px solid var(--s200)", background: "var(--s50)", display: "flex", gap: 8, flexShrink: 0 }}>
              <button type="button" onClick={() => setStep(2)} className="btn btn-ghost btn-sm">
                ← שלב 2
              </button>
              <button type="button" onClick={() => setStep(4)} disabled={planningState.items.length === 0}
                className="btn btn-primary"
                style={{ flex: 1, cursor: planningState.items.length === 0 ? "not-allowed" : "pointer", opacity: planningState.items.length === 0 ? 0.5 : 1 }}>
                שלב 4 ←
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
            <div className="mb-4" style={{ border: "1px solid var(--s200)", borderRadius: 10, overflow: "hidden" }}>
              {Object.keys(planningState.boq).length === 0
                ? <p className="text-xs text-slate-500 p-3">אין נתוני BOQ להצגה.</p>
                : (
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: "var(--navy)", color: "#fff" }}>
                        <th style={{ textAlign: "right", padding: "8px 10px", fontWeight: 600, fontSize: 11 }}>סוג</th>
                        <th style={{ textAlign: "right", padding: "8px 10px", fontWeight: 600, fontSize: 11 }}>תת-סוג</th>
                        <th style={{ textAlign: "center", padding: "8px 6px", fontWeight: 600, fontSize: 11 }}>פריטים</th>
                        <th style={{ textAlign: "center", padding: "8px 6px", fontWeight: 600, fontSize: 11 }}>אורך</th>
                        <th style={{ textAlign: "center", padding: "8px 6px", fontWeight: 600, fontSize: 11 }}>שטח</th>
                        <th style={{ textAlign: "center", padding: "8px 6px", fontWeight: 600, fontSize: 11 }}>גובה/עובי</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(planningState.boq).map(([key, value], rowIdx) => {
                        const row = value as { type?: string; subtype?: string; count?: number; total_length_m?: number; total_area_m2?: number };
                        const color = getCategoryColor(row.type, row.subtype);
                        const catEntry = Object.values(planningState.categories).find(c => c.type === row.type && c.subtype === row.subtype);
                        const heightVal = catEntry?.params?.height_or_thickness;
                        return (
                          <tr key={key} style={{ background: rowIdx % 2 === 0 ? hexToRgba(color, 0.05) : "#fff", borderBottom: "1px solid var(--s100)" }}>
                            <td style={{ padding: "7px 10px", fontWeight: 700, color }}>{row.type ?? "—"}</td>
                            <td style={{ padding: "7px 10px", color: "var(--text-1)" }}>{row.subtype ?? "—"}</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)" }}>{row.count ?? 0}</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)", fontFamily: "monospace" }}>{(row.total_length_m ?? 0).toFixed(2)} מ&apos;</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)", fontFamily: "monospace" }}>{(row.total_area_m2 ?? 0).toFixed(2)} מ&quot;ר</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: heightVal ? "var(--navy)" : "var(--s400)", fontFamily: "monospace", fontWeight: heightVal ? 600 : 400 }}>
                              {heightVal != null ? `${heightVal} מ'` : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )
              }
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <button type="button" onClick={() => void handleFinalize()} disabled={loading}
                className={`btn ${loading ? "btn-ghost" : "btn-orange"}`}
                style={loading ? { cursor: "not-allowed", opacity: .7 } : {}}>
                {loading ? "שומר..." : "שמירה סופית"}
              </button>
              <button type="button" onClick={() => setStep(5)} className="btn btn-primary">גזרות עבודה ▶</button>
              <button type="button" onClick={() => setStep(3)} className="btn btn-ghost">← חזור לשלב 3</button>
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
                  style={{ touchAction: "none" }}
                  onMouseDown={handleSecMouseDown}
                  onMouseMove={handleSecMouseMove}
                  onMouseUp={handleSecMouseUp}
                  onMouseLeave={() => setSecDrawing(false)}
                  onTouchStart={makeTouchHandler(handleSecMouseDown)}
                  onTouchMove={makeTouchHandler(handleSecMouseMove)}
                  onTouchEnd={makeTouchHandler(handleSecMouseUp)}
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
                className="w-full px-3 py-2 rounded-lg bg-[var(--navy)] text-white text-xs font-semibold disabled:opacity-40 hover:bg-[#162d56]"
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
                style={{ padding: "11px 20px", borderRadius: 10, background: loading ? "#94a3b8" : "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer", boxShadow: loading ? "none" : "0 3px 12px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
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
