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
  confirmAutoSegments,
  confirmAutoSegmentsBatch,
  deleteAutoSegment,
  fetchBoqSummary,
  type AutoSegment,
  type AutoAnalyzeVisionData,
  type BoqSummary,
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
  "תקרה": ["גבס", "אקוסטית", "חשופה", "צבועה"],
  "דלתות וחלונות": ["דלת פנים", "דלת כניסה", "חלון אלומיניום", "חלון עץ", "ויטרינה"],
  "אינסטלציה": ["מים קרים", "מים חמים", "ביוב", "גז"],
  "חשמל": ["תאורה", "שקעים", "לוח חשמל", "גנרטור"],
  "טיח וצבע": ["טיח פנים", "טיח חוץ", "צבע פנים", "צבע חוץ"],
  "עמודים": ["עמוד בטון", "עמוד מתכת", "קורה"],
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

// Maps suggested_subtype (from backend) ג†’ Hebrew label + icon
function getFixGroupLabel(subtype: string): { label: string; icon: string } {
  const t = subtype.toLowerCase();
  if (t.includes("כיור") || t.includes("אסלה")) return { label: "כיורים ואסלות", icon: "🚰" };
  if (t.includes("אמבטיה") || t.includes("מקלחת")) return { label: "אמבטיות ומקלחות", icon: "🚿" };
  if (t.includes("ריהוט") || t.includes("מכשיר")) return { label: "ריהוט ומכשירים", icon: "🛋️" };
  if (t.includes("דלת")) return { label: "דלתות", icon: "🚪" };
  if (t.includes("חלון")) return { label: "חלונות", icon: "🪟" };
  if (t.includes("עמוד")) return { label: "עמודים", icon: "🏗️" };
  if (t.includes("מדרגות")) return { label: "מדרגות", icon: "🪜" };
  if (t.includes("מעלית")) return { label: "מעליות", icon: "🛗" };
  if (t.includes("קורה")) return { label: "קורות", icon: "🔩" };
  return { label: subtype || "אחר", icon: "📌" };
}

// ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€
// CategoryPickerModal ג€” choose or create a category for pending shapes
// ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€
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
          שיוך {pendingCount} {pendingCount === 1 ? "פריט" : "פריטים"} לקטגוריה
        </div>
        <p style={{ fontSize: 12, color: "#64748b", marginBottom: 16 }}>
          כל הפריטים שסומנו ישויכו לקטגוריה שתבחר.
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
                  <span style={{ fontWeight: 600, fontSize: 14, color: "#1e293b" }}>{cat.type} - {cat.subtype}</span>
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
              <div style={{ display: "flex", alignItems: "center", marginTop: 4 }}>
                <div style={{ width: 14, height: 14, borderRadius: "50%", background: getCategoryColor(newType, newSubtype), display: "inline-block", marginLeft: 6, flexShrink: 0 }}/>
                <select
                  style={{ flex: 1, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                  value={newType}
                  onChange={(e) => setNewType(e.target.value)}
                >
                  {Object.keys(CATEGORY_SUBTYPES).map(t => <option key={t}>{t}</option>)}
                </select>
              </div>
            </label>
            <label style={{ fontSize: 13 }}>
              תת-סוג
              <div style={{ display: "flex", alignItems: "center", marginTop: 4 }}>
                <div style={{ width: 14, height: 14, borderRadius: "50%", background: getCategoryColor(newType, newSubtype), display: "inline-block", marginLeft: 6, flexShrink: 0 }}/>
                <select
                  style={{ flex: 1, border: "1px solid #cbd5e1", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}
                  value={newSubtype}
                  onChange={(e) => setNewSubtype(e.target.value)}
                >
                  {subtypeOptions.map((sub) => <option key={sub}>{sub}</option>)}
                </select>
              </div>
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
              style={{ padding: "12px 0", borderRadius: 12, background: "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 15, cursor: "pointer" }}
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
          ביטול - המשך לצייר
        </button>
      </div>
    </div>
  );
};

// ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€
// ZoomModal ג€” fullscreen plan viewer with scroll-zoom + drag-pan + drawing
// Now supports pending shapes (unassigned, shown in amber)
// ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€ג"€
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

  // ג"€ג"€ Touch support for ZoomModal ג"€ג"€
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
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 14 }}>תצוגה מוגדלת - שלב 3</span>
        <span style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>Scroll לזום · Alt+גרור לתזוזה · לחץ לסימון</span>

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
            שיוך {pendingShapes.length} {pendingShapes.length === 1 ? "פריט" : "פריטים"} לקטגוריה
          </button>
        )}
        {pendingShapes.length === 0 && (
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", padding: "4px 8px" }}>ציור -&gt; שיוך לקטגוריה</span>
        )}

        {/* zoom controls */}
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button type="button" onClick={() => setZoom(z => Math.max(0.25, z * 0.8))} style={{ width: 28, height: 28, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 16 }}>גˆ’</button>
          <span style={{ color: "#fff", fontSize: 12, minWidth: 42, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => setZoom(z => Math.min(10, z * 1.25))} style={{ width: 28, height: 28, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 16 }}>+</button>
          <button type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} style={{ padding: "4px 8px", borderRadius: 6, border: "none", background: "rgba(255,255,255,0.15)", color: "#fff", cursor: "pointer", fontSize: 11 }}>איפוס</button>
        </div>

        <button type="button" onClick={onClose} style={{ padding: "5px 14px", borderRadius: 6, background: "#EF4444", border: "none", color: "#fff", fontWeight: 700, cursor: "pointer", fontSize: 13 }}>סגור</button>
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
              ממתין לשיוך ({pendingShapes.length}) - לחץ על הכפתור בסרגל למעלה לבחירת קטגוריה
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {pendingShapes.map((s) => (
                <div key={s.id} style={{ background: hexToRgba(PENDING_COLOR, 0.15), border: `1px solid ${hexToRgba(PENDING_COLOR, 0.4)}`, borderRadius: 6, padding: "2px 8px", fontSize: 11, color: "#fef3c7", display: "flex", gap: 5, alignItems: "center" }}>
                  <span>{s.object_type === "line" ? "קו" : s.object_type === "rect" ? "מלבן" : "חופשי"}</span>
                  <button type="button" onClick={() => onDeletePending(s.id)} style={{ background: "none", border: "none", color: "#F87171", cursor: "pointer", fontSize: 11, padding: 0 }}>מחק</button>
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
                <span>{cat?.subtype ?? item.category} | {item.type} | {(item.length_m_effective ?? item.length_m).toFixed(2)} מ'</span>
                <button type="button" onClick={() => void onDeleteItem(item.uid)} style={{ background: "none", border: "none", color: "#F87171", cursor: "pointer", fontSize: 11, padding: 0 }}>מחק</button>
              </div>
            );
          })}
          {planningState.items.length === 0 && <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>חלונות שסומנו נדרש שיוך?.</span>}
        </div>
      </div>
    </div>
  );
};

// ─── Step 3 toolbar components ───────────────────────────────────────────────

// Floating Toolbar Button
const ToolbarButton: React.FC<{
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}> = ({ active, onClick, label, icon }) => (
  <button
    type="button"
    onClick={onClick}
    title={label}
    style={{
      display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
      padding: "10px 14px", borderRadius: 8,
      background: active ? "#1e3a5f" : "transparent",
      border: "none",
      borderBottom: active ? "3px solid #e67e22" : "3px solid transparent",
      color: active ? "#fff" : "#475569",
      cursor: "pointer",
      fontSize: 12,
      fontWeight: 600,
      minWidth: 60,
      transition: "background 0.15s",
    }}
  >
    {icon}
    <span>{label}</span>
  </button>
);

const SelectIcon = () => (
  <svg width={22} height={22} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 3l14 9-7 1-4 7z"/>
  </svg>
);

const LineIcon = () => (
  <svg width={22} height={22} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
    <line x1="4" y1="20" x2="20" y2="4"/>
    <circle cx="4" cy="20" r="2" fill="currentColor"/>
    <circle cx="20" cy="4" r="2" fill="currentColor"/>
  </svg>
);

const RectIcon = () => (
  <svg width={22} height={22} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
  </svg>
);

const PathIcon = () => (
  <svg width={22} height={22} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 20 Q8 5 12 12 Q16 19 21 4"/>
  </svg>
);

interface ContextPopoverProps {
  item: { uid: string; type: string; category: string; length_m?: number; length_m_effective?: number };
  position: { x: number; y: number };
  categories: Record<string, { key: string; type: string; subtype: string }>;
  onClose: () => void;
  onDelete: () => void;
}

const ContextPopover: React.FC<ContextPopoverProps> = ({ item, position, categories, onClose, onDelete }) => {
  const cat = categories[item.category];
  return (
    <div
      style={{
        position: "absolute",
        left: position.x,
        top: position.y,
        transform: "translate(-50%, -100%)",
        zIndex: 30,
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        boxShadow: "0 8px 24px rgba(15,23,42,0.16)",
        padding: "12px 14px",
        minWidth: 200,
        direction: "rtl",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "#1e3a5f" }}>
          {cat ? `${cat.type} — ${cat.subtype}` : "ללא קטגוריה"}
        </span>
        <button type="button" onClick={onClose} style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: 16, lineHeight: 1, padding: 0 }}>×</button>
      </div>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10 }}>
        <span>{item.type === "line" ? "קו" : item.type === "rect" ? "מלבן" : "חופשי"}</span>
        {item.length_m_effective != null && (
          <span style={{ marginRight: 8 }}>{item.length_m_effective.toFixed(2)} מ'</span>
        )}
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button type="button" onClick={onClose} style={{ flex: 1, padding: "5px 0", borderRadius: 6, background: "#f1f5f9", border: "none", color: "#334155", fontSize: 11, cursor: "pointer" }}>סגור</button>
        <button type="button" onClick={onDelete} style={{ padding: "5px 10px", borderRadius: 6, background: "#fef2f2", border: "1px solid #fecaca", color: "#dc2626", fontSize: 11, cursor: "pointer" }}>מחק</button>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// PlanningPage — main wizard
// ─────────────────────────────────────────────────────────────────────────────
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

  // ג"€ג"€ Pending shapes (drawn but not yet assigned to a category) ג"€ג"€
  const [pendingShapes, setPendingShapes] = React.useState<PendingShape[]>([]);
  const [categoryPickerOpen, setCategoryPickerOpen] = React.useState(false);

  // ג"€ג"€ Category management state (used in step 3 side panel) ג"€ג"€
  const [categoriesDraft, setCategoriesDraft] = React.useState<Record<string, PlanningCategory>>({});
  const [newType, setNewType] = React.useState("קירות");
  const [newSubtype, setNewSubtype] = React.useState("בטון");
  const [newParamValue, setNewParamValue] = React.useState<number>(2.6);
  const [newParamNote, setNewParamNote] = React.useState("");
  const [addCatOpen, setAddCatOpen] = React.useState(false);

  // ג"€ג"€ Auto-analyze state ג"€ג"€
  const [autoSegments, setAutoSegments] = React.useState<AutoSegment[] | null>(null);
  const [autoVisionData, setAutoVisionData] = React.useState<AutoAnalyzeVisionData | null>(null);
  const [visionCatSuggestions, setVisionCatSuggestions] = React.useState<{ type: string; subtype: string; paramValue: number }[]>([]);
  const [visionActiveCard, setVisionActiveCard] = React.useState<string | null>(null);
  const [autoLoading, setAutoLoading] = React.useState(false);
  const [autoSelected, setAutoSelected] = React.useState<Set<string>>(new Set());
  const [selectedSegIds, setSelectedSegIds] = React.useState<Set<string>>(new Set());
  const [autoConfirmedKeys, setAutoConfirmedKeys] = React.useState<Record<string, string>>({}); // segIdג†’catKey
  const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(new Set(["walls"]));
  const [bulkOpen, setBulkOpen] = React.useState(false);
  const [bulkCatKeys, setBulkCatKeys] = React.useState<Record<string, string>>({}); // "type/subtype"ג†’catKey
  const [lastAddedUid, setLastAddedUid] = React.useState<string | null>(null);
  const [focusedUid, setFocusedUid] = React.useState<string | null>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement | null>(null);

  // ג"€ג"€ Auto-approve UX state ג"€ג"€
  const [autoApproveThreshold, setAutoApproveThreshold] = React.useState(90); // percentage 70-100
  const [autoApproveToast, setAutoApproveToast] = React.useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = React.useState<string | null>(null);
  const [showPendingOnly, setShowPendingOnly] = React.useState(true);
  const [contextMenu, setContextMenu] = React.useState<{ x: number; y: number; segId: string } | null>(null);
  const [popoverType, setPopoverType] = React.useState<string>("");
  const [popoverSubtype, setPopoverSubtype] = React.useState<string>("");
  const segmentListRefs = React.useRef<Record<string, HTMLDivElement | null>>({});
  const segmentPanelRef = React.useRef<HTMLDivElement | null>(null);
  const [bulkSmallCat, setBulkSmallCat] = React.useState<string>(""); // for small items bulk categorize
  const [approveAllMsg, setApproveAllMsg] = React.useState<string | null>(null);
  const [reviewMode, setReviewMode] = React.useState(false);
  const [reviewQueue, setReviewQueue] = React.useState<string[]>([]);
  const [reviewIndex, setReviewIndex] = React.useState(0);
  // ג"€ג"€ Category highlight mode ג"€ג"€
  const [highlightedClass, setHighlightedClass] = React.useState<string | null>(null);
  const [highlightedType, setHighlightedType] = React.useState<string | null>(null);
  const [boqData, setBoqData] = React.useState<BoqSummary | null>(null);
  const [boqLoading, setBoqLoading] = React.useState(false);
  const [boqVisible, setBoqVisible] = React.useState(false);
  const [confFilter, setConfFilter] = React.useState<"all" | "high" | "low">("all");
  const [hoveredSegId, setHoveredSegId] = React.useState<string | null>(null);

  // Step 3 select mode
  const [selectMode, setSelectMode] = React.useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [selectedPopoverItem, setSelectedPopoverItem] = React.useState<any>(null);
  const [popoverPosition, setPopoverPosition] = React.useState({ x: 0, y: 0 });

  // ג"€ג"€ Zone state ג"€ג"€
  const [zoneDrawing, setZoneDrawing] = React.useState(false);
  const [zoneStart, setZoneStart] = React.useState<Point | null>(null);
  const [zoneEnd, setZoneEnd] = React.useState<Point | null>(null);
  const [zoneTemp, setZoneTemp] = React.useState<Point | null>(null);
  const [zoneCatKey, setZoneCatKey] = React.useState<string>("");
  const zoneCanvasRef = React.useRef<SVGSVGElement | null>(null);

  // ג"€ג"€ Text items state ג"€ג"€
  const [textRows, setTextRows] = React.useState<TextItemPayload[]>([
    { category_key: "__manual__", description: "", quantity: 1, unit: "יח'", note: "" }
  ]);

  // ג"€ג"€ Step 5 ג€” Sections (׳’׳–׳¨׳•׳× ׳¢׳‘׳•׳”׳”) ג"€ג"€
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

  // ג"€ג"€ Drawing state (main canvas) ג"€ג"€
  const [drawMode, setDrawMode] = React.useState<DrawMode>("line");
  const [drawing, setDrawing] = React.useState(false);
  const [startPoint, setStartPoint] = React.useState<Point | null>(null);
  const [tempPoint, setTempPoint] = React.useState<Point | null>(null);
  const [pathPoints, setPathPoints] = React.useState<Point[]>([]);

  // ג"€ג"€ Opening / wall confirmation prompts ג"€ג"€
  const [openingPrompt, setOpeningPrompt] = React.useState<{ itemUid: string; gapId?: string; gapLengthM?: number } | null>(null);
  const [wallPrompt, setWallPrompt] = React.useState<{ itemUid: string; overlapRatio?: number } | null>(null);

  // ג"€ג"€ Calibration state ג"€ג"€
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
  const [zoomPercent, setZoomPercent] = React.useState(200);

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

  // ג"€ג"€ Load plans ג"€ג"€
  const loadPlans = React.useCallback(async () => {
    const data = await listWorkshopPlans();
    setPlans(data);
    if (!selectedPlanId && data.length > 0) setSelectedPlanId(data[0].id);
  }, [selectedPlanId]);

  const loadPlanningState = React.useCallback(async (planId: string) => {
    const state = await getPlanningState(planId);
    setPlanningState(state);
    setCategoriesDraft(state.categories);
    // ׳©׳—׳–׳¨ ׳×׳•׳¦׳׳•׳× ׳ ׳™׳×׳•׳— ׳׳•׳˜׳•׳׳˜׳™ ׳©׳ ׳©׳׳¨׳• ׳‘׳”׳׳˜׳”׳‘׳™׳™׳¡
    if (state.auto_segments && state.auto_segments.length > 0 && autoSegments === null) {
      setAutoSegments(state.auto_segments);
      setAutoSelected(new Set(
        state.auto_segments
          .filter(s => s.suggested_subtype !== "׳₪׳¨׳˜ ׳§׳˜׳")
          .map(s => s.segment_id)
      ));
    }
  }, [autoSegments]);

  React.useEffect(() => {
    void loadPlans().catch(() => setError("שגיאה בטעינת תוכניות."));
  }, [loadPlans]);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    setZoomPercent(200);
    setLoading(true);
    void loadPlanningState(selectedPlanId)
      .catch(() => setError("שגיאה בטעינת נתוני הגדרת התכולה."))
      .finally(() => setLoading(false));
  }, [selectedPlanId, loadPlanningState]);

  const selectedPlan = plans.find((p) => p.id === selectedPlanId) ?? null;

  const updateDisplaySizeFromImage = (img: HTMLImageElement | null) => {
    if (!img) return;
    const naturalW = img.naturalWidth || planningState?.image_width || 1;
    const naturalH = img.naturalHeight || planningState?.image_height || 1;
    const containerW = canvasContainerRef.current?.clientWidth  || 920;
    const containerH = canvasContainerRef.current?.clientHeight || 600;
    const usableW = Math.max(400, containerW - 32);
    const scaleByWidth = usableW / naturalW;
    const scaleByHeight = (containerH * 1.8) / naturalH; // allow 1.8x container height before capping
    const scale = Math.max(0.15, Math.min(scaleByWidth, scaleByHeight));
    setBaseDisplaySize({
      width:  Math.max(1, Math.round(naturalW * scale)),
      height: Math.max(1, Math.round(naturalH * scale)),
    });
  };

  const toLocalPoint = (targetRef: React.RefObject<HTMLElement | null>, clientX: number, clientY: number): Point => {
    const rect = targetRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: Math.max(0, Math.min(displaySize.width, clientX - rect.left)), y: Math.max(0, Math.min(displaySize.height, clientY - rect.top)) };
  };

  // ג"€ג"€ Touch support: wraps a mouse handler to accept touch events ג"€ג"€
  const makeTouchHandler = <T extends SVGSVGElement>(
    handler: React.MouseEventHandler<T>
  ): React.TouchEventHandler<T> =>
    (e: React.TouchEvent<T>) => {
      e.preventDefault();
      const touch = e.touches[0] ?? e.changedTouches[0];
      if (!touch) return;
      handler({ clientX: touch.clientX, clientY: touch.clientY, currentTarget: e.currentTarget, preventDefault: () => {} } as unknown as React.MouseEvent<T>);
    };

  // ג"€ג"€ Category helpers ג"€ג"€
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

  // ג"€ג"€ Assign pending shapes to a category ג"€ג"€
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
        setError(`׳ ׳©׳׳¨׳• ${saved} ׳׳×׳•׳ ${pendingShapes.length} ׳₪׳¨׳™׳˜׳™׳. ${failures.length} ׳ ׳›׳©׳׳• ג€” ׳ ׳¡׳” ׳©׳•׳‘.`);
      } else {
        setError(`׳©׳’׳™׳׳” ׳‘׳©׳™׳•׳ ׳₪׳¨׳™׳˜׳™׳: ${failures[0]}`);
      }
    } finally { setLoading(false); }
  };

  // ג"€ג"€ Create new category then assign ג"€ג"€
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

  // ג"€ג"€ Canvas: main page drawing (adds to pending) ג"€ג"€
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
    // Store as pending with display_scale applied (raw canvas coords ג†’ natural coords)
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

  // ג"€ג"€ Delete handlers ג"€ג"€
  const handleDeleteItem = async (uid: string) => {
    if (!selectedPlanId) return;
    try {
      const state = await deletePlanningItem(selectedPlanId, uid);
      setPlanningState(state);
    } catch { setError("שגיאה במחיקת פריט."); }
  };

  const handleBatchAssign = async (categoryKey: string) => {
    if (!selectedPlanId || selectedSegIds.size === 0) return;
    setCategoryPickerOpen(false);
    setLoading(true);
    try {
      const ids = [...selectedSegIds];
      const state = await confirmAutoSegmentsBatch(selectedPlanId, ids, categoryKey);
      setPlanningState(state);
      setAutoSegments(prev => prev ? prev.filter(s => !ids.includes(s.segment_id)) : prev);
      setSelectedSegIds(new Set());
    } catch { setError("שגיאה בשיוך מרובה."); }
    finally { setLoading(false); }
  };

  const startReviewMode = () => {
    if (!planningState || planningState.items.length === 0) return;
    setReviewQueue(planningState.items.map((i) => i.uid));
    setReviewIndex(0);
    setReviewMode(true);
    setSelectMode(false);
    setSelectedPopoverItem(null);
  };

  const stopReviewMode = () => {
    setReviewMode(false);
    setReviewQueue([]);
    setReviewIndex(0);
  };

  const reviewAdvance = (updatedQueue?: string[]) => {
    const q = updatedQueue ?? reviewQueue;
    if (reviewIndex + 1 >= q.length) {
      stopReviewMode();
    } else {
      setReviewIndex((i) => i + 1);
    }
  };

  const reviewDeleteCurrent = async () => {
    const uid = reviewQueue[reviewIndex];
    if (!uid) return;
    await handleDeleteItem(uid);
    const newQueue = reviewQueue.filter((id) => id !== uid);
    setReviewQueue(newQueue);
    if (reviewIndex >= newQueue.length) {
      stopReviewMode();
    }
  };

  // ג"€ג"€ Calibration ג"€ג"€
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

  // ג"€ג"€ Opening / Wall prompts ג"€ג"€
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

  // ג"€ג"€ Finalize ג"€ג"€
  const handleFinalize = async () => {
    if (!selectedPlanId) return;
    setLoading(true);
    try {
      const state = await finalizePlanning(selectedPlanId);
      setPlanningState(state); setError("");
      setFinalizeNotice(`נשמר בהצלחה: ${new Date().toLocaleTimeString("he-IL")}`);
    } catch { setError("שגיאה בשמירה סופית של התכולה."); } finally { setLoading(false); }
  };

  // ג"€ג"€ Section handlers ג"€ג"€
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

  // ג"€ג"€ Section canvas mouse handlers ג"€ג"€
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

  // ג"€ג"€ Vision material ג†’ category matcher ג"€ג"€
  // -- Vision material -> category matcher --
  const matchMaterialToCategory = (material: string): { type: string; subtype: string } | null => {
    const m = material.toLowerCase();
    if (m.includes("\u05d1\u05d8\u05d5\u05df \u05de\u05d5\u05d7\u05dc\u05e7")) return { type: "\u05e8\u05d9\u05e6\u05d5\u05e3", subtype: "\u05d1\u05d8\u05d5\u05df \u05de\u05d5\u05d7\u05dc\u05e7" };
    if (m.includes("\u05d1\u05d8\u05d5\u05df")) return { type: "\u05e7\u05d9\u05e8\u05d5\u05ea", subtype: "\u05d1\u05d8\u05d5\u05df" };
    if (m.includes("\u05d1\u05dc\u05d5\u05e7\u05d9\u05dd")) return { type: "\u05e7\u05d9\u05e8\u05d5\u05ea", subtype: "\u05d1\u05dc\u05d5\u05e7\u05d9\u05dd" };
    if (m.includes("\u05de\u05d7\u05d9\u05e6\u05d4 \u05e7\u05dc\u05d4")) return { type: "\u05e7\u05d9\u05e8\u05d5\u05ea", subtype: "\u05de\u05d7\u05d9\u05e6\u05d4 \u05e7\u05dc\u05d4" };
    if (m.includes("\u05d2\u05e8\u05e0\u05d9\u05d8") || m.includes("\u05e4\u05d5\u05e8\u05e6\u05dc\u05df")) return { type: "\u05e8\u05d9\u05e6\u05d5\u05e3", subtype: "\u05d2\u05e8\u05e0\u05d9\u05d8 \u05e4\u05d5\u05e8\u05e6\u05dc\u05df" };
    if (m.includes("\u05e7\u05e8\u05de\u05d9\u05e7\u05d4")) return { type: "\u05e8\u05d9\u05e6\u05d5\u05e3", subtype: "\u05e7\u05e8\u05de\u05d9\u05e7\u05d4" };
    if (m.includes("\u05e4\u05e8\u05e7\u05d8")) return { type: "\u05e8\u05d9\u05e6\u05d5\u05e3", subtype: "\u05e4\u05e8\u05e7\u05d8" };
    if (m.includes("\u05ea\u05e7\u05e8\u05d4") && m.includes("\u05d2\u05d1\u05e1")) return { type: "\u05ea\u05e7\u05e8\u05d4", subtype: "\u05d2\u05d1\u05e1" };
    if (m.includes("\u05d2\u05d1\u05e1") && (m.includes("\u05de\u05d7\u05d9\u05e6\u05d4") || m.includes("\u05e7\u05d9\u05e8"))) return { type: "\u05e7\u05d9\u05e8\u05d5\u05ea", subtype: "\u05d2\u05d1\u05e1" };
    if (m.includes("\u05d2\u05d1\u05e1")) return { type: "\u05e7\u05d9\u05e8\u05d5\u05ea", subtype: "\u05d2\u05d1\u05e1" };
    if (m.includes("\u05d8\u05d9\u05d7") && m.includes("\u05d7\u05d5\u05e5")) return { type: "\u05d8\u05d9\u05d7 \u05d5\u05e6\u05d1\u05e2", subtype: "\u05d8\u05d9\u05d7 \u05d7\u05d5\u05e5" };
    if (m.includes("\u05d8\u05d9\u05d7") && m.includes("\u05e4\u05e0\u05d9\u05dd")) return { type: "\u05d8\u05d9\u05d7 \u05d5\u05e6\u05d1\u05e2", subtype: "\u05d8\u05d9\u05d7 \u05e4\u05e0\u05d9\u05dd" };
    if (m.includes("\u05e6\u05d1\u05e2") && m.includes("\u05d7\u05d5\u05e5")) return { type: "\u05d8\u05d9\u05d7 \u05d5\u05e6\u05d1\u05e2", subtype: "\u05e6\u05d1\u05e2 \u05d7\u05d5\u05e5" };
    if (m.includes("\u05e6\u05d1\u05e2") && m.includes("\u05e4\u05e0\u05d9\u05dd")) return { type: "\u05d8\u05d9\u05d7 \u05d5\u05e6\u05d1\u05e2", subtype: "\u05e6\u05d1\u05e2 \u05e4\u05e0\u05d9\u05dd" };
    if (m.includes("\u05d3\u05dc\u05ea") && m.includes("\u05db\u05e0\u05d9\u05e1\u05d4")) return { type: "\u05d3\u05dc\u05ea\u05d5\u05ea \u05d5\u05d7\u05dc\u05d5\u05e0\u05d5\u05ea", subtype: "\u05d3\u05dc\u05ea \u05db\u05e0\u05d9\u05e1\u05d4" };
    if (m.includes("\u05d3\u05dc\u05ea")) return { type: "\u05d3\u05dc\u05ea\u05d5\u05ea \u05d5\u05d7\u05dc\u05d5\u05e0\u05d5\u05ea", subtype: "\u05d3\u05dc\u05ea \u05e4\u05e0\u05d9\u05dd" };
    if (m.includes("\u05d7\u05dc\u05d5\u05df") && m.includes("\u05e2\u05e5")) return { type: "\u05d3\u05dc\u05ea\u05d5\u05ea \u05d5\u05d7\u05dc\u05d5\u05e0\u05d5\u05ea", subtype: "\u05d7\u05dc\u05d5\u05df \u05e2\u05e5" };
    if (m.includes("\u05d7\u05dc\u05d5\u05df")) return { type: "\u05d3\u05dc\u05ea\u05d5\u05ea \u05d5\u05d7\u05dc\u05d5\u05e0\u05d5\u05ea", subtype: "\u05d7\u05dc\u05d5\u05df \u05d0\u05dc\u05d5\u05de\u05d9\u05e0\u05d9\u05d5\u05dd" };
    if (m.includes("\u05d5\u05d9\u05d8\u05e8\u05d9\u05e0\u05d4")) return { type: "\u05d3\u05dc\u05ea\u05d5\u05ea \u05d5\u05d7\u05dc\u05d5\u05e0\u05d5\u05ea", subtype: "\u05d5\u05d9\u05d8\u05e8\u05d9\u05e0\u05d4" };
    if (m.includes("\u05e2\u05de\u05d5\u05d3") && m.includes("\u05de\u05ea\u05db\u05ea")) return { type: "\u05e2\u05de\u05d5\u05d3\u05d9\u05dd", subtype: "\u05e2\u05de\u05d5\u05d3 \u05de\u05ea\u05db\u05ea" };
    if (m.includes("\u05e2\u05de\u05d5\u05d3")) return { type: "\u05e2\u05de\u05d5\u05d3\u05d9\u05dd", subtype: "\u05e2\u05de\u05d5\u05d3 \u05d1\u05d8\u05d5\u05df" };
    if (m.includes("\u05e7\u05d5\u05e8\u05d4")) return { type: "\u05e2\u05de\u05d5\u05d3\u05d9\u05dd", subtype: "\u05e7\u05d5\u05e8\u05d4" };
    if (m.includes("\u05d0\u05e7\u05d5\u05e1\u05d8\u05d9")) return { type: "\u05ea\u05e7\u05e8\u05d4", subtype: "\u05d0\u05e7\u05d5\u05e1\u05d8\u05d9\u05ea" };
    return null;
  };

  // ג"€ג"€ Auto-create categories from Vision materials ג"€ג"€
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

  // ג"€ג"€ Auto-analyze handlers ג"€ג"€
  const handleAutoAnalyze = async () => {
    if (!selectedPlanId) return;
    setAutoLoading(true);
    try {
      const result = await autoAnalyzePlan(selectedPlanId);
      setAutoSegments(result.segments);
      setAutoVisionData(result.vision_data ?? null);
      setVisionActiveCard(null);
      // Pre-select all except unidentified small fixtures ("׳₪׳¨׳˜ ׳§׳˜׳")
      setAutoSelected(new Set(
        result.segments.filter(s => s.suggested_subtype !== "׳₪׳¨׳˜ ׳§׳˜׳").map(s => s.segment_id)
      ));
      // Build category suggestions ג€” primary: from detected segments; secondary: Vision materials
      {
        const seen = new Set<string>();
        const suggestions: { type: string; subtype: string; paramValue: number }[] = [];

        // Primary: unique (type, subtype) pairs from wall segments ג€” guaranteed valid
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
                suggestions.push({ ...match, paramValue: match.type === "׳¨׳™׳¦׳•׳£" ? 0.012 : 2.6 });
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
      setError(`׳©׳’׳™׳׳” ׳‘׳ ׳™׳×׳•׳— ׳׳•׳˜׳•׳׳˜׳™: ${detail}`);
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
      setError(`׳©׳’׳™׳׳” ׳‘׳׳™׳©׳•׳¨ ׳׳–׳•׳¨׳™׳: ${detail}`);
    } finally { setLoading(false); }
  };

  // ג"€ג"€ Auto-approve by confidence threshold ג"€ג"€
  const handleAutoApproveByThreshold = async () => {
    if (!selectedPlanId || !autoSegments || !planningState) return;
    const thresh = autoApproveThreshold / 100;
    const toApprove = autoSegments.filter(s =>
      s.confidence >= thresh &&
      !autoConfirmedKeys[s.segment_id] && // not already confirmed
      s.element_class !== "fixture"
    );
    if (toApprove.length === 0) {
      setAutoApproveToast("אין פריטים מעל הסף שעדיין לא אושרו");
      setTimeout(() => setAutoApproveToast(null), 3000);
      return;
    }
    // Find or create matching category keys
    const newConfirmedKeys: Record<string, string> = { ...autoConfirmedKeys };
    for (const seg of toApprove) {
      const match = Object.values(planningState.categories).find(
        c => c.type === seg.suggested_type && c.subtype === seg.suggested_subtype
      );
      if (match) {
        newConfirmedKeys[seg.segment_id] = match.key;
      }
    }
    // Only confirm those with a category key found
    const confirmable = toApprove.filter(s => newConfirmedKeys[s.segment_id]);
    if (confirmable.length === 0) {
      setAutoApproveToast("לא נמצאו קטגוריות תואמות - צור קטגוריות קודם");
      setTimeout(() => setAutoApproveToast(null), 3000);
      return;
    }
    setLoading(true);
    try {
      let lastState = planningState;
      for (const seg of confirmable) {
        const catKey = newConfirmedKeys[seg.segment_id];
        const { data } = await apiClient.post<PlanningState>(
          `/manager/planning/${encodeURIComponent(selectedPlanId)}/confirm-auto-segment`,
          { segment_id: seg.segment_id, category_key: catKey, bbox: seg.bbox }
        );
        lastState = data;
      }
      setPlanningState(lastState);
      // Remove confirmed segments from autoSegments
      setAutoSegments(prev => prev ? prev.filter(s => !newConfirmedKeys[s.segment_id] || !confirmable.find(c => c.segment_id === s.segment_id)) : null);
      setAutoConfirmedKeys(newConfirmedKeys);
      setAutoApproveToast(`ג… ׳׳•׳©׳¨׳• ${confirmable.length} ׳₪׳¨׳™׳˜׳™׳ ׳׳•׳˜׳•׳׳˜׳™׳×`);
      setTimeout(() => setAutoApproveToast(null), 4000);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳׳™׳©׳•׳¨ ׳׳•׳˜׳•׳׳˜׳™: ${detail}`);
    } finally { setLoading(false); }
  };

  const handleLoadBoq = async () => {
    if (!selectedPlanId) return;
    setBoqLoading(true);
    try {
      const summary = await fetchBoqSummary(selectedPlanId);
      setBoqData(summary);
      setBoqVisible(true);
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳˜׳¢׳™׳ ׳× ׳›׳×׳‘ ׳›׳׳•׳™׳•׳×: ${detail}`);
    } finally {
      setBoqLoading(false);
    }
  };

  const handleDeleteSegment = async (segId: string) => {
    if (!selectedPlanId) return;
    try {
      await deleteAutoSegment(selectedPlanId, segId);
      setAutoSegments(prev => prev ? prev.filter(s => s.segment_id !== segId) : null);
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳׳—׳™׳§׳× ׳¡׳’׳׳ ׳˜: ${detail}`);
    }
  };

  // ג"€ג"€ Confirm single segment from context menu ג"€ג"€
  const handleConfirmSingleSegment = async (segId: string, categoryKey: string) => {
    if (!selectedPlanId || !autoSegments || !planningState) return;
    const seg = autoSegments.find(s => s.segment_id === segId);
    if (!seg || !categoryKey) return;
    setContextMenu(null);
    setLoading(true);
    try {
      const { data } = await apiClient.post<PlanningState>(
        `/manager/planning/${encodeURIComponent(selectedPlanId)}/confirm-auto-segment`,
        { segment_id: seg.segment_id, category_key: categoryKey, bbox: seg.bbox }
      );
      setPlanningState(data);
      setAutoSegments(prev => prev ? prev.filter(s => s.segment_id !== segId) : null);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳׳™׳©׳•׳¨: ${detail}`);
    } finally { setLoading(false); }
  };

  // ג"€ג"€ Bulk confirm a group of segments (for small items) ג"€ג"€
  const handleBulkConfirmGroup = async (segIds: string[], categoryKey: string) => {
    if (!selectedPlanId || !planningState || !autoSegments) return;
    const segs = autoSegments.filter(s => segIds.includes(s.segment_id));
    if (segs.length === 0 || !categoryKey) return;
    setLoading(true);
    try {
      let lastState = planningState;
      for (const seg of segs) {
        const { data } = await apiClient.post<PlanningState>(
          `/manager/planning/${encodeURIComponent(selectedPlanId)}/confirm-auto-segment`,
          { segment_id: seg.segment_id, category_key: categoryKey, bbox: seg.bbox }
        );
        lastState = data;
      }
      setPlanningState(lastState);
      setAutoSegments(prev => prev ? prev.filter(s => !segIds.includes(s.segment_id)) : null);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳©׳™׳•׳ ׳§׳‘׳•׳¦׳”: ${detail}`);
    } finally { setLoading(false); }
  };

  // ג"€ג"€ Zone handlers ג"€ג"€
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
      setError(`׳©׳’׳™׳׳” ׳‘׳”׳•׳¡׳₪׳× ׳׳–׳•׳¨: ${detail}`);
    } finally { setLoading(false); }
  };

  // ג"€ג"€ Text item handlers ג"€ג"€
  const handleAddTextRow = () => {
    setTextRows(prev => [...prev, { category_key: "__manual__", description: "", quantity: 1, unit: "יח'", note: "" }]);
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
      setTextRows([{ category_key: "__manual__", description: "", quantity: 1, unit: "יח'", note: "" }]);
      setError("");
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data?.detail as string | undefined) || e.message : String(e);
      setError(`׳©׳’׳™׳׳” ׳‘׳©׳׳™׳¨׳× ׳₪׳¨׳™׳˜׳™ ׳˜׳§׳¡׳˜: ${detail}`);
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

  // ג"€ג"€ Focus / zoom to an item on canvas ג"€ג"€
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

  const handleSelectItem = (uid: string, e: React.MouseEvent) => {
    if (!selectMode || !planningState) return;
    e.stopPropagation();
    const item = planningState.items.find(i => i.uid === uid);
    if (!item) return;
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const containerRect = canvasContainerRef.current?.getBoundingClientRect();
    if (!containerRect) return;
    setSelectedPopoverItem(item);
    setPopoverPosition({
      x: rect.left + rect.width / 2 - containerRect.left,
      y: rect.top - containerRect.top - 10,
    });
  };

  // ג"€ג"€ Real-time BOQ preview (per category totals) ג"€ג"€
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

  // ג"€ג"€ Line Snap constants ג"€ג"€
  const SNAP_ANGLE_DEG = 8;   // degrees ג€” snap to nearest 45ֲ° if within this tolerance
  const SNAP_ENDPOINT_PX = 14; // pixels ג€” snap to existing endpoint if within this distance

  // Snap a candidate end-point toward 45ֲ°/90ֲ° angles or existing endpoints
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
    // Angle snap: constrain to multiples of 45ֲ° from anchor
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
            <span style={{ fontSize: 18 }}>✓</span>
            <span className="text-sm font-bold" style={{ color: "#92400E" }}>x ? x? x</span>
          </div>
          <div className="text-sm mb-3" style={{ color: "#B45309" }}>
            ? חלונות שסומנו ?? <strong>{openingPrompt.gapLengthM?.toFixed(2)} מ'</strong> - ? x? x, ?? x ?? x
          </div>
          <div className="flex gap-2 flex-wrap">
            <button type="button" onClick={() => void handleResolveOpening("door")} style={{ padding: "8px 18px", borderRadius: 9, background: "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(255,75,75,0.3)" }}>?</button>
            <button type="button" onClick={() => void handleResolveOpening("window")} style={{ padding: "8px 18px", borderRadius: 9, background: "#F97316", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(249,115,22,0.3)" }}>x</button>
            <button type="button" onClick={() => void handleResolveOpening("none")} style={{ padding: "8px 16px", borderRadius: 9, background: "#fff", color: "#64748b", border: "1px solid #CBD5E1", fontWeight: 600, fontSize: 13, cursor: "pointer" }}>x ?</button>
          </div>
        </div>
      )}

      {/* Wall confirmation prompt */}
      {wallPrompt && (
        <div className="rounded-lg p-4" style={{ background: "#EFF6FF", border: "1px solid #93C5FD", borderRight: "5px solid #3B82F6", boxShadow: "0 2px 8px rgba(59,130,246,0.1)" }}>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 18 }}>✓</span>
            <span className="text-sm font-bold" style={{ color: "#1E3A8A" }}>סך שטח ?? ?</span>
          </div>
          <div className="text-sm mb-3" style={{ color: "#1D4ED8" }}>
            ?? סך שטח ? ? (x?: <strong>{wallPrompt.overlapRatio != null ? `${Math.round(wallPrompt.overlapRatio * 100)}%` : "x x"}</strong>). ? x? ?? סך שטח
          </div>
          <div className="flex gap-2 flex-wrap">
            <button type="button" onClick={() => void handleResolveWall(true)} style={{ padding: "8px 18px", borderRadius: 9, background: "var(--navy)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", boxShadow: "0 2px 6px rgba(27,58,107,0.3)" }}>x, ?? ?</button>
            <button type="button" onClick={() => void handleResolveWall(false)} style={{ padding: "8px 16px", borderRadius: 9, background: "#fff", color: "#64748b", border: "1px solid #CBD5E1", fontWeight: 600, fontSize: 13, cursor: "pointer" }}>x?</button>
          </div>
        </div>
      )}

      {/* Loading overlay for plan load */}
      {loading && !planningState && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", background: "#F0F9FF", border: "1px solid #BAE6FD", borderRadius: 10, fontSize: 13, color: "#0369A1" }}>
          <span style={{ animation: "spin 1s linear infinite", display: "inline-block", fontSize: 16 }}>x</span>
          <span>טוען נתוני תכנון...</span>
        </div>
      )}

      {/* Step nav ג€” horizontal stepper strip */}
      <div style={{ background: "#fff", borderBottom: "1px solid var(--s200)", padding: "16px 20px", flexShrink: 0, margin: "0 -24px", paddingLeft: 28, paddingRight: 28 }}>
        <div className="wizard-steps">
          {(
            [
                            { s: 1 as WizardStep, label: "בחירת\nתוכנית",  canGo: true },
              { s: 2 as WizardStep, label: "כיול\nסקייל",    canGo: canStep2 },
              { s: 3 as WizardStep, label: "הגדרת\nתכולה",   canGo: canStep3 },
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

      {/* ג"€ג"€ STEP 1: Pick plan ג"€ג"€ */}
      {step === 1 && (
        <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-5 space-y-4">
          <div>
            <p className="text-base font-bold text-[var(--navy)] mb-1">בחר תוכנית לעבודה</p>
            <p className="text-xs text-slate-400">בחר מהרשימה תוכנית שהועלתה בסדנת עבודה.</p>
          </div>
          {plans.length === 0 ? (
            <div className="rounded-lg p-4 text-sm text-amber-800" style={{ background: "#FFFBEB", border: "1px solid #FCD34D" }}>
              אין תוכניות זמינות. העלה קודם תוכנית ב"סדנת עבודה".
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
              <span style={{ fontSize: 28 }}>תוכנית</span>
              <div className="text-sm">
                <div className="font-bold text-[#1E3A8A]">{selectedPlan.plan_name}</div>
                <div className="text-xs text-blue-600 mt-0.5">
                  {selectedPlan.total_wall_length_m != null && <span>אורך קירות: <strong>{selectedPlan.total_wall_length_m.toFixed(1)} מ'</strong></span>}
                  {selectedPlan.concrete_length_m != null && <span className="mr-3">בטון: {selectedPlan.concrete_length_m.toFixed(1)} מ'</span>}
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-end">
            <button type="button" onClick={() => setStep(2)} disabled={!selectedPlanId}
              style={{ height: 48, padding: "0 28px", borderRadius: 10, background: selectedPlanId ? "var(--navy)" : "var(--s300)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: selectedPlanId ? "pointer" : "not-allowed", transition: "all 0.15s" }}>
              המשך לשלב 2
            </button>
          </div>
        </div>
      )}

      {/* ג"€ג"€ STEP 2: Calibration only ג"€ג"€ */}
      {step === 2 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr,300px] gap-4">
          {/* Calibration canvas */}
          <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
            <p className="text-sm font-semibold text-[#31333F] mb-1">שלב 2: כיול סקייל</p>
            {planningState.scale_px_per_meter > 0 && planningState.scale_px_per_meter !== 200 && (
              <div style={{ background: "#F0FDF4", border: "1px solid #86EFAC", borderRight: "4px solid #22C55E", borderRadius: 10, padding: "10px 14px", marginBottom: 12, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>✓</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: "#15803D" }}>כיול קיים</div>
                  <div style={{ fontSize: 11, color: "#166534" }}>סקייל: {planningState.scale_px_per_meter.toFixed(1)} px/m - ניתן לדלג לשלב 3</div>
                </div>
                <button type="button" onClick={() => setStep(3)}
                  style={{ padding: "7px 16px", borderRadius: 9, background: "#15803D", color: "#fff", border: "none", fontWeight: 700, fontSize: 12, cursor: "pointer" }}>
                  דלג לשלב 3
                </button>
              </div>
            )}
            <p className="text-xs text-slate-500 mb-3">גרור קו על אורך ידוע בתוכנית, הזן את האורך האמיתי ולחץ "עדכן סקייל".</p>
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
                עדכן סקייל
              </button>
              <button type="button" onClick={() => { setCalStart(null); setCalEnd(null); setCalTemp(null); setCalDrawing(false); }}
                className="btn btn-ghost btn-full" style={{ marginTop: 6 }}>
                נקה קו
              </button>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800 space-y-1">
              <p className="font-semibold">טיפ</p>
              <p>גרור קו על קיר שאורכו ידוע (למשל 5 מטר). הכיול יחושב אוטומטית.</p>
              <p>אחרי כיול מדויק, תוצאות המדידה יהיו מדויקות יותר.</p>
            </div>
          </div>

          <div className="xl:col-span-2" style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
            <button type="button" onClick={() => setStep(1)} className="btn btn-ghost">חזור לשלב 1</button>
            <button type="button" onClick={() => setStep(3)} className="btn btn-orange">המשך לשלב 3</button>
          </div>
        </div>
      )}

      {/* ג"€ג"€ ZoomModal ג"€ג"€ */}
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

      {/* ג"€ג"€ Category Picker Modal ג"€ג"€ */}
      {categoryPickerOpen && planningState && (
        <CategoryPickerModal
          categories={planningState.categories}
          pendingCount={pendingShapes.length}
          onPick={(key) => { if (selectedSegIds.size > 0) { void handleBatchAssign(key); } else { void handleAssignCategory(key); } }}
          onCreateAndPick={(type, subtype, param, note) => { void handleCreateAndAssign(type, subtype, param, note); }}
          onCancel={() => setCategoryPickerOpen(false)}
        />
      )}

      {/* STEP 3: Full-screen redesign */}
      {step === 3 && planningState && (
        <div style={{
          position: "fixed",
          inset: 0,
          zIndex: 100,
          display: "flex",
          flexDirection: "column",
          background: "#eef2f7",
          direction: "rtl",
        }}>
          {/* Slim header bar */}
          <div style={{
            height: 48,
            background: "#1e3a5f",
            display: "flex",
            alignItems: "center",
            padding: "0 16px",
            gap: 12,
            flexShrink: 0,
            borderBottom: "1px solid rgba(255,255,255,0.1)",
          }}>
            <span style={{ color: "rgba(255,255,255,0.6)", fontSize: 11 }}>שלב 3 / 5 — הגדרת תכולה</span>
            <span style={{ color: "#fff", fontWeight: 600, fontSize: 13 }}>{selectedPlan?.plan_name ?? ""}</span>
            <div style={{ flex: 1 }} />
            {/* Zoom controls */}
            <button type="button" onClick={() => setZoomPercent(p => Math.max(50, p - 20))} style={{ width: 26, height: 26, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.12)", color: "#fff", cursor: "pointer", fontSize: 14 }}>−</button>
            <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 11, minWidth: 38, textAlign: "center" }}>{zoomPercent}%</span>
            <button type="button" onClick={() => setZoomPercent(p => Math.min(300, p + 20))} style={{ width: 26, height: 26, borderRadius: 6, border: "none", background: "rgba(255,255,255,0.12)", color: "#fff", cursor: "pointer", fontSize: 14 }}>+</button>
            <button type="button" onClick={() => setZoomPercent(100)} style={{ height: 26, padding: "0 8px", borderRadius: 6, border: "none", background: "rgba(255,255,255,0.12)", color: "#fff", cursor: "pointer", fontSize: 10 }}>איפוס</button>
            <button type="button" onClick={() => setZoomModalOpen(true)} style={{ marginRight: 4, padding: "4px 10px", borderRadius: 6, background: "rgba(255,255,255,0.12)", border: "1px solid rgba(255,255,255,0.2)", color: "#fff", fontSize: 11, cursor: "pointer" }}>
              ⛶ מסך מלא
            </button>
            <button type="button" onClick={() => setStep(2)} style={{ padding: "4px 12px", borderRadius: 6, background: "transparent", border: "1px solid rgba(255,255,255,0.3)", color: "#fff", fontSize: 12, cursor: "pointer" }}>← אחור</button>
            <button type="button" onClick={() => setStep(4)} style={{ padding: "4px 12px", borderRadius: 6, background: "#e67e22", border: "none", color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>המשך →</button>
          </div>

          {/* Main area: canvas + sidebar */}
          <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

            {/* Canvas workspace */}
            <div
              ref={canvasContainerRef}
              style={{
                flex: 1,
                overflow: "auto",
                position: "relative",
                backgroundColor: "#eef2f7",
                backgroundImage: "radial-gradient(circle, #94a3b8 1px, transparent 1px)",
                backgroundSize: "22px 22px",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                padding: "12px 16px 16px 16px",
              }}
            >
              {/* Floating toolbar wrapper — sticky at top, centered */}
              <div style={{
                position: "sticky",
                top: 0,
                zIndex: 20,
                width: "100%",
                display: "flex",
                justifyContent: "center",
                paddingBottom: 8,
                pointerEvents: "none",
              }}>
              <div style={{
                pointerEvents: "auto",
                display: "inline-flex",
                gap: 4,
                background: "rgba(255, 255, 255, 0.92)",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                border: "1px solid rgba(203, 213, 225, 0.9)",
                borderRadius: 12,
                padding: "5px 8px",
                boxShadow: "0 4px 20px rgba(15, 23, 42, 0.12)",
                alignItems: "center",
              }}>
                <ToolbarButton
                  active={selectMode}
                  onClick={() => { setSelectMode(true); setStep3Tab("auto"); }}
                  icon={<SelectIcon />}
                  label="בחירה"
                />
                <div style={{ width: 1, height: 20, background: "#e2e8f0", margin: "0 4px" }} />
                <ToolbarButton
                  active={!selectMode && step3Tab === "manual" && drawMode === "line"}
                  onClick={() => { setSelectMode(false); setStep3Tab("manual"); setDrawMode("line"); setContextMenu(null); }}
                  icon={<LineIcon />}
                  label="קיר / קו"
                />
                <ToolbarButton
                  active={!selectMode && step3Tab === "manual" && drawMode === "rect"}
                  onClick={() => { setSelectMode(false); setStep3Tab("manual"); setDrawMode("rect"); setContextMenu(null); }}
                  icon={<RectIcon />}
                  label="מרובע"
                />
                <ToolbarButton
                  active={!selectMode && step3Tab === "zone"}
                  onClick={() => { setSelectMode(false); setStep3Tab("zone"); setContextMenu(null); }}
                  icon={<RectIcon />}
                  label="שטח רצפה"
                />
                <ToolbarButton
                  active={!selectMode && step3Tab === "manual" && drawMode === "path"}
                  onClick={() => { setSelectMode(false); setStep3Tab("manual"); setDrawMode("path"); setContextMenu(null); }}
                  icon={<PathIcon />}
                  label="חופשי"
                />
                {pendingShapes.length > 0 && (
                  <>
                    <div style={{ width: 1, height: 20, background: "#e2e8f0", margin: "0 4px" }} />
                    <button
                      type="button"
                      onClick={() => setCategoryPickerOpen(true)}
                      style={{
                        display: "flex", alignItems: "center", gap: 5,
                        padding: "4px 10px", borderRadius: 8,
                        background: "#F59E0B", border: "none",
                        color: "#fff", fontWeight: 700, fontSize: 11,
                        cursor: "pointer",
                      }}
                    >
                      שייך {pendingShapes.length} פריטים
                    </button>
                  </>
                )}
                {selectedSegIds.size > 0 && (
                  <>
                    <div style={{ width: 1, height: 20, background: "#e2e8f0", margin: "0 4px" }} />
                    <button
                      type="button"
                      onClick={() => setCategoryPickerOpen(true)}
                      style={{ padding: "4px 10px", borderRadius: 8, background: "#e67e22", border: "none", color: "#fff", fontWeight: 700, fontSize: 11, cursor: "pointer" }}
                    >
                      שייך {selectedSegIds.size} נבחרים ←
                    </button>
                  </>
                )}
              </div>
              </div>{/* end toolbar sticky wrapper */}

              {/* The canvas itself */}
              <div style={{ position: "relative" }}>
                <PlanningCanvasErrorBoundary>
                  <div className="relative select-none" style={{ flexShrink: 0, boxShadow: "0 24px 60px rgba(15,23,42,.18)", border: "1px solid #7C8EA3", background: "#fff", borderRadius: 18, overflow: "hidden" }}>
                    <img
                      ref={drawingImageRef}
                      src={imageUrl}
                      alt="plan"
                      className="block"
                      style={{ width: displaySize.width, height: displaySize.height, filter: "contrast(1.1) brightness(1.05) saturate(.92)" }}
                      onLoad={() => updateDisplaySizeFromImage(drawingImageRef.current)}
                      draggable={false}
                    />

                    {/* Auto segments overlay */}
                    {step3Tab === "auto" && autoSegments !== null && autoSegments.length > 0 && (
                      <svg width={displaySize.width} height={displaySize.height} className="absolute inset-0" onClick={() => { setContextMenu(null); setSelectedSegmentId(null); setSelectedPopoverItem(null); }}>
                        {highlightedClass && <rect x={0} y={0} width={displaySize.width} height={displaySize.height} fill="rgba(15,23,42,.12)" style={{ pointerEvents: "none" }} />}
                        {autoSegments.filter(seg => seg.suggested_subtype !== "פרט קטן").map((seg, idx) => {
                          const [bx, by, bw, bh] = seg.bbox.map(v => v * displayScale);
                          const isSelected = selectedSegmentId === seg.segment_id;
                          const summaryKey = (seg.suggested_type ?? seg.element_class) + "|" + (seg.suggested_subtype ?? seg.wall_type ?? "");
                          if (highlightedClass && summaryKey !== highlightedClass) return null;
                          const strokeColor = isSelected ? "#0F172A" : seg.category_color ?? (seg.confidence >= .75 ? "#15803D" : seg.confidence >= .5 ? "#B45309" : "#DC2626");
                          const fillColor = seg.category_color ?? (seg.element_class === "fixture" ? "#0EA5E9" : seg.element_class === "room" ? "#7C3AED" : "#2563EB");
                          return (
                            <g key={seg.segment_id} style={{ cursor: "pointer" }}
                              onMouseEnter={() => setHoveredSegId(seg.segment_id)}
                              onMouseLeave={() => setHoveredSegId(null)}
                              onClick={(ev) => {
                                ev.stopPropagation();
                                setSelectedSegmentId(seg.segment_id);
                                const cats = Object.values(planningState.categories);
                                const match = cats.find(cat => cat.type === seg.suggested_type && cat.subtype === seg.suggested_subtype) ?? cats.find(cat => cat.subtype === seg.suggested_subtype) ?? cats[0];
                                if (match) { setPopoverType(match.type); setPopoverSubtype(match.subtype); }
                                const cRect = canvasContainerRef.current?.getBoundingClientRect() ?? { left: 0, top: 0 };
                                const cScrollLeft = canvasContainerRef.current?.scrollLeft ?? 0;
                                const cScrollTop  = canvasContainerRef.current?.scrollTop  ?? 0;
                                const vpX = cRect.left - cScrollLeft + bx + bw / 2 - 120;
                                const vpY = cRect.top  - cScrollTop  + by - 220;
                                setContextMenu({
                                  x: Math.max(8, Math.min(window.innerWidth  - 256, vpX)),
                                  y: Math.max(56, Math.min(window.innerHeight - 260, vpY)),
                                  segId: seg.segment_id,
                                });
                              }}
                            >
                              <rect x={bx} y={by} width={bw} height={bh} fill={fillColor} fillOpacity={isSelected ? .26 : .08} stroke={strokeColor} strokeWidth={isSelected ? 3 : 2} rx={seg.element_class === "room" ? 5 : 1} strokeDasharray={seg.element_class === "fixture" ? "5 3" : "none"} />
                              <text x={bx + Math.max(10, bw / 2)} y={by + Math.max(12, bh / 2)} fill="#fff" fontSize={seg.element_class === "room" ? 10 : 9} fontWeight="700" textAnchor="middle" stroke="rgba(15,23,42,.35)" strokeWidth={.6} style={{ pointerEvents: "none" }}>{seg.element_class === "room" ? (seg.room_name ?? seg.label) : String(idx + 1)}</text>
                              {/* Checkbox */}
                              <rect
                                x={bx + 4} y={by + 4} width={14} height={14}
                                fill={selectedSegIds.has(seg.segment_id) ? "#e67e22" : "rgba(255,255,255,0.9)"}
                                stroke="#e67e22" strokeWidth={1.5} rx={3}
                                onClick={(ev) => {
                                  ev.stopPropagation();
                                  setSelectedSegIds(prev => {
                                    const next = new Set(prev);
                                    next.has(seg.segment_id) ? next.delete(seg.segment_id) : next.add(seg.segment_id);
                                    return next;
                                  });
                                }}
                                style={{ cursor: "pointer" }}
                              />
                              {selectedSegIds.has(seg.segment_id) && (
                                <text x={bx + 11} y={by + 14} fill="#fff" fontSize={10} fontWeight="bold"
                                  textAnchor="middle" style={{ pointerEvents: "none" }}>✓</text>
                              )}
                              {hoveredSegId === seg.segment_id && (
                                <g onClick={(ev) => { ev.stopPropagation(); void handleDeleteSegment(seg.segment_id); setContextMenu(null); }} style={{ cursor: "pointer" }}>
                                  <circle cx={bx + bw - 8} cy={by + 8} r={8} fill="#DC2626" />
                                  <text x={bx + bw - 8} y={by + 11} fill="#fff" fontSize={10} fontWeight="700" textAnchor="middle" style={{ pointerEvents: "none" }}>×</text>
                                </g>
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    )}

                    {/* Context menu popover */}
                    {contextMenu && (() => {
                      const cats = Object.values(planningState.categories);
                      const types = Array.from(new Set(cats.map(cat => cat.type))).filter(Boolean);
                      const activeType = popoverType || types[0] || "";
                      const subtypes = cats.filter(cat => cat.type === activeType).map(cat => cat.subtype);
                      const activeSubtype = popoverSubtype || subtypes[0] || "";
                      const match = cats.find(cat => cat.type === activeType && cat.subtype === activeSubtype) ?? null;
                      const seg = autoSegments?.find(x => x.segment_id === contextMenu.segId);
                      return (
                        <div onClick={ev => ev.stopPropagation()} style={{ position: "fixed", left: contextMenu.x, top: contextMenu.y, width: 240, background: "#fff", border: "1px solid #CBD5E1", borderRadius: 12, boxShadow: "0 18px 38px rgba(15,23,42,.18)", padding: 14, direction: "rtl", zIndex: 9999 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid #E2E8F0" }}>
                            <span style={{ fontSize: 14, fontWeight: 700, color: "#0F172A" }}>הגדרת אלמנט</span>
                            <span style={{ fontSize: 10, fontWeight: 700, color: match ? "#166534" : "#92400E", background: match ? "#DCFCE7" : "#FEF3C7", borderRadius: 999, padding: "3px 8px" }}>{match ? "מוכן לאישור" : "נדרש שיוך"}</span>
                          </div>
                          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 10 }}>{seg?.label ?? seg?.suggested_subtype ?? "אלמנט נבחר"}</div>
                          <label style={{ display: "block", fontSize: 11, color: "#64748B", marginBottom: 4 }}>סוג:</label>
                          <select value={activeType} onChange={ev => { const nextType = ev.target.value; setPopoverType(nextType); setPopoverSubtype(cats.find(cat => cat.type === nextType)?.subtype ?? ""); }} style={{ width: "100%", border: "1px solid #CBD5E1", borderRadius: 8, padding: "7px 9px", fontSize: 12, marginBottom: 10, background: "#fff" }}>
                            {types.map(type => <option key={type} value={type}>{type}</option>)}
                          </select>
                          <label style={{ display: "block", fontSize: 11, color: "#64748B", marginBottom: 4 }}>תת-קטגוריה:</label>
                          <select value={activeSubtype} onChange={ev => setPopoverSubtype(ev.target.value)} style={{ width: "100%", border: "1px solid #CBD5E1", borderRadius: 8, padding: "7px 9px", fontSize: 12, marginBottom: 12, background: "#fff" }}>
                            {subtypes.map(subtype => <option key={subtype} value={subtype}>{subtype}</option>)}
                          </select>
                          <div style={{ display: "flex", gap: 8 }}>
                            <button type="button" disabled={!match} onClick={ev => { ev.stopPropagation(); if (!match) return; void handleConfirmSingleSegment(contextMenu.segId, match.key); setContextMenu(null); }} style={{ flex: 1, height: 34, borderRadius: 8, border: "none", background: match ? "#15803D" : "#CBD5E1", color: "#fff", fontSize: 12, fontWeight: 700, cursor: match ? "pointer" : "not-allowed" }}>אשר</button>
                            <button type="button" onClick={ev => { ev.stopPropagation(); void handleDeleteSegment(contextMenu.segId); setContextMenu(null); }} style={{ width: 70, height: 34, borderRadius: 8, border: "1px solid #FECACA", background: "#fff", color: "#DC2626", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>מחק</button>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Auto: empty state */}
                    {step3Tab === "auto" && autoSegments === null && (
                      <div className="absolute inset-0 flex items-center justify-center" style={{ background: "rgba(255,255,255,.12)" }}>
                        <div style={{ background: "rgba(255,255,255,.94)", border: "1px solid #CBD5E1", borderRadius: 14, padding: "18px 22px", textAlign: "center", boxShadow: "0 12px 28px rgba(15,23,42,.14)" }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#0F172A", marginBottom: 6 }}>נתח את התוכנית כדי להציג זיהויים</div>
                          <div style={{ fontSize: 12, color: "#64748B" }}>לאחר הניתוח אפשר לבחור אלמנט ולשייך אותו ישירות מעל השרטוט.</div>
                        </div>
                      </div>
                    )}

                    {/* Zone drawing SVG */}
                    {step3Tab === "zone" && (
                      <svg ref={zoneCanvasRef} width={displaySize.width} height={displaySize.height} className="absolute inset-0 cursor-crosshair" style={{ touchAction: "none" }}
                        onMouseDown={handleZoneMouseDown} onMouseMove={handleZoneMouseMove} onMouseUp={handleZoneMouseUp} onMouseLeave={() => setZoneDrawing(false)}
                        onTouchStart={makeTouchHandler(handleZoneMouseDown)} onTouchMove={makeTouchHandler(handleZoneMouseMove)} onTouchEnd={makeTouchHandler(handleZoneMouseUp)}
                      >
                        {planningState.items.filter(it => it.type === "zone" || it.type === "rect").map(item => {
                          const obj = item.raw_object;
                          const cat = planningState.categories[item.category];
                          const color = getCategoryColor(cat?.type, cat?.subtype);
                          return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale} width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale} fill={hexToRgba(color, .18)} stroke={color} strokeWidth={2} rx={3} />;
                        })}
                        {zoneStart && (zoneEnd ?? zoneTemp) && (() => {
                          const end = zoneEnd ?? zoneTemp;
                          return <rect x={Math.min(zoneStart.x, end!.x)} y={Math.min(zoneStart.y, end!.y)} width={Math.abs(end!.x - zoneStart.x)} height={Math.abs(end!.y - zoneStart.y)} fill="rgba(37,99,235,.16)" stroke="#2563EB" strokeWidth={2} strokeDasharray="6 4" />;
                        })()}
                      </svg>
                    )}

                    {/* Manual drawing SVG */}
                    {step3Tab === "manual" && (
                      <svg ref={drawingSurfaceRef} width={displaySize.width} height={displaySize.height} className="absolute inset-0 cursor-crosshair" style={{ touchAction: "none" }}
                        onMouseDown={selectMode ? undefined : handleCanvasMouseDown}
                        onMouseMove={selectMode ? undefined : handleCanvasMouseMove}
                        onMouseUp={selectMode ? undefined : handleCanvasMouseUp}
                        onMouseLeave={() => setDrawing(false)}
                        onTouchStart={selectMode ? undefined : makeTouchHandler(handleCanvasMouseDown)}
                        onTouchMove={selectMode ? undefined : makeTouchHandler(handleCanvasMouseMove)}
                        onTouchEnd={selectMode ? undefined : makeTouchHandler(handleCanvasMouseUp)}
                      >
                        {planningState.items.map(item => {
                          const obj = item.raw_object;
                          const cat = planningState.categories[item.category];
                          const color = getCategoryColor(cat?.type, cat?.subtype);
                          const isReviewTarget = reviewMode && reviewQueue[reviewIndex] === item.uid;
                          const reviewStrokeWidth = isReviewTarget ? 5 : 2;
                          const reviewFilter = isReviewTarget ? "drop-shadow(0 0 6px rgba(230, 126, 34, 0.9))" : undefined;
                          if (item.type === "line") return <line key={item.uid} x1={Number(obj.x1) * displayScale} y1={Number(obj.y1) * displayScale} x2={Number(obj.x2) * displayScale} y2={Number(obj.y2) * displayScale} stroke={isReviewTarget ? "#e67e22" : color} strokeWidth={reviewStrokeWidth} strokeLinecap="round" style={{ cursor: selectMode ? "pointer" : "inherit", filter: reviewFilter }} onClick={selectMode ? (e) => handleSelectItem(item.uid, e as unknown as React.MouseEvent) : undefined} />;
                          if (item.type === "rect" || item.type === "zone") return <rect key={item.uid} x={Number(obj.x) * displayScale} y={Number(obj.y) * displayScale} width={Number(obj.width) * displayScale} height={Number(obj.height) * displayScale} fill={hexToRgba(isReviewTarget ? "#e67e22" : color, .15)} stroke={isReviewTarget ? "#e67e22" : color} strokeWidth={reviewStrokeWidth} style={{ cursor: selectMode ? "pointer" : "inherit", filter: reviewFilter }} onClick={selectMode ? (e) => handleSelectItem(item.uid, e as unknown as React.MouseEvent) : undefined} />;
                          const pts = Array.isArray(obj.points) ? (obj.points as number[][]).map(([px, py]) => `${px * displayScale},${py * displayScale}`).join(" ") : "";
                          return <polyline key={item.uid} points={pts} fill="none" stroke={isReviewTarget ? "#e67e22" : color} strokeWidth={reviewStrokeWidth} style={{ cursor: selectMode ? "pointer" : "inherit", filter: reviewFilter }} onClick={selectMode ? (e) => handleSelectItem(item.uid, e as unknown as React.MouseEvent) : undefined} />;
                        })}
                        {pendingShapes.map(renderPendingOnCanvas)}
                        {drawing && startPoint && tempPoint && drawMode === "line" && <line x1={startPoint.x} y1={startPoint.y} x2={tempPoint.x} y2={tempPoint.y} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                        {drawing && startPoint && tempPoint && drawMode === "rect" && <rect x={Math.min(startPoint.x, tempPoint.x)} y={Math.min(startPoint.y, tempPoint.y)} width={Math.abs(tempPoint.x - startPoint.x)} height={Math.abs(tempPoint.y - startPoint.y)} fill={hexToRgba(PENDING_COLOR, .15)} stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                        {drawing && drawMode === "path" && pathPoints.length > 1 && <polyline points={pathPoints.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke={PENDING_COLOR} strokeWidth={2} strokeDasharray="6 3" />}
                      </svg>
                    )}

                    {/* Context popover for select mode */}
                    {selectedPopoverItem && selectMode && (
                      <ContextPopover
                        item={selectedPopoverItem}
                        position={popoverPosition}
                        categories={planningState.categories}
                        onClose={() => setSelectedPopoverItem(null)}
                        onDelete={() => { void handleDeleteItem(selectedPopoverItem.uid); setSelectedPopoverItem(null); }}
                      />
                    )}

                    {/* Floating review pill */}
                    {reviewMode && reviewQueue.length > 0 && (() => {
                      const uid = reviewQueue[reviewIndex];
                      const item = planningState?.items.find((i) => i.uid === uid);
                      const cat = item ? planningState?.categories[item.category] : null;
                      const label = cat
                        ? `${cat.type} — ${cat.subtype}`
                        : `פריט ${reviewIndex + 1}`;
                      return (
                        <div style={{
                          position: "absolute",
                          bottom: 20,
                          left: "50%",
                          transform: "translateX(-50%)",
                          background: "#ffffff",
                          padding: "10px 18px",
                          borderRadius: 30,
                          boxShadow: "0 8px 28px rgba(15, 23, 42, 0.18)",
                          border: "2px solid #e67e22",
                          display: "flex",
                          alignItems: "center",
                          gap: 12,
                          zIndex: 30,
                          direction: "rtl",
                          whiteSpace: "nowrap",
                        }}>
                          <span style={{ fontWeight: 700, color: "#1e3a5f", fontSize: 12 }}>
                            {label} — {reviewIndex + 1}/{reviewQueue.length}
                          </span>
                          <button
                            type="button"
                            onClick={() => reviewAdvance()}
                            style={{
                              padding: "5px 14px", borderRadius: 8,
                              background: "#15803d", border: "none",
                              color: "#fff", fontWeight: 700, fontSize: 12, cursor: "pointer"
                            }}
                          >
                            ✓ המשך
                          </button>
                          <button
                            type="button"
                            onClick={() => void reviewDeleteCurrent()}
                            style={{
                              padding: "5px 14px", borderRadius: 8,
                              background: "#fff", border: "1px solid #ef4444",
                              color: "#ef4444", fontWeight: 700, fontSize: 12, cursor: "pointer"
                            }}
                          >
                            ✕ מחיקה
                          </button>
                          <button
                            type="button"
                            onClick={stopReviewMode}
                            style={{
                              padding: "5px 10px", borderRadius: 8,
                              background: "transparent", border: "none",
                              color: "#94a3b8", fontSize: 12, cursor: "pointer"
                            }}
                          >
                            סגור
                          </button>
                        </div>
                      );
                    })()}
                  </div>
                </PlanningCanvasErrorBoundary>
              </div>
            </div>

            {/* Sidebar (right) */}
            <div style={{
              width: 260,
              flexShrink: 0,
              background: "rgba(255,255,255,0.96)",
              borderRight: "1px solid #e2e8f0",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}>
              {/* Sidebar header */}
              <div style={{ padding: "14px 16px 10px", borderBottom: "1px solid #f1f5f9" }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>
                  הגדרת תכולה
                </div>
                <div style={{ fontSize: 12, color: "#94a3b8" }}>
                  {planningState.items.length ?? 0} פריטים משויכים
                </div>
                <button
                  type="button"
                  onClick={() => { setStep3Tab("auto"); void handleAutoAnalyze(); }}
                  disabled={autoLoading}
                  style={{ marginTop: 10, width: "100%", height: 36, borderRadius: 8, border: "none", background: "#E67E22", color: "#fff", fontSize: 12, fontWeight: 700, cursor: autoLoading ? "not-allowed" : "pointer", opacity: autoLoading ? .65 : 1 }}
                >
                  {autoLoading ? "מנתח..." : "נתח תוכנית"}
                </button>
              </div>

              {/* Review mode button / badge */}
              {planningState.items.length > 0 && !reviewMode && (
                <div style={{ padding: "6px 12px 0" }}>
                  <button
                    type="button"
                    onClick={startReviewMode}
                    style={{
                      width: "100%", padding: "7px 0", borderRadius: 8,
                      background: "#e67e22", border: "none",
                      color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer",
                      display: "flex", alignItems: "center", justifyContent: "center", gap: 6
                    }}
                  >
                    <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    התחל סקירת פריטים
                  </button>
                </div>
              )}
              {reviewMode && (
                <div style={{ padding: "6px 12px 0" }}>
                  <div style={{
                    background: "#fff7ed", border: "1px solid #fed7aa",
                    borderRadius: 8, padding: "7px 10px",
                    fontSize: 11, color: "#c2410c", fontWeight: 600,
                    display: "flex", alignItems: "center", gap: 6
                  }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#e67e22", display: "inline-block" }} />
                    מצב סריקה פעיל — {reviewIndex + 1}/{reviewQueue.length}
                  </div>
                </div>
              )}

              {/* Aggregated summaries */}
              <div style={{ flex: 1, overflowY: "auto", padding: "10px 12px" }}>
                {(() => {
                  const groups: Record<string, { type: string; subtype: string; count: number; totalLength: number; color: string }> = {};
                  for (const item of (planningState?.items ?? [])) {
                    const cat = planningState?.categories[item.category];
                    if (!cat) continue;
                    const key = `${cat.type}:${cat.subtype}`;
                    if (!groups[key]) groups[key] = { type: cat.type, subtype: cat.subtype, count: 0, totalLength: 0, color: getCategoryColor(cat.type, cat.subtype) };
                    groups[key].count++;
                    groups[key].totalLength += Number(item.length_m_effective ?? item.length_m ?? 0);
                  }
                  const entries = Object.entries(groups);

                  if (autoSegments !== null && autoSegments.length > 0) {
                    const detGroups = new Map<string, { label: string; count: number; length: number; color: string; key: string }>();
                    autoSegments.filter(seg => seg.suggested_subtype !== "פרט קטן").forEach(seg => {
                      const typeLabel = seg.suggested_type ?? (seg.element_class === "room" ? "חדרים" : seg.element_class === "fixture" ? "אביזרים" : "קירות");
                      const subtypeLabel = seg.suggested_subtype ?? seg.wall_type ?? "ללא סוג";
                      const k = typeLabel + "|" + subtypeLabel;
                      const row = detGroups.get(k) ?? { key: k, label: typeLabel + " - " + subtypeLabel, count: 0, length: 0, color: seg.category_color ?? (seg.element_class === "fixture" ? "#0EA5E9" : seg.element_class === "room" ? "#7C3AED" : "#2563EB") };
                      row.count += 1;
                      row.length += Number(seg.length_m ?? 0);
                      detGroups.set(k, row);
                    });
                    return Array.from(detGroups.values()).sort((a, b) => b.count - a.count).map(row => {
                      const active = highlightedClass === row.key;
                      return (
                        <button key={row.key} type="button"
                          onClick={() => { if (active) { setHighlightedClass(null); setHighlightedType(null); } else { setHighlightedClass(row.key); setHighlightedType(row.label); } }}
                          style={{ textAlign: "right", width: "100%", padding: "8px 10px", borderRadius: 8, border: `1px solid ${active ? row.color : '#E2E8F0'}`, background: active ? hexToRgba(row.color, .1) : "#f8fafc", cursor: "pointer", marginBottom: 6 }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
                            <span style={{ width: 8, height: 8, borderRadius: "50%", background: row.color, flexShrink: 0, display: "inline-block" }} />
                            <span style={{ fontSize: 12, fontWeight: 700, color: "#1e293b", flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{row.label}</span>
                            <span style={{ fontSize: 11, color: "#64748b", fontWeight: 700 }}>{row.count}</span>
                          </div>
                          <div style={{ fontSize: 11, color: "#94a3b8", paddingRight: 15 }}>
                            {row.length > 0 ? `${row.length.toFixed(1)} מ'` : "ללא מידה"}
                          </div>
                        </button>
                      );
                    });
                  }

                  if (entries.length === 0) return (
                    <div style={{ fontSize: 12, color: "#94a3b8", textAlign: "center", padding: "24px 0" }}>
                      עדיין אין פריטים משויכים
                    </div>
                  );
                  return entries.map(([key, g]) => {
                    const groupItems = planningState.items.filter(it => {
                      const cat = planningState.categories[it.category];
                      return cat && `${cat.type}:${cat.subtype}` === key;
                    });
                    return (
                      <div key={key} style={{ marginBottom: 6, borderRadius: 8, background: "#f8fafc", border: "1px solid #e2e8f0", overflow: "hidden" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "8px 10px 3px" }}>
                          <span style={{ width: 8, height: 8, borderRadius: "50%", background: g.color, flexShrink: 0, display: "inline-block" }} />
                          <span style={{ fontSize: 12, fontWeight: 700, color: "#1e293b" }}>{g.type}</span>
                          <span style={{ fontSize: 11, color: "#64748b" }}>{g.subtype}</span>
                        </div>
                        <div style={{ fontSize: 11, color: "#94a3b8", paddingRight: 15, paddingBottom: groupItems.length > 0 ? 4 : 8, paddingLeft: 10 }}>
                          {g.count} {g.count === 1 ? "פריט" : "פריטים"}{g.totalLength > 0 && ` · ${g.totalLength.toFixed(1)} מ'`}
                        </div>
                        {groupItems.map(item => (
                          <div key={item.uid} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 10px", borderTop: "1px solid #f1f5f9" }}>
                            <span style={{ flex: 1, fontSize: 10, color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {Number(item.length_m_effective ?? item.length_m ?? 0).toFixed(1)} מ'
                            </span>
                            <button
                              type="button"
                              onClick={() => void handleDeleteItem(item.uid)}
                              style={{ marginRight: "auto", padding: "2px 8px", borderRadius: 6, border: "1px solid #fecaca", background: "#fff", color: "#dc2626", fontSize: 11, cursor: "pointer" }}
                            >×</button>
                          </div>
                        ))}
                      </div>
                    );
                  });
                })()}
              </div>

              {/* Approve all button */}
              {planningState.items.length > 0 && (
                <div style={{ padding: "6px 12px" }}>
                  {approveAllMsg ? (
                    <div style={{
                      fontSize: 11, color: "#15803d", fontWeight: 600,
                      padding: "6px 10px", background: "#f0fdf4",
                      border: "1px solid #86efac", borderRadius: 7, textAlign: "center"
                    }}>
                      ✓ {approveAllMsg}
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        if (autoSegments && autoSegments.length > 0 && Object.keys(autoConfirmedKeys).length > 0) {
                          void handleConfirmAutoSegments(false);
                        } else {
                          setApproveAllMsg("אין פריטים לאישור");
                          setTimeout(() => setApproveAllMsg(null), 2000);
                        }
                      }}
                      style={{
                        width: "100%", padding: "6px 0", borderRadius: 7,
                        background: "transparent",
                        border: "1px solid #cbd5e1",
                        color: "#334155", fontSize: 11, fontWeight: 600, cursor: "pointer"
                      }}
                    >
                      ✓ אשר את הכל
                    </button>
                  )}
                </div>
              )}

              {/* Zone category picker (shown when zone tab active) */}
              {step3Tab === "zone" && (
                <div style={{ padding: "8px 12px", background: "#f8fafc", borderTop: "1px solid #e2e8f0" }}>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6, fontWeight: 600 }}>קטגוריה לשטח שצוירה:</div>
                  <select
                    className="pro-field"
                    value={zoneCatKey}
                    onChange={(e) => setZoneCatKey(e.target.value)}
                    style={{ width: "100%", marginBottom: 6, fontSize: 12 }}
                  >
                    <option value="">בחר קטגוריה...</option>
                    {Object.values(planningState.categories).map(cat => (
                      <option key={cat.key} value={cat.key}>{cat.type} - {cat.subtype}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => void handleAddZone()}
                    disabled={!zoneStart || !zoneEnd || !zoneCatKey}
                    style={{ width: "100%", padding: "6px 0", borderRadius: 7, background: (!zoneStart || !zoneEnd || !zoneCatKey) ? "#CBD5E1" : "#1e3a5f", border: "none", color: "#fff", fontSize: 11, fontWeight: 700, cursor: (!zoneStart || !zoneEnd || !zoneCatKey) ? "not-allowed" : "pointer" }}
                  >
                    הוסף שטח
                  </button>
                </div>
              )}

              {/* Pending indicator */}
              {pendingShapes.length > 0 && (
                <div style={{ padding: "8px 12px", background: "#fffbeb", borderTop: "1px solid #fcd34d" }}>
                  <div style={{ fontSize: 11, color: "#b45309", fontWeight: 600, marginBottom: 5 }}>
                    {pendingShapes.length} פריטים ממתינים לשיוך
                  </div>
                  <button
                    type="button"
                    onClick={() => setCategoryPickerOpen(true)}
                    style={{ width: "100%", padding: "6px 0", borderRadius: 7, background: "#F59E0B", border: "none", color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer" }}
                  >
                    + שייך לקטגוריה
                  </button>
                </div>
              )}

              {/* Add category button */}
              <div style={{ padding: "10px 12px", borderTop: "1px solid #f1f5f9" }}>
                <button
                  type="button"
                  onClick={() => setAddCatOpen(prev => !prev)}
                  style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "2px dashed #e67e22",
                    background: addCatOpen ? "#fff7ed" : "#fff", color: "#c96a10", fontWeight: 700, fontSize: 13,
                    cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, marginBottom: 8 }}
                >
                  <span style={{ fontSize: 18, lineHeight: 1 }}>+</span> הוסף קטגוריה
                </button>
                {addCatOpen && (
                  <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 10, padding: 14, marginBottom: 12, direction: "rtl" }}>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 8, fontWeight: 600 }}>סוג</div>
                    <select className="pro-field" value={newType} onChange={ev => setNewType(ev.target.value)} style={{ fontSize: 13, marginBottom: 8, width: "100%" }}>
                      {Object.keys(CATEGORY_SUBTYPES).map(t => <option key={t}>{t}</option>)}
                    </select>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 8, fontWeight: 600 }}>תת-סוג</div>
                    <select className="pro-field" value={newSubtype} onChange={ev => setNewSubtype(ev.target.value)} style={{ fontSize: 13, marginBottom: 12, width: "100%" }}>
                      {(CATEGORY_SUBTYPES[newType] ?? ["כללי"]).map(s => <option key={s}>{s}</option>)}
                    </select>
                    <button type="button"
                      style={{ width: "100%", padding: "9px 0", borderRadius: 9, background: "#e67e22", color: "#fff",
                        border: "none", fontWeight: 700, fontSize: 13, cursor: "pointer" }}
                      onClick={() => { handleAddCategory(); void handleSaveCategories(); setAddCatOpen(false); }}
                    >✓ צור קטגוריה</button>
                  </div>
                )}
              </div>

              {/* BOQ button */}
              <div style={{ padding: "0 12px 12px" }}>
                <button
                  type="button"
                  onClick={() => { void handleLoadBoq(); setBoqVisible(true); }}
                  disabled={boqLoading || !autoSegments?.length}
                  style={{
                    width: "100%", padding: "7px 0", borderRadius: 8,
                    background: (boqLoading || !autoSegments?.length) ? "#CBD5E1" : "#1e3a5f",
                    border: "none", color: "#fff", fontSize: 12, cursor: (boqLoading || !autoSegments?.length) ? "not-allowed" : "pointer",
                    fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                  }}
                >
                  <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                  {boqLoading ? "טוען..." : "הפק כתב כמויות"}
                </button>

                <button
                  type="button"
                  onClick={() => setStep(4)}
                  disabled={planningState.items.length === 0}
                  style={{ width: "100%", marginTop: 8, padding: "7px 0", borderRadius: 8, background: planningState.items.length === 0 ? "#CBD5E1" : "#15803D", border: "none", color: "#fff", fontSize: 12, fontWeight: 700, cursor: planningState.items.length === 0 ? "not-allowed" : "pointer" }}
                >
                  אשר ועבור לשלב 4 ←
                </button>
              </div>
            </div>
          </div>

          {/* CategoryPickerModal */}
          {categoryPickerOpen && planningState && (
            <CategoryPickerModal
              categories={categoriesDraft}
              pendingCount={pendingShapes.length}
              onPick={(key) => void handleAssignCategory(key)}
              onCreateAndPick={(type, sub, val, note) => void handleCreateAndAssign(type, sub, val, note)}
              onCancel={() => setCategoryPickerOpen(false)}
            />
          )}

          {/* ZoomModal */}
          {zoomModalOpen && imageUrl && planningState && (
            <ZoomModal
              imageUrl={imageUrl}
              planningState={planningState}
              pendingShapes={pendingShapes}
              displayScale={displayScale}
              onClose={() => setZoomModalOpen(false)}
              onDrawComplete={(shape) => setPendingShapes((prev) => [...prev, shape])}
              onAssignCategory={() => setCategoryPickerOpen(true)}
              onDeletePending={(id) => setPendingShapes((prev) => prev.filter((s) => s.id !== id))}
              onDeleteItem={handleDeleteItem}
            />
          )}
        </div>
      )}

      {/* ג"€ג"€ STEP 4: BOQ + Save ג"€ג"€ */}
      
{step === 4 && planningState && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
            <p className="text-sm font-semibold mb-3 text-[#31333F]">שלב 4: כתב כמויות (BOQ) ושמירה</p>
            {finalizeNotice && (
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-green-800 bg-green-50 rounded-lg px-4 py-3" style={{ border: "1px solid #86EFAC", borderRight: "5px solid #22C55E" }}>
                <span style={{ fontSize: 20 }}>?</span>
                <span className="flex-1">{finalizeNotice}</span>
                <button type="button" onClick={() => setFinalizeNotice("")} style={{ color: "#16A34A", background: "none", border: "none", cursor: "pointer", fontSize: 15 }}>סגור</button>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סך הפריטים</p>
                <p className="font-semibold text-[#31333F]">{planningState.items.length}</p>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סך אורך</p>
                <p className="font-semibold text-[#31333F]">{planningState.totals.total_length_m.toFixed(2)} מ&apos;</p>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs">
                <p className="text-slate-500">סך שטח</p>
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
                            <td style={{ padding: "7px 10px", fontWeight: 700, color }}>{row.type ?? "-"}</td>
                            <td style={{ padding: "7px 10px", color: "var(--text-1)" }}>{row.subtype ?? "-"}</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)" }}>{row.count ?? 0}</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)", fontFamily: "monospace" }}>{(row.total_length_m ?? 0).toFixed(2)} מ&apos;</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: "var(--text-2)", fontFamily: "monospace" }}>{(row.total_area_m2 ?? 0).toFixed(2)} מ&quot;ר</td>
                            <td style={{ padding: "7px 6px", textAlign: "center", color: heightVal ? "var(--navy)" : "var(--s400)", fontFamily: "monospace", fontWeight: heightVal ? 600 : 400 }}>
                              {heightVal != null ? String(heightVal) + " מ'" : "-"}
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
                className={loading ? "btn btn-ghost" : "btn btn-orange"}
                style={loading ? { cursor: "not-allowed", opacity: .7 } : {}}>
                {loading ? "שומר..." : "שמירה סופית"}
              </button>
              <button type="button" onClick={() => setStep(5)} className="btn btn-primary">גזרות עבודה</button>
              <button type="button" onClick={() => setStep(3)} className="btn btn-ghost">חזור לשלב 3</button>
            </div>
          </div>
        </div>
      )}

      {step === 5 && planningState && (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr,340px] gap-4">
          {/* Left: canvas + section overlays */}
          <div className="space-y-3">
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
              <p className="text-sm font-semibold mb-2 text-[#31333F]">׳©׳׳‘ 5: ׳’׳–׳¨׳•׳× ׳¢׳‘׳•׳”׳”</p>
              <p className="text-xs text-slate-500 mb-3">
                ׳¡׳׳ ׳׳–׳•׳¨ ׳¢׳ ׳”׳’׳¨׳׳•׳©׳” ׳׳›׳ ׳’׳–׳¨׳” ׳•׳׳׳ ׳©׳ ׳§׳‘׳׳ ׳•׳¢׳•׳‘׳” ׳׳—׳¨׳׳™. ׳’׳–׳¨׳•׳× ׳׳™׳ ׳ ׳—׳•׳‘׳” ג€” ׳ ׳™׳×׳ ׳׳”׳׳’.
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
              <p className="text-[10px] text-slate-400 mt-1">׳’׳¨׳•׳¨ ׳¢׳ ׳”׳’׳¨׳׳•׳©׳” ׳׳¡׳׳ ׳×׳—׳•׳ ׳’׳–׳¨׳” (׳׳•׳₪׳¦׳™׳•׳ ׳׳™)</p>
            </div>
          </div>

          {/* Right: form + list */}
          <div className="space-y-4">
            {/* Add section form */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4 space-y-3">
              <p className="text-sm font-semibold text-[#31333F]">׳”׳•׳¡׳£ ׳’׳–׳¨׳”</p>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">׳©׳ ׳”׳’׳–׳¨׳” (׳׳•׳₪׳¦׳™׳•׳ ׳׳™)</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="׳’׳–׳¨׳” ׳¦׳₪׳•׳ ׳™׳× / ׳§׳•׳׳” ׳׳³..."
                    value={secName}
                    onChange={e => setSecName(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">׳©׳ ׳§׳‘׳׳ ׳׳‘׳¦׳¢</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="׳©׳ ׳”׳§׳‘׳׳..."
                    value={secContractor}
                    onChange={e => setSecContractor(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">׳©׳ ׳¢׳•׳‘׳” ׳׳—׳¨׳׳™</label>
                  <input
                    type="text"
                    className="w-full border border-slate-300 rounded px-2 py-1 text-xs"
                    placeholder="׳©׳ ׳”׳¢׳•׳‘׳”..."
                    value={secWorker}
                    onChange={e => setSecWorker(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600 block mb-0.5">׳¦׳‘׳¢</label>
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
                    ׳׳–׳•׳¨ ׳¡׳•׳׳ ג" | {Math.round(Math.abs(secEnd.x - secStart.x))}ֳ—{Math.round(Math.abs(secEnd.y - secStart.y))} px
                    <button type="button" className="text-red-400 hover:text-red-600 mr-2" onClick={() => { setSecStart(null); setSecEnd(null); }}>ג• ׳ ׳§׳”</button>
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={() => void handleAddSection()}
                disabled={loading || (!secContractor.trim() && !secWorker.trim())}
                className="w-full px-3 py-2 rounded-lg bg-[var(--navy)] text-white text-xs font-semibold disabled:opacity-40 hover:bg-[#162d56]"
              >
                {loading ? "׳©׳•׳׳¨..." : "+ ׳”׳•׳¡׳£ ׳’׳–׳¨׳”"}
              </button>
            </div>

            {/* Existing sections list */}
            <div className="bg-white rounded-lg border border-[#E6E6EA] shadow-sm p-4">
              <p className="text-sm font-semibold mb-2 text-[#31333F]">׳’׳–׳¨׳•׳× ׳§׳™׳™׳׳•׳× ({planningState.sections.length})</p>
              {planningState.sections.length === 0 ? (
                <p className="text-xs text-slate-400">׳׳™׳ ׳’׳–׳¨׳•׳× ׳¢׳”׳™׳™׳. ׳ ׳™׳×׳ ׳׳”׳׳’ ׳¢׳ ׳©׳׳‘ ׳–׳”.</p>
              ) : (
                <div className="space-y-2">
                  {planningState.sections.map((sec) => (
                    <div key={sec.uid} className="rounded-lg px-3 py-2 text-xs flex items-start justify-between gap-2"
                      style={{ background: `${sec.color}12`, border: `1px solid ${sec.color}55` }}>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: sec.color, display: "inline-block" }} />
                          <span className="font-semibold text-slate-700">{sec.name || "׳׳׳ ׳©׳"}</span>
                        </div>
                        <div className="text-slate-500">נ— ׳§׳‘׳׳: <span className="text-slate-700 font-medium">{sec.contractor || "ג€”"}</span></div>
                        <div className="text-slate-500">נ‘· ׳¢׳•׳‘׳”: <span className="text-slate-700 font-medium">{sec.worker || "ג€”"}</span></div>
                        {sec.width > 0 && (
                          <div className="text-slate-400 mt-0.5">
                            ׳׳–׳•׳¨: {Math.round(sec.width)}ֳ—{Math.round(sec.height)} px
                          </div>
                        )}
                      </div>
                      <button type="button" onClick={() => void handleDeleteSection(sec.uid)}
                        className="text-red-400 hover:text-red-600 flex-shrink-0 text-sm">מחק</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button type="button" onClick={() => void handleFinalize()} disabled={loading}
                style={{ padding: "11px 20px", borderRadius: 10, background: loading ? "#94a3b8" : "var(--orange)", color: "#fff", border: "none", fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer", boxShadow: loading ? "none" : "0 3px 12px rgba(255,75,75,0.3)", transition: "all 0.15s" }}>
                {loading ? "׳©׳•׳׳¨..." : "נ’¾ ׳©׳׳™׳¨׳” ׳¡׳•׳₪׳™׳×"}
              </button>
              <button type="button" onClick={() => setStep(4)}
                style={{ padding: "9px 16px", borderRadius: 10, background: "#fff", color: "#64748b", border: "1.5px solid #CBD5E1", fontSize: 13, cursor: "pointer", fontWeight: 500 }}>
                ג† ׳—׳–׳•׳¨ ׳׳©׳׳‘ 4
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









